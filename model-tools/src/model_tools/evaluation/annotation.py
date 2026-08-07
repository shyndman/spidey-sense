"""Annotate evaluation images with Grounding DINO, preferring CUDA when available.

The stage deliberately keeps source metadata, images, and per-sample outputs inside
``EvaluationPaths``. It emits no per-image diagnostics: callers can inspect the
content-free ``StageFailure`` records in the data volume when a sample fails.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Protocol, TypeGuard

from .contracts import (
    AnnotationRecord,
    Detection,
    SampleManifest,
    StageFailure,
    StageSummary,
)
from .events import AggregateEvent, emit_event
from .paths import EvaluationPaths

CHECKPOINT: Final[str] = "IDEA-Research/grounding-dino-base"
PROMPT: Final[str] = (
    "spider. tarantula. black widow. wolf spider. garden spider. barn spider."
)
TARGET_PHRASES: Final[tuple[str, ...]] = (
    "spider",
    "tarantula",
    "black widow",
    "wolf spider",
    "garden spider",
    "barn spider",
)
MAX_DETECTIONS: Final[int] = 20
PART_SUFFIX: Final[str] = ".part"


class _ContextManager(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _ImageSource(_ContextManager, Protocol):
    def convert(self, mode: str) -> object: ...


def _is_callable(value: object) -> TypeGuard[Callable[..., object]]:
    return callable(value)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def _is_context_manager(value: object) -> TypeGuard[_ContextManager]:
    enter: object = getattr(value, "__enter__", None)
    exit_method: object = getattr(value, "__exit__", None)
    return _is_callable(enter) and _is_callable(exit_method)


def _is_image_source(value: object) -> TypeGuard[_ImageSource]:
    convert: object = getattr(value, "convert", None)
    return _is_context_manager(value) and _is_callable(convert)


@dataclass(frozen=True, slots=True)
class RankedProposal:
    """A detector proposal before conversion to the public contract model."""

    confidence: float
    phrase: str
    box_xyxy: tuple[float, float, float, float]
    query_index: int


def _as_nested_numbers(value: object) -> object:
    """Convert tensor-like fake/model values to ordinary Python numeric lists."""
    current = value
    for method_name in ("detach", "cpu"):
        method: object = getattr(current, method_name, None)
        if _is_callable(method):
            current = method()
    tolist: object = getattr(current, "tolist", None)
    if _is_callable(tolist):
        current = tolist()
    return current


def _strip_batch(value: object, *, expected_width: int | None = None) -> object:
    value = _as_nested_numbers(value)
    if not _is_sequence(value):
        return value
    first_value = value[0] if value else None
    if _is_sequence(first_value):
        # Logits are [queries, tokens] and boxes are [queries, 4]. A batch adds
        # one more sequence level, which is the only shape this helper removes.
        first_item = first_value[0] if first_value else None
        if expected_width is None:
            if _is_sequence(first_item):
                return first_value
        elif len(first_value) != expected_width or _is_sequence(first_item):
            return first_value
    return value


def _finite_number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _sigmoid(value: object) -> float:
    number = _finite_number(value)
    if number >= 0.0:
        result = 1.0 / (1.0 + math.exp(-number))
    else:
        exp_number = math.exp(number)
        result = exp_number / (1.0 + exp_number)
    return min(1.0, max(0.0, result))


def convert_normalized_boxes_to_pixel_xyxy(
    boxes: object,
    *,
    width: int,
    height: int,
) -> tuple[tuple[float, float, float, float], ...]:
    """Convert normalized ``cx, cy, width, height`` boxes to finite pixel XYXY.

    Grounding DINO emits normalized center-format boxes. Clipping is intentional:
    the public contract describes source-pixel coordinates, and detector proposals
    just outside an image still remain useful evidence rather than being rejected.
    """
    if width <= 0 or height <= 0:
        return ()
    rows = _strip_batch(boxes, expected_width=4)
    if not _is_sequence(rows):
        return ()
    converted: list[tuple[float, float, float, float]] = []
    for row in rows:
        if not _is_sequence(row) or len(row) < 4:
            continue
        cx = _finite_number(row[0])
        cy = _finite_number(row[1])
        box_width = abs(_finite_number(row[2]))
        box_height = abs(_finite_number(row[3]))
        left = min(float(width), max(0.0, (cx - box_width / 2.0) * width))
        top = min(float(height), max(0.0, (cy - box_height / 2.0) * height))
        right = min(float(width), max(0.0, (cx + box_width / 2.0) * width))
        bottom = min(float(height), max(0.0, (cy + box_height / 2.0) * height))
        converted.append((left, top, right, bottom))
    return tuple(converted)


def rank_target_aligned_proposals(
    logits: object,
    boxes: object,
    *,
    width: int,
    height: int,
    token_spans: Mapping[str, Sequence[int]] | None = None,
    phrases: Sequence[str] = TARGET_PHRASES,
) -> tuple[RankedProposal, ...]:
    """Rank every query by sigmoid detector confidence aligned to target phrases.

    No score threshold is applied. For each query, the phrase with the strongest
    token confidence wins; all queries then receive a deterministic confidence
    ordering (descending confidence, then original query order).
    """
    logit_rows = _strip_batch(logits)
    box_rows = _strip_batch(boxes, expected_width=4)
    if not _is_sequence(logit_rows) or not _is_sequence(box_rows):
        return ()
    phrase_values = tuple(str(phrase) for phrase in phrases)
    if not phrase_values:
        return ()
    converted_boxes = convert_normalized_boxes_to_pixel_xyxy(
        box_rows,
        width=width,
        height=height,
    )
    ranked: list[RankedProposal] = []
    for query_index, row in enumerate(logit_rows):
        if query_index >= len(converted_boxes):
            break
        if _is_sequence(row):
            values = tuple(_sigmoid(item) for item in row)
        else:
            values = (_sigmoid(row),)
        if not values:
            continue
        phrase_scores: list[tuple[float, str]] = []
        for phrase in phrase_values:
            if token_spans is None and len(values) == len(phrase_values):
                indexes = (phrase_values.index(phrase),)
            else:
                indexes = token_spans.get(phrase, ()) if token_spans is not None else ()
            selected = tuple(
                values[index] for index in indexes if 0 <= index < len(values)
            )
            score = max(selected, default=max(values))
            phrase_scores.append((score, phrase))
        confidence, phrase = max(
            phrase_scores,
            key=lambda item: (item[0], -phrase_values.index(item[1])),
        )
        ranked.append(
            RankedProposal(
                confidence=_finite_number(confidence),
                phrase=phrase,
                box_xyxy=converted_boxes[query_index],
                query_index=query_index,
            )
        )
    ranked.sort(key=lambda proposal: (-proposal.confidence, proposal.query_index))
    return tuple(ranked[:MAX_DETECTIONS])


def _extract_output(outputs: object, name: str) -> object:
    value: object = getattr(outputs, name, None)
    if value is not None:
        return value
    if _is_mapping(outputs):
        return outputs.get(name)
    return None


def _phrase_token_spans(tokenizer: object) -> dict[str, tuple[int, ...]]:
    """Return token positions overlapping each phrase in the approved prompt."""
    if not _is_callable(tokenizer):
        return {}
    try:
        encoded = tokenizer(
            PROMPT,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
    except (AttributeError, TypeError, ValueError):
        return {}
    if not _is_mapping(encoded):
        return {}
    offset_value = encoded.get("offset_mapping")
    if not _is_sequence(offset_value):
        return {}
    offsets = offset_value
    first_value = offsets[0] if offsets else None
    if len(offsets) == 1 and _is_sequence(first_value):
        first_item = first_value[0] if first_value else None
        if _is_sequence(first_item):
            offsets = first_value
    phrase_ranges: dict[str, tuple[int, int]] = {}
    search_start = 0
    for phrase in TARGET_PHRASES:
        start = PROMPT.find(phrase, search_start)
        if start < 0:
            continue
        phrase_ranges[phrase] = (start, start + len(phrase))
        search_start = start + len(phrase)
    spans: dict[str, tuple[int, ...]] = {}
    for phrase, (phrase_start, phrase_end) in phrase_ranges.items():
        indexes: list[int] = []
        for index, offset in enumerate(offsets):
            if not _is_sequence(offset) or len(offset) < 2:
                continue
            start = _finite_number(offset[0])
            end = _finite_number(offset[1])
            if end > phrase_start and start < phrase_end:
                indexes.append(index)
        spans[phrase] = tuple(indexes)
    return spans


def build_annotation(
    sample: SampleManifest,
    processor: object,
    model: object,
    image: object,
    device: Literal["cpu", "cuda"] = "cpu",
) -> AnnotationRecord:
    if not _is_callable(processor):
        raise TypeError("processor is not callable")
    processed = processor(images=image, text=PROMPT, return_tensors="pt")
    to_device: object = getattr(processed, "to", None)
    if _is_callable(to_device):
        processed = to_device(device)
    if not _is_callable(model):
        raise TypeError("model is not callable")
    if _is_mapping(processed):
        outputs = model(**processed)
    else:
        outputs = model(processed)
    logits = _extract_output(outputs, "logits")
    boxes = _extract_output(outputs, "pred_boxes")
    if logits is None or boxes is None:
        raise ValueError("detector output missing required tensors")
    tokenizer: object = getattr(processor, "tokenizer", None)
    spans = _phrase_token_spans(tokenizer) if tokenizer is not None else None
    proposals = rank_target_aligned_proposals(
        logits,
        boxes,
        width=sample.width,
        height=sample.height,
        token_spans=spans,
    )
    detections = tuple(
        Detection(
            rank=index,
            phrase=proposal.phrase,
            confidence=proposal.confidence,
            box_xyxy=proposal.box_xyxy,
        )
        for index, proposal in enumerate(proposals, start=1)
    )
    max_confidence = max(
        (detection.confidence for detection in detections),
        default=0.0,
    )
    return AnnotationRecord(
        sample_id=sample.sample_id,
        detections=detections,
        max_confidence=max_confidence,
    )


def _json_bytes(model: object) -> bytes:
    dump: object = getattr(model, "model_dump", None)
    if not _is_callable(dump):
        raise TypeError("contract model cannot be serialized")
    payload = dump(mode="json")
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def _write_atomic(destination: Path, model: object) -> None:
    _ = destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    payload = _json_bytes(model)
    try:
        with partial.open("wb") as handle:
            _ = handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _ = partial.replace(destination)
    finally:
        _ = partial.unlink(missing_ok=True)


def _valid_annotation(path: Path, sample_id: str) -> bool:
    try:
        record = AnnotationRecord.model_validate_json(path.read_bytes())
    except (OSError, ValueError, TypeError):
        return False
    return record.sample_id == sample_id


def _failure_path(paths: EvaluationPaths, manifest_path: Path) -> Path:
    stable_name = hashlib.sha256(manifest_path.name.encode("utf-8")).hexdigest()
    return paths.errors / f"{stable_name}.json"


def _write_failure(
    paths: EvaluationPaths,
    manifest_path: Path,
    *,
    code: str,
    sample_id: str | None,
) -> None:
    try:
        failure = StageFailure(stage="annotate", code=code, sample_id=sample_id)
        _write_atomic(_failure_path(paths, manifest_path), failure)
    except (OSError, TypeError, ValueError):
        # A failure record must never expose the underlying path/content. If the
        # volume itself is unavailable there is nowhere safe to report it.
        return


def load_runtime(
    paths: EvaluationPaths,
) -> tuple[object, object, object, Literal["cpu", "cuda"]]:
    """Load detector dependencies lazily and choose one shared execution device."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    cache_dir = paths.cache / "huggingface"
    _ = cache_dir.mkdir(parents=True, exist_ok=True)
    processor_class: object = getattr(transformers, "AutoProcessor", None)
    processor_factory: object = getattr(processor_class, "from_pretrained", None)
    if not _is_callable(processor_factory):
        raise ImportError("transformers processor loader unavailable")
    model_class: object = getattr(
        transformers,
        "AutoModelForZeroShotObjectDetection",
        None,
    )
    model_factory: object = getattr(model_class, "from_pretrained", None)
    if not _is_callable(model_factory):
        raise ImportError("transformers detector loader unavailable")
    processor = processor_factory(CHECKPOINT, cache_dir=str(cache_dir))
    model = model_factory(
        CHECKPOINT,
        cache_dir=str(cache_dir),
    )
    device: Literal["cpu", "cuda"] = "cpu"
    cuda: object = getattr(torch, "cuda", None)
    is_available: object = getattr(cuda, "is_available", None)
    if _is_callable(is_available):
        available = is_available()
        if isinstance(available, bool) and available:
            device = "cuda"
    to_device: object = getattr(model, "to", None)
    if _is_callable(to_device):
        _ = to_device(device)
    eval_method: object = getattr(model, "eval", None)
    if _is_callable(eval_method):
        _ = eval_method()
    return torch, processor, model, device


def _manifest_files(paths: EvaluationPaths) -> tuple[Path, ...]:
    return tuple(sorted(paths.manifests.glob("*.json"), key=lambda path: path.name))


def _emit_stage_event(
    event: AggregateEvent,
    summary: StageSummary | None = None,
    *,
    attempted: int | None = None,
    completed: int | None = None,
    skipped: int | None = None,
    failed: int | None = None,
    processed: int = 0,
) -> None:
    if summary is not None:
        attempted = summary.attempted
        completed = summary.completed
        skipped = summary.skipped
        failed = summary.failed
    if (
        attempted is None
        or completed is None
        or skipped is None
        or failed is None
    ):
        raise ValueError("stage event counts are required")
    emit_event(
        event,
        stage="annotate",
        attempted=attempted,
        completed=completed,
        skipped=skipped,
        failed=failed,
        processed=processed,
    )


def _complete(
    summary: StageSummary,
    *,
    processed: int = 0,
) -> StageSummary:
    _emit_stage_event("complete", summary, processed=processed)
    return summary


def annotate(paths: EvaluationPaths) -> StageSummary:
    """Resume annotation, retaining all detector proposals including weak ones."""
    paths.ensure()
    manifest_paths = _manifest_files(paths)
    attempted = completed = skipped = failed = 0
    pending: list[tuple[Path, SampleManifest]] = []
    for manifest_path in manifest_paths:
        attempted += 1
        try:
            sample = SampleManifest.model_validate_json(manifest_path.read_bytes())
        except (OSError, ValueError, TypeError):
            failed += 1
            _write_failure(
                paths,
                manifest_path,
                code="manifest_invalid",
                sample_id=None,
            )
            continue
        destination = paths.annotation_path(sample.sample_id)
        if destination.exists() and _valid_annotation(destination, sample.sample_id):
            skipped += 1
            continue
        pending.append((manifest_path, sample))
    _emit_stage_event(
        "start",
        attempted=attempted,
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
    processed = 0
    if not pending:
        return _complete(
            StageSummary(
                attempted=attempted,
                completed=completed,
                skipped=skipped,
                failed=failed,
            )
        )
    _emit_stage_event(
        "model_loading",
        attempted=attempted,
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
    try:
        torch, processor, model, device = load_runtime(paths)
    except Exception:
        # Keep model-loading diagnostics out of stdout and failure records.
        for manifest_path, sample in pending:
            failed += 1
            _write_failure(
                paths,
                manifest_path,
                code="model_unavailable",
                sample_id=sample.sample_id,
            )
            processed += 1
            if processed % 25 == 0:
                _emit_stage_event(
                    "progress",
                    attempted=attempted,
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                    processed=processed,
                )
        return _complete(
            StageSummary(
                attempted=attempted,
                completed=completed,
                skipped=skipped,
                failed=failed,
            ),
            processed=processed,
        )
    no_grad: object = getattr(torch, "no_grad", None)
    _emit_stage_event(
        "model_ready",
        attempted=attempted,
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
    try:
        image_module = importlib.import_module("PIL.Image")
    except Exception:
        for manifest_path, sample in pending:
            failed += 1
            _write_failure(
                paths,
                manifest_path,
                code="image_loader_unavailable",
                sample_id=sample.sample_id,
            )
            processed += 1
            if processed % 25 == 0:
                _emit_stage_event(
                    "progress",
                    attempted=attempted,
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                    processed=processed,
                )
        return _complete(
            StageSummary(
                attempted=attempted,
                completed=completed,
                skipped=skipped,
                failed=failed,
            ),
            processed=processed,
        )
    image_opener: object = getattr(image_module, "open", None)
    if not _is_callable(image_opener):
        for manifest_path, sample in pending:
            failed += 1
            _write_failure(
                paths,
                manifest_path,
                code="image_loader_unavailable",
                sample_id=sample.sample_id,
            )
            processed += 1
            if processed % 25 == 0:
                _emit_stage_event(
                    "progress",
                    attempted=attempted,
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                    processed=processed,
                )
        return _complete(
            StageSummary(
                attempted=attempted,
                completed=completed,
                skipped=skipped,
                failed=failed,
            ),
            processed=processed,
        )
    for manifest_path, sample in pending:
        try:
            image_path = paths.image_path(sample.image_relative_path).resolve()
            _ = image_path.relative_to(paths.images.resolve())
            if not image_path.is_file():
                raise FileNotFoundError
            source_image = image_opener(image_path)
            if not _is_image_source(source_image):
                raise TypeError("Pillow image source unavailable")
            context: _ContextManager
            if _is_callable(no_grad):
                candidate = no_grad()
                if not _is_context_manager(candidate):
                    raise TypeError("torch no_grad context unavailable")
                context = candidate
            else:
                context = NullContext()
            with source_image, context:
                image = source_image.convert("RGB")
                record = build_annotation(sample, processor, model, image, device)
            _write_atomic(paths.annotation_path(sample.sample_id), record)
        except Exception:
            failed += 1
            _write_failure(
                paths,
                manifest_path,
                code="annotation_failed",
                sample_id=sample.sample_id,
            )
        else:
            completed += 1
        processed += 1
        if processed % 25 == 0:
            _emit_stage_event(
                "progress",
                attempted=attempted,
                completed=completed,
                skipped=skipped,
                failed=failed,
                processed=processed,
            )
    return _complete(
        StageSummary(
            attempted=attempted,
            completed=completed,
            skipped=skipped,
            failed=failed,
        ),
        processed=processed,
    )


class NullContext:
    def __enter__(self) -> NullContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None
