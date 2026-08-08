"""Annotation stage package."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final, Literal, Protocol, TypeGuard

from ..acquisition.models import SampleManifest
from .models import AnnotationRecord, Detection, RankedProposal

TARGET_PHRASES: Final[tuple[str, ...]] = (
    "spider",
    "tarantula",
    "black widow",
    "wolf spider",
    "garden spider",
    "barn spider",
)
PROMPT: Final[str] = (
    "spider. tarantula. black widow. wolf spider. garden spider. barn spider."
)
MAX_DETECTIONS: Final[int] = 20


class _Callable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


def _is_callable(value: object) -> TypeGuard[_Callable]:
    return callable(value)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


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


def _rank_proposal(
    row: object,
    *,
    query_index: int,
    box_xyxy: tuple[float, float, float, float],
    phrase_values: tuple[str, ...],
    token_spans: Mapping[str, Sequence[int]] | None,
) -> RankedProposal | None:
    if _is_sequence(row):
        values = tuple(_sigmoid(item) for item in row)
    else:
        values = (_sigmoid(row),)
    if not values:
        return None
    phrase_scores: list[tuple[float, str]] = []
    for phrase in phrase_values:
        if token_spans is None and len(values) == len(phrase_values):
            indexes = (phrase_values.index(phrase),)
        else:
            indexes = token_spans.get(phrase, ()) if token_spans is not None else ()
        selected = tuple(values[index] for index in indexes if 0 <= index < len(values))
        score = max(selected, default=max(values))
        phrase_scores.append((score, phrase))
    confidence, phrase = max(
        phrase_scores,
        key=lambda item: (item[0], -phrase_values.index(item[1])),
    )
    return RankedProposal(
        confidence=_finite_number(confidence),
        phrase=phrase,
        box_xyxy=box_xyxy,
        query_index=query_index,
    )


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
        proposal = _rank_proposal(
            row,
            query_index=query_index,
            box_xyxy=converted_boxes[query_index],
            phrase_values=phrase_values,
            token_spans=token_spans,
        )
        if proposal is not None:
            ranked.append(proposal)
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
