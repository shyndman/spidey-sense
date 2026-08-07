"""Model-isolated ONNX scoring for the persistent evaluation corpus."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from model_tools.metadata import ArtifactMetadata, InputMetadata

from .contracts import SampleManifest, ScoreRecord, StageFailure, StageSummary
from .paths import EvaluationPaths

PROBABILITY_COUNT = 1_000
SCORE_FAILURE_PREFIX = "SCORE_"


class _InferenceSession(Protocol):
    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[object]: ...


class _OnnxRuntime(Protocol):
    def InferenceSession(
        self,
        path: str,
        *,
        providers: list[str],
    ) -> _InferenceSession: ...


ort = cast(_OnnxRuntime, cast(object, importlib.import_module("onnxruntime")))


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A validated metadata contract paired with its ONNX Runtime session."""

    metadata: ArtifactMetadata
    session: _InferenceSession


def softmax_logits(logits: NDArray[np.float32]) -> tuple[float, ...]:
    """Return a finite, normalized probability row from 1,000 logits."""

    if logits.shape != (PROBABILITY_COUNT,) or not bool(np.all(np.isfinite(logits))):
        raise ValueError("invalid logits")
    shifted: NDArray[np.float64] = logits.astype(np.float64) - float(np.max(logits))
    exponentials: NDArray[np.float64] = np.exp(shifted)
    total_value: np.float64 = np.sum(exponentials, dtype=np.float64)
    total = float(total_value)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid softmax sum")
    probabilities: NDArray[np.float64] = exponentials / total
    if not bool(np.all(np.isfinite(probabilities))):
        raise ValueError("invalid softmax probabilities")
    return tuple(float(value) for value in probabilities.flat)


def preprocess_image(path: Path, metadata: InputMetadata) -> NDArray[np.float32]:
    """Apply the fixed contain, bilinear, black-pad, NCHW evaluation transform.

    Evaluation inputs are corpus JPEGs, so alpha compositing cannot affect this
    validation-only implementation. The geometry and normalization match the
    checked-in model contract used to compare every registered checkpoint.
    """
    if not math.isfinite(metadata.pixel_scale) or metadata.pixel_scale <= 0:
        raise ValueError("invalid pixel scale")

    _, channels, target_height, target_width = metadata.shape
    if channels != 3:
        raise ValueError("evaluation input must have three channels")
    with Image.open(path) as source:
        _ = source.load()
        image = source.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("invalid image dimensions")

    scale = min(target_width / image.width, target_height / image.height)
    width = min(target_width, max(1, math.floor(image.width * scale + 0.5)))
    height = min(target_height, max(1, math.floor(image.height * scale + 0.5)))
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    contained = np.zeros((target_height, target_width, channels), dtype=np.float32)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    contained[top : top + height, left : left + width] = pixels
    contained *= np.float32(metadata.pixel_scale)

    mean = np.asarray(metadata.mean, dtype=np.float32)
    deviation = np.asarray(metadata.standard_deviation, dtype=np.float32)
    if not bool(np.all(np.isfinite(mean))) or not bool(
        np.all(np.isfinite(deviation) & (deviation > 0))
    ):
        raise ValueError("invalid normalization")
    normalized = (contained - mean) / deviation
    tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    if tensor.shape != (1, channels, target_height, target_width) or not bool(
        np.all(np.isfinite(tensor))
    ):
        raise ValueError("invalid preprocessed tensor")
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _load_model(paths: EvaluationPaths, model_id: str) -> LoadedModel:
    bundle = paths.model_bundle(model_id)
    metadata_path = bundle / f"{model_id}.metadata.json"
    metadata = ArtifactMetadata.model_validate_json(metadata_path.read_bytes())
    if metadata.model.id != model_id:
        raise ValueError("model metadata ID mismatch")
    model_path = bundle / metadata.model.filename
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return LoadedModel(metadata=metadata, session=session)


def _score_sample(
    model: LoadedModel,
    manifest: SampleManifest,
    image_path: Path,
) -> ScoreRecord:
    tensor = preprocess_image(image_path, model.metadata.input)
    outputs = model.session.run(
        [model.metadata.output.name],
        {model.metadata.input.name: tensor},
    )
    if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
        raise ValueError("invalid inference output")
    logits = cast(NDArray[np.float32], outputs[0])
    if logits.shape != (1, PROBABILITY_COUNT):
        raise ValueError("invalid inference output shape")
    row: NDArray[np.float32] = logits[0, :]
    probabilities = softmax_logits(row)
    blocked_indices = tuple(record.index for record in model.metadata.classes.blocked)
    blocked_score = sum(probabilities[index] for index in blocked_indices)
    return ScoreRecord(
        model_id=model.metadata.model.id,
        sample_id=manifest.sample_id,
        probabilities=probabilities,
        blocked_score=blocked_score,
        top_index=int(np.argmax(row)),
    )


def _existing_score(path: Path, model_id: str, sample_id: str) -> bool:
    if not path.exists():
        return False
    try:
        score = ScoreRecord.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return False
    return score.model_id == model_id and score.sample_id == sample_id


def _write_failure(
    paths: EvaluationPaths,
    model_id: str,
    code: str,
    sample_id: str | None = None,
) -> None:
    failure = StageFailure(stage="score", code=code, sample_id=sample_id)
    try:
        failure_path = paths.model_error_path(model_id, sample_id)
    except ValueError:
        failure_path = paths.model_error_path(model_id)
        failure = StageFailure(stage="score", code=code)
    paths.write_json(failure_path, failure)


def _collect_pending(
    paths: EvaluationPaths,
    model_id: str,
) -> tuple[int, list[tuple[SampleManifest, Path]], int, int]:
    manifest_paths = tuple(sorted(paths.manifests.glob("*.json")))
    pending: list[tuple[SampleManifest, Path]] = []
    skipped = 0
    failed = 0
    for manifest_path in manifest_paths:
        try:
            manifest = SampleManifest.model_validate_json(manifest_path.read_bytes())
            score_path = paths.score_path(model_id, manifest.sample_id)
            _ = paths.model_error_path(model_id, manifest.sample_id)
        except (OSError, ValueError):
            failed += 1
            _write_failure(
                paths,
                model_id,
                f"{SCORE_FAILURE_PREFIX}INVALID_MANIFEST",
                manifest_path.stem,
            )
            continue
        if _existing_score(score_path, model_id, manifest.sample_id):
            skipped += 1
        else:
            pending.append((manifest, score_path))
    return len(manifest_paths), pending, skipped, failed


def _score_pending(
    paths: EvaluationPaths,
    model_id: str,
    model: LoadedModel,
    pending: list[tuple[SampleManifest, Path]],
) -> tuple[int, int]:
    completed = 0
    failed = 0
    for manifest, score_path in pending:
        try:
            image_path = paths.image_path(manifest.image_relative_path)
            score = _score_sample(model, manifest, image_path)
            paths.write_json(score_path, score)
        except (OSError, RuntimeError, ValueError):
            failed += 1
            _write_failure(
                paths,
                model_id,
                f"{SCORE_FAILURE_PREFIX}SAMPLE",
                manifest.sample_id,
            )
        else:
            completed += 1
    return completed, failed


def score_model(paths: EvaluationPaths, model_id: str) -> StageSummary:
    """Score every manifest for one registered model without emitting sample data."""

    paths.ensure_model(model_id)
    attempted, pending, skipped, failed = _collect_pending(paths, model_id)
    if not pending:
        return StageSummary(
            attempted=attempted,
            completed=0,
            skipped=skipped,
            failed=failed,
        )
    try:
        model = _load_model(paths, model_id)
    except (OSError, RuntimeError, ValueError):
        _write_failure(paths, model_id, f"{SCORE_FAILURE_PREFIX}MODEL_LOAD")
        return StageSummary(
            attempted=attempted,
            completed=0,
            skipped=skipped,
            failed=failed + len(pending),
        )

    completed, scoring_failures = _score_pending(paths, model_id, model, pending)
    return StageSummary(
        attempted=attempted,
        completed=completed,
        skipped=skipped,
        failed=failed + scoring_failures,
    )
