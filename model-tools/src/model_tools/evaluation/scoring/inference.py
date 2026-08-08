"""Eager ONNX Runtime loading and one-sample scoring."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from model_tools.metadata import ArtifactMetadata

from ..acquisition.models import SampleManifest
from ..storage.layout import EvaluationPaths
from .models import InferenceSessionProtocol, LoadedModel, ScoreRecord
from .preprocessing import PROBABILITY_COUNT, preprocess_image, softmax_logits


class _OnnxRuntime(Protocol):
    def InferenceSession(
        self,
        path: str,
        *,
        providers: list[str],
    ) -> InferenceSessionProtocol: ...


ort = cast(_OnnxRuntime, cast(object, importlib.import_module("onnxruntime")))


def load_model(paths: EvaluationPaths, model_id: str) -> LoadedModel:
    bundle = paths.model_bundle(model_id)
    metadata_path = bundle / f"{model_id}.metadata.json"
    metadata = ArtifactMetadata.model_validate_json(metadata_path.read_bytes())
    if metadata.model.id != model_id:
        raise ValueError("model metadata ID mismatch")
    model_path = bundle / metadata.model.filename
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return LoadedModel(metadata=metadata, session=session)


def score_sample(
    model: LoadedModel, manifest: SampleManifest, image_path: Path
) -> ScoreRecord:
    tensor = preprocess_image(image_path, model.metadata.input)
    outputs = model.session.run(
        [model.metadata.output.name], {model.metadata.input.name: tensor}
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
        probabilities=tuple(probabilities),
        blocked_score=blocked_score,
        top_index=int(np.argmax(row)),
    )
