"""Typed scoring stage records and runtime wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, NonNegativeInt, SkipValidation, field_validator

from model_tools.metadata import ArtifactMetadata

from ..acquisition.models import SampleManifest
from ..base import EvaluationModel, RuntimeEvaluationModel
from ..storage.layout import EvaluationPaths

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class InferenceSessionProtocol(Protocol):
    def run(
        self, output_names: list[str], input_feed: dict[str, NDArray[np.float32]]
    ) -> list[object]: ...


class ScoreRecord(EvaluationModel):
    schema_version: Literal[2] = 2
    model_id: str
    sample_id: str
    probabilities: tuple[Probability, ...] = Field(min_length=1000, max_length=1000)
    blocked_score: Probability
    top_index: Annotated[NonNegativeInt, Field(le=999)]

    @field_validator("model_id", "sample_id")
    @classmethod
    def non_empty_join_id(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class PendingScore(EvaluationModel):
    manifest: SampleManifest
    score_path: Path


class PendingScores(EvaluationModel):
    attempted: NonNegativeInt
    items: tuple[PendingScore, ...]
    skipped: NonNegativeInt
    failed: NonNegativeInt


class ScoreCounts(EvaluationModel):
    completed: NonNegativeInt
    failed: NonNegativeInt


class LoadedModel(RuntimeEvaluationModel):
    metadata: ArtifactMetadata
    session: SkipValidation[InferenceSessionProtocol]


class ScoringRequest(EvaluationModel):
    paths: EvaluationPaths
    model_id: str


__all__ = [
    "InferenceSessionProtocol",
    "LoadedModel",
    "PendingScore",
    "PendingScores",
    "ScoreCounts",
    "ScoreRecord",
    "ScoringRequest",
]
