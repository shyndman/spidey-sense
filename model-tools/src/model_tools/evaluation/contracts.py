"""Strict schema-versioned records shared by evaluation pipeline stages.

These models are deliberately immutable and reject unknown keys or weakly typed
values. JSON arrays are represented as tuples in Python so records cannot be
mutated after validation.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from .paths import EvaluationPaths

__all__ = [
    "AnnotationRecord",
    "Detection",
    "EvaluationModel",
    "EvaluationPaths",
    "SampleManifest",
    "ScoreRecord",
    "StageFailure",
    "StageSummary",
]

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Pixel = Annotated[float, Field(ge=0.0)]


class EvaluationModel(BaseModel):
    """Base configuration for every persisted evaluation record."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class SampleManifest(EvaluationModel):
    """Provenance and deterministic split assignment for one corpus sample."""

    schema_version: Literal[1] = 1
    sample_id: str
    source: Literal["inaturalist", "coco2017"]
    source_id: str
    source_category: str
    expected_presence: Literal["positive", "hard_negative", "broad_negative"]
    source_url: str
    license: str
    image_relative_path: str
    sha256: str
    perceptual_hash: str
    duplicate_group: str
    split: Literal["calibration", "test"]
    width: PositiveInt
    height: PositiveInt

    @field_validator(
        "sample_id",
        "source_id",
        "source_category",
        "source_url",
        "license",
        "image_relative_path",
        "sha256",
        "perceptual_hash",
        "duplicate_group",
    )
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Reject empty provenance values while preserving strict strings."""

        if not value:
            raise ValueError("must not be empty")
        return value


class Detection(EvaluationModel):
    """One detector box, ranked by confidence and aligned to a fixed phrase."""

    schema_version: Literal[1] = 1
    rank: Annotated[PositiveInt, Field(le=20)]
    phrase: str
    confidence: Probability
    box_xyxy: tuple[Pixel, Pixel, Pixel, Pixel]

    @field_validator("phrase")
    @classmethod
    def non_empty_phrase(cls, value: str) -> str:
        """Require a meaningful target phrase for each detector result."""

        if not value:
            raise ValueError("must not be empty")
        return value


class AnnotationRecord(EvaluationModel):
    """Top twenty target-aligned detections for one manifest sample."""

    schema_version: Literal[1] = 1
    sample_id: str
    detections: tuple[Detection, ...] = Field(max_length=20)
    max_confidence: Probability

    @field_validator("sample_id")
    @classmethod
    def non_empty_sample_id(cls, value: str) -> str:
        """Reject records that cannot be joined to a manifest."""

        if not value:
            raise ValueError("must not be empty")
        return value


class ScoreRecord(EvaluationModel):
    """One model's 1,000 output probabilities for one sample."""

    schema_version: Literal[2] = 2
    model_id: str
    sample_id: str
    probabilities: tuple[Probability, ...] = Field(min_length=1000, max_length=1000)
    blocked_score: Probability
    top_index: Annotated[NonNegativeInt, Field(le=999)]

    @field_validator("model_id", "sample_id")
    @classmethod
    def non_empty_join_id(cls, value: str) -> str:
        """Reject records that cannot be joined to a model and manifest."""

        if not value:
            raise ValueError("must not be empty")
        return value


class StageFailure(EvaluationModel):
    """Opaque aggregate-stage failure metadata safe to persist under ``errors``."""

    schema_version: Literal[1] = 1
    stage: str
    code: str
    sample_id: str | None = None

    @field_validator("stage", "code")
    @classmethod
    def non_empty_failure_text(cls, value: str) -> str:
        """Keep failure identifiers stable and useful without content or URLs."""

        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("sample_id")
    @classmethod
    def non_empty_failure_sample_id(cls, value: str | None) -> str | None:
        """Allow an opaque sample identifier, but never an empty one."""

        if value == "":
            raise ValueError("must not be empty when provided")
        return value


class StageSummary(EvaluationModel):
    """Numeric aggregate counts emitted by an evaluation stage."""

    schema_version: Literal[1] = 1
    attempted: NonNegativeInt
    completed: NonNegativeInt
    skipped: NonNegativeInt
    failed: NonNegativeInt

    @model_validator(mode="after")
    def counts_partition_attempts(self) -> StageSummary:
        """Ensure every attempted item has exactly one terminal outcome."""

        if self.completed + self.skipped + self.failed != self.attempted:
            raise ValueError("completed + skipped + failed must equal attempted")
        return self
