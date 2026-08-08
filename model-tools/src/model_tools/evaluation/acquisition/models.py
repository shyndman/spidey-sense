"""Typed domain, source-wire, and runtime records for acquisition."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict, PositiveInt, SkipValidation, field_validator

from ..base import EvaluationModel, MutableEvaluationModel, RuntimeEvaluationModel
from ..storage.layout import EvaluationPaths

Source = Literal["inaturalist", "coco2017"]
ExpectedPresence = Literal["positive", "hard_negative", "broad_negative"]
Split = Literal["calibration", "test"]


class AcquisitionFailure(RuntimeError):
    """A source item did not satisfy the corpus contract."""


class SampleManifest(EvaluationModel):
    """Provenance and deterministic split assignment for one corpus sample."""

    schema_version: Literal[1] = 1
    sample_id: str
    source: Source
    source_id: str
    source_category: str
    expected_presence: ExpectedPresence
    source_url: str
    license: str
    image_relative_path: str
    sha256: str
    perceptual_hash: str
    duplicate_group: str
    split: Split
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
        if not value:
            raise ValueError("must not be empty")
        return value


class AcquisitionRequest(EvaluationModel):
    """Input to the acquisition stage facade."""

    paths: EvaluationPaths


class Candidate(EvaluationModel):
    """One deterministic source item eligible for materialization."""

    source: Source
    source_id: str
    source_category: str
    expected_presence: ExpectedPresence
    source_url: str
    license: str
    image_name: str
    image_bytes: bytes | None = None
    archive_member: str | None = None


class AcceptedSample(EvaluationModel):
    """A candidate whose bytes and JPEG metadata satisfy the corpus contract."""

    candidate: Candidate
    sample_id: str
    image_path: Path
    sha256: str
    perceptual_hash: int
    width: PositiveInt
    height: PositiveInt
    duplicate_group: str = ""
    split: Split = "calibration"


class AcquisitionQuota(EvaluationModel):
    """One ordered source/category quota."""

    source: Source
    category: str
    expected_presence: ExpectedPresence
    quota: int


class CategoryProgress(MutableEvaluationModel):
    """Mutable counters for one source category."""

    attempted: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    materialization_attempted: int = 0


class Shortage(EvaluationModel):
    """Unfilled quota count for one category."""

    category: str
    count: int


class DecodedImageMetadata(EvaluationModel):
    """JPEG dimensions and perceptual hash."""

    width: PositiveInt
    height: PositiveInt
    perceptual_hash: int


class CandidateStream(RuntimeEvaluationModel):
    """Lazy source candidate stream and source-failure count."""

    candidates: SkipValidation[Iterator[Candidate]]
    source_failures: int


class CandidateBatch(EvaluationModel):
    """Finite source candidate batch and source-failure count."""

    candidates: tuple[Candidate, ...]
    source_failures: int


class Photo(EvaluationModel):
    """Tolerantly parsed iNaturalist photo wire row."""

    id: int | str | None = None
    url: str | None = None
    license: str | int | float | bool | None = None
    license_code: str | int | float | bool | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", frozen=True, strict=False
    )


class Observation(EvaluationModel):
    """Tolerantly parsed iNaturalist observation wire row."""

    id: int | str | None = None
    quality_grade: str | int | float | bool | None = None
    photos: tuple[Photo, ...] | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", frozen=True, strict=False
    )


class SourcePage(EvaluationModel):
    """Tolerantly parsed source page containing observation rows."""

    results: tuple[Observation, ...] | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", frozen=True, strict=False
    )


class ArchiveImage(EvaluationModel):
    """Tolerantly parsed COCO archive image row."""

    id: int | str | None = None
    file_name: str | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", frozen=True, strict=False
    )


class PhotoSelection(EvaluationModel):
    """Selected public photo metadata for one observation."""

    source_id: str
    image_url: str
    license: str


class PhotoOption(EvaluationModel):
    """Selected photo plus its source ordering key."""

    photo_id: str
    selection: PhotoSelection


class SourcePageResult(EvaluationModel):
    """Source page rows, fingerprint, and pagination state."""

    rows: tuple[Observation, ...]
    fingerprint: tuple[str, ...] | None
    has_next_page: bool


__all__ = [
    "AcceptedSample",
    "AcquisitionFailure",
    "AcquisitionQuota",
    "AcquisitionRequest",
    "ArchiveImage",
    "Candidate",
    "CandidateBatch",
    "CandidateStream",
    "CategoryProgress",
    "DecodedImageMetadata",
    "ExpectedPresence",
    "Observation",
    "Photo",
    "PhotoOption",
    "PhotoSelection",
    "SampleManifest",
    "Shortage",
    "Source",
    "SourcePage",
    "SourcePageResult",
    "Split",
]
