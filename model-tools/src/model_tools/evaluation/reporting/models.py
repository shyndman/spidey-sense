"""Typed report graph and cross-stage rows."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ConfigDict, Field, NonNegativeInt, RootModel

from ..acquisition.models import SampleManifest
from ..annotation.models import AnnotationRecord
from ..base import EvaluationModel, MutableEvaluationModel
from ..scoring.models import ScoreRecord
from ..storage.layout import EvaluationPaths


class ScoreObservation(EvaluationModel):
    is_positive: bool
    score: float


class ScoredSample(EvaluationModel):
    manifest: SampleManifest
    score: ScoreRecord


class AnnotatedScoredSample(EvaluationModel):
    manifest: SampleManifest
    score: ScoreRecord
    annotation: AnnotationRecord


class Confusion(EvaluationModel):
    threshold: float
    tp: NonNegativeInt
    fn: NonNegativeInt
    fp: NonNegativeInt
    tn: NonNegativeInt
    tpr: float | None
    fpr: float | None
    precision: float | None


class Measurement(EvaluationModel):
    count: NonNegativeInt
    total: NonNegativeInt
    rate: float | None


class TestMeasurements(EvaluationModel):
    target: Measurement
    lookalike: Measurement
    ordinary_negative: Measurement


class OperatingPoint(EvaluationModel):
    threshold: float
    calibration: Confusion
    test: TestMeasurements


class ScoreStatistics(EvaluationModel):
    count: NonNegativeInt
    mean_blocked_score: float | None
    minimum_blocked_score: float | None
    maximum_blocked_score: float | None


class ConfidenceDecile(EvaluationModel):
    lower: float
    upper: float
    count: NonNegativeInt
    detector_miss_count: NonNegativeInt
    mean_confidence: float | None
    mean_absolute_pixel_area: float | None
    mean_relative_image_area: float | None
    mean_blocked_score: float | None
    positive_count: NonNegativeInt
    negative_count: NonNegativeInt


class AreaBucket(EvaluationModel):
    lower: float
    upper: float | None
    count: NonNegativeInt
    positive_count: NonNegativeInt
    negative_count: NonNegativeInt
    mean_blocked_score: float | None


class AreaBins(EvaluationModel):
    absolute_pixel_area: tuple[AreaBucket, ...]
    relative_image_area: tuple[AreaBucket, ...]
    detector_miss_count: NonNegativeInt


class ReportStageSummary(EvaluationModel):
    attempted: NonNegativeInt
    completed: NonNegativeInt
    skipped: NonNegativeInt
    failed: NonNegativeInt


class RecordAccounting(EvaluationModel):
    manifest_records: NonNegativeInt
    annotation_records: NonNegativeInt
    score_records: NonNegativeInt
    joined_records: NonNegativeInt
    failure_count: NonNegativeInt


class FailureCounts(RootModel[dict[str, NonNegativeInt]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class FailureSummary(EvaluationModel):
    total: NonNegativeInt
    by_code: FailureCounts


class ThresholdCurves(EvaluationModel):
    calibration: tuple[Confusion, ...]
    test: tuple[Confusion, ...]


class OperatingPoints(RootModel[dict[str, OperatingPoint]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class ScoreStatisticsMap(RootModel[dict[str, ScoreStatistics]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class ModelReport(EvaluationModel):
    schema_version: Literal[2] = 2
    model_id: str
    stage_summary: ReportStageSummary
    record_accounting: RecordAccounting
    failures: FailureSummary
    sample_count: NonNegativeInt
    detector_miss_count: NonNegativeInt
    threshold_curves: ThresholdCurves
    standard_operating_points: OperatingPoints
    by_source_category: ScoreStatisticsMap
    by_expected_presence: ScoreStatisticsMap
    detector_confidence_deciles: tuple[ConfidenceDecile, ...]
    top_box_area_bins: AreaBins


class ModelReports(RootModel[dict[str, ModelReport]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class EvaluationReports(EvaluationModel):
    schema_version: Literal[2] = 2
    models: ModelReports


class ModelReportRequest(EvaluationModel):
    paths: EvaluationPaths
    model_id: str


class ReportsRequest(EvaluationModel):
    paths: EvaluationPaths


class MutableAreaBucket(MutableEvaluationModel):
    lower: float
    upper: float | None
    count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    score_values: list[float] = Field(default_factory=list)


class MutableConfidenceBucket(MutableEvaluationModel):
    lower: float
    upper: float
    count: int = 0
    detector_miss_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    confidence_values: list[float] = Field(default_factory=list)
    absolute_area_values: list[float] = Field(default_factory=list)
    relative_area_values: list[float] = Field(default_factory=list)
    score_values: list[float] = Field(default_factory=list)


__all__ = [
    "ScoreObservation",
    "ScoredSample",
    "AnnotatedScoredSample",
    "Confusion",
    "Measurement",
    "TestMeasurements",
    "OperatingPoint",
    "ScoreStatistics",
    "ConfidenceDecile",
    "AreaBucket",
    "AreaBins",
    "ReportStageSummary",
    "RecordAccounting",
    "FailureCounts",
    "FailureSummary",
    "ThresholdCurves",
    "OperatingPoints",
    "ScoreStatisticsMap",
    "ModelReports",
    "ModelReport",
    "EvaluationReports",
    "ModelReportRequest",
    "ReportsRequest",
    "MutableAreaBucket",
    "MutableConfidenceBucket",
]
