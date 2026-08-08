"""Reporting-owned boundary model coverage."""

from pathlib import Path

from model_tools.evaluation.acquisition.models import SampleManifest
from model_tools.evaluation.reporting.models import (
    AnnotatedScoredSample,
    AreaBins,
    AreaBucket,
    ConfidenceDecile,
    Confusion,
    EvaluationReports,
    FailureCounts,
    FailureSummary,
    Measurement,
    ModelReport,
    ModelReportRequest,
    ModelReports,
    MutableAreaBucket,
    MutableConfidenceBucket,
    OperatingPoint,
    OperatingPoints,
    RecordAccounting,
    ReportsRequest,
    ReportStageSummary,
    ScoredSample,
    ScoreObservation,
    ScoreStatistics,
    ScoreStatisticsMap,
    ThresholdCurves,
)
from model_tools.evaluation.reporting.models import (
    TestMeasurements as ReportTestMeasurements,
)
from model_tools.evaluation.reporting.records import RecordReadResult
from model_tools.evaluation.storage.layout import EvaluationPaths
from pydantic import BaseModel


def test_reporting_graph_is_typed_and_round_trips() -> None:
    records: dict[str, SampleManifest] = {}
    confusion = Confusion(
        threshold=0.5, tp=1, fn=0, fp=0, tn=1, tpr=1.0, fpr=0.0, precision=1.0
    )
    measurement = Measurement(count=1, total=1, rate=1.0)
    test_measurements = ReportTestMeasurements(
        target=measurement, lookalike=measurement, ordinary_negative=measurement
    )
    report = ModelReport(
        model_id="model-proxy",
        stage_summary=ReportStageSummary(attempted=1, completed=1, skipped=0, failed=0),
        record_accounting=RecordAccounting(
            manifest_records=1,
            annotation_records=1,
            score_records=1,
            joined_records=1,
            failure_count=0,
        ),
        failures=FailureSummary(total=0, by_code=FailureCounts(root={})),
        sample_count=1,
        detector_miss_count=0,
        threshold_curves=ThresholdCurves(calibration=(confusion,), test=(confusion,)),
        standard_operating_points=OperatingPoints(
            root={
                "proxy": OperatingPoint(
                    threshold=0.5, calibration=confusion, test=test_measurements
                )
            }
        ),
        by_source_category=ScoreStatisticsMap(
            root={
                "proxy": ScoreStatistics(
                    count=1,
                    mean_blocked_score=0.2,
                    minimum_blocked_score=0.2,
                    maximum_blocked_score=0.2,
                )
            }
        ),
        by_expected_presence=ScoreStatisticsMap(root={}),
        detector_confidence_deciles=(
            ConfidenceDecile(
                lower=0.0,
                upper=1.0,
                count=1,
                detector_miss_count=0,
                mean_confidence=0.5,
                mean_absolute_pixel_area=1.0,
                mean_relative_image_area=0.1,
                mean_blocked_score=0.2,
                positive_count=1,
                negative_count=0,
            ),
        ),
        top_box_area_bins=AreaBins(
            absolute_pixel_area=(
                AreaBucket(
                    lower=0.0,
                    upper=None,
                    count=1,
                    positive_count=1,
                    negative_count=0,
                    mean_blocked_score=0.2,
                ),
            ),
            relative_image_area=(),
            detector_miss_count=0,
        ),
    )
    values: tuple[BaseModel, ...] = (
        ScoredSample.model_construct(),
        AnnotatedScoredSample.model_construct(),
        RecordReadResult[SampleManifest](records=records, failures={}, attempted=0),
        ScoreObservation(is_positive=True, score=0.2),
        confusion,
        measurement,
        test_measurements,
        OperatingPoint(threshold=0.5, calibration=confusion, test=test_measurements),
        ScoreStatistics(
            count=1,
            mean_blocked_score=0.2,
            minimum_blocked_score=0.2,
            maximum_blocked_score=0.2,
        ),
        ConfidenceDecile(
            lower=0.0,
            upper=1.0,
            count=1,
            detector_miss_count=0,
            mean_confidence=0.5,
            mean_absolute_pixel_area=1.0,
            mean_relative_image_area=0.1,
            mean_blocked_score=0.2,
            positive_count=1,
            negative_count=0,
        ),
        AreaBucket(
            lower=0.0,
            upper=None,
            count=1,
            positive_count=1,
            negative_count=0,
            mean_blocked_score=0.2,
        ),
        report.stage_summary,
        report.record_accounting,
        report.failures,
        report.failures.by_code,
        report.threshold_curves,
        report.standard_operating_points,
        report.by_source_category,
        report.by_expected_presence,
        report.top_box_area_bins,
        ModelReports(root={"proxy": report}),
        EvaluationReports(models=ModelReports(root={"proxy": report})),
        ModelReportRequest(
            paths=EvaluationPaths(root=Path("/proxy")), model_id="proxy"
        ),
        ReportsRequest(paths=EvaluationPaths(root=Path("/proxy"))),
        MutableAreaBucket(lower=0.0, upper=None),
        MutableConfidenceBucket(lower=0.0, upper=1.0),
        report,
    )
    assert all(isinstance(value, BaseModel) for value in values)
    assert ModelReport.model_validate_json(report.model_dump_json()) == report
    assert (
        EvaluationReports.model_validate_json(
            EvaluationReports(
                models=ModelReports(root={"proxy": report})
            ).model_dump_json()
        ).models.root["proxy"]
        == report
    )
