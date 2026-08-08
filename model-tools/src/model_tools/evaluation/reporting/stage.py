"""Reporting orchestration."""

from __future__ import annotations

from collections import Counter
from typing import Literal, NamedTuple

from ..acquisition.models import SampleManifest
from ..annotation.models import AnnotationRecord
from ..scoring.models import ScoreRecord
from ..storage.json import REPORT_JSON_PROFILE, write_model
from .metrics import operating_points, threshold_curves
from .models import (
    AnnotatedScoredSample,
    EvaluationReports,
    FailureCounts,
    FailureSummary,
    ModelReport,
    ModelReportRequest,
    ModelReports,
    OperatingPoints,
    RecordAccounting,
    ReportsRequest,
    ReportStageSummary,
    ScoredSample,
    ScoreStatistics,
    ScoreStatisticsMap,
    ThresholdCurves,
)
from .records import RecordReadResult, read_records, typed_records
from .stratification import area_bins, confidence_deciles

StatsKey = Literal["source_category", "expected_presence"]


class _ReportRecords(NamedTuple):
    manifests: RecordReadResult[SampleManifest]
    annotations: RecordReadResult[AnnotationRecord]
    scores: RecordReadResult[ScoreRecord]
    manifest_values: dict[str, SampleManifest]
    annotation_values: dict[str, AnnotationRecord]
    score_values: dict[str, ScoreRecord]
    failures: Counter[str]


def _read_report_records(request: ModelReportRequest) -> _ReportRecords:
    paths = request.paths
    paths.ensure()
    manifests = read_records(paths.manifests, SampleManifest)
    annotations = read_records(paths.annotations, AnnotationRecord)
    scores = read_records(paths.model_scores(request.model_id), ScoreRecord)
    failures: Counter[str] = Counter()
    failures.update(manifests.failures)
    failures.update(annotations.failures)
    failures.update(scores.failures)
    return _ReportRecords(
        manifests,
        annotations,
        scores,
        typed_records(manifests, SampleManifest),
        typed_records(annotations, AnnotationRecord),
        typed_records(scores, ScoreRecord),
        failures,
    )


def _join_report_samples(
    records: _ReportRecords, model_id: str
) -> tuple[tuple[AnnotatedScoredSample, ...], tuple[ScoredSample, ...]]:
    scores = records.score_values
    wrong = [
        sample_id for sample_id, score in scores.items() if score.model_id != model_id
    ]
    for sample_id in wrong:
        del scores[sample_id]
    records.failures["wrong_model_score"] += len(wrong)
    joined: list[AnnotatedScoredSample] = []
    for sample_id in sorted(records.manifest_values):
        if sample_id not in records.annotation_values:
            records.failures["missing_annotation"] += 1
        if sample_id not in scores:
            records.failures["missing_score"] += 1
        if sample_id in records.annotation_values and sample_id in scores:
            joined.append(
                AnnotatedScoredSample(
                    manifest=records.manifest_values[sample_id],
                    annotation=records.annotation_values[sample_id],
                    score=scores[sample_id],
                )
            )
    records.failures["unmatched_annotation"] += len(
        set(records.annotation_values) - set(records.manifest_values)
    )
    records.failures["unmatched_score"] += len(
        set(scores) - set(records.manifest_values)
    )
    joined_samples = tuple(joined)
    scored_samples = tuple(
        ScoredSample(manifest=item.manifest, score=item.score)
        for item in joined_samples
    )
    return joined_samples, scored_samples


def _stats_key(manifest: SampleManifest, key: StatsKey) -> str:
    if key == "source_category":
        return manifest.source_category
    return manifest.expected_presence


def _stats(samples: tuple[ScoredSample, ...], key: StatsKey) -> ScoreStatisticsMap:
    groups: dict[str, list[float]] = {}
    for item in samples:
        groups.setdefault(_stats_key(item.manifest, key), []).append(
            item.score.blocked_score
        )
    return ScoreStatisticsMap(
        {
            key: ScoreStatistics(
                count=len(values),
                mean_blocked_score=sum(values) / len(values) if values else None,
                minimum_blocked_score=min(values) if values else None,
                maximum_blocked_score=max(values) if values else None,
            )
            for key, values in sorted(groups.items())
        }
    )


def build_model_report(request: ModelReportRequest) -> ModelReport:
    records = _read_report_records(request)
    joined_samples, scored_samples = _join_report_samples(records, request.model_id)
    total_failures = sum(records.failures.values())
    calibration, test = threshold_curves(scored_samples)
    try:
        points = operating_points(scored_samples)
    except ValueError:
        points = {}
    report = ModelReport(
        model_id=request.model_id,
        stage_summary=ReportStageSummary(
            attempted=len(joined_samples) + total_failures,
            completed=len(joined_samples),
            skipped=0,
            failed=total_failures,
        ),
        record_accounting=RecordAccounting(
            manifest_records=records.manifests.attempted,
            annotation_records=records.annotations.attempted,
            score_records=records.scores.attempted,
            joined_records=len(joined_samples),
            failure_count=total_failures,
        ),
        failures=FailureSummary(
            total=total_failures,
            by_code=FailureCounts(
                {key: value for key, value in records.failures.items() if value}
            ),
        ),
        sample_count=len(joined_samples),
        detector_miss_count=sum(
            not item.annotation.detections for item in joined_samples
        ),
        threshold_curves=ThresholdCurves(calibration=calibration, test=test),
        standard_operating_points=OperatingPoints(points),
        by_source_category=_stats(scored_samples, "source_category"),
        by_expected_presence=_stats(scored_samples, "expected_presence"),
        detector_confidence_deciles=confidence_deciles(joined_samples),
        top_box_area_bins=area_bins(joined_samples),
    )
    write_model(
        request.paths.model_report_path(request.model_id),
        report,
        profile=REPORT_JSON_PROFILE,
    )
    return report


def run(request: ReportsRequest) -> EvaluationReports:
    from ..registry import registered_model_ids

    model_ids: tuple[str, ...] = registered_model_ids(request.paths).root
    return EvaluationReports(
        models=ModelReports(
            {
                model_id: build_model_report(
                    ModelReportRequest(paths=request.paths, model_id=model_id)
                )
                for model_id in model_ids
            }
        )
    )


__all__ = ["build_model_report", "run"]
