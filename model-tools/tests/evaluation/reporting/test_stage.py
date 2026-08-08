"""Focused numeric contract tests for aggregate evaluation reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

from model_tools.evaluation.acquisition.models import SampleManifest
from model_tools.evaluation.annotation.models import AnnotationRecord, Detection
from model_tools.evaluation.reporting import build_model_report
from model_tools.evaluation.reporting.models import (
    AnnotatedScoredSample,
    ModelReportRequest,
)
from model_tools.evaluation.scoring.models import ScoreRecord
from model_tools.evaluation.storage.layout import EvaluationPaths

ExpectedPresence = Literal["positive", "hard_negative", "broad_negative"]
Split = Literal["calibration", "test"]


_MODEL_ID = "evaluation-model"


def _manifest(
    sample_id: str,
    *,
    category: str,
    expected_presence: ExpectedPresence,
    split: Split,
) -> SampleManifest:
    return SampleManifest(
        sample_id=sample_id,
        source="inaturalist",
        source_id=f"observation-{sample_id}",
        source_category=category,
        expected_presence=expected_presence,
        source_url="https://example.invalid/metadata",
        license="CC BY",
        image_relative_path=f"images/{sample_id}.jpg",
        sha256=(sample_id.encode().hex() + "0" * 64)[:64],
        perceptual_hash="0" * 16,
        duplicate_group=f"group-{sample_id}",
        split=split,
        width=100,
        height=100,
    )


def _annotation(
    sample_id: str,
    confidence: float,
    box: tuple[float, float, float, float],
) -> AnnotationRecord:
    return AnnotationRecord(
        sample_id=sample_id,
        detections=(
            Detection(
                rank=1,
                phrase="proxy",
                confidence=confidence,
                box_xyxy=box,
            ),
        ),
        max_confidence=confidence,
    )


def _score(sample_id: str, blocked_score: float) -> ScoreRecord:
    probabilities = tuple((index + 1) / 1001 for index in range(1000))
    return ScoreRecord(
        model_id=_MODEL_ID,
        sample_id=sample_id,
        probabilities=probabilities,
        blocked_score=blocked_score,
        top_index=0,
    )


def _write_record(
    directory: Path,
    record: SampleManifest | AnnotationRecord | ScoreRecord,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.sample_id}.json"
    _ = path.write_text(record.model_dump_json(), encoding="utf-8")


def _write_fixture(paths: EvaluationPaths) -> None:
    records = (
        AnnotatedScoredSample(
            manifest=_manifest(
                "cal-positive",
                category="proxy-group-a",
                expected_presence="positive",
                split="calibration",
            ),
            annotation=_annotation("cal-positive", 0.05, (0, 0, 10, 10)),
            score=_score("cal-positive", 0.2),
        ),
        AnnotatedScoredSample(
            manifest=_manifest(
                "cal-negative",
                category="proxy-background",
                expected_presence="broad_negative",
                split="calibration",
            ),
            annotation=_annotation("cal-negative", 0.25, (0, 0, 20, 20)),
            score=_score("cal-negative", 0.7),
        ),
        AnnotatedScoredSample(
            manifest=_manifest(
                "test-positive",
                category="proxy-group-b",
                expected_presence="positive",
                split="test",
            ),
            annotation=_annotation("test-positive", 0.95, (0, 0, 50, 50)),
            score=_score("test-positive", 0.8),
        ),
        AnnotatedScoredSample(
            manifest=_manifest(
                "test-negative",
                category="proxy-background",
                expected_presence="broad_negative",
                split="test",
            ),
            annotation=AnnotationRecord(
                sample_id="test-negative",
                detections=(),
                max_confidence=0.0,
            ),
            score=_score("test-negative", 0.1),
        ),
    )
    for record in records:
        _write_record(paths.manifests, record.manifest)
        _write_record(paths.annotations, record.annotation)
        _write_record(paths.model_scores(_MODEL_ID), record.score)


def test_report_keeps_split_curves_and_strata_aggregate_only(
    tmp_path: Path,
) -> None:
    paths = EvaluationPaths(root=tmp_path)
    _write_fixture(paths)
    report = build_model_report(ModelReportRequest(paths=paths, model_id=_MODEL_ID))

    assert report.stage_summary.attempted == 4
    assert report.stage_summary.completed == 4
    assert report.stage_summary.skipped == 0
    assert report.stage_summary.failed == 0
    curves = report.threshold_curves
    assert [item.threshold for item in curves.calibration] == [0.2, 0.7]
    assert curves.calibration[0].tp == 1
    assert curves.calibration[0].fn == 0
    assert curves.calibration[0].fp == 1
    assert curves.calibration[0].tn == 0
    assert curves.calibration[1].tp == 0
    assert curves.calibration[1].fn == 1
    assert [item.threshold for item in curves.test] == [0.1, 0.8]
    assert curves.test[0].tp == 1
    assert curves.test[0].fn == 0
    assert curves.test[0].fp == 1
    assert curves.test[0].tn == 0
    assert curves.test[1].tp == 1
    assert curves.test[1].fn == 0
    assert curves.test[1].fp == 0
    assert curves.test[1].tn == 1
    assert "proxy-group-a" in report.by_source_category.root
    assert "positive" in report.by_expected_presence.root

    deciles = report.detector_confidence_deciles
    expected: object = cast(
        object,
        json.loads(
            (Path(__file__).with_name("expected_model_report.json")).read_text(
                encoding="utf-8"
            )
        ),
    )
    dumped: object = report.model_dump(mode="json")
    assert dumped == expected
    assert len(deciles) == 10
    assert deciles[0].count == 1
    assert deciles[0].detector_miss_count == 1
    assert deciles[9].count == 1
    assert deciles[9].mean_absolute_pixel_area == 2500.0
    assert report.detector_miss_count == 1

    area_bins = report.top_box_area_bins
    assert area_bins.absolute_pixel_area[1].count == 2
    assert area_bins.absolute_pixel_area[2].count == 1
    assert area_bins.relative_image_area[1].count == 2
    assert area_bins.relative_image_area[4].count == 1
    assert area_bins.detector_miss_count == 1
    assert paths.model_report_path(_MODEL_ID).is_file()
    assert not list(paths.reports.glob("*.part"))
    serialized: object = cast(
        object,
        json.loads(paths.model_report_path(_MODEL_ID).read_text(encoding="utf-8")),
    )
    assert serialized == expected

    def keys(value: object) -> list[str]:
        if isinstance(value, dict):
            mapping = cast(dict[object, object], value)
            return [key for key in mapping if isinstance(key, str)] + [
                nested
                for child in mapping.values()
                if isinstance(child, (dict, list))
                for nested in keys(cast(object, child))
            ]
        if isinstance(value, list):
            items = cast(list[object], value)
            return [
                nested
                for child in items
                if isinstance(child, (dict, list))
                for nested in keys(cast(object, child))
            ]
        return []

    all_keys = keys(dumped)
    for forbidden in (
        "sample_id",
        "source_url",
        "image_relative_path",
        "probabilities",
        "phrase",
    ):
        assert forbidden not in all_keys


def test_report_accounts_for_missing_invalid_and_unmatched_records(
    tmp_path: Path,
) -> None:
    paths = EvaluationPaths(root=tmp_path)
    manifest = _manifest(
        "missing",
        category="proxy-unmatched",
        expected_presence="hard_negative",
        split="test",
    )
    _write_record(paths.manifests, manifest)
    paths.annotations.mkdir(parents=True, exist_ok=True)
    _ = (paths.annotations / "invalid.json").write_text("not json", encoding="utf-8")
    paths.model_scores(_MODEL_ID).mkdir(parents=True, exist_ok=True)
    _write_record(paths.model_scores(_MODEL_ID), _score("extra", 0.4))

    report = build_model_report(ModelReportRequest(paths=paths, model_id=_MODEL_ID))

    expected_failure_fields: dict[str, object] = {
        "stage_summary": {
            "attempted": 4,
            "completed": 0,
            "skipped": 0,
            "failed": 4,
        },
        "record_accounting": {
            "manifest_records": 1,
            "annotation_records": 1,
            "score_records": 1,
            "joined_records": 0,
            "failure_count": 4,
        },
        "failures": {
            "total": 4,
            "by_code": {
                "invalid_annotation": 1,
                "missing_annotation": 1,
                "missing_score": 1,
                "unmatched_score": 1,
            },
        },
        "sample_count": 0,
        "detector_miss_count": 0,
    }
    failure_dump = report.model_dump(mode="json")
    assert {
        key: failure_dump[key] for key in expected_failure_fields
    } == expected_failure_fields
