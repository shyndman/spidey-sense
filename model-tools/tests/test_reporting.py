"""Focused numeric contract tests for aggregate evaluation reporting."""

from __future__ import annotations

import json
from pathlib import Path

from model_tools.evaluation.contracts import (
    AnnotationRecord,
    Detection,
    SampleManifest,
    ScoreRecord,
)
from model_tools.evaluation.paths import EvaluationPaths
from model_tools.evaluation.reporting import _report_model

_MODEL_ID = "evaluation-model"


def _manifest(
    sample_id: str,
    *,
    category: str,
    expected_presence: str,
    split: str,
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
    path.write_text(record.model_dump_json(), encoding="utf-8")


def _write_fixture(paths: EvaluationPaths) -> None:
    records = (
        (
            _manifest(
                "cal-positive",
                category="target-class-a",
                expected_presence="positive",
                split="calibration",
            ),
            _annotation("cal-positive", 0.05, (0, 0, 10, 10)),
            _score("cal-positive", 0.2),
        ),
        (
            _manifest(
                "cal-negative",
                category="ordinary-negative",
                expected_presence="broad_negative",
                split="calibration",
            ),
            _annotation("cal-negative", 0.25, (0, 0, 20, 20)),
            _score("cal-negative", 0.7),
        ),
        (
            _manifest(
                "test-positive",
                category="target-class-b",
                expected_presence="positive",
                split="test",
            ),
            _annotation("test-positive", 0.95, (0, 0, 50, 50)),
            _score("test-positive", 0.8),
        ),
        (
            _manifest(
                "test-negative",
                category="ordinary-negative",
                expected_presence="broad_negative",
                split="test",
            ),
            AnnotationRecord(
                sample_id="test-negative",
                detections=(),
                max_confidence=0.0,
            ),
            _score("test-negative", 0.1),
        ),
    )
    for manifest, annotation, score in records:
        _write_record(paths.manifests, manifest)
        _write_record(paths.annotations, annotation)
        _write_record(paths.model_scores(_MODEL_ID), score)


def test_report_keeps_split_curves_and_strata_aggregate_only(tmp_path: Path) -> None:
    paths = EvaluationPaths(tmp_path)
    _write_fixture(paths)
    result = _report_model(paths, _MODEL_ID)

    assert result["stage_summary"] == {
        "attempted": 4,
        "completed": 4,
        "skipped": 0,
        "failed": 0,
    }
    curves = result["threshold_curves"]
    assert isinstance(curves, dict)
    assert [item["threshold"] for item in curves["calibration"]] == [0.2, 0.7]
    assert curves["calibration"][0]["tp"] == 1
    assert curves["calibration"][0]["fn"] == 0
    assert curves["calibration"][0]["fp"] == 1
    assert curves["calibration"][0]["tn"] == 0
    assert curves["calibration"][1]["tp"] == 0
    assert curves["calibration"][1]["fn"] == 1
    assert [item["threshold"] for item in curves["test"]] == [0.1, 0.8]
    assert curves["test"][0]["tp"] == 1
    assert curves["test"][0]["fn"] == 0
    assert curves["test"][0]["fp"] == 1
    assert curves["test"][0]["tn"] == 0
    assert curves["test"][1]["tp"] == 1
    assert curves["test"][1]["fn"] == 0
    assert curves["test"][1]["fp"] == 0
    assert curves["test"][1]["tn"] == 1
    assert "target-class-a" in result["by_source_category"]
    assert "positive" in result["by_expected_presence"]

    deciles = result["detector_confidence_deciles"]
    assert len(deciles) == 10
    assert deciles[0]["count"] == 1
    assert deciles[0]["detector_miss_count"] == 1
    assert deciles[9]["count"] == 1
    assert deciles[9]["mean_absolute_pixel_area"] == 2500.0
    assert result["detector_miss_count"] == 1

    area_bins = result["top_box_area_bins"]
    assert area_bins["absolute_pixel_area"][1]["count"] == 2
    assert area_bins["absolute_pixel_area"][2]["count"] == 1
    assert area_bins["relative_image_area"][1]["count"] == 2
    assert area_bins["relative_image_area"][4]["count"] == 1
    assert area_bins["detector_miss_count"] == 1
    assert paths.model_report_path(_MODEL_ID).is_file()
    assert not list(paths.reports.glob("*.part"))

    serialized = json.loads(
        paths.model_report_path(_MODEL_ID).read_text(encoding="utf-8")
    )
    assert serialized == result

    def keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return [key for key in value] + [
                nested for child in value.values() for nested in keys(child)
            ]
        if isinstance(value, list):
            return [nested for child in value for nested in keys(child)]
        return []

    all_keys = keys(result)
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
    paths = EvaluationPaths(tmp_path)
    manifest = _manifest(
        "missing",
        category="lookalike-class",
        expected_presence="hard_negative",
        split="test",
    )
    _write_record(paths.manifests, manifest)
    paths.annotations.mkdir(parents=True, exist_ok=True)
    (paths.annotations / "invalid.json").write_text("not json", encoding="utf-8")
    paths.model_scores(_MODEL_ID).mkdir(parents=True, exist_ok=True)
    _write_record(paths.model_scores(_MODEL_ID), _score("extra", 0.4))

    result = _report_model(paths, _MODEL_ID)

    assert result["record_accounting"]["joined_records"] == 0
    assert result["failures"]["total"] == 4
    assert result["failures"]["by_code"] == {
        "invalid_annotation": 1,
        "missing_annotation": 1,
        "missing_score": 1,
        "unmatched_score": 1,
    }
