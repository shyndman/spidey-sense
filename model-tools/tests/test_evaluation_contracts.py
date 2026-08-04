"""Numeric-only tests for persisted evaluation contracts and paths."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from model_tools.evaluation import (
    AnnotationRecord,
    Detection,
    EvaluationPaths,
    SampleManifest,
    ScoreRecord,
    StageFailure,
    StageSummary,
    atomic_write_json,
    read_json,
)


def _manifest() -> SampleManifest:
    return SampleManifest(
        sample_id="sample-1",
        source="coco2017",
        source_id="42",
        source_category="broad_negative",
        expected_presence="broad_negative",
        source_url="https://example.invalid/metadata/42",
        license="CC-BY-4.0",
        image_relative_path="images/sample-1.jpg",
        sha256="a" * 64,
        perceptual_hash="0" * 16,
        duplicate_group="sample-1",
        split="calibration",
        width=32,
        height=24,
    )


def test_contracts_reject_unknown_and_weak_values() -> None:
    with pytest.raises(ValidationError):
        SampleManifest.model_validate({**_manifest().model_dump(), "unexpected": 1})
    with pytest.raises(ValidationError):
        SampleManifest.model_validate({**_manifest().model_dump(), "width": True})


def test_detection_and_annotation_bounds() -> None:
    detection = Detection(
        rank=1,
        phrase="target",
        confidence=0.5,
        box_xyxy=(0.0, 1.0, 10.0, 12.0),
    )
    record = AnnotationRecord(
        sample_id="sample-1",
        detections=(detection,),
        max_confidence=0.5,
    )
    assert record.detections[0].rank == 1
    with pytest.raises(ValidationError):
        Detection(
            rank=21,
            phrase="target",
            confidence=0.5,
            box_xyxy=(0.0, 1.0, 10.0, 12.0),
        )
    with pytest.raises(ValidationError):
        AnnotationRecord(
            sample_id="sample-1",
            detections=tuple(
                Detection(
                    rank=1,
                    phrase="target",
                    confidence=0.5,
                    box_xyxy=(0.0, 1.0, 10.0, 12.0),
                )
                for _ in range(21)
            ),
            max_confidence=0.5,
        )


def test_score_and_stage_lengths() -> None:
    score = ScoreRecord(
        sample_id="sample-1",
        probabilities=(0.001,) * 1000,
        blocked_score=0.2,
        top_index=3,
    )
    assert len(score.probabilities) == 1000
    with pytest.raises(ValidationError):
        ScoreRecord(
            sample_id="sample-1",
            probabilities=(0.001,) * 999,
            blocked_score=0.2,
            top_index=3,
        )
    assert StageFailure(stage="score", code="missing_output", sample_id="opaque")
    assert StageSummary(attempted=3, completed=1, skipped=1, failed=1)
    with pytest.raises(ValidationError):
        StageSummary(attempted=3, completed=1, skipped=1, failed=0)


def test_paths_create_layout_and_atomic_json(tmp_path: Path) -> None:
    paths = EvaluationPaths(tmp_path / "data")
    paths.ensure()
    expected = {
        "images",
        "manifests",
        "annotations",
        "scores",
        "errors",
        "reports",
        "downloads",
        "models",
        "cache",
        "tmp",
    }
    assert {path.name for path in paths.root.iterdir()} == expected
    assert paths.image_path("images/sample-1.jpg") == paths.images / "sample-1.jpg"
    destination = paths.manifest_path("sample-1")
    atomic_write_json(destination, _manifest())
    assert destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()
    loaded = read_json(destination, SampleManifest)
    assert isinstance(loaded, SampleManifest)
    assert loaded.sample_id == "sample-1"


def test_root_relative_image_path_matches_scoring_resolution(tmp_path: Path) -> None:
    paths = EvaluationPaths(tmp_path / "data")
    manifest = _manifest()
    resolved = paths.image_path(manifest.image_relative_path)
    assert manifest.image_relative_path.startswith("images/")
    assert resolved == paths.root / manifest.image_relative_path

    for invalid_path in (
        "images//sample-1.jpg",
        "images/nested/sample-1.jpg",
        "images/../sample-1.jpg",
        "images/foo\\bar.jpg",
        "images/",
    ):
        with pytest.raises(ValueError):
            paths.image_path(invalid_path)
