"""Numeric tests for standard calibration and held-out operating points."""

from __future__ import annotations

from model_tools.evaluation.contracts import SampleManifest, ScoreRecord
from model_tools.evaluation.operating_points import operating_points

_MODEL_ID = "evaluation-model"


def _sample(
    sample_id: str,
    *,
    presence: str,
    split: str,
    score: float,
) -> tuple[SampleManifest, ScoreRecord]:
    manifest = SampleManifest(
        sample_id=sample_id,
        source="coco2017",
        source_id=sample_id,
        source_category=f"neutral-{presence}",
        expected_presence=presence,
        source_url="https://example.invalid/metadata",
        license="CC-BY-4.0",
        image_relative_path=f"images/{sample_id}.jpg",
        sha256="a" * 64,
        perceptual_hash="0" * 16,
        duplicate_group=sample_id,
        split=split,
        width=100,
        height=100,
    )
    record = ScoreRecord(
        model_id=_MODEL_ID,
        sample_id=sample_id,
        probabilities=(0.001,) * 1000,
        blocked_score=score,
        top_index=0,
    )
    return manifest, record


def test_operating_points_exclude_lookalikes_from_calibration_selection() -> None:
    samples = [
        _sample("cal-target-1", presence="positive", split="calibration", score=0.9),
        _sample("cal-target-2", presence="positive", split="calibration", score=0.8),
        _sample(
            "cal-ordinary-1",
            presence="broad_negative",
            split="calibration",
            score=0.2,
        ),
        _sample(
            "cal-ordinary-2",
            presence="broad_negative",
            split="calibration",
            score=0.1,
        ),
        _sample(
            "cal-lookalike",
            presence="hard_negative",
            split="calibration",
            score=0.95,
        ),
        _sample("test-target", presence="positive", split="test", score=0.85),
        _sample(
            "test-lookalike",
            presence="hard_negative",
            split="test",
            score=0.9,
        ),
        _sample(
            "test-ordinary",
            presence="broad_negative",
            split="test",
            score=0.3,
        ),
    ]

    result = operating_points(samples)

    for objective in (
        "maximum_calibration_f1",
        "ordinary_negative_fpr_at_most_1_percent",
        "target_tpr_at_least_97_percent",
    ):
        assert result[objective]["threshold"] == 0.8
    assert result["maximum_calibration_f1"]["calibration"] == {
        "threshold": 0.8,
        "tp": 2,
        "fn": 0,
        "fp": 0,
        "tn": 2,
        "tpr": 1.0,
        "fpr": 0.0,
        "precision": 1.0,
    }
    assert result["maximum_calibration_f1"]["test"] == {
        "target": {"count": 1, "total": 1, "rate": 1.0},
        "lookalike": {"count": 1, "total": 1, "rate": 1.0},
        "ordinary_negative": {"count": 0, "total": 1, "rate": 0.0},
    }


def test_zero_fpr_objective_allows_threshold_above_highest_score() -> None:
    samples = [
        _sample("cal-target", presence="positive", split="calibration", score=0.8),
        _sample(
            "cal-ordinary",
            presence="broad_negative",
            split="calibration",
            score=0.9,
        ),
    ]

    result = operating_points(samples)

    threshold = result["ordinary_negative_fpr_at_most_1_percent"]["threshold"]
    assert threshold > 0.9
    assert (
        result["ordinary_negative_fpr_at_most_1_percent"]["calibration"]["fpr"] == 0.0
    )
