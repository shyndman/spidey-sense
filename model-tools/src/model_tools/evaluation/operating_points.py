"""Select shared calibration objectives and measure held-out model behavior."""

from __future__ import annotations

import math
from typing import TypedDict

from .contracts import SampleManifest, ScoreRecord


class Confusion(TypedDict):
    threshold: float
    tp: int
    fn: int
    fp: int
    tn: int
    tpr: float | None
    fpr: float | None
    precision: float | None


def confusion(values: list[tuple[bool, float]], threshold: float) -> Confusion:
    """Compute one binary confusion record at a fixed threshold."""

    tp = fn = fp = tn = 0
    for is_positive, score in values:
        predicted_positive = score >= threshold
        if is_positive and predicted_positive:
            tp += 1
        elif is_positive:
            fn += 1
        elif predicted_positive:
            fp += 1
        else:
            tn += 1
    positives = tp + fn
    negatives = fp + tn
    predicted_positives = tp + fp
    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / positives if positives else None,
        "fpr": fp / negatives if negatives else None,
        "precision": tp / predicted_positives if predicted_positives else None,
    }


def _required_rate(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"calibration {name} is undefined")
    return value


def _f1(record: Confusion) -> float:
    denominator = 2 * record["tp"] + record["fp"] + record["fn"]
    return (2 * record["tp"] / denominator) if denominator else 0.0


def _calibration_values(
    samples: list[tuple[SampleManifest, ScoreRecord]],
) -> list[tuple[bool, float]]:
    return [
        (manifest.expected_presence == "positive", score.blocked_score)
        for manifest, score in samples
        if manifest.split == "calibration"
        and manifest.expected_presence in {"positive", "broad_negative"}
    ]


def _select_thresholds(
    samples: list[tuple[SampleManifest, ScoreRecord]],
) -> dict[str, float]:
    values = _calibration_values(samples)
    if not values:
        raise ValueError("calibration target and ordinary-negative scores are required")
    observed_thresholds = {score for _, score in values}
    thresholds = sorted(
        {
            *observed_thresholds,
            math.nextafter(max(observed_thresholds), math.inf),
        }
    )
    candidates = [(threshold, confusion(values, threshold)) for threshold in thresholds]
    maximum_f1 = max(
        candidates,
        key=lambda item: (
            _f1(item[1]),
            _required_rate(item[1]["tpr"], "TPR"),
            -_required_rate(item[1]["fpr"], "FPR"),
            item[0],
        ),
    )[0]
    ordinary_candidates = [
        item for item in candidates if _required_rate(item[1]["fpr"], "FPR") <= 0.01
    ]
    target_candidates = [
        item for item in candidates if _required_rate(item[1]["tpr"], "TPR") >= 0.97
    ]
    if not ordinary_candidates or not target_candidates:
        raise ValueError("calibration scores cannot satisfy standard objectives")
    return {
        "maximum_calibration_f1": maximum_f1,
        "ordinary_negative_fpr_at_most_1_percent": min(
            item[0] for item in ordinary_candidates
        ),
        "target_tpr_at_least_97_percent": max(item[0] for item in target_candidates),
        "fixed_0_5": 0.5,
    }


def _measurement(scores: list[float], threshold: float) -> dict[str, object]:
    positive_count = sum(score >= threshold for score in scores)
    return {
        "count": positive_count,
        "total": len(scores),
        "rate": positive_count / len(scores) if scores else None,
    }


def operating_points(
    samples: list[tuple[SampleManifest, ScoreRecord]],
) -> dict[str, object]:
    """Select standard calibration thresholds and measure every held-out set."""

    thresholds = _select_thresholds(samples)
    test_scores = {
        presence: [
            score.blocked_score
            for manifest, score in samples
            if manifest.split == "test" and manifest.expected_presence == presence
        ]
        for presence in ("positive", "hard_negative", "broad_negative")
    }
    calibration_values = _calibration_values(samples)
    return {
        name: {
            "threshold": threshold,
            "calibration": confusion(calibration_values, threshold),
            "test": {
                "target": _measurement(test_scores["positive"], threshold),
                "lookalike": _measurement(test_scores["hard_negative"], threshold),
                "ordinary_negative": _measurement(
                    test_scores["broad_negative"], threshold
                ),
            },
        }
        for name, threshold in thresholds.items()
    }


__all__ = ["confusion", "operating_points"]
