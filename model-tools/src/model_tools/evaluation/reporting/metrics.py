"""Threshold confusion and operating-point metrics."""

from __future__ import annotations

import math

from .models import (
    Confusion,
    Measurement,
    OperatingPoint,
    ScoredSample,
    ScoreObservation,
    TestMeasurements,
)


def confusion(
    values: list[ScoreObservation] | tuple[ScoreObservation, ...], threshold: float
) -> Confusion:
    tp = fn = fp = tn = 0
    for value in values:
        predicted_positive = value.score >= threshold
        if value.is_positive and predicted_positive:
            tp += 1
        elif value.is_positive:
            fn += 1
        elif predicted_positive:
            fp += 1
        else:
            tn += 1
    positives, negatives, predicted = tp + fn, fp + tn, tp + fp
    return Confusion(
        threshold=threshold,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        tpr=tp / positives if positives else None,
        fpr=fp / negatives if negatives else None,
        precision=tp / predicted if predicted else None,
    )


def _calibration_values(
    samples: tuple[ScoredSample, ...],
) -> tuple[ScoreObservation, ...]:
    return tuple(
        ScoreObservation(
            is_positive=item.manifest.expected_presence == "positive",
            score=item.score.blocked_score,
        )
        for item in samples
        if item.manifest.split == "calibration"
        and item.manifest.expected_presence in {"positive", "broad_negative"}
    )


def _required_rate(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"calibration {name} is undefined")
    return value


def _f1(record: Confusion) -> float:
    denominator = 2 * record.tp + record.fp + record.fn
    return 2 * record.tp / denominator if denominator else 0.0


def _select_thresholds(samples: tuple[ScoredSample, ...]) -> dict[str, float]:
    values = _calibration_values(samples)
    if not values:
        raise ValueError("calibration target and ordinary-negative scores are required")
    observed = {value.score for value in values}
    thresholds = sorted({*observed, math.nextafter(max(observed), math.inf)})
    candidates = [(threshold, confusion(values, threshold)) for threshold in thresholds]
    maximum = max(
        candidates,
        key=lambda item: (
            _f1(item[1]),
            _required_rate(item[1].tpr, "TPR"),
            -_required_rate(item[1].fpr, "FPR"),
            item[0],
        ),
    )[0]
    ordinary = [
        item for item in candidates if _required_rate(item[1].fpr, "FPR") <= 0.01
    ]
    target = [item for item in candidates if _required_rate(item[1].tpr, "TPR") >= 0.97]
    if not ordinary or not target:
        raise ValueError("calibration scores cannot satisfy standard objectives")
    return {
        "maximum_calibration_f1": maximum,
        "ordinary_negative_fpr_at_most_1_percent": min(item[0] for item in ordinary),
        "target_tpr_at_least_97_percent": max(item[0] for item in target),
        "fixed_0_5": 0.5,
    }


def _measurement(scores: list[float], threshold: float) -> Measurement:
    count = sum(score >= threshold for score in scores)
    return Measurement(
        count=count, total=len(scores), rate=count / len(scores) if scores else None
    )


def operating_points(samples: tuple[ScoredSample, ...]) -> dict[str, OperatingPoint]:
    thresholds = _select_thresholds(samples)
    test_scores = {
        presence: [
            item.score.blocked_score
            for item in samples
            if item.manifest.split == "test"
            and item.manifest.expected_presence == presence
        ]
        for presence in ("positive", "hard_negative", "broad_negative")
    }
    calibration = _calibration_values(samples)
    return {
        name: OperatingPoint(
            threshold=threshold,
            calibration=confusion(calibration, threshold),
            test=TestMeasurements(
                target=_measurement(test_scores["positive"], threshold),
                lookalike=_measurement(test_scores["hard_negative"], threshold),
                ordinary_negative=_measurement(
                    test_scores["broad_negative"], threshold
                ),
            ),
        )
        for name, threshold in thresholds.items()
    }


def threshold_curves(
    samples: tuple[ScoredSample, ...],
) -> tuple[tuple[Confusion, ...], tuple[Confusion, ...]]:
    curves: list[tuple[Confusion, ...]] = []
    for split in ("calibration", "test"):
        values = tuple(
            ScoreObservation(
                is_positive=item.manifest.expected_presence == "positive",
                score=item.score.blocked_score,
            )
            for item in samples
            if item.manifest.split == split
        )
        curves.append(
            tuple(
                confusion(values, threshold)
                for threshold in sorted({value.score for value in values})
            )
        )
    return curves[0], curves[1]


__all__ = ["confusion", "operating_points", "threshold_curves"]
