"""Numeric tests for standard calibration and held-out operating points."""

from typing import Literal, TypedDict, TypeGuard, cast

from model_tools.evaluation.contracts import SampleManifest, ScoreRecord
from model_tools.evaluation.operating_points import operating_points

_MODEL_ID = "evaluation-model"
_Presence = Literal["positive", "hard_negative", "broad_negative"]
_Split = Literal["calibration", "test"]


class _Confusion(TypedDict):
    threshold: float
    tp: int
    fn: int
    fp: int
    tn: int
    tpr: float | None
    fpr: float | None
    precision: float | None


class _Measurement(TypedDict):
    count: int
    total: int
    rate: float | None


class _TestMeasurements(TypedDict):
    target: _Measurement
    lookalike: _Measurement
    ordinary_negative: _Measurement


class _OperatingPoint(TypedDict):
    threshold: float
    calibration: _Confusion
    test: _TestMeasurements


def _is_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    object_dict = cast(dict[object, object], value)
    return all(isinstance(key, str) for key in object_dict)


def _required_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{field} must be a float")
    return value


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int")
    return value


def _optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _required_float(value, field)


def _measurement(value: object, field: str) -> _Measurement:
    if not _is_object_dict(value):
        raise TypeError(f"{field} must be an object")
    return {
        "count": _required_int(value.get("count"), f"{field}.count"),
        "total": _required_int(value.get("total"), f"{field}.total"),
        "rate": _optional_float(value.get("rate"), f"{field}.rate"),
    }


def _confusion(value: object, field: str) -> _Confusion:
    if not _is_object_dict(value):
        raise TypeError(f"{field} must be an object")
    return {
        "threshold": _required_float(
            value.get("threshold"), f"{field}.threshold"
        ),
        "tp": _required_int(value.get("tp"), f"{field}.tp"),
        "fn": _required_int(value.get("fn"), f"{field}.fn"),
        "fp": _required_int(value.get("fp"), f"{field}.fp"),
        "tn": _required_int(value.get("tn"), f"{field}.tn"),
        "tpr": _optional_float(value.get("tpr"), f"{field}.tpr"),
        "fpr": _optional_float(value.get("fpr"), f"{field}.fpr"),
        "precision": _optional_float(
            value.get("precision"), f"{field}.precision"
        ),
    }


def _test_measurements(value: object, field: str) -> _TestMeasurements:
    if not _is_object_dict(value):
        raise TypeError(f"{field} must be an object")
    return {
        "target": _measurement(value.get("target"), f"{field}.target"),
        "lookalike": _measurement(value.get("lookalike"), f"{field}.lookalike"),
        "ordinary_negative": _measurement(
            value.get("ordinary_negative"), f"{field}.ordinary_negative"
        ),
    }


def _operating_point(result: dict[str, object], objective: str) -> _OperatingPoint:
    point = result.get(objective)
    if not _is_object_dict(point):
        raise TypeError(f"{objective} must be an object")
    return {
        "threshold": _required_float(point.get("threshold"), "threshold"),
        "calibration": _confusion(
            point.get("calibration"), f"{objective}.calibration"
        ),
        "test": _test_measurements(point.get("test"), f"{objective}.test"),
    }


def _sample(
    sample_id: str,
    *,
    presence: _Presence,
    split: _Split,
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
        assert _operating_point(result, objective)["threshold"] == 0.8
    maximum_calibration_f1 = _operating_point(result, "maximum_calibration_f1")
    assert maximum_calibration_f1["calibration"] == {
        "threshold": 0.8,
        "tp": 2,
        "fn": 0,
        "fp": 0,
        "tn": 2,
        "tpr": 1.0,
        "fpr": 0.0,
        "precision": 1.0,
    }
    assert maximum_calibration_f1["test"] == {
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

    threshold = _operating_point(
        result, "ordinary_negative_fpr_at_most_1_percent"
    )["threshold"]
    assert threshold > 0.9
    assert (
        _operating_point(result, "ordinary_negative_fpr_at_most_1_percent")[
            "calibration"
        ]["fpr"]
        == 0.0
    )
