"""Aggregate threshold and detector-stratified evaluation reports.

The reporting stage only consumes strict contract records.  It never emits a
sample-level row: all output is grouped numeric evidence suitable for a later
threshold decision.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Final, Literal

from .contracts import AnnotationRecord, SampleManifest, ScoreRecord
from .operating_points import Confusion, confusion, operating_points
from .paths import EvaluationPaths

_CONFIDENCE_BIN_COUNT: Final[int] = 10
_RELATIVE_AREA_EDGES: Final[tuple[float, ...]] = (
    0.0,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)
_ABSOLUTE_AREA_EDGES: Final[tuple[float, ...]] = (
    0.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
)


@dataclass(slots=True)
class _AreaBucket:
    lower: float
    upper: float | None
    count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    score_sum: float = 0.0

    def add(self, is_positive: bool, blocked_score: float) -> None:
        self.count += 1
        self.positive_count += 1 if is_positive else 0
        self.negative_count += 0 if is_positive else 1
        self.score_sum += blocked_score

    def as_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "mean_blocked_score": (self.score_sum / self.count if self.count else None),
        }


@dataclass(slots=True)
class _ConfidenceBucket:
    lower: float
    upper: float
    count: int = 0
    detector_miss_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    confidence_values: list[float] = field(default_factory=list)
    absolute_area_values: list[float] = field(default_factory=list)
    relative_area_values: list[float] = field(default_factory=list)
    score_values: list[float] = field(default_factory=list)

    def add_miss(self) -> None:
        self.detector_miss_count += 1

    def add_hit(
        self,
        is_positive: bool,
        confidence: float,
        blocked_score: float,
        areas: tuple[float, float] | None,
    ) -> None:
        self.count += 1
        self.positive_count += 1 if is_positive else 0
        self.negative_count += 0 if is_positive else 1
        self.confidence_values.append(confidence)
        self.score_values.append(blocked_score)
        if areas is not None:
            self.absolute_area_values.append(areas[0])
            self.relative_area_values.append(areas[1])

    def as_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "detector_miss_count": self.detector_miss_count,
            "mean_confidence": _mean(self.confidence_values),
            "mean_absolute_pixel_area": _mean(self.absolute_area_values),
            "mean_relative_image_area": _mean(self.relative_area_values),
            "mean_blocked_score": _mean(self.score_values),
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


def _read_records[RecordT: (SampleManifest, AnnotationRecord, ScoreRecord)](
    directory: Path,
    model: type[RecordT],
) -> tuple[dict[str, RecordT], Counter[str], int]:
    """Read strict JSON records, returning records and aggregate failure codes."""
    records: dict[str, RecordT] = {}
    failures: Counter[str] = Counter()
    attempted = 0
    if model is SampleManifest:
        record_kind = "manifest"
    elif model is AnnotationRecord:
        record_kind = "annotation"
    elif model is ScoreRecord:
        record_kind = "score"
    else:
        raise ValueError("unsupported evaluation record model")
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        attempted += 1
        try:
            value = model.model_validate_json(path.read_bytes())
        except (OSError, ValueError, TypeError):
            failures[f"invalid_{record_kind}"] += 1
            continue
        sample_id = value.sample_id
        if sample_id in records:
            failures[f"duplicate_{record_kind}"] += 1
            continue
        records[sample_id] = value
    return records, failures, attempted


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    """Persist one aggregate JSON report without exposing a partial report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    payload = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    with part.open("w", encoding="utf-8") as handle:
        _ = handle.write(payload)
        handle.flush()
        _ = os.fsync(handle.fileno())
    os.replace(part, path)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _score_stats(values: list[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "mean_blocked_score": _mean(values),
        "minimum_blocked_score": min(values) if values else None,
        "maximum_blocked_score": max(values) if values else None,
    }


def _threshold_curves(
    samples: list[tuple[SampleManifest, ScoreRecord]],
) -> dict[str, list[Confusion]]:
    by_split: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for manifest, score in samples:
        by_split[manifest.split].append(
            (manifest.expected_presence == "positive", score.blocked_score)
        )
    curves: dict[str, list[Confusion]] = {}
    for split in ("calibration", "test"):
        values = by_split.get(split, [])
        thresholds = sorted({score for _, score in values})
        curves[split] = [confusion(values, threshold) for threshold in thresholds]
    return curves


def _category_stats(
    samples: list[tuple[SampleManifest, ScoreRecord]],
    key: Literal["source_category", "expected_presence"],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for manifest, score in samples:
        category = (
            manifest.source_category
            if key == "source_category"
            else manifest.expected_presence
        )
        grouped[category].append(score.blocked_score)
    return {category: _score_stats(grouped[category]) for category in sorted(grouped)}


def _detector_confidence(annotation: AnnotationRecord) -> float | None:
    if not annotation.detections:
        return None
    return max(float(detection.confidence) for detection in annotation.detections)


def _box_areas(
    manifest: SampleManifest,
    annotation: AnnotationRecord,
) -> tuple[float, float] | None:
    if not annotation.detections:
        return None
    detection = max(
        annotation.detections,
        key=lambda item: (item.confidence, -item.rank),
    )
    x1, y1, x2, y2 = detection.box_xyxy
    absolute = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    relative = absolute / float(manifest.width * manifest.height)
    return absolute, relative


def _area_buckets(
    samples: list[tuple[SampleManifest, AnnotationRecord, ScoreRecord]],
) -> dict[str, object]:
    edge_sets: dict[str, tuple[float, ...]] = {
        "absolute_pixel_area": _ABSOLUTE_AREA_EDGES,
        "relative_image_area": _RELATIVE_AREA_EDGES,
    }
    buckets: dict[str, list[_AreaBucket]] = {}
    for name, edges in edge_sets.items():
        buckets[name] = [
            _AreaBucket(
                lower=edges[index],
                upper=edges[index + 1] if index + 1 < len(edges) else None,
            )
            for index in range(len(edges) - 1)
        ]
        buckets[name].append(_AreaBucket(lower=edges[-1], upper=None))

    detector_miss_count = 0
    for manifest, annotation, score in samples:
        areas = _box_areas(manifest, annotation)
        if areas is None:
            detector_miss_count += 1
            continue
        absolute, relative = areas
        is_positive = manifest.expected_presence == "positive"
        for name, value in (
            ("absolute_pixel_area", absolute),
            ("relative_image_area", relative),
        ):
            edges = edge_sets[name]
            index = next(
                (
                    candidate
                    for candidate in range(len(edges) - 1)
                    if edges[candidate] <= value < edges[candidate + 1]
                ),
                len(edges) - 1,
            )
            buckets[name][index].add(is_positive, score.blocked_score)
    return {
        **{
            name: [bucket.as_dict() for bucket in area_buckets]
            for name, area_buckets in buckets.items()
        },
        "detector_miss_count": detector_miss_count,
    }


def _confidence_deciles(
    samples: list[tuple[SampleManifest, AnnotationRecord, ScoreRecord]],
) -> list[dict[str, object]]:
    buckets = [
        _ConfidenceBucket(
            lower=index / _CONFIDENCE_BIN_COUNT,
            upper=(index + 1) / _CONFIDENCE_BIN_COUNT,
        )
        for index in range(_CONFIDENCE_BIN_COUNT)
    ]
    for manifest, annotation, score in samples:
        confidence = _detector_confidence(annotation)
        if confidence is None:
            # Detector misses are retained in a numeric aggregate, not dropped.
            buckets[0].add_miss()
            continue
        index = min(
            _CONFIDENCE_BIN_COUNT - 1,
            int(confidence * _CONFIDENCE_BIN_COUNT),
        )
        buckets[index].add_hit(
            manifest.expected_presence == "positive",
            confidence,
            score.blocked_score,
            _box_areas(manifest, annotation),
        )
    return [bucket.as_dict() for bucket in buckets]


def _standard_operating_points(
    samples: list[tuple[SampleManifest, ScoreRecord]],
) -> dict[str, object]:
    """Return no objectives only when required calibration groups are absent."""

    calibration_presence = {
        manifest.expected_presence
        for manifest, _score in samples
        if manifest.split == "calibration"
    }
    if not {"positive", "broad_negative"}.issubset(calibration_presence):
        return {}
    return operating_points(samples)


def _report_model(paths: EvaluationPaths, model_id: str) -> dict[str, object]:
    """Join one model's records and emit deterministic aggregate evidence."""
    paths.ensure()
    manifests, manifest_failures, manifest_attempted = _read_records(
        paths.manifests,
        SampleManifest,
    )
    annotations, annotation_failures, annotation_attempted = _read_records(
        paths.annotations,
        AnnotationRecord,
    )
    scores, score_failures, score_attempted = _read_records(
        paths.model_scores(model_id),
        ScoreRecord,
    )
    failures: Counter[str] = Counter()
    failures.update(manifest_failures)
    failures.update(annotation_failures)
    failures.update(score_failures)
    mismatched_score_ids = [
        sample_id for sample_id, score in scores.items() if score.model_id != model_id
    ]
    for sample_id in mismatched_score_ids:
        del scores[sample_id]
    failures["wrong_model_score"] += len(mismatched_score_ids)

    joined: list[tuple[SampleManifest, AnnotationRecord, ScoreRecord]] = []
    for sample_id in sorted(manifests):
        annotation = annotations.get(sample_id)
        score = scores.get(sample_id)
        if annotation is None:
            failures["missing_annotation"] += 1
        if score is None:
            failures["missing_score"] += 1
        if annotation is not None and score is not None:
            joined.append((manifests[sample_id], annotation, score))
    failures["unmatched_annotation"] += len(set(annotations).difference(manifests))
    failures["unmatched_score"] += len(set(scores).difference(manifests))

    pairs = [(manifest, score) for manifest, _annotation, score in joined]
    report_data: dict[str, object] = {
        "schema_version": 2,
        "model_id": model_id,
        "stage_summary": {
            "attempted": len(joined) + sum(failures.values()),
            "completed": len(joined),
            "skipped": 0,
            "failed": sum(failures.values()),
        },
        "record_accounting": {
            "manifest_records": manifest_attempted,
            "annotation_records": annotation_attempted,
            "score_records": score_attempted,
            "joined_records": len(joined),
            "failure_count": sum(failures.values()),
        },
        "failures": {
            "total": sum(failures.values()),
            "by_code": {code: count for code, count in failures.items() if count},
        },
        "sample_count": len(joined),
        "detector_miss_count": sum(
            not annotation.detections for _, annotation, _ in joined
        ),
        "threshold_curves": _threshold_curves(pairs),
        "standard_operating_points": _standard_operating_points(pairs),
        "by_source_category": _category_stats(pairs, "source_category"),
        "by_expected_presence": _category_stats(pairs, "expected_presence"),
        "detector_confidence_deciles": _confidence_deciles(joined),
        "top_box_area_bins": _area_buckets(joined),
    }
    _atomic_write_json(paths.model_report_path(model_id), report_data)
    return report_data


def report(paths: EvaluationPaths) -> dict[str, object]:
    """Report every registered model without mixing model-owned records."""

    from .model_bundle import registered_model_ids

    paths.ensure()
    reports = {
        model_id: _report_model(paths, model_id)
        for model_id in registered_model_ids(paths)
    }
    return {"schema_version": 2, "models": reports}


__all__ = ["report"]
