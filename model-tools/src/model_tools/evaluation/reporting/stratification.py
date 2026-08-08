"""Detector confidence and box-area stratification."""

from __future__ import annotations

from statistics import fmean

from ..annotation.models import AnnotationRecord
from .models import (
    AnnotatedScoredSample,
    AreaBins,
    AreaBucket,
    ConfidenceDecile,
    MutableAreaBucket,
    MutableConfidenceBucket,
)

_RELATIVE_AREA_EDGES = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
_ABSOLUTE_AREA_EDGES = (0.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _confidence(annotation: AnnotationRecord) -> float | None:
    return max(
        (float(detection.confidence) for detection in annotation.detections),
        default=None,
    )


def _areas(sample: AnnotatedScoredSample) -> tuple[float, float] | None:
    if not sample.annotation.detections:
        return None
    detection = max(
        sample.annotation.detections, key=lambda item: (item.confidence, -item.rank)
    )
    x1, y1, x2, y2 = detection.box_xyxy
    absolute = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return absolute, absolute / float(sample.manifest.width * sample.manifest.height)


def confidence_deciles(
    samples: tuple[AnnotatedScoredSample, ...],
) -> tuple[ConfidenceDecile, ...]:
    buckets = [
        MutableConfidenceBucket(lower=index / 10, upper=(index + 1) / 10)
        for index in range(10)
    ]
    for sample in samples:
        positive = sample.manifest.expected_presence == "positive"
        confidence = _confidence(sample.annotation)
        if confidence is None:
            buckets[0].detector_miss_count += 1
            continue
        bucket = buckets[min(9, int(confidence * 10))]
        bucket.count += 1
        bucket.positive_count += positive
        bucket.negative_count += not positive
        bucket.confidence_values.append(confidence)
        bucket.score_values.append(sample.score.blocked_score)
        areas = _areas(sample)
        if areas:
            bucket.absolute_area_values.append(areas[0])
            bucket.relative_area_values.append(areas[1])
    return tuple(
        ConfidenceDecile(
            lower=bucket.lower,
            upper=bucket.upper,
            count=bucket.count,
            detector_miss_count=bucket.detector_miss_count,
            mean_confidence=_mean(bucket.confidence_values),
            mean_absolute_pixel_area=_mean(bucket.absolute_area_values),
            mean_relative_image_area=_mean(bucket.relative_area_values),
            mean_blocked_score=_mean(bucket.score_values),
            positive_count=bucket.positive_count,
            negative_count=bucket.negative_count,
        )
        for bucket in buckets
    )


def _area_edges() -> dict[str, tuple[float, ...]]:
    return {
        "absolute_pixel_area": _ABSOLUTE_AREA_EDGES,
        "relative_image_area": _RELATIVE_AREA_EDGES,
    }


def _area_buckets(
    edges: dict[str, tuple[float, ...]],
) -> dict[str, list[MutableAreaBucket]]:
    return {
        name: [
            MutableAreaBucket(
                lower=edge,
                upper=values[index + 1] if index + 1 < len(values) else None,
            )
            for index, edge in enumerate(values)
        ]
        for name, values in edges.items()
    }


def _area_bucket_index(edge_values: tuple[float, ...], value: float) -> int:
    return next(
        (
            index
            for index in range(len(edge_values) - 1)
            if edge_values[index] <= value < edge_values[index + 1]
        ),
        len(edge_values) - 1,
    )


def _add_area_sample(
    sample: AnnotatedScoredSample,
    areas: tuple[float, float],
    edges: dict[str, tuple[float, ...]],
    buckets: dict[str, list[MutableAreaBucket]],
) -> None:
    positive = sample.manifest.expected_presence == "positive"
    for name, value in zip(
        ("absolute_pixel_area", "relative_image_area"), areas, strict=True
    ):
        index = _area_bucket_index(edges[name], value)
        bucket = buckets[name][index]
        bucket.count += 1
        bucket.positive_count += positive
        bucket.negative_count += not positive
        bucket.score_values.append(sample.score.blocked_score)


def _build_area_bucket(bucket: MutableAreaBucket) -> AreaBucket:
    return AreaBucket(
        lower=bucket.lower,
        upper=bucket.upper,
        count=bucket.count,
        positive_count=bucket.positive_count,
        negative_count=bucket.negative_count,
        mean_blocked_score=_mean(bucket.score_values),
    )


def area_bins(samples: tuple[AnnotatedScoredSample, ...]) -> AreaBins:
    edges = _area_edges()
    buckets = _area_buckets(edges)
    misses = 0
    for sample in samples:
        areas = _areas(sample)
        if areas is None:
            misses += 1
            continue
        _add_area_sample(sample, areas, edges, buckets)
    return AreaBins(
        absolute_pixel_area=tuple(
            _build_area_bucket(bucket) for bucket in buckets["absolute_pixel_area"]
        ),
        relative_image_area=tuple(
            _build_area_bucket(bucket) for bucket in buckets["relative_image_area"]
        ),
        detector_miss_count=misses,
    )


__all__ = ["confidence_deciles", "area_bins"]
