"""Duplicate-group naming and calibration/test assignment."""

from __future__ import annotations

from collections.abc import Sequence

from .deduplication import ordered_groups
from .models import AcceptedSample


def assign_groups_and_splits(
    samples: Sequence[AcceptedSample],
) -> tuple[AcceptedSample, ...]:
    """Return copies with deterministic groups and the existing split policy."""

    ordered = ordered_groups(samples)
    names: dict[str, str] = {}
    for group in ordered:
        group_name = f"phash-{min(item.sample_id for item in group)}"
        for item in group:
            names[item.sample_id] = group_name

    totals: dict[tuple[str, str], int] = {}
    for item in samples:
        key = (item.candidate.source_category, item.candidate.expected_presence)
        totals[key] = totals.get(key, 0) + 1
    targets = {key: (value + 1) // 2 for key, value in totals.items()}
    calibration: dict[tuple[str, str], int] = {key: 0 for key in totals}
    splits: dict[str, str] = {}
    for group in ordered:
        contribution: dict[tuple[str, str], int] = {}
        for item in group:
            key = (item.candidate.source_category, item.candidate.expected_presence)
            contribution[key] = contribution.get(key, 0) + 1
        calibration_gain = sum(
            max(0, targets[key] - calibration[key]) * count
            for key, count in contribution.items()
        )
        test_gain = sum(
            max(0, calibration[key] - targets[key]) * count
            for key, count in contribution.items()
        )
        use_calibration = calibration_gain >= test_gain
        split = "calibration" if use_calibration else "test"
        for item in group:
            splits[item.sample_id] = split
        if use_calibration:
            for key, count in contribution.items():
                calibration[key] += count

    return tuple(
        item.model_copy(
            update={
                "duplicate_group": names[item.sample_id],
                "split": splits[item.sample_id],
            }
        )
        for item in samples
    )


__all__ = ["assign_groups_and_splits"]
