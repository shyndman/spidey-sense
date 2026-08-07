"""Deterministic, resumable corpus acquisition orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

from . import coco, corpus, inaturalist, materialization, model_bundle
from .acquisition_types import (
    AcceptedSample,
    AcquisitionFailure,
    Candidate,
    ExpectedPresence,
    Source,
)
from .contracts import EvaluationPaths, StageSummary
from .events import emit_event

_MATERIALIZATION_PROGRESS_ATTEMPT_CADENCE: Final[int] = 25


@dataclass(slots=True)
class _CategoryStats:
    attempted: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    materialization_attempted: int = 0


def _quota(
    source: Source,
    category: str,
    expected: ExpectedPresence,
    quota: int,
) -> tuple[Source, str, ExpectedPresence, int]:
    return source, category, expected, quota


def _source_quotas() -> tuple[tuple[Source, str, ExpectedPresence, int], ...]:
    return tuple(
        _quota("inaturalist", category, "positive", inaturalist.POSITIVE_QUOTA)
        for category in inaturalist.INATURALIST_BUCKETS
    ) + tuple(
        _quota(
            "inaturalist",
            category,
            "hard_negative",
            inaturalist.HARD_NEGATIVE_QUOTA,
        )
        for category in inaturalist.INATURALIST_HARD_NEGATIVE_BUCKETS
    ) + (_quota("coco2017", "coco2017_validation", "broad_negative", coco.COCO_QUOTA),)

def _category_present(
    accepted: list[AcceptedSample],
    source: Source,
    category: str,
    expected: ExpectedPresence,
) -> int:
    return sum(
        item.candidate.source == source
        and item.candidate.source_category == category
        and item.candidate.expected_presence == expected
        for item in accepted
    )


def _summary(attempted: int, completed: int, skipped: int, failed: int) -> StageSummary:
    return StageSummary(
        attempted=attempted,
        completed=completed,
        skipped=skipped,
        failed=failed,
    )


def _emit_progress(category: str, quota: int, stats: _CategoryStats) -> None:
    if stats.materialization_attempted % _MATERIALIZATION_PROGRESS_ATTEMPT_CADENCE:
        return
    emit_event(
        "progress",
        category=category,
        quota=quota,
        attempted=stats.attempted,
        completed=stats.completed,
        skipped=stats.skipped,
        failed=stats.failed,
    )


def _candidate_stream(
    paths: EvaluationPaths,
    source: Source,
    category: str,
    expected: ExpectedPresence,
    need: int,
) -> tuple[Iterator[Candidate], int]:
    if source == "coco2017":
        candidates, source_failed = coco.candidates(paths, need)
        if not source_failed:
            emit_event(
                "source_page",
                category=category,
                page=1,
                count=len(candidates),
                quota=need,
            )
        return iter(candidates), source_failed
    return inaturalist.iter_candidates(category, expected, need), 0

def _source_failure_code(source: Source) -> str:
    return "COCO_SOURCE_FAILED" if source == "coco2017" else "INAT_QUERY_FAILED"


def _record_materialization_failure(
    paths: EvaluationPaths,
    category: str,
    candidate: Candidate,
    error: AcquisitionFailure,
    quota: int,
    stats: _CategoryStats,
) -> None:
    code = corpus.failure_code(error)
    corpus.write_failure(paths, code, materialization.candidate_id(candidate))
    emit_event(
        "error",
        category=category,
        code=code,
        attempted=1,
        failed=1,
    )
    _emit_progress(category, quota, stats)


def _consume_candidates(
    paths: EvaluationPaths,
    accepted: list[AcceptedSample],
    occupied_sha: dict[str, AcceptedSample],
    occupied_ids: set[str],
    source: Source,
    category: str,
    expected: ExpectedPresence,
    quota: int,
    candidates: Iterator[Candidate],
    stats: _CategoryStats,
) -> int:
    source_failed = 0
    try:
        while _category_present(accepted, source, category, expected) < quota:
            try:
                candidate = next(candidates)
            except StopIteration:
                break
            stats.attempted += 1
            stats.materialization_attempted += 1
            try:
                item = materialization.materialize_candidate(paths, candidate)
            except AcquisitionFailure as error:
                stats.failed += 1
                _record_materialization_failure(
                    paths, category, candidate, error, quota, stats
                )
                continue
            if item.sha256 in occupied_sha:
                canonical = occupied_sha[item.sha256]
                if item.image_path != canonical.image_path:
                    item.image_path.unlink(missing_ok=True)
                stats.skipped += 1
            elif item.sample_id in occupied_ids:
                stats.skipped += 1
            else:
                accepted.append(item)
                occupied_sha[item.sha256] = item
                occupied_ids.add(item.sample_id)
                stats.completed += 1
            _emit_progress(category, quota, stats)
    except AcquisitionFailure:
        source_failed = 1
    finally:
        close = getattr(candidates, "close", None)
        if callable(close):
            _ = close()
    return source_failed


def _record_source_failure(
    paths: EvaluationPaths,
    source: Source,
    category: str,
    source_failed: int,
) -> None:
    if not source_failed:
        return
    source_code = _source_failure_code(source)
    corpus.write_failure(paths, source_code, None)
    emit_event(
        "error",
        category=category,
        code=source_code,
        attempted=source_failed,
        failed=source_failed,
    )


def _record_shortage(category: str, shortage: int) -> None:
    if not shortage:
        return
    emit_event(
        "error",
        category=category,
        code="QUOTA_SHORTAGE",
        count=shortage,
        attempted=shortage,
        failed=shortage,
    )


def _emit_category_complete(
    category: str,
    quota: int,
    present: int,
    stats: _CategoryStats,
    shortage: int = 0,
) -> None:
    emit_event(
        "category_complete",
        category=category,
        count=present,
        quota=quota,
        attempted=stats.attempted + shortage,
        completed=stats.completed,
        skipped=stats.skipped,
        failed=stats.failed + shortage,
    )


def _run_category(
    paths: EvaluationPaths,
    accepted: list[AcceptedSample],
    occupied_sha: dict[str, AcceptedSample],
    occupied_ids: set[str],
    source: Source,
    category: str,
    expected: ExpectedPresence,
    quota: int,
) -> tuple[_CategoryStats, int]:
    present = _category_present(accepted, source, category, expected)
    stats = _CategoryStats()
    if present >= quota:
        stats.attempted = present
        stats.skipped = present
        _emit_category_complete(category, quota, present, stats)
        return stats, 0
    candidates, source_failed = _candidate_stream(
        paths, source, category, expected, quota - present
    )
    source_failed = max(
        source_failed,
        _consume_candidates(
            paths,
            accepted,
            occupied_sha,
            occupied_ids,
            source,
            category,
            expected,
            quota,
            candidates,
            stats,
        ),
    )
    _record_source_failure(paths, source, category, source_failed)
    present = _category_present(accepted, source, category, expected)
    shortage = max(0, quota - present)
    _record_shortage(category, shortage)
    _emit_category_complete(category, quota, present, stats, shortage)
    return stats, shortage


def _complete(summary: StageSummary) -> StageSummary:
    emit_event(
        "complete",
        attempted=summary.attempted,
        completed=summary.completed,
        skipped=summary.skipped,
        failed=summary.failed,
    )
    return summary


def acquire(paths: EvaluationPaths) -> StageSummary:
    """Acquire the approved corpus, resuming valid samples atomically."""
    paths.ensure()
    accepted = corpus.load_existing(paths)
    resumed = len(accepted)
    emit_event(
        "start",
        attempted=0,
        completed=0,
        skipped=0,
        failed=0,
        resumed=resumed,
    )
    if not model_bundle.provision_model_bundle(paths):
        corpus.write_failure(paths, "MODEL_BUNDLE_FAILED", None)
        emit_event("error", code="MODEL_BUNDLE_FAILED", attempted=1, failed=1)
        return _complete(_summary(1, 0, 0, 1))
    occupied_sha = {item.sha256: item for item in accepted}
    occupied_ids = {item.sample_id for item in accepted}
    attempted = completed = skipped = failed = 0
    shortages: list[tuple[str, int]] = []
    for source, category, expected, quota in _source_quotas():
        stats, shortage = _run_category(
            paths,
            accepted,
            occupied_sha,
            occupied_ids,
            source,
            category,
            expected,
            quota,
        )
        attempted += stats.attempted
        completed += stats.completed
        skipped += stats.skipped
        failed += stats.failed
        if shortage:
            shortages.append((category, shortage))
    corpus.assign_groups_and_splits(accepted)
    for item in accepted:
        corpus.write_manifest(paths, item)
    for category, count in shortages:
        attempted += count
        failed += count
        corpus.write_shortage(paths, category, count)
    return _complete(_summary(attempted, completed, skipped, failed))


__all__ = ["acquire"]
