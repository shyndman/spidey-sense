"""Acquisition stage orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import model_tools.evaluation.acquisition.materialization as materialization
import model_tools.evaluation.acquisition.repository as repository
import model_tools.evaluation.acquisition.sources.coco as coco
import model_tools.evaluation.acquisition.sources.inaturalist as inaturalist
import model_tools.evaluation.registry as registry

from ..application import StageExecution, StageSummary
from ..events import AggregateEvent, AggregateEventRecord, emit_event
from ..storage.layout import EvaluationPaths
from .models import (
    AcceptedSample,
    AcquisitionFailure,
    AcquisitionQuota,
    AcquisitionRequest,
    Candidate,
    CandidateStream,
    CategoryProgress,
    ExpectedPresence,
    Shortage,
    Source,
)
from .splitting import assign_groups_and_splits

_MATERIALIZATION_PROGRESS_ATTEMPT_CADENCE: Final[int] = 25


def _emit(
    event: AggregateEvent,
    *,
    category: str | None = None,
    page: int | None = None,
    count: int | None = None,
    quota: int | None = None,
    attempted: int | None = None,
    completed: int | None = None,
    skipped: int | None = None,
    failed: int | None = None,
    resumed: int | None = None,
    processed: int | None = None,
    code: str | None = None,
) -> None:
    emit_event(
        AggregateEventRecord(
            stage="acquire",
            event=event,
            category=category,
            page=page,
            count=count,
            quota=quota,
            attempted=attempted,
            completed=completed,
            skipped=skipped,
            failed=failed,
            resumed=resumed,
            processed=processed,
            code=code,
        )
    )


def _source_quotas() -> tuple[AcquisitionQuota, ...]:
    return (
        tuple(
            AcquisitionQuota(
                source="inaturalist",
                category=category,
                expected_presence="positive",
                quota=inaturalist.POSITIVE_QUOTA,
            )
            for category in inaturalist.INATURALIST_BUCKETS
        )
        + tuple(
            AcquisitionQuota(
                source="inaturalist",
                category=category,
                expected_presence="hard_negative",
                quota=inaturalist.HARD_NEGATIVE_QUOTA,
            )
            for category in inaturalist.INATURALIST_HARD_NEGATIVE_BUCKETS
        )
        + (
            AcquisitionQuota(
                source="coco2017",
                category="coco2017_validation",
                expected_presence="broad_negative",
                quota=coco.COCO_QUOTA,
            ),
        )
    )


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


def _emit_progress(category: str, quota: int, stats: CategoryProgress) -> None:
    if stats.materialization_attempted % _MATERIALIZATION_PROGRESS_ATTEMPT_CADENCE:
        return
    _emit(
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
) -> CandidateStream:
    if source == "coco2017":
        batch = coco.candidates(paths, need)
        if not batch.source_failures:
            _emit(
                "source_page",
                category=category,
                page=1,
                count=len(batch.candidates),
                quota=need,
            )
        return CandidateStream(
            candidates=iter(batch.candidates),
            source_failures=batch.source_failures,
        )
    return CandidateStream(
        candidates=inaturalist.iter_candidates(category, expected, need),
        source_failures=0,
    )


def _source_failure_code(source: Source) -> str:
    return "COCO_SOURCE_FAILED" if source == "coco2017" else "INAT_QUERY_FAILED"


def _record_materialization_failure(
    paths: EvaluationPaths,
    category: str,
    candidate: Candidate,
    error: AcquisitionFailure,
    quota: int,
    stats: CategoryProgress,
) -> None:
    code = repository.failure_code(error)
    repository.write_failure(paths, code, materialization.candidate_id(candidate))
    _emit("error", category=category, code=code, attempted=1, failed=1)
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
    stats: CategoryProgress,
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
    paths: EvaluationPaths, source: Source, category: str, source_failed: int
) -> None:
    if not source_failed:
        return
    source_code = _source_failure_code(source)
    repository.write_failure(paths, source_code, None)
    _emit(
        "error",
        category=category,
        code=source_code,
        attempted=source_failed,
        failed=source_failed,
    )


def _record_shortage(category: str, shortage: int) -> None:
    if shortage:
        _emit(
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
    stats: CategoryProgress,
    shortage: int = 0,
) -> None:
    _emit(
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
    quota_row: AcquisitionQuota,
) -> tuple[CategoryProgress, int]:
    source, category, expected, quota = (
        quota_row.source,
        quota_row.category,
        quota_row.expected_presence,
        quota_row.quota,
    )
    present = _category_present(accepted, source, category, expected)
    stats = CategoryProgress()
    if present >= quota:
        stats.attempted = present
        stats.skipped = present
        _emit_category_complete(category, quota, present, stats)
        return stats, 0
    stream = _candidate_stream(paths, source, category, expected, quota - present)
    source_failed = max(
        stream.source_failures,
        _consume_candidates(
            paths,
            accepted,
            occupied_sha,
            occupied_ids,
            source,
            category,
            expected,
            quota,
            stream.candidates,
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
    _emit(
        "complete",
        attempted=summary.attempted,
        completed=summary.completed,
        skipped=summary.skipped,
        failed=summary.failed,
    )
    return summary


def run(request: AcquisitionRequest) -> StageExecution:
    """Acquire the approved corpus, resuming valid samples atomically."""
    paths = request.paths
    paths.ensure()
    accepted = repository.load_existing(paths)
    resumed = len(accepted)
    _emit("start", attempted=0, completed=0, skipped=0, failed=0, resumed=resumed)
    if not registry.provision_model_bundle(paths):
        repository.write_failure(paths, "MODEL_BUNDLE_FAILED", None)
        _emit("error", code="MODEL_BUNDLE_FAILED", attempted=1, failed=1)
    occupied_sha = {item.sha256: item for item in accepted}
    occupied_ids = {item.sample_id for item in accepted}
    attempted = completed = skipped = failed = 0
    shortages: list[Shortage] = []
    for quota_row in _source_quotas():
        stats, shortage = _run_category(
            paths, accepted, occupied_sha, occupied_ids, quota_row
        )
        attempted += stats.attempted
        completed += stats.completed
        skipped += stats.skipped
        failed += stats.failed
        if shortage:
            shortages.append(Shortage(category=quota_row.category, count=shortage))
    accepted = list(assign_groups_and_splits(accepted))
    for item in accepted:
        repository.write_manifest(paths, item)
    for shortage in shortages:
        attempted += shortage.count
        failed += shortage.count
        repository.write_shortage(paths, shortage.category, shortage.count)
    summary = _complete(_summary(attempted, completed, skipped, failed))
    return StageExecution(stage="acquire", summary=summary)


__all__ = ["run"]
