"""Scoring stage orchestration."""

from __future__ import annotations

from ..application import StageExecution, StageSummary
from .inference import load_model
from .models import ScoringRequest
from .repository import (
    SCORE_FAILURE_PREFIX,
    collect_pending,
    score_pending,
    write_failure,
)


def run(request: ScoringRequest) -> StageExecution:
    paths, model_id = request.paths, request.model_id
    paths.ensure_model(model_id)
    pending = collect_pending(paths, model_id)
    if not pending.items:
        summary = StageSummary(
            attempted=pending.attempted,
            completed=0,
            skipped=pending.skipped,
            failed=pending.failed,
        )
        return StageExecution(stage="score", summary=summary)
    try:
        model = load_model(paths, model_id)
    except (OSError, RuntimeError, ValueError):
        write_failure(paths, model_id, f"{SCORE_FAILURE_PREFIX}MODEL_LOAD")
        summary = StageSummary(
            attempted=pending.attempted,
            completed=0,
            skipped=pending.skipped,
            failed=pending.failed + len(pending.items),
        )
        return StageExecution(stage="score", summary=summary)
    counts = score_pending(paths, model_id, model, pending.items)
    summary = StageSummary(
        attempted=pending.attempted,
        completed=counts.completed,
        skipped=pending.skipped,
        failed=pending.failed + counts.failed,
    )
    return StageExecution(stage="score", summary=summary)
