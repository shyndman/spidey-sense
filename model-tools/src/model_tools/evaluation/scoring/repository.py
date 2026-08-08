"""Score cache discovery and persistence."""

from __future__ import annotations

from pathlib import Path

from ..acquisition.models import SampleManifest
from ..application import StageFailure
from ..storage.json import SCORE_JSON_PROFILE, write_model
from ..storage.layout import EvaluationPaths
from .inference import score_sample
from .models import LoadedModel, PendingScore, PendingScores, ScoreCounts, ScoreRecord

SCORE_FAILURE_PREFIX = "SCORE_"


def _existing_score(path: Path, model_id: str, sample_id: str) -> bool:
    if not path.exists():
        return False
    try:
        score = ScoreRecord.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return False
    return score.model_id == model_id and score.sample_id == sample_id


def write_failure(
    paths: EvaluationPaths, model_id: str, code: str, sample_id: str | None = None
) -> None:
    failure = StageFailure(stage="score", code=code, sample_id=sample_id)
    try:
        failure_path = paths.model_error_path(model_id, sample_id)
    except ValueError:
        failure_path = paths.model_error_path(model_id)
        failure = StageFailure(stage="score", code=code)
    write_model(failure_path, failure, profile=SCORE_JSON_PROFILE)


def collect_pending(paths: EvaluationPaths, model_id: str) -> PendingScores:
    manifest_paths = tuple(sorted(paths.manifests.glob("*.json")))
    pending: list[PendingScore] = []
    skipped = failed = 0
    for manifest_path in manifest_paths:
        try:
            manifest = SampleManifest.model_validate_json(manifest_path.read_bytes())
            score_path = paths.score_path(model_id, manifest.sample_id)
            _ = paths.model_error_path(model_id, manifest.sample_id)
        except (OSError, ValueError):
            failed += 1
            write_failure(
                paths,
                model_id,
                f"{SCORE_FAILURE_PREFIX}INVALID_MANIFEST",
                manifest_path.stem,
            )
            continue
        if _existing_score(score_path, model_id, manifest.sample_id):
            skipped += 1
        else:
            pending.append(PendingScore(manifest=manifest, score_path=score_path))
    return PendingScores(
        attempted=len(manifest_paths),
        items=tuple(pending),
        skipped=skipped,
        failed=failed,
    )


def score_pending(
    paths: EvaluationPaths,
    model_id: str,
    model: LoadedModel,
    pending: tuple[PendingScore, ...],
) -> ScoreCounts:
    completed = failed = 0
    for item in pending:
        try:
            score = score_sample(
                model,
                item.manifest,
                paths.image_path(item.manifest.image_relative_path),
            )
            write_model(item.score_path, score, profile=SCORE_JSON_PROFILE)
        except (OSError, RuntimeError, ValueError):
            failed += 1
            write_failure(
                paths,
                model_id,
                f"{SCORE_FAILURE_PREFIX}SAMPLE",
                item.manifest.sample_id,
            )
        else:
            completed += 1
    return ScoreCounts(completed=completed, failed=failed)
