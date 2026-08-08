"""Annotation persistence and resume validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from ..application import StageFailure
from ..storage.json import ANNOTATION_JSON_PROFILE, write_model
from ..storage.layout import EvaluationPaths
from .models import AnnotationRecord


def valid_annotation(path: Path, sample_id: str) -> bool:
    try:
        record = AnnotationRecord.model_validate_json(path.read_bytes())
    except (OSError, ValueError, TypeError):
        return False
    return record.sample_id == sample_id


def _failure_path(paths: EvaluationPaths, manifest_path: Path) -> Path:
    stable_name = hashlib.sha256(manifest_path.name.encode("utf-8")).hexdigest()
    return paths.errors / f"{stable_name}.json"


def write_annotation(destination: Path, value: BaseModel) -> None:
    write_model(destination, value, profile=ANNOTATION_JSON_PROFILE)


def write_failure(
    paths: EvaluationPaths,
    manifest_path: Path,
    *,
    code: str,
    sample_id: str | None,
) -> None:
    try:
        failure = StageFailure(stage="annotate", code=code, sample_id=sample_id)
        write_model(
            _failure_path(paths, manifest_path),
            failure,
            profile=ANNOTATION_JSON_PROFILE,
        )
    except (OSError, TypeError, ValueError):
        return
