"""Shared contracts and orchestration for the container-only evaluation pipeline."""

from .contracts import (
    AnnotationRecord,
    Detection,
    SampleManifest,
    ScoreRecord,
    StageFailure,
    StageSummary,
)
from .paths import EvaluationPaths, atomic_write_json, read_json, write_json_atomic

__all__ = [
    "AnnotationRecord",
    "Detection",
    "EvaluationPaths",
    "SampleManifest",
    "ScoreRecord",
    "StageFailure",
    "StageSummary",
    "atomic_write_json",
    "read_json",
    "write_json_atomic",
]
