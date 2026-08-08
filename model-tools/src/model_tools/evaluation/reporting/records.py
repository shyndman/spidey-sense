"""Read persisted evaluation records."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import NonNegativeInt

from ..acquisition.models import SampleManifest
from ..annotation.models import AnnotationRecord
from ..base import EvaluationModel
from ..scoring.models import ScoreRecord
from ..storage.json import read_model


class RecordReadResult[T: EvaluationModel](EvaluationModel):
    records: dict[str, T]
    failures: dict[str, NonNegativeInt]
    attempted: NonNegativeInt


def read_records[T: EvaluationModel](
    directory: Path, model: type[T]
) -> RecordReadResult[T]:
    records: dict[str, T] = {}
    failures: Counter[str] = Counter()
    attempted = 0
    if model is SampleManifest:
        kind = "manifest"
    elif model is AnnotationRecord:
        kind = "annotation"
    elif model is ScoreRecord:
        kind = "score"
    else:
        raise ValueError("unsupported evaluation record model")
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        attempted += 1
        try:
            value = read_model(path, model)
        except (OSError, ValueError, TypeError):
            failures[f"invalid_{kind}"] += 1
            continue
        sample_id = value.sample_id
        if sample_id in records:
            failures[f"duplicate_{kind}"] += 1
            continue
        records[sample_id] = value
    return RecordReadResult(
        records=records, failures=dict(failures), attempted=attempted
    )


def typed_records[T: EvaluationModel](
    result: RecordReadResult[T], model: type[T]
) -> dict[str, T]:
    return {key: model.model_validate(value) for key, value in result.records.items()}


__all__ = ["RecordReadResult", "read_records", "typed_records"]
