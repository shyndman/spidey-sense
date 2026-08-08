"""Scoring-owned persisted model tests."""

import pytest
from model_tools.evaluation.scoring.models import ScoreRecord
from pydantic import BaseModel, ValidationError


def _score() -> ScoreRecord:
    return ScoreRecord(
        model_id="model-proxy",
        sample_id="sample-proxy",
        probabilities=(0.001,) * 1000,
        blocked_score=0.2,
        top_index=3,
    )


def test_score_record_is_strict_and_round_trips() -> None:
    value = _score()
    assert isinstance(value, BaseModel)
    assert ScoreRecord.model_validate_json(value.model_dump_json()) == value
    with pytest.raises(ValidationError):
        _ = ScoreRecord(
            model_id="model-proxy",
            sample_id="sample-proxy",
            probabilities=(0.001,) * 999,
            blocked_score=0.2,
            top_index=3,
        )
