"""Typed aggregate event shape and JSON behavior."""

import json
from typing import cast

import pytest
from model_tools.evaluation.events import AggregateEventRecord, emit_event
from pydantic import BaseModel


def test_event_record_is_typed_and_round_trips(
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = AggregateEventRecord(
        stage="acquire", event="progress", category="proxy", attempted=1, completed=1
    )
    assert isinstance(record, BaseModel)
    assert AggregateEventRecord.model_validate_json(record.model_dump_json()) == record
    emit_event(record)
    payload = cast(object, json.loads(capsys.readouterr().err))
    assert isinstance(payload, dict)
    payload = cast(dict[str, object], payload)
    assert payload == {
        "stage": "acquire",
        "event": "progress",
        "category": "unknown",
        "attempted": 1,
        "completed": 1,
    }
    assert payload["stage"] == "acquire"
    assert payload["event"] == "progress"
    assert payload["category"] == "unknown"
    assert payload["attempted"] == 1
    assert "code" not in payload
