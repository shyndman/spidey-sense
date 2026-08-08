"""Annotation-owned persisted model tests."""

import pytest
from model_tools.evaluation.annotation.models import AnnotationRecord, Detection
from pydantic import BaseModel, ValidationError


def _detection() -> Detection:
    return Detection(
        rank=1, phrase="proxy", confidence=0.5, box_xyxy=(0.0, 1.0, 10.0, 12.0)
    )


def test_annotation_models_validate_bounds_and_round_trip() -> None:
    detection = _detection()
    record = AnnotationRecord(
        sample_id="sample-proxy", detections=(detection,), max_confidence=0.5
    )
    assert isinstance(detection, BaseModel)
    assert isinstance(record, BaseModel)
    assert AnnotationRecord.model_validate_json(record.model_dump_json()) == record
    with pytest.raises(ValidationError):
        _ = Detection(
            rank=21, phrase="proxy", confidence=0.5, box_xyxy=(0.0, 1.0, 10.0, 12.0)
        )
    with pytest.raises(ValidationError):
        _ = AnnotationRecord(
            sample_id="sample-proxy",
            detections=tuple(detection for _ in range(21)),
            max_confidence=0.5,
        )
