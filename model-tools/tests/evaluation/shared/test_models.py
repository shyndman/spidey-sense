"""Shared base-model configuration tests."""

import pytest
from model_tools.evaluation.base import (
    EvaluationModel,
    MutableEvaluationModel,
    RuntimeEvaluationModel,
)
from pydantic import BaseModel, ValidationError


def test_base_models_are_pydantic_and_configuration_is_strict() -> None:
    class Value(EvaluationModel):
        value: int

    class Runtime(RuntimeEvaluationModel):
        value: object

    class Mutable(MutableEvaluationModel):
        value: int

    assert isinstance(Value(value=1), BaseModel)
    assert isinstance(Runtime(value=object()), BaseModel)
    mutable = Mutable(value=1)
    mutable.value = 2
    assert mutable.value == 2
    with pytest.raises(ValidationError):
        _ = Value.model_validate({"value": 1, "extra": 2})
