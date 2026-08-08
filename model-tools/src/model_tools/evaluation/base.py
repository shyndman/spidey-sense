"""Strict Pydantic bases used at evaluation module boundaries."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class EvaluationModel(BaseModel):
    """Immutable, strict model for semantic evaluation values."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class RuntimeEvaluationModel(BaseModel):
    """Strict immutable model that may wrap opaque runtime objects."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class MutableEvaluationModel(BaseModel):
    """Strict mutable model for local stage accumulators only."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=False,
        strict=True,
        validate_assignment=True,
    )


__all__ = ["EvaluationModel", "MutableEvaluationModel", "RuntimeEvaluationModel"]
