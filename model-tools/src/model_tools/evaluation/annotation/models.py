"""Typed annotation records and detector runtime values."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, Protocol

from pydantic import Field, PositiveInt, SkipValidation, field_validator

from ..base import EvaluationModel, RuntimeEvaluationModel
from ..storage.layout import EvaluationPaths

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Pixel = Annotated[float, Field(ge=0.0)]


class AnnotationRequest(EvaluationModel):
    paths: EvaluationPaths


class Detection(EvaluationModel):
    schema_version: Literal[1] = 1
    rank: PositiveInt = Field(le=20)
    phrase: str
    confidence: Probability
    box_xyxy: tuple[Pixel, Pixel, Pixel, Pixel]

    @field_validator("phrase")
    @classmethod
    def non_empty_phrase(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class AnnotationRecord(EvaluationModel):
    schema_version: Literal[1] = 1
    sample_id: str
    detections: tuple[Detection, ...] = Field(max_length=20)
    max_confidence: Probability

    @field_validator("sample_id")
    @classmethod
    def non_empty_sample_id(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class RankedProposal(EvaluationModel):
    confidence: float
    phrase: str
    box_xyxy: tuple[float, float, float, float]
    query_index: int


class TokenizerProtocol(Protocol):
    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ) -> object: ...


class ProcessorProtocol(Protocol):
    tokenizer: TokenizerProtocol

    def __call__(
        self,
        *,
        images: object,
        text: str,
        return_tensors: str,
    ) -> object: ...


class ModelProtocol(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...

    def to(self, device: Literal["cpu", "cuda"]) -> object: ...

    def eval(self) -> object: ...


class CudaProtocol(Protocol):
    def is_available(self) -> bool: ...


class TorchProtocol(Protocol):
    cuda: CudaProtocol
    no_grad: Callable[[], object]


class DetectorRuntime(RuntimeEvaluationModel):
    torch: SkipValidation[TorchProtocol]
    processor: SkipValidation[ProcessorProtocol]
    model: SkipValidation[ModelProtocol]
    device: Literal["cpu", "cuda"]


__all__ = [
    "AnnotationRecord",
    "AnnotationRequest",
    "Detection",
    "DetectorRuntime",
    "RankedProposal",
]
