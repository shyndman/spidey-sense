"""Typed records and explicit orchestration for evaluation stages."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, cast

from pydantic import (
    ConfigDict,
    NonNegativeInt,
    RootModel,
    field_validator,
    model_validator,
)

from .base import EvaluationModel
from .reporting.models import EvaluationReports

if TYPE_CHECKING:
    from .storage.layout import EvaluationPaths

StageName = Literal["acquire", "annotate", "score", "report"]
SingleStageCommand = Literal["acquire", "annotate", "score", "report"]


class StageFailure(EvaluationModel):
    schema_version: Literal[1] = 1
    stage: str
    code: str
    sample_id: str | None = None

    @field_validator("stage", "code")
    @classmethod
    def non_empty_failure_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("sample_id")
    @classmethod
    def non_empty_failure_sample_id(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("must not be empty when provided")
        return value


class StageSummary(EvaluationModel):
    schema_version: Literal[1] = 1
    attempted: NonNegativeInt
    completed: NonNegativeInt
    skipped: NonNegativeInt
    failed: NonNegativeInt

    @model_validator(mode="after")
    def counts_partition_attempts(self) -> StageSummary:
        if self.completed + self.skipped + self.failed != self.attempted:
            raise ValueError("completed + skipped + failed must equal attempted")
        return self


class StageExecution(EvaluationModel):
    stage: Literal["acquire", "annotate", "score"]
    summary: StageSummary


class StageOutputs(RootModel[dict[StageName, StageSummary | EvaluationReports]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class EvaluationRunResult(EvaluationModel):
    stages: StageOutputs


class _Request(Protocol):
    paths: EvaluationPaths


class _ScoringRequest(_Request, Protocol):
    model_id: str


class _RequestType(Protocol):
    def __call__(self, *, paths: EvaluationPaths) -> _Request: ...


class _ScoringRequestType(Protocol):
    def __call__(self, *, paths: EvaluationPaths, model_id: str) -> _ScoringRequest: ...


class _StageModule(Protocol):
    def run(self, request: _Request) -> StageExecution: ...


class _ReportingModule(Protocol):
    def run(self, request: _Request) -> EvaluationReports: ...


class _AcquisitionModelsModule(Protocol):
    AcquisitionRequest: _RequestType


class _AnnotationModelsModule(Protocol):
    AnnotationRequest: _RequestType


class _ReportingModelsModule(Protocol):
    ReportsRequest: _RequestType


class _ScoringModelsModule(Protocol):
    ScoringRequest: _ScoringRequestType


def _run_score(paths: EvaluationPaths) -> StageExecution:
    from .registry import registered_model_ids

    scoring = cast(
        _StageModule,
        cast(object, importlib.import_module("model_tools.evaluation.scoring")),
    )
    models = cast(
        _ScoringModelsModule,
        cast(object, importlib.import_module("model_tools.evaluation.scoring.models")),
    )

    summaries = tuple(
        scoring.run(models.ScoringRequest(paths=paths, model_id=model_id)).summary
        for model_id in registered_model_ids(paths).root
    )

    return StageExecution(
        stage="score",
        summary=StageSummary(
            attempted=sum(summary.attempted for summary in summaries),
            completed=sum(summary.completed for summary in summaries),
            skipped=sum(summary.skipped for summary in summaries),
            failed=sum(summary.failed for summary in summaries),
        ),
    )


def run_stage(
    command: SingleStageCommand, paths: EvaluationPaths
) -> StageExecution | EvaluationReports:
    """Run one named stage with lazy loading and return its typed result.

    Only the requested stage implementation and models are imported; scoring
    dispatches across registered model IDs, while reporting returns reports.
    """
    if command == "acquire":
        acquisition = cast(
            _StageModule,
            cast(object, importlib.import_module("model_tools.evaluation.acquisition")),
        )
        models = cast(
            _AcquisitionModelsModule,
            cast(
                object,
                importlib.import_module("model_tools.evaluation.acquisition.models"),
            ),
        )
        return acquisition.run(models.AcquisitionRequest(paths=paths))
    if command == "annotate":
        annotation = cast(
            _StageModule,
            cast(object, importlib.import_module("model_tools.evaluation.annotation")),
        )
        models = cast(
            _AnnotationModelsModule,
            cast(
                object,
                importlib.import_module("model_tools.evaluation.annotation.models"),
            ),
        )
        return annotation.run(models.AnnotationRequest(paths=paths))
    if command == "score":
        return _run_score(paths)

    reporting = cast(
        _ReportingModule,
        cast(object, importlib.import_module("model_tools.evaluation.reporting")),
    )
    models = cast(
        _ReportingModelsModule,
        cast(
            object, importlib.import_module("model_tools.evaluation.reporting.models")
        ),
    )
    return reporting.run(models.ReportsRequest(paths=paths))


def run_all(paths: EvaluationPaths) -> tuple[EvaluationRunResult, int]:
    """Run stages in order, stopping on failure and returning present outputs.

    Acquisition, annotation, and scoring run before reporting; a failed
    summary stops the sequence and leaves later-stage results absent. The
    returned status is nonzero on early stop and zero after reporting.
    """
    stages: dict[StageName, StageSummary | EvaluationReports] = {}
    for command in ("acquire", "annotate", "score"):
        execution = cast(StageExecution, run_stage(command, paths))
        stages[command] = execution.summary
        if execution.summary.failed:
            return EvaluationRunResult(stages=StageOutputs(stages)), 1

    reporting = cast(
        _ReportingModule,
        cast(object, importlib.import_module("model_tools.evaluation.reporting")),
    )
    models = cast(
        _ReportingModelsModule,
        cast(
            object, importlib.import_module("model_tools.evaluation.reporting.models")
        ),
    )
    stages["report"] = reporting.run(models.ReportsRequest(paths=paths))
    return EvaluationRunResult(stages=StageOutputs(stages)), 0


__all__ = [
    "EvaluationRunResult",
    "SingleStageCommand",
    "StageExecution",
    "StageFailure",
    "StageName",
    "StageOutputs",
    "StageSummary",
    "run_all",
    "run_stage",
]
