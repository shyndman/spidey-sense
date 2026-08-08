"""Application orchestration behavior."""

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import pytest
from model_tools.evaluation.acquisition.models import AcquisitionRequest
from model_tools.evaluation.annotation.models import AnnotationRequest
from model_tools.evaluation.application import (
    EvaluationRunResult,
    StageExecution,
    StageFailure,
    StageOutputs,
    StageSummary,
    run_all,
    run_stage,
)
from model_tools.evaluation.registry import RegisteredModels
from model_tools.evaluation.reporting.models import (
    EvaluationReports,
    ModelReports,
    ReportsRequest,
)
from model_tools.evaluation.scoring.models import ScoringRequest
from model_tools.evaluation.storage.layout import EvaluationPaths
from pydantic import BaseModel


class _RunnableModule[RequestT, ResultT](types.ModuleType):
    run: Callable[[RequestT], ResultT]

    def __init__(self, name: str, run: Callable[[RequestT], ResultT]) -> None:
        super().__init__(name)
        self.run = run

def _execution(stage: str, *, failed: int = 0) -> StageExecution:
    return StageExecution(
        stage=cast("Literal['acquire', 'annotate', 'score']", stage),
        summary=StageSummary(
            attempted=1, completed=0 if failed else 1, skipped=0, failed=failed
        ),
    )


def _module[RequestT, ResultT](
    name: str, run: Callable[[RequestT], ResultT]
) -> types.ModuleType:
    return _RunnableModule(name, run)


def _reports() -> EvaluationReports:
    return EvaluationReports(models=ModelReports(root={}))


def test_run_stage_uses_typed_package_facade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[AcquisitionRequest] = []

    def fake_run(request: AcquisitionRequest) -> StageExecution:
        seen.append(request)
        return _execution("acquire")

    monkeypatch.setitem(
        sys.modules,
        "model_tools.evaluation.acquisition",
        _module("model_tools.evaluation.acquisition", fake_run),
    )
    result = run_stage("acquire", EvaluationPaths(root=tmp_path))
    assert result == _execution("acquire")
    assert seen[0].paths.root == tmp_path


def test_run_all_orders_and_aggregates_registered_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = EvaluationPaths(root=tmp_path)
    calls: list[str] = []

    def acquire_run(_: AcquisitionRequest) -> StageExecution:
        calls.append("acquire")
        return _execution("acquire")

    def annotate_run(_: AnnotationRequest) -> StageExecution:
        calls.append("annotate")
        return _execution("annotate")

    def score_run(request: ScoringRequest) -> StageExecution:
        calls.append(f"score:{request.model_id}")
        return StageExecution(
            stage="score",
            summary=StageSummary(attempted=1, completed=1, skipped=0, failed=0),
        )

    def report_run(_: ReportsRequest) -> EvaluationReports:
        calls.append("report")
        return _reports()

    monkeypatch.setitem(
        sys.modules,
        "model_tools.evaluation.acquisition",
        _module("acquisition", acquire_run),
    )
    monkeypatch.setitem(
        sys.modules,
        "model_tools.evaluation.annotation",
        _module("annotation", annotate_run),
    )
    monkeypatch.setitem(
        sys.modules, "model_tools.evaluation.scoring", _module("scoring", score_run)
    )
    monkeypatch.setitem(
        sys.modules,
        "model_tools.evaluation.reporting",
        _module("reporting", report_run),
    )

    def registered_ids(_: EvaluationPaths) -> RegisteredModels:
        return RegisteredModels(root=("first", "second"))

    monkeypatch.setattr(
        "model_tools.evaluation.registry.registered_model_ids", registered_ids
    )
    result, status = run_all(paths)
    assert isinstance(result, EvaluationRunResult)
    assert status == 0
    assert calls == ["acquire", "annotate", "score:first", "score:second", "report"]
    assert set(result.stages.root) == {"acquire", "annotate", "score", "report"}
    summary = result.stages.root["score"]
    assert isinstance(summary, StageSummary)
    assert summary.attempted == 2 and summary.completed == 2 and summary.failed == 0


def test_run_all_failure_contains_only_present_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def failed_acquire(_: AcquisitionRequest) -> StageExecution:
        calls.append("acquire")
        return _execution("acquire", failed=1)

    monkeypatch.setitem(
        sys.modules,
        "model_tools.evaluation.acquisition",
        _module("acquisition", failed_acquire),
    )
    result, status = run_all(EvaluationPaths(root=tmp_path))
    assert status == 1 and calls == ["acquire"]
    assert set(result.stages.root) == {"acquire"}
    assert all(key not in result.stages.root for key in ("annotate", "score", "report"))


def test_application_records_are_pydantic_and_round_trip() -> None:
    records = (
        StageFailure(stage="acquire", code="failed"),
        StageSummary(attempted=1, completed=1, skipped=0, failed=0),
        _execution("acquire"),
        StageOutputs(
            root={
                "acquire": StageSummary(attempted=1, completed=1, skipped=0, failed=0)
            }
        ),
        EvaluationRunResult(stages=StageOutputs(root={})),
    )
    assert all(isinstance(record, BaseModel) for record in records)
    for record in records:
        assert type(record).model_validate(record.model_dump()) == record
