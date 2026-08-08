"""CLI parser, serialization, and process-status behavior."""

import json
import sys
from pathlib import Path
from typing import Literal, Protocol, cast

import pytest
from model_tools.evaluation import cli
from model_tools.evaluation.application import (
    EvaluationRunResult,
    SingleStageCommand,
    StageExecution,
    StageOutputs,
    StageSummary,
)
from model_tools.evaluation.reporting.models import EvaluationReports, ModelReports
from model_tools.evaluation.storage.layout import EvaluationPaths


def _json_object(text: str) -> dict[str, object]:
    payload = cast(object, json.loads(text))
    if not isinstance(payload, dict):
        raise AssertionError("expected a JSON object")
    return cast(dict[str, object], payload)


class _Arguments(Protocol):
    data_dir: Path
    command: str


def _execution(
    stage: Literal["acquire", "annotate", "score"], *, failed: int = 0
) -> StageExecution:
    return StageExecution(
        stage=stage,
        summary=StageSummary(
            attempted=1,
            completed=0 if failed else 1,
            skipped=0,
            failed=failed,
        ),
    )


def test_parser_exposes_approved_commands_and_default() -> None:
    parser = cli.build_parser()
    namespace = cast(_Arguments, cast(object, parser.parse_args(["acquire"])))
    assert namespace.data_dir == Path("/data")
    for command in ("acquire", "annotate", "score", "report", "all"):
        parsed = cast(_Arguments, cast(object, parser.parse_args([command])))
        assert parsed.command == command


def test_score_status_and_output_are_delegated_to_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_run_stage(
        command: SingleStageCommand, paths: EvaluationPaths
    ) -> StageExecution:
        calls.append((command, paths.root))
        return _execution("score")

    monkeypatch.setattr(cli, "run_stage", fake_run_stage)
    assert cli.main(["score", "--data-dir", str(tmp_path)]) == 0
    assert calls == [("score", tmp_path)]
    captured = capsys.readouterr().out
    assert captured.count("\n") == 1
    assert "sample-1" not in captured


def test_all_failed_result_is_serialized_and_status_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = EvaluationRunResult(
        stages=StageOutputs(root={"acquire": _execution("acquire", failed=1).summary})
    )

    def fake_run_all(_: EvaluationPaths) -> tuple[EvaluationRunResult, int]:
        return result, 1

    monkeypatch.setattr(cli, "run_all", fake_run_all)
    assert cli.main(["all", "--data-dir", str(tmp_path)]) == 1
    stages = cast(dict[str, object], _json_object(capsys.readouterr().out)["stages"])
    assert set(stages) == {"acquire"}
    assert "annotate" not in stages
    assert "score" not in stages
    assert "report" not in stages


def test_acquire_keeps_one_aggregate_stdout_value_while_events_use_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def acquire_with_events(
        command: SingleStageCommand, _paths: EvaluationPaths
    ) -> StageExecution:
        assert command == "acquire"
        _ = sys.stderr.write('{"stage":"acquire","event":"start","resumed":0}\n')
        return _execution("acquire")

    monkeypatch.setattr(cli, "run_stage", acquire_with_events)
    assert cli.main(["acquire", "--data-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("\n") == 1
    assert _json_object(captured.out) == {
        "stage": "acquire",
        "summary": {
            "attempted": 1,
            "completed": 1,
            "failed": 0,
            "schema_version": 1,
            "skipped": 0,
        },
    }
    assert _json_object(captured.err) == {
        "stage": "acquire",
        "event": "start",
        "resumed": 0,
    }


def test_direct_report_status_is_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def report_stage(
        _: SingleStageCommand, _paths: EvaluationPaths
    ) -> EvaluationReports:
        return EvaluationReports(models=ModelReports(root={}))

    monkeypatch.setattr(cli, "run_stage", report_stage)
    assert cli.main(["report", "--data-dir", str(tmp_path)]) == 0
    assert _json_object(capsys.readouterr().out)["schema_version"] == 2


@pytest.mark.parametrize("command", ["acquire", "annotate", "score"])
def test_direct_failed_stage_status_is_one(
    command: Literal["acquire", "annotate", "score"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_stage(
        name: SingleStageCommand, _paths: EvaluationPaths
    ) -> StageExecution:
        assert name == command
        return _execution(command, failed=1)

    monkeypatch.setattr(cli, "run_stage", failed_stage)
    assert cli.main([command, "--data-dir", str(tmp_path)]) == 1
