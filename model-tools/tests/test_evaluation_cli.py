"""CLI dispatch tests using only numeric metadata and mocked stages."""

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pytest
from model_tools.evaluation import cli, model_bundle, scoring
from model_tools.evaluation.contracts import StageSummary
from model_tools.evaluation.paths import EvaluationPaths


class _ParserNamespace(Protocol):
    command: str
    data_dir: Path


class _AcquisitionModule(Protocol):
    acquire: Callable[[EvaluationPaths], StageSummary]


def _module_with_acquire(
    name: str,
    acquire: Callable[[EvaluationPaths], StageSummary],
) -> types.ModuleType:
    module = types.ModuleType(name)
    typed_module = cast(_AcquisitionModule, cast(object, module))
    typed_module.acquire = acquire
    return module


def _json_object(text: str) -> dict[str, object]:
    payload = cast(object, json.loads(text))
    if not isinstance(payload, dict):
        raise AssertionError("expected a JSON object")
    return cast(dict[str, object], payload)


def test_parser_exposes_approved_commands_and_default() -> None:
    parser = cli.build_parser()
    acquire = cast(
        _ParserNamespace, cast(object, parser.parse_args(["acquire"]))
    )
    assert acquire.data_dir == Path("/data")
    for command in ("acquire", "annotate", "score", "report", "all"):
        parsed = cast(
            _ParserNamespace, cast(object, parser.parse_args([command]))
        )
        assert parsed.command == command


def test_score_runs_every_registered_model_without_sample_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_score_model(
        paths: EvaluationPaths, model_id: str
    ) -> StageSummary:
        calls.append((paths.root, model_id))
        return StageSummary(attempted=1, completed=1, skipped=0, failed=0)

    monkeypatch.setattr(scoring, "score_model", fake_score_model)

    def registered_model_ids(_paths: EvaluationPaths) -> tuple[str, ...]:
        return ("baseline-model", "competitor-model")

    monkeypatch.setattr(
        model_bundle, "registered_model_ids", registered_model_ids
    )
    assert cli.main(["score", "--data-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr().out
    assert captured.count("\n") == 1
    assert "sample-1" not in captured
    assert calls == [
        (tmp_path, "baseline-model"),
        (tmp_path, "competitor-model"),
    ]


def test_all_stops_after_failed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called: list[str] = []

    def failed_acquire(_paths: EvaluationPaths) -> StageSummary:
        called.append("acquire")
        return StageSummary(attempted=1, completed=0, skipped=0, failed=1)

    module = _module_with_acquire(
        "model_tools.evaluation.acquisition", failed_acquire
    )
    monkeypatch.setitem(sys.modules, "model_tools.evaluation.acquisition", module)
    assert cli.main(["all", "--data-dir", str(tmp_path)]) == 1
    assert called == ["acquire"]
    assert '"failed": 1' in capsys.readouterr().out


def test_acquire_keeps_one_aggregate_stdout_value_while_events_use_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def acquire_with_events(_paths: EvaluationPaths) -> StageSummary:
        _ = sys.stderr.write('{"stage":"acquire","event":"start","resumed":0}\n')
        return StageSummary(attempted=2, completed=2, skipped=0, failed=0)

    module = _module_with_acquire(
        "model_tools.evaluation.acquisition", acquire_with_events
    )
    monkeypatch.setitem(sys.modules, "model_tools.evaluation.acquisition", module)
    assert cli.main(["acquire", "--data-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("\n") == 1
    assert _json_object(captured.out) == {
        "stage": "acquire",
        "summary": {
            "attempted": 2,
            "completed": 2,
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
