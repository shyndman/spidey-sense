"""CLI dispatch tests using only numeric manifest metadata and mocked subprocesses."""

import json
import sys
import types
from pathlib import Path

from model_tools.evaluation import cli


def test_parser_exposes_approved_commands_and_default() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["acquire"]).data_dir == Path("/data")
    for command in ("acquire", "annotate", "score", "report", "all"):
        assert parser.parse_args([command]).command == command


def test_score_invocation_streams_stderr_and_suppresses_stdout(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return cli.subprocess.CompletedProcess(
            command,
            0,
            stdout="sample-1\n",
            stderr="sample-1 failed\n",
        )

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["score", "--data-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr().out
    assert captured.count("\n") == 1
    assert "sample-1" not in captured
    assert calls == [
        (
            [
                "pnpm",
                "--dir",
                "/app/extension",
                "exec",
                "tsx",
                "scripts/evaluate-model.ts",
                "--data-dir",
                str(tmp_path),
            ],
            {
                "check": True,
                "stdout": cli.subprocess.PIPE,
                "stderr": None,
                "text": True,
            },
        )
    ]


def test_all_stops_after_failed_stage(tmp_path: Path, monkeypatch, capsys) -> None:
    called: list[str] = []

    def failed_acquire(_paths):
        called.append("acquire")
        return cli.StageSummary(attempted=1, completed=0, skipped=0, failed=1)

    module = types.ModuleType("model_tools.evaluation.acquisition")
    module.acquire = failed_acquire
    monkeypatch.setitem(sys.modules, "model_tools.evaluation.acquisition", module)
    assert cli.main(["all", "--data-dir", str(tmp_path)]) == 1
    assert called == ["acquire"]
    assert '"failed": 1' in capsys.readouterr().out


def test_acquire_keeps_one_aggregate_stdout_value_while_events_use_stderr(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def acquire_with_events(_paths):
        _ = sys.stderr.write(
            '{"stage":"acquire","event":"start","resumed":0}\n'
        )
        return cli.StageSummary(attempted=2, completed=2, skipped=0, failed=0)

    module = types.ModuleType("model_tools.evaluation.acquisition")
    module.acquire = acquire_with_events
    monkeypatch.setitem(sys.modules, "model_tools.evaluation.acquisition", module)
    assert cli.main(["acquire", "--data-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == {
        "stage": "acquire",
        "summary": {
            "attempted": 2,
            "completed": 2,
            "failed": 0,
            "schema_version": 1,
            "skipped": 0,
        },
    }
    assert json.loads(captured.err) == {
        "stage": "acquire",
        "event": "start",
        "resumed": 0,
    }
