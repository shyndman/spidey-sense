"""Typed command-line orchestration for the container-only evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from .contracts import StageSummary
from .paths import EvaluationPaths

CommandName = Literal["acquire", "annotate", "score", "report", "all"]


def build_parser() -> argparse.ArgumentParser:
    """Build the five-command evaluator parser with a volume-root default."""

    parser = argparse.ArgumentParser(prog="model-tools-evaluation")
    _ = parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data"),
        dest="data_dir",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name in ("acquire", "annotate", "score", "report", "all"):
        command = commands.add_parser(command_name)
        # Suppress this default so a global --data-dir remains effective while
        # also accepting the ergonomic `command --data-dir /data` spelling.
        _ = command.add_argument(
            "--data-dir",
            type=Path,
            default=argparse.SUPPRESS,
            dest="data_dir",
        )
    return parser


def _summary_payload(stage: str, summary: StageSummary) -> dict[str, object]:
    return {"stage": stage, "summary": summary.model_dump(mode="json")}


def _emit(payload: object) -> None:
    """Emit one aggregate JSON value and never stage-level sample details."""

    _ = sys.stdout.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _run_score(paths: EvaluationPaths) -> StageSummary:
    """Invoke the TypeScript scorer with live stderr and suppressed stdout."""
    manifest_ids: set[str] = {path.stem for path in paths.manifests.glob("*.json")}

    before_ids = {
        path.stem for path in paths.scores.glob("*.json") if path.stem in manifest_ids
    }
    command = [
        "pnpm",
        "--dir",
        "/app/extension",
        "exec",
        "tsx",
        "scripts/evaluate-model.ts",
        "--data-dir",
        str(paths.root),
    ]
    try:
        _ = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        skipped = len(before_ids)
        failed = max(1, len(manifest_ids) - skipped)
        return StageSummary(
            attempted=skipped + failed,
            completed=0,
            skipped=skipped,
            failed=failed,
        )
    after_ids = {
        path.stem for path in paths.scores.glob("*.json") if path.stem in manifest_ids
    }
    skipped = len(before_ids)
    completed = len(after_ids - before_ids)
    failed = len(manifest_ids) - skipped - completed
    return StageSummary(
        attempted=len(manifest_ids),
        completed=completed,
        skipped=skipped,
        failed=failed,
    )


def _run_named(
    command: CommandName,
    paths: EvaluationPaths,
) -> StageSummary | dict[str, object]:
    """Lazy-load one stage implementation and execute it for the data root."""

    if command == "acquire":
        from .acquisition import acquire

        return acquire(paths)
    if command == "annotate":
        from .annotation import annotate

        return annotate(paths)
    if command == "score":
        return _run_score(paths)
    if command == "report":
        from .reporting import report

        return report(paths)
    raise ValueError(f"unsupported single stage: {command}")


def _run_all(paths: EvaluationPaths) -> tuple[dict[str, object], int]:
    """Run stages in order and stop before the next stage after any failures."""

    summaries: dict[str, object] = {}
    for stage in ("acquire", "annotate", "score"):
        result = cast(StageSummary, _run_named(cast(CommandName, stage), paths))
        summaries[stage] = result.model_dump(mode="json")
        if result.failed:
            return {"stages": summaries}, 1
    from .reporting import report

    summaries["report"] = report(paths)
    return {"stages": summaries}, 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one evaluator command and return a process exit status."""

    namespace = build_parser().parse_args(argv)
    command = cast(CommandName, namespace.command)
    paths = EvaluationPaths(cast(Path, namespace.data_dir))
    paths.ensure()
    if command == "all":
        payload, status = _run_all(paths)
        _emit(payload)
        return status
    result = _run_named(command, paths)
    if isinstance(result, StageSummary):
        _emit(_summary_payload(command, result))
        return 1 if result.failed else 0
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
