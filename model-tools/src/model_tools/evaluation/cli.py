"""Command-line parser and output serialization for evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .application import (
    SingleStageCommand,
    StageExecution,
    run_all,
    run_stage,
)
from .reporting.models import EvaluationReports
from .storage.layout import EvaluationPaths


def build_parser() -> argparse.ArgumentParser:
    """Build the five-command evaluation CLI and its data-directory options.

    The ``acquire``, ``annotate``, ``score``, ``report``, and ``all`` commands
    share a default data directory, with command-local options able to
    override it.
    """
    parser = argparse.ArgumentParser(prog="model-tools-evaluation")
    _ = parser.add_argument(
        "--data-dir", type=Path, default=Path("/data"), dest="data_dir"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name in ("acquire", "annotate", "score", "report", "all"):
        command = commands.add_parser(command_name)
        _ = command.add_argument(
            "--data-dir", type=Path, default=argparse.SUPPRESS, dest="data_dir"
        )
    return parser


def _emit(payload: object) -> None:
    _ = sys.stdout.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _payload(result: StageExecution | EvaluationReports) -> object:
    return result.model_dump(mode="json")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a selected stage or the ordered aggregate and write one JSON
    line to stdout.

    Single-stage commands return status one only for failed stage summaries;
    ``all`` writes the aggregate result and returns its orchestration status.
    """
    namespace = build_parser().parse_args(argv)
    paths = EvaluationPaths(root=cast(Path, namespace.data_dir))
    paths.ensure()
    command = cast(str, namespace.command)
    if command == "all":
        result, status = run_all(paths)
        _emit(result.model_dump(mode="json"))
        return status
    stage = run_stage(cast(SingleStageCommand, command), paths)
    _emit(_payload(stage))
    if isinstance(stage, StageExecution):
        return 1 if stage.summary.failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
