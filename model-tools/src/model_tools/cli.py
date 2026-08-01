"""Command-line entrypoint for model tooling."""

from argparse import ArgumentParser
from importlib.metadata import version
from typing import Final

_DISTRIBUTION: Final = "model-tools"


def build_parser() -> ArgumentParser:
    """Build the command-line parser."""
    parser = ArgumentParser(prog="model-tools", description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version(_DISTRIBUTION)}",
    )
    return parser


def main() -> None:
    """Run the model-tools command-line interface."""
    build_parser().parse_args()
