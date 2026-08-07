"""Command-line entrypoint for reproducible model bundle acquisition."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal

from structlog.stdlib import BoundLogger, get_logger

from .acquire import BundlePaths, acquire_bundle, verify_bundle
from .metadata import ArtifactMetadata

_DISTRIBUTION: Final = "model-tools"
_LOGGER: BoundLogger = get_logger()


class _CommandNamespace(Namespace):
    """Typed attributes populated by the command-line parser."""

    command: str | None = None
    manifest: Path | None = None
    output_dir: Path | None = None
    bundle_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class AcquireCommand:
    """Arguments for acquiring or reusing the pinned model bundle."""

    name: Literal["acquire"]
    manifest: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class VerifyCommand:
    """Arguments for verifying an existing model bundle offline."""

    name: Literal["verify"]
    manifest: Path
    bundle_dir: Path


type Command = AcquireCommand | VerifyCommand


def build_parser() -> ArgumentParser:
    """Build the command-line parser."""

    parser = ArgumentParser(prog="model-tools", description=__doc__)
    _ = parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version(_DISTRIBUTION)}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser(
        "acquire",
        help="Download or reuse the pinned model and generate verified metadata.",
    )
    _ = acquire.add_argument("--manifest", required=True, type=Path)
    _ = acquire.add_argument("--output-dir", required=True, type=Path)

    verify = subparsers.add_parser(
        "verify",
        help="Verify an existing model bundle without network access.",
    )
    _ = verify.add_argument("--manifest", required=True, type=Path)
    _ = verify.add_argument("--bundle-dir", required=True, type=Path)
    return parser


def _command_from_namespace(
    parser: ArgumentParser, namespace: _CommandNamespace
) -> Command:
    command = namespace.command
    manifest = namespace.manifest
    if not isinstance(manifest, Path):
        parser.error("--manifest must be a path")
    if command == "acquire":
        output_dir = namespace.output_dir
        if not isinstance(output_dir, Path):
            parser.error("--output-dir must be a path")
        return AcquireCommand(
            name="acquire",
            manifest=manifest,
            output_dir=output_dir,
        )
    if command == "verify":
        bundle_dir = namespace.bundle_dir
        if not isinstance(bundle_dir, Path):
            parser.error("--bundle-dir must be a path")
        return VerifyCommand(
            name="verify",
            manifest=manifest,
            bundle_dir=bundle_dir,
        )
    parser.error(f"unknown command: {command}")


def _run_command(command: Command) -> tuple[BundlePaths, ArtifactMetadata]:
    if isinstance(command, AcquireCommand):
        paths = acquire_bundle(command.manifest, command.output_dir)
        metadata = verify_bundle(
            command.manifest,
            command.output_dir,
            run_inference=False,
        )
        return paths, metadata

    metadata = verify_bundle(
        command.manifest,
        command.bundle_dir,
        run_inference=True,
    )
    return (
        BundlePaths(
            model=command.bundle_dir / metadata.model.filename,
            metadata=command.bundle_dir / f"{metadata.model.id}.metadata.json",
        ),
        metadata,
    )


def _log_success(
    command: Command,
    paths: BundlePaths,
    metadata: ArtifactMetadata,
) -> None:
    _LOGGER.info(
        "model_bundle_verified",
        command=command.name,
        model_id=metadata.model.id,
        model_path=str(paths.model),
        metadata_path=str(paths.metadata),
        sha256=metadata.model.sha256,
        input={
            "name": metadata.input.name,
            "dataType": metadata.input.data_type,
            "shape": metadata.input.shape,
        },
        output={
            "name": metadata.output.name,
            "dataType": metadata.output.data_type,
            "shape": metadata.output.shape,
        },
    )


def main() -> None:
    """Acquire or verify a model bundle and report one structured result."""

    parser = build_parser()
    namespace = parser.parse_args(namespace=_CommandNamespace())
    command = _command_from_namespace(parser, namespace)
    try:
        paths, metadata = _run_command(command)
    except Exception as error:
        _LOGGER.error(
            "model_bundle_command_failed",
            command=command.name,
            stage=command.name,
            error=str(error),
        )
        raise SystemExit(1) from None
    _log_success(command, paths, metadata)
