"""Acquire or reproducibly build one pinned ONNX model artifact."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Final

from .metadata import OnnxModelSource, SourceManifest, TimmSafetensorsModelSource
from .model_export import export_timm_safetensors

DOWNLOAD_CHUNK_BYTES: Final = 1_048_576
DOWNLOAD_TIMEOUT_SECONDS: Final = 30
PART_SUFFIX: Final = ".part"


def artifact_sha256(manifest: SourceManifest) -> str:
    """Return the expected hash of the final ONNX artifact."""

    source = manifest.model
    return (
        source.sha256 if isinstance(source, OnnxModelSource) else source.artifact_sha256
    )


def artifact_size_bytes(manifest: SourceManifest) -> int:
    """Return the expected byte size of the final ONNX artifact."""

    source = manifest.model
    return (
        source.size_bytes
        if isinstance(source, OnnxModelSource)
        else source.artifact_size_bytes
    )


def file_matches(path: Path, *, expected_size: int, expected_sha256: str) -> bool:
    """Check one artifact without exposing its content."""

    try:
        return path.stat().st_size == expected_size and sha256(path) == expected_sha256
    except OSError:
        return False


def require_file(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    """Require an exact pinned artifact."""

    if not file_matches(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    ):
        raise RuntimeError(f"artifact failed size or SHA-256 verification: {path}")


def sha256(path: Path) -> str:
    """Hash one file incrementally."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def open_url(url: str) -> Generator[BinaryIO]:
    """Open a pinned source with a bounded connection timeout."""

    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        yield response


def _download(url: str, destination: Path, *, size: int, digest: str) -> None:
    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    actual_digest = hashlib.sha256()
    actual_size = 0
    try:
        with open_url(url) as response, partial.open("wb") as target:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                target.write(chunk)
                actual_digest.update(chunk)
                actual_size += len(chunk)
        if actual_size != size:
            raise RuntimeError("downloaded model source failed size verification")
        if actual_digest.hexdigest() != digest:
            raise RuntimeError("downloaded model source failed SHA-256 verification")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _source_path(destination: Path, source: TimmSafetensorsModelSource) -> Path:
    return destination.with_name(f".{source.id}.source.safetensors")


def acquire_model_artifact(
    manifest: SourceManifest,
    destination: Path,
    validate: Callable[[Path], None],
) -> None:
    """Download an ONNX graph or build one from exact pinned timm weights."""

    source = manifest.model
    if isinstance(source, OnnxModelSource):
        staged = destination.with_name(f".{destination.name}.download")
        try:
            _download(
                str(source.url),
                staged,
                size=source.size_bytes,
                digest=source.sha256,
            )
            validate(staged)
            staged.replace(destination)
        finally:
            staged.unlink(missing_ok=True)
        return

    weights_path = _source_path(destination, source)
    if not file_matches(
        weights_path,
        expected_size=source.size_bytes,
        expected_sha256=source.sha256,
    ):
        _download(
            str(source.url),
            weights_path,
            size=source.size_bytes,
            digest=source.sha256,
        )

    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    try:
        export_timm_safetensors(weights_path, partial, manifest)
        require_file(
            partial,
            expected_size=source.artifact_size_bytes,
            expected_sha256=source.artifact_sha256,
        )
        validate(partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


__all__ = [
    "acquire_model_artifact",
    "artifact_sha256",
    "artifact_size_bytes",
    "file_matches",
    "open_url",
    "require_file",
    "sha256",
]
