"""Download, archive extraction, and atomic acquisition byte movement."""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path
from types import TracebackType
from typing import IO, BinaryIO, Final, Protocol, cast

from ..storage.layout import EvaluationPaths
from .hashing import decode_jpeg, sha256_file
from .models import AcceptedSample, AcquisitionFailure, Candidate

DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 60
PART_SUFFIX: Final[str] = ".part"


class _ReadableBytes(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _HTTPResponse(_ReadableBytes, Protocol):
    def __enter__(self) -> _HTTPResponse: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


def _is_http_response(value: object) -> bool:
    return all(
        callable(getattr(value, name, None))
        for name in ("read", "__enter__", "__exit__")
    )


def image_path(paths: EvaluationPaths, image_name: str) -> Path:
    """Resolve one candidate filename beneath the corpus image directory."""

    relative = Path(image_name)
    if (
        "\\" in image_name
        or relative.is_absolute()
        or len(relative.parts) != 1
        or relative.parts[0] in {"", ".", ".."}
    ):
        raise AcquisitionFailure("METADATA_INVALID")
    return paths.images / relative


def candidate_id(candidate: Candidate) -> str:
    """Return the stable opaque identity used for sample records."""

    return f"{candidate.source}-{candidate.source_id}"


def materialize_candidate(
    paths: EvaluationPaths,
    candidate: Candidate,
    archive_path: Path | None = None,
) -> AcceptedSample:
    """Materialize and validate one candidate, atomically where bytes change."""

    sample_id = candidate_id(candidate)
    destination = image_path(paths, candidate.image_name)
    if candidate.archive_member is not None:
        source_archive = (
            archive_path
            if archive_path is not None
            else paths.downloads / f"{candidate.archive_member.partition('/')[0]}.zip"
        )
        extract_archive_member(source_archive, candidate.archive_member, destination)
    elif candidate.image_bytes is not None:
        write_bytes_atomic(destination, candidate.image_bytes)
    elif not destination.is_file():
        download_file(candidate.source_url, destination)
    try:
        sha256 = sha256_file(destination)
        metadata = decode_jpeg(destination)
    except Exception as error:
        destination.unlink(missing_ok=True)
        if isinstance(error, AcquisitionFailure):
            raise
        raise AcquisitionFailure("JPEG_INVALID") from error
    return AcceptedSample(
        candidate=candidate,
        sample_id=sample_id,
        image_path=destination,
        sha256=sha256,
        perceptual_hash=metadata.perceptual_hash,
        width=metadata.width,
        height=metadata.height,
    )


def ensure_archive(url: str, destination: Path) -> None:
    """Download a valid ZIP archive through a same-directory partial file."""

    if destination.is_file() and zipfile.is_zipfile(destination):
        return
    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    partial.unlink(missing_ok=True)
    try:
        with open_url(url) as response, partial.open("wb") as output:
            _ = copy_stream(response, output)
            output.flush()
            _ = os.fsync(output.fileno())
        if not zipfile.is_zipfile(partial):
            raise AcquisitionFailure("ARCHIVE_INVALID")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def download_file(url: str, destination: Path) -> None:
    """Download one file with atomic replacement and sanitized failures."""

    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    partial.unlink(missing_ok=True)
    try:
        with open_url(url) as response, partial.open("wb") as output:
            _ = copy_stream(response, output)
            output.flush()
            _ = os.fsync(output.fileno())
        os.replace(partial, destination)
    except Exception as error:
        partial.unlink(missing_ok=True)
        if isinstance(error, AcquisitionFailure):
            raise
        raise AcquisitionFailure("DOWNLOAD_FAILED") from error


def write_bytes_atomic(destination: Path, payload: bytes) -> None:
    """Write bytes through a flushed, fsynced, same-directory partial file."""

    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    try:
        with partial.open("wb") as output:
            _ = output.write(payload)
            output.flush()
            _ = os.fsync(output.fileno())
        os.replace(partial, destination)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise AcquisitionFailure("WRITE_FAILED") from error


def extract_archive_member(
    archive_path: Path,
    member: str,
    destination: Path,
) -> None:
    """Extract one archive member to a destination atomically."""

    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    partial.unlink(missing_ok=True)
    try:
        with (
            zipfile.ZipFile(archive_path) as archive,
            archive.open(member) as source,
            partial.open("wb") as output,
        ):
            _ = copy_stream(source, output)
            output.flush()
            _ = os.fsync(output.fileno())
        os.replace(partial, destination)
    except Exception as error:
        partial.unlink(missing_ok=True)
        if isinstance(error, AcquisitionFailure):
            raise
        raise AcquisitionFailure("DOWNLOAD_FAILED") from error


def open_url(url: str) -> _HTTPResponse:
    """Open a source URL with the evaluator's stable user agent and timeout."""

    request = urllib.request.Request(
        url, headers={"User-Agent": "spidey-sense-evaluator/1"}
    )
    response = cast(
        object,
        urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS),
    )
    if not _is_http_response(response):
        raise TypeError("urlopen returned an incompatible response")
    return cast(_HTTPResponse, response)


def copy_stream(source: _ReadableBytes | IO[bytes], output: BinaryIO) -> int:
    """Copy bytes in bounded chunks and return the number copied."""

    total = 0
    while True:
        chunk = source.read(1_048_576)
        if not chunk:
            return total
        _ = output.write(chunk)
        total += len(chunk)


__all__ = [
    "DOWNLOAD_TIMEOUT_SECONDS",
    "PART_SUFFIX",
    "candidate_id",
    "copy_stream",
    "download_file",
    "ensure_archive",
    "extract_archive_member",
    "image_path",
    "materialize_candidate",
    "open_url",
    "write_bytes_atomic",
]
