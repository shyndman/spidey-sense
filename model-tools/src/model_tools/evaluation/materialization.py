"""Shared acquisition materialization, transport, and image primitives.

This module owns byte movement and local image validation only. Source adapters
provide candidates; corpus persistence and split policy live in :mod:`corpus`.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import IO, BinaryIO, Final, Protocol, TypeGuard, cast

from .acquisition_types import (
    AcceptedSample,
    AcquisitionFailure,
    Candidate,
    JsonObject,
)
from .paths import EvaluationPaths

DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 60
PART_SUFFIX: Final[str] = ".part"


class _PillowImage(Protocol):
    """Minimal typed surface used by the local Pillow decoder."""

    format: str | None
    size: tuple[int, int]

    def load(self) -> object: ...

    def convert(self, mode: str) -> _PillowImage: ...

    def resize(self, size: tuple[int, int], resample: int) -> _PillowImage: ...

    def getdata(self) -> Iterable[int | float]: ...


class _PillowImageContext(Protocol):
    def __enter__(self) -> _PillowImage: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _PillowResampling(Protocol):
    LANCZOS: int


class _PillowModule(Protocol):
    Resampling: _PillowResampling

    def open(self, path: Path) -> _PillowImageContext: ...


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


def _is_pillow_module(value: object) -> TypeGuard[_PillowModule]:
    open_method: object = getattr(value, "open", None)
    resampling: object = getattr(value, "Resampling", None)
    lanczos: object = getattr(resampling, "LANCZOS", None)
    return callable(open_method) and isinstance(lanczos, int)


def _is_http_response(value: object) -> TypeGuard[_HTTPResponse]:
    read: object = getattr(value, "read", None)
    enter: object = getattr(value, "__enter__", None)
    exit_method: object = getattr(value, "__exit__", None)
    return callable(read) and callable(enter) and callable(exit_method)


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
            else paths.downloads
            / f"{candidate.archive_member.partition('/')[0]}.zip"
        )
        extract_archive_member(source_archive, candidate.archive_member, destination)
    elif candidate.image_bytes is not None:
        write_bytes_atomic(destination, candidate.image_bytes)
    elif not destination.is_file():
        download_file(candidate.source_url, destination)
    try:
        sha256 = sha256_file(destination)
        width, height, perceptual_hash_value = decode_jpeg(destination)
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
        perceptual_hash=perceptual_hash_value,
        width=width,
        height=height,
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
    return response


def fetch_json(url: str) -> JsonObject:
    """Fetch and decode one JSON object from a source endpoint."""

    with open_url(url) as response:
        raw = response.read()
    try:
        loaded = cast(object, json.loads(raw))
    except (TypeError, ValueError) as error:
        raise AcquisitionFailure("METADATA_INVALID") from error
    payload = json_object(loaded)
    if payload is None:
        raise AcquisitionFailure("METADATA_INVALID")
    return payload


def json_object(value: object) -> JsonObject | None:
    """Convert a JSON mapping to a string-keyed object, rejecting other values."""

    if not isinstance(value, Mapping):
        return None
    mapping = cast(Mapping[object, object], value)
    result: JsonObject = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def copy_stream(source: _ReadableBytes | IO[bytes], output: BinaryIO) -> int:
    """Copy bytes in bounded chunks and return the number copied."""

    total = 0
    while True:
        chunk = source.read(1_048_576)
        if not chunk:
            return total
        _ = output.write(chunk)
        total += len(chunk)


def sha256_file(path: Path) -> str:
    """Hash one file without retaining its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pillow_module() -> _PillowModule:
    try:
        module = import_module("PIL.Image")
    except ImportError as error:
        raise AcquisitionFailure("PILLOW_UNAVAILABLE") from error
    if not _is_pillow_module(module):
        raise AcquisitionFailure("PILLOW_UNAVAILABLE")
    return module


def decode_jpeg(path: Path) -> tuple[int, int, int]:
    """Decode one JPEG and return its dimensions plus a 64-bit average hash."""

    image_module = _load_pillow_module()
    try:
        with image_module.open(path) as image:
            if image.format != "JPEG":
                raise AcquisitionFailure("JPEG_REQUIRED")
            _ = image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise AcquisitionFailure("JPEG_DIMENSIONS_INVALID")
            return width, height, perceptual_hash(image)
    except AcquisitionFailure:
        raise
    except Exception as error:
        raise AcquisitionFailure("JPEG_INVALID") from error


def perceptual_hash(image: _PillowImage) -> int:
    """Compute a deterministic 64-bit average hash without retaining pixels."""

    image_module = _load_pillow_module()
    gray = image.convert("L").resize((8, 8), image_module.Resampling.LANCZOS)
    values = list(gray.getdata())
    mean = sum(values) / len(values)
    result = 0
    for value in values:
        result = (result << 1) | int(value >= mean)
    return result & ((1 << 64) - 1)


def hamming_distance(left: int, right: int) -> int:
    """Return the bit distance between two perceptual hashes."""

    return (left ^ right).bit_count()


__all__ = [
    "DOWNLOAD_TIMEOUT_SECONDS",
    "PART_SUFFIX",
    "candidate_id",
    "copy_stream",
    "decode_jpeg",
    "download_file",
    "ensure_archive",
    "extract_archive_member",
    "fetch_json",
    "hamming_distance",
    "image_path",
    "json_object",
    "materialize_candidate",
    "open_url",
    "perceptual_hash",
    "sha256_file",
    "write_bytes_atomic",
]
