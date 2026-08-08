"""Hashing and local JPEG decoding primitives."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import Protocol, TypeGuard

from .models import AcquisitionFailure, DecodedImageMetadata


class _PillowImage(Protocol):
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


def _is_pillow_module(value: object) -> TypeGuard[_PillowModule]:
    open_method: object = getattr(value, "open", None)
    resampling: object = getattr(value, "Resampling", None)
    lanczos: object = getattr(resampling, "LANCZOS", None)
    return callable(open_method) and isinstance(lanczos, int)


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


def decode_jpeg(path: Path) -> DecodedImageMetadata:
    """Decode one JPEG and return dimensions plus a 64-bit average hash."""

    image_module = _load_pillow_module()
    try:
        with image_module.open(path) as image:
            if image.format != "JPEG":
                raise AcquisitionFailure("JPEG_REQUIRED")
            _ = image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise AcquisitionFailure("JPEG_DIMENSIONS_INVALID")
            return DecodedImageMetadata(
                width=width,
                height=height,
                perceptual_hash=perceptual_hash(image),
            )
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


__all__ = ["decode_jpeg", "hamming_distance", "perceptual_hash", "sha256_file"]
