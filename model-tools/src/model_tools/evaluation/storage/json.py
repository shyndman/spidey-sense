"""Typed JSON persistence with the evaluation byte profiles."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from ..base import EvaluationModel


class JsonWriteProfile(EvaluationModel):
    """JSON encoder and partial-file behavior for one record family."""

    sort_keys: bool
    ensure_ascii: bool
    dot_partial: bool
    allow_nan: bool


ACQUISITION_JSON_PROFILE = JsonWriteProfile(
    sort_keys=True,
    ensure_ascii=True,
    dot_partial=True,
    allow_nan=True,
)
ANNOTATION_JSON_PROFILE = JsonWriteProfile(
    sort_keys=False,
    ensure_ascii=True,
    dot_partial=True,
    allow_nan=True,
)
SCORE_JSON_PROFILE = JsonWriteProfile(
    sort_keys=True,
    ensure_ascii=False,
    dot_partial=False,
    allow_nan=False,
)
REPORT_JSON_PROFILE = JsonWriteProfile(
    sort_keys=True,
    ensure_ascii=True,
    dot_partial=True,
    allow_nan=True,
)


def read_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    """Read and strictly validate one JSON object or array model."""

    payload = Path(path).read_bytes()
    return model_type.model_validate_json(payload)


def write_model(
    path: Path,
    value: BaseModel,
    *,
    profile: JsonWriteProfile,
) -> None:
    """Serialize and atomically persist one validated model."""

    payload = value.model_dump(mode="json")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=profile.ensure_ascii,
            sort_keys=profile.sort_keys,
            separators=(",", ":"),
            allow_nan=profile.allow_nan,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, encoded, dot_partial=profile.dot_partial)


def atomic_write_bytes(path: Path, payload: bytes, *, dot_partial: bool) -> None:
    """Persist bytes through a same-directory fsynced atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_name = (
        f".{destination.name}.part" if dot_partial else f"{destination.name}.part"
    )
    partial = destination.with_name(partial_name)
    try:
        with partial.open("wb") as handle:
            _ = handle.write(payload)
            _ = handle.flush()
            _ = os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


__all__ = [
    "ACQUISITION_JSON_PROFILE",
    "ANNOTATION_JSON_PROFILE",
    "JsonWriteProfile",
    "REPORT_JSON_PROFILE",
    "SCORE_JSON_PROFILE",
    "atomic_write_bytes",
    "read_model",
    "write_model",
]
