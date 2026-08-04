"""Persistent Docker-volume paths and crash-safe JSON record helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel

JsonValue = (
    Mapping[str, object]
    | list[object]
    | tuple[object, ...]
    | str
    | int
    | float
    | bool
    | None
)


@dataclass(frozen=True, slots=True)
class EvaluationPaths:
    """All evaluation data locations rooted inside the persistent ``/data`` volume."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def images(self) -> Path:
        """JPEG corpus directory."""
        return self.root / "images"

    @property
    def manifests(self) -> Path:
        """Per-sample source manifests directory."""
        return self.root / "manifests"

    @property
    def annotations(self) -> Path:
        """Per-sample detector annotation directory."""
        return self.root / "annotations"

    @property
    def scores(self) -> Path:
        """Per-sample MobileNet score directory."""
        return self.root / "scores"

    @property
    def errors(self) -> Path:
        """Opaque stage-failure directory."""
        return self.root / "errors"

    @property
    def reports(self) -> Path:
        """Aggregate report directory."""
        return self.root / "reports"

    @property
    def downloads(self) -> Path:
        """Download/cache staging directory."""
        return self.root / "downloads"

    @property
    def models(self) -> Path:
        """Locally persisted detector and model checkpoints directory."""
        return self.root / "models"

    @property
    def cache(self) -> Path:
        """Reusable network and model cache directory."""
        return self.root / "cache"

    @property
    def tmp(self) -> Path:
        """Scratch directory for resumable stage work."""
        return self.root / "tmp"

    def ensure(self) -> None:
        """Create every required directory, including the volume root."""
        for directory in (
            self.root,
            self.images,
            self.manifests,
            self.annotations,
            self.scores,
            self.errors,
            self.reports,
            self.downloads,
            self.models,
            self.cache,
            self.tmp,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, sample_id: str) -> Path:
        """Return the atomic JSON path for one manifest sample."""
        return self._sample_json_path(self.manifests, sample_id)

    def annotation_path(self, sample_id: str) -> Path:
        """Return the atomic JSON path for one annotation sample."""
        return self._sample_json_path(self.annotations, sample_id)

    def score_path(self, sample_id: str) -> Path:
        """Return the atomic JSON path for one score sample."""
        return self._sample_json_path(self.scores, sample_id)

    def image_path(self, image_relative_path: str) -> Path:
        """Resolve root-relative ``images/<filename>`` inside the data volume."""
        relative = Path(image_relative_path)
        if (
            not image_relative_path
            or image_relative_path.endswith("/")
            or "\\" in image_relative_path
            or relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != self.images.name
            or relative.parts[1] in {"", ".", ".."}
            or image_relative_path != f"{self.images.name}/{relative.parts[1]}"
        ):
            raise ValueError("image path must be root-relative images/<filename>")
        return self.images / relative.parts[1]

    def error_path(self, stage: str, sample_id: str | None = None) -> Path:
        """Return an opaque JSON failure path, optionally scoped to a sample."""
        stage_name = _safe_component(stage)
        name = (
            stage_name
            if sample_id is None
            else f"{stage_name}-{_safe_component(sample_id)}"
        )
        return self.errors / f"{name}.json"

    def report_path(self, name: str = "aggregate.json") -> Path:
        """Return an aggregate report path under ``reports``."""
        return self.reports / f"{_safe_component(Path(name).stem)}.json"

    def write_json(self, path: Path, value: JsonValue | BaseModel) -> None:
        """Write one JSON object atomically through a same-directory ``.part`` file."""
        atomic_write_json(path, value)

    def read_json[T: BaseModel](
        self,
        path: Path,
        model_type: type[T] | None = None,
    ) -> T | dict[str, object] | list[object]:
        """Read JSON and optionally validate it against a strict Pydantic model."""
        return read_json(path, model_type)

    @staticmethod
    def _sample_json_path(directory: Path, sample_id: str) -> Path:
        return directory / f"{_safe_component(sample_id)}.json"


def _safe_component(value: str) -> str:
    """Reject path separators and traversal in opaque record identifiers."""
    if not value or value in {".", ".."}:
        raise ValueError("path component must be a non-empty name")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError("path component must not contain path separators")
    return value


def _json_payload(value: JsonValue | BaseModel) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def atomic_write_json(path: Path, value: JsonValue | BaseModel) -> None:
    """Persist JSON with flush, fsync, close, and atomic rename semantics."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f"{destination.name}.part")
    try:
        with part.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_payload(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            _ = handle.write("\n")
            _ = handle.flush()
            _ = os.fsync(handle.fileno())
        os.replace(part, destination)
    finally:
        part.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: JsonValue | BaseModel) -> None:
    """Compatibility spelling for the public atomic JSON writer."""
    atomic_write_json(path, value)


def read_json[T: BaseModel](
    path: Path,
    model_type: type[T] | None = None,
) -> T | dict[str, object] | list[object]:
    """Read a JSON file and optionally return a validated model instance."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = cast(object, json.load(handle))
    if isinstance(payload, dict):
        json_value: dict[str, object] | list[object] = cast(
            dict[str, object],
            payload,
        )
    elif isinstance(payload, list):
        json_value = cast(list[object], payload)
    else:
        raise ValueError("JSON root must be an object or array")
    if model_type is None:
        return json_value
    return model_type.model_validate(json_value)
