"""Persistent Docker-volume paths for evaluation data."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from pydantic import field_validator

from ..base import EvaluationModel


class EvaluationPaths(EvaluationModel):
    """All evaluation data locations rooted inside the persistent ``/data`` volume."""

    root: Path

    @field_validator("root", mode="before")
    @classmethod
    def coerce_root_path(cls, value: str | PathLike[str]) -> Path:
        """Retain the dataclass API's string-to-``Path`` coercion."""

        return Path(value)

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
        """Root directory for model-isolated score records."""

        return self.root / "scores"

    @property
    def errors(self) -> Path:
        """Root directory for shared and model-isolated failure records."""

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
        """Root directory for isolated model bundles."""

        return self.root / "models"

    @property
    def cache(self) -> Path:
        """Reusable network and model cache directory."""

        return self.root / "cache"

    @property
    def tmp(self) -> Path:
        """Scratch directory for resumable stage work."""

        return self.root / "tmp"

    def model_bundle(self, model_id: str) -> Path:
        """Return one model's isolated artifact bundle directory."""

        return self.models / _safe_component(model_id)

    def model_scores(self, model_id: str) -> Path:
        """Return one model's isolated per-sample score directory."""

        return self.scores / _safe_component(model_id)

    def model_errors(self, model_id: str) -> Path:
        """Return one model's isolated score-failure directory."""

        return self.errors / _safe_component(model_id)

    def model_reports(self, model_id: str) -> Path:
        """Return one model's isolated aggregate report directory."""

        return self.reports / _safe_component(model_id)

    def ensure_model(self, model_id: str) -> None:
        """Create all runtime directories owned by one registered model."""

        for directory in (
            self.model_bundle(model_id),
            self.model_scores(model_id),
            self.model_errors(model_id),
            self.model_reports(model_id),
        ):
            directory.mkdir(parents=True, exist_ok=True)

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

    def score_path(self, model_id: str, sample_id: str) -> Path:
        """Return one model's atomic per-sample score path."""

        return self._sample_json_path(self.model_scores(model_id), sample_id)

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

    def model_report_path(
        self,
        model_id: str,
        name: str = "aggregate.json",
    ) -> Path:
        """Return one model's aggregate report path."""

        return self.model_reports(model_id) / f"{_safe_component(Path(name).stem)}.json"

    def model_error_path(self, model_id: str, sample_id: str | None = None) -> Path:
        """Return one model's opaque score-failure path."""

        name = "score" if sample_id is None else f"score-{_safe_component(sample_id)}"
        return self.model_errors(model_id) / f"{name}.json"

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


__all__ = ["EvaluationPaths"]
