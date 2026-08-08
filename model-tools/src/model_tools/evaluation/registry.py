"""Provision and discover the registered evaluation model bundles."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import ClassVar, Protocol, cast

from pydantic import ConfigDict, RootModel

from .storage.layout import EvaluationPaths


class _ModelIdentity(Protocol):
    id: str


class _SourceManifest(Protocol):
    model: _ModelIdentity


class _BundlePaths(Protocol):
    model: Path
    metadata: Path


class _AcquireModule(Protocol):
    def load_source_manifest(self, path: Path) -> _SourceManifest: ...

    def acquire_bundle(self, manifest_path: Path, output_dir: Path) -> _BundlePaths: ...


class _ArtifactMetadata(Protocol):
    model: _ModelIdentity

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray
    ) -> _ArtifactMetadata: ...


class _MetadataModule(Protocol):
    ArtifactMetadata: type[_ArtifactMetadata]


_acquire_module = cast(
    _AcquireModule,
    cast(object, importlib.import_module("model_tools.acquire")),
)
load_source_manifest = _acquire_module.load_source_manifest
acquire_bundle = _acquire_module.acquire_bundle
_metadata_module = cast(
    _MetadataModule,
    cast(object, importlib.import_module("model_tools.metadata")),
)
ArtifactMetadata = _metadata_module.ArtifactMetadata


class RegisteredModels(RootModel[tuple[str, ...]]):
    """Manifest-order model identifiers available for evaluation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        strict=True,
        validate_assignment=True,
    )

    root: tuple[str, ...]


def _manifest_paths() -> tuple[Path, ...]:
    configured = os.environ.get("MODEL_SOURCE_MANIFEST")
    if configured:
        path = Path(configured)
        return (path,) if path.is_file() else ()

    candidates = (
        Path("/app/model-tools/model-sources"),
        Path(__file__).resolve().parents[3] / "model-sources",
    )
    directory = next((path for path in candidates if path.is_dir()), None)
    if directory is None:
        return ()
    return tuple(sorted(directory.glob("*.toml"), key=lambda path: path.name))


def provision_model_bundle(paths: EvaluationPaths) -> bool:
    """Ensure every registered model has an isolated verified bundle."""

    manifests = _manifest_paths()
    if not manifests:
        return False
    try:
        model_ids: set[str] = set()
        for manifest_path in manifests:
            manifest = load_source_manifest(manifest_path)
            model_id = manifest.model.id
            if model_id in model_ids:
                return False
            model_ids.add(model_id)
            _ = acquire_bundle(manifest_path, paths.model_bundle(model_id))
    except Exception:
        return False
    return True


def registered_model_ids(paths: EvaluationPaths) -> RegisteredModels:
    """Return model IDs whose isolated metadata validates exactly."""

    manifests = _manifest_paths()
    if not manifests:
        raise ValueError("no registered model manifests")
    model_ids: list[str] = []
    seen: set[str] = set()
    for manifest_path in manifests:
        manifest = load_source_manifest(manifest_path)
        model_id = manifest.model.id
        if model_id in seen:
            raise ValueError(f"duplicate registered model ID: {model_id}")
        seen.add(model_id)
        metadata_path = paths.model_bundle(model_id) / f"{model_id}.metadata.json"
        try:
            metadata = ArtifactMetadata.model_validate_json(metadata_path.read_bytes())
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid registered model bundle: {model_id}") from error
        if metadata.model.id != model_id:
            raise ValueError(f"model bundle ID does not match manifest: {model_id}")
        model_ids.append(model_id)
    return RegisteredModels(root=tuple(model_ids))


__all__ = ["RegisteredModels", "provision_model_bundle", "registered_model_ids"]
