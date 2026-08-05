"""Provision every pinned model bundle used by evaluation."""

from __future__ import annotations

import os
from pathlib import Path

from model_tools.acquire import acquire_bundle, load_source_manifest
from model_tools.metadata import ArtifactMetadata

from .paths import EvaluationPaths


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


def registered_model_ids(paths: EvaluationPaths) -> tuple[str, ...]:
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
    return tuple(model_ids)


__all__ = ["provision_model_bundle", "registered_model_ids"]
