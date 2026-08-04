"""Provision the pinned model bundle used by evaluation."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypeGuard

from .paths import EvaluationPaths


class _BundleAcquirer(Protocol):
    def __call__(self, manifest_path: Path, destination: Path) -> object: ...


class _BundleModule(Protocol):
    acquire_bundle: _BundleAcquirer


def _is_bundle_module(value: object) -> TypeGuard[_BundleModule]:
    acquire_bundle: object = getattr(value, "acquire_bundle", None)
    return callable(acquire_bundle)


def provision_model_bundle(paths: EvaluationPaths) -> bool:
    """Ensure the pinned MobileNet bundle exists in the persistent volume."""

    configured = os.environ.get("MODEL_SOURCE_MANIFEST")
    candidates = ((Path(configured),) if configured else ()) + (
        Path("/app/model-tools/model-source.toml"),
        Path(__file__).resolve().parents[3] / "model-source.toml",
    )
    manifest_path = next(
        (candidate for candidate in candidates if candidate.is_file()), None
    )
    if manifest_path is None:
        return False
    try:
        module = import_module("model_tools.acquire")
        if not _is_bundle_module(module):
            return False
        _ = module.acquire_bundle(manifest_path, paths.models)
    except Exception:
        return False
    return True


__all__ = ["provision_model_bundle"]
