"""Manifest, resume, and sanitized acquisition failure persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from ..application import StageFailure
from ..events import STABLE_FAILURE_CODES
from ..storage.json import ACQUISITION_JSON_PROFILE, write_model
from ..storage.layout import EvaluationPaths
from .hashing import sha256_file
from .models import (
    AcceptedSample,
    AcquisitionFailure,
    Candidate,
    SampleManifest,
)


def _json_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)


def load_existing(paths: EvaluationPaths) -> list[AcceptedSample]:
    """Load only manifests whose images and persisted hashes remain valid."""

    result: list[AcceptedSample] = []
    for manifest_path in sorted(paths.manifests.glob("*.json"), key=lambda p: p.name):
        try:
            loaded = _json_object(
                cast(object, json.loads(manifest_path.read_text(encoding="utf-8")))
            )
            if loaded is None:
                continue
            manifest = SampleManifest.model_validate(loaded)
            relative_path = Path(manifest.image_relative_path)
            if (
                len(relative_path.parts) != 2
                or relative_path.parts[0] != paths.images.name
            ):
                continue
            image_path_value = resolve_image_path(paths, manifest.image_relative_path)
            if image_path_value.parent != paths.images:
                continue
            if (
                not image_path_value.is_file()
                or sha256_file(image_path_value) != manifest.sha256
            ):
                continue
            result.append(
                AcceptedSample(
                    candidate=Candidate(
                        source=manifest.source,
                        source_id=manifest.source_id,
                        source_category=manifest.source_category,
                        expected_presence=manifest.expected_presence,
                        source_url=manifest.source_url,
                        license=manifest.license,
                        image_name=image_path_value.name,
                    ),
                    sample_id=manifest.sample_id,
                    image_path=image_path_value,
                    sha256=manifest.sha256,
                    perceptual_hash=coerce_hash(manifest.perceptual_hash),
                    width=manifest.width,
                    height=manifest.height,
                    duplicate_group=manifest.duplicate_group,
                    split=manifest.split,
                )
            )
        except (OSError, ValueError, TypeError, OverflowError, UnicodeError):
            continue
    return result


def resolve_image_path(paths: EvaluationPaths, image_relative_path: str) -> Path:
    """Resolve a root-relative ``images/<filename>`` path without traversal."""

    relative = Path(image_relative_path)
    if (
        len(relative.parts) != 2
        or relative.parts[0] != paths.images.name
        or relative.parts[1] in {"", ".", ".."}
        or "\\" in image_relative_path
        or relative.is_absolute()
        or image_relative_path != f"{paths.images.name}/{relative.parts[1]}"
    ):
        raise ValueError("image path must be root-relative images/<filename>")
    return paths.images / relative.parts[1]


def coerce_hash(value: str | int) -> int:
    """Parse persisted hexadecimal perceptual hashes."""

    if isinstance(value, int):
        return value
    text = value.strip().lower()
    return int(text, 16) if text.startswith("0x") else int(text, 16)


def write_manifest(paths: EvaluationPaths, item: AcceptedSample) -> None:
    """Persist one source manifest if its exact JSON payload has changed."""

    manifest = SampleManifest(
        schema_version=1,
        sample_id=item.sample_id,
        source=item.candidate.source,
        source_id=item.candidate.source_id,
        source_category=item.candidate.source_category,
        expected_presence=item.candidate.expected_presence,
        source_url=item.candidate.source_url,
        license=item.candidate.license,
        image_relative_path=str(item.image_path.relative_to(paths.root)),
        sha256=item.sha256,
        perceptual_hash=f"{item.perceptual_hash:016x}",
        duplicate_group=item.duplicate_group,
        split=item.split,
        width=item.width,
        height=item.height,
    )
    payload = manifest.model_dump(mode="json")
    destination = paths.manifest_path(item.sample_id)
    if destination.is_file():
        try:
            loaded = _json_object(
                cast(object, json.loads(destination.read_text(encoding="utf-8")))
            )
            if loaded == payload:
                return
        except (OSError, ValueError, TypeError, UnicodeError):
            pass
    write_model(destination, manifest, profile=ACQUISITION_JSON_PROFILE)


def write_failure(paths: EvaluationPaths, code: str, sample_id: str | None) -> None:
    """Persist only sanitized stage, code, and sample identity under errors."""

    safe_code = code if code in STABLE_FAILURE_CODES else "ACQUISITION_FAILED"
    safe_sample = (
        ""
        if sample_id is None
        else "".join(
            character if character.isalnum() or character in "_-" else "_"
            for character in sample_id
        )
    )
    filename = f"acquire-{safe_code}-{safe_sample or 'stage'}.json"
    failure = StageFailure(
        schema_version=1,
        stage="acquire",
        code=safe_code,
        sample_id=safe_sample or None,
    )
    write_model(paths.errors / filename, failure, profile=ACQUISITION_JSON_PROFILE)


def write_shortage(paths: EvaluationPaths, category: str, count: int) -> None:
    """Persist one sanitized quota-shortage record."""

    del count
    safe_category = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in category
    )
    filename = f"acquire-shortage-{safe_category}.json"
    failure = StageFailure(
        schema_version=1,
        stage="acquire",
        code="QUOTA_SHORTAGE",
        sample_id=None,
    )
    write_model(paths.errors / filename, failure, profile=ACQUISITION_JSON_PROFILE)


def failure_code(error: AcquisitionFailure) -> str:
    """Reduce an exception to the stable failure code contract."""

    raw: object = error.args[0] if error.args else ""
    if isinstance(raw, str) and raw in STABLE_FAILURE_CODES:
        return raw
    return "ACQUISITION_FAILED"


__all__ = [
    "coerce_hash",
    "failure_code",
    "load_existing",
    "resolve_image_path",
    "write_failure",
    "write_manifest",
    "write_shortage",
]
