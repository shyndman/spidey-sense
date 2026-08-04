"""Resumable corpus state, grouping, split assignment, and persistence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .acquisition_types import (
    STABLE_FAILURE_CODES,
    AcceptedSample,
    AcquisitionFailure,
    Candidate,
    JsonObject,
)
from .contracts import SampleManifest, StageFailure
from .materialization import (
    PART_SUFFIX,
    hamming_distance,
    sha256_file,
)
from .paths import EvaluationPaths


def load_existing(paths: EvaluationPaths) -> list[AcceptedSample]:
    """Load only manifests whose images and persisted hashes remain valid."""

    result: list[AcceptedSample] = []
    for manifest_path in sorted(paths.manifests.glob("*.json"), key=lambda p: p.name):
        try:
            loaded = cast(
                object,
                json.loads(manifest_path.read_text(encoding="utf-8")),
            )
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


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], left: int, right: int) -> None:
    root_left, root_right = _find(parent, left), _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _mark_near(
    parent: list[int],
    tree: list[tuple[int, dict[int, int]]],
    target: int,
    item_index: int,
) -> None:
    pending = [0]
    while pending:
        current = pending.pop()
        node_hash, children = tree[current]
        distance = hamming_distance(node_hash, target)
        if distance <= 4:
            _union(parent, item_index, current)
        pending.extend(
            child
            for edge, child in children.items()
            if max(0, distance - 4) <= edge <= distance + 4
        )


def _hash_groups(items: list[AcceptedSample]) -> list[list[AcceptedSample]]:
    parent = list(range(len(items)))
    tree: list[tuple[int, dict[int, int]]] = []
    for index, item in enumerate(items):
        if tree:
            _mark_near(parent, tree, item.perceptual_hash, index)
        if not tree:
            tree.append((item.perceptual_hash, {}))
            continue
        node_index = 0
        while True:
            node_hash, children = tree[node_index]
            distance = hamming_distance(node_hash, item.perceptual_hash)
            child = children.get(distance)
            if child is None:
                children[distance] = len(tree)
                tree.append((item.perceptual_hash, {}))
                break
            node_index = child
    groups: dict[int, list[AcceptedSample]] = {}
    for index, item in enumerate(items):
        groups.setdefault(_find(parent, index), []).append(item)
    return sorted(
        groups.values(), key=lambda group: min(item.sample_id for item in group)
    )


def _assign_duplicate_groups(
    ordered_groups: list[list[AcceptedSample]],
) -> None:
    for group in ordered_groups:
        group_name = f"phash-{min(item.sample_id for item in group)}"
        for item in group:
            item.duplicate_group = group_name


def _assign_splits(
    items: list[AcceptedSample],
    ordered_groups: list[list[AcceptedSample]],
) -> None:
    totals: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item.candidate.source_category, item.candidate.expected_presence)
        totals[key] = totals.get(key, 0) + 1
    targets = {key: (value + 1) // 2 for key, value in totals.items()}
    calibration: dict[tuple[str, str], int] = {key: 0 for key in totals}
    for group in ordered_groups:
        contribution: dict[tuple[str, str], int] = {}
        for item in group:
            key = (item.candidate.source_category, item.candidate.expected_presence)
            contribution[key] = contribution.get(key, 0) + 1
        calibration_gain = sum(
            max(0, targets[key] - calibration[key]) * count
            for key, count in contribution.items()
        )
        test_gain = sum(
            max(0, calibration[key] - targets[key]) * count
            for key, count in contribution.items()
        )
        use_calibration = calibration_gain >= test_gain
        for item in group:
            item.split = "calibration" if use_calibration else "test"
        if use_calibration:
            for key, count in contribution.items():
                calibration[key] += count


def assign_groups_and_splits(items: list[AcceptedSample]) -> None:
    """Union near hashes and assign complete groups to deterministic splits."""
    ordered_groups = _hash_groups(items)
    _assign_duplicate_groups(ordered_groups)
    _assign_splits(items, ordered_groups)


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
    payload = cast(JsonObject, manifest.model_dump(mode="json"))
    destination = paths.manifest_path(item.sample_id)
    if destination.is_file():
        try:
            loaded = cast(
                object,
                json.loads(destination.read_text(encoding="utf-8")),
            )
            if loaded == payload:
                return
        except (OSError, ValueError, TypeError, UnicodeError):
            pass
    write_json_atomic(destination, payload)


def write_json_atomic(destination: Path, payload: Mapping[str, object]) -> None:
    """Persist one JSON object through a same-directory partial file."""

    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    try:
        with partial.open("w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            _ = output.write("\n")
            output.flush()
            _ = os.fsync(output.fileno())
        os.replace(partial, destination)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise AcquisitionFailure("MANIFEST_WRITE_FAILED") from error


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
    payload = cast(JsonObject, failure.model_dump(mode="json"))
    write_json_atomic(paths.errors / filename, payload)


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
    payload = cast(JsonObject, failure.model_dump(mode="json"))
    write_json_atomic(paths.errors / filename, payload)


def failure_code(error: AcquisitionFailure) -> str:
    """Reduce an exception to the stable failure code contract."""

    raw: object = error.args[0] if error.args else ""
    if isinstance(raw, str) and raw in STABLE_FAILURE_CODES:
        return raw
    return "ACQUISITION_FAILED"


__all__ = [
    "PART_SUFFIX",
    "assign_groups_and_splits",
    "coerce_hash",
    "failure_code",
    "load_existing",
    "resolve_image_path",
    "write_failure",
    "write_json_atomic",
    "write_manifest",
    "write_shortage",
]
