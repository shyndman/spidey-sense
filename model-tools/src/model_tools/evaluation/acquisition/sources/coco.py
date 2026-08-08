"""COCO 2017 validation source discovery and candidate construction."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import model_tools.evaluation.acquisition.materialization as materialization

from ...storage.layout import EvaluationPaths
from ..models import AcquisitionFailure, ArchiveImage, Candidate, CandidateBatch

COCO_QUOTA: Final[int] = 3_000
COCO_IMAGES_URL: Final[str] = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL: Final[str] = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def _json_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    mapping = cast(Mapping[object, object], value)
    result: dict[str, object] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def candidates(paths: EvaluationPaths, need: int) -> CandidateBatch:
    """Return deterministic COCO validation candidates and source failures."""

    image_archive = paths.downloads / "val2017.zip"
    annotation_archive = paths.downloads / "annotations_trainval2017.zip"
    del need
    try:
        materialization.ensure_archive(COCO_IMAGES_URL, image_archive)
        materialization.ensure_archive(COCO_ANNOTATIONS_URL, annotation_archive)
        rows = _read_coco_images(annotation_archive)
        result: list[Candidate] = []
        with zipfile.ZipFile(image_archive) as archive:
            available = set(archive.namelist())
            for row in sorted(rows, key=_coco_row_id):
                raw_id = row.id
                filename = row.file_name
                if not isinstance(raw_id, (int, str)) or not isinstance(filename, str):
                    continue
                member = f"val2017/{filename}"
                if member not in available:
                    continue
                source_id = str(raw_id)
                result.append(
                    Candidate(
                        source="coco2017",
                        source_id=source_id,
                        source_category="coco2017_validation",
                        expected_presence="broad_negative",
                        source_url=(
                            f"http://images.cocodataset.org/val2017/{filename}"
                        ),
                        license="COCO 2017",
                        image_name=f"coco-{source_id}.jpg",
                        archive_member=member,
                    )
                )
        return CandidateBatch(candidates=tuple(result), source_failures=0)
    except Exception:
        return CandidateBatch(candidates=(), source_failures=1)


def _coco_row_id(row: ArchiveImage) -> int:
    value = row.id
    return int(value) if isinstance(value, (int, str)) else -1


def _read_coco_images(annotation_archive: Path) -> tuple[ArchiveImage, ...]:
    with zipfile.ZipFile(annotation_archive) as archive:
        member = "annotations/instances_val2017.json"
        with archive.open(member) as handle:
            loaded = cast(object, json.load(handle))
    payload = _json_object(loaded)
    rows = payload.get("images") if payload is not None else None
    if not isinstance(rows, list):
        raise AcquisitionFailure("COCO_ANNOTATIONS_INVALID")
    raw_rows: list[object] = cast(list[object], rows)
    parsed: list[ArchiveImage] = []
    for raw_row in raw_rows:
        row = _json_object(raw_row)
        if row is None or row.get("id") is None:
            continue
        raw_id = row.get("id")
        if not isinstance(raw_id, (int, str)):
            continue
        raw_name = row.get("file_name")
        if raw_name is not None and not isinstance(raw_name, str):
            row["file_name"] = None
        try:
            parsed.append(ArchiveImage.model_validate(row))
        except (TypeError, ValueError):
            continue
    return tuple(parsed)


__all__ = [
    "COCO_ANNOTATIONS_URL",
    "COCO_IMAGES_URL",
    "COCO_QUOTA",
    "candidates",
]
