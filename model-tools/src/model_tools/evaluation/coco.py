"""COCO 2017 validation source discovery and candidate construction."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from . import materialization
from .acquisition_types import AcquisitionFailure, Candidate, JsonObject
from .paths import EvaluationPaths

COCO_QUOTA: Final[int] = 3_000
COCO_IMAGES_URL: Final[str] = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL: Final[str] = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def candidates(paths: EvaluationPaths, need: int) -> tuple[list[Candidate], int]:
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
                raw_id = row.get("id")
                filename = row.get("file_name")
                if not isinstance(raw_id, (int, str)) or not isinstance(
                    filename, str
                ):
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
                            "http://images.cocodataset.org/val2017/"
                            f"{filename}"
                        ),
                        license="COCO 2017",
                        image_name=f"coco-{source_id}.jpg",
                        archive_member=member,
                    )
                )
        return result, 0
    except Exception:
        return [], 1


def _coco_row_id(row: Mapping[str, object]) -> int:
    value = row.get("id")
    return int(value) if isinstance(value, (int, str)) else -1


def _read_coco_images(annotation_archive: Path) -> list[JsonObject]:
    with zipfile.ZipFile(annotation_archive) as archive:
        member = "annotations/instances_val2017.json"
        with archive.open(member) as handle:
            loaded = cast(object, json.load(handle))
    payload = materialization.json_object(loaded)
    rows = payload.get("images") if payload is not None else None
    if not isinstance(rows, list):
        raise AcquisitionFailure("COCO_ANNOTATIONS_INVALID")
    raw_rows: list[object] = cast(list[object], rows)
    result = [
        row
        for raw_row in raw_rows
        if (
            (row := materialization.json_object(raw_row)) is not None
            and row.get("id") is not None
        )
    ]
    return result


__all__ = [
    "COCO_ANNOTATIONS_URL",
    "COCO_IMAGES_URL",
    "COCO_QUOTA",
    "candidates",
]
