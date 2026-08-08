"""Typed aggregate lifecycle events and sanitized stderr emission."""

from __future__ import annotations

import json
import sys
from typing import Final, Literal

from .base import EvaluationModel

AcquisitionEvent = Literal[
    "start",
    "source_page",
    "progress",
    "category_complete",
    "error",
    "complete",
]
AggregateEvent = AcquisitionEvent | Literal["model_loading", "model_ready"]
AggregateStage = Literal["acquire", "annotate", "score"]

STABLE_FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        "MODEL_BUNDLE_FAILED",
        "COCO_SOURCE_FAILED",
        "INAT_QUERY_FAILED",
        "DOWNLOAD_FAILED",
        "WRITE_FAILED",
        "ARCHIVE_INVALID",
        "COCO_ANNOTATIONS_INVALID",
        "PILLOW_UNAVAILABLE",
        "JPEG_REQUIRED",
        "JPEG_DIMENSIONS_INVALID",
        "JPEG_INVALID",
        "METADATA_INVALID",
        "MANIFEST_WRITE_FAILED",
        "QUOTA_SHORTAGE",
        "ACQUISITION_FAILED",
    }
)

EVENT_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "argiope_aurantia",
        "araneus_cavaticus",
        "araneus_diadematus",
        "latrodectus_mactans",
        "theraphosidae",
        "lycosidae",
        "scorpiones",
        "ticks_and_mites",
        "insecta",
        "crabs",
        "coco2017_validation",
    }
)


class AggregateEventRecord(EvaluationModel):
    """One sanitized aggregate event safe for stderr emission."""

    stage: AggregateStage
    event: AggregateEvent
    category: str | None = None
    page: int | None = None
    count: int | None = None
    quota: int | None = None
    attempted: int | None = None
    completed: int | None = None
    skipped: int | None = None
    failed: int | None = None
    resumed: int | None = None
    processed: int | None = None
    code: str | None = None


def emit_event(record: AggregateEventRecord) -> None:
    """Write one sanitized aggregate lifecycle event to stderr.

    Events intentionally contain no source IDs, URLs, paths, exception text, or
    image metadata. This keeps live progress useful without exposing samples.
    """

    sanitized = record.model_copy(
        update={
            "category": (
                record.category
                if record.category is None or record.category in EVENT_CATEGORIES
                else "unknown"
            ),
            "code": (
                record.code
                if record.code is None or record.code in STABLE_FAILURE_CODES
                else "ACQUISITION_FAILED"
            ),
        }
    )
    payload = sanitized.model_dump(mode="json", exclude_none=True)
    _ = sys.stderr.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )
    _ = sys.stderr.flush()


__all__ = [
    "AcquisitionEvent",
    "AggregateEvent",
    "AggregateEventRecord",
    "AggregateStage",
    "EVENT_CATEGORIES",
    "STABLE_FAILURE_CODES",
    "emit_event",
]
