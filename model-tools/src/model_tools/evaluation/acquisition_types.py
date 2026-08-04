"""Typed records and stable identifiers shared by acquisition modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

Source = Literal["inaturalist", "coco2017"]
ExpectedPresence = Literal["positive", "hard_negative", "broad_negative"]
Split = Literal["calibration", "test"]
AcquisitionEvent = Literal[
    "start",
    "source_page",
    "progress",
    "category_complete",
    "error",
    "complete",
]
JsonObject = dict[str, object]

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

# Categories are intentionally a closed, sanitized set for aggregate events.
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


class AcquisitionFailure(RuntimeError):
    """A source item did not satisfy the corpus contract."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One deterministic source item eligible for materialization."""

    source: Source
    source_id: str
    source_category: str
    expected_presence: ExpectedPresence
    source_url: str
    license: str
    image_name: str
    image_bytes: bytes | None = None
    archive_member: str | None = None


@dataclass(slots=True)
class AcceptedSample:
    """A candidate whose bytes and JPEG metadata satisfy the corpus contract."""

    candidate: Candidate
    sample_id: str
    image_path: Path
    sha256: str
    perceptual_hash: int
    width: int
    height: int
    duplicate_group: str = ""
    split: Split = "calibration"


__all__ = [
    "AcquisitionEvent",
    "AcquisitionFailure",
    "AcceptedSample",
    "Candidate",
    "EVENT_CATEGORIES",
    "ExpectedPresence",
    "JsonObject",
    "STABLE_FAILURE_CODES",
    "Source",
    "Split",
]
