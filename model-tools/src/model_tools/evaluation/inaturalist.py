"""Lazy, deterministic iNaturalist candidate selection."""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator, Mapping
from typing import Final, cast

from . import materialization
from .acquisition_types import (
    AcquisitionFailure,
    Candidate,
    ExpectedPresence,
)
from .events import emit_event

INATURALIST_BUCKETS: Final[tuple[str, ...]] = (
    "argiope_aurantia",
    "araneus_cavaticus",
    "araneus_diadematus",
    "latrodectus_mactans",
    "theraphosidae",
    "lycosidae",
)
INATURALIST_HARD_NEGATIVE_BUCKETS: Final[tuple[str, ...]] = (
    "scorpiones",
    "ticks_and_mites",
    "insecta",
    "crabs",
)
INATURALIST_TAXON_IDS: Final[dict[str, int]] = {
    "argiope_aurantia": 67707,
    "araneus_cavaticus": 143959,
    "araneus_diadematus": 52628,
    "latrodectus_mactans": 47381,
    "theraphosidae": 47424,
    "lycosidae": 47416,
    "scorpiones": 48894,
    "ticks_and_mites": 52788,
    "insecta": 47158,
    "crabs": 121639,
}
INATURALIST_PROGRESS_PAGE_CADENCE: Final[int] = 5
POSITIVE_QUOTA: Final[int] = 1_000
HARD_NEGATIVE_QUOTA: Final[int] = 750
INATURALIST_PAGE_SIZE: Final[int] = 200
PUBLIC_PHOTO_LICENSES: Final[frozenset[str]] = frozenset(
    {
        "cc0",
        "cc-by",
        "cc-by-sa",
        "cc-by-nd",
        "cc-by-nc",
        "cc-by-nc-sa",
        "cc-by-nc-nd",
        "pd",
        "public-domain",
    }
)


def _fetch_page(
    category: str,
    page: int,
    need: int,
    previous_fingerprint: tuple[str, ...] | None,
) -> tuple[list[dict[str, object]], tuple[str, ...] | None, bool]:
    try:
        payload = materialization.fetch_json(inat_url(category, page))
    except Exception as error:
        raise AcquisitionFailure("INAT_QUERY_FAILED") from error
    raw_rows_value = payload.get("results", [])
    if not isinstance(raw_rows_value, list):
        emit_event("source_page", category=category, page=page, count=0, quota=need)
        return [], None, False
    raw_rows: list[object] = cast(list[object], raw_rows_value)
    emit_event(
        "source_page",
        category=category,
        page=page,
        count=len(raw_rows),
        quota=need,
    )
    if not raw_rows:
        return [], None, False
    rows = [
        row
        for raw_row in raw_rows
        if (row := materialization.json_object(raw_row)) is not None
    ]
    fingerprint = tuple(str(row.get("id", "")) for row in rows)
    if fingerprint == previous_fingerprint:
        return [], None, False
    return rows, fingerprint, len(raw_rows) >= INATURALIST_PAGE_SIZE


def _page_candidates(
    rows: list[dict[str, object]],
    category: str,
    expected: ExpectedPresence,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for observation in sorted(rows, key=_observation_key):
        chosen = choose_observation_photo(observation)
        if chosen is None:
            continue
        source_id, image_url, license_name = chosen
        candidates.append(
            Candidate(
                source="inaturalist",
                source_id=source_id,
                source_category=category,
                expected_presence=expected,
                source_url=image_url,
                license=license_name,
                image_name=f"inat-{source_id}.jpg",
            )
        )
    return candidates


def iter_candidates(
    category: str, expected: ExpectedPresence, need: int
) -> Iterator[Candidate]:
    """Yield page-lazy iNaturalist candidates in stable source-id order.

    At most one page and its selected candidates are retained at a time. The
    caller controls iteration, allowing it to stop as soon as its quota is full.
    """

    page = 1
    previous_fingerprint: tuple[str, ...] | None = None
    while True:
        rows, fingerprint, has_next_page = _fetch_page(
            category, page, need, previous_fingerprint
        )
        if fingerprint is None:
            return
        previous_fingerprint = fingerprint
        page_candidates = _page_candidates(rows, category, expected)
        if page % INATURALIST_PROGRESS_PAGE_CADENCE == 0:
            emit_event(
                "progress",
                category=category,
                page=page,
                count=len(page_candidates),
                quota=need,
            )
        yield from page_candidates
        if not has_next_page:
            return
        page += 1


def inat_url(category: str, page: int) -> str:
    taxon_id = INATURALIST_TAXON_IDS.get(category)
    if taxon_id is None:
        raise AcquisitionFailure("INAT_QUERY_FAILED")
    query = urllib.parse.urlencode(
        {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": ",".join(sorted(PUBLIC_PHOTO_LICENSES)),
            "per_page": INATURALIST_PAGE_SIZE,
            "page": page,
            "order_by": "id",
            "order": "asc",
        }
    )
    return f"https://api.inaturalist.org/v1/observations?{query}"


def _observation_key(row: Mapping[str, object]) -> tuple[int, str]:
    raw_id = row.get("id", 0)
    if isinstance(raw_id, int):
        numeric = raw_id
    elif isinstance(raw_id, str):
        try:
            numeric = int(raw_id)
        except ValueError:
            numeric = 0
    else:
        numeric = 0
    return numeric, str(raw_id)


def choose_observation_photo(
    observation: Mapping[str, object],
) -> tuple[str, str, str] | None:
    if str(observation.get("quality_grade", "")) != "research":
        return None
    raw_id = observation.get("id")
    if raw_id is None:
        return None
    photos = observation.get("photos")
    if not isinstance(photos, list):
        return None
    raw_photos: list[object] = cast(list[object], photos)
    options: list[tuple[str, str, str]] = []
    for raw_photo in raw_photos:
        photo = materialization.json_object(raw_photo)
        if photo is None:
            continue
        license_name = (
            str(photo.get("license", photo.get("license_code", ""))).strip().lower()
        )
        image_url = photo.get("url")
        if license_name not in PUBLIC_PHOTO_LICENSES or not isinstance(image_url, str):
            continue
        photo_id = str(photo.get("id", ""))
        options.append((photo_id, image_url, license_name))
    if not options:
        return None
    _, image_url, license_name = min(options, key=lambda item: (item[0], item[1]))
    return str(raw_id), image_url, license_name


__all__ = [
    "HARD_NEGATIVE_QUOTA",
    "INATURALIST_BUCKETS",
    "INATURALIST_HARD_NEGATIVE_BUCKETS",
    "INATURALIST_PAGE_SIZE",
    "INATURALIST_PROGRESS_PAGE_CADENCE",
    "INATURALIST_TAXON_IDS",
    "POSITIVE_QUOTA",
    "PUBLIC_PHOTO_LICENSES",
    "choose_observation_photo",
    "inat_url",
]
