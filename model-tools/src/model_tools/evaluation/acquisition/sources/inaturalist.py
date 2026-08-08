"""Lazy, deterministic iNaturalist candidate selection."""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Iterator, Mapping
from typing import Final, cast

import model_tools.evaluation.acquisition.materialization as materialization

from ...events import AggregateEvent, AggregateEventRecord, emit_event
from ..models import (
    AcquisitionFailure,
    Candidate,
    ExpectedPresence,
    Observation,
    Photo,
    PhotoOption,
    PhotoSelection,
    SourcePage,
    SourcePageResult,
)


def _emit(
    event: AggregateEvent,
    *,
    category: str,
    page: int,
    count: int,
    quota: int,
) -> None:
    emit_event(
        AggregateEventRecord(
            stage="acquire",
            event=event,
            category=category,
            page=page,
            count=count,
            quota=quota,
        )
    )


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


def fetch_json(url: str) -> dict[str, object]:
    """Fetch one source JSON object with source-local raw decoding."""

    with materialization.open_url(url) as response:
        raw = response.read()
    try:
        loaded = cast(object, json.loads(raw))
    except (TypeError, ValueError) as error:
        raise AcquisitionFailure("METADATA_INVALID") from error
    payload = _json_object(loaded)
    if payload is None:
        raise AcquisitionFailure("METADATA_INVALID")
    return payload


def _parse_photo(raw: object) -> Photo | None:
    payload = _json_object(raw)
    if payload is None:
        return None
    raw_id = payload.get("id")
    if raw_id is not None and not isinstance(raw_id, (int, str)):
        return None
    raw_url = payload.get("url")
    if raw_url is not None and not isinstance(raw_url, str):
        payload["url"] = None
    try:
        return Photo.model_validate(payload)
    except (TypeError, ValueError):
        return None


def _parse_observation(raw: object) -> Observation | None:
    payload = _json_object(raw)
    if payload is None:
        return None
    raw_id = payload.get("id")
    if raw_id is not None and not isinstance(raw_id, (int, str)):
        return None
    raw_photos = payload.get("photos")
    photos: tuple[Photo, ...] | None = None
    if isinstance(raw_photos, list):
        raw_photos = cast(list[object], raw_photos)
        parsed = tuple(
            photo
            for raw_photo in raw_photos
            if (photo := _parse_photo(raw_photo)) is not None
        )
        photos = parsed
    raw_quality = payload.get("quality_grade")
    quality: str | int | float | bool | None = (
        raw_quality if isinstance(raw_quality, (str, int, float, bool)) else None
    )
    try:
        return Observation(id=raw_id, quality_grade=quality, photos=photos)
    except (TypeError, ValueError):
        return None


def _fetch_page(
    category: str,
    page: int,
    need: int,
    previous_fingerprint: tuple[str, ...] | None,
) -> SourcePageResult:
    try:
        payload = fetch_json(inat_url(category, page))
    except Exception as error:
        raise AcquisitionFailure("INAT_QUERY_FAILED") from error
    raw_rows_value = payload.get("results", [])
    if not isinstance(raw_rows_value, list):
        _emit("source_page", category=category, page=page, count=0, quota=need)
        return SourcePageResult(rows=(), fingerprint=None, has_next_page=False)
    raw_rows: list[object] = cast(list[object], raw_rows_value)
    _emit(
        "source_page",
        category=category,
        page=page,
        count=len(raw_rows),
        quota=need,
    )
    if not raw_rows:
        return SourcePageResult(rows=(), fingerprint=None, has_next_page=False)
    page_model = SourcePage.model_validate(
        {
            "results": tuple(
                row
                for raw_row in raw_rows
                if (row := _parse_observation(raw_row)) is not None
            )
        }
    )
    rows = page_model.results or ()
    fingerprint = tuple(str(row.id if row.id is not None else "") for row in rows)
    if fingerprint == previous_fingerprint:
        return SourcePageResult(rows=(), fingerprint=None, has_next_page=False)
    return SourcePageResult(
        rows=rows,
        fingerprint=fingerprint,
        has_next_page=len(raw_rows) >= INATURALIST_PAGE_SIZE,
    )


def _page_candidates(
    rows: tuple[Observation, ...],
    category: str,
    expected: ExpectedPresence,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for observation in sorted(rows, key=_observation_key):
        chosen = choose_observation_photo(observation)
        if chosen is None:
            continue
        candidates.append(
            Candidate(
                source="inaturalist",
                source_id=chosen.source_id,
                source_category=category,
                expected_presence=expected,
                source_url=chosen.image_url,
                license=chosen.license,
                image_name=f"inat-{chosen.source_id}.jpg",
            )
        )
    return candidates


def iter_candidates(
    category: str, expected: ExpectedPresence, need: int
) -> Iterator[Candidate]:
    """Yield page-lazy iNaturalist candidates in stable source-id order."""

    page = 1
    previous_fingerprint: tuple[str, ...] | None = None
    while True:
        page_result = _fetch_page(category, page, need, previous_fingerprint)
        if page_result.fingerprint is None:
            return
        previous_fingerprint = page_result.fingerprint
        page_candidates = _page_candidates(page_result.rows, category, expected)
        if page % INATURALIST_PROGRESS_PAGE_CADENCE == 0:
            _emit(
                "progress",
                category=category,
                page=page,
                count=len(page_candidates),
                quota=need,
            )
        yield from page_candidates
        if not page_result.has_next_page:
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


def _observation_key(row: Observation) -> tuple[int, str]:
    raw_id = row.id if row.id is not None else 0
    if isinstance(raw_id, int):
        numeric = raw_id
    elif type(raw_id) is str:
        try:
            numeric = int(raw_id)
        except ValueError:
            numeric = 0
    else:
        numeric = 0
    return numeric, str(raw_id)


def choose_observation_photo(observation: Observation) -> PhotoSelection | None:
    if str(observation.quality_grade or "") != "research":
        return None
    raw_id = observation.id
    if raw_id is None:
        return None
    if observation.photos is None:
        return None
    options: list[PhotoOption] = []
    for photo in observation.photos:
        license_name = (
            str(
                photo.license if photo.license is not None else photo.license_code or ""
            )
            .strip()
            .lower()
        )
        image_url = photo.url
        if license_name not in PUBLIC_PHOTO_LICENSES or not isinstance(image_url, str):
            continue
        photo_id = str(photo.id if photo.id is not None else "")
        options.append(
            PhotoOption(
                photo_id=photo_id,
                selection=PhotoSelection(
                    source_id=str(raw_id),
                    image_url=image_url,
                    license=license_name,
                ),
            )
        )
    if not options:
        return None
    return min(
        options, key=lambda item: (item.photo_id, item.selection.image_url)
    ).selection


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
    "fetch_json",
    "inat_url",
    "iter_candidates",
]
