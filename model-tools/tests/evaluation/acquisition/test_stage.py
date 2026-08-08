"""Focused acquisition invariants using numeric and mocked source fixtures."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlsplit

import pytest
from model_tools.evaluation import registry
from model_tools.evaluation.acquisition import (
    materialization,
    repository,
    splitting,
    stage,
)
from model_tools.evaluation.acquisition import run as acquisition_run
from model_tools.evaluation.acquisition.models import (
    AcceptedSample,
    AcquisitionFailure,
    AcquisitionRequest,
    Candidate,
    CandidateBatch,
    ExpectedPresence,
    Observation,
    Photo,
)
from model_tools.evaluation.acquisition.sources import coco, inaturalist
from model_tools.evaluation.storage.json import JsonWriteProfile
from model_tools.evaluation.storage.layout import EvaluationPaths
from pydantic import BaseModel


def _observation(source_id: int) -> dict[str, object]:
    return {
        "id": source_id,
        "quality_grade": "research",
        "photos": [
            {
                "id": source_id,
                "url": f"https://example.invalid/{source_id}.jpg",
                "license": "cc-by",
            }
        ],
    }


def _accepted(
    candidate: Candidate,
    image_path: Path,
    sha256: str,
    sample_id: str | None = None,
) -> AcceptedSample:
    return AcceptedSample(
        candidate=candidate,
        sample_id=sample_id or f"inaturalist-{candidate.source_id}",
        image_path=image_path,
        sha256=sha256,
        perceptual_hash=1,
        width=1,
        height=1,
    )


def _results_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"results": rows}


def _json_object(text: str) -> dict[str, object]:
    loaded = cast(object, json.loads(text))
    if not isinstance(loaded, dict):
        raise AssertionError("expected a JSON object")
    return cast(dict[str, object], loaded)


def _events(stderr: str) -> list[dict[str, object]]:
    return [_json_object(line) for line in stderr.splitlines() if line]


def test_observation_photo_selection_requires_research_grade_and_public_license() -> (
    None
):
    observation = Observation(
        id=42,
        quality_grade="research",
        photos=(
            Photo(
                id=9,
                url="https://example.invalid/late.jpg",
                license="cc-by",
            ),
            Photo(
                id=2,
                url="https://example.invalid/early.jpg",
                license="cc-by-sa",
            ),
            Photo(
                id=1,
                url="https://example.invalid/private.jpg",
                license="all-rights-reserved",
            ),
        ),
    )
    selected = inaturalist.choose_observation_photo(observation)
    assert selected is not None
    assert (selected.source_id, selected.image_url, selected.license) == (
        "42",
        "https://example.invalid/early.jpg",
        "cc-by-sa",
    )
    assert (
        inaturalist.choose_observation_photo(
            observation.model_copy(update={"quality_grade": "needs_id"})
        )
        is None
    )


def test_near_duplicate_groups_never_cross_deterministic_split() -> None:
    candidate = Candidate(
        source="inaturalist",
        source_id="1",
        source_category="argiope_aurantia",
        expected_presence="positive",
        source_url="https://example.invalid/image.jpg",
        license="cc-by",
        image_name="inat-1.jpg",
    )
    items = [
        AcceptedSample(
            candidate=candidate,
            sample_id=str(index),
            image_path=Path(f"{index}.jpg"),
            sha256=str(index),
            perceptual_hash=hash_value,
            width=1,
            height=1,
        )
        for index, hash_value in enumerate(
            (
                0,
                1,
                1 << 5,
                (1 << 20) | (1 << 21) | (1 << 22) | (1 << 23) | (1 << 24),
            )
        )
    ]
    items = list(splitting.assign_groups_and_splits(items))
    grouped: dict[str, set[str]] = {}
    for item in items:
        grouped.setdefault(item.duplicate_group, set()).add(item.split)
    assert (
        items[0].duplicate_group == items[1].duplicate_group == items[2].duplicate_group
    )
    assert items[3].duplicate_group != items[0].duplicate_group


def test_shortage_summary_is_numeric_and_errors_contain_no_source_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = EvaluationPaths(root=tmp_path)

    def fake_inat(
        _category: str, _expected: ExpectedPresence, _need: int
    ) -> Iterator[Candidate]:
        return iter(())

    def fake_coco(_paths: EvaluationPaths, _need: int) -> CandidateBatch:
        return CandidateBatch(candidates=(), source_failures=0)

    def fake_provision(_paths: EvaluationPaths) -> bool:
        return True

    monkeypatch.setattr(inaturalist, "iter_candidates", fake_inat)
    monkeypatch.setattr(coco, "candidates", fake_coco)
    monkeypatch.setattr(registry, "provision_model_bundle", fake_provision)
    summary = acquisition_run(AcquisitionRequest(paths=paths)).summary
    assert summary.attempted == 12_000
    assert summary.completed == 0
    assert summary.skipped == 0
    assert summary.failed == 12_000
    for error_path in paths.errors.glob("*.json"):
        payload = _json_object(error_path.read_text(encoding="utf-8"))
        assert set(payload) == {"schema_version", "stage", "code", "sample_id"}
        assert "http" not in error_path.read_text(encoding="utf-8").lower()
    events = _events(capsys.readouterr().err)
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "complete"
    assert {event["event"] for event in events} >= {
        "category_complete",
        "error",
    }
    assert events[-1]["attempted"] == summary.attempted
    assert events[-1]["failed"] == summary.failed
    forbidden = (str(tmp_path), "sample-secret", "raw source failure", "http")
    assert all(
        not any(token in json.dumps(event).lower() for token in forbidden)
        for event in events
    )


def test_every_inaturalist_category_url_uses_numeric_taxon_id() -> None:
    expected = {
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
    assert inaturalist.INATURALIST_TAXON_IDS == expected
    for category, taxon_id in expected.items():
        query = parse_qs(urlsplit(inaturalist.inat_url(category, 3)).query)
        assert query["taxon_id"] == [str(taxon_id)]
        assert "taxon_name" not in query


def test_source_page_and_cadenced_progress_events_are_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(inaturalist, "INATURALIST_PAGE_SIZE", 2)
    monkeypatch.setattr(inaturalist, "INATURALIST_PROGRESS_PAGE_CADENCE", 2)

    def fake_fetch(_url: str) -> dict[str, object]:
        page = int(parse_qs(urlsplit(_url).query)["page"][0])
        if page == 3:
            return _results_payload([])
        rows: list[dict[str, object]] = [
            {"id": page * 10 + index} for index in range(2)
        ]
        return _results_payload(rows)

    monkeypatch.setattr(inaturalist, "fetch_json", fake_fetch)
    candidate_iterator = inaturalist.iter_candidates("insecta", "hard_negative", 1)
    _ = list(candidate_iterator)
    events = _events(capsys.readouterr().err)
    assert [event["event"] for event in events] == [
        "source_page",
        "source_page",
        "progress",
        "source_page",
    ]
    assert events[2]["page"] == inaturalist.INATURALIST_PROGRESS_PAGE_CADENCE
    allowed = {
        "stage",
        "event",
        "category",
        "page",
        "count",
        "quota",
        "attempted",
        "completed",
        "skipped",
        "failed",
        "resumed",
        "code",
    }
    assert all(set(event) <= allowed for event in events)


def test_materialization_errors_never_leak_source_or_exception_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = EvaluationPaths(root=tmp_path)
    candidate = Candidate(
        source="inaturalist",
        source_id="sample-secret",
        source_category="argiope_aurantia",
        expected_presence="positive",
        source_url="https://secret.invalid/private.jpg",
        license="cc-by",
        image_name="sample-secret.jpg",
    )

    def fake_inat(
        category: str, _expected: ExpectedPresence, _need: int
    ) -> Iterator[Candidate]:
        if category == "argiope_aurantia":
            return iter((candidate,))
        return iter(())

    def fake_coco(_paths: EvaluationPaths, _need: int) -> CandidateBatch:
        return CandidateBatch(candidates=(), source_failures=0)

    def fake_materialize(_paths: EvaluationPaths, _item: Candidate) -> AcceptedSample:
        raise AcquisitionFailure("raw source failure https://secret.invalid")

    def fake_provision(_paths: EvaluationPaths) -> bool:
        return True

    monkeypatch.setattr(inaturalist, "iter_candidates", fake_inat)
    monkeypatch.setattr(coco, "candidates", fake_coco)
    monkeypatch.setattr(materialization, "materialize_candidate", fake_materialize)
    monkeypatch.setattr(registry, "provision_model_bundle", fake_provision)
    _ = acquisition_run(AcquisitionRequest(paths=paths))
    stderr = capsys.readouterr().err
    assert "secret.invalid" not in stderr
    assert "sample-secret" not in stderr
    assert str(tmp_path) not in stderr
    assert "raw source failure" not in stderr
    assert all(
        set(_json_object(line))
        <= {
            "stage",
            "event",
            "category",
            "page",
            "count",
            "quota",
            "attempted",
            "completed",
            "skipped",
            "failed",
            "resumed",
            "code",
        }
        for line in stderr.splitlines()
        if line
    )


def test_inaturalist_candidates_are_page_lazy_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inaturalist, "INATURALIST_PAGE_SIZE", 2)
    calls: list[int] = []

    def fake_fetch(url: str) -> dict[str, object]:
        page = int(parse_qs(urlsplit(url).query)["page"][0])
        calls.append(page)
        rows = [_observation(page * 10 + index) for index in range(2)]
        return _results_payload(rows)

    monkeypatch.setattr(inaturalist, "fetch_json", fake_fetch)
    candidate_iterator = cast(
        Generator[Candidate],
        inaturalist.iter_candidates("insecta", "hard_negative", 1),
    )
    assert calls == []
    assert not isinstance(candidate_iterator, list)
    first = next(candidate_iterator)
    assert first.source_id == "10"
    assert calls == [1]
    candidate_iterator.close()


def test_inaturalist_repeated_full_page_stops_with_bounded_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inaturalist, "INATURALIST_PAGE_SIZE", 2)
    calls: list[int] = []
    repeated_page = _results_payload([_observation(10), _observation(11)])

    def fake_fetch(url: str) -> dict[str, object]:
        page = int(parse_qs(urlsplit(url).query)["page"][0])
        calls.append(page)
        return repeated_page

    monkeypatch.setattr(inaturalist, "fetch_json", fake_fetch)
    candidate_iterator = inaturalist.iter_candidates("insecta", "hard_negative", 1)
    assert list(candidate_iterator)
    assert calls == [1, 2]


def test_acquire_stops_huge_inaturalist_source_when_quota_is_filled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inaturalist, "INATURALIST_BUCKETS", ("insecta",))
    monkeypatch.setattr(inaturalist, "INATURALIST_HARD_NEGATIVE_BUCKETS", ())
    monkeypatch.setattr(inaturalist, "POSITIVE_QUOTA", 1)
    monkeypatch.setattr(inaturalist, "HARD_NEGATIVE_QUOTA", 0)
    monkeypatch.setattr(coco, "COCO_QUOTA", 0)
    monkeypatch.setattr(inaturalist, "INATURALIST_PAGE_SIZE", 2)

    def fake_provision(_paths: EvaluationPaths) -> bool:
        return True

    monkeypatch.setattr(registry, "provision_model_bundle", fake_provision)
    paths = EvaluationPaths(root=tmp_path)
    calls: list[int] = []

    def fake_fetch(url: str) -> dict[str, object]:
        page = int(parse_qs(urlsplit(url).query)["page"][0])
        calls.append(page)
        rows = [_observation(page * 10 + index) for index in range(2)]
        return _results_payload(rows)

    def fake_materialize(
        paths: EvaluationPaths, candidate: Candidate
    ) -> AcceptedSample:
        return _accepted(
            candidate,
            paths.images / candidate.image_name,
            f"sha-{candidate.source_id}",
        )

    monkeypatch.setattr(inaturalist, "fetch_json", fake_fetch)
    monkeypatch.setattr(materialization, "materialize_candidate", fake_materialize)
    summary = acquisition_run(AcquisitionRequest(paths=paths)).summary
    assert summary.completed == 1
    assert summary.failed == 0
    assert calls == [1]


def test_acquire_uses_later_pages_after_resumed_duplicate_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(inaturalist, "INATURALIST_BUCKETS", ("insecta",))
    monkeypatch.setattr(inaturalist, "INATURALIST_HARD_NEGATIVE_BUCKETS", ())
    monkeypatch.setattr(inaturalist, "POSITIVE_QUOTA", 2)
    monkeypatch.setattr(inaturalist, "HARD_NEGATIVE_QUOTA", 0)
    monkeypatch.setattr(coco, "COCO_QUOTA", 0)
    monkeypatch.setattr(inaturalist, "INATURALIST_PAGE_SIZE", 2)
    monkeypatch.setattr(stage, "_MATERIALIZATION_PROGRESS_ATTEMPT_CADENCE", 1)

    def fake_provision(_paths: EvaluationPaths) -> bool:
        return True

    monkeypatch.setattr(registry, "provision_model_bundle", fake_provision)
    paths = EvaluationPaths(root=tmp_path)
    resumed_candidate = Candidate(
        source="inaturalist",
        source_id="100",
        source_category="insecta",
        expected_presence="positive",
        source_url="https://example.invalid/100.jpg",
        license="cc-by",
        image_name="inat-100.jpg",
    )
    resumed = _accepted(
        resumed_candidate,
        paths.images / "existing.jpg",
        "existing-sha",
    )

    def fake_load_existing(_paths: EvaluationPaths) -> list[AcceptedSample]:
        return [resumed]

    monkeypatch.setattr(repository, "load_existing", fake_load_existing)
    pages: dict[int, dict[str, object]] = {
        1: _results_payload([_observation(100), _observation(101)]),
        2: _results_payload([_observation(102), _observation(103)]),
    }
    calls: list[int] = []

    def fake_fetch(url: str) -> dict[str, object]:
        page = int(parse_qs(urlsplit(url).query)["page"][0])
        calls.append(page)
        return pages[page]

    def fake_materialize(
        paths: EvaluationPaths, candidate: Candidate
    ) -> AcceptedSample:
        if candidate.source_id == "101":
            raise AcquisitionFailure("DOWNLOAD_FAILED")
        sha = "existing-sha" if candidate.source_id == "100" else "new-sha"
        return _accepted(
            candidate,
            paths.images / f"alias-{candidate.source_id}.jpg",
            sha,
        )

    monkeypatch.setattr(inaturalist, "fetch_json", fake_fetch)
    monkeypatch.setattr(materialization, "materialize_candidate", fake_materialize)

    def fake_write_manifest(_paths: EvaluationPaths, _item: AcceptedSample) -> None:
        return None

    monkeypatch.setattr(repository, "write_manifest", fake_write_manifest)
    summary = acquisition_run(AcquisitionRequest(paths=paths)).summary
    assert summary.completed == 1
    assert summary.skipped == 1
    assert summary.failed == 1
    assert calls == [1, 2]
    events = _events(capsys.readouterr().err)
    progress_indices = [
        index for index, event in enumerate(events) if event["event"] == "progress"
    ]
    complete_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "category_complete"
    )
    assert progress_indices
    assert max(progress_indices) < complete_index
    assert all(
        set(event)
        <= {
            "stage",
            "event",
            "category",
            "page",
            "count",
            "quota",
            "attempted",
            "completed",
            "skipped",
            "failed",
            "resumed",
            "code",
        }
        for event in events
    )


def test_write_manifest_skips_identical_proxy_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = EvaluationPaths(root=tmp_path)
    candidate = Candidate(
        source="coco2017",
        source_id="source-proxy",
        source_category="category-proxy",
        expected_presence="broad_negative",
        source_url="https://example.invalid/proxy",
        license="license-proxy",
        image_name="sample-proxy.jpg",
    )
    item = _accepted(
        candidate,
        paths.images / "sample-proxy.jpg",
        sha256="a" * 64,
        sample_id="sample-proxy",
    ).model_copy(update={"duplicate_group": "group-proxy"})
    repository.write_manifest(paths, item)

    calls = 0

    def fail_if_rewritten(
        _path: Path, _value: BaseModel, *, profile: JsonWriteProfile
    ) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError(f"unexpected manifest rewrite with {profile=}")

    monkeypatch.setattr(repository, "write_model", fail_if_rewritten)
    repository.write_manifest(paths, item)
    assert calls == 0
