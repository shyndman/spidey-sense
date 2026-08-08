"""Persistent layout and typed JSON boundary tests."""

import os
from pathlib import Path
from typing import cast

import pytest
from model_tools.evaluation.acquisition.models import SampleManifest
from model_tools.evaluation.base import EvaluationModel
from model_tools.evaluation.storage.json import (
    ACQUISITION_JSON_PROFILE,
    ANNOTATION_JSON_PROFILE,
    REPORT_JSON_PROFILE,
    SCORE_JSON_PROFILE,
    JsonWriteProfile,
    read_model,
    write_model,
)
from model_tools.evaluation.storage.layout import EvaluationPaths


class _ProfileProbe(EvaluationModel):
    zulu: str
    alpha: str
    count: int


def _manifest() -> SampleManifest:
    return SampleManifest(
        sample_id="sample-proxy",
        source="coco2017",
        source_id="source-proxy",
        source_category="category-proxy",
        expected_presence="broad_negative",
        source_url="https://example.invalid/proxy",
        license="license-proxy",
        image_relative_path="images/sample-proxy.jpg",
        sha256="a" * 64,
        perceptual_hash="0" * 16,
        duplicate_group="group-proxy",
        split="calibration",
        width=32,
        height=24,
    )


def test_paths_coerce_proxy_root_and_create_layout(tmp_path: Path) -> None:
    proxy = EvaluationPaths(root=cast(Path, cast(object, "/proxy")))
    assert proxy.root == Path("/proxy")
    paths = EvaluationPaths(root=Path(str(tmp_path / "data")))
    assert paths.root == Path(tmp_path / "data")
    paths.ensure()
    paths.ensure_model("model-proxy")
    assert {path.name for path in paths.root.iterdir()} == {
        "images",
        "manifests",
        "annotations",
        "scores",
        "errors",
        "reports",
        "downloads",
        "models",
        "cache",
        "tmp",
    }
    assert (
        paths.image_path("images/sample-proxy.jpg") == paths.images / "sample-proxy.jpg"
    )
    assert (
        paths.score_path("model-proxy", "sample-proxy")
        == paths.scores / "model-proxy" / "sample-proxy.json"
    )
    assert (
        paths.model_error_path("model-proxy", "sample-proxy")
        == paths.errors / "model-proxy" / "score-sample-proxy.json"
    )


def test_paths_reject_ambiguous_image_paths(tmp_path: Path) -> None:
    paths = EvaluationPaths(root=tmp_path / "data")
    for invalid_path in (
        "images//sample.jpg",
        "images/nested/sample.jpg",
        "images/../sample.jpg",
        "images/foo\\bar.jpg",
        "images/",
    ):
        with pytest.raises(ValueError):
            _ = paths.image_path(invalid_path)


def test_all_json_profiles_have_exact_bytes_and_clean_partials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace
    recorded_sources: list[Path] = []

    def replace_spy(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        recorded_sources.append(Path(source))
        real_replace(source, destination)

    monkeypatch.setattr("model_tools.evaluation.storage.json.os.replace", replace_spy)

    probe = _ProfileProbe(zulu="café-proxy", alpha="proxy", count=7)
    profiles: tuple[tuple[str, JsonWriteProfile, bytes], ...] = (
        (
            "acquisition",
            ACQUISITION_JSON_PROFILE,
            b'{"alpha":"proxy","count":7,"zulu":"caf\\u00e9-proxy"}\n',
        ),
        (
            "annotation",
            ANNOTATION_JSON_PROFILE,
            b'{"zulu":"caf\\u00e9-proxy","alpha":"proxy","count":7}\n',
        ),
        (
            "score",
            SCORE_JSON_PROFILE,
            b'{"alpha":"proxy","count":7,"zulu":"caf\xc3\xa9-proxy"}\n',
        ),
        (
            "report",
            REPORT_JSON_PROFILE,
            b'{"alpha":"proxy","count":7,"zulu":"caf\\u00e9-proxy"}\n',
        ),
    )
    for name, profile, expected in profiles:
        destination = tmp_path / name / "probe.json"
        write_model(destination, probe, profile=profile)
        assert destination.read_bytes() == expected
        assert destination.read_bytes().endswith(b"\n")
        assert not destination.with_name(".probe.json.part").exists()
        assert not destination.with_name("probe.json.part").exists()
    assert [source.name for source in recorded_sources] == [
        ".probe.json.part",
        ".probe.json.part",
        "probe.json.part",
        ".probe.json.part",
    ]


def test_typed_json_read_write_and_cleanup(tmp_path: Path) -> None:
    destination = EvaluationPaths(root=tmp_path / "data").manifest_path("sample-proxy")
    write_model(destination, _manifest(), profile=ACQUISITION_JSON_PROFILE)
    assert destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()
    loaded = read_model(destination, SampleManifest)
    assert loaded == _manifest()
