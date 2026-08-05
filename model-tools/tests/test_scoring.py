"""Harmless-proxy tests for validation-only model scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from model_tools.evaluation import scoring
from model_tools.evaluation.contracts import SampleManifest, ScoreRecord
from model_tools.evaluation.paths import EvaluationPaths
from model_tools.metadata import (
    ArtifactMetadata,
    ClassGroups,
    InputMetadata,
    LabelRecord,
    ModelMetadata,
    OutputMetadata,
)

_MODEL_ID = "evaluation-model"


class _FakeSession:
    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[object]:
        assert output_names == ["output"]
        assert input_feed["input"].shape == (1, 3, 4, 4)
        logits = np.zeros((1, 1000), dtype=np.float32)
        logits[0, 3] = 2.0
        return [logits]


def _input_metadata() -> InputMetadata:
    return InputMetadata(
        name="input",
        data_type="float32",
        layout="NCHW",
        shape=(None, 3, 4, 4),
        color_space="RGB",
        resize_mode="contain",
        allow_upscale=True,
        interpolation="bilinear",
        padding_mode="black",
        pixel_scale=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        standard_deviation=(1.0, 1.0, 1.0),
    )


def _artifact_metadata() -> ArtifactMetadata:
    labels = tuple(
        LabelRecord(index=index, synset=f"n{index:08d}", label=f"class-{index}")
        for index in range(1000)
    )
    return ArtifactMetadata(
        schema_version=2,
        model=ModelMetadata(
            id=_MODEL_ID,
            filename=f"{_MODEL_ID}.onnx",
            sha256="a" * 64,
            size_bytes=1,
            format="onnx",
            opset=17,
            source_url="https://example.invalid/model",
            source_revision="pinned",
        ),
        input=_input_metadata(),
        output=OutputMetadata(
            name="output",
            data_type="float32",
            shape=(None, 1000),
            activation="softmax",
            labels=labels,
        ),
        classes=ClassGroups(blocked=(labels[3],), debug=(labels[4],)),
    )


def _write_proxy_image(path: Path) -> None:
    image = Image.new("RGB", (4, 2), color=(255, 255, 255))
    image.save(path, format="JPEG", quality=100, subsampling=0)


def test_preprocess_contains_and_black_pads_proxy(tmp_path: Path) -> None:
    image_path = tmp_path / "proxy.jpg"
    _write_proxy_image(image_path)

    assert _input_metadata().pixel_scale != 1.0
    tensor = scoring.preprocess_image(image_path, _input_metadata())

    assert tensor.shape == (1, 3, 4, 4)
    assert bool(np.all(tensor[:, :, 0, :] == 0.0))
    assert bool(np.all(tensor[:, :, 3, :] == 0.0))
    assert bool(np.all(tensor[:, :, 1:3, :] == 1.0))


def test_score_model_writes_isolated_record_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = EvaluationPaths(tmp_path)
    paths.ensure()
    image_path = paths.images / "sample-1.jpg"
    _write_proxy_image(image_path)
    manifest = SampleManifest(
        sample_id="sample-1",
        source="coco2017",
        source_id="proxy-1",
        source_category="ordinary-negative",
        expected_presence="broad_negative",
        source_url="https://example.invalid/metadata",
        license="CC-BY-4.0",
        image_relative_path="images/sample-1.jpg",
        sha256="b" * 64,
        perceptual_hash="0" * 16,
        duplicate_group="proxy-1",
        split="test",
        width=4,
        height=2,
    )
    paths.write_json(paths.manifest_path(manifest.sample_id), manifest)
    loaded = scoring.LoadedModel(
        metadata=_artifact_metadata(),
        session=_FakeSession(),
    )
    load_calls = 0

    def fake_load(_paths, _model_id):
        nonlocal load_calls
        load_calls += 1
        return loaded

    monkeypatch.setattr(scoring, "_load_model", fake_load)

    first = scoring.score_model(paths, _MODEL_ID)
    second = scoring.score_model(paths, _MODEL_ID)

    assert load_calls == 1
    assert first.model_dump() == {
        "schema_version": 1,
        "attempted": 1,
        "completed": 1,
        "skipped": 0,
        "failed": 0,
    }
    assert second.completed == 0
    assert second.skipped == 1
    score = ScoreRecord.model_validate_json(
        paths.score_path(_MODEL_ID, "sample-1").read_bytes()
    )
    assert score.model_id == _MODEL_ID
    assert score.sample_id == "sample-1"
    assert score.top_index == 3
    assert paths.model_errors(_MODEL_ID).is_dir()
