"""Pure numeric contract tests for detector proposal annotation helpers."""

from __future__ import annotations

import math

from model_tools.evaluation import annotation
from model_tools.evaluation.annotation import (
    MAX_DETECTIONS,
    _build_annotation,
    convert_normalized_boxes_to_pixel_xyxy,
    rank_target_aligned_proposals,
)
from model_tools.evaluation.contracts import SampleManifest
from model_tools.evaluation.paths import EvaluationPaths


def test_box_conversion_is_pixel_xyxy_and_finite() -> None:
    boxes = convert_normalized_boxes_to_pixel_xyxy(
        [[0.5, 0.5, 0.5, 0.25], [float("nan"), float("inf"), 2.0, 2.0]],
        width=640,
        height=480,
    )

    assert boxes[0] == (160.0, 180.0, 480.0, 300.0)
    assert all(math.isfinite(value) for box in boxes for value in box)
    assert all(0.0 <= value <= 640.0 for value in boxes[1][::2])
    assert all(0.0 <= value <= 480.0 for value in boxes[1][1::2])


def test_ranking_retains_very_low_confidence_proposals_without_cutoff() -> None:
    logits = [[-50.0, -50.0], [-4.0, -3.0], [4.0, -5.0]]
    boxes = [[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]]

    proposals = rank_target_aligned_proposals(
        logits,
        boxes,
        width=100,
        height=80,
        phrases=("spider", "tarantula"),
    )

    assert len(proposals) == 3
    assert [proposal.query_index for proposal in proposals] == [2, 1, 0]
    assert proposals[-1].confidence > 0.0
    assert proposals[-1].confidence < 1e-10


def test_ranking_caps_to_top_twenty_and_uses_token_spans() -> None:
    logits = [[-10.0, -10.0, -10.0] for _ in range(MAX_DETECTIONS + 3)]
    logits[2] = [-10.0, 8.0, -10.0]
    boxes = [[0.5, 0.5, 0.2, 0.2] for _ in logits]

    proposals = rank_target_aligned_proposals(
        logits,
        boxes,
        width=32,
        height=32,
        token_spans={"spider": (0,), "tarantula": (1,), "black widow": (2,)},
        phrases=("spider", "tarantula", "black widow"),
    )

    assert len(proposals) == MAX_DETECTIONS
    assert proposals[0].query_index == 2
    assert proposals[0].phrase == "tarantula"
    assert all(math.isfinite(proposal.confidence) for proposal in proposals)
    assert all(
        math.isfinite(value) for proposal in proposals for value in proposal.box_xyxy
    )


def test_build_annotation_emits_contract_record_from_numeric_model_output() -> None:
    class FakeProcessor:
        def __call__(self, **_kwargs: object) -> dict[str, object]:
            return {"pixel_values": [[0.0]]}

    class FakeModel:
        def __call__(self, **_kwargs: object) -> dict[str, object]:
            return {
                "logits": [[-12.0] * 6 for _ in range(MAX_DETECTIONS + 5)],
                "pred_boxes": [[0.5, 0.5, 0.2, 0.2] for _ in range(MAX_DETECTIONS + 5)],
            }

    sample = SampleManifest(
        sample_id="sample-1",
        source="inaturalist",
        source_id="observation-1",
        source_category="argiope_aurantia",
        expected_presence="positive",
        source_url="source-1",
        license="cc0",
        image_relative_path="sample-1.jpg",
        sha256="a" * 64,
        perceptual_hash="b" * 16,
        duplicate_group="group-1",
        split="calibration",
        width=64,
        height=48,
    )

    record = _build_annotation(sample, FakeProcessor(), FakeModel(), image=None)

    assert record.sample_id == sample.sample_id
    assert len(record.detections) == MAX_DETECTIONS
    assert [detection.rank for detection in record.detections] == list(
        range(1, MAX_DETECTIONS + 1)
    )
    assert record.max_confidence > 0.0
    assert all(math.isfinite(detection.confidence) for detection in record.detections)
    assert all(
        math.isfinite(value)
        for detection in record.detections
        for value in detection.box_xyxy
    )


def test_load_runtime_selects_cuda_and_moves_model_to_it(tmp_path, monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

    class FakeModel:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def to(self, device: str) -> FakeModel:
            self.devices.append(device)
            return self

        def eval(self) -> FakeModel:
            return self

    fake_model = FakeModel()

    class FakeProcessorClass:
        @staticmethod
        def from_pretrained(_checkpoint: str, *, cache_dir: str) -> object:
            return object()

    class FakeModelClass:
        @staticmethod
        def from_pretrained(_checkpoint: str, *, cache_dir: str) -> FakeModel:
            return fake_model

    class FakeTransformers:
        AutoProcessor = FakeProcessorClass
        AutoModelForZeroShotObjectDetection = FakeModelClass

    modules: dict[str, object] = {
        "torch": FakeTorch(),
        "transformers": FakeTransformers,
    }
    monkeypatch.setattr(
        annotation.importlib,
        "import_module",
        lambda name: modules[name],
    )

    _torch, _processor, model, device = annotation._load_runtime(
        EvaluationPaths(tmp_path)
    )

    assert device == "cuda"
    assert model is fake_model
    assert fake_model.devices == ["cuda"]


def test_load_runtime_falls_back_to_cpu_without_cuda(tmp_path, monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    class FakeModel:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def to(self, device: str) -> FakeModel:
            self.devices.append(device)
            return self

        def eval(self) -> FakeModel:
            return self

    fake_model = FakeModel()

    class FakeProcessorClass:
        @staticmethod
        def from_pretrained(_checkpoint: str, *, cache_dir: str) -> object:
            return object()

    class FakeModelClass:
        @staticmethod
        def from_pretrained(_checkpoint: str, *, cache_dir: str) -> FakeModel:
            return fake_model

    class FakeTransformers:
        AutoProcessor = FakeProcessorClass
        AutoModelForZeroShotObjectDetection = FakeModelClass

    modules: dict[str, object] = {
        "torch": FakeTorch(),
        "transformers": FakeTransformers,
    }
    monkeypatch.setattr(
        annotation.importlib,
        "import_module",
        lambda name: modules[name],
    )

    _torch, _processor, model, device = annotation._load_runtime(
        EvaluationPaths(tmp_path)
    )

    assert device == "cpu"
    assert model is fake_model
    assert fake_model.devices == ["cpu"]


def test_build_annotation_moves_processed_inputs_to_selected_device() -> None:
    class FakeProcessed(dict[str, object]):
        def __init__(self) -> None:
            super().__init__(pixel_values=[[0.0]])
            self.devices: list[str] = []

        def to(self, device: str) -> FakeProcessed:
            self.devices.append(device)
            return self

    processed = FakeProcessed()

    class FakeProcessor:
        def __call__(self, **_kwargs: object) -> FakeProcessed:
            return processed

    class FakeModel:
        def __call__(self, **_kwargs: object) -> dict[str, object]:
            return {
                "logits": [[-12.0] * 6 for _ in range(MAX_DETECTIONS + 5)],
                "pred_boxes": [
                    [0.5, 0.5, 0.2, 0.2] for _ in range(MAX_DETECTIONS + 5)
                ],
            }

    sample = SampleManifest(
        sample_id="sample-device",
        source="inaturalist",
        source_id="observation-device",
        source_category="argiope_aurantia",
        expected_presence="positive",
        source_url="source-device",
        license="cc0",
        image_relative_path="sample-device.jpg",
        sha256="a" * 64,
        perceptual_hash="b" * 16,
        duplicate_group="group-device",
        split="calibration",
        width=64,
        height=48,
    )

    _ = _build_annotation(
        sample,
        FakeProcessor(),
        FakeModel(),
        image=None,
        device="cuda",
    )

    assert processed.devices == ["cuda"]

