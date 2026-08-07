"""Behavioral tests for reproducible model acquisition using harmless model data."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast, override

import pytest
from model_tools.acquire import (
    AcquisitionError,
    acquire_bundle,
    build_metadata,
    check_model,
    load_source_manifest,
    parse_labels,
    verify_bundle,
)
from model_tools.metadata import TimmSafetensorsModelSource
from onnx import TensorProto, helper

_BLOCKED_SYNSET: Final = "n00000001"
_DEBUG_SYNSET: Final = "n00000002"
_CLASS_COUNT: Final = 12


class _PayloadServer(ThreadingHTTPServer):
    payloads: dict[str, bytes] = {}


class _PayloadHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = cast(_PayloadServer, self.server)
        payload = server.payloads.get(self.path)
        if payload is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        _ = self.wfile.write(payload)

    @override
    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def _serve(payloads: dict[str, bytes]) -> Generator[str]:
    server = _PayloadServer(("127.0.0.1", 0), _PayloadHandler)
    server.payloads = payloads
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = cast(tuple[str, int], server.server_address)
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _model_bytes() -> bytes:
    graph_input = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        ["batch_size", 3, 2, 2],
    )
    graph_output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        ["batch_size", _CLASS_COUNT],
    )
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["input"], ["output"], axis=1)],
        "synthetic-acquisition-model",
        [graph_input],
        [graph_output],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    model.ir_version = 8
    check_model(model, full_check=True)
    return model.SerializeToString()


def _labels_bytes() -> bytes:
    return "".join(
        f"n{index:08d} harmless class {index}\n" for index in range(_CLASS_COUNT)
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(
    path: Path,
    *,
    model_url: str,
    labels_url: str,
    model_payload: bytes,
    labels_payload: bytes,
    model_sha256: str | None = None,
    model_size_bytes: int | None = None,
    blocked_synset: str = _BLOCKED_SYNSET,
    input_height: int = 2,
) -> None:
    _ = path.write_text(
        f'''schema_version = 3

[model]
id = "synthetic-model"
filename = "synthetic-model.onnx"
url = "{model_url}"
revision = "test-model-revision"
sha256 = "{model_sha256 or _sha256(model_payload)}"
size_bytes = {model_size_bytes if model_size_bytes is not None else len(model_payload)}
license = "apache-2.0"
format = "onnx"
opset = 12

[labels]
url = "{labels_url}"
revision = "test-labels-revision"
sha256 = "{_sha256(labels_payload)}"
size_bytes = {len(labels_payload)}
count = {_CLASS_COUNT}

[graph.input]
name = "input"
data_type = "float32"
batch_dimension = "batch_size"
channels = 3
height = {input_height}
width = 2

[graph.output]
name = "output"
data_type = "float32"
batch_dimension = "batch_size"
classes = {_CLASS_COUNT}

[preprocessing]
color_space = "RGB"
layout = "NCHW"
resize_mode = "contain"
allow_upscale = true
interpolation = "bilinear"
padding_mode = "black"
pixel_scale = 0.0039215686274509803
mean = [0.485, 0.456, 0.406]
standard_deviation = [0.229, 0.224, 0.225]

[postprocessing]
activation = "softmax"

[classes]
blocked_synsets = ["{blocked_synset}"]
debug_synsets = ["{_DEBUG_SYNSET}"]
''',
        encoding="utf-8",
    )


def test_checked_in_source_manifest_is_strictly_valid() -> None:
    manifest_path = Path(__file__).parents[1] / "model-sources/mobilenetv2-12.toml"

    manifest = load_source_manifest(manifest_path)

    assert manifest.model.id == "mobilenetv2-12"
    assert manifest.graph.input.name == "input"
    assert manifest.preprocessing.mean == (0.485, 0.456, 0.406)
    assert manifest.classes.blocked_synsets == (
        "n01773157",
        "n01773549",
        "n01773797",
        "n01774384",
        "n01774750",
        "n01775062",
    )


def test_checked_in_tinyvit_manifest_pins_conversion_contract() -> None:
    manifest_path = Path(__file__).parents[1] / "model-sources/tinyvit-5m-224-dist.toml"

    manifest = load_source_manifest(manifest_path)

    assert isinstance(manifest.model, TimmSafetensorsModelSource)
    assert manifest.model.id == "tinyvit-5m-224-dist"
    assert manifest.model.revision == "b46989c2a2c95e7612b919a6f5e2cd7dcf4f1271"
    assert manifest.model.exporter_version == "1.0.28"
    assert manifest.model.artifact_sha256 == (
        "ab65a070d1540d6beb9543193d470ac0f2b5f3063b411783856d0c4c45a1e44f"
    )
    assert manifest.graph.input.height == 224
    assert manifest.graph.output.classes == 1000


def test_checked_in_mobilenetv4_manifest_pins_conversion_contract() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "model-sources/mobilenetv4-conv-medium-224.toml"
    )

    manifest = load_source_manifest(manifest_path)

    assert isinstance(manifest.model, TimmSafetensorsModelSource)
    assert manifest.model.id == "mobilenetv4-conv-medium-224"
    assert manifest.model.license == "apache-2.0"
    assert manifest.model.format == "timm-safetensors"
    assert (
        manifest.model.architecture
        == "mobilenetv4_conv_medium.e500_r224_in1k"
    )
    assert manifest.model.revision == "02a09fbfb82b289e871ba8255f9da58c056fb13b"
    assert manifest.model.sha256 == (
        "10f9f8057d574f71d22c52d31cadab8af0bc325841fd93939c86d45d20998cad"
    )
    assert manifest.model.size_bytes == 39179184
    assert manifest.model.exporter_version == "1.0.28"
    assert manifest.model.opset == 17
    assert manifest.model.artifact_sha256 == (
        "ff9c0617abcd88fc364afb8bc5784c0fc3ddb90fb4ecb165d00146ed8235a945"
    )
    assert manifest.model.artifact_size_bytes == 38763301
    assert manifest.graph.input.data_type == "float32"
    assert manifest.graph.input.batch_dimension == "batch_size"
    assert (
        manifest.graph.input.channels,
        manifest.graph.input.height,
        manifest.graph.input.width,
    ) == (3, 224, 224)
    assert manifest.graph.output.data_type == "float32"
    assert manifest.graph.output.batch_dimension == "batch_size"
    assert manifest.graph.output.classes == 1000
    assert manifest.preprocessing.color_space == "RGB"
    assert manifest.preprocessing.layout == "NCHW"
    assert manifest.preprocessing.resize_mode == "contain"
    assert manifest.preprocessing.allow_upscale is True
    assert manifest.preprocessing.interpolation == "bilinear"
    assert manifest.preprocessing.padding_mode == "black"
    assert manifest.preprocessing.mean == (0.485, 0.456, 0.406)
    assert manifest.preprocessing.standard_deviation == (0.229, 0.224, 0.225)
    assert manifest.postprocessing.activation == "softmax"
    assert manifest.classes.blocked_synsets == (
        "n01773157",
        "n01773549",
        "n01773797",
        "n01774384",
        "n01774750",
        "n01775062",
    )
    assert manifest.classes.debug_synsets == ("n07753592",)


def test_invalid_source_manifest_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid.toml"
    _ = manifest_path.write_text("schema_version = 3\n", encoding="utf-8")

    with pytest.raises(AcquisitionError, match="invalid source manifest"):
        _ = load_source_manifest(manifest_path)


def test_acquire_downloads_then_reuses_bundle_offline(tmp_path: Path) -> None:
    model_payload = _model_bytes()
    labels_payload = _labels_bytes()
    manifest_path = tmp_path / "model-source.toml"
    bundle_dir = tmp_path / "bundle"

    with _serve({"/model": model_payload, "/labels": labels_payload}) as base_url:
        _write_manifest(
            manifest_path,
            model_url=f"{base_url}/model",
            labels_url=f"{base_url}/labels",
            model_payload=model_payload,
            labels_payload=labels_payload,
        )
        paths = acquire_bundle(manifest_path, bundle_dir)

    assert paths.model.read_bytes() == model_payload
    assert paths.metadata.is_file()
    assert list(bundle_dir.glob("*.part")) == []

    metadata = verify_bundle(manifest_path, bundle_dir, run_inference=True)
    assert metadata.classes.blocked[0].synset == _BLOCKED_SYNSET
    assert metadata.classes.debug[0].synset == _DEBUG_SYNSET
    first_metadata = paths.metadata.read_bytes()

    reused_paths = acquire_bundle(manifest_path, bundle_dir)

    assert reused_paths == paths
    assert paths.metadata.read_bytes() == first_metadata


def test_checksum_failure_removes_partial_model(tmp_path: Path) -> None:
    model_payload = _model_bytes()
    labels_payload = _labels_bytes()
    manifest_path = tmp_path / "model-source.toml"
    bundle_dir = tmp_path / "bundle"

    with _serve({"/model": model_payload, "/labels": labels_payload}) as base_url:
        _write_manifest(
            manifest_path,
            model_url=f"{base_url}/model",
            labels_url=f"{base_url}/labels",
            model_payload=model_payload,
            labels_payload=labels_payload,
            model_sha256="0" * 64,
        )
        with pytest.raises(AcquisitionError, match="SHA-256"):
            _ = acquire_bundle(manifest_path, bundle_dir)

    assert not (bundle_dir / "synthetic-model.onnx").exists()
    assert list(bundle_dir.glob("*.part")) == []


def test_size_failure_removes_partial_model(tmp_path: Path) -> None:
    model_payload = _model_bytes()
    labels_payload = _labels_bytes()
    manifest_path = tmp_path / "model-source.toml"
    bundle_dir = tmp_path / "bundle"

    with _serve({"/model": model_payload, "/labels": labels_payload}) as base_url:
        _write_manifest(
            manifest_path,
            model_url=f"{base_url}/model",
            labels_url=f"{base_url}/labels",
            model_payload=model_payload,
            labels_payload=labels_payload,
            model_size_bytes=len(model_payload) + 1,
        )
        with pytest.raises(AcquisitionError, match="size"):
            _ = acquire_bundle(manifest_path, bundle_dir)

    assert not (bundle_dir / "synthetic-model.onnx").exists()
    assert list(bundle_dir.glob("*.part")) == []


def test_graph_contract_failure_does_not_promote_model(tmp_path: Path) -> None:
    model_payload = _model_bytes()
    labels_payload = _labels_bytes()
    manifest_path = tmp_path / "model-source.toml"
    bundle_dir = tmp_path / "bundle"

    with _serve({"/model": model_payload, "/labels": labels_payload}) as base_url:
        _write_manifest(
            manifest_path,
            model_url=f"{base_url}/model",
            labels_url=f"{base_url}/labels",
            model_payload=model_payload,
            labels_payload=labels_payload,
            input_height=3,
        )
        with pytest.raises(AcquisitionError, match="unexpected tensor contract"):
            _ = acquire_bundle(manifest_path, bundle_dir)

    assert not (bundle_dir / "synthetic-model.onnx").exists()
    assert list(bundle_dir.glob("*.part")) == []


def test_label_parser_rejects_duplicate_synsets() -> None:
    with pytest.raises(AcquisitionError, match="duplicate label"):
        _ = parse_labels(
            "n00000001 first\nn00000001 second\n",
            expected_count=2,
        )


def test_metadata_rejects_missing_group_synset(tmp_path: Path) -> None:
    model_payload = _model_bytes()
    labels_payload = _labels_bytes()
    manifest_path = tmp_path / "model-source.toml"
    _write_manifest(
        manifest_path,
        model_url="https://example.invalid/model",
        labels_url="https://example.invalid/labels",
        model_payload=model_payload,
        labels_payload=labels_payload,
        blocked_synset="n99999999",
    )
    manifest = load_source_manifest(manifest_path)
    labels = parse_labels(
        labels_payload.decode(),
        expected_count=_CLASS_COUNT,
    )

    with pytest.raises(AcquisitionError, match="missing blocked synset"):
        _ = build_metadata(manifest, labels)



def test_checked_in_mobilenetv4_conv_large_manifest_pins_conversion_contract() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "model-sources/mobilenetv4-conv-large-256.toml"
    )

    manifest = load_source_manifest(manifest_path)

    assert isinstance(manifest.model, TimmSafetensorsModelSource)
    assert manifest.model.id == "mobilenetv4-conv-large-256"
    assert manifest.model.filename == "mobilenetv4-conv-large-256.onnx"
    assert manifest.model.license == "apache-2.0"
    assert manifest.model.format == "timm-safetensors"
    assert (
        manifest.model.architecture
        == "mobilenetv4_conv_large.e500_r256_in1k"
    )
    assert manifest.model.revision == "033de0bd74b5261f7339c5de242a2e4785808aa3"
    assert manifest.model.sha256 == (
        "9f86412962cd3360dab5759cbbc64495fefda68f7bae44b6b3142ac5853cb266"
    )
    assert manifest.model.size_bytes == 131022456
    assert manifest.model.exporter_version == "1.0.28"
    assert manifest.model.opset == 17
    assert manifest.model.artifact_sha256 == (
        "78741de009a5b405a85b12c8cdefe2f1c5a4a8516543eabd52c828975bfa850c"
    )
    assert manifest.model.artifact_size_bytes == 130113418
    assert manifest.graph.input.name == "input"
    assert manifest.graph.input.data_type == "float32"
    assert manifest.graph.input.batch_dimension == "batch_size"
    assert (
        manifest.graph.input.channels,
        manifest.graph.input.height,
        manifest.graph.input.width,
    ) == (3, 256, 256)
    assert manifest.graph.output.name == "output"
    assert manifest.graph.output.data_type == "float32"
    assert manifest.graph.output.batch_dimension == "batch_size"
    assert manifest.graph.output.classes == 1000
    assert manifest.preprocessing.color_space == "RGB"
    assert manifest.preprocessing.layout == "NCHW"
    assert manifest.preprocessing.resize_mode == "contain"
    assert manifest.preprocessing.allow_upscale is True
    assert manifest.preprocessing.interpolation == "bilinear"
    assert manifest.preprocessing.padding_mode == "black"
    assert manifest.preprocessing.pixel_scale == 0.0039215686274509803
    assert manifest.preprocessing.mean == (0.485, 0.456, 0.406)
    assert manifest.preprocessing.standard_deviation == (0.229, 0.224, 0.225)
    assert manifest.postprocessing.activation == "softmax"
    assert manifest.labels.revision == "4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177"
    assert manifest.labels.sha256 == (
        "acf75ef0abe89694b19056e0796401068b459c457baa30335f240c7692857355"
    )
    assert manifest.labels.size_bytes == 31675
    assert manifest.labels.count == 1000
    assert manifest.classes.blocked_synsets == (
        "n01773157",
        "n01773549",
        "n01773797",
        "n01774384",
        "n01774750",
        "n01775062",
    )
    assert manifest.classes.debug_synsets == ("n07753592",)


def test_checked_in_mobilenetv4_conv_small_manifest_pins_conversion_contract() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "model-sources/mobilenetv4-conv-small-224.toml"
    )

    manifest = load_source_manifest(manifest_path)

    assert isinstance(manifest.model, TimmSafetensorsModelSource)
    assert manifest.model.id == "mobilenetv4-conv-small-224"
    assert manifest.model.filename == "mobilenetv4-conv-small-224.onnx"
    assert manifest.model.license == "apache-2.0"
    assert manifest.model.format == "timm-safetensors"
    assert (
        manifest.model.architecture
        == "mobilenetv4_conv_small.e2400_r224_in1k"
    )
    assert manifest.model.revision == "331fb803779522b685cf942e15f914fb6741c1eb"
    assert manifest.model.sha256 == (
        "7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4"
    )
    assert manifest.model.size_bytes == 15223016
    assert manifest.model.exporter_version == "1.0.28"
    assert manifest.model.opset == 17
    assert manifest.model.artifact_sha256 == (
        "afa4b3e2e717a50607beb67a4822c8347bac172ec898ee4e88e6bcc58462f9bb"
    )
    assert manifest.model.artifact_size_bytes == 15069040
    assert manifest.graph.input.name == "input"
    assert manifest.graph.input.data_type == "float32"
    assert manifest.graph.input.batch_dimension == "batch_size"
    assert (
        manifest.graph.input.channels,
        manifest.graph.input.height,
        manifest.graph.input.width,
    ) == (3, 224, 224)
    assert manifest.graph.output.name == "output"
    assert manifest.graph.output.data_type == "float32"
    assert manifest.graph.output.batch_dimension == "batch_size"
    assert manifest.graph.output.classes == 1000
    assert manifest.preprocessing.color_space == "RGB"
    assert manifest.preprocessing.layout == "NCHW"
    assert manifest.preprocessing.resize_mode == "contain"
    assert manifest.preprocessing.allow_upscale is True
    assert manifest.preprocessing.interpolation == "bilinear"
    assert manifest.preprocessing.padding_mode == "black"
    assert manifest.preprocessing.pixel_scale == 0.0039215686274509803
    assert manifest.preprocessing.mean == (0.485, 0.456, 0.406)
    assert manifest.preprocessing.standard_deviation == (0.229, 0.224, 0.225)
    assert manifest.postprocessing.activation == "softmax"
    assert manifest.labels.revision == "4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177"
    assert manifest.labels.sha256 == (
        "acf75ef0abe89694b19056e0796401068b459c457baa30335f240c7692857355"
    )
    assert manifest.labels.size_bytes == 31675
    assert manifest.labels.count == 1000
    assert manifest.classes.blocked_synsets == (
        "n01773157",
        "n01773549",
        "n01773797",
        "n01774384",
        "n01774750",
        "n01775062",
    )
    assert manifest.classes.debug_synsets == ("n07753592",)


def test_checked_in_mobilenetv3_small_manifest_pins_conversion_contract() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "model-sources/mobilenetv3-small-100-224.toml"
    )

    manifest = load_source_manifest(manifest_path)

    assert isinstance(manifest.model, TimmSafetensorsModelSource)
    assert manifest.model.id == "mobilenetv3-small-100-224"
    assert manifest.model.filename == "mobilenetv3-small-100-224.onnx"
    assert manifest.model.license == "apache-2.0"
    assert manifest.model.format == "timm-safetensors"
    assert manifest.model.architecture == "tf_mobilenetv3_small_100.in1k"
    assert manifest.model.revision == "6ea7c1da494cbf1ef6682a942221a9fdc3a78126"
    assert manifest.model.sha256 == (
        "7567f2534992e11894549df6550a4ad251bd1d6997b2f24651fdeef87f45ca90"
    )
    assert manifest.model.size_bytes == 10241912
    assert manifest.model.exporter_version == "1.0.28"
    assert manifest.model.opset == 17
    assert manifest.model.artifact_sha256 == (
        "b533cd20a4246f0154c5bf3edce6085c27deab42175793f9f7030d691c0f11db"
    )
    assert manifest.model.artifact_size_bytes == 10249621
    assert manifest.graph.input.name == "input"
    assert manifest.graph.input.data_type == "float32"
    assert manifest.graph.input.batch_dimension == "batch_size"
    assert (
        manifest.graph.input.channels,
        manifest.graph.input.height,
        manifest.graph.input.width,
    ) == (3, 224, 224)
    assert manifest.graph.output.name == "output"
    assert manifest.graph.output.data_type == "float32"
    assert manifest.graph.output.batch_dimension == "batch_size"
    assert manifest.graph.output.classes == 1000
    assert manifest.preprocessing.color_space == "RGB"
    assert manifest.preprocessing.layout == "NCHW"
    assert manifest.preprocessing.resize_mode == "contain"
    assert manifest.preprocessing.allow_upscale is True
    assert manifest.preprocessing.interpolation == "bilinear"
    assert manifest.preprocessing.padding_mode == "black"
    assert manifest.preprocessing.pixel_scale == 0.0039215686274509803
    assert manifest.preprocessing.mean == (0.5, 0.5, 0.5)
    assert manifest.preprocessing.standard_deviation == (0.5, 0.5, 0.5)
    assert manifest.postprocessing.activation == "softmax"
    assert manifest.labels.revision == "4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177"
    assert manifest.labels.sha256 == (
        "acf75ef0abe89694b19056e0796401068b459c457baa30335f240c7692857355"
    )
    assert manifest.labels.size_bytes == 31675
    assert manifest.labels.count == 1000
    assert manifest.classes.blocked_synsets == (
        "n01773157",
        "n01773549",
        "n01773797",
        "n01774384",
        "n01774750",
        "n01775062",
    )
    assert manifest.classes.debug_synsets == ("n07753592",)


def test_checked_in_mobilenetv3_large_manifest_pins_conversion_contract() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "model-sources/mobilenetv3-large-100-224.toml"
    )

    manifest = load_source_manifest(manifest_path)

    assert isinstance(manifest.model, TimmSafetensorsModelSource)
    assert manifest.model.id == "mobilenetv3-large-100-224"
    assert manifest.model.filename == "mobilenetv3-large-100-224.onnx"
    assert manifest.model.license == "apache-2.0"
    assert manifest.model.format == "timm-safetensors"
    assert (
        manifest.model.architecture
        == "mobilenetv3_large_100.ra4_e3600_r224_in1k"
    )
    assert manifest.model.revision == "26a545b7c557e54fa13caff7e76da391f6bcbc1a"
    assert manifest.model.sha256 == (
        "bd595a2d59301fc6264c4aabb203f93ec17b0aa7bdd7532c8d9e1cf07dfc9bdc"
    )
    assert manifest.model.size_bytes == 22058008
    assert manifest.model.exporter_version == "1.0.28"
    assert manifest.model.opset == 17
    assert manifest.model.artifact_sha256 == (
        "aea18dca394e39dcc73b408d5493e3e980d38db351b7d97061bc97967a7294ed"
    )
    assert manifest.model.artifact_size_bytes == 21919309
    assert manifest.graph.input.name == "input"
    assert manifest.graph.input.data_type == "float32"
    assert manifest.graph.input.batch_dimension == "batch_size"
    assert (
        manifest.graph.input.channels,
        manifest.graph.input.height,
        manifest.graph.input.width,
    ) == (3, 224, 224)
    assert manifest.graph.output.name == "output"
    assert manifest.graph.output.data_type == "float32"
    assert manifest.graph.output.batch_dimension == "batch_size"
    assert manifest.graph.output.classes == 1000
    assert manifest.preprocessing.color_space == "RGB"
    assert manifest.preprocessing.layout == "NCHW"
    assert manifest.preprocessing.resize_mode == "contain"
    assert manifest.preprocessing.allow_upscale is True
    assert manifest.preprocessing.interpolation == "bilinear"
    assert manifest.preprocessing.padding_mode == "black"
    assert manifest.preprocessing.pixel_scale == 0.0039215686274509803
    assert manifest.preprocessing.mean == (0.5, 0.5, 0.5)
    assert manifest.preprocessing.standard_deviation == (0.5, 0.5, 0.5)
    assert manifest.postprocessing.activation == "softmax"
    assert manifest.labels.revision == "4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177"
    assert manifest.labels.sha256 == (
        "acf75ef0abe89694b19056e0796401068b459c457baa30335f240c7692857355"
    )
    assert manifest.labels.size_bytes == 31675
    assert manifest.labels.count == 1000
    assert manifest.classes.blocked_synsets == (
        "n01773157",
        "n01773549",
        "n01773797",
        "n01774384",
        "n01774750",
        "n01775062",
    )
    assert manifest.classes.debug_synsets == ("n07753592",)