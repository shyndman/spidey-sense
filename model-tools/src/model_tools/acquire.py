"""Acquire and verify the exact model bundle consumed by the extension.

Acquisition is intentionally reproducible and single-copy: pinned remote inputs are
verified while streaming directly beside the extension's generated public assets.
A locally valid bundle is fully reusable offline, while partial or unverified data
is never promoted to a final artifact path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import tomllib
import urllib.request
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final, Protocol, cast

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto
from pydantic import ValidationError

from .metadata import (
    ArtifactMetadata,
    ClassGroups,
    InputMetadata,
    LabelRecord,
    ModelMetadata,
    OutputMetadata,
    SourceManifest,
)


class _LoadModel(Protocol):
    def __call__(
        self,
        path: str,
        *,
        load_external_data: bool,
    ) -> onnx.ModelProto: ...


class _CheckModel(Protocol):
    def __call__(self, model: onnx.ModelProto, *, full_check: bool) -> None: ...


load_model = cast(_LoadModel, onnx.load)
check_model = cast(_CheckModel, onnx.checker.check_model)


class _InferenceSession(Protocol):
    def run(
        self,
        output_names: Sequence[str],
        input_feed: dict[str, NDArray[np.float32]],
    ) -> Sequence[object]: ...


class _OnnxRuntime(Protocol):
    def InferenceSession(
        self,
        path: str,
        *,
        providers: Sequence[str],
    ) -> _InferenceSession: ...


ort = cast(_OnnxRuntime, importlib.import_module("onnxruntime"))

DOWNLOAD_CHUNK_BYTES: Final = 1_048_576
DOWNLOAD_TIMEOUT_SECONDS: Final = 30
METADATA_SUFFIX: Final = ".metadata.json"
PART_SUFFIX: Final = ".part"
SOFTMAX_SUM_TOLERANCE: Final = 1e-5


@dataclass(frozen=True, slots=True)
class BundlePaths:
    """Final paths for one generated extension model bundle."""

    model: Path
    metadata: Path


class AcquisitionError(RuntimeError):
    """A pinned source or generated artifact violated its contract."""


def load_source_manifest(path: Path) -> SourceManifest:
    """Load and strictly validate the checked-in TOML source manifest."""

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return SourceManifest.model_validate(data)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise AcquisitionError(f"invalid source manifest {path}: {error}") from error


def acquire_bundle(manifest_path: Path, output_dir: Path) -> BundlePaths:
    """Create or reuse a verified bundle, downloading only invalid components."""

    manifest = load_source_manifest(manifest_path)
    paths = _bundle_paths(manifest, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _file_matches(
        paths.model,
        expected_size=manifest.model.size_bytes,
        expected_sha256=manifest.model.sha256,
    ):
        try:
            verify_bundle(manifest_path, output_dir, run_inference=True)
        except AcquisitionError:
            pass
        else:
            return paths
    else:
        _download_model(manifest, paths.model)

    _verify_graph(paths.model, manifest)
    _run_inference(paths.model, manifest)
    labels = _download_labels(manifest)
    metadata = _build_metadata(manifest, labels)
    _write_metadata(metadata, paths.metadata)
    verify_bundle(manifest_path, output_dir, run_inference=True)
    return paths


def verify_bundle(
    manifest_path: Path,
    bundle_dir: Path,
    *,
    run_inference: bool,
) -> ArtifactMetadata:
    """Verify a generated bundle entirely offline and return its typed metadata."""

    manifest = load_source_manifest(manifest_path)
    paths = _bundle_paths(manifest, bundle_dir)
    _require_file(
        paths.model,
        expected_size=manifest.model.size_bytes,
        expected_sha256=manifest.model.sha256,
    )
    _verify_graph(paths.model, manifest)
    metadata = _load_metadata(paths.metadata)
    expected = _build_metadata(manifest, metadata.output.labels)
    if metadata != expected:
        raise AcquisitionError("metadata does not match the pinned source contract")
    if paths.metadata.read_bytes() != _serialize_metadata(metadata):
        raise AcquisitionError("metadata is not canonically serialized")
    if run_inference:
        _run_inference(paths.model, manifest)
    return metadata


def _bundle_paths(manifest: SourceManifest, output_dir: Path) -> BundlePaths:
    model_path = output_dir / manifest.model.filename
    metadata_path = output_dir / f"{manifest.model.id}{METADATA_SUFFIX}"
    return BundlePaths(model=model_path, metadata=metadata_path)


def _file_matches(path: Path, *, expected_size: int, expected_sha256: str) -> bool:
    try:
        return path.stat().st_size == expected_size and _sha256(path) == expected_sha256
    except OSError:
        return False


def _require_file(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    if not _file_matches(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    ):
        raise AcquisitionError(f"artifact failed size or SHA-256 verification: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _open_url(url: str) -> Generator[BinaryIO]:
    try:
        with urllib.request.urlopen(
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            yield response
    except OSError as error:
        raise AcquisitionError(f"download failed for {url}: {error}") from error


def _download_model(manifest: SourceManifest, destination: Path) -> None:
    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            _open_url(str(manifest.model.url)) as response,
            partial.open("wb") as target,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if (
            size != manifest.model.size_bytes
            or digest.hexdigest() != manifest.model.sha256
        ):
            raise AcquisitionError(
                "downloaded model failed size or SHA-256 verification"
            )
        _verify_graph(partial, manifest)
        _run_inference(partial, manifest)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _download_labels(manifest: SourceManifest) -> tuple[LabelRecord, ...]:
    with _open_url(str(manifest.labels.url)) as response:
        payload = response.read(manifest.labels.size_bytes + 1)
    if len(payload) != manifest.labels.size_bytes:
        raise AcquisitionError("downloaded labels failed size verification")
    if hashlib.sha256(payload).hexdigest() != manifest.labels.sha256:
        raise AcquisitionError("downloaded labels failed SHA-256 verification")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AcquisitionError("downloaded labels are not UTF-8") from error
    return _parse_labels(text, expected_count=manifest.labels.count)


def _parse_labels(text: str, *, expected_count: int) -> tuple[LabelRecord, ...]:
    lines = text.splitlines()
    if len(lines) != expected_count or any(not line for line in lines):
        raise AcquisitionError(f"expected exactly {expected_count} non-empty labels")
    records: list[LabelRecord] = []
    seen_synsets: set[str] = set()
    for index, line in enumerate(lines):
        try:
            synset, label = line.split(" ", maxsplit=1)
        except ValueError as error:
            raise AcquisitionError(f"malformed label at index {index}") from error
        if not synset or not label or synset in seen_synsets:
            raise AcquisitionError(f"invalid or duplicate label at index {index}")
        seen_synsets.add(synset)
        records.append(LabelRecord(index=index, synset=synset, label=label))
    return tuple(records)


def _verify_graph(path: Path, manifest: SourceManifest) -> None:
    try:
        model = load_model(str(path), load_external_data=False)
        check_model(model, full_check=True)
    except (OSError, ValueError, onnx.checker.ValidationError) as error:
        raise AcquisitionError(f"ONNX graph validation failed: {error}") from error

    default_opsets = [
        item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
    ]
    if default_opsets != [manifest.model.opset]:
        raise AcquisitionError(f"unexpected ONNX opsets: {default_opsets}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise AcquisitionError("model must expose exactly one graph input and output")

    graph_input = model.graph.input[0]
    _verify_tensor(
        graph_input,
        name=manifest.graph.input.name,
        dimensions=(
            manifest.graph.input.batch_dimension,
            manifest.graph.input.channels,
            manifest.graph.input.height,
            manifest.graph.input.width,
        ),
    )
    graph_output = model.graph.output[0]
    _verify_tensor(
        graph_output,
        name=manifest.graph.output.name,
        dimensions=(
            manifest.graph.output.batch_dimension,
            manifest.graph.output.classes,
        ),
    )


def _verify_tensor(
    value: onnx.ValueInfoProto,
    *,
    name: str,
    dimensions: tuple[str | int, ...],
) -> None:
    tensor_type = value.type.tensor_type
    actual_dimensions: tuple[str | int, ...] = tuple(
        dimension.dim_param if dimension.HasField("dim_param") else dimension.dim_value
        for dimension in tensor_type.shape.dim
    )
    if (
        value.name != name
        or tensor_type.elem_type != TensorProto.FLOAT
        or actual_dimensions != dimensions
    ):
        raise AcquisitionError(
            f"unexpected tensor contract for {value.name}: "
            f"type={tensor_type.elem_type}, shape={actual_dimensions}"
        )


def _run_inference(path: Path, manifest: SourceManifest) -> None:
    try:
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        tensor: NDArray[np.float32] = np.zeros(
            (
                1,
                manifest.graph.input.channels,
                manifest.graph.input.height,
                manifest.graph.input.width,
            ),
            dtype=np.float32,
        )
        raw_result = session.run(
            [manifest.graph.output.name],
            {manifest.graph.input.name: tensor},
        )[0]
    except (OSError, RuntimeError, ValueError) as error:
        raise AcquisitionError(f"ONNX Runtime inference failed: {error}") from error

    if not isinstance(raw_result, np.ndarray):
        raise AcquisitionError("inference output is not a dense tensor")
    result = cast(NDArray[np.float32], raw_result)
    expected_shape = (1, manifest.graph.output.classes)
    if result.shape != expected_shape or not bool(np.all(np.isfinite(result))):
        raise AcquisitionError(
            f"inference returned invalid output: shape={result.shape}"
        )
    shifted = result - np.max(result, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    if not bool(
        np.allclose(
            np.sum(probabilities, axis=1),
            np.ones(1),
            atol=SOFTMAX_SUM_TOLERANCE,
            rtol=0,
        )
    ):
        raise AcquisitionError("inference softmax probabilities do not sum to one")


def _build_metadata(
    manifest: SourceManifest,
    labels: tuple[LabelRecord, ...],
) -> ArtifactMetadata:
    if len(labels) != manifest.labels.count:
        raise AcquisitionError(f"expected exactly {manifest.labels.count} labels")
    if any(record.index != index for index, record in enumerate(labels)):
        raise AcquisitionError("label indices must be contiguous and zero-based")

    by_synset = {record.synset: record for record in labels}
    if len(by_synset) != len(labels):
        raise AcquisitionError("label synsets must be unique")
    blocked = _resolve_group(manifest.classes.blocked_synsets, by_synset, "blocked")
    debug = _resolve_group(manifest.classes.debug_synsets, by_synset, "debug")
    if set(manifest.classes.blocked_synsets) & set(manifest.classes.debug_synsets):
        raise AcquisitionError("blocked and debug class groups overlap")

    return ArtifactMetadata(
        schema_version=1,
        model=_build_model_metadata(manifest),
        input=_build_input_metadata(manifest),
        output=OutputMetadata(
            name=manifest.graph.output.name,
            data_type=manifest.graph.output.data_type,
            shape=(None, manifest.graph.output.classes),
            activation=manifest.postprocessing.activation,
            labels=labels,
        ),
        classes=ClassGroups(blocked=blocked, debug=debug),
    )


def _build_model_metadata(manifest: SourceManifest) -> ModelMetadata:
    source = manifest.model
    return ModelMetadata(
        id=source.id,
        filename=source.filename,
        sha256=source.sha256,
        size_bytes=source.size_bytes,
        format=source.format,
        opset=source.opset,
        source_url=source.url,
        source_revision=source.revision,
    )


def _build_input_metadata(manifest: SourceManifest) -> InputMetadata:
    source = manifest.graph.input
    preprocessing = manifest.preprocessing
    return InputMetadata(
        name=source.name,
        data_type=source.data_type,
        layout=preprocessing.layout,
        shape=(None, source.channels, source.height, source.width),
        color_space=preprocessing.color_space,
        resize_mode=preprocessing.resize_mode,
        resize_shortest_side=preprocessing.resize_shortest_side,
        interpolation=preprocessing.interpolation,
        crop_mode=preprocessing.crop_mode,
        crop_width=preprocessing.crop_width,
        crop_height=preprocessing.crop_height,
        pixel_scale=preprocessing.pixel_scale,
        mean=preprocessing.mean,
        standard_deviation=preprocessing.standard_deviation,
    )


def _resolve_group(
    synsets: tuple[str, ...],
    labels: dict[str, LabelRecord],
    group: str,
) -> tuple[LabelRecord, ...]:
    try:
        return tuple(labels[synset] for synset in synsets)
    except KeyError as error:
        raise AcquisitionError(f"missing {group} synset: {error.args[0]}") from error


def _serialize_metadata(metadata: ArtifactMetadata) -> bytes:
    payload = metadata.model_dump(mode="json", by_alias=True)
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()


def _write_metadata(metadata: ArtifactMetadata, destination: Path) -> None:
    partial = destination.with_name(f".{destination.name}{PART_SUFFIX}")
    try:
        partial.write_bytes(_serialize_metadata(metadata))
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _load_metadata(path: Path) -> ArtifactMetadata:
    try:
        return ArtifactMetadata.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise AcquisitionError(f"invalid artifact metadata {path}: {error}") from error
