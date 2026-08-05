"""Atomic converted-model acquisition tests using harmless synthetic bytes."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pytest

from model_tools import model_artifact
from model_tools.acquire import load_source_manifest
from model_tools.metadata import TimmSafetensorsModelSource


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_converted_artifact_verifies_before_promotion_and_cleans_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_payload = b"harmless synthetic weights"
    final_payload = b"harmless synthetic ONNX"
    manifest = load_source_manifest(
        Path(__file__).parents[1] / "model-sources/tinyvit-5m-224-dist.toml"
    )
    source = manifest.model
    assert isinstance(source, TimmSafetensorsModelSource)
    manifest = manifest.model_copy(
        update={
            "model": source.model_copy(
                update={
                    "sha256": _digest(source_payload),
                    "size_bytes": len(source_payload),
                    "artifact_sha256": _digest(final_payload),
                    "artifact_size_bytes": len(final_payload),
                }
            )
        }
    )
    download_count = 0

    @contextmanager
    def fake_open_url(_url: str):
        nonlocal download_count
        download_count += 1
        yield BytesIO(source_payload)

    exported_payload = b"wrong artifact"

    def fake_export(weights: Path, destination: Path, _manifest) -> None:
        assert weights.read_bytes() == source_payload
        destination.write_bytes(exported_payload)

    monkeypatch.setattr(model_artifact, "open_url", fake_open_url)
    monkeypatch.setattr(model_artifact, "export_timm_safetensors", fake_export)
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"existing artifact")
    validation_calls = 0

    def validate(path: Path) -> None:
        nonlocal validation_calls
        validation_calls += 1
        assert path != destination
        assert path.read_bytes() == final_payload

    with pytest.raises(RuntimeError, match="verification"):
        model_artifact.acquire_model_artifact(manifest, destination, validate)
    assert destination.read_bytes() == b"existing artifact"
    assert validation_calls == 0
    assert not (tmp_path / ".model.onnx.part").exists()

    exported_payload = final_payload

    def reject_after_validation(path: Path) -> None:
        validate(path)
        raise RuntimeError("graph rejected")

    with pytest.raises(RuntimeError, match="graph rejected"):
        model_artifact.acquire_model_artifact(
            manifest,
            destination,
            reject_after_validation,
        )
    assert destination.read_bytes() == b"existing artifact"
    assert not (tmp_path / ".model.onnx.part").exists()

    model_artifact.acquire_model_artifact(manifest, destination, validate)
    assert destination.read_bytes() == final_payload
    assert validation_calls == 2
    assert download_count == 1
