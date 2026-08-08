"""Acquisition-owned persisted model tests."""

import pytest
from model_tools.evaluation.acquisition.models import SampleManifest
from pydantic import BaseModel, ValidationError


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


def test_manifest_is_strict_and_round_trips() -> None:
    value = _manifest()
    assert isinstance(value, BaseModel)
    assert SampleManifest.model_validate_json(value.model_dump_json()) == value
    with pytest.raises(ValidationError):
        _ = SampleManifest.model_validate({**value.model_dump(), "unexpected": 1})
    with pytest.raises(ValidationError):
        _ = SampleManifest.model_validate({**value.model_dump(), "width": True})
