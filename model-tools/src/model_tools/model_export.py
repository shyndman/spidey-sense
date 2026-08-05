"""Deterministically convert pinned timm weights into browser-ready ONNX."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Protocol, cast

import torch
from torch import Tensor
from torch.nn import Module

from .metadata import SourceManifest, TimmSafetensorsModelSource


class _SafeTensorsTorch(Protocol):
    def load_file(self, filename: str, device: str) -> dict[str, Tensor]: ...


class _OnnxExport(Protocol):
    def __call__(
        self,
        model: Module,
        args: tuple[Tensor, ...],
        f: str,
        *,
        input_names: Sequence[str],
        output_names: Sequence[str],
        dynamic_axes: Mapping[str, Mapping[int, str]],
        opset_version: int,
        dynamo: Literal[False],
    ) -> object: ...


def _load_model(
    weights_path: Path,
    source: TimmSafetensorsModelSource,
) -> Module:
    installed_version = version("timm")
    if installed_version != source.exporter_version:
        raise RuntimeError(
            f"timm exporter version {installed_version} does not match "
            f"pinned version {source.exporter_version}"
        )

    # HACK: Import timm only when conversion is requested. Its eager torchvision
    # registration can fail in CPU development environments whose optional
    # torchvision operators do not match Torch; acquisition of existing ONNX
    # models must remain usable there. Remove this after timm no longer imports
    # torchvision at package import time.
    from timm import create_model

    manual_seed = cast(Callable[[int], object], torch.manual_seed)
    _ = manual_seed(0)
    model = create_model(source.architecture, pretrained=False)
    safe_tensors = cast(
        _SafeTensorsTorch,
        importlib.import_module("safetensors.torch"),
    )
    state = safe_tensors.load_file(str(weights_path), device="cpu")
    _ = model.load_state_dict(state, strict=True)
    _ = model.eval()
    return model


def _export_model(
    model: Module,
    destination: Path,
    manifest: SourceManifest,
    source: TimmSafetensorsModelSource,
) -> None:
    sample = torch.zeros(
        (
            1,
            manifest.graph.input.channels,
            manifest.graph.input.height,
            manifest.graph.input.width,
        ),
        dtype=torch.float32,
    )
    # HACK: PyTorch's newer dynamo exporter changes graph serialization across
    # releases and currently needs extra exporter packages. The legacy exporter
    # produces a byte-identical TinyViT graph under the fully locked toolchain,
    # which lets acquisition verify the generated artifact by SHA-256. Remove
    # this only after pinning and recording a replacement graph artifact.
    export = cast(_OnnxExport, torch.onnx.export)
    _ = export(
        model,
        (sample,),
        str(destination),
        input_names=[manifest.graph.input.name],
        output_names=[manifest.graph.output.name],
        dynamic_axes={
            manifest.graph.input.name: {0: manifest.graph.input.batch_dimension},
            manifest.graph.output.name: {0: manifest.graph.output.batch_dimension},
        },
        opset_version=source.opset,
        dynamo=False,
    )


def export_timm_safetensors(
    weights_path: Path,
    destination: Path,
    manifest: SourceManifest,
) -> None:
    """Export one pinned timm checkpoint with a dynamic batch dimension."""

    source = manifest.model
    if not isinstance(source, TimmSafetensorsModelSource):
        raise TypeError("timm export requires safetensors source metadata")
    _export_model(_load_model(weights_path, source), destination, manifest, source)


__all__ = ["export_timm_safetensors"]
