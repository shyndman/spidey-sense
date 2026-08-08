"""Annotation stage package."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import TracebackType
from typing import Literal, Protocol, TypeGuard, cast

from ..storage.layout import EvaluationPaths
from .models import DetectorRuntime, ModelProtocol, ProcessorProtocol, TorchProtocol

CHECKPOINT = "IDEA-Research/grounding-dino-base"


class _ContextManager(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _ImageSource(_ContextManager, Protocol):
    def convert(self, mode: str) -> object: ...


def is_callable(value: object) -> TypeGuard[Callable[..., object]]:
    return callable(value)


def is_context_manager(value: object) -> TypeGuard[_ContextManager]:
    enter: object = getattr(value, "__enter__", None)
    exit_method: object = getattr(value, "__exit__", None)
    return is_callable(enter) and is_callable(exit_method)


def is_image_source(value: object) -> TypeGuard[_ImageSource]:
    convert: object = getattr(value, "convert", None)
    return is_context_manager(value) and is_callable(convert)


def load_runtime(
    paths: EvaluationPaths,
) -> DetectorRuntime:
    """Load detector dependencies lazily and choose one shared execution device."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    cache_dir = paths.cache / "huggingface"
    _ = cache_dir.mkdir(parents=True, exist_ok=True)
    processor_class: object = getattr(transformers, "AutoProcessor", None)
    processor_factory: object = getattr(processor_class, "from_pretrained", None)
    if not is_callable(processor_factory):
        raise ImportError("transformers processor loader unavailable")
    model_class: object = getattr(
        transformers,
        "AutoModelForZeroShotObjectDetection",
        None,
    )
    model_factory: object = getattr(model_class, "from_pretrained", None)
    if not is_callable(model_factory):
        raise ImportError("transformers detector loader unavailable")
    processor = processor_factory(CHECKPOINT, cache_dir=str(cache_dir))
    model = model_factory(
        CHECKPOINT,
        cache_dir=str(cache_dir),
    )
    device: Literal["cpu", "cuda"] = "cpu"
    cuda: object = getattr(torch, "cuda", None)
    is_available: object = getattr(cuda, "is_available", None)
    if is_callable(is_available):
        available = is_available()
        if isinstance(available, bool) and available:
            device = "cuda"
    to_device: object = getattr(model, "to", None)
    if is_callable(to_device):
        _ = to_device(device)
    eval_method: object = getattr(model, "eval", None)
    if is_callable(eval_method):
        _ = eval_method()
    return DetectorRuntime(
        torch=cast(TorchProtocol, cast(object, torch)),
        processor=cast(ProcessorProtocol, processor),
        model=cast(ModelProtocol, model),
        device=device,
    )


class NullContext:
    def __enter__(self) -> NullContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None
