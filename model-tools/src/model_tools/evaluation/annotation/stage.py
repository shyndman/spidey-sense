"""Annotation stage package."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

from ..acquisition.models import SampleManifest
from ..application import StageExecution, StageSummary
from ..base import MutableEvaluationModel
from ..events import AggregateEvent, AggregateEventRecord, emit_event
from ..storage.layout import EvaluationPaths
from .detector import build_annotation
from .models import AnnotationRecord, AnnotationRequest, DetectorRuntime
from .repository import valid_annotation, write_annotation, write_failure
from .runtime import (
    NullContext,
    is_callable,
    is_context_manager,
    is_image_source,
    load_runtime,
)


class _AnnotationState(MutableEvaluationModel):
    attempted: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    processed: int = 0


class _AnnotationRunner:
    paths: EvaluationPaths
    state: _AnnotationState

    def __init__(self, request: AnnotationRequest) -> None:
        self.paths = request.paths
        self.state = _AnnotationState()

    def run(self) -> StageSummary:
        self.paths.ensure()
        pending = self._collect_pending()
        self._emit("start")
        if not pending:
            return self._finish()
        self._emit("model_loading")
        try:
            runtime = load_runtime(self.paths)
            torch = runtime.torch
            _ = runtime.processor
            _ = runtime.model
            _ = runtime.device
        except Exception:
            self._fail_pending(pending, "model_unavailable")
            return self._finish()
        no_grad: object = getattr(torch, "no_grad", None)
        self._emit("model_ready")
        try:
            image_module = self._load_image_module()
        except Exception:
            self._fail_pending(pending, "image_loader_unavailable")
            return self._finish()
        image_opener = getattr(image_module, "open", None)
        if not is_callable(image_opener):
            self._fail_pending(pending, "image_loader_unavailable")
            return self._finish()
        self._process_pending(pending, runtime, image_opener, no_grad)
        return self._finish()

    def _collect_pending(self) -> list[tuple[Path, SampleManifest]]:
        pending: list[tuple[Path, SampleManifest]] = []
        for manifest_path in _manifest_files(self.paths):
            self.state.attempted += 1
            try:
                sample = SampleManifest.model_validate_json(manifest_path.read_bytes())
            except (OSError, ValueError, TypeError):
                self.state.failed += 1
                write_failure(
                    self.paths, manifest_path, code="manifest_invalid", sample_id=None
                )
                continue
            destination = self.paths.annotation_path(sample.sample_id)
            if destination.exists() and valid_annotation(destination, sample.sample_id):
                self.state.skipped += 1
                continue
            pending.append((manifest_path, sample))
        return pending

    def _load_image_module(self) -> object:
        return importlib.import_module("PIL.Image")

    def _fail_pending(
        self,
        pending: list[tuple[Path, SampleManifest]],
        code: str,
    ) -> None:
        for manifest_path, sample in pending:
            self.state.failed += 1
            write_failure(
                self.paths, manifest_path, code=code, sample_id=sample.sample_id
            )
            self._processed()

    def _process_pending(
        self,
        pending: list[tuple[Path, SampleManifest]],
        runtime: DetectorRuntime,
        image_opener: Callable[..., object],
        no_grad: object,
    ) -> None:
        for manifest_path, sample in pending:
            try:
                record = self._annotate_one(sample, runtime, image_opener, no_grad)
                write_annotation(self.paths.annotation_path(sample.sample_id), record)
            except Exception:
                self.state.failed += 1
                write_failure(
                    self.paths,
                    manifest_path,
                    code="annotation_failed",
                    sample_id=sample.sample_id,
                )
            else:
                self.state.completed += 1
            self._processed()

    def _annotate_one(
        self,
        sample: SampleManifest,
        runtime: DetectorRuntime,
        image_opener: Callable[..., object],
        no_grad: object,
    ) -> AnnotationRecord:
        image_path = self.paths.image_path(sample.image_relative_path).resolve()
        _ = image_path.relative_to(self.paths.images.resolve())
        if not image_path.is_file():
            raise FileNotFoundError
        source_image = image_opener(image_path)
        if not is_image_source(source_image):
            raise TypeError("Pillow image source unavailable")
        with source_image:
            image = source_image.convert("RGB")
            if is_callable(no_grad):
                candidate = no_grad()
                if not is_context_manager(candidate):
                    raise TypeError("torch no_grad context unavailable")
                with candidate:
                    return build_annotation(
                        sample, runtime.processor, runtime.model, image, runtime.device
                    )
            with NullContext():
                return build_annotation(
                    sample, runtime.processor, runtime.model, image, runtime.device
                )

    def _processed(self) -> None:
        self.state.processed += 1
        if self.state.processed % 25 == 0:
            self._emit("progress", processed=self.state.processed)

    def _emit(self, event: AggregateEvent, *, processed: int = 0) -> None:
        _emit_stage_event(
            event,
            attempted=self.state.attempted,
            completed=self.state.completed,
            skipped=self.state.skipped,
            failed=self.state.failed,
            processed=processed,
        )

    def _finish(self) -> StageSummary:
        summary = StageSummary(
            attempted=self.state.attempted,
            completed=self.state.completed,
            skipped=self.state.skipped,
            failed=self.state.failed,
        )
        return _complete(summary, processed=self.state.processed)


def _manifest_files(paths: EvaluationPaths) -> tuple[Path, ...]:
    return tuple(sorted(paths.manifests.glob("*.json"), key=lambda path: path.name))


def _emit_stage_event(
    event: AggregateEvent,
    summary: StageSummary | None = None,
    *,
    attempted: int | None = None,
    completed: int | None = None,
    skipped: int | None = None,
    failed: int | None = None,
    processed: int = 0,
) -> None:
    if summary is not None:
        attempted = summary.attempted
        completed = summary.completed
        skipped = summary.skipped
        failed = summary.failed
    if attempted is None or completed is None or skipped is None or failed is None:
        raise ValueError("stage event counts are required")
    emit_event(
        AggregateEventRecord(
            stage="annotate",
            event=event,
            attempted=attempted,
            completed=completed,
            skipped=skipped,
            failed=failed,
            processed=processed,
        )
    )


def _complete(summary: StageSummary, *, processed: int = 0) -> StageSummary:
    _emit_stage_event("complete", summary, processed=processed)
    return summary


def _run_annotation(request: AnnotationRequest) -> StageSummary:
    """Resume annotation, retaining all detector proposals including weak ones."""
    return _AnnotationRunner(request).run()


def run(request: AnnotationRequest) -> StageExecution:
    return StageExecution(stage="annotate", summary=_run_annotation(request))
