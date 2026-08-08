"""Typed evaluation reporting stage facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import (
        EvaluationReports,
        ModelReport,
        ModelReportRequest,
        ReportsRequest,
    )


def build_model_report(request: ModelReportRequest) -> ModelReport:
    from .stage import build_model_report as _build_model_report

    return _build_model_report(request)


def run(request: ReportsRequest) -> EvaluationReports:
    from .stage import run as _run

    return _run(request)


__all__ = ["run", "build_model_report"]
