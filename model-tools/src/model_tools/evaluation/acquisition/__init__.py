"""Corpus acquisition stage facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..application import StageExecution
    from .models import AcquisitionRequest


def run(request: AcquisitionRequest) -> StageExecution:
    from .stage import run as _run

    return _run(request)


__all__ = ["run"]
