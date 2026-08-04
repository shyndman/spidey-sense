"""Aggregate-only acquisition lifecycle events."""

from __future__ import annotations

import json
import sys

from .acquisition_types import (
    EVENT_CATEGORIES,
    STABLE_FAILURE_CODES,
    AcquisitionEvent,
    JsonObject,
)


def emit_event(
    event: AcquisitionEvent,
    *,
    category: str | None = None,
    page: int | None = None,
    count: int | None = None,
    quota: int | None = None,
    attempted: int | None = None,
    completed: int | None = None,
    skipped: int | None = None,
    failed: int | None = None,
    resumed: int | None = None,
    code: str | None = None,
) -> None:
    """Write one sanitized aggregate lifecycle event to stderr.

    Events intentionally contain no source IDs, URLs, paths, exception text, or
    image metadata. This keeps live progress useful without exposing samples.
    """

    payload: JsonObject = {"stage": "acquire", "event": event}
    if category is not None:
        payload["category"] = (
            category if category in EVENT_CATEGORIES else "unknown"
        )
    for key, value in (
        ("page", page),
        ("count", count),
        ("quota", quota),
        ("attempted", attempted),
        ("completed", completed),
        ("skipped", skipped),
        ("failed", failed),
        ("resumed", resumed),
    ):
        if value is not None:
            payload[key] = value
    if code is not None:
        payload["code"] = (
            code if code in STABLE_FAILURE_CODES else "ACQUISITION_FAILED"
        )
    _ = sys.stderr.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    _ = sys.stderr.flush()


__all__ = ["emit_event"]
