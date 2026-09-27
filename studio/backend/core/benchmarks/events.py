# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The scoring pass's side of the event stream.

The other two passes (``materialize`` and ``grade``) run in nanochat's
virtualenv and emit through ``nanochat.events``, which is the channel that
project provides. This module produces byte-identical records for the scoring
pass so all three write one interleaved JSONL stream that the orchestrator tails
once.

The record shape is therefore fixed by nanochat and is not this side's to
change: ``{"kind", "ts", "pid", ...fields}``, one JSON object per line, flushed
per record. Emitting anything else would mean the orchestrator had to know which
pass wrote a line.

Two properties are load-bearing and are the reason this is not a ``print``:

  * Flush per record. A scoring pass killed mid-benchmark must leave the
    problems it already graded on disk, or the run reports zero for work it did.
  * Non-finite floats become null. ``NaN`` is not valid JSON, and a strict
    parser downstream would drop the record and the progress with it. A gap in a
    chart is better than a silent hole in the event stream.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

_handle = None
_handle_path: Optional[str] = None


def _clean(value: Any) -> Any:
    """Make a value survive a JSON round trip.

    Mirrors ``nanochat.events._clean``: a float that is NaN or infinite has no
    JSON representation, and reporting it as a stale or invented number would be
    worse than reporting nothing.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def emit(kind: str, **fields: Any) -> bool:
    """Append one record. Returns whether it was written. Never raises.

    A driver problem must not kill a run that is otherwise scoring fine, so
    every failure here is swallowed: the run still finishes, and the missing
    record shows up as a gap rather than as a lost run.
    """
    global _handle, _handle_path
    try:
        path = os.environ.get("NANOCHAT_EVENT_FILE") or os.environ.get(
            "UNSLOTH_BENCHMARK_EVENT_FILE"
        )
        if not path:
            return False
        if _handle is None or _handle_path != path:
            if _handle is not None:
                try:
                    _handle.close()
                except Exception:
                    pass
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            _handle = open(path, "a", encoding="utf-8", buffering=1)
            _handle_path = path
        record: dict[str, Any] = {"kind": kind, "ts": time.time(), "pid": os.getpid()}
        record.update(_clean(fields))
        _handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _handle.flush()
        return True
    except Exception:
        return False


def stop_requested() -> bool:
    """Whether the orchestrator has asked the run to stop.

    The scoring pass polls this between batches. A benchmark of 14k problems
    would otherwise be un-stoppable, and the sentinel is what lets a stop take
    effect with the problems already scored still reported.
    """
    path = os.environ.get("NANOCHAT_STOP_FILE") or os.environ.get("UNSLOTH_BENCHMARK_STOP_FILE")
    return bool(path) and os.path.exists(path)
