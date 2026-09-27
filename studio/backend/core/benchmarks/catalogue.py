# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The benchmark picker, read from nanochat's own registry.

The table of what can be benchmarked lives in ``tasks/registry.py`` inside the
nanochat checkout, and is not restated here. That is deliberate: a benchmark
added upstream would otherwise need a matching change in this file to become
selectable, and a benchmark renamed upstream would leave this app offering a key
that no longer resolves. Reading the registry means the picker is whatever
nanochat says it is.

What this module adds is the part only this app knows: which of those benchmarks
can actually be run on this host, and what the defaults should be.

Host gating
-----------
A benchmark is unsupported when running it here would be wrong rather than slow:

  * HumanEval executes model-written code behind a POSIX ``resource`` memory cap.
    Windows has no ``resource`` module. nanochat's own registry flags this
    (``requires_posix_sandbox``) and skips the cap where it is missing, which is
    a weaker guarantee, not an equivalent one. Rather than execute untrusted
    generated code with no memory cap on a Windows host, the benchmark is
    reported unsupported and the picker says why.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from core.nanochat import environment

logger = logging.getLogger(__name__)

# The bounding a run gets when the user does not choose one. Every benchmark
# here is larger than a quick check (MMLU's test split is 14k problems), so an
# unbounded default would be an overnight run with no warning. The UI states the
# cap on the run preview rather than applying it silently.
DEFAULT_MAX_PROBLEMS = 400

# Benchmarks whose own registry sets no cap and which the picker offers first.
# Kept as a fallback only: when nanochat reports its own defaults they win, so a
# change upstream is picked up rather than pinned here.
_FALLBACK_DEFAULTS = ("arc-easy", "arc-challenge", "mmlu", "gsm8k", "humaneval")


def posix_sandbox_available() -> bool:
    """Whether this host can run code behind a memory cap.

    Checked by importing rather than by platform name: nanochat's own sandbox
    needs ``resource`` and a shell, and a name test would report Windows as
    capable of something it cannot do.
    """
    if os.name == "nt":
        return False
    try:
        import resource  # noqa: F401
    except ImportError:
        return False
    return True


def _unsupported_reason(spec: dict, sandbox: bool) -> Optional[str]:
    if spec.get("requires_posix_sandbox") and not sandbox:
        return "runs model-written code in a sandbox that needs a POSIX memory cap"
    return None


def benchmarks(*, force: bool = False) -> list[dict]:
    """The benchmark table, normalised for the API and annotated for this host.

    Never raises. A tab that cannot reach nanochat still has to render, with an
    explanation, rather than failing.
    """
    from core.nanochat import catalogue

    raw = catalogue.get_catalogue(force=force).get("benchmarks") or {}
    sandbox = posix_sandbox_available()

    rows: list[dict] = []
    for key, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        reason = _unsupported_reason(spec, sandbox)
        rows.append(
            {
                "key": str(key),
                "label": str(spec.get("label") or key),
                "task_name": str(spec.get("task_name") or key),
                "notes": str(spec.get("notes") or ""),
                "baseline": float(spec.get("baseline") or 0.0),
                # nanochat calls this `categorical`; the API says `kind` because
                # "categorical" alone reads as a scoring method rather than as
                # what kind of question it is.
                "kind": "categorical" if spec.get("categorical") else "generative",
                "requires_posix_sandbox": bool(spec.get("requires_posix_sandbox")),
                "max_problems": spec.get("max_problems"),
                "is_default": bool(spec.get("is_default")),
                "supported": reason is None,
                "unavailable_reason": reason,
            }
        )

    # Registry order, which is the order a picker should read in.
    rows.sort(key=lambda row: row["key"])
    return rows


def default_benchmarks() -> list[str]:
    """The keys a fresh run starts with.

    nanochat's own defaults when it is reachable, because those are the set its
    published pipeline is measured on; otherwise the hard-coded list above,
    filtered down to what this host can run so the default is never a selection
    the user cannot start.
    """
    from core.nanochat import catalogue

    reported = list(catalogue.get_catalogue().get("default_chat_benchmarks") or [])
    keys = reported or list(_FALLBACK_DEFAULTS)

    supported = {row["key"] for row in benchmarks() if row["supported"]}
    chosen = [key for key in keys if key in supported]
    if chosen:
        return chosen
    # A host that supports none of the defaults still gets something to run.
    return sorted(supported)


def availability() -> tuple[bool, Optional[str]]:
    """Whether the datasets can be fetched, and why not when they cannot."""
    status = environment.environment_status()
    if status.ready():
        return True, None
    return False, status.blocking_reason or "nanochat has not been set up yet"


def resolve_specs(keys: list[str]) -> tuple[list[dict], list[str]]:
    """Look requested keys up, returning the rows and the ones not found.

    An unknown key is reported rather than raised on: the request is validated
    against this same table, and a run that started with a key that has since
    disappeared upstream should still get the benchmarks that did resolve.
    """
    table = {row["key"]: row for row in benchmarks()}
    found: list[dict] = []
    missing: list[str] = []
    for key in keys:
        row = table.get(key)
        if row is None:
            missing.append(key)
        else:
            found.append(row)
    return found, missing


def unsupported_keys(keys: list[str]) -> dict[str, str]:
    """The requested keys this host cannot run, mapped to why."""
    rows, _ = resolve_specs(keys)
    return {
        row["key"]: row["unavailable_reason"] or "unsupported on this host"
        for row in rows
        if not row["supported"]
    }


def catalogue_payload() -> dict[str, Any]:
    """The whole picker, in the shape ``BenchmarkCatalogueResponse`` expects."""
    available, reason = availability()
    return {
        "benchmarks": benchmarks(),
        "default_benchmarks": default_benchmarks(),
        "available": available,
        "reason": reason,
        "posix_sandbox": posix_sandbox_available(),
    }
