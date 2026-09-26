# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reads the nanochat catalogue (datasets, SFT tasks, benchmarks) from nanochat.

The tables live in ``nanochat/dataset.py``, ``tasks/registry.py`` and
``nanochat/registry.py``, in the nanochat virtualenv, which this process cannot
import: it has a different pinned torch.

So rather than mirror them here (and have the two drift the first time either is
edited), this asks nanochat to print its own tables as JSON and caches the answer.
The trade is a subprocess on first call; the cache means the pickers do not pay
it repeatedly.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from typing import Any, Optional

from core.nanochat import environment
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# The catalogue changes only when nanochat is updated, so a long cache is safe.
# Long enough that opening the tab repeatedly does not spawn a process each time.
_CACHE_TTL_SECONDS = 900
_FETCH_TIMEOUT_SECONDS = 180

_lock = threading.Lock()
_cache: dict[str, Any] = {}
_cached_at: float = 0.0

# Shown when nanochat cannot be reached, so the pickers still render something
# honest instead of an empty list the user cannot interpret.
_FALLBACK: dict[str, Any] = {
    "schema_version": 0,
    "unavailable": True,
    "reason": "nanochat has not been set up yet",
    "datasets": {},
    "default_dataset": None,
    "training_tasks": {},
    "default_training_tasks": [],
    "benchmarks": {},
    "default_chat_benchmarks": [],
    "base_evals": {},
    "default_base_evals": [],
    "stages": [],
    "trained_on_by_default": [],
    "registry_problems": [],
}


def invalidate_cache() -> None:
    """Drop the cache. Called after an install or checkout update."""
    global _cached_at
    with _lock:
        _cache.clear()
        _cached_at = 0.0


def get_catalogue(force: bool = False) -> dict[str, Any]:
    """The catalogue, from cache when fresh.

    Never raises. A picker that cannot load still has to render, with an
    explanation, rather than failing the page.
    """
    global _cached_at
    with _lock:
        if not force and _cache and (time.time() - _cached_at) < _CACHE_TTL_SECONDS:
            return _cache

    payload = _fetch()

    with _lock:
        _cache.clear()
        _cache.update(payload)
        _cached_at = time.time()
        return _cache


def _fetch() -> dict[str, Any]:
    status = environment.environment_status()
    if not status.checkout_present or not status.venv_present:
        reason = status.blocking_reason or "nanochat is not installed"
        return {**_FALLBACK, "reason": reason}

    try:
        proc = subprocess.run(
            [str(environment.venv_python()), "-m", "nanochat.registry"],
            cwd=str(environment.checkout_root()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_FETCH_TIMEOUT_SECONDS,
            **windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return {**_FALLBACK, "reason": "nanochat took too long to describe itself"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read the nanochat catalogue: %s", exc)
        return {**_FALLBACK, "reason": str(exc)}

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit {proc.returncode}"
        logger.warning("nanochat.registry failed: %s", reason)
        return {**_FALLBACK, "reason": reason}

    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("nanochat.registry printed invalid JSON: %s", exc)
        return {**_FALLBACK, "reason": "nanochat returned unreadable output"}

    if not isinstance(parsed, dict):
        return {**_FALLBACK, "reason": "nanochat returned an unexpected shape"}

    # Merge over the fallback so a field added in a newer nanochat is visible and
    # a field removed does not turn into a KeyError downstream.
    merged = {**_FALLBACK, **parsed, "unavailable": False}
    merged.pop("reason", None)
    if parsed.get("registry_problems"):
        logger.warning(
            "nanochat reports registry problems: %s", parsed["registry_problems"]
        )
    return merged


def datasets() -> dict[str, Any]:
    return get_catalogue().get("datasets") or {}


def benchmarks() -> dict[str, Any]:
    return get_catalogue().get("benchmarks") or {}


def training_tasks() -> dict[str, Any]:
    return get_catalogue().get("training_tasks") or {}


def base_evals() -> dict[str, Any]:
    return get_catalogue().get("base_evals") or {}


def stages() -> list[dict]:
    return get_catalogue().get("stages") or []


def default_selection() -> dict[str, Any]:
    """The defaults a fresh run should start with.

    These are nanochat's own defaults, taken from its tables rather than restated
    here, so a run started from the UI matches the published pipeline.
    """
    cat = get_catalogue()
    return {
        "dataset": cat.get("default_dataset"),
        "sft_tasks": list(cat.get("default_training_tasks") or []),
        "chat_benchmarks": list(cat.get("default_chat_benchmarks") or []),
        "base_benchmarks": list(cat.get("default_base_evals") or []),
    }
