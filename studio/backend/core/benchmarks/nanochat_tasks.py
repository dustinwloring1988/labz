# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turning a nanochat benchmark key into the ``Task`` object that defines it.

Run inside nanochat's virtualenv, by ``materialize`` and ``grade``.

Why this table exists
---------------------
nanochat decides which dataset and split a benchmark reads in a dict local to
``scripts/chat_eval.run_chat_eval``::

    'MMLU': partial(MMLU, subset="all", split="test"),
    'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
    ...

That table is the definition, and it is unreachable: a dict inside a function
body cannot be imported. So it is mirrored here, and the mirror is checked
against the registry on every use (see ``missing_factories``) so the two cannot
drift apart quietly. A benchmark added upstream shows up as a named gap rather
than as a run that silently omitted it -- which is the failure mode that matters,
because a leaderboard missing a benchmark looks exactly like a leaderboard where
that model happened to do badly.

Everything below the factory table is nanochat's, reached through its own public
surface: ``Task.get_example`` produces the items and ``Task.evaluate`` grades
them. Neither is restated here.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

# Kept in step with scripts/chat_eval.py's table, and checked against
# tasks.registry.BENCHMARKS by missing_factories() on every pass.
_FACTORIES: dict[str, Callable[[], Any]] = {}


def _load_factories() -> dict[str, Callable[[], Any]]:
    """The task constructors, imported from nanochat's own task modules.

    Imported rather than reimplemented so the classes, and therefore
    ``get_example`` and ``evaluate``, are exactly the ones nanochat ships.
    """
    if _FACTORIES:
        return _FACTORIES
    from tasks.arc import ARC
    from tasks.gsm8k import GSM8K
    from tasks.humaneval import HumanEval
    from tasks.mmlu import MMLU

    # The test split for every benchmark. nanochat trains on the train splits and
    # reports on test, so anything else would measure set the model saw.
    _FACTORIES.update(
        {
            "HumanEval": HumanEval,
            "MMLU": partial(MMLU, subset="all", split="test"),
            "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
            "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
            "GSM8K": partial(GSM8K, subset="main", split="test"),
        }
    )
    return _FACTORIES


def missing_factories() -> list[str]:
    """Registry task names this table cannot build.

    Non-empty means nanochat defines a benchmark this mirror has not caught up
    with. The passes report it rather than skipping the benchmark quietly.
    """
    factories = _load_factories()
    from tasks.registry import BENCHMARKS

    missing: list[str] = []
    for key, spec in BENCHMARKS.items():
        task_name = spec.get("task_name")
        if task_name and task_name not in factories:
            missing.append(f"{key} (task_name={task_name})")
    return missing


def task_name_for(key: str) -> str:
    """The registry's task name for a benchmark key."""
    from tasks.registry import BENCHMARKS

    spec = BENCHMARKS.get(key)
    if spec is None:
        known = ", ".join(sorted(BENCHMARKS))
        raise KeyError(f"Unknown benchmark {key!r}. Available: {known}")
    return str(spec["task_name"])


def build_task(key: str) -> Any:
    """The ``Task`` object for a benchmark key, ready to yield its items.

    Resolution goes key -> registry task_name -> constructor, so a key this side
    has never heard of still works as long as nanochat's table covers it.
    """
    factories = _load_factories()
    task_name = task_name_for(key)
    factory = factories.get(task_name)
    if factory is None:
        raise KeyError(
            f"No task constructor for {task_name!r} (benchmark {key!r}). "
            "nanochat added a benchmark this app has not been taught to build; "
            "add it to _load_factories()."
        )
    return factory()


def baseline_for(key: str) -> float:
    """The benchmark's chance level, for centring a score."""
    from tasks.registry import baseline_for as _baseline_for

    return float(_baseline_for(task_name_for(key)))
