# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Identity of a materialized item cache entry.

Items depend only on the benchmark, its dataset, and the run's problem cap, and
are shared across runs so a second suite of the same benchmark skips the download.

The cap is part of the identity and that is not incidental. The cached file is the
*truncated* item list rather than the dataset, so a file written for 400 problems
is a different thing from one written for 8. Keyed on the dataset alone, a run
capped at 8 found and reused the 400-problem file and scored all 400 -- reporting a
total the user never asked for, and marking the run "sampled" for the wrong reason.
Nothing in the stamp had changed, so nothing looked wrong.

This lives apart from ``materialize`` because that module is only importable inside
nanochat's virtualenv, and a wrong cache key does its damage in *this* process.
The policy is therefore testable where it is relied on, rather than only in the one
interpreter that cannot exercise it.
"""

from __future__ import annotations

from pathlib import Path

# Where each benchmark's dataset lives, used only to build a cache key. Which
# dataset a benchmark actually reads is nanochat's business and is resolved by its
# own task classes; this table exists so the key changes if a dataset is re-fetched.
#
# Mirrors the splits in nanochat's tasks/*.py. A benchmark absent here falls back to
# a key of its own name, which is safe: it just means the entry is not invalidated
# by a dataset re-fetch.
TASK_SPECS: dict[str, tuple[str, str, str]] = {
    "mmlu": ("cais/mmlu", "all", "test"),
    "arc-easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "arc-challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "humaneval": ("openai_humaneval", "openai_humaneval", "test"),
}


def _cap_part(max_problems: int | None) -> str:
    # "all" rather than a number for the uncapped case, so it cannot collide with
    # a cap that happens to stringify the same way.
    return "all" if max_problems is None else str(int(max_problems))


def item_cache_stamp(key: str, max_problems: int | None) -> str:
    """The cache key for one benchmark at one problem cap.

    Read from the caller rather than from the environment, so the two callers --
    the check and the write -- cannot disagree about the cap within one run.
    """
    spec = TASK_SPECS.get(key)
    if spec is None:
        return f"{key}:{_cap_part(max_problems)}"
    repo, subset, split = spec
    return f"{repo}/{subset}/{split}:{_dataset_component(repo, subset, split)}:{_cap_part(max_problems)}"


def _dataset_component(repo: str, subset: str, split: str) -> str:
    """The size and mtime of a dataset's shard manifest, or ``unknown``.

    nanochat's ``load_hub_dataset`` writes ``manifest.json`` last, after the shards
    land, so its presence means the download finished. Size plus mtime is enough to
    notice a re-fetch without depending on the directory layout.

    ``unknown`` when nanochat's own module is unreachable, which is the normal case
    outside its virtualenv. The cap still discriminates, and the cap is the half
    that was silently wrong; a dataset re-fetch then re-materializes on the next cap
    change rather than on its own.
    """
    try:
        from nanochat.common import get_base_dir
        base = get_base_dir()
    except Exception:  # noqa: BLE001
        return "unknown"
    manifest = Path(base) / "task_data" / repo.replace("/", "--") / subset / split / "manifest.json"
    try:
        stat = manifest.stat()
    except OSError:
        # Not downloaded yet, which is a valid state: the download happens during
        # the pass that will write this entry.
        return "absent"
    return f"{stat.st_size}:{int(stat.st_mtime)}"
