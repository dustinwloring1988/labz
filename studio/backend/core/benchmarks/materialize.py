# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Materialize benchmark items, in nanochat's own virtualenv.

Run as ``<nanochat venv python> materialize.py --checkout <root> --out <dir>
--keys a,b,c --max-problems N``.

This is the first of the three passes (see ``core/benchmarks/__init__.py`` for
why they are split). It runs here because ``tasks/*.py`` and the parquet
downloader they use live in the nanochat checkout, and because that interpreter
already has everything they need: pyarrow, urllib, a lock file. It needs no
torch, and none is loaded, so nanochat's pinned build is irrelevant to it.

What it writes
--------------
One JSONL file per benchmark, each line the item exactly as nanochat's
``Task.get_example`` returns it, plus a manifest. Conversations are stored whole
rather than flattened to a prompt and an answer, because the grading pass needs
them whole: ``Task.evaluate()`` reads the gold turn out of the conversation, and
GSM8K's gold turn is a list of parts with tool calls interleaved, which cannot
be reduced to a string without reimplementing the thing being reused.

Caching
-------
Items depend only on the benchmark and its dataset, not on the model, so they
are cached under a content-addressed directory and reused across runs. The cache
key includes the dataset shard manifest's mtime, so a dataset that is re-fetched
upstream invalidates rather than silently serving stale problems.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional


def _bootstrap_checkout() -> None:
    """Make this file's siblings and the nanochat checkout importable.

    This module lives in the backend tree and is invoked by absolute path, so
    Python puts *this* directory on sys.path and nanochat's own packages are
    invisible. nanochat's scripts never hit this because they run with the
    checkout as the working directory.

    This file's own directory is added explicitly as well, rather than relying on
    ``sys.path[0]`` happening to be it. That is true when the file is run as a
    script and false when it is imported and its ``main`` called, which is a
    supported way to drive it and a confusing way for it to fail.

    Done at import time rather than inside main() because the event emitter below
    is imported at module scope, and resolving the path after that import has
    already failed would be too late. The orchestrator sets
    ``UNSLOTH_NANOCHAT_CHECKOUT``; the argv scan is the fallback that lets this
    file be run by hand.
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)

    root = os.environ.get("UNSLOTH_NANOCHAT_CHECKOUT")
    if not root and "--checkout" in sys.argv:
        index = sys.argv.index("--checkout")
        if index + 1 < len(sys.argv):
            root = sys.argv[index + 1]
    if root and root not in sys.path:
        sys.path.insert(0, root)


_bootstrap_checkout()

# The event channel is nanochat's own. This process is inside its checkout with
# its interpreter, so it uses the emitter the project ships rather than a
# parallel one; the orchestrator tails the resulting file either way.
from nanochat.events import emit_event  # noqa: E402

# A sibling, not a `core.benchmarks` path: this process is running under nanochat's
# interpreter with only this directory and the checkout on the path, so the package
# name is not importable here even though the module is the same file.
from item_cache import item_cache_stamp  # noqa: E402


def _log(message: str) -> None:
    print(message, flush = True)
    emit_event("log", line = message)


def _user_text(conversation: dict) -> str:
    """The user turn, as a string.

    A conversation is a list of turns and only the first is ever the user's, so
    this is total rather than defensive.
    """
    for message in conversation.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content)
    return ""


def materialize_one(task: Any, key: str, limit: Optional[int], out_dir: Path) -> dict:
    """Write one benchmark's items, and report what was written."""
    total = int(task.num_examples())
    take = total if limit is None else min(total, int(limit))
    target = out_dir / f"{key}.jsonl"

    written = 0
    with target.open("w", encoding = "utf-8") as handle:
        for index in range(take):
            conversation = task.get_example(index)
            # Stored whole: the grading pass needs the gold turn, and a
            # GSM8K one is a list of parts rather than a string.
            handle.write(
                json.dumps(
                    {"index": index, "conversation": conversation},
                    ensure_ascii = False,
                )
                + "\n"
            )
            written += 1

    return {"key": key, "path": str(target), "written": written, "available": total}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = "Materialize benchmark items.")
    # Defaults from the environment rather than required, because the orchestrator
    # sets UNSLOTH_NANOCHAT_CHECKOUT and a flag that had to repeat it would be a
    # second place for the two to disagree. The flag stays for running this by hand.
    parser.add_argument(
        "--checkout",
        default = os.environ.get("UNSLOTH_NANOCHAT_CHECKOUT"),
        help = "nanochat checkout root. Defaults to $UNSLOTH_NANOCHAT_CHECKOUT.",
    )
    parser.add_argument("--out", required = True)
    parser.add_argument("--keys", required = True, help = "comma-separated benchmark keys")
    parser.add_argument("--max-problems", type = int, default = None)
    args = parser.parse_args(argv)

    if not args.checkout:
        _log("no nanochat checkout: pass --checkout or set UNSLOTH_NANOCHAT_CHECKOUT")
        return 2
    checkout = Path(args.checkout)
    os.environ.setdefault("NANOCHAT_BASE_DIR", str(checkout / ".base"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents = True, exist_ok = True)
    keys = [key.strip() for key in args.keys.split(",") if key.strip()]

    try:
        from tasks.registry import BENCHMARKS  # noqa: F401
    except Exception:
        _log("could not import nanochat's task registry")
        _log(traceback.format_exc())
        return 2

    import nanochat_tasks

    # Checked before anything is downloaded. A benchmark this app cannot build a
    # task for is reported by name, because a run that quietly omitted one would
    # produce a leaderboard that looks complete and is not.
    gaps = nanochat_tasks.missing_factories()
    if gaps:
        _log(f"this app cannot build these benchmarks yet: {', '.join(gaps)}")

    manifest: dict[str, Any] = {"stamps": {}, "items": {}, "unsupported": gaps}
    failures: dict[str, str] = {}

    for key in keys:
        stamp = item_cache_stamp(key, args.max_problems)
        cached = out_dir / f"{key}.jsonl"
        stamp_file = out_dir / f"{key}.stamp"

        # A cache hit skips the download and the per-item rendering, which for
        # MMLU is the bulk of this pass.
        if cached.exists() and stamp_file.exists():
            try:
                if stamp_file.read_text(encoding = "utf-8").strip() == stamp:
                    with cached.open("r", encoding = "utf-8") as handle:
                        written = sum(1 for _ in handle)
                    _log(f"{key}: reusing {written} cached items")
                    manifest["stamps"][key] = stamp
                    manifest["items"][key] = {
                        "path": str(cached),
                        "written": written,
                    }
                    continue
            except OSError:
                pass

        try:
            _log(f"{key}: materializing")
            # Building the task resolves the dataset, downloading it on first
            # use. That is the slow part and it prints; the lines land in the
            # in-app console so the wait is visible rather than silent.
            task = nanochat_tasks.build_task(key)
            result = materialize_one(task, key, args.max_problems, out_dir)
            # Re-read the stamp now that the task has been built. Before the
            # download the manifest does not exist, so the pre-build stamp says
            # "absent" and the next run would re-materialize every time, which
            # for MMLU is the whole point of having a cache.
            stamp = item_cache_stamp(key, args.max_problems)
            stamp_file.write_text(stamp, encoding = "utf-8")
            _log(f"{key}: {result['written']} of {result['available']} items")
            manifest["stamps"][key] = stamp
            manifest["items"][key] = result
        except Exception as exc:  # noqa: BLE001
            # One benchmark failing must not lose the others: a run that can do
            # MMLU and ARC is still worth running even if a download 404s.
            detail = f"{type(exc).__name__}: {exc}"
            _log(f"{key}: failed - {detail}")
            failures[key] = detail

    manifest["failures"] = failures
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii = False, indent = 2), encoding = "utf-8"
    )

    if not manifest["items"]:
        emit_event("materialized", items = {}, failures = failures)
        return 1
    emit_event("materialized", items = manifest["items"], failures = failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
