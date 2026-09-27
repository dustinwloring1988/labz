# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Grade predictions, in nanochat's own virtualenv.

Run as ``<nanochat venv python> grade.py --checkout <root> --items <dir>
--predictions <dir> --results <path> --keys a,b,c``.

The last of the three passes (see ``core/benchmarks/__init__.py``). It runs in
nanochat's interpreter because grading *is* nanochat's ``Task.evaluate()``, and
that is the whole point of reusing its benchmarks rather than reimplementing
them: GSM8K's answer normalisation, HumanEval's sandboxed execution of
model-written code, and the multiple-choice letter assertions are all decided
there. Restating any of them here would be a second copy to keep in step with
the first, and a copy that drifts is worse than no copy, because it fails
silently as a wrong number.

What this pass deliberately does not do
---------------------------------------
It does not re-derive the gold answer. It reads it out of the same conversation
the materializer stored, so the thing graded against is the thing nanochat
defines as correct rather than a normalised copy of it.

Predictions are matched to items by index, not by position, and a missing
prediction counts as wrong rather than being dropped. Dropping it would let a
run that failed halfway report a flattering accuracy over the problems it
managed to answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional


def _bootstrap_checkout() -> None:
    """Make this file's siblings and the nanochat checkout importable.

    See the same function in ``materialize.py`` for why both directories are added
    explicitly instead of trusting ``sys.path[0]``.
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

from nanochat.events import emit_event  # noqa: E402


def _log(message: str) -> None:
    print(message, flush=True)
    emit_event("log", line=message)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line means the scoring pass was killed
                # mid-write. Everything before it is still valid, so the run
                # reports what it finished rather than nothing.
                continue
    return rows


def _as_prediction_text(value: Any) -> str:
    """A prediction as the string ``Task.evaluate`` expects.

    A letter prediction is stored as a bare letter, which is already what the
    multiple-choice tasks compare against.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def grade_one(
    task: Any,
    key: str,
    items_path: Path,
    predictions_path: Path,
    exact_path: Path | None = None,
) -> dict:
    """Grade one benchmark, returning its score row.

    Also writes a per-item verdict file when ``exact_path`` is given. The totals
    alone cannot say which *way* a judge's verdict differed from this one, and a
    judge that only ever raises a score is a different finding from one that
    lowers half of them -- so the item-level verdicts are kept for that
    comparison rather than re-derived from the totals afterwards.
    """
    started = time.time()
    items = _load_jsonl(items_path)
    predictions = {
        int(row.get("index", -1)): row
        for row in _load_jsonl(predictions_path)
        if isinstance(row.get("index"), int)
    }

    correct = 0
    total = 0
    ungraded = 0
    per_item: list[dict] = []

    for row in items:
        index = int(row.get("index", -1))
        conversation = row.get("conversation")
        if not isinstance(conversation, dict):
            continue
        total += 1
        prediction = predictions.get(index)
        if prediction is None:
            # Counted, not skipped: see the module docstring.
            ungraded += 1
            per_item.append({"index": index, "exact": None, "graded": False})
            continue
        text = _as_prediction_text(prediction.get("prediction"))
        try:
            outcome = task.evaluate(conversation, text)
        except AssertionError as exc:
            # The multiple-choice tasks assert the prediction is one of the
            # letters, which a real model violates routinely. That is a wrong
            # answer, not a broken run, so it is counted wrong and the reason
            # is recorded once rather than raised.
            if ungraded == 0:
                _log(f"{key}: {exc}")
            ungraded += 1
            per_item.append({"index": index, "exact": False, "graded": True})
            continue
        except Exception as exc:  # noqa: BLE001
            _log(f"{key}: item {index} failed to grade - {type(exc).__name__}: {exc}")
            ungraded += 1
            per_item.append({"index": index, "exact": None, "graded": False})
            continue
        was_correct = bool(outcome)
        correct += int(was_correct)
        per_item.append({"index": index, "exact": was_correct, "graded": True})

    if exact_path is not None:
        try:
            exact_path.parent.mkdir(parents=True, exist_ok=True)
            with exact_path.open("w", encoding="utf-8") as handle:
                for entry in per_item:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            # The totals are correct without this file; only the breakdown is
            # lost, so a write failure must not fail the grade.
            _log(f"{key}: could not write per-item verdicts - {exc}")

    accuracy = (correct / total) if total else None
    return {
        "key": key,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "elapsed_seconds": time.time() - started,
        "ungraded": ungraded,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Grade benchmark predictions.")
    # Defaults from the environment, for the same reason as the materializer: the
    # orchestrator already sets it, and a flag that repeats it is a second place
    # for the two to disagree.
    parser.add_argument(
        "--checkout",
        default=os.environ.get("UNSLOTH_NANOCHAT_CHECKOUT"),
        help="nanochat checkout root. Defaults to $UNSLOTH_NANOCHAT_CHECKOUT.",
    )
    parser.add_argument("--items", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument(
        "--exact-dir",
        default=None,
        help="Where to write per-item exact verdicts, for comparison against a judge.",
    )
    args = parser.parse_args(argv)

    if not args.checkout:
        _log("no nanochat checkout: pass --checkout or set UNSLOTH_NANOCHAT_CHECKOUT")
        return 2
    checkout = Path(args.checkout)
    items_dir = Path(args.items)
    predictions_dir = Path(args.predictions)

    keys = [key.strip() for key in args.keys.split(",") if key.strip()]

    try:
        from tasks.registry import BENCHMARKS  # noqa: F401
    except Exception:
        _log("could not import nanochat's task registry")
        _log(traceback.format_exc())
        return 2

    import nanochat_tasks

    results: dict[str, Any] = {"scores": {}, "failures": {}}

    for key in keys:
        items_path = items_dir / f"{key}.jsonl"
        predictions_path = predictions_dir / f"{key}.jsonl"
        if not items_path.exists():
            results["failures"][key] = "no items were materialized"
            continue
        if not predictions_path.exists():
            # Distinct from a zero score: nothing was attempted, so reporting
            # 0% would blame the model for a pass that never ran.
            results["failures"][key] = "the scoring pass produced no predictions"
            continue
        try:
            task = nanochat_tasks.build_task(key)
            exact_path = None
            if args.exact_dir:
                exact_path = Path(args.exact_dir) / f"{key}.exact.jsonl"
            row = grade_one(task, key, items_path, predictions_path, exact_path)
            results["scores"][key] = row
            accuracy = row["accuracy"]
            _log(
                f"{key}: {row['correct']}/{row['total']}"
                + (f" = {accuracy * 100:.1f}%" if accuracy is not None else "")
            )
            # Spelled out rather than **row: the row already carries `key`, and
            # passing both raises inside emit_event -- which the handler below
            # would then catch and turn into a failure, replacing a score that
            # had already been computed and stored.
            emit_event(
                "graded",
                key=key,
                correct=row["correct"],
                total=row["total"],
                accuracy=accuracy,
                elapsed_seconds=row["elapsed_seconds"],
                ungraded=row["ungraded"],
            )
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            _log(f"{key}: grading failed - {detail}")
            # A benchmark that already produced a score keeps it: a failure in
            # reporting is not a failure to grade, and dropping a good score
            # because an event could not be written would be the wrong trade.
            results["failures"].setdefault(key, detail)

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if results["scores"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
