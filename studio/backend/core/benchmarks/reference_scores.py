# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Published scores, shown on the leaderboard as reference rows.

Why these exist
---------------
A leaderboard with one row is not a leaderboard. Without something to compare
the first measurement against, the first run lands in an empty table and reads
as a broken feature rather than as a starting point.

Why they are marked ``reference`` and never mixed in silently
------------------------------------------------------------
A published score and a score this app measured are different kinds of number,
and the differences are not cosmetic:

  * The published figures were produced by somebody else's harness, with its own
    prompt rendering, its own shot count and its own answer parsing. This app
    scores through nanochat's tasks, which render multiple choice as
    ``- choice=letter`` and parse generative answers with nanochat's own
    normalisers. Same benchmark name, different measurement.
  * The shot counts differ, and a multiple-choice score moves several points
    with it.
  * Nothing here was measured on this machine, on this GPU, in this quantisation.

So these rows are anchors, not comparators. Each carries the source it came from
and is rendered with a badge; the API never merges them into a local average.
Every figure below is transcribed from the model card named in its ``note``, and
a benchmark that card does not report is left absent rather than filled in from a
second, differently-configured source. An absent column is honest; a number
spliced from two protocols is not.

To correct or extend this
-------------------------
This table is the only place a published figure is recorded. Keys are the
benchmark keys from ``core/nanochat``'s registry (``mmlu``, ``arc-easy``,
``arc-challenge``, ``gsm8k``, ``humaneval``); a key this registry does not define
is ignored, so adding a benchmark upstream does not need a change here.
"""

from __future__ import annotations

from typing import Optional

# Fraction correct, not percent.
#
# ``ARC-Easy`` and ``ARC-Challenge`` are absent for most rows below because the
# model cards that report the other three do not report them. That is left as a
# gap on purpose; see the module docstring.
REFERENCE_SCORES: list[dict] = [
    {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_label": "Llama 3.1 8B Instruct",
        "scores": {
            "mmlu": 0.68,
            "gsm8k": 0.845,
            "humaneval": 0.335,
        },
        "note": (
            "Meta Llama 3.1 model card, instruction-tuned results: MMLU 5-shot, "
            "GSM8K 8-shot CoT maj@1, HumanEval pass@1."
        ),
    },
    {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_label": "Qwen2.5 7B Instruct",
        "scores": {
            "mmlu": 0.74,
            "gsm8k": 0.916,
            "humaneval": 0.848,
        },
        "note": (
            "Qwen2.5 technical report, 7B-Instruct evaluation table: MMLU 5-shot, "
            "GSM8K and HumanEval as reported there."
        ),
    },
]


def references_for(benchmark_keys: list[str]) -> list[dict]:
    """The published rows, projected onto the benchmarks the picker offers.

    Anything outside ``benchmark_keys`` is dropped so a stale entry cannot put a
    column on the board that no local run can ever fill.
    """
    wanted = set(benchmark_keys)
    rows: list[dict] = []
    for entry in REFERENCE_SCORES:
        scores = {
            key: float(value)
            for key, value in (entry.get("scores") or {}).items()
            if key in wanted and value is not None
        }
        if not scores:
            continue
        rows.append(
            {
                "model_id": entry["model_id"],
                "model_label": entry["model_label"],
                "scores": scores,
                "source_note": entry.get("note"),
            }
        )
    return rows


def reference_score(benchmark_keys: list[str], model_id: str, key: str) -> Optional[float]:
    """One published figure, or None. Used to show a run against its own baseline."""
    for row in references_for(benchmark_keys):
        if row["model_id"] == model_id:
            return row["scores"].get(key)
    return None
