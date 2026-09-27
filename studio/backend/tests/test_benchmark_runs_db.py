# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark storage: what gets ranked, and what is kept off the board.

The two rules worth protecting are that a truncated score cannot outrank a
full-set one, and that a model's row is replaced rather than duplicated. Both fail
silently when broken -- the board still renders, just with a number that is
flattering and wrong.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import pytest

from storage import benchmark_runs_db as db


@pytest.fixture(autouse = True)
def _isolated_db(tmp_path, monkeypatch):
    """A fresh database per test, so ordering between them cannot matter."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    db.get_connection().close()


def _run(
    run_id: str,
    *,
    model_id: str = "me/model",
    accuracy: float = 0.5,
    truncated: bool = False,
    mode: str = "logits",
    created_at: float = 1000.0,
    baseline: float = 0.25,
    keys: tuple[str, ...] = ("mmlu",),
    owner_subject: str = "admin",
) -> str:
    return db.save_run(
        {
            "run_id": run_id,
            "model_id": model_id,
            "model_label": model_id,
            "format": "safetensors",
            "status": "completed",
            "scoring_mode": mode,
            "requested": list(keys),
            "max_problems": 400,
            "started_at": created_at,
            "elapsed_seconds": 60.0,
            "scores": [
                {
                    "key": key,
                    "label": key,
                    "kind": "categorical",
                    "status": "complete",
                    "accuracy": accuracy,
                    "baseline": baseline,
                    "correct": 200,
                    "total": 400,
                    "truncated": truncated,
                }
                for key in keys
            ],
        },
        owner_subject = owner_subject,
    )


# --- centring ---


def test_a_centred_score_maps_guessing_to_zero_and_perfect_to_one():
    # The only figure comparable across benchmarks, so its ends have to be exact.
    for accuracy, expected in ((0.25, 0.0), (1.0, 1.0), (0.625, 0.5)):
        run = _run("centred", accuracy = accuracy, baseline = 0.25)
        assert db.get_run(run, owner_subject = "admin")["composite"] == pytest.approx(expected)


def test_a_zero_baseline_benchmark_is_centred_on_perfect():
    # A generative benchmark has no chance level, so its raw accuracy is already
    # the centred figure rather than something to divide.
    run = _run("zero", accuracy = 0.4, baseline = 0.0)
    assert db.get_run(run, owner_subject = "admin")["composite"] == pytest.approx(0.4)


# --- truncated runs ---


def test_a_truncated_score_gets_no_composite_however_high_it_is():
    # 0.99 from 400 of MMLU's 14,042 problems is a lucky sample, and ranking it
    # above a full-set 0.7 would be ranking sampling noise. It is stored, marked,
    # and left without a composite so it sorts last.
    run = _run("capped", accuracy = 0.99, truncated = True)
    stored = db.get_run(run, owner_subject = "admin")
    assert stored["composite"] is None
    assert stored["scores"][0]["accuracy"] == pytest.approx(0.99)
    assert stored["scores"][0]["truncated"] is True
    assert stored["scores"][0]["centered"] is None


def test_a_full_set_run_outranks_a_truncated_one_for_the_same_model():
    _run("capped", accuracy = 0.99, truncated = True, created_at = 2000.0)
    _run("full", accuracy = 0.30, truncated = False, created_at = 1000.0)
    best = db.best_per_model(["mmlu"], owner_subject = "admin")
    assert len(best) == 1
    assert best[0]["run_id"] == "full"


# --- one row per model ---


def test_a_model_appears_once_with_its_best_run():
    _run("first", model_id = "me/model", accuracy = 0.4, created_at = 1000.0)
    _run("second", model_id = "me/model", accuracy = 0.7, created_at = 2000.0)
    _run("third", model_id = "me/model", accuracy = 0.1, created_at = 3000.0)
    best = db.best_per_model(["mmlu"], owner_subject = "admin")
    assert len(best) == 1
    # The middle run: the best composite wins, and the newer worse run does not.
    assert best[0]["run_id"] == "second"


def test_a_re_run_replaces_the_model_row_rather_than_adding_one():
    # Re-running the same checkpoint with a larger cap should improve its row, not
    # leave two rows for one model.
    _run("a", model_id = "me/model", accuracy = 0.4, created_at = 1000.0)
    _run("b", model_id = "me/model", accuracy = 0.8, created_at = 2000.0)
    rows = db.best_per_model(["mmlu"], owner_subject = "admin")
    assert [row["run_id"] for row in rows] == ["b"]
    # Both runs are still in history; the board just shows the best.
    assert len(db.list_runs(owner_subject = "admin")) == 2


def test_two_models_are_both_kept():
    _run("a", model_id = "me/small", accuracy = 0.3)
    _run("b", model_id = "me/large", accuracy = 0.8)
    rows = db.best_per_model(["mmlu"], owner_subject = "admin")
    assert {row["model_id"] for row in rows} == {"me/small", "me/large"}


# --- scoring modes stay separate ---


def test_a_generated_run_is_not_merged_into_the_exact_ones():
    # The board groups by mode, so the storage has to keep them distinguishable.
    _run("exact", model_id = "me/exact", mode = "logits", accuracy = 0.5)
    _run("gen", model_id = "me/gen", mode = "generated", accuracy = 0.9)
    rows = {row["run_id"]: row for row in db.best_per_model(["mmlu"], owner_subject = "admin")}
    assert rows["exact"]["scoring_mode"] == "logits"
    assert rows["gen"]["scoring_mode"] == "generated"


# --- validation ---


def test_an_accuracy_outside_zero_to_one_is_dropped():
    # A score outside the range is a bug upstream. Storing it would put a number
    # on the board that no reader could interpret.
    run = _run("bad", accuracy = 1.4)
    stored = db.get_run(run, owner_subject = "admin")
    assert stored["scores"][0]["accuracy"] is None
    assert stored["composite"] is None


def test_a_run_with_no_usable_score_is_still_stored():
    # It happened, and history should say so. It just carries no composite, so it
    # cannot reach the top of the board.
    run = _run("empty", accuracy = 0.5)
    db.save_run(
        {
            "run_id": run,
            "model_id": "me/model",
            "model_label": "me/model",
            "status": "error",
            "requested": ["mmlu"],
            "started_at": 1.0,
            "scores": [],
        },
        owner_subject = "admin",
    )
    stored = db.get_run(run, owner_subject = "admin")
    assert stored["composite"] is None
    assert stored["scores"] == []


def test_a_model_with_no_requested_benchmark_is_left_off_the_board():
    # A run that touched none of the columns being shown has nothing to say about
    # them, and appears as a row of empty cells.
    _run("gsm", model_id = "me/math", keys = ("gsm8k",))
    assert db.best_per_model(["mmlu"], owner_subject = "admin") == []
    assert len(db.best_per_model(["gsm8k"], owner_subject = "admin")) == 1


# --- ownership and deletion ---


def test_one_account_cannot_read_or_delete_another_accounts_run():
    _run("mine", owner_subject = "admin")
    assert db.get_run("mine", owner_subject = "someone-else") is None
    assert db.delete_run("mine", owner_subject = "someone-else") is False
    assert db.get_run("mine", owner_subject = "admin") is not None


def test_deleting_a_run_takes_its_scores_with_it():
    # Through the foreign key, not by hand in the accessor: a score row whose run
    # is gone would be an orphan the board could still read.
    _run("doomed")
    assert db.delete_run("doomed", owner_subject = "admin") is True
    assert db.get_run("doomed", owner_subject = "admin") is None
    conn = db.get_connection()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM benchmark_scores WHERE run_id = ?", ("doomed",)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_deleting_an_unknown_run_reports_that_nothing_was_removed():
    assert db.delete_run("no-such-run", owner_subject = "admin") is False


def test_re_persisting_a_run_replaces_its_scores():
    # A run can be written twice (a stop after grading, then a completion). A
    # leftover score from the first attempt would put two numbers for one
    # benchmark on one row.
    run = _run("twice", keys = ("mmlu", "arc-easy"))
    assert len(db.get_run(run, owner_subject = "admin")["scores"]) == 2
    _run("twice", keys = ("mmlu",))
    stored = db.get_run(run, owner_subject = "admin")
    assert [score["key"] for score in stored["scores"]] == ["mmlu"]
