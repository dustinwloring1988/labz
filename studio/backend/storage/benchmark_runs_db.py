# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable storage for benchmark runs and their scores.

A benchmark result that only exists in the orchestrator's memory is not a
leaderboard entry, so a finished run is written here. The tables are declared in
``studio_db._ensure_schema`` alongside everything else; this module owns only the
accessors, which is the split the newer domains use.

Two things the leaderboard needs that are easy to get wrong, and so are stated
here rather than left to the query:

  * A truncated score is not a score. A run capped at 400 problems out of MMLU's
    14k has an estimate, not a measurement, so it is stored with ``truncated``
    set and excluded from composites. Ranking it alongside a full-set run would
    let a cheap run outrank a thorough one on a difference that is sampling
    noise.
  * ``logits`` and ``generated`` are different measurements. The board sorts
    within a mode and the mode travels with every row, so nothing is ever
    compared to something it should not be.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Optional

from core.training.account_jobs import account_is_retired
from storage.studio_db import get_connection as _studio_connection

# Only these are ever ranked. A run that errored has no scores to compare, and
# one that was stopped has whatever finished before the stop, which is real but
# is not what the user asked to measure.
RANKABLE_STATUSES = ("completed", "stopped")


def get_connection():
    if account_is_retired():
        raise RuntimeError("Account is retired")
    return _studio_connection()


def _centered(accuracy: Optional[float], baseline: float, truncated: bool) -> Optional[float]:
    """Share of the guess-to-perfect gap that was closed.

    The only figure comparable across benchmarks, so it is what a composite is
    the mean of. None when the run was truncated (it is an estimate, not a
    measurement) or when the baseline leaves no room to improve.
    """
    if accuracy is None or truncated:
        return None
    room = 1.0 - baseline
    if room <= 0:
        return None
    return (accuracy - baseline) / room


def save_run(state: dict, owner_subject: Optional[str] = None) -> str:
    """Write a finished run and its scores, replacing any earlier attempt.

    Scores go in one transaction with the run. A run row without its scores would
    appear on the board as a model that scored nothing, which reads as a
    measurement rather than as a write that did not finish.
    """
    run_id = str(state.get("run_id") or "")
    if not run_id:
        raise ValueError("a run needs an id")

    scores = [row for row in (state.get("scores") or []) if isinstance(row, dict)]
    usable = [
        row
        for row in scores
        if row.get("status") == "complete" and _valid_accuracy(row.get("accuracy"))
    ]
    centered = [
        _centered(
            _valid_accuracy(row.get("accuracy")),
            float(row.get("baseline") or 0.0),
            bool(row.get("truncated")),
        )
        for row in usable
    ]
    centered_values = [value for value in centered if value is not None]
    composite = sum(centered_values) / len(centered_values) if centered_values else None

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO benchmark_runs (
                id, owner_subject, model_id, model_label, model_format, lora_path,
                load_in_4bit, status, scoring_mode, benchmarks_json, max_problems,
                composite, error, created_at, duration_seconds,
                judge_model_id, judge_model_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                model_label = excluded.model_label,
                status = excluded.status,
                scoring_mode = excluded.scoring_mode,
                benchmarks_json = excluded.benchmarks_json,
                max_problems = excluded.max_problems,
                composite = excluded.composite,
                error = excluded.error,
                duration_seconds = excluded.duration_seconds,
                judge_model_id = excluded.judge_model_id,
                judge_model_label = excluded.judge_model_label
            """,
            (
                run_id,
                owner_subject,
                str(state.get("model_id") or ""),
                str(state.get("model_label") or run_id),
                str(state.get("format") or "safetensors"),
                state.get("lora_path"),
                1 if state.get("load_in_4bit") else 0,
                str(state.get("status") or "completed"),
                state.get("scoring_mode"),
                json.dumps(list(state.get("requested") or []), ensure_ascii = False),
                state.get("max_problems"),
                composite,
                state.get("error"),
                float(state.get("started_at") or time.time()),
                float(state.get("elapsed_seconds") or 0.0),
                state.get("judge_model_id"),
                state.get("judge_model_label"),
            ),
        )
        # Replaced wholesale rather than merged: a re-persisted run is a
        # correction, and leaving a score from the earlier attempt behind would
        # put two numbers for one benchmark on one row.
        conn.execute("DELETE FROM benchmark_scores WHERE run_id = ?", (run_id,))
        conn.executemany(
            """
            INSERT INTO benchmark_scores (
                run_id, benchmark_key, label, kind, status, accuracy, baseline,
                centered, correct, total, elapsed_seconds, truncated, scoring_mode, error,
                judge_model, exact_accuracy, judge_accuracy, judge_judged,
                judge_unreadable, disagreements, judge_raised
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    str(row.get("key") or ""),
                    str(row.get("label") or row.get("key") or ""),
                    str(row.get("kind") or "categorical"),
                    str(row.get("status") or "pending"),
                    _valid_accuracy(row.get("accuracy")),
                    float(row.get("baseline") or 0.0),
                    _centered(
                        _valid_accuracy(row.get("accuracy")),
                        float(row.get("baseline") or 0.0),
                        bool(row.get("truncated")),
                    ),
                    int(row.get("correct") or 0),
                    int(row.get("total") or 0),
                    float(row.get("elapsed_seconds") or 0.0),
                    1 if row.get("truncated") else 0,
                    row.get("scoring_mode") or state.get("scoring_mode"),
                    row.get("error"),
                    # The judge's verdict and the exact one both travel, on every
                    # judged row. Collapsing them to one column here would be the
                    # point of the whole feature lost.
                    row.get("judge_model"),
                    _valid_accuracy(row.get("exact_accuracy")),
                    _valid_accuracy(row.get("judge_accuracy")),
                    int(row.get("judge_judged") or 0),
                    int(row.get("judge_unreadable") or 0),
                    int(row.get("disagreements") or 0),
                    int(row.get("judge_raised") or 0),
                )
                for row in scores
                if row.get("key")
            ],
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    return run_id


def _valid_accuracy(value: Any) -> Optional[float]:
    """An accuracy in 0..1, or None.

    A score outside that range is a bug upstream, and storing it would put a
    number on the leaderboard that no reader could interpret.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    if result < 0.0 or result > 1.0:
        return None
    return result


def _scores_for(conn: sqlite3.Connection, run_ids: list[str]) -> dict[str, list[dict]]:
    if not run_ids:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"SELECT * FROM benchmark_scores WHERE run_id IN ({placeholders}) ORDER BY benchmark_key",
        run_ids,
    ).fetchall()
    grouped: dict[str, list[dict]] = {run_id: [] for run_id in run_ids}
    for row in rows:
        grouped.setdefault(row["run_id"], []).append(
            {
                "key": row["benchmark_key"],
                "label": row["label"],
                "kind": row["kind"],
                "status": row["status"],
                "accuracy": row["accuracy"],
                "baseline": row["baseline"],
                "centered": row["centered"],
                "correct": row["correct"],
                "total": row["total"],
                "elapsed_seconds": row["elapsed_seconds"],
                "truncated": bool(row["truncated"]),
                "scoring_mode": row["scoring_mode"],
                "error": row["error"],
                "judge_model": row["judge_model"],
                "exact_accuracy": row["exact_accuracy"],
                "judge_accuracy": row["judge_accuracy"],
                "judge_judged": row["judge_judged"],
                "judge_unreadable": row["judge_unreadable"],
                "disagreements": row["disagreements"],
                "judge_raised": row["judge_raised"],
            }
        )
    return grouped


def _run_from_row(row: sqlite3.Row, scores: list[dict]) -> dict:
    return {
        "run_id": row["id"],
        "model_id": row["model_id"],
        "model_label": row["model_label"],
        "format": row["model_format"],
        "lora_path": row["lora_path"],
        "load_in_4bit": bool(row["load_in_4bit"]),
        "status": row["status"],
        "scoring_mode": row["scoring_mode"],
        # Named on the run so a leaderboard row from last month says which model
        # produced its generative figures. A judged number with no author is not
        # a number anyone can compare against a new one.
        "judge_model_id": row["judge_model_id"],
        "judge_model_label": row["judge_model_label"],
        "benchmarks_run": json.loads(row["benchmarks_json"] or "[]"),
        "max_problems": row["max_problems"],
        "composite": row["composite"],
        "error": row["error"],
        "created_at": row["created_at"],
        "duration_seconds": row["duration_seconds"],
        "scores": scores,
    }


def list_runs(*, limit: int = 50, owner_subject: Optional[str] = None) -> list[dict]:
    """Stored runs, newest first."""
    conn = get_connection()
    try:
        sql = "SELECT * FROM benchmark_runs"
        params: list[Any] = []
        if owner_subject is not None:
            sql += " WHERE owner_subject = ? OR owner_subject IS NULL"
            params.append(owner_subject)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))

        rows = conn.execute(sql, params).fetchall()
        scores = _scores_for(conn, [row["id"] for row in rows])
        return [_run_from_row(row, scores.get(row["id"], [])) for row in rows]
    finally:
        conn.close()


def get_run(run_id: str, *, owner_subject: Optional[str] = None) -> Optional[dict]:
    conn = get_connection()
    try:
        sql = "SELECT * FROM benchmark_runs WHERE id = ?"
        params: list[Any] = [run_id]
        if owner_subject is not None:
            sql += " AND (owner_subject = ? OR owner_subject IS NULL)"
            params.append(owner_subject)
        row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return _run_from_row(row, _scores_for(conn, [run_id]).get(run_id, []))
    finally:
        conn.close()


def delete_run(run_id: str, *, owner_subject: Optional[str] = None) -> bool:
    conn = get_connection()
    try:
        sql = "DELETE FROM benchmark_runs WHERE id = ?"
        params: list[Any] = [run_id]
        if owner_subject is not None:
            sql += " AND (owner_subject = ? OR owner_subject IS NULL)"
            params.append(owner_subject)
        cursor = conn.execute(sql, params)
        # The score rows go with it through ON DELETE CASCADE, which is why they
        # are declared with a foreign key rather than deleted by hand here.
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def best_per_model(benchmark_keys: list[str], *, owner_subject: Optional[str] = None) -> list[dict]:
    """The best stored run for each model, for the leaderboard.

    "Best" is the highest composite, and a model is identified by its id rather
    than its label so that re-running a checkpoint improves its row instead of
    adding a second one. Ties and the truncated-score exclusion both fall out of
    the query: a run with no untruncated score has a NULL composite and sorts
    last, so it is present but never on top.
    """
    wanted = list(benchmark_keys)
    conn = get_connection()
    try:
        sql = "SELECT * FROM benchmark_runs"
        params: list[Any] = []
        if owner_subject is not None:
            sql += " WHERE (owner_subject = ? OR owner_subject IS NULL)"
            params.append(owner_subject)
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []

        scores = _scores_for(conn, [row["id"] for row in rows])
        best: dict[str, tuple[dict, list[dict]]] = {}
        for row in rows:
            run = _run_from_row(row, scores.get(row["id"], []))
            identity = run["model_id"]
            current = best.get(identity)
            if current is None or _better(run, current[0]):
                best[identity] = (run, run["scores"])

        runs = [entry[0] for entry in best.values()]
        if wanted:
            # A run that touched none of the requested benchmarks has nothing to
            # say about this board, so it is left off rather than shown as a row
            # of empty cells.
            runs = [run for run in runs if any(score["key"] in wanted for score in run["scores"])]
        runs.sort(key = lambda run: (run["composite"] is None, -(run["composite"] or 0.0)))
        return runs
    finally:
        conn.close()


def _better(candidate: dict, incumbent: dict) -> bool:
    """Whether a run should replace another as a model's best.

    A run with a composite beats one without. Two with composites are compared on
    it, and a tie goes to the newer run, because re-running the same checkpoint
    with a larger problem cap is the usual reason for the tie and the larger cap
    is the better measurement.
    """
    left, right = candidate.get("composite"), incumbent.get("composite")
    if (left is None) != (right is None):
        return left is not None
    if left is None:
        return False
    if left != right:
        return left > right
    return float(candidate.get("created_at") or 0.0) > float(incumbent.get("created_at") or 0.0)
