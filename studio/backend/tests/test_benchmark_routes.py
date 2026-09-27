# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Benchmark tab's HTTP surface.

What matters here is that the tab cannot ask for something that would produce a
misleading number: an unknown benchmark, one this host cannot run, or a suite
with nothing in it. Each of those is rejected at the edge with a message the
picker can show, rather than accepted and turned into a zero.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.benchmarks as routes_benchmarks
from storage import benchmark_runs_db as db

_KEYS = ("mmlu", "arc-easy", "arc-challenge", "gsm8k", "humaneval")


@pytest.fixture(autouse = True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    db.get_connection().close()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes_benchmarks.router, prefix = "/api/benchmarks")
    app.dependency_overrides[routes_benchmarks.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False)


def _save(
    run_id: str,
    *,
    model_id: str,
    label: str,
    scores: dict,
    truncated: bool = False,
    mode: str = "logits",
) -> None:
    db.save_run(
        {
            "run_id": run_id,
            "model_id": model_id,
            "model_label": label,
            "format": "safetensors",
            "status": "completed",
            "scoring_mode": mode,
            "requested": list(scores),
            "max_problems": 400,
            "started_at": 1000.0,
            "elapsed_seconds": 120.0,
            "scores": [
                {
                    "key": key,
                    "label": key,
                    "kind": "categorical",
                    "status": "complete",
                    "accuracy": value,
                    "baseline": 0.25,
                    "correct": 100,
                    "total": 400,
                    "truncated": truncated,
                }
                for key, value in scores.items()
            ],
        },
        owner_subject = "admin",
    )


# --- catalogue ---


def test_the_catalogue_always_answers(client):
    # Even with no nanochat environment the picker has to render, with a reason,
    # rather than failing the page.
    body = client.get("/api/benchmarks/benchmarks").json()
    assert "benchmarks" in body
    assert "default_benchmarks" in body
    assert "available" in body
    assert isinstance(body["posix_sandbox"], bool)


# --- start validation ---


def test_an_unknown_benchmark_is_refused_with_the_available_ones_named(client):
    response = client.post(
        "/api/benchmarks/start",
        json = {"model_id": "m", "model_path": "C:/m", "benchmarks": ["not-real"]},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "not-real" in detail
    # The picker shows the reason, so it has to say what the options are.
    assert "Available" in detail


def test_a_suite_with_nothing_selected_is_refused(client):
    response = client.post(
        "/api/benchmarks/start",
        json = {"model_id": "m", "model_path": "C:/m", "benchmarks": []},
    )
    assert response.status_code == 422


def test_an_unexpected_field_is_refused_rather_than_ignored(client):
    # extra="forbid", so a stale client cannot send a field this side no longer
    # honours and watch it look like it took effect.
    response = client.post(
        "/api/benchmarks/start",
        json = {
            "model_id": "m",
            "model_path": "C:/m",
            "benchmarks": ["mmlu"],
            "not_a_field": 1,
        },
    )
    assert response.status_code == 422


def test_an_out_of_range_problem_cap_is_refused(client):
    response = client.post(
        "/api/benchmarks/start",
        json = {
            "model_id": "m",
            "model_path": "C:/m",
            "benchmarks": ["mmlu"],
            "max_problems": 0,
        },
    )
    assert response.status_code == 422


# --- status ---


def test_status_is_a_404_when_no_run_has_ever_started(client):
    # Distinct from an idle state, so the tab can tell "superseded" from "never".
    assert client.get("/api/benchmarks/status").status_code == 404


# --- leaderboard ---


def test_the_leaderboard_is_empty_rather_than_absent(client):
    body = client.get("/api/benchmarks/leaderboard").json()
    assert body["rows"] == []


def test_a_local_run_and_a_published_score_are_both_listed_and_labelled(client):
    _save("mine", model_id = "me/m", label = "my model", scores = {"mmlu": 0.5})
    body = client.get("/api/benchmarks/leaderboard?benchmarks=mmlu").json()
    sources = {row["source"] for row in body["rows"]}
    assert "local" in sources
    # The published rows are the reason a first run has something to sit against.
    if "reference" in sources:
        reference = next(row for row in body["rows"] if row["source"] == "reference")
        # Every published figure says where it came from. A number with no
        # provenance on a leaderboard is indistinguishable from a measurement.
        assert reference["source_note"]
        assert reference["run_id"] is None


def test_published_scores_can_be_turned_off(client):
    _save("mine", model_id = "me/m", label = "my model", scores = {"mmlu": 0.5})
    body = client.get("/api/benchmarks/leaderboard?benchmarks=mmlu&include_references=false").json()
    assert all(row["source"] == "local" for row in body["rows"])


def test_a_local_run_is_ranked_above_a_published_one(client):
    # Same board, but the reader's own measurement leads: a published figure from
    # another machine should never outrank a measurement taken on this one.
    _save("mine", model_id = "me/m", label = "my model", scores = {"mmlu": 0.05})
    body = client.get("/api/benchmarks/leaderboard?benchmarks=mmlu").json()
    local = [index for index, row in enumerate(body["rows"]) if row["source"] == "local"]
    reference = [index for index, row in enumerate(body["rows"]) if row["source"] == "reference"]
    if local and reference:
        assert min(local) < min(reference)


def test_a_truncated_run_is_listed_and_marked_rather_than_hidden(client):
    # Hiding it would leave a user's only run invisible, which reads as a broken
    # feature. It carries no composite, so it cannot reach the top either.
    _save("capped", model_id = "me/m", label = "capped", scores = {"mmlu": 0.9}, truncated = True)
    body = client.get("/api/benchmarks/leaderboard?benchmarks=mmlu").json()
    row = next(row for row in body["rows"] if row["source"] == "local")
    assert row["scores"]["mmlu"] == pytest.approx(0.9)
    assert row["truncated_keys"] == ["mmlu"]
    assert row["composite"] is None


def test_the_board_is_projected_onto_the_requested_benchmarks_only(client):
    _save("both", model_id = "me/m", label = "both", scores = {"mmlu": 0.5, "gsm8k": 0.2})
    body = client.get("/api/benchmarks/leaderboard?benchmarks=mmlu").json()
    row = next(row for row in body["rows"] if row["source"] == "local")
    assert set(row["scores"]) == {"mmlu"}
    assert body["sort_keys"] == ["mmlu"]


def test_generated_and_exact_runs_are_never_ranked_as_one_group(client):
    _save("exact", model_id = "me/exact", label = "exact", scores = {"mmlu": 0.4}, mode = "logits")
    _save(
        "gen",
        model_id = "me/gen",
        label = "gen",
        scores = {"mmlu": 0.95},
        mode = "generated",
    )
    body = client.get("/api/benchmarks/leaderboard?benchmarks=mmlu").json()
    order = [row["run_id"] for row in body["rows"] if row["source"] == "local"]
    # The weaker method scored higher. Grouped, the exact measurement still leads,
    # because ranking them together would let the method decide the ranking.
    assert order == ["exact", "gen"]


# --- history and deletion ---


def test_history_lists_stored_runs(client):
    _save("one", model_id = "me/m", label = "one", scores = {"mmlu": 0.5})
    body = client.get("/api/benchmarks/history").json()
    assert [run["run_id"] for run in body["runs"]] == ["one"]
    assert body["runs"][0]["scores"][0]["key"] == "mmlu"


def test_deleting_an_unknown_run_is_a_404(client):
    assert client.delete("/api/benchmarks/runs/nope").status_code == 404


def test_a_run_can_be_removed_from_the_board(client):
    _save("bye", model_id = "me/m", label = "bye", scores = {"mmlu": 0.5})
    assert client.delete("/api/benchmarks/runs/bye").status_code == 200
    assert client.get("/api/benchmarks/leaderboard?benchmarks=mmlu").json()["rows"] == [
        row
        for row in client.get("/api/benchmarks/leaderboard?benchmarks=mmlu").json()["rows"]
        if row["source"] == "reference"
    ]


def test_a_running_run_cannot_be_deleted_out_from_under_itself(client, monkeypatch):
    # The orchestrator owns a live run and writes to its row; deleting it here
    # would leave the tab scoring into a run the board no longer has.
    class FakeRun:
        run_id = "live"
        status = "running"

        def is_active(self) -> bool:
            return True

    monkeypatch.setattr(
        routes_benchmarks,
        "_manager",
        lambda: type("M", (), {"current": staticmethod(lambda: FakeRun())})(),
    )
    response = client.delete("/api/benchmarks/runs/live")
    assert response.status_code == 409


# --- the run's own state ---


def test_every_score_reaches_the_api_with_its_centred_figure():
    """`centered` is the only per-benchmark figure comparable across benchmarks,
    so it has to survive the trip to the UI.

    It is a method on ScoreState rather than a field, so the obvious
    ``asdict(score)`` drops it. Nothing about that fails to compile or to render:
    the column just reads as blank forever, which is why it is pinned here rather
    than left to the shape of the dataclass.
    """
    from core.benchmarks.orchestrator import (
        BenchmarkConfig,
        BenchmarkRun,
        RunState,
        ScoreState,
    )

    state = RunState(run_id = "r", requested = ["mmlu"])
    state.scores = [
        ScoreState(
            key = "mmlu",
            label = "MMLU",
            status = "complete",
            accuracy = 0.625,
            baseline = 0.25,
            correct = 250,
            total = 400,
        )
    ]
    row = state.to_dict()["scores"][0]
    assert "centered" in row
    # (0.625 - 0.25) / 0.75 == 0.5: half the gap between guessing and perfect.
    assert row["centered"] == pytest.approx(0.5)

    # And the run's own headline figure travels for the same reason. It is a method
    # too, so the obvious spelling would leave the badge on the live view blank for
    # the entire run.
    assert "composite" in state.to_dict()
    assert state.to_dict()["composite"] == pytest.approx(0.5)

    # And for a run built the way the orchestrator builds one, so this is not only
    # about the hand-assembled state above.
    run = BenchmarkRun(
        run_id = "r2",
        config = BenchmarkConfig(
            model_id = "m",
            model_label = "L",
            model_path = "C:/m",
            benchmarks = ["mmlu"],
        ),
    )
    # The baseline is set here rather than left to the catalogue: where it comes
    # from depends on whether nanochat is installed, and this is about the
    # arithmetic reaching the API, not about where the chance level is read from.
    run.state.scores[0].status = "complete"
    run.state.scores[0].baseline = 0.25
    run.state.scores[0].accuracy = 0.25
    assert run.state.to_dict()["scores"][0]["centered"] == pytest.approx(0.0)


def test_a_truncated_score_reaches_the_api_without_a_centred_figure():
    from core.benchmarks.orchestrator import RunState, ScoreState

    state = RunState(run_id = "r", requested = ["mmlu"])
    state.scores = [
        ScoreState(key = "mmlu", status = "complete", accuracy = 0.9, baseline = 0.25, truncated = True)
    ]
    # A sample is an estimate, so it gets no centred figure and no composite. The
    # raw accuracy still travels, because the reader is entitled to see it.
    assert state.to_dict()["scores"][0]["centered"] is None
    assert state.to_dict()["scores"][0]["accuracy"] == pytest.approx(0.9)
    assert state.composite() is None


def test_an_uncapped_run_materializes_every_problem_rather_than_none():
    """A run with no problem cap must pass "no cap" down, not zero.

    The cap reaches the materializer as an optional flag, and sending 0 for it
    reads as a cap of zero: the pass writes no items, the run then fails with "no
    benchmark items could be prepared", and the message blames the wrong thing for
    a request that asked for everything.
    """
    from core.benchmarks.orchestrator import BenchmarkConfig, BenchmarkRun

    run = BenchmarkRun(
        run_id = "uncapped",
        config = BenchmarkConfig(
            model_id = "m",
            model_label = "L",
            model_path = "C:/m",
            benchmarks = ["mmlu"],
            max_problems = None,
        ),
    )
    commands: list[list[str]] = []
    run._run_child = lambda phase, command, **kwargs: commands.append(command) or 0
    run._read_manifest = lambda: {"items": {"mmlu": {"written": 14042}}, "failures": {}}

    assert run._pass_materialize() is True
    # The flag is absent, so the materializer falls back to the benchmark's own
    # size rather than being handed a cap of zero.
    assert "--max-problems" not in commands[0]
    # And the run is not reported as truncated, because nothing was cut short.
    assert run.state.scores[0].truncated is False


def test_a_materialize_failure_names_the_benchmark_that_failed():
    from core.benchmarks.orchestrator import BenchmarkConfig, BenchmarkRun

    run = BenchmarkRun(
        run_id = "failed",
        config = BenchmarkConfig(
            model_id = "m",
            model_label = "L",
            model_path = "C:/m",
            benchmarks = ["mmlu", "gsm8k"],
        ),
    )
    run._run_child = lambda phase, command, **kwargs: 1
    run._read_manifest = lambda: {
        # A previous run's benchmark is still in the shared cache, so the manifest
        # is not empty even though nothing this run asked for made it.
        "items": {"arc-easy": {"written": 10}},
        "failures": {"mmlu": "404 on download", "gsm8k": "404 on download"},
    }

    assert run._pass_materialize() is False
    # The user is told which benchmark and why, rather than being invited to
    # re-run and watch the same thing fail.
    assert "mmlu" in run.state.error
    assert "404 on download" in run.state.error


def test_a_capped_run_passes_its_cap_down():
    from core.benchmarks.orchestrator import BenchmarkConfig, BenchmarkRun

    run = BenchmarkRun(
        run_id = "capped",
        config = BenchmarkConfig(
            model_id = "m",
            model_label = "L",
            model_path = "C:/m",
            benchmarks = ["mmlu"],
            max_problems = 400,
        ),
    )
    commands: list[list[str]] = []
    run._run_child = lambda phase, command, **kwargs: commands.append(command) or 0
    run._read_manifest = lambda: {"items": {"mmlu": {"written": 400}}, "failures": {}}

    assert run._pass_materialize() is True
    index = commands[0].index("--max-problems")
    assert commands[0][index + 1] == "400"


def test_a_local_inventory_path_is_resolved_to_something_loadable(tmp_path):
    """The inventory reports where a model lives, not a directory a loader opens.

    For an HF cache entry the two differ: the repo directory has no
    ``config.json``, only its snapshot does. Handing the repo root to the loader
    fails with "Unrecognized model ... should have a model_type key", which is
    the regression this pins -- the entry was offered in the picker and failed
    only when the user pressed Start.
    """
    from pathlib import Path

    import routes.benchmarks as route

    # A cache-shaped tree: repo root with a blobs dir, and a snapshot holding the
    # config. The root on its own is not loadable, which is the whole point.
    repo = tmp_path / "models--someone--SomeModel"
    snapshot = repo / "snapshots" / "abc123"
    (repo / "blobs").mkdir(parents = True)
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text('{"model_type": "llama"}', encoding = "utf-8")

    assert not (repo / "config.json").is_file()
    assert route._is_loadable(str(repo)) is False
    assert route._is_loadable(str(snapshot)) is True


def test_a_directory_with_neither_manifest_is_not_offered(tmp_path):
    from routes.benchmarks import _is_loadable

    empty = tmp_path / "not-a-model"
    empty.mkdir()
    assert _is_loadable(str(empty)) is False
    assert _is_loadable("") is False
    # A missing path is not an error, just not offerable.
    assert _is_loadable(str(tmp_path / "gone")) is False


def test_the_composite_averages_only_the_scores_that_finished():
    from core.benchmarks.orchestrator import RunState, ScoreState

    state = RunState(run_id = "r", requested = ["mmlu", "gsm8k", "arc-easy"])
    state.scores = [
        ScoreState(key = "mmlu", status = "complete", accuracy = 0.625, baseline = 0.25),
        ScoreState(key = "gsm8k", status = "complete", accuracy = 0.4, baseline = 0.0),
        # Still running, so it is not in the average. Averaging a partial suite as
        # if it were the whole one is how a suite score stops meaning its label.
        ScoreState(key = "arc-easy", status = "running", accuracy = None, baseline = 0.25),
    ]
    assert state.composite() == pytest.approx((0.5 + 0.4) / 2)
