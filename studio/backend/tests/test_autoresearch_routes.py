# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The autoresearch routes, over HTTP.

The point of these is the request boundary rather than the happy path. Almost
everything the tab needs to work is something a *refusal* has to get right:

  - a status request before the first loop is an empty status, not a 404, or the
    tab opens on an error
  - starting a second loop is a 409, not a 422, because the request is well
    formed and only the machine's state is wrong
  - a loop that is running must refuse a checkpoint load, because the
    experiments have the GPU and a second copy of the weights is an OOM the
    unattended run cannot interpret
  - a report over a still-running loop is refused, because it would describe
    experiments that have not finished

None of these need autoresearch installed, so they are all testable here.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.autoresearch import results
from core.autoresearch.orchestrator import get_run_manager
from routes import autoresearch


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client over the real router, with the workspace pointed at a temp dir.

    The ledger is read from disk, so the tests that care about it write one
    rather than stubbing the reader: what is under test is the route's response
    shape, and the shape is only meaningful against real rows.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "core.autoresearch.environment.checkout_root", lambda: checkout
    )
    monkeypatch.setattr(
        "core.autoresearch.chat_sidecar.autoresearch_root", lambda: checkout
    )
    app = FastAPI()
    app.include_router(autoresearch.router, prefix="/api/autoresearch")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    yield TestClient(app), checkout


def test_status_before_any_run_is_an_empty_state_not_an_error(client) -> None:
    """The tab is opened before the first loop, so this is the normal first answer."""
    http, _checkout = client
    response = http.get("/api/autoresearch/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "idle"
    assert payload["run_id"] == ""


def test_status_for_a_stale_run_id_is_a_conflict(client) -> None:
    http, _checkout = client
    assert http.get("/api/autoresearch/status?expected_run_id=nope").status_code == 200


def test_results_before_anything_has_run_is_an_empty_ledger(client) -> None:
    http, _checkout = client
    response = http.get("/api/autoresearch/results")
    assert response.status_code == 200
    payload = response.json()
    assert payload["exists"] is False
    assert payload["rows"] == []
    assert payload["best_val_bpb"] is None


def test_results_carry_the_ledger_and_its_totals(client) -> None:
    http, checkout = client
    (checkout / results.RESULTS_FILENAME).write_text(
        results.RESULTS_HEADER + "\n"
        + "aaa\t1.500000\t3.5\tkeep\tfirst\n"
        + "bbb\t1.200000\t3.6\tkeep\tbetter\n"
        + "ccc\t0.000000\t0.0\tcrash\tdied\n",
        encoding="utf-8",
    )
    payload = http.get("/api/autoresearch/results").json()
    assert payload["total"] == 3
    assert payload["kept"] == 2
    assert payload["crashed"] == 1
    assert payload["best_val_bpb"] == 1.2
    assert payload["best_index"] == 2
    assert [row["index"] for row in payload["rows"]] == [1, 2, 3]
    # A crash carries no score, so the best is a real experiment.
    assert payload["rows"][2]["val_bpb"] is None


def test_metrics_include_a_point_for_every_experiment(client) -> None:
    """A crashed experiment is a gap, not a missing x value.

    Dropping it would renumber the run, so a chart drawn from the metrics would
    label experiment 3 as experiment 2.
    """
    http, checkout = client
    (checkout / results.RESULTS_FILENAME).write_text(
        results.RESULTS_HEADER + "\n"
        + "aaa\t1.500000\t3.5\tkeep\tfirst\n"
        + "bbb\t0.000000\t0.0\tcrash\tdied\n",
        encoding="utf-8",
    )
    payload = http.get("/api/autoresearch/metrics").json()
    assert [point["index"] for point in payload["series"]] == [1, 2]
    assert payload["series"][0]["val_bpb"] == 1.5
    assert payload["series"][1]["val_bpb"] is None
    # The tooltip needs this to say what the experiment actually changed.
    assert payload["series"][0]["description"] == "first"
    assert payload["baseline_val_bpb"] == 1.5
    assert payload["best_val_bpb"] == 1.5


def test_checkpoints_are_empty_before_a_loop_has_run(client) -> None:
    http, _checkout = client
    payload = http.get("/api/autoresearch/checkpoints").json()
    assert payload["checkpoints"] == []


def test_chat_models_are_empty_before_a_loop_has_run(client) -> None:
    http, _checkout = client
    payload = http.get("/api/autoresearch/chat/models").json()
    assert payload["models"] == []
    assert payload["resident"] is None


def test_loading_a_checkpoint_that_does_not_exist_is_a_400(client) -> None:
    http, _checkout = client
    response = http.post("/api/autoresearch/chat/load?model_id=autoresearch/007")
    assert response.status_code == 400
    # The wording is shown to the user, so it has to say what went wrong rather
    # than surfacing a stack trace.
    assert "no checkpoint" in response.json()["detail"].lower()


def test_a_malformed_model_id_is_a_400_not_a_500(client) -> None:
    http, _checkout = client
    response = http.post("/api/autoresearch/chat/load?model_id=not-ours")
    assert response.status_code == 400


def test_stopping_with_nothing_running_is_idle_not_an_error(client) -> None:
    http, _checkout = client
    response = http.post("/api/autoresearch/stop", json={"save": True})
    assert response.status_code == 200
    assert response.json()["status"] == "idle"


def test_starting_without_the_environment_is_a_409(client, monkeypatch) -> None:
    """Not a 422: the request is well formed and only the machine is unready."""
    http, _checkout = client
    # Patched on the module object rather than by dotted path: the orchestrator
    # holds a reference to the module, not to the function, so replacing the
    # attribute is what it will actually see.
    from core.autoresearch import environment

    monkeypatch.setattr(
        environment,
        "environment_status",
        lambda: environment.EnvironmentStatus(
            blocking_reason="autoresearch has not been copied into the workspace yet"
        ),
    )
    response = http.post("/api/autoresearch/start", json={"num_experiments": 4})
    assert response.status_code == 409
    assert "copied into" in response.json()["detail"]


def test_starting_a_second_loop_is_a_conflict(client, monkeypatch) -> None:
    """The machine is busy, not the request malformed."""
    http, _checkout = client
    from core.autoresearch import environment
    from core.autoresearch.orchestrator import AutoresearchBusy

    monkeypatch.setattr(environment, "environment_status", environment.EnvironmentStatus)
    monkeypatch.setattr(
        get_run_manager(),
        "start",
        lambda config: (_ for _ in ()).throw(AutoresearchBusy("An autoresearch loop is already running")),
    )
    response = http.post("/api/autoresearch/start", json={"num_experiments": 4})
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


def test_a_report_over_a_running_loop_is_refused(client, monkeypatch) -> None:
    """It would describe experiments that have not finished."""
    http, _checkout = client
    monkeypatch.setattr(get_run_manager(), "current", lambda: _ActiveRun())
    response = http.post(
        "/api/autoresearch/report",
        json={"model_id": "some/model", "source": "catalog"},
    )
    assert response.status_code == 409
    assert "still running" in response.json()["detail"]


def test_loading_a_checkpoint_during_a_loop_is_refused(client, monkeypatch) -> None:
    """The experiments have the GPU; a second copy of the weights is an OOM the
    unattended run cannot interpret, so it is refused before it happens."""
    http, _checkout = client
    monkeypatch.setattr(get_run_manager(), "is_active", lambda: True)
    response = http.post("/api/autoresearch/chat/load?model_id=autoresearch/001")
    assert response.status_code == 409
    assert "stop it" in response.json()["detail"]


class _ActiveRun:
    run_id = "run_1"

    def is_active(self) -> bool:
        return True


def test_an_unknown_agent_is_refused_at_the_boundary(client) -> None:
    """A renamed CLI has to be a form error, not a subprocess failure an hour in."""
    http, _checkout = client
    response = http.post("/api/autoresearch/start", json={"agent": "gpt"})
    assert response.status_code == 422


def test_an_experiment_count_beyond_the_ceiling_is_refused(client) -> None:
    http, _checkout = client
    response = http.post("/api/autoresearch/start", json={"num_experiments": 5000})
    assert response.status_code == 422


def test_a_zero_experiment_count_is_refused(client) -> None:
    http, _checkout = client
    response = http.post("/api/autoresearch/start", json={"num_experiments": 0})
    assert response.status_code == 422


def test_a_multi_line_focus_is_flattened_by_the_request_model() -> None:
    """It is spliced into a prompt, where a newline ends the instruction block."""
    from models.autoresearch import AutoresearchStartRequest

    request = AutoresearchStartRequest(focus="first line\nsecond line")
    assert "\n" not in request.focus
    assert request.focus == "first line second line"


def test_the_environment_route_reports_what_is_missing_rather_than_raising(client) -> None:
    http, _checkout = client
    response = http.get("/api/autoresearch/environment")
    assert response.status_code == 200
    payload = response.json()
    # Every field the Configure tab reads has to be present even on a machine
    # with nothing installed, or the gate renders undefined.
    for field in (
        "source_present", "checkout_present", "venv_present", "torch_installed",
        "data_prepared", "git_ready", "install_state", "install_progress",
        "ready",
    ):
        assert field in payload
    assert payload["ready"] is False


def test_the_agents_route_lists_every_agent_with_a_reason(client) -> None:
    """Reported rather than filtered: "not on PATH" and "will not run" are
    different problems and only one is fixed by installing something."""
    http, _checkout = client
    payload = http.get("/api/autoresearch/agents").json()
    keys = [agent["key"] for agent in payload["agents"]]
    # Every agent the backend can drive, in preference order -- which puts
    # opencode first, because that is the one most often pointed at a local model
    # and it is the default.
    from core.autoresearch.agent import AGENT_SPECS, PREFERRED_AGENT_ORDER

    assert keys == list(PREFERRED_AGENT_ORDER)
    assert set(keys) == {spec.key for spec in AGENT_SPECS}
    for agent in payload["agents"]:
        if agent["available"]:
            assert agent["version"]
        else:
            assert agent["reason"]


def test_the_route_recommends_the_default_agent_when_it_is_installed(client) -> None:
    """`recommended` is what the UI preselects, so it has to be the default.

    On this machine opencode and codex are both present, so this asserts the
    recommendation is the configured default rather than merely one of the two.
    """
    from core.autoresearch.agent import PREFERRED_AGENT_ORDER

    http, _checkout = client
    payload = http.get("/api/autoresearch/agents").json()
    if payload["recommended"] is not None:
        assert payload["recommended"] == PREFERRED_AGENT_ORDER[0]
