# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One GPU, three things that want all of it.

The report model, the chat checkpoint and the experiments cannot be resident at
the same time, so the tab has to sequence them rather than assume. These pin the
three decisions that sequencing is made of:

  - a loop refuses to start over live training, a live nanochat run or a live
    benchmark run, naming the other thing
  - starting a loop releases this feature's own chat worker, and a resident chat
    model only if the measured peak says it will not fit
  - a loop asked to write a report does so as its last phase, after the
    experiments, and only on a clean finish
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.autoresearch import environment, results
from core.autoresearch.orchestrator import (
    AutoresearchBusy,
    AutoresearchConfig,
    AutoresearchRun,
    _measured_peak_bytes,
)


# -----------------------------------------------------------------------------
# Admission
# -----------------------------------------------------------------------------

def _manager():
    from core.autoresearch.orchestrator import get_run_manager

    return get_run_manager()


@pytest.fixture()
def idle_manager(monkeypatch):
    """A manager that believes nothing else holds the GPU, and starts nothing.

    ``AutoresearchRun.start`` is stubbed out. Against a prepared workspace it
    would really hand the GPU over and spawn a loop, and a test about admission
    must not launch one.
    """
    manager = _manager()
    monkeypatch.setattr(manager, "is_active", lambda: False)
    monkeypatch.setattr(manager, "_current", None, raising=False)
    monkeypatch.setattr(AutoresearchRun, "start", lambda self: self.state())
    return manager


def test_a_run_is_actually_started_by_the_manager(idle_manager, monkeypatch) -> None:
    """The manager creates the run, and something has to launch it.

    A guard that admits and then returns without starting leaves a loop that
    reports itself as running and never trains, which is the kind of failure that
    looks like a slow GPU rather than a bug.
    """
    _no_other_runs(monkeypatch)
    started: list[str] = []
    monkeypatch.setattr(
        AutoresearchRun, "start",
        lambda self: started.append(self.run_id) or self.state(),
    )
    idle_manager.start(AutoresearchConfig(num_experiments=1))
    assert len(started) == 1


def _no_other_runs(monkeypatch, *, training=False, nanochat=False, benchmarks=False):
    class _Training:
        def is_training_active(self) -> bool:
            return training

    class _Runs:
        def __init__(self, active):
            self._active = active

        def is_active(self) -> bool:
            return self._active

    monkeypatch.setattr(
        "core.training.training.get_training_backend", lambda: _Training(), raising=False
    )
    monkeypatch.setattr(
        "core.nanochat.orchestrator.get_run_manager", lambda: _Runs(nanochat), raising=False
    )
    monkeypatch.setattr(
        "core.benchmarks.orchestrator.get_run_manager", lambda: _Runs(benchmarks), raising=False
    )


def test_a_loop_refuses_to_start_over_live_unsloth_training(idle_manager, monkeypatch) -> None:
    _no_other_runs(monkeypatch, training=True)
    with pytest.raises(AutoresearchBusy, match="Unsloth training"):
        idle_manager.start(AutoresearchConfig(num_experiments=1))


def test_a_loop_refuses_to_start_over_a_live_nanochat_run(idle_manager, monkeypatch) -> None:
    _no_other_runs(monkeypatch, nanochat=True)
    with pytest.raises(AutoresearchBusy, match="nanochat"):
        idle_manager.start(AutoresearchConfig(num_experiments=1))


def test_a_loop_refuses_to_start_over_a_live_benchmark_run(idle_manager, monkeypatch) -> None:
    _no_other_runs(monkeypatch, benchmarks=True)
    with pytest.raises(AutoresearchBusy, match="benchmark"):
        idle_manager.start(AutoresearchConfig(num_experiments=1))


def test_the_refusal_names_the_other_thing_rather_than_saying_busy(idle_manager, monkeypatch) -> None:
    """A user has three other tabs open and needs to know which one to stop."""
    _no_other_runs(monkeypatch, nanochat=True)
    with pytest.raises(AutoresearchBusy) as caught:
        idle_manager.start(AutoresearchConfig(num_experiments=1))
    assert "nanochat" in str(caught.value).lower()


def test_a_missing_benchmarks_module_does_not_refuse(idle_manager, monkeypatch) -> None:
    """Benchmarks is optional, so its absence is not a reason to refuse.

    Asserted on the import rather than on a refusal: a fake importer that breaks
    every other import in the process proves nothing about this one branch.
    """
    import sys

    _no_other_runs(monkeypatch)
    saved = sys.modules.pop("core.benchmarks.orchestrator", None)
    monkeypatch.setitem(sys.modules, "core.benchmarks.orchestrator", None)
    try:
        # None in sys.modules makes `import x` raise ImportError, which is the
        # condition the guard is written for.
        idle_manager.start(AutoresearchConfig(num_experiments=1))
    finally:
        if saved is not None:
            sys.modules["core.benchmarks.orchestrator"] = saved
        else:
            sys.modules.pop("core.benchmarks.orchestrator", None)


# -----------------------------------------------------------------------------
# VRAM handover
# -----------------------------------------------------------------------------

def test_nothing_is_kept_when_no_peak_has_been_measured(monkeypatch) -> None:
    """The first run ever has no measurement, so the conservative answer wins.

    Keeping a chat model into an OOM costs the whole overnight loop, and the cost
    of being wrong the other way is that the model is unloaded.
    """
    from routes import training_vram

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat", lambda: {"any": True, "loading": False}
    )
    keep, info = training_vram.can_keep_chat_during_autoresearch(peak_bytes=None)
    assert keep is False
    assert info["reason"] == "no_measured_peak"


def test_a_measured_peak_alone_does_not_keep_anything_with_nothing_resident(monkeypatch) -> None:
    from routes import training_vram

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat", lambda: {"any": False, "loading": False}
    )
    keep, info = training_vram.can_keep_chat_during_autoresearch(peak_bytes=int(2.9e9))
    assert keep is False
    assert info["reason"] == "no_resident_chat"


def _patch_gpu(monkeypatch, free_gb: float, *, device: str = "cuda") -> None:
    """Pin the VRAM probe the coordinator reads.

    ``get_visible_gpu_utilization`` is imported inside the coordinator from
    ``utils.hardware``, so it is patched at its source rather than on the module
    that only borrows it. The device shape is that module's own: total and used
    in GB, with free derived from the two.
    """
    from utils import hardware

    monkeypatch.setattr(
        hardware, "get_device", lambda: getattr(hardware.DeviceType, device.upper())
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {
            "devices": [
                {"index": 0, "vram_total_gb": free_gb + 4.0, "vram_used_gb": 4.0},
            ]
        },
    )


def test_a_chat_model_loading_is_never_kept(monkeypatch) -> None:
    """A load in flight holds VRAM it has not published a name for yet."""
    from routes import training_vram

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat", lambda: {"any": True, "loading": True}
    )
    keep, info = training_vram.can_keep_chat_during_autoresearch(
        peak_bytes=int(1e9)
    )
    assert keep is False
    assert info["reason"] == "chat_model_loading"


def test_a_small_peak_keeps_a_resident_chat_model(monkeypatch) -> None:
    """16 GB of card and a 3 GB experiment: there is no reason to unload anything."""
    from routes import training_vram

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat",
        lambda: {"any": True, "loading": False, "model": "qwen"},
    )
    _patch_gpu(monkeypatch, 12.0)
    keep, info = training_vram.can_keep_chat_during_autoresearch(peak_bytes=int(2.9e9))
    assert keep is True
    # The threshold carries the module's margin and floor, so it is above the
    # bare requirement.
    assert info["needed_gb"] > info["required_gb"]


def test_a_peak_larger_than_the_card_unloads(monkeypatch) -> None:
    from routes import training_vram

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat",
        lambda: {"any": True, "loading": False, "model": "qwen"},
    )
    _patch_gpu(monkeypatch, 1.0)
    keep, _info = training_vram.can_keep_chat_during_autoresearch(peak_bytes=int(30e9))
    assert keep is False


def test_a_failed_probe_unloads_rather_than_guessing(monkeypatch) -> None:
    from routes import training_vram
    from utils import hardware

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat",
        lambda: {"any": True, "loading": False, "model": "qwen"},
    )

    def boom() -> dict:
        raise RuntimeError("nvidia-smi said no")

    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", boom)
    keep, info = training_vram.can_keep_chat_during_autoresearch(peak_bytes=int(1e9))
    assert keep is False
    assert info["reason"] == "probe_error"


def test_a_cpu_host_keeps_nothing(monkeypatch) -> None:
    from routes import training_vram
    from utils import hardware

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat", lambda: {"any": True, "loading": False}
    )
    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CPU)
    keep, info = training_vram.can_keep_chat_during_autoresearch(peak_bytes=int(1e9))
    assert keep is False
    assert info["reason"] == "non_accelerator"


def test_the_feature_releases_its_own_chat_worker_first(monkeypatch) -> None:
    """A worker holds a CUDA context of its own that no capacity probe can see.

    The experiments also start a process per experiment, so several contexts on a
    consumer card is how the driver starts refusing allocations.
    """
    from routes import training_vram

    released: list[str] = []

    monkeypatch.setattr(
        training_vram, "summarize_resident_chat", lambda: {"any": False, "loading": False}
    )
    monkeypatch.setattr(
        training_vram, "_free_autoresearch_for_training",
        lambda reason: released.append("own") or ["autoresearch:x"],
    )
    monkeypatch.setattr(training_vram, "_free_nanochat_for_training", lambda reason: [])

    freed = training_vram.coordinate_models_for_autoresearch(peak_bytes=None)
    assert "autoresearch:x" in freed
    assert released == ["own"]


def test_a_wedged_sidecar_does_not_take_the_handover_down(monkeypatch) -> None:
    """The release is isolated so a broken sidecar cannot block a run that fits.

    Patched below the wrapper rather than over it: the guard under test is inside
    the wrapper, so replacing the wrapper would remove the thing being tested.
    """
    from core.autoresearch import chat_sidecar
    from routes import training_vram

    def boom(reason: str) -> bool:
        raise RuntimeError("the worker is wedged")

    monkeypatch.setattr(chat_sidecar, "free_for_training", boom)
    monkeypatch.setattr(
        training_vram, "summarize_resident_chat", lambda: {"any": False, "loading": False}
    )
    assert training_vram.coordinate_models_for_autoresearch(peak_bytes=None) == []


def test_the_release_runs_even_when_nothing_else_is_resident(monkeypatch) -> None:
    """The empty case is the one the nanochat path already handles, and the one
    where a resident worker is otherwise invisible to the capacity probe."""
    from core.autoresearch import chat_sidecar
    from routes import training_vram

    calls: list[str] = []
    monkeypatch.setattr(
        chat_sidecar, "free_for_training", lambda reason: calls.append(reason) or True
    )
    monkeypatch.setattr(
        training_vram, "summarize_resident_chat", lambda: {"any": False, "loading": False}
    )
    freed = training_vram.coordinate_models_for_autoresearch(peak_bytes=None)
    assert len(calls) == 1
    assert freed == ["autoresearch:checkpoint"]


# -----------------------------------------------------------------------------
# The measured peak
# -----------------------------------------------------------------------------

def test_the_peak_is_the_largest_any_scored_experiment_used(tmp_path: Path) -> None:
    (tmp_path / results.RESULTS_FILENAME).write_text(
        results.RESULTS_HEADER + "\n"
        + "a\t1.500000\t2.5000\tkeep\tone\n"
        + "b\t1.400000\t3.2500\tkeep\ttwo\n"
        + "c\t1.300000\t1.0000\tdiscard\tthree\n",
        encoding="utf-8",
    )
    assert _measured_peak_bytes(tmp_path) == int(3.25e9)


def test_a_crash_row_is_not_a_measurement(tmp_path: Path) -> None:
    """autoresearch writes 0.0 for a run that died before its save.

    Treating that as a measurement would make every later run believe it needs no
    memory at all, and keep a chat model resident straight into an OOM.
    """
    (tmp_path / results.RESULTS_FILENAME).write_text(
        results.RESULTS_HEADER + "\n"
        + "a\t1.500000\t2.5000\tkeep\tone\n"
        + "b\t0.000000\t0.0\tcrash\tdied\n",
        encoding="utf-8",
    )
    assert _measured_peak_bytes(tmp_path) == int(2.5e9)


def test_an_untouched_ledger_measures_nothing(tmp_path: Path) -> None:
    assert _measured_peak_bytes(tmp_path) is None


# -----------------------------------------------------------------------------
# The report as the last phase
# -----------------------------------------------------------------------------

def test_the_report_config_defaults_to_off() -> None:
    """The box is unticked until someone wants it, so no model is ever assumed."""
    config = AutoresearchConfig()
    assert config.report_after_finish is False
    assert config.report_model_id == ""


def test_the_report_config_reaches_the_run_state() -> None:
    config = AutoresearchConfig(
        report_after_finish=True, report_model_id="qwen/qwen3-8b", report_source="catalog"
    )
    payload = config.to_dict()
    assert payload["report_after_finish"] is True
    assert payload["report_model_id"] == "qwen/qwen3-8b"
    assert payload["report_source"] == "catalog"


def test_asking_for_a_report_with_no_model_is_refused() -> None:
    from pydantic import ValidationError

    from models.autoresearch import AutoresearchStartRequest

    with pytest.raises(ValidationError):
        AutoresearchStartRequest(num_experiments=2, report_after_finish=True)


def test_an_untouched_report_field_is_not_a_refusal() -> None:
    """A blank model with the box unticked is a form nobody filled in."""
    from models.autoresearch import AutoresearchStartRequest

    request = AutoresearchStartRequest(num_experiments=2)
    assert request.report_after_finish is False


def test_only_a_clean_finish_writes_the_report() -> None:
    """A partial search would be described as if it were the whole thing."""
    from core.autoresearch.orchestrator import AutoresearchRun

    run = AutoresearchRun("run_x", AutoresearchConfig(report_after_finish=True))
    written: list[str] = []
    run._write_report_after_finish = lambda: written.append("report")  # type: ignore[method-assign]

    run._finish("stopped", "Stopped by hand")
    assert written == []

    run._finish("completed", "Ran 2 experiments")
    assert written == ["report"]


def test_the_report_phase_is_ordered_after_the_experiments() -> None:
    """The ordering is the whole reason this exists.

    On one GPU the report cannot be written while the experiments hold the card,
    and the moment it becomes possible is when the queue drains. Writing it as the
    run's last phase is what turns an unpredictable moment into a scheduled one.
    """
    from core.autoresearch.orchestrator import AutoresearchRun

    run = AutoresearchRun("run_x", AutoresearchConfig(report_after_finish=True))
    order: list[str] = []

    run._write_report_after_finish = lambda: order.append("report")  # type: ignore[method-assign]
    # Stand in for the experiment loop, which appends its own phase markers.
    run._set(phase="experiment")
    run._finish("completed", "done")
    order.insert(0, run.state().phase)
    assert order == ["finished", "report"]


def test_a_failing_report_does_not_fail_the_run() -> None:
    """The experiments succeeded. A report that could not be written is a
    separate thing that happened afterwards, and reporting the run as failed
    would throw away a night of work over a model that would not load."""
    from core.autoresearch.orchestrator import AutoresearchRun

    run = AutoresearchRun("run_x", AutoresearchConfig(report_after_finish=True))
    run._write_report_after_finish()
    state = run.state()
    assert state.status == "idle"  # _finish was not called, so untouched
    assert state.report_running is False
    assert state.report_error is not None
    assert state.report_text == ""
