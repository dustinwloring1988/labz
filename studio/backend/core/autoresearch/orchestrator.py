# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Running N autoresearch experiments, one agent invocation each.

The shape of a run here is a chain of separate processes, and that is deliberate
rather than incidental. Each experiment is a coding agent that edits one file,
trains a model for five minutes, reads a number, and commits or reverts. None of
that can happen in this process, and a run can be twelve experiments long — hours
of unattended work — so it must survive the studio being closed and reopened.

The parts:

  - ``AutoresearchRun`` owns the sequencer thread. For each experiment it starts
    the agent, supervises it, harvests the outcome, and records it.
  - The agent reports on stdout; the outcome is read from ``run.log``, which the
    project's own program.md says is the source of truth. Nothing here parses
    agent prose for a score, because a reworded log line upstream must not be
    able to change what the results table says.
  - Run state lives in memory and the SSE stream polls it, the same arrangement
    nanochat uses: the state is owned by a sequencer thread in a different
    process boundary, so being pushed to would mean the producer knowing about
    the consumer. A JSONL mirror is written alongside it as a diagnostic, since
    an overnight run is exactly the kind of thing a user wants to read after the
    fact rather than only on screen.

What the studio owns and the agent does not: the count, the stop button, the
ledger, and the checkpoints. What the agent owns: the idea, the edit, the
training run, and the keep-or-discard call. Splitting it there is what keeps the
loop honest — the studio records what happened without interpreting it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.autoresearch import agent as agent_mod
from core.autoresearch import environment, results
from utils.paths.storage_roots import account_path
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    is_process_shutting_down,
    terminate_pid,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# How often the event tail is read.
_READ_POLL_SECONDS = 0.4
# Console lines kept in memory and per stream frame.
_MAX_LOG_LINES = 4000
_MAX_LOG_LINES_PER_FRAME = 200
# How much of the agent's own stderr is kept for explaining a failure. Enough for
# the line that names the cause, which in every failure seen so far is the last
# ERROR or a usage line.
_MAX_AGENT_TAIL_LINES = 40
# A line worth quoting at the user: the ones an agent CLI writes when it is
# refusing to do the work, as opposed to the banner and session preamble it
# writes while agreeing to.
_AGENT_ERROR_MARKERS = ("error", "fatal", "failed", "not found", "not supported", "denied")
# How many times the same crash reason may repeat before the run gives up on it.
#
# Separate from `max_consecutive_failures` because the two answer different
# questions. That setting asks "has the machine stopped working", where any
# failure counts and giving up early would throw away real experiments. This asks
# "is another attempt going to say the same thing", where a verbatim repeat is not
# a result at all -- a refused model, a missing dataset, a bad path -- and paying
# for it again teaches nothing.
#
# Two, not three: one failure can be a genuine one-off, and by the second the
# machine, the brief and the agent have all been ruled out by elimination. A user
# who wants to keep going starts another run, which is cheap compared to the
# eleven minutes an experiment costs.
REPEATED_CRASH_LIMIT = 2
# How long a process is given to exit after a stop before it is killed. The
# agent is mid-write to train.py at this point, so the grace period is long
# enough for it to notice and clean up.
_STOP_GRACE_SECONDS = 90
_KILL_TIMEOUT_SECONDS = 20

# The phases a run moves through, as data rather than as branches, so the UI can
# render the pipeline without the orchestrator and the UI having to agree on an
# ordering.
PHASE_PREPARING = "preparing"
PHASE_EXPERIMENT = "experiment"
PHASE_FINISHED = "finished"


class AutoresearchBusy(RuntimeError):
    """Raised when a run is requested while one is already going."""


class AutoresearchNotInstalled(RuntimeError):
    """Raised when a run is requested before the environment exists."""


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class AutoresearchConfig:
    """What the user chose on the Configure tab."""

    # The headline setting: how many experiments to run.
    num_experiments: int = 8
    # Which coding agent runs each experiment.
    #
    # opencode, not codex: it is the one of the three most often pointed at a
    # local model, because its provider configuration is where a user sets one up.
    # A loop measuring one specific machine should default to running on that
    # machine's own weights rather than on a cloud account.
    agent: str = "opencode"
    # Extra arguments passed to the agent CLI, verbatim.
    #
    # This is the lever for pointing the experiment loop at a local model. The
    # agent is a separate program with its own model setting that the studio
    # cannot know, and a loop that trains on a cloud model while the user
    # believes it is running on theirs is worse than one that does not run. Each
    # CLI spells this differently -- `--model`, a provider flag, a base URL -- so
    # the field takes whatever that CLI accepts rather than guessing.
    agent_args: list[str] = field(default_factory = list)
    # Whether the agent may act without asking. Needed for unattended operation
    # and therefore on by default, but surfaced in the UI rather than assumed.
    skip_permissions: bool = True
    # Per-experiment wall clock for the agent, including its training run.
    agent_timeout_seconds: int = agent_mod.DEFAULT_AGENT_TIMEOUT_SECONDS
    # Stop the loop after this many consecutive failures. Zero means never, which
    # is the honest default: a crash is information, and a loop that gives up
    # after two of them cannot tell a bad idea from a bad machine.
    max_consecutive_failures: int = 0
    # Whether to prepare the dataset and tokenizer before the first experiment.
    prepare_data: bool = True
    # A one-line steer for the agent, prepended to program.md's instructions.
    focus: str = ""
    # Keep going when an experiment is stopped by hand rather than treating it as
    # the end of the run.
    continue_on_failure: bool = True
    # Write the report as soon as the loop finishes, with this model, rather than
    # making the user come back and press the button.
    #
    # This is the last phase of the sequence the one-GPU layout forces: report,
    # then experiments, then report again. The experiments own the GPU for hours,
    # so a report written afterwards cannot be something the user starts while
    # they are still going -- and asking them to be present at the right moment
    # for a phase that starts when a queue drains is asking them to be present at
    # an unpredictable time.
    report_after_finish: bool = False
    report_model_id: str = ""
    report_source: str = "catalog"  # "catalog" | "autoresearch" | "agent"
    # Which CLI, when the report source is an agent. Defaults to the loop's own
    # agent so "write it with the same thing" needs no second decision.
    report_agent_key: str = ""
    report_agent_args: list[str] = field(default_factory = list)

    def to_dict(self) -> dict:
        return {
            "num_experiments": self.num_experiments,
            "agent": self.agent,
            "agent_args": list(self.agent_args),
            "skip_permissions": self.skip_permissions,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "max_consecutive_failures": self.max_consecutive_failures,
            "prepare_data": self.prepare_data,
            "focus": self.focus,
            "continue_on_failure": self.continue_on_failure,
            "report_after_finish": self.report_after_finish,
            "report_model_id": self.report_model_id,
            "report_source": self.report_source,
            "report_agent_key": self.report_agent_key,
            "report_agent_args": list(self.report_agent_args),
        }


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------


@dataclass
class ExperimentInfo:
    """One experiment's row in the run's stepper."""

    index: int
    status: str = "pending"  # pending | running | kept | discarded | crashed | stopped
    message: str = ""
    error: Optional[str] = None
    val_bpb: Optional[float] = None
    memory_gb: Optional[float] = None
    commit: str = ""
    description: str = ""
    summary: str = ""
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    checkpoint: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return max(0.0, self.ended_at - self.started_at)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "val_bpb": self.val_bpb,
            "memory_gb": self.memory_gb,
            "commit": self.commit,
            "description": self.description,
            "summary": self.summary,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "checkpoint": self.checkpoint,
        }


@dataclass
class RunState:
    """Everything the UI reads about a run."""

    run_id: str = ""
    status: str = "idle"  # idle | running | completed | error | stopped
    phase: str = PHASE_PREPARING
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    experiment: int = 0
    total_experiments: int = 0
    progress_percent: float = 0.0

    # Live numbers from the experiment in flight, read out of the run log.
    step: int = 0
    loss: Optional[float] = None
    tok_per_sec: Optional[float] = None
    mfu: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None

    best_val_bpb: Optional[float] = None
    baseline_val_bpb: Optional[float] = None
    kept: int = 0
    discarded: int = 0
    crashed: int = 0

    experiments: list[ExperimentInfo] = field(default_factory = list)
    warnings: list[str] = field(default_factory = list)

    # The report the loop asked for, written after the experiments so it can be
    # read on the tab without a second click. Held here rather than only on disk
    # because the UI is watching this run and the report is its last phase.
    report_text: str = ""
    report_model: str = ""
    report_saved_to: Optional[str] = None
    report_error: Optional[str] = None
    report_running: bool = False

    resolved_config: dict = field(default_factory = dict)
    git_branch: str = ""
    git_head: str = ""

    stop_requested: bool = False

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "experiment": self.experiment,
            "total_experiments": self.total_experiments,
            "progress_percent": self.progress_percent,
            "step": self.step,
            "loss": self.loss,
            "tok_per_sec": self.tok_per_sec,
            "mfu": self.mfu,
            "peak_memory_gb": self.peak_memory_gb,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "best_val_bpb": self.best_val_bpb,
            "baseline_val_bpb": self.baseline_val_bpb,
            "kept": self.kept,
            "discarded": self.discarded,
            "crashed": self.crashed,
            "experiments": [e.to_dict() for e in self.experiments],
            "warnings": self.warnings,
            "report_text": self.report_text,
            "report_model": self.report_model,
            "report_saved_to": self.report_saved_to,
            "report_error": self.report_error,
            "report_running": self.report_running,
            "resolved_config": self.resolved_config,
            "git_branch": self.git_branch,
            "git_head": self.git_head,
            "stop_requested": self.stop_requested,
        }


# -----------------------------------------------------------------------------
# The run
# -----------------------------------------------------------------------------


class AutoresearchRun:
    """One request for N experiments, and everything it produced."""

    def __init__(self, run_id: str, config: AutoresearchConfig) -> None:
        self.run_id = run_id
        self.config = config

        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._state = RunState(
            run_id = run_id,
            total_experiments = config.num_experiments,
            resolved_config = config.to_dict(),
            phase = PHASE_PREPARING,
            message = "Preparing the experiment loop",
        )
        self._experiments: list[ExperimentInfo] = [
            ExperimentInfo(index = i + 1) for i in range(config.num_experiments)
        ]
        self._log_lines: list[dict] = []
        self._log_seq = 0
        # The agent's own stderr, kept apart from the console so a failure can be
        # explained with the line that names the cause.
        self._agent_tail: list[str] = []
        # The previous experiment's crash reason and how many times in a row it has
        # repeated, which is what distinguishes a loop that is going nowhere from a
        # run having a bad experiment. See REPEATED_CRASH_LIMIT.
        self._previous_crash_reason: Optional[str] = None
        self._repeat_failures = 0
        self._last_crash_reason: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        root = environment.checkout_root()
        self._run_dir = account_path(f"outputs/autoresearch/{run_id}")
        self._event_file = self._run_dir / "events.jsonl"
        self._stop_file = self._run_dir / "stop.request"
        self._root = root

    # --- paths -----------------------------------------------------------

    @property
    def checkout_root(self) -> Path:
        return environment.checkout_root()

    @property
    def train_log(self) -> Path:
        return self.checkout_root / results.TRAIN_LOG_NAME

    def output_dir(self) -> Path:
        return self._run_dir

    # --- state ------------------------------------------------------------

    def state(self) -> RunState:
        with self._lock:
            snapshot = RunState(
                run_id = self._state.run_id,
                status = self._state.status,
                phase = self._state.phase,
                message = self._state.message,
                error = self._state.error,
                started_at = self._state.started_at,
                ended_at = self._state.ended_at,
                experiment = self._state.experiment,
                total_experiments = self._state.total_experiments,
                progress_percent = self._state.progress_percent,
                step = self._state.step,
                loss = self._state.loss,
                tok_per_sec = self._state.tok_per_sec,
                mfu = self._state.mfu,
                peak_memory_gb = self._state.peak_memory_gb,
                elapsed_seconds = self._state.elapsed_seconds,
                eta_seconds = self._state.eta_seconds,
                best_val_bpb = self._state.best_val_bpb,
                baseline_val_bpb = self._state.baseline_val_bpb,
                kept = self._state.kept,
                discarded = self._state.discarded,
                crashed = self._state.crashed,
                experiments = [ExperimentInfo(**e.__dict__) for e in self._experiments],
                warnings = list(self._state.warnings),
                report_text = self._state.report_text,
                report_model = self._state.report_model,
                report_saved_to = self._state.report_saved_to,
                report_error = self._state.report_error,
                report_running = self._state.report_running,
                resolved_config = dict(self._state.resolved_config),
                git_branch = self._state.git_branch,
                git_head = self._state.git_head,
                stop_requested = self._state.stop_requested,
            )
        return snapshot

    def is_active(self) -> bool:
        with self._lock:
            return self._state.status == "running"

    def log_tail(self, limit: int = 400) -> list[dict]:
        with self._lock:
            return list(self._log_lines[-limit:])

    def logs_since(
        self,
        cursor: int,
        limit: int = _MAX_LOG_LINES_PER_FRAME,
    ) -> tuple[list[dict], int]:
        with self._lock:
            fresh = [line for line in self._log_lines if line["seq"] > cursor][:limit]
            next_cursor = fresh[-1]["seq"] if fresh else cursor
            return fresh, next_cursor

    # --- lifecycle ---------------------------------------------------------

    def _free_vram_for_run(self) -> list[str]:
        """Release whatever cannot stay resident, and say what was released.

        Called once, before the first experiment, rather than per experiment: the
        memory picture does not change between attempts, and unloading a model
        the user is reading a report in on every single iteration would be a
        hundred interruptions for one necessary one.

        The peak passed to the probe is the largest this project has recorded, so
        the decision on the first run of a fresh workspace is the conservative
        one -- nothing has been measured, so nothing is assumed to fit.
        """
        from routes.training_vram import coordinate_models_for_autoresearch

        peak = _measured_peak_bytes(self.checkout_root)
        try:
            freed = coordinate_models_for_autoresearch(peak_bytes = peak)
        except Exception:
            # Coordination is best effort by design, like every other call into
            # it. A failure here must not stop a run that would have fit, and a
            # user who needs to know can free things themselves.
            logger.debug("could not coordinate models for autoresearch", exc_info = True)
            return []
        if freed:
            logger.info("freed VRAM before autoresearch: %s", ", ".join(freed))
            self._append_log(f"> released for the GPU: {', '.join(freed)}")
        return freed

    def start(self) -> RunState:
        status = environment.environment_status()
        if not status.ready():
            raise AutoresearchNotInstalled(
                status.blocking_reason or "autoresearch is not installed"
            )

        spec = agent_mod.get_spec(self.config.agent)
        if spec is None:
            raise AutoresearchNotInstalled(f"unknown agent: {self.config.agent!r}")
        available = {a.key: a for a in agent_mod.detect_agents()}
        if self.config.agent not in available or not available[self.config.agent].available:
            reason = (
                available.get(self.config.agent).reason
                if available.get(self.config.agent)
                else "not found"
            )
            raise AutoresearchNotInstalled(
                f"{spec.label} cannot be used: {reason}. Install it, or pick another agent."
            )

        # Hand the GPU over before anything is launched, not after the first
        # experiment has already OOMed. This is the step that lets a chat model be
        # loaded while the user reads a report and still be safe when they start a
        # loop: whatever cannot stay is released here, and the UI is told what so
        # the release is not a mystery.
        freed = self._free_vram_for_run()

        self._run_dir.mkdir(parents = True, exist_ok = True)
        # A fresh event file per run: the reader follows byte offsets, and an
        # old file's contents would be replayed as this run's first frames.
        try:
            self._event_file.unlink()
        except OSError:
            pass

        results.ensure_results_file(self.checkout_root)
        git = agent_mod.git_state(str(self.checkout_root))
        with self._lock:
            self._state.status = "running"
            self._state.started_at = time.time()
            self._state.git_branch = git.branch
            self._state.git_head = git.head
        if git.dirty:
            self._warn(
                "The experiment repository has uncommitted changes, so this run's "
                "baseline is not the last recorded experiment. A previous run was "
                "probably interrupted between editing train.py and committing."
            )

        self._thread = threading.Thread(
            target = self._run_loop, name = f"autoresearch-{self.run_id}", daemon = True
        )
        self._thread.start()
        return self.state()

    def stop(self, *, force: bool = False) -> dict:
        """Ask the loop to finish after the experiment in flight.

        Cooperative, with a hard deadline. The agent is mid-experiment and its
        work is a file edit plus a git commit; killing it at an arbitrary moment
        can leave train.py changed with no record of the change, which the next
        experiment would then build on. So the sentinel is written and the
        process is given a grace period before anything is killed.

        ``force`` shortens that grace to nothing, which is what shutdown needs:
        there is no user left to finish an experiment for.
        """
        with self._lock:
            self._state.stop_requested = True
            self._state.message = "Stopping after the current experiment"
        self._stop_requested.set()
        try:
            self._stop_file.parent.mkdir(parents = True, exist_ok = True)
            self._stop_file.write_text("stop\n", encoding = "utf-8")
        except OSError:
            pass
        if force:
            with self._proc_lock:
                proc = self._proc
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
        return {"status": "stopping", "run_id": self.run_id}

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread is None:
            return True
        return self._thread.join(timeout = timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- the loop ----------------------------------------------------------

    def _run_loop(self) -> None:
        root = self.checkout_root
        try:
            # The dataset and tokenizer are not optional: train.py reads the
            # tokenizer from the cache and dies without it, so an unprepared
            # workspace would fail every experiment identically.
            if self.config.prepare_data and not environment.is_data_prepared():
                self._set(phase = PHASE_PREPARING, message = "Preparing data (first run only)")
                self._append_log("> preparing data and tokenizer")
                environment.prepare_data_async()
                deadline = time.time() + 30 * 60
                while not environment.is_data_prepared():
                    if self._stop_requested.is_set() or time.time() > deadline:
                        raise RuntimeError("data preparation did not finish in time")
                    if not self.is_alive():
                        break
                    time.sleep(1.0)
                if not environment.is_data_prepared():
                    raise RuntimeError("data preparation did not finish")
                self._append_log("> data ready")

            consecutive_failures = 0
            summary = results.read_results(root)
            baseline = summary.baseline()

            for index in range(1, self.config.num_experiments + 1):
                if self._stop_requested.is_set():
                    break

                outcome = self._run_experiment(index, summary)
                if outcome is None:
                    break  # stopped
                if outcome == "crash":
                    consecutive_failures += 1
                    if (
                        self.config.max_consecutive_failures
                        and consecutive_failures >= self.config.max_consecutive_failures
                    ):
                        self._finish(
                            "error",
                            f"{consecutive_failures} experiments in a row failed; stopping",
                        )
                        return
                    # A crash that comes back word for word is not a new result.
                    #
                    # `max_consecutive_failures` counts any failure, because
                    # different failures are information: a loop that gave up at two
                    # could not tell a bad idea from a bad machine. But an identical
                    # repeat is a configuration that cannot work -- a model the
                    # agent's endpoint refuses, a missing dataset -- and re-running
                    # it is not an experiment, it is paying the same eleven minutes
                    # again to learn the same thing. Two is enough to tell a
                    # one-off from a loop that is going nowhere, and a user who
                    # meant to push on can start another run.
                    reason = self._last_crash_reason
                    self._repeat_failures, stop = _count_repeat(
                        self._previous_crash_reason, reason, self._repeat_failures
                    )
                    self._previous_crash_reason = reason
                    if stop:
                        self._finish(
                            "error",
                            f"the same failure repeated {self._repeat_failures} times in a "
                            f"row, which is a setup problem rather than a result: {reason}",
                        )
                        return
                else:
                    consecutive_failures = 0
                    self._repeat_failures = 0
                    self._previous_crash_reason = None

                # Re-read the ledger rather than patching the summary: the agent
                # wrote the row, and its own account of what happened is the
                # record. Re-reading also picks up a row the agent added by hand.
                summary = results.read_results(root)
                if baseline is None:
                    baseline = summary.baseline()

            if self._stop_requested.is_set():
                self._finish("stopped", "Stopped after the current experiment")
            else:
                self._finish("completed", f"Ran {self.config.num_experiments} experiments")
        except Exception as exc:  # noqa: BLE001 - the state carries the reason
            logger.exception("autoresearch run %s failed", self.run_id)
            self._finish("error", str(exc))
        finally:
            self._kill_process()

    def _run_experiment(self, index: int, summary: results.ResultsSummary) -> Optional[str]:
        """One agent invocation. Returns "crash", "ok", or None if stopped."""
        root = self.checkout_root
        record = self._experiment(index)
        best = summary.best()
        baseline = summary.baseline()

        prompt = agent_mod.build_experiment_prompt(
            index = index,
            total = self.config.num_experiments,
            best_bpb = best.val_bpb if best else None,
            completed = summary.total,
            baseline_bpb = baseline.val_bpb if baseline else None,
        )
        if self.config.focus.strip():
            prompt += f"\nA focus for this run, from the user:\n  {self.config.focus.strip()}\n"

        try:
            # The brief goes to a file, not to argv: see agent._PROMPT_INSTRUCTION.
            agent_mod.write_prompt(root, prompt)
            command = agent_mod.build_command(
                agent_key = self.config.agent,
                prompt = prompt,
                cwd = str(root),
                skip_permissions = self.config.skip_permissions,
                timeout_seconds = self.config.agent_timeout_seconds,
                extra_args = self.config.agent_args,
            )
        except (ValueError, RuntimeError) as exc:
            self._finish("error", str(exc))
            return None

        with self._lock:
            record.status = "running"
            record.started_at = time.time()
            self._state.experiment = index
            self._state.phase = PHASE_EXPERIMENT
            self._state.message = f"Experiment {index} of {self.config.num_experiments}"
            self._state.step = 0
            self._state.loss = None
            self._state.elapsed_seconds = 0.0
            self._state.eta_seconds = None
            self._state.progress_percent = (index - 1) / max(1, self.config.num_experiments) * 100.0

        self._append_log("")
        self._append_log(f"> experiment {index} of {self.config.num_experiments}")
        self._append_log(f"> agent: {self.config.agent}")
        # Reset per experiment: the previous one's failures are not evidence about
        # this one, and quoting them would be a lie.
        with self._lock:
            self._agent_tail = []
            # Cleared alongside the tail so a repeat is only ever judged against a
            # reason this experiment actually produced. A stale one from the last
            # experiment would make the first crash of a new kind look like a
            # repeat of the old kind.
            self._last_crash_reason = None

        # Each experiment starts from a clean slate for the things the previous
        # one wrote. run.log in particular is appended to by the agent, and
        # leaving it would make the next experiment's summary block
        # indistinguishable from this one's.
        try:
            self.train_log.unlink()
        except OSError:
            pass

        started = time.time()
        try:
            returncode = self._supervise(command, record)
        except Exception as exc:  # noqa: BLE001
            record.status = "crashed"
            record.error = str(exc)
            # A failure to even run the agent is a crash like any other, and a
            # repeatable one -- an agent that is not installed fails this way every
            # time -- so it counts towards the repeat limit like a graded crash.
            self._last_crash_reason = str(exc)
            record.ended_at = time.time()
            self._append_log(f"> experiment {index} failed: {exc}", stream = "stderr")
            return "crash"

        record.ended_at = time.time()
        elapsed = record.ended_at - started

        if self._stop_requested.is_set() and returncode != 0:
            record.status = "stopped"
            record.message = "Stopped by hand"
            return None

        outcome = self._harvest(index, record, returncode, elapsed)
        return "crash" if outcome.status == "crashed" else "ok"

    def _supervise(self, command: agent_mod.AgentCommand, record: ExperimentInfo) -> int:
        """Run the agent, pumping its output into the console until it exits.

        Both streams are drained by their own thread. One reader loop for both
        would let a chatty stderr starve stdout, and the agent's progress output
        is exactly what the user is watching.
        """
        env = environment.experiment_environment(
            # The agent inherits the same cache dir the experiments use, so its
            # `uv run train.py` finds the tokenizer data preparation wrote.
            extra = {"AUTORESEARCH_STOP_FILE": str(self._stop_file)},
        )
        try:
            proc = subprocess.Popen(
                command.argv,
                cwd = command.cwd,
                env = env,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                stdin = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                bufsize = 1,
                **child_popen_kwargs(),
                **windows_hidden_subprocess_kwargs(),
            )
        except Exception as exc:
            raise RuntimeError(f"could not start {command.argv[0]}: {exc}") from exc

        with self._proc_lock:
            self._proc = proc
        adopt_pid(proc.pid)

        train_log_seen = 0.0
        deadline = time.monotonic() + command.timeout_seconds
        try:
            pump = threading.Thread(
                target = self._drain,
                args = (proc.stdout, "stdout"),
                name = "autoresearch-stdout",
                daemon = True,
            )
            pump_err = threading.Thread(
                target = self._drain,
                args = (proc.stderr, "stderr"),
                name = "autoresearch-stderr",
                daemon = True,
            )
            pump.start()
            pump_err.start()

            while True:
                code = proc.poll()
                if code is not None:
                    break
                if self._stop_requested.is_set():
                    self._terminate(proc)
                    raise RuntimeError("stopped by hand")
                if time.monotonic() > deadline:
                    self._kill_process()
                    self._append_log(
                        f"> the agent ran for longer than "
                        f"{command.timeout_seconds // 60} minutes and was stopped",
                        stream = "stderr",
                    )
                    raise RuntimeError(
                        f"the agent exceeded its {command.timeout_seconds}s budget; "
                        f"raise the per-experiment timeout if this is expected"
                    )
                # The agent's training run writes to run.log rather than to its
                # own stdout, so the live numbers have to be read from there.
                now = time.monotonic()
                if now - train_log_seen > 1.0:
                    train_log_seen = now
                    self._pump_train_log(record)
                time.sleep(0.2)
            pump.join(timeout = 5)
            pump_err.join(timeout = 5)
            return code
        finally:
            with self._proc_lock:
                if self._proc is proc:
                    self._proc = None
            # The pipes are closed here rather than in the drain threads so that
            # a process that outlived them cannot keep the run's stdout open.
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def _drain(self, stream, name: str) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                # autoresearch prints its training progress with \r and no
                # newline, so a raw split leaves a hundred partial lines in one.
                # The tail is what a terminal shows too.
                for piece in str(line).rstrip("\n").split("\r"):
                    piece = piece.rstrip()
                    if piece:
                        self._append_log(piece, stream = name)
                        if name == "stderr":
                            self._note_agent_output(piece)
        except Exception:
            pass

    def _note_agent_output(self, line: str) -> None:
        """Keep the tail of what the agent said, to explain a failure with.

        The console already shows all of it, but the console is a hundred lines
        of scrollback and the experiment row is one line. When an experiment has
        no score, "the run did not print a val_bpb" is a true sentence that
        tells the user nothing, when the actual cause -- an agent that cannot
        reach its model, a disk that filled, a tool that is not on PATH -- is
        sitting three lines above it.
        """
        with self._lock:
            self._agent_tail.append(line)
            if len(self._agent_tail) > _MAX_AGENT_TAIL_LINES:
                del self._agent_tail[:-_MAX_AGENT_TAIL_LINES]

    def _pump_train_log(self, record: ExperimentInfo) -> None:
        """Read the live numbers out of run.log.

        train.py's progress line is one \r-overwritten line, so only the last
        segment matters. It is the sole source of step, loss and throughput while
        an experiment trains: there is no tensorboard and no metrics file in the
        project, and adding one would mean editing the file the agent owns.
        """
        try:
            raw = self.train_log.read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            return
        if not raw:
            return
        tail = raw.split("\r")[-1].strip()
        if not tail:
            return
        step = _int_from(tail, "step")
        loss = _float_from(tail, "loss")
        tok_per_sec = _int_from(tail, "tok/sec")
        mfu = _float_from(tail, "mfu")
        remaining = _int_from(tail, "remaining")
        with self._lock:
            if step is not None:
                self._state.step = step
            if loss is not None:
                self._state.loss = loss
            if tok_per_sec is not None:
                self._state.tok_per_sec = tok_per_sec
            if mfu is not None:
                self._state.mfu = mfu
            if record.started_at is not None:
                self._state.elapsed_seconds = time.time() - record.started_at
            if remaining is not None:
                self._state.eta_seconds = remaining
            # A step is worth a fraction of an experiment, so the bar moves
            # within one rather than sitting still for five minutes. Capped at
            # 90% because the last 10% is the evaluation, which has no step
            # counter of its own and would otherwise look like a hang.
            if step:
                self._state.progress_percent = min(
                    90.0,
                    ((self._state.experiment - 1) + 0.9)
                    / max(1, self._state.total_experiments)
                    * 100.0,
                )

    def _harvest(
        self, index: int, record: ExperimentInfo, returncode: int, elapsed: float
    ) -> ExperimentInfo:
        """Turn a finished agent invocation into a recorded experiment.

        The score comes from run.log, not from the agent's closing message. The
        agent's own account is kept as the summary, but the number in the table
        is the one the project's evaluation printed: an agent that misreads its
        own log, or summarises optimistically, must not be able to improve the
        recorded result.
        """
        try:
            log_text = self.train_log.read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            log_text = ""
        outcome = agent_mod.parse_run_log(log_text)

        record.summary = ""
        record.commit = _git_head(self.checkout_root)
        record.val_bpb = outcome.val_bpb
        record.memory_gb = _float_or_none(outcome.summary.get("peak_vram_mb"), scale = 1.0 / 1024.0)

        if not outcome.ok:
            record.status = "crashed"
            record.error = _explain_failure(self._agent_tail, returncode)
            # Kept where the reason is decided, so the repeat check compares the
            # same text the reader is shown rather than re-deriving it from a log
            # that has since moved on.
            self._last_crash_reason = record.error
            for line in outcome.tail[-12:]:
                self._append_log(f"  {line}", stream = "stderr")
            # A crash is recorded whatever the agent managed to do, so the ledger
            # and the UI agree on what happened.
            results.append_result(
                self.checkout_root,
                commit = record.commit or "(none)",
                val_bpb = None,
                memory_gb = None,
                status = results.STATUS_CRASH,
                description = record.error or "the run did not finish",
            )
            self._append_log(
                f"> experiment {index} crashed after {elapsed:.0f}s: {record.error}",
                stream = "stderr",
            )
            return record

        status = _row_status(results.read_results(self.checkout_root), index)
        record.status = {
            results.STATUS_KEEP: "kept",
            results.STATUS_DISCARD: "discarded",
        }.get(status, "kept")
        record.message = f"val bpb {outcome.val_bpb:.6f}"
        record.description = _row_description(results.read_results(self.checkout_root), index)

        checkpoint = results.capture_checkpoint(
            self.checkout_root,
            experiment = index,
            commit = record.commit,
            val_bpb = outcome.val_bpb,
            num_params_m = _float_or_none(outcome.summary.get("num_params_M")),
            depth = _int_or_none(outcome.summary.get("depth")),
        )
        if checkpoint is not None:
            record.checkpoint = checkpoint.name
        else:
            self._warn(
                f"Experiment {index} finished but left no checkpoint behind, so it "
                f"cannot be chatted with later."
            )

        self._append_log(
            f"> experiment {index}: val_bpb {outcome.val_bpb:.6f} ({record.status}, {elapsed:.0f}s)"
        )
        return record

    # --- state helpers -----------------------------------------------------

    def _experiment(self, index: int) -> ExperimentInfo:
        with self._lock:
            return self._experiments[index - 1]

    def _set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self._state, key, value)

    def _warn(self, message: str) -> None:
        with self._lock:
            if message in self._state.warnings:
                return
            self._state.warnings = (self._state.warnings + [message])[-20:]
        self._append_log(f"> {message}", stream = "stderr")

    def _append_log(
        self,
        line: str,
        *,
        stream: str = "stdout",
    ) -> None:
        with self._lock:
            self._log_seq += 1
            entry = {
                "seq": self._log_seq,
                "stream": stream,
                "line": line,
                "ts": time.time(),
            }
            self._log_lines.append(entry)
            if len(self._log_lines) > _MAX_LOG_LINES:
                del self._log_lines[:-_MAX_LOG_LINES]
        self._write_event({"kind": "log", "stream": stream, "line": line})

    def _write_event(self, payload: dict) -> None:
        """Append one record to the run's event file.

        Best effort. The in-memory state is authoritative and the file is a
        replay aid for a reconnecting client, so a failed write must never take
        down the experiment that is running.
        """
        try:
            self._run_dir.mkdir(parents = True, exist_ok = True)
            with self._event_file.open("a", encoding = "utf-8", newline = "\n") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError:
            pass

    def _finish(self, status: str, message: str) -> None:
        with self._lock:
            self._state.status = status
            self._state.phase = PHASE_FINISHED
            self._state.message = message
            self._state.ended_at = time.time()
            self._state.progress_percent = (
                100.0 if status == "completed" else self._state.progress_percent
            )
        self._append_log(f"> {message}")
        self._write_event({"kind": "finished", "status": status, "message": message})
        # The chat worker and a training run cannot both have the GPU, and the
        # experiments are over, so anything the user loaded for a look is
        # released rather than left to collide with the next loop.
        try:
            from core.autoresearch import chat_sidecar
            chat_sidecar.free_for_training(f"autoresearch run {self.run_id} {status}")
        except Exception:
            logger.debug("could not free the autoresearch chat worker", exc_info = True)

        # The last phase: hand the GPU to the model that writes the report. Only
        # on a clean finish -- a stopped or failed run produced a partial search,
        # and a report over it would read as an assessment of work that never
        # happened.
        if self.config.report_after_finish and status == "completed":
            self._write_report_after_finish()

    def _write_report_after_finish(self) -> None:
        """Run the report as the run's final phase.

        Blocking, on the sequencer thread, so the report is ordered after the
        experiments rather than racing them. That ordering is the whole point: on
        one GPU the report cannot be written while the experiments hold the card,
        and the moment it becomes possible is the moment the queue drains, which
        is not a moment a user can be asked to be watching for.
        """
        from core.autoresearch import report as report_mod

        with self._lock:
            self._state.report_running = True
            self._state.report_model = self.config.report_model_id
        self._append_log(f"> writing the report with {self.config.report_model_id}")
        self._write_event({"kind": "report_started", "model": self.config.report_model_id})

        collected: list[str] = []
        try:
            request = report_mod.ReportRequest(
                model_id = self.config.report_model_id,
                max_tokens = 2000,
                temperature = 0.7,
                source = self.config.report_source,
                # The report defaults to the same agent, and the same arguments,
                # that ran the experiments. Writing it with a different model
                # than the one that did the work is a choice; having to make that
                # choice again in a second field is not.
                agent_key = self.config.report_agent_key or self.config.agent,
                agent_args = list(self.config.report_agent_args or self.config.agent_args),
            )
            for cumulative in _drive_report(request):
                collected.append(cumulative)
            text = collected[-1] if collected else ""
            saved = report_mod.save_report(self.checkout_root, self.run_id, text)
        except Exception as exc:  # noqa: BLE001 - the experiments still succeeded
            logger.warning("the autoresearch report failed", exc_info = True)
            with self._lock:
                self._state.report_error = str(exc)
                self._state.report_running = False
            self._append_log(f"> the report failed: {exc}", stream = "stderr")
            self._write_event({"kind": "report_failed", "error": str(exc)})
            return

        with self._lock:
            self._state.report_text = text
            self._state.report_saved_to = saved
            self._state.report_running = False
        self._append_log("> the report is ready on the Report tab")
        self._write_event({"kind": "report_done", "saved_to": saved})

    def _terminate(self, proc: subprocess.Popen) -> None:
        """Ask, then insist."""
        try:
            terminate_pid(proc.pid, timeout = float(_STOP_GRACE_SECONDS), owner_verified = True)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout = float(_KILL_TIMEOUT_SECONDS))
        except subprocess.TimeoutExpired:
            self._kill_process()

    def _kill_process(self) -> None:
        with self._proc_lock:
            proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Reading helpers, all off run.log's progress line
# -----------------------------------------------------------------------------


def _field(line: str, name: str) -> Optional[str]:
    """The value for ``name`` on a train.py progress line.

    The line does not use one convention. ``step 00042`` has no colon while
    ``loss: 4.12`` does, so both forms are tried rather than one being assumed:
    getting that wrong is invisible, and the symptom is a progress bar that sits
    still for a five-minute training pass.
    """
    value = _value_after(line, f"{name}:")
    if value is None:
        value = _value_after(line, f"{name} ")
    return value


def _value_after(line: str, marker: str) -> Optional[str]:
    start = line.find(marker)
    if start < 0:
        return None
    rest = line[start + len(marker) :]
    # Stop at the next field so "loss" does not pick up what follows it.
    for separator in ("|", ",", "  "):
        cut = rest.find(separator)
        if cut >= 0:
            rest = rest[:cut]
    rest = rest.strip().rstrip("%").strip()
    return rest or None


# The leading number of a value, ignoring any unit suffix.
#
# train.py writes its timings and its countdown with units attached: "dt:
# 118.32ms", "remaining: 246s". Taking the whole token fails on both, and the
# failure is invisible because the field simply reads as absent.
_LEADING_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _number_from(line: str, name: str) -> Optional[str]:
    value = _field(line, name)
    if value is None:
        return None
    match = _LEADING_NUMBER_RE.match(value)
    return match.group(0) if match else None


def _int_from(line: str, name: str) -> Optional[int]:
    value = _number_from(line, name)
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _float_from(line: str, name: str) -> Optional[float]:
    value = _number_from(line, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _float_or_none(value: Optional[str], scale: float = 1.0) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _drive_report(request) -> Iterator[str]:
    """Run the report generator to completion, yielding each cumulative snapshot.

    ``report.generate_report`` is async because the catalog path is: it goes
    through the studio's OpenAI-compatible completions, which is an asyncio
    route. This sequencer thread is not on an event loop, so the coroutine gets
    its own loop on a worker thread and the snapshots come back across a queue.

    Hand-rolled rather than a helper because there is exactly one caller and the
    alternative -- making the whole sequencer async -- would mean every log line
    and every git call in this file is now on the event loop.
    """
    import asyncio
    import queue as queue_mod
    import threading as threading_mod

    from core.autoresearch import report as report_mod

    outbox: "queue_mod.Queue" = queue_mod.Queue()
    sentinel = object()

    async def pump() -> None:
        try:
            # fastapi_request=None marks the background case: there is no HTTP
            # client, so the report goes straight at whichever backend is
            # resident rather than through a completions route built around a
            # real Request. The loop's own event is the only thing that can stop
            # it, and it is passed below.
            async for cumulative in report_mod.generate_report(
                request, fastapi_request = None, owner = "autoresearch"
            ):
                outbox.put(cumulative)
        except Exception as exc:  # noqa: BLE001
            outbox.put(exc)
        finally:
            outbox.put(sentinel)

    def runner() -> None:
        try:
            asyncio.run(pump())
        except Exception as exc:  # noqa: BLE001
            outbox.put(exc)
            outbox.put(sentinel)

    thread = threading_mod.Thread(target = runner, name = "autoresearch-report", daemon = True)
    thread.start()
    while True:
        item = outbox.get()
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


class _OffLoopRequest:
    """Retired. A fabricated Request does not survive the completions path.

    It read ``request.url.path`` for its audit log and ``request.headers.get`` for
    auth, so a stub with a ``headers()`` *method* failed deep in the inference
    stack with ``'function' object has no attribute 'get'``. The background report
    now passes ``None`` and drives the resident backend directly; see
    ``report._stream_resident_model``.
    """


def _measured_peak_bytes(root: Path) -> Optional[int]:
    """The largest peak VRAM any recorded experiment used, in bytes.

    Read from the ledger rather than tracked in memory, so it survives a restart
    and so the first run of a new session is judged on what the project actually
    did rather than on nothing.

    A crash row is skipped: autoresearch writes ``0.0`` for a run that died before
    its save, and treating that as a measurement would make every subsequent run
    believe it needs no memory at all.
    """
    summary = results.read_results(root)
    peaks = [
        row.memory_gb
        for row in summary.rows
        if row.memory_gb is not None and row.memory_gb > 0 and row.val_bpb is not None
    ]
    if not peaks:
        return None
    return int(max(peaks) * 1e9)


def _git_head(root: Path) -> str:
    """The commit the experiment left HEAD on, for the ledger's first column."""
    if not shutil.which("git"):
        return ""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd = str(root),
            capture_output = True,
            text = True,
            timeout = 30,
            encoding = "utf-8",
            errors = "replace",
            **windows_hidden_subprocess_kwargs(),
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _row_status(summary: results.ResultsSummary, index: int) -> str:
    """The status of the row the agent just wrote.

    Matched on position from the end, not on commit: the agent may have amended,
    rebased or left the commit untouched, and the row it appended is the last
    one either way. A ledger that gained no row means the agent did not get that
    far, which is a crash rather than a keep.
    """
    if len(summary.rows) < index:
        return results.STATUS_CRASH
    return summary.rows[index - 1].status


def _row_description(summary: results.ResultsSummary, index: int) -> str:
    if len(summary.rows) < index:
        return ""
    return summary.rows[index - 1].description


def _count_repeat(
    previous: str | None, reason: str | None, repeats: int
) -> tuple[int, bool]:
    """How many times this crash has now repeated in a row, and whether to stop.

    Split out from the sequencer so the rule can be read -- and tested -- without a
    thread, an agent and a checkout. `previous` is the last crash's reason, so a
    run of identical reasons accumulates and anything else resets the count.

    A crash with no reason is never treated as a repeat: an unattributable failure
    is the one case where two identical-looking outcomes might be two different
    problems, and stopping on a guess would hide a real one.
    """
    if reason and reason == previous:
        repeats += 1
    else:
        repeats = 1
    return repeats, bool(reason) and repeats >= REPEATED_CRASH_LIMIT


def _explain_failure(agent_tail: list[str], returncode: int) -> str:
    """The most useful one-line reason an experiment produced no score.

    Two failure shapes have to be told apart, because the fix is different:

      - the **agent** never got to the training run. An agent CLI that cannot
        reach its model, is not authenticated, or is missing a tool says so and
        exits. Nothing about val_bpb is wrong here; the loop simply never ran.
      - the **training run** started and then died. That is what program.md's own
        "empty grep means it crashed" rule describes, and the log tail is the
        evidence.

    Quoting the agent's own line is the difference between a user knowing their
    model is misconfigured and a user staring at "no val_bpb" with no idea.
    """
    for line in reversed(agent_tail or []):
        lowered = line.lower()
        if not any(marker in lowered for marker in _AGENT_ERROR_MARKERS):
            continue
        # The agents quote their structured error twice, once per event. One copy
        # is enough, and the first is the readable sentence rather than the JSON.
        cleaned = line.strip()
        if cleaned.startswith("ERROR: [") and "message" in cleaned:
            quoted = re.search(r'"message"\s*:\s*"([^"]+)"', cleaned)
            if quoted:
                return quoted.group(1)
        if len(cleaned) > 300:
            cleaned = cleaned[:299].rstrip() + "…"
        return cleaned
    if returncode != 0:
        return (
            f"the agent exited with code {returncode} before it produced a val_bpb. "
            f"Its output is in the console above."
        )
    return "the run did not print a val_bpb, so it failed before evaluation"


# -----------------------------------------------------------------------------
# The manager
# -----------------------------------------------------------------------------


class AutoresearchRunManager:
    """The single active run, and the lock that keeps the GPU to itself.

    One at a time is not a simplification. Two autoresearch loops on one GPU
    would both be autotuning batch sizes against the same free VRAM, and the
    loser of that race gets an OOM it cannot interpret.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: Optional[AutoresearchRun] = None
        self._recent: list[AutoresearchRun] = []

    def current(self) -> Optional[AutoresearchRun]:
        with self._lock:
            return self._current

    def is_active(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.is_active()

    def start(self, config: AutoresearchConfig) -> AutoresearchRun:
        from core.training.lifecycle import training_lifecycle_guard

        if is_process_shutting_down():
            raise AutoresearchBusy("LABZ is shutting down")
        with self._lock:
            if self._current is not None and self._current.is_active():
                raise AutoresearchBusy("An autoresearch loop is already running")
        # Refuse rather than contend. One GPU, and every one of these wants all
        # of it: two loops running together means one of them OOMs at a step it
        # cannot predict, which for an unattended run is an experiment lost with
        # no explanation in the log. Checked here rather than inside the run so
        # the refusal names the other thing by name.
        with training_lifecycle_guard():
            from core.training.training import get_training_backend

            if get_training_backend().is_training_active():
                raise AutoresearchBusy("LABZ training is running; stop it first")

            from core.nanochat.orchestrator import get_run_manager as nanochat_runs

            if nanochat_runs().is_active():
                raise AutoresearchBusy("A nanochat run is in progress; stop it first")

            try:
                from core.benchmarks.orchestrator import get_run_manager as benchmark_runs
                if benchmark_runs().is_active():
                    raise AutoresearchBusy("A benchmark run is in progress; stop it first")
            except ImportError:
                # Benchmarks is optional; its absence is not a reason to refuse.
                pass

            run_id = f"run_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
            run = AutoresearchRun(run_id, config)
            with self._lock:
                self._current = run
                self._recent.append(run)
                if len(self._recent) > 5:
                    del self._recent[:-5]
        # Outside the guard, like nanochat and benchmarks: the guard serialises the
        # admission decision, not the run's lifetime. Holding it for the whole
        # loop would make a concurrent start block on an RLock for hours and then
        # be answered by a check that has long since gone stale.
        run.start()
        return run

    def stop(self) -> dict:
        with self._lock:
            run = self._current
        if run is None or not run.is_active():
            return {"status": "idle"}
        return run.stop()

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Stop the active loop, killing if it does not comply.

        Called on backend shutdown. Leaving an autoresearch loop running would
        outlive the app and hold the GPU, and unlike a training run there is no
        way to reattach to it: the agent it spawned dies with the session. So it
        is stopped unconditionally rather than politely.
        """
        with self._lock:
            run = self._current
        if run is None or not run.is_active():
            return True
        run.stop(force = True)
        return run.wait(timeout)

    def clear(self) -> None:
        with self._lock:
            self._current = None


_manager: Optional[AutoresearchRunManager] = None
_manager_lock = threading.Lock()


def get_run_manager() -> AutoresearchRunManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = AutoresearchRunManager()
        return _manager


def estimate_duration_seconds(config: AutoresearchConfig) -> int:
    """What a run of this size costs, roughly.

    Used only to set expectations before the user commits. autoresearch's budget
    is a fixed wall clock per experiment, so the estimate is close to linear in
    the count, and the honest answer is a range rather than a number.
    """
    return int(config.num_experiments * agent_mod.SECONDS_PER_EXPERIMENT_ESTIMATE)
