# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sequencing a nanochat run: stages, live progress, and stopping.

A nanochat run is a chain of separate processes (download data, train a tokenizer,
pretrain, evaluate, SFT, evaluate, RL), each run in nanochat's own virtualenv.
This module owns that chain.

Design notes
------------
The child reports progress as JSONL appended to one file per run
(``NANOCHAT_EVENT_FILE``); a reader thread follows that file and folds records
into the run state. Nothing here parses stdout for numbers, so a reworded log
message upstream cannot break the UI.

Why one stage process at a time, rather than the whole chain in one process:
each stage must be able to fail, be retried, and release the GPU between
pretraining and SFT. A chain in a single process would hold VRAM across the
boundary and turn any stage failure into "start the whole thing over".

Stopping is cooperative first: the sentinel file lets a stage finish its current
step and write a valid checkpoint, which is what makes a stopped run resumable.
Only if the stage ignores the sentinel for too long is it killed, because a stage
that will never notice (a download, a hung compile) must not hold the run open.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from core.nanochat import environment, presets
from utils.paths.storage_roots import account_path
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    is_process_shutting_down,
    pid_is_running,
    terminate_pid,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# How often the reader thread re-checks the event file for new bytes.
_READ_POLL_SECONDS = 0.4

# Bounds on what the UI keeps in memory. A long run emits tens of thousands of
# metric events; keeping them all would grow without limit, and the charts only
# ever need a downsampled series.
_MAX_METRIC_POINTS = 4000
# The console is the whole point of a run's log, so it keeps more than the other
# buffers, but it is still bounded: a stage can print a progress bar a few times
# a second for hours.
_MAX_LOG_LINES = 4000
# Most log lines a single SSE frame may carry. A frame is capped so one very
# chatty stage cannot blow up the stream buffer; the rest arrives on the next tick.
_MAX_LOG_LINES_PER_FRAME = 200
_MAX_SAMPLES = 200
_MAX_BENCHMARKS = 400

# A stage that ignores the stop sentinel for this long gets killed. Generous
# because the slowest thing it might be doing is a torch.compile or a checkpoint
# write, and killing mid-write would leave an unusable checkpoint.
_STOP_GRACE_SECONDS = 90

# How long to wait for a killed stage to actually disappear.
_KILL_TIMEOUT_SECONDS = 20


class NanochatBusy(RuntimeError):
    """Another nanochat run, or an Unsloth training run, owns the GPU."""


class NanochatNotInstalled(RuntimeError):
    """The nanochat environment is not ready."""


def _hms(seconds: float) -> str:
    """A duration as ``1h 04m 09s`` / ``4m 09s`` / ``9s``, for a console line.

    Coarse on purpose: this is a log line the user reads while waiting, not a
    metric, so trailing precision would only make the column ragged.
    """
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class NanochatConfig:
    """A validated run configuration, as the UI sends it.

    Mirrors the nanochat CLI flags the UI exposes. Defaults match nanochat's own
    so a run started with nothing set behaves like the published pipeline.
    """

    # --- model ---
    depth: int = presets.DEFAULT_DEPTH
    aspect_ratio: int = presets.DEFAULT_ASPECT_RATIO
    head_dim: int = presets.DEFAULT_HEAD_DIM
    max_seq_len: int = 2048
    window_pattern: str = "L"
    model_tag: str | None = None

    # --- data ---
    dataset: str = "climbmix-400b"
    num_shards: int = 32
    vocab_size: int = presets.DEFAULT_VOCAB_SIZE
    train_tokenizer: bool = True

    # --- horizon ---
    num_iterations: int = 1000
    param_data_ratio: float = 12.0
    total_batch_size: int = 32768
    device_batch_size: int = 4
    fp8: bool = False

    # --- learning rates ---
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.28
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05

    # --- evaluation cadence ---
    eval_every: int = 100
    eval_tokens: int = 2 * 1024 * 1024
    core_metric_every: int = 500
    sample_every: int = 200
    save_every: int = -1

    # --- SFT ---
    sft_tasks: list[str] = field(default_factory=lambda: ["smoltalk", "mmlu", "gsm8k"])
    mmlu_epochs: int = 3
    gsm8k_epochs: int = 4
    sft_iterations: int = 400
    sft_sample_every: int = 100

    # --- RL ---
    run_rl: bool = False
    rl_epochs: int = 1
    rl_examples_per_step: int = 16
    rl_num_samples: int = 16
    rl_max_new_tokens: int = 256
    rl_eval_every: int = 60
    rl_save_every: int = 60

    # --- which stages to run ---
    base_benchmarks: list[str] = field(default_factory=lambda: ["core", "bpb", "sample"])
    chat_benchmarks: list[str] = field(
        default_factory=lambda: ["arc-easy", "arc-challenge", "mmlu", "gsm8k", "humaneval"]
    )
    core_metric_max_per_task: int = 500

    # --- lifecycle ---
    stages: list[str] = field(default_factory=lambda: [
        "dataset", "tokenizer", "pretrain", "base_eval", "sft", "chat_eval", "rl",
    ])

    def resolved_model_tag(self) -> str:
        return self.model_tag or f"d{self.depth}"


# -----------------------------------------------------------------------------
# Live state
# -----------------------------------------------------------------------------

@dataclass
class MetricPoint:
    step: int
    value: Optional[float]
    total_steps: Optional[int] = None


@dataclass
class SampleRecord:
    stage: str
    step: int
    mode: str
    prompt: str
    completion: str
    reward: Optional[float] = None
    advantage: Optional[float] = None
    ts: float = 0.0


@dataclass
class BenchmarkRecord:
    stage: str
    name: str
    accuracy: Optional[float] = None
    baseline: Optional[float] = None
    centered: Optional[float] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    is_partial: bool = False
    ts: float = 0.0


@dataclass
class StageState:
    key: str
    status: str = "pending"  # pending | running | ok | failed | skipped | stopped
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    pid: Optional[int] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else time.time()
        return end - self.started_at


@dataclass
class RunState:
    """Everything the API and UI read about a run."""

    run_id: str
    status: str = "idle"  # idle | running | completed | error | stopped
    phase: str = "idle"
    current_stage: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    # Progress
    step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0

    # Current training numbers
    loss: Optional[float] = None
    val_bpb: Optional[float] = None
    chatcore: Optional[float] = None
    reward: Optional[float] = None
    tok_per_sec: Optional[float] = None
    mfu: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    eta_seconds: Optional[float] = None
    elapsed_seconds: float = 0.0

    # Derived model facts, once nanochat reports its resolved config
    resolved_config: Optional[dict] = None
    num_params: Optional[int] = None

    stages: dict[str, StageState] = field(default_factory=dict)
    checkpoints: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_seconds"] = (
            (self.ended_at or time.time()) - self.started_at if self.started_at else 0.0
        )
        return data


# -----------------------------------------------------------------------------
# Event file tailing
# -----------------------------------------------------------------------------

class _EventTail:
    """Follows a JSONL file that another process is appending to.

    Two things this has to get right, because both lose data silently:

      - Only complete lines are handed to the parser. The writer is routinely
        mid-record, and a half-written line is not JSON.
      - A record split across two reads is carried over, not dropped. nanochat
        flushes per record but the reader polls on a timer, so any record that
        lands across a poll boundary arrives in two pieces. The ``config`` event
        is the largest and the most important, so losing it would leave the UI
        with no resolved step count and no way to draw a progress bar.

    Offsets are byte counts and the file is read in binary, so correctness does
    not depend on any text decoder's internal position cookie.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._carry = b""

    def reset(self) -> None:
        self._offset = 0
        self._carry = b""

    def read_new(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._offset:
            # Truncated or replaced (a new run reused the path). Start over.
            self._offset = 0
            self._carry = b""
        if size == self._offset:
            return []

        try:
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                # Advance past everything actually read, and keep the incomplete
                # tail for next time.
                self._offset += len(chunk)
        except OSError:
            return []

        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            # No complete record yet: hold everything for the next read.
            self._carry += chunk
            return []

        complete = self._carry + chunk[: last_newline + 1]
        self._carry = chunk[last_newline + 1:]

        records = []
        for raw in complete.split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                # A malformed line is a bug upstream, not a reason to kill a run
                # that is otherwise making progress.
                logger.warning("nanochat event was not valid JSON: %s", raw[:200])
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records


# -----------------------------------------------------------------------------
# The run
# -----------------------------------------------------------------------------

class NanochatRun:
    """One nanochat run, from start to the last stage.

    Owns at most one child process at a time. The reader thread and the stage
    loop run on daemon threads; the lock guards the mutable state they share.
    """

    def __init__(self, run_id: str, config: NanochatConfig) -> None:
        self.run_id = run_id
        self.config = config
        self.state = RunState(run_id=run_id)
        self.state.started_at = time.time()
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._tail: Optional[_EventTail] = None
        self._metrics: dict[str, list[MetricPoint]] = {}
        self._samples: list[SampleRecord] = []
        self._benchmarks: list[BenchmarkRecord] = []
        self._log_lines: list[dict] = []
        self._log_seq = 0
        self._run_dir = account_path(f"outputs/nanochat/{run_id}")
        self._event_file = self._run_dir / "events.jsonl"
        self._stop_file = self._run_dir / "stop.request"
        self._nanochat_base = account_path("outputs/nanochat/base")
        self._listeners: list[Callable[[RunState], None]] = []

        for key in config.stages:
            self.state.stages[key] = StageState(key=key)

    # -- paths ------------------------------------------------------------
    @property
    def nanochat_base_dir(self) -> Path:
        """Where nanochat keeps checkpoints, tokenizer and data.

        Inside the account workspace rather than the user's real
        ``~/.cache/nanochat``, so accounts stay isolated and uninstalling the app
        does not leave gigabytes behind.
        """
        return self._nanochat_base

    def checkpoints_dir(self, source: str) -> Path:
        folder = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}
        return self.nanochat_base_dir / folder[source] / self.config.resolved_model_tag()

    def result_file(self, name: str) -> Path:
        return self._run_dir / name

    # -- public API -------------------------------------------------------
    def start(self) -> RunState:
        status = environment.environment_status()
        if not status.ready():
            raise NanochatNotInstalled(status.blocking_reason or "nanochat is not installed")

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self.nanochat_base_dir.mkdir(parents=True, exist_ok=True)
        # A stale sentinel from a previous run would stop this one immediately.
        self._stop_file.unlink(missing_ok=True)
        self._event_file.unlink(missing_ok=True)
        self._tail = _EventTail(self._event_file)

        self.state.status = "running"
        self.state.phase = "starting"
        self.state.message = "Starting nanochat"

        self._thread = threading.Thread(
            target=self._run_stages, name=f"nanochat-run-{self.run_id}", daemon=True
        )
        self._thread.start()
        return self.state

    def stop(self, *, save: bool = True) -> dict:
        """Ask the run to stop.

        Cooperative first: the sentinel lets the current stage finish its step and
        write a checkpoint. ``save=False`` escalates to a kill, which can leave
        the last checkpoint short of the current step.
        """
        with self._lock:
            if self.state.status != "running":
                return {"status": self.state.status, "message": "Not running"}
            self._stop_requested.set()
            self.state.message = "Stopping" if save else "Stopping without saving"

        if not save:
            self._kill_current_stage()
            return {"status": "stopping", "message": "Killed the current stage"}

        try:
            self._stop_file.parent.mkdir(parents=True, exist_ok=True)
            self._stop_file.touch()
        except OSError as exc:
            logger.warning("could not write the nanochat stop sentinel: %s", exc)
            # Without the sentinel the stage will never notice, so kill instead of
            # leaving the run wedged.
            self._kill_current_stage()
            return {"status": "stopping", "message": "Stop signal failed; stage killed"}

        return {"status": "stopping", "message": "Will stop after the current step"}

    def is_active(self) -> bool:
        return self.state.status == "running"

    def metrics(self) -> dict[str, list[dict]]:
        with self._lock:
            return {
                key: [asdict(point) for point in points]
                for key, points in self._metrics.items()
            }

    def samples(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return [asdict(s) for s in self._samples[-limit:]]

    def benchmarks(self) -> list[dict]:
        with self._lock:
            return [asdict(b) for b in self._benchmarks]

    def log_tail(self, limit: int = 200) -> list[dict]:
        with self._lock:
            return list(self._log_lines[-limit:])

    def logs_since(self, cursor: int, limit: int = _MAX_LOG_LINES_PER_FRAME) -> tuple[list[dict], int]:
        """Console lines the caller has not seen, and the cursor to ask from next.

        A monotonic sequence number is what makes this resumable. Re-sending the
        tail would duplicate lines the console already rendered every time a
        reconnect happened, and a reconnect is the normal case, not the rare one.
        """
        with self._lock:
            fresh = [entry for entry in self._log_lines if entry["seq"] > cursor][:limit]
            return fresh, (fresh[-1]["seq"] if fresh else cursor)

    # -- stage sequencing -------------------------------------------------
    def _run_stages(self) -> None:
        try:
            for key in self.config.stages:
                if self._stop_requested.is_set():
                    self._finish("stopped", "Stopped before " + key)
                    return
                stage = self.state.stages.get(key)
                if stage is None:
                    continue
                if key == "rl" and not self.config.run_rl:
                    stage.status = "skipped"
                    stage.message = "Reinforcement learning is off"
                    continue
                outcome = self._run_stage(key)
                if outcome == "ok":
                    continue
                self._finish(outcome, f"{key} {outcome}")
                return
            self._finish("completed", "Finished")
        except Exception as exc:  # a bug here must still close the run out
            logger.exception("nanochat run %s failed", self.run_id)
            self._finish("error", str(exc))

    def _stage_argv(self, key: str) -> tuple[str, list[str]]:
        """(module, argv) for a stage. Every value nanochat accepts is passed
        explicitly rather than relying on its defaults, so a run is reproducible
        from the stored config alone."""
        cfg = self.config
        tag = cfg.resolved_model_tag()

        if key == "dataset":
            return "nanochat.dataset", [
                "--dataset", cfg.dataset,
                "-n", str(cfg.num_shards),
            ]

        if key == "tokenizer":
            # The tokenizer is trained on whatever the corpus is. Re-training it
            # invalidates every existing checkpoint, so it is opt-out.
            return "scripts.tok_train", [
                "--vocab-size", str(cfg.vocab_size),
                "--max-chars", "200000000",
            ]

        if key == "pretrain":
            return "scripts.base_train", [
                "--depth", str(cfg.depth),
                "--aspect-ratio", str(cfg.aspect_ratio),
                "--head-dim", str(cfg.head_dim),
                "--max-seq-len", str(cfg.max_seq_len),
                "--window-pattern", cfg.window_pattern,
                "--dataset", cfg.dataset,
                "--target-param-data-ratio", str(cfg.param_data_ratio),
                "--num-iterations", str(cfg.num_iterations),
                "--total-batch-size", str(cfg.total_batch_size),
                "--device-batch-size", str(cfg.device_batch_size),
                "--embedding-lr", str(cfg.embedding_lr),
                "--unembedding-lr", str(cfg.unembedding_lr),
                "--matrix-lr", str(cfg.matrix_lr),
                "--scalar-lr", str(cfg.scalar_lr),
                "--weight-decay", str(cfg.weight_decay),
                "--warmup-steps", str(cfg.warmup_steps),
                "--warmdown-ratio", str(cfg.warmdown_ratio),
                "--final-lr-frac", str(cfg.final_lr_frac),
                "--eval-every", str(cfg.eval_every),
                "--eval-tokens", str(cfg.eval_tokens),
                "--core-metric-every", str(cfg.core_metric_every),
                "--core-metric-max-per-task", str(cfg.core_metric_max_per_task),
                "--sample-every", str(cfg.sample_every),
                "--save-every", str(cfg.save_every),
                "--model-tag", tag,
                "--run", "dummy",
            ] + (["--fp8"] if cfg.fp8 else [])

        if key == "base_eval":
            return "scripts.base_eval", [
                "--eval", ",".join(cfg.base_benchmarks) or "bpb",
                "--model-tag", tag,
                "--device-batch-size", str(cfg.device_batch_size),
                "--max-per-task", str(cfg.core_metric_max_per_task),
                "--json-out", str(self.result_file("base_eval.json")),
            ]

        if key == "sft":
            return "scripts.chat_sft", [
                "--model-tag", tag,
                "--max-seq-len", str(cfg.max_seq_len),
                "--device-batch-size", str(cfg.device_batch_size),
                "--total-batch-size", str(cfg.total_batch_size),
                "--num-iterations", str(cfg.sft_iterations),
                "--eval-every", str(max(1, cfg.sft_iterations // 4)),
                "--chatcore-every", str(max(1, cfg.sft_iterations // 4)),
                "--sample-every", str(cfg.sft_sample_every),
                "--sft-tasks", ",".join(cfg.sft_tasks),
                "--mmlu-epochs", str(cfg.mmlu_epochs),
                "--gsm8k-epochs", str(cfg.gsm8k_epochs),
                "--benchmarks", ",".join(cfg.chat_benchmarks),
                "--run", "dummy",
            ]

        if key == "chat_eval":
            return "scripts.chat_eval", [
                "-i", "sft",
                "--model-tag", tag,
                "-a", "|".join(cfg.chat_benchmarks),
                "-m", "256",
                "-b", str(cfg.device_batch_size),
                "--json-out", str(self.result_file("chat_eval.json")),
            ]

        if key == "rl":
            return "scripts.chat_rl", [
                "--model-tag", tag,
                "--num-epochs", str(cfg.rl_epochs),
                "--device-batch-size", str(cfg.device_batch_size),
                "--examples-per-step", str(cfg.rl_examples_per_step),
                "--num-samples", str(cfg.rl_num_samples),
                "--max-new-tokens", str(cfg.rl_max_new_tokens),
                "--eval-every", str(cfg.rl_eval_every),
                "--save-every", str(cfg.rl_save_every),
                "--run", "dummy",
            ]

        raise ValueError(f"unknown nanochat stage: {key}")

    def _run_stage(self, key: str) -> str:
        module, argv = self._stage_argv(key)
        stage = self.state.stages[key]

        with self._lock:
            stage.status = "running"
            stage.started_at = time.time()
            self.state.current_stage = key
            self.state.phase = key
            self.state.message = f"Running {key}"
            # Each stage starts from zero so a stale step from a previous stage
            # cannot make the progress bar jump.
            self.state.step = 0
            self.state.total_steps = 0
            self.state.progress_percent = 0.0

        env = environment.stage_environment(
            event_file=self._event_file,
            stop_file=self._stop_file,
            base_dir=self.nanochat_base_dir,
        )
        command = environment.stage_command(module, argv)

        if is_process_shutting_down():
            stage.status = "failed"
            stage.error = "Unsloth is shutting down"
            return "failed"

        # Free VRAM before the child allocates any. A resident chat model is kept
        # only if the run still fits alongside it, because on a consumer card the
        # alternative is an out-of-memory failure hours into training.
        try:
            self._free_vram_for_stage(key)
        except Exception:
            # Never let a teardown problem block the run: the worst case is the
            # child OOMs, which is recoverable and visible.
            logger.warning("could not free VRAM before nanochat stage %s", key, exc_info=True)

        try:
            proc = subprocess.Popen(
                command,
                cwd=str(environment.checkout_root()),
                env=env,
                # Both streams are piped into the in-app console. This used to be
                # DEVNULL, which is why opening the page flashed a black console
                # window with nothing in it: the child owned a console nobody drew
                # on, and its output went nowhere at all.
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                **child_popen_kwargs(),
                **windows_hidden_subprocess_kwargs(),
            )
        except Exception as exc:
            stage.status = "failed"
            stage.error = f"could not start {module}: {exc}"
            return "failed"

        with self._proc_lock:
            self._proc = proc
        stage.pid = proc.pid
        adopt_pid(proc.pid)

        self._append_log(f"> starting {key}: {' '.join(command)}", stage=key)

        # Both streams are drained on their own threads: a stage that writes a
        # traceback and then waits would deadlock against a full pipe, and
        # reading them on one thread would let a chatty stdout starve stderr.
        stderr_tail: list[str] = []

        def _drain(stream: Any, name: str) -> None:
            try:
                for raw in stream:
                    # Split on the newline delimiter the stream yields on, rather
                    # than trusting one read to be one line. A stage that writes
                    # several lines in a single flush would otherwise arrive as a
                    # single console entry with embedded newlines in it.
                    for piece in raw.split("\n"):
                        if name == "stderr":
                            stderr_tail.append(piece)
                            if len(stderr_tail) > 60:
                                del stderr_tail[:-60]
                        self._append_log(piece, stream=name, stage=key)
            except Exception:
                # The pipe closing as the process exits is normal, not an error.
                pass

        readers = [
            threading.Thread(
                target=_drain,
                args=(proc.stdout, "stdout"),
                name=f"nanochat-stdout-{key}",
                daemon=True,
            ),
            threading.Thread(
                target=_drain,
                args=(proc.stderr, "stderr"),
                name=f"nanochat-stderr-{key}",
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        exit_code = self._supervise(proc, key)

        # The readers are daemon threads blocked on read; closing the pipes is what
        # unblocks them. They are joined with a short timeout rather than waited on
        # outright, because a reader that has already lost the race must not be able
        # to hold up the next stage.
        for handle in (proc.stdout, proc.stderr):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        for reader in readers:
            reader.join(timeout=1.0)
        with self._proc_lock:
            self._proc = None
        stage.ended_at = time.time()

        if exit_code == 0:
            stage.status = "stopped" if self._stop_requested.is_set() else "ok"
            self._append_log(
                f"> {key} finished in {_hms(stage.duration_seconds or 0.0)}", stage=key
            )
            return "ok" if not self._stop_requested.is_set() else "stopped"
        if self._stop_requested.is_set():
            # A stage that exits non-zero because it was killed mid-step is a
            # stop, not a failure.
            stage.status = "stopped"
            self._append_log(f"> {key} stopped", stage=key)
            return "stopped"
        stage.status = "failed"
        detail = "\n".join(stderr_tail[-12:]) or f"exit code {exit_code}"
        stage.error = detail
        self._append_log(f"> {key} failed (exit {exit_code})", stream="stderr", stage=key)
        return "failed"

    def _free_vram_for_stage(self, key: str) -> list[str]:
        """Release GPU memory the stage is about to need.

        Only the GPU-hungry stages ask for this. The download and tokenizer
        stages are CPU work, and tearing down a loaded chat model before
        downloading a few hundred megabytes of parquet would be gratuitous.
        """
        if key not in ("pretrain", "sft", "rl", "base_eval", "chat_eval"):
            return []
        from routes.training_vram import coordinate_models_for_nanochat

        freed = coordinate_models_for_nanochat(
            depth = self.config.depth,
            total_batch_size = self.config.total_batch_size,
            max_seq_len = self.config.max_seq_len,
            vocab_size = self.config.vocab_size,
        )
        if freed:
            logger.info("freed VRAM before nanochat %s: %s", key, ", ".join(freed))
        return freed

    def _supervise(self, proc: subprocess.Popen, key: str) -> int:
        """Wait for a stage, pumping its events, and honouring a stop request."""
        deadline_poll = 0.0
        while True:
            if self._stop_requested.is_set():
                # Give the stage a chance to notice the sentinel and save.
                if not self._stop_file.exists():
                    self._stop_file.parent.mkdir(parents=True, exist_ok=True)
                    self._stop_file.touch()
                elapsed = time.time() - getattr(self, "_stop_requested_at", time.time())
                if elapsed > _STOP_GRACE_SECONDS:
                    self._kill_current_stage()
            self._pump_events(key)
            try:
                return proc.wait(timeout=_READ_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                continue

    # -- events -----------------------------------------------------------
    def _pump_events(self, key: str) -> None:
        if self._tail is None:
            return
        for record in self._tail.read_new():
            try:
                self._apply_event(key, record)
            except Exception:
                logger.exception("nanochat event handling failed: %s", record.get("kind"))

    def _apply_event(self, stage_key: str, record: dict) -> None:
        kind = record.get("kind")
        handler = getattr(self, f"_on_{kind}", None)
        if handler is not None:
            handler(stage_key, record)
        if kind == "log":
            self._append_log(str(record.get("line", "")))

    def _append_log(self, line: str, *, stream: str = "stdout", stage: str | None = None) -> None:
        """Buffer one console line for the in-app terminal.

        A progress bar redraws in place by writing ``\\r`` rather than ``\\n``, so
        a text-mode pipe hands over all of its passes as one line. Only the last
        pass is what a terminal would have shown, so that is what is kept;
        rendering the whole string would show every frame of the bar concatenated.
        """
        text = line.rsplit("\r", 1)[-1].strip()
        if not text:
            return
        with self._lock:
            self._log_seq += 1
            self._log_lines.append(
                {
                    "seq": self._log_seq,
                    "stream": stream,
                    "stage": stage or self.state.current_stage,
                    "line": text,
                    "ts": time.time(),
                }
            )
            if len(self._log_lines) > _MAX_LOG_LINES:
                del self._log_lines[:-_MAX_LOG_LINES]

    def _on_stage_start(self, stage_key: str, record: dict) -> None:
        with self._lock:
            self.state.message = f"{record.get('stage', stage_key)} starting"

    def _on_config(self, stage_key: str, record: dict) -> None:
        with self._lock:
            self.state.resolved_config = record
            self.state.total_steps = int(record.get("total_steps") or record.get("num_iterations") or 0)
            if record.get("num_params"):
                self.state.num_params = int(record["num_params"])
            if record.get("peak_memory_bytes"):
                self.state.peak_memory_bytes = int(record["peak_memory_bytes"])

    def _on_metric(self, stage_key: str, record: dict) -> None:
        step = int(record.get("step") or 0)
        series = "loss"
        if "loss" in record and record.get("loss") is not None:
            value = float(record["loss"])
        elif "reward" in record and record.get("reward") is not None:
            series = "reward"
            value = float(record["reward"])
        else:
            return
        total = record.get("total_steps")
        with self._lock:
            self.state.step = step
            if total:
                self.state.total_steps = int(total)
            if record.get("loss") is not None:
                self.state.loss = float(record["loss"])
            if record.get("reward") is not None:
                self.state.reward = float(record["reward"])
            if record.get("tok_per_sec") is not None:
                self.state.tok_per_sec = int(record["tok_per_sec"])
            if record.get("mfu") is not None:
                self.state.mfu = float(record["mfu"])
            if record.get("elapsed_seconds") is not None:
                self.state.elapsed_seconds = float(record["elapsed_seconds"])
            if record.get("eta_seconds") is not None:
                self.state.eta_seconds = float(record["eta_seconds"])
            if record.get("progress_percent") is not None:
                self.state.progress_percent = float(record["progress_percent"])
            elif self.state.total_steps:
                self.state.progress_percent = 100.0 * step / self.state.total_steps
            self._append_metric_locked(series, step, value)
            # Throughput and MFU ride along on every metric event and are what the
            # UI's performance charts draw, so they are recorded here rather than
            # left to be reconstructed from a single scalar. Their own series: a
            # second axis on the loss chart would read as a relationship between
            # two things that have none.
            for extra in ("tok_per_sec", "mfu"):
                raw = record.get(extra)
                if raw is None:
                    continue
                self._append_metric_locked(extra, step, float(raw))

    def _append_metric_locked(self, series: str, step: int, value: float) -> None:
        """Add a point to a named series. Caller holds ``self._lock``."""
        points = self._metrics.setdefault(series, [])
        if points and step <= points[-1].step:
            return
        points.append(
            MetricPoint(step=step, value=value, total_steps=self.state.total_steps or None)
        )
        if len(points) > _MAX_METRIC_POINTS:
            del points[: len(points) - _MAX_METRIC_POINTS]

    def _on_eval(self, stage_key: str, record: dict) -> None:
        metric = record.get("metric")
        value = record.get("value")
        step = int(record.get("step") or 0)
        with self._lock:
            if metric == "val_bpb" and value is not None:
                self.state.val_bpb = float(value)
            elif metric in ("chatcore", "chatcore_subset") and value is not None:
                self.state.chatcore = float(value)
            if record.get("peak_memory_bytes"):
                self.state.peak_memory_bytes = int(record["peak_memory_bytes"])
            # The evaluation scalars also form series, which is what makes the
            # validation curve possible: the state keeps only the latest value, so
            # a run that evaluated 40 times would otherwise show one point.
            if metric in ("val_bpb", "chatcore", "chatcore_subset") and value is not None and step > 0:
                self._append_metric_locked(
                    "chatcore" if metric == "chatcore_subset" else metric, step, float(value)
                )
        if value is None:
            return
        # Every eval event becomes a benchmark row, not just a chosen few: a
        # missing metric in this list silently drops a result the user asked for,
        # which is worse than carrying a row they did not select.
        #
        # The split is folded into the name because base_eval reports train and
        # val separately and they are not interchangeable.
        split = record.get("split")
        name = f"{metric}:{split}" if split else str(metric)
        self._benchmarks.append(BenchmarkRecord(
            stage=record.get("stage", stage_key),
            name=name,
            metric=metric,
            value=float(value),
            # pass@k is an accuracy; the others are scalars on their own scale.
            accuracy=float(value) if metric == "pass_at_k" else None,
            is_partial=bool(record.get("is_partial")),
            ts=time.time(),
        ))
        with self._lock:
            del self._benchmarks[:-_MAX_BENCHMARKS]

    def _on_benchmark(self, stage_key: str, record: dict) -> None:
        self._benchmarks.append(BenchmarkRecord(
            stage=record.get("stage", stage_key),
            name=str(record.get("name", "unknown")),
            accuracy=_as_float(record.get("accuracy")),
            baseline=_as_float(record.get("baseline")),
            centered=_as_float(record.get("centered")),
            ts=time.time(),
        ))
        with self._lock:
            del self._benchmarks[:-_MAX_BENCHMARKS]

    def _on_sample(self, stage_key: str, record: dict) -> None:
        completion = record.get("completion")
        self._samples.append(SampleRecord(
            stage=record.get("stage", stage_key),
            step=int(record.get("step") or 0),
            mode=str(record.get("mode", "completion")),
            prompt=str(record.get("prompt", "")),
            completion="" if completion is None else str(completion),
            reward=_as_float(record.get("reward")),
            advantage=_as_float(record.get("advantage")),
            ts=time.time(),
        ))
        with self._lock:
            del self._samples[:-_MAX_SAMPLES]

    def _on_checkpoint(self, stage_key: str, record: dict) -> None:
        entry = {
            "stage": record.get("stage", stage_key),
            "source": record.get("source"),
            "step": record.get("step"),
            "path": record.get("path"),
            "model_tag": record.get("model_tag"),
            "num_params": record.get("num_params"),
            "ts": time.time(),
        }
        with self._lock:
            self.state.checkpoints.append(entry)
            del self.state.checkpoints[:-64]

    def _on_warning(self, stage_key: str, record: dict) -> None:
        message = str(record.get("message", ""))
        if not message:
            return
        with self._lock:
            self.state.warnings.append(message)
            del self.state.warnings[:-40]

    def _on_stopping(self, stage_key: str, record: dict) -> None:
        with self._lock:
            self.state.message = "Stopping at the next step boundary"

    def _on_stage_end(self, stage_key: str, record: dict) -> None:
        status = record.get("status", "ok")
        stage = self.state.stages.get(stage_key)
        if stage is not None and stage.ended_at is None:
            stage.ended_at = time.time()
        if record.get("val_bpb") is not None:
            with self._lock:
                self.state.val_bpb = float(record["val_bpb"])
        if record.get("peak_memory_bytes"):
            with self._lock:
                self.state.peak_memory_bytes = int(record["peak_memory_bytes"])
        if record.get("checkpoint_dir"):
            with self._lock:
                self.state.message = f"{stage_key} finished ({status})"

    # -- teardown ---------------------------------------------------------
    def _kill_current_stage(self) -> None:
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        pid = proc.pid
        logger.info("killing nanochat stage pid=%s", pid)
        try:
            # owner_verified: we still hold the Popen, so we can prove the pid is
            # ours without re-deriving it from a start time.
            terminate_pid(pid, timeout=_KILL_TIMEOUT_SECONDS, owner_verified=True)
        except Exception:
            logger.warning("could not terminate nanochat pid %s", pid, exc_info=True)
            try:
                proc.kill()
            except Exception:
                pass

    def _finish(self, status: str, message: str) -> None:
        with self._lock:
            self.state.status = status
            self.state.phase = "idle"
            self.state.message = message
            self.state.ended_at = time.time()
            self.state.current_stage = None
            if status == "error":
                stage = self.state.current_stage
                if stage is None:
                    # Attribute the failure to the stage that was running.
                    for candidate in self.state.stages.values():
                        if candidate.status == "running":
                            stage = candidate.key
                            break
                if stage:
                    self.state.error = message
        self._stop_file.unlink(missing_ok=True)

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# Manager
# -----------------------------------------------------------------------------

class NanochatRunManager:
    """The single active run, and the lock that keeps the GPU to itself.

    One run at a time, and mutually exclusive with Unsloth's own training: both
    want the whole GPU, and running them together would have one of them OOM at an
    unpredictable point hours into a run.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: Optional[NanochatRun] = None
        self._recent: list[NanochatRun] = []

    def current(self) -> Optional[NanochatRun]:
        with self._lock:
            return self._current

    def is_active(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.is_active()

    def start(self, config: NanochatConfig) -> NanochatRun:
        from core.training.lifecycle import training_lifecycle_guard

        with self._lock:
            if self._current is not None and self._current.is_active():
                raise NanochatBusy("A nanochat run is already in progress")
            # Serialised against Unsloth training, which owns the same GPU.
            with training_lifecycle_guard():
                from core.training.training import get_training_backend

                backend = get_training_backend()
                if backend.is_training_active():
                    raise NanochatBusy("Unsloth training is running; stop it first")

                run = NanochatRun(run_id=uuid.uuid4().hex[:12], config=config)
                self._current = run
                self._recent.append(run)
                del self._recent[:-20]

        run.start()
        return run

    def stop(self, *, save: bool = True) -> dict:
        with self._lock:
            run = self._current
        if run is None:
            return {"status": "idle", "message": "No nanochat run"}
        return run.stop(save=save)

    def status(self) -> RunState:
        with self._lock:
            run = self._current
        if run is None:
            return RunState(run_id="", status="idle", phase="idle", message="No nanochat run")
        return run.state

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Stop the active run, killing if it does not comply.

        Called on backend shutdown: leaving a nanochat process running would
        outlive the app and hold VRAM, so it is stopped unconditionally rather
        than politely.
        """
        with self._lock:
            run = self._current
        if run is None or not run.is_active():
            return True
        run.stop(save=False)
        return run.wait(timeout)


_manager: Optional[NanochatRunManager] = None
_manager_lock = threading.Lock()


def get_run_manager() -> NanochatRunManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = NanochatRunManager()
        return _manager
