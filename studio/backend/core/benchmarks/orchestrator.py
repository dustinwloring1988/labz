# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sequencing a benchmark run: three passes, live progress, and stopping.

A run is a chain of child processes (see ``core/benchmarks/__init__.py`` for why
it is three and not one)::

    materialize   nanochat venv   downloads and writes items
    score         backend venv    loads the model, writes predictions
    grade         nanochat venv   task.evaluate() -> accuracy

Modelled on ``core/nanochat/orchestrator.py`` deliberately: that one already
solved the awkward parts -- a progress file another process appends to, a
cooperative stop that escalates to a kill, releasing VRAM before the child
allocates any, and refusing to start while training owns the GPU. Benchmarking
has the same awkward parts, so it reuses the shape rather than inventing a second
one.

Progress is a JSONL file the children append to and a reader thread follows, so
nothing here parses stdout for numbers and a reworded log line upstream cannot
break the UI.

What is different from nanochat, and why
---------------------------------------
* The passes are ordered by dependency, not by pipeline stage, and a failure in
  one skips the rest rather than retrying: a benchmark whose items never
  materialized cannot be scored, and a model that would not load cannot be
  graded. Retrying either would loop on a deterministic failure.
* The model is loaded once, for the whole suite, not per benchmark. Loading a 7B
  model takes minutes and the suite is minutes, so a per-benchmark load would
  dominate the run.
* Results are persisted when the run finishes, because a benchmark result that
  only exists in memory is not a leaderboard entry.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.benchmarks import catalogue
from core.nanochat import environment
from utils.paths.storage_roots import account_path
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    is_process_shutting_down,
    terminate_pid,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# How often the reader thread re-checks the event file for new bytes.
_READ_POLL_SECONDS = 0.4

# Bounds on what the UI keeps in memory. A long suite emits a progress event per
# batch, so without a cap this grows for as long as the tab is open.
_MAX_LOG_LINES = 4000
_MAX_LOG_LINES_PER_FRAME = 200
# Benchmarks per run. The picker caps this well below; this is the ceiling for a
# request that arrives with the field omitted.
_MAX_BENCHMARKS_PER_RUN = 32

# A pass that ignores the stop sentinel for this long gets killed. Generous
# because the slowest thing it may be doing is a model load or a dataset
# download, and killing mid-download loses the partial file.
_STOP_GRACE_SECONDS = 60
_KILL_TIMEOUT_SECONDS = 20

# Item cache. Shared across runs because items depend only on the benchmark and
# its dataset, so a second run of the same benchmark skips the download. Keyed by
# benchmark inside, and invalidated by the dataset stamp the materializer writes.
_ITEMS_DIRNAME = "benchmark-items"

# Progress is weighted rather than counted: materializing MMLU is one long step
# and scoring it is thousands of small ones, so an equal split would show a run
# as a quarter done for most of its life.
_PHASE_WEIGHTS = {"materialize": 0.25, "score": 0.65, "grade": 0.10}


class BenchmarkBusy(RuntimeError):
    """Another benchmark run, or an LABZ training run, owns the GPU."""


class BenchmarkUnavailable(RuntimeError):
    """The nanochat environment that owns the datasets is not set up."""


@dataclass
class BenchmarkConfig:
    """One run's settings, resolved and validated before anything starts."""

    model_id: str
    model_label: str
    model_path: str
    format: str = "safetensors"
    lora_path: Optional[str] = None
    load_in_4bit: bool = False

    benchmarks: list[str] = field(default_factory = list)
    max_problems: Optional[int] = catalogue.DEFAULT_MAX_PROBLEMS
    max_new_tokens: int = 512
    batch_size: int = 8
    max_seq_length: int = 2048
    trust_remote_code: bool = False
    hf_token: Optional[str] = None

    # A second model that scores the generative answers, after the model under
    # test has answered. Optional, and only meaningful for the generative
    # benchmarks: a categorical answer is a single letter that exact logit
    # matching already measures, so a judge there could only add noise.
    judge_model_id: Optional[str] = None
    judge_model_label: Optional[str] = None
    judge_model_path: Optional[str] = None
    judge_max_new_tokens: int = 256

    def judge_enabled(self, kind: str) -> bool:
        """Whether a judge is both configured and wanted for this kind.

        Kind is passed in rather than looked up so the caller decides, from the
        benchmark table it already read, and the two answers cannot disagree.
        """
        return bool(self.judge_model_path) and kind == "generative"

    def scoring_mode_hint(self) -> str:
        """The mode a run will use, before anything has actually been scored.

        A hint, not a fact: the scorer may fall back to generation if a tokenizer
        cannot represent the answer letters as single tokens, and the recorded
        mode comes from what it did. This is what the UI shows while queued.
        """
        return "generated" if self.format == "gguf" else "logits"


@dataclass
class ScoreState:
    """One benchmark's progress within a run."""

    key: str
    label: str = ""
    kind: str = "categorical"
    status: str = "pending"  # pending | running | complete | skipped | failed
    accuracy: Optional[float] = None
    baseline: float = 0.0
    correct: int = 0
    total: int = 0
    elapsed_seconds: float = 0.0
    truncated: bool = False
    scoring_mode: Optional[str] = None
    error: Optional[str] = None

    # What a judge model said about the answers, kept beside the exact verdict
    # rather than in place of it. nanochat's grading is the definition and is
    # deterministic; a judge is a second opinion that can be wrong in a different
    # way, so a disagreement is something to show rather than something to
    # resolve. None on a run with no judge, and on a categorical benchmark even
    # when a judge ran, because a letter is not something a second model improves.
    judge_model: Optional[str] = None
    exact_accuracy: Optional[float] = None
    judge_accuracy: Optional[float] = None
    judge_judged: int = 0
    judge_unreadable: int = 0
    # Where the two disagree, and by how much. Reported because "the judge and the
    # exact grader disagree on 38% of these" is a statement about the benchmark,
    # and hiding it would leave only the number that was chosen.
    disagreements: int = 0
    judge_raised: int = 0

    def judged(self) -> bool:
        return self.judge_accuracy is not None

    def centered(self) -> Optional[float]:
        """Share of the gap between guessing and perfect that was closed.

        The only figure comparable across benchmarks, and so what a suite
        composite is the mean of. Undefined when the baseline is 1 (nothing left
        to close) or when the run was cut short, so it stays None rather than
        being reported as a number it is not.
        """
        if self.accuracy is None or self.truncated:
            return None
        room = 1.0 - self.baseline
        if room <= 0:
            return None
        return (self.accuracy - self.baseline) / room


@dataclass
class RunState:
    """Everything the API and UI read about the active run."""

    run_id: str
    status: str = "idle"  # idle | running | completed | error | stopped
    phase: str = "idle"  # idle | materialize | score | grade | done
    message: str = ""
    error: Optional[str] = None

    model_id: str = ""
    model_label: str = ""
    format: str = "safetensors"
    lora_path: Optional[str] = None
    load_in_4bit: bool = False
    judge_model_id: Optional[str] = None
    judge_model_label: Optional[str] = None

    requested: list[str] = field(default_factory = list)
    current: Optional[str] = None
    scores: list[ScoreState] = field(default_factory = list)
    scoring_mode: Optional[str] = None

    progress_percent: float = 0.0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.ended_at or time.time()) - self.started_at

    def composite(self) -> Optional[float]:
        """Mean of the centred scores that finished untruncated.

        None until one has, so the UI never shows a composite over nothing, and
        the benchmarks included are the ones that contributed -- a suite where
        only the categorical half finished must not report as a suite score.
        """
        values = [score.centered() for score in self.scores]
        usable = [value for value in values if value is not None]
        if not usable:
            return None
        return sum(usable) / len(usable)

    def to_dict(self) -> dict:
        # Scores are mapped rather than handed to asdict: `centered` is a method
        # on ScoreState, not a field, so asdict would drop it and every score
        # would reach the UI with no centred figure -- which is the one number
        # that can be compared across benchmarks.
        return {
            "run_id": self.run_id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "error": self.error,
            "model_id": self.model_id,
            "model_label": self.model_label,
            "format": self.format,
            "lora_path": self.lora_path,
            "load_in_4bit": self.load_in_4bit,
            "judge_model_id": self.judge_model_id,
            "judge_model_label": self.judge_model_label,
            "requested": list(self.requested),
            "current": self.current,
            "scores": [{**asdict(score), "centered": score.centered()} for score in self.scores],
            "scoring_mode": self.scoring_mode,
            # Present even though it is a method, for the same reason `centered` is
            # on each score: the suite composite is the headline number on the run,
            # and omitting it would leave the live view and the header badge blank
            # for the whole run and blank on completion.
            "composite": self.composite(),
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds(),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class _EventTail:
    """Follows a JSONL file that another process is appending to.

    Byte offsets and binary reads, so correctness does not depend on any text
    decoder's internal position, and a record split across two polls is carried
    over rather than dropped -- the scoring pass flushes per record, but the
    reader polls on a timer, so a record routinely lands across a poll boundary
    and losing it would silently understate progress.
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
            # Truncated or replaced. Start over.
            self._offset = 0
            self._carry = b""
        if size == self._offset:
            return []

        try:
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset += len(chunk)
        except OSError:
            return []

        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            self._carry += chunk
            return []

        complete = self._carry + chunk[: last_newline + 1]
        self._carry = chunk[last_newline + 1 :]

        records: list[dict] = []
        for raw in complete.split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw.decode("utf-8", errors = "replace"))
            except json.JSONDecodeError:
                # A malformed line is a bug upstream, not a reason to lose a run
                # that is otherwise making progress.
                logger.warning("benchmark event was not valid JSON: %s", raw[:200])
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records


class BenchmarkRun:
    """One benchmark run, from the first download to the persisted scores."""

    def __init__(self, run_id: str, config: BenchmarkConfig) -> None:
        self.run_id = run_id
        self.config = config
        self.state = RunState(
            run_id = run_id,
            model_id = config.model_id,
            model_label = config.model_label,
            format = config.format,
            lora_path = config.lora_path,
            load_in_4bit = config.load_in_4bit,
            judge_model_id = config.judge_model_id,
            judge_model_label = config.judge_model_label,
            requested = list(config.benchmarks),
            started_at = time.time(),
        )
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._stop_requested_at = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._tail: Optional[_EventTail] = None
        self._log_lines: list[dict] = []
        self._log_seq = 0

        self._run_dir = account_path(f"outputs/benchmarks/{run_id}")
        self._event_file = self._run_dir / "events.jsonl"
        self._stop_file = self._run_dir / "stop.request"
        self._predictions_dir = self._run_dir / "predictions"
        self._results_file = self._run_dir / "results.json"
        self._config_file = self._run_dir / "config.json"
        # The judge's own answers, kept so a disagreement can be read rather than
        # only counted.
        self._judged_dir = self._run_dir / "judged"
        # Per-item exact verdicts, captured at grading time so the judge's
        # agreements and disagreements can be counted item by item. The totals
        # alone cannot say which way a disagreement went.
        self._exact_verdict: dict[tuple[str, int], bool] = {}
        # Shared across runs: items depend on the benchmark, not the model.
        self._items_dir = account_path(_ITEMS_DIRNAME)
        # nanochat's data directory, pointed inside the account workspace so
        # accounts stay isolated and a reset stays inside the app.
        self._nanochat_base = account_path("outputs/nanochat/base")

        for key in config.benchmarks:
            # The label and kind are properties of the benchmark, not of this run,
            # so they are resolved once here rather than re-read on each use. The
            # kind in particular decides whether a judge is wanted at all, and
            # re-reading the catalogue to ask that mid-run meant the answer could
            # change under a run that had already committed to a suite.
            self.state.scores.append(
                ScoreState(
                    key = key,
                    label = self._label(key),
                    kind = self._kind(key),
                    baseline = self._baseline(key),
                )
            )

    def _baseline(self, key: str) -> float:
        rows, _missing = catalogue.resolve_specs([key])
        return float(rows[0].get("baseline") or 0.0) if rows else 0.0

    def _label(self, key: str) -> str:
        rows, _missing = catalogue.resolve_specs([key])
        return str(rows[0].get("label") or key) if rows else key

    def _kind(self, key: str) -> str:
        rows, _missing = catalogue.resolve_specs([key])
        return str(rows[0].get("kind") or "categorical") if rows else "categorical"

    # -- paths ------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def items_dir(self) -> Path:
        return self._items_dir

    def _score_state(self, key: str) -> Optional[ScoreState]:
        for score in self.state.scores:
            if score.key == key:
                return score
        return None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> RunState:
        available, reason = catalogue.availability()
        if not available:
            raise BenchmarkUnavailable(reason or "nanochat has not been set up yet")

        self._run_dir.mkdir(parents = True, exist_ok = True)
        self._items_dir.mkdir(parents = True, exist_ok = True)
        self._predictions_dir.mkdir(parents = True, exist_ok = True)
        # A stale sentinel from a previous run would stop this one immediately.
        self._stop_file.unlink(missing_ok = True)
        self._event_file.unlink(missing_ok = True)
        self._tail = _EventTail(self._event_file)

        with self._lock:
            self.state.status = "running"
            self.state.phase = "materialize"
            self.state.message = "Preparing benchmarks"

        self._thread = threading.Thread(
            target = self._run_passes, name = f"benchmark-run-{self.run_id}", daemon = True
        )
        self._thread.start()
        return self.state

    def stop(self) -> dict:
        """Ask the run to stop.

        Cooperative first: the sentinel lets the current pass finish what it is
        doing, and the scoring pass polls it between batches so a stop takes
        effect with the problems already graded still reported. Only a pass that
        ignores it for too long is killed.
        """
        with self._lock:
            if self.state.status != "running":
                return {"status": self.state.status, "message": "Not running"}
            self._stop_requested.set()
            self._stop_requested_at = time.time()
            self.state.message = "Stopping"

        try:
            self._stop_file.parent.mkdir(parents = True, exist_ok = True)
            self._stop_file.touch()
        except OSError as exc:
            logger.warning("could not write the benchmark stop sentinel: %s", exc)
            self._kill_current()
            return {"status": "stopping", "message": "Stop signal failed; pass killed"}

        return {"status": "stopping", "message": "Will stop after the current batch"}

    def is_active(self) -> bool:
        return self.state.status == "running"

    def logs_since(
        self,
        cursor: int,
        limit: int = _MAX_LOG_LINES_PER_FRAME,
    ) -> tuple[list[dict], int]:
        """Console lines the caller has not seen, and the cursor to ask from next.

        A monotonic sequence number is what makes this resumable: re-sending the
        tail would duplicate lines the console already rendered on every
        reconnect, and a reconnect is the normal case here.
        """
        with self._lock:
            fresh = [entry for entry in self._log_lines if entry["seq"] > cursor][:limit]
            return fresh, (fresh[-1]["seq"] if fresh else cursor)

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- pass sequencing --------------------------------------------------
    def _run_passes(self) -> None:
        try:
            if self._stop_requested.is_set():
                self._finish("stopped", "Stopped before starting")
                return

            if not self._pass_materialize():
                return
            if self._stop_requested.is_set():
                self._finish("stopped", "Stopped before scoring")
                return
            if not self._pass_score():
                return
            if self._stop_requested.is_set():
                self._finish("stopped", "Stopped before grading")
                return

            # Grading before the judge, not after, for two reasons. It costs
            # milliseconds and produces the exact verdict, so the judge can be told
            # which answers exact marking rejected instead of re-deriving that. And
            # it means a judge that cannot load leaves real scores behind rather
            # than taking the whole run down with it.
            graded = self._pass_grade()
            if not graded:
                self._finish("error", "No benchmark produced a score")
                return

            if self._stop_requested.is_set():
                self._finish("stopped", "Stopped after grading")
                return

            # Last, and in its own process, so the model under test has already
            # exited and the judge is the only thing in VRAM.
            self._pass_judge()

            self._finish("completed", "Finished")
        except Exception as exc:  # a bug here must still close the run out
            logger.exception("benchmark run %s failed", self.run_id)
            self._finish("error", str(exc))

    def _pass_judge(self) -> None:
        """Score the generative answers with a second model, if one is configured.

        Never fails the run. A judge is an optional second opinion, so a judge
        that will not load, or a verdict it could not format, leaves the exact
        scores exactly as they were and says so on the run.
        """
        wanted = [key for key in self.config.benchmarks if self._judge_wanted(key)]
        if not wanted:
            return
        if not self.config.judge_model_path:
            return

        self._set_phase("judge", f"Judging with {self.config.judge_model_label}")
        self._append_log(f"> judge: {self.config.judge_model_label}")

        config = {
            **self._score_pass_config(),
            "judge_model_id": self.config.judge_model_id,
            "judge_model_label": self.config.judge_model_label,
            "judge_model_path": self.config.judge_model_path,
            "judge_max_new_tokens": self.config.judge_max_new_tokens,
            "judge_benchmarks": wanted,
            "judged_dir": str(self._judged_dir),
            "judge_results": str(self._run_dir / "judge.json"),
        }
        try:
            self._config_file.write_text(
                json.dumps(config, ensure_ascii = False, indent = 2), encoding = "utf-8"
            )
        except OSError as exc:
            self._append_log(f"judge: could not write its configuration - {exc}")
            return

        self._free_vram()

        exit_code = self._run_child(
            "judge",
            [
                sys.executable,
                "-m",
                "core.benchmarks.judge",
                "--config",
                str(self._config_file),
            ],
        )
        self._absorb_judge_results()
        if exit_code != 0:
            self._append_log("judge: did not finish; the exact scores are unaffected")

    def _judge_wanted(self, key: str) -> bool:
        """Whether a judge applies to this benchmark.

        Reads the kind the run resolved when it was built, rather than asking the
        catalogue again. A run cannot start without nanochat, so the kind is
        already known; re-reading it here would mean the answer could differ from
        the one the run was configured against.
        """
        score = self._score_state(key)
        return self.config.judge_enabled(score.kind if score else "categorical")

    def _absorb_judge_results(self) -> None:
        """Fold the judge's verdicts in beside the exact ones.

        The exact verdict is never overwritten. It is what nanochat defines, it
        is deterministic, and the whole reason to run a judge is that the two can
        differ -- so the difference is the result, not a problem to resolve.
        """
        try:
            with (self._run_dir / "judge.json").open("r", encoding = "utf-8") as handle:
                results = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(results, dict):
            return

        with self._lock:
            for key, row in (results or {}).items():
                score = self._score_state(key)
                if score is None or not isinstance(row, dict):
                    continue
                accuracy = _as_float(row.get("judge_accuracy"))
                judged = int(row.get("judge_judged") or 0)
                unreadable = int(row.get("judge_unreadable") or 0)
                examined = judged + unreadable
                score.exact_accuracy = score.accuracy
                score.judge_accuracy = accuracy
                score.judge_judged = judged
                score.judge_unreadable = unreadable
                score.judge_model = row.get("judge_model")
                score.elapsed_seconds += float(row.get("judge_elapsed_seconds") or 0.0)
                if accuracy is None or judged == 0:
                    # Nothing to average. This is the one case that cannot be
                    # reported, because a judge that read no answer at all has not
                    # measured the benchmark -- as opposed to one that read some of
                    # it, which has measured part of the suite and is reported as
                    # such.
                    score.error = "the judge returned no readable verdict"
                    continue
                # The judge is the reported figure for a generative benchmark,
                # because that is what the user asked it for and because it is
                # the reading that does not punish a correct answer for its
                # formatting. The exact verdict stays on the row next to it.
                score.accuracy = accuracy
                score.correct = int(row.get("judge_correct") or 0)
                # The denominator is the readable verdicts, because that is the
                # population `judge_accuracy` is an average over and accuracy has
                # to stay equal to correct / total. The answers that went unread
                # are not silently folded in as failures; they are reported as
                # `judge_unreadable` and shown as coverage beside the score.
                score.total = judged
                score.scoring_mode = "judged"
                if unreadable:
                    # A judge that read part of a suite has still measured part of
                    # it, and the number is worth seeing. What is not acceptable is
                    # seeing it without its denominator, so this says so in the
                    # log and the row carries the counts. A partial judge is also
                    # ranked below every exact row, so it cannot outrank a
                    # measurement that covered the whole suite.
                    self._append_log(
                        f"> {key}: the judge read {judged} of {examined} answers; "
                        f"{unreadable} were not readable and are not counted as wrong"
                    )
                self._note_disagreement(score, row)
        self._refresh_run_scoring_mode()

    def _refresh_run_scoring_mode(self) -> None:
        """Recompute the run's mode from what its scores ended up being.

        The grade pass sets the run's mode, and the judge pass then changes
        individual scores to `judged` without touching it. That left a judged run
        reporting `generated`, which is worse than a stale label: the leaderboard
        groups and ranks on the run's mode, so the row would have been filed with
        the exact-answer runs and the separate judged group would never appear.

        A run whose scores disagree about their method is `mixed` rather than
        picking one, which is also where the board already ranks it -- below every
        single-method row, since it is not a clean measurement of anything.
        """
        with self._lock:
            modes = {
                score.scoring_mode
                for score in self.state.scores
                if score.status == "complete" and score.scoring_mode
            }
            if not modes:
                return
            self.state.scoring_mode = modes.pop() if len(modes) == 1 else "mixed"

    def _note_disagreement(self, score, row: dict) -> None:
        """Record how the judge's verdict differs from the exact one.

        Counted from the per-item records rather than from the totals, because the
        totals cannot say *which way* each disagreement went, and a judge that
        only ever raises a score is a different finding from one that lowers half
        of them.
        """
        path = self._judged_dir / f"{score.key}.jsonl"
        raised = 0
        lowered = 0
        for record in _load_jsonl(path):
            verdict = record.get("verdict")
            if not isinstance(verdict, dict) or record.get("skipped"):
                continue
            index = record.get("index")
            exact_here = self._exact_verdict.get((score.key, index))
            if exact_here is None:
                continue
            judged_here = bool(verdict.get("correct"))
            if judged_here == bool(exact_here):
                continue
            if judged_here:
                raised += 1
            else:
                lowered += 1
        score.judge_raised = raised
        score.disagreements = raised + lowered
        if score.disagreements:
            logger.info(
                "benchmark judge disagreed with exact matching on %s items (%d raised, %d lowered)",
                score.key,
                score.disagreements,
                raised,
                lowered,
            )

    def _pass_materialize(self) -> bool:
        self._set_phase("materialize", "Preparing benchmarks")
        self._append_log(f"> materializing {', '.join(self.config.benchmarks)}")

        command = [
            str(environment.venv_python()),
            str(_pass_path("materialize.py")),
            # No --checkout: the pass reads UNSLOTH_NANOCHAT_CHECKOUT, which
            # _run_child sets. One source, so the path the child resolves its
            # imports from is the same one the parent resolved.
            "--out",
            str(self._items_dir),
            "--keys",
            ",".join(self.config.benchmarks),
        ]
        if self.config.max_problems is not None:
            # Only sent when there is a cap. Sending 0 for "no cap" would read as
            # a cap of zero and materialize nothing, and the run would then fail
            # with "no benchmark items could be prepared" -- a message about the
            # wrong thing entirely, for a request that asked for everything.
            command += ["--max-problems", str(self.config.max_problems)]

        self._run_child(
            "materialize",
            command,
            # Items are shared, so the stop sentinel must not reach this pass: a
            # stop during a run should not leave the cache half-written for the
            # next one.
            use_stop_file = False,
        )

        manifest = self._read_manifest()
        if not manifest.get("items"):
            self._fail(self._materialize_failure(self.config.benchmarks, manifest))
            return False

        for key, detail in (manifest.get("failures") or {}).items():
            score = self._score_state(key)
            if score is not None:
                score.status = "failed"
                score.error = str(detail)

        ready = [key for key in self.config.benchmarks if key in manifest.get("items", {})]
        if not ready:
            # Reachable when the shared item cache still holds a previous run's
            # benchmarks, so the manifest is non-empty while nothing this run
            # asked for is in it.
            self._fail(self._materialize_failure(self.config.benchmarks, manifest))
            return False

        # Narrow the run to what actually materialized. Scoring a benchmark with
        # no items would report a zero it did not earn.
        self.config.benchmarks = ready
        self.state.requested = list(ready)
        with self._lock:
            keep = set(ready)
            self.state.scores = [score for score in self.state.scores if score.key in keep]
        self._set_progress(0.25)
        return True

    def _materialize_failure(self, requested: list[str], manifest: dict) -> str:
        """Why no benchmark was ready, in terms of the benchmarks asked for.

        The manifest is shared across runs, so "the manifest is empty" and "the
        manifest has other benchmarks in it" are different situations, and naming
        the per-benchmark reason is the difference between a message the user can
        act on and one they can only re-run and watch fail again.
        """
        failures = manifest.get("failures") or {}
        named = [f"{key} ({reason})" for key, reason in failures.items() if key in requested]
        if named:
            return "Could not prepare: " + "; ".join(named)
        return "No benchmark items could be prepared"

    def _read_manifest(self) -> dict:
        try:
            with (self._items_dir / "manifest.json").open("r", encoding = "utf-8") as handle:
                parsed = json.load(handle)
            return parsed if isinstance(parsed, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _score_pass_config(self) -> dict:
        """The scoring pass's configuration, shared with the judge pass.

        One definition rather than two so the judge cannot end up with a
        different view of which items or which model than the run it is scoring.
        """
        return {
            "model_id": self.config.model_id,
            "model_path": self.config.model_path,
            "format": self.config.format,
            "lora_path": self.config.lora_path,
            "load_in_4bit": self.config.load_in_4bit,
            "benchmarks": list(self.config.benchmarks),
            "max_problems": self.config.max_problems,
            "max_new_tokens": self.config.max_new_tokens,
            "batch_size": self.config.batch_size,
            "max_seq_length": self.config.max_seq_length,
            "trust_remote_code": self.config.trust_remote_code,
            "hf_token": self.config.hf_token,
            "items_dir": str(self._items_dir),
            "predictions_dir": str(self._predictions_dir),
        }

    def _pass_score(self) -> bool:
        self._set_phase("score", f"Scoring {self.config.model_label}")

        # Written to disk rather than passed as argv: it is a dozen fields, and a
        # path is not subject to shell quoting on any platform.
        try:
            self._config_file.write_text(
                json.dumps(self._score_pass_config(), ensure_ascii = False, indent = 2),
                encoding = "utf-8",
            )
        except OSError as exc:
            self._fail(f"could not write the run configuration: {exc}")
            return False

        # The model is about to be loaded, which is the largest allocation in the
        # run. Anything resident from chat would have to be given back first or
        # the load fails on a card that is not actually full.
        self._free_vram()

        exit_code = self._run_child(
            "score",
            [
                sys.executable,
                "-m",
                "core.benchmarks.score",
                "--config",
                str(self._config_file),
            ],
        )
        if self._stop_requested.is_set():
            return True
        if exit_code != 0:
            # The scorer reports per-benchmark failures through events, so this
            # is only reached when it could not run at all.
            self._fail("the model could not be loaded or scored")
            return False
        return True

    def _free_vram(self) -> list[str]:
        """Release chat and STT models before the scoring pass loads a model.

        Unconditional, unlike the nanochat path which weighs the run's own
        requirement first: how much a benchmark needs depends on the model being
        benchmarked and the quantisation chosen, and getting that wrong means an
        out-of-memory failure minutes into a run. Handing the memory back is
        cheap, and a resident chat model reloads on the next message.
        """
        try:
            from routes.training_vram import (
                free_chat_models_for_training,
                free_stt_model_for_training,
            )
            freed = list(free_stt_model_for_training(reason = "benchmark run"))
            freed += list(free_chat_models_for_training(reason = "benchmark run"))
        except Exception:
            # Never let a teardown problem block the run: the worst case is that
            # the load fails, which is visible and recoverable.
            logger.warning("could not free VRAM before benchmarking", exc_info = True)
            return []
        if freed:
            logger.info("freed VRAM before benchmarking: %s", ", ".join(freed))
            self._append_log(f"released for this run: {', '.join(freed)}")
        return freed

    def _pass_grade(self) -> bool:
        self._set_phase("grade", "Grading")
        exit_code = self._run_child(
            "grade",
            [
                str(environment.venv_python()),
                str(_pass_path("grade.py")),
                "--items",
                str(self._items_dir),
                "--predictions",
                str(self._predictions_dir),
                "--results",
                str(self._results_file),
                "--exact-dir",
                str(self._judged_dir),
                "--keys",
                ",".join(self.config.benchmarks),
            ],
        )
        self._absorb_results()
        graded = any(score.status == "complete" for score in self.state.scores)
        if graded:
            # Grading is the last pass, so reaching here means the bar is full
            # whether or not every benchmark produced a score. A suite where one
            # benchmark failed is still a suite that ran.
            self._set_progress(1.0)
        elif exit_code != 0:
            self._fail("grading produced no scores")
        return graded

    def _absorb_results(self) -> None:
        """Fold the grader's output into the run state.

        Tolerant by design: a run that graded three of four benchmarks has three
        real scores, and losing them because the fourth failed would be the wrong
        trade.
        """
        try:
            with self._results_file.open("r", encoding = "utf-8") as handle:
                results = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(results, dict):
            return

        with self._lock:
            for key, row in (results.get("scores") or {}).items():
                score = self._score_state(key)
                if score is None:
                    continue
                score.status = "complete"
                score.accuracy = _as_float(row.get("accuracy"))
                score.correct = int(row.get("correct") or 0)
                score.total = int(row.get("total") or 0)
                score.elapsed_seconds = float(row.get("elapsed_seconds") or 0.0)
                score.truncated = _is_truncated(self.config.max_problems, score.total)
            for key, detail in (results.get("failures") or {}).items():
                score = self._score_state(key)
                if score is not None and score.status != "complete":
                    score.status = "failed"
                    score.error = str(detail)

            self._capture_exact_verdicts(results)
            mode = self._mode_from_predictions()
            if mode:
                self.state.scoring_mode = mode
                for score in self.state.scores:
                    if score.status == "complete":
                        score.scoring_mode = mode

    def _capture_exact_verdicts(self, results: dict) -> None:
        """Ask the grader which items it marked right, one verdict per item.

        The grader does not write a per-item file, so this re-asks it only if one
        exists; where it does not, the disagreement count falls back to comparing
        totals. Recorded because the totals cannot say which *way* a disagreement
        went, and a judge that only ever raises a score is a different finding
        from one that lowers half of them.
        """
        for key, row in (results.get("scores") or {}).items():
            path = self._judged_dir / f"{key}.exact.jsonl"
            if not path.exists():
                continue
            for record in _load_jsonl(path):
                index = record.get("index")
                verdict = record.get("exact")
                if isinstance(index, int) and isinstance(verdict, bool):
                    self._exact_verdict[(key, index)] = verdict

    def _mode_from_predictions(self) -> Optional[str]:
        """Which scoring path actually ran, read off what was written.

        Taken from the predictions rather than assumed from the format: the
        safetensors scorer falls back to generation when a tokenizer cannot
        represent the answer letters as single tokens, and a run that quietly
        reported "logits" then would be claiming a measurement it did not make.
        """
        modes: set[str] = set()
        for key in self.config.benchmarks:
            path = self._predictions_dir / f"{key}.jsonl"
            if not path.exists():
                continue
            try:
                with path.open("r", encoding = "utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if "letter_scores" in row:
                            modes.add("logits")
                        else:
                            modes.add("generated")
                        break
            except (OSError, json.JSONDecodeError):
                continue
        if not modes:
            return None
        if len(modes) > 1:
            return "mixed"
        return next(iter(modes))

    # -- child process ----------------------------------------------------
    def _run_child(
        self,
        phase: str,
        command: list[str],
        *,
        use_stop_file: bool = True,
    ) -> int:
        """Run one pass, pumping its events, and honouring a stop request."""
        if is_process_shutting_down():
            self._fail("LABZ is shutting down")
            return 1

        env = environment.stage_environment(
            event_file = self._event_file,
            stop_file = self._stop_file if use_stop_file else None,
            base_dir = self._nanochat_base,
        )
        # The two nanochat passes import from the checkout, and this file is
        # invoked by absolute path from the backend tree, so the checkout is not on
        # their sys.path. Passed as an env var rather than left to them to find
        # from argv, because an import that resolves at the top of a module and
        # fails there is a much worse failure than one that never gets that far.
        env["UNSLOTH_NANOCHAT_CHECKOUT"] = str(environment.checkout_root())

        if phase == "score":
            # Only the scoring pass needs this. It runs in this interpreter's
            # environment rather than nanochat's, so the backend tree has to be
            # importable for `core.benchmarks.score` to resolve. Scoped to that one
            # pass on purpose: putting the backend tree on the nanochat passes'
            # path would make the whole backend importable inside a virtualenv
            # whose torch differs from it, which is the confusion this whole
            # three-way split exists to avoid.
            backend_root = str(_backend_root())
            env["PYTHONPATH"] = os.pathsep.join([backend_root, env.get("PYTHONPATH", "")]).rstrip(
                os.pathsep
            )

        self._append_log(f"> {phase}: {' '.join(command[1:])}")

        try:
            proc = subprocess.Popen(
                command,
                cwd = str(_backend_root()),
                env = env,
                # Both streams are piped into the in-app console. This used to be
                # DEVNULL in the nanochat path, which meant opening the page
                # flashed a console window nothing drew on.
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                stdin = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                **child_popen_kwargs(),
                **windows_hidden_subprocess_kwargs(),
            )
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"{phase}: could not start - {exc}")
            return 1

        with self._proc_lock:
            self._proc = proc
        adopt_pid(proc.pid)

        stderr_tail: list[str] = []

        def _drain(stream, name: str) -> None:
            try:
                for raw in stream:
                    # Split on the newline the stream yields on rather than
                    # trusting one read to be one line: a pass that writes several
                    # lines in a single flush would otherwise arrive as one
                    # console entry with newlines in it.
                    for piece in raw.split("\n"):
                        if name == "stderr":
                            stderr_tail.append(piece)
                            if len(stderr_tail) > 60:
                                del stderr_tail[:-60]
                        self._append_log(piece, phase = phase)
            except Exception:
                # The pipe closing as the process exits is normal.
                pass

        readers = [
            threading.Thread(
                target = _drain,
                args = (proc.stdout, "stdout"),
                name = f"benchmark-{phase}-stdout",
                daemon = True,
            ),
            threading.Thread(
                target = _drain,
                args = (proc.stderr, "stderr"),
                name = f"benchmark-{phase}-stderr",
                daemon = True,
            ),
        ]
        for reader in readers:
            reader.start()

        exit_code = self._supervise(proc, phase)

        for handle in (proc.stdout, proc.stderr):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        for reader in readers:
            reader.join(timeout = 1.0)

        with self._proc_lock:
            self._proc = None

        return exit_code

    def _supervise(self, proc: subprocess.Popen, phase: str) -> int:
        while True:
            if self._stop_requested.is_set() and phase != "materialize":
                if not self._stop_file.exists():
                    self._stop_file.parent.mkdir(parents = True, exist_ok = True)
                    self._stop_file.touch()
                if time.time() - self._stop_requested_at > _STOP_GRACE_SECONDS:
                    self._kill_current()
            self._pump_events()
            try:
                return proc.wait(timeout = _READ_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                continue

    def _kill_current(self) -> None:
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        pid = proc.pid
        logger.info("killing benchmark pass pid=%s", pid)
        try:
            # owner_verified: we still hold the Popen, so the pid is provably ours
            # without re-deriving it from a start time.
            terminate_pid(pid, timeout = _KILL_TIMEOUT_SECONDS, owner_verified = True)
        except Exception:
            logger.warning("could not terminate benchmark pid %s", pid, exc_info = True)
            try:
                proc.kill()
            except Exception:
                pass

    # -- events -----------------------------------------------------------
    def _pump_events(self) -> None:
        if self._tail is None:
            return
        for record in self._tail.read_new():
            kind = record.get("kind")
            handler = getattr(self, f"_on_{kind}", None)
            if handler is not None:
                try:
                    handler(record)
                except Exception:
                    logger.exception("benchmark event handling failed: %s", record.get("kind"))
            if kind == "log":
                self._append_log(str(record.get("line", "")))

    def _append_log(
        self,
        line: str,
        *,
        phase: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._log_seq += 1
            self._log_lines.append(
                {
                    "seq": self._log_seq,
                    "line": line,
                    "stream": "stderr" if line.startswith("Traceback") else "stdout",
                    "phase": phase,
                    "ts": time.time(),
                }
            )
            overflow = len(self._log_lines) - _MAX_LOG_LINES
            if overflow > 0:
                del self._log_lines[:overflow]

    def _on_progress(self, record: dict) -> None:
        """A scoring progress tick: move the bar within the score phase."""
        scored = int(record.get("scored") or 0)
        total = int(record.get("total") or 0)
        if total <= 0:
            return
        weight = _PHASE_WEIGHTS["score"]
        self._set_progress(_PHASE_WEIGHTS["materialize"] + weight * (scored / total))

    def _on_scored(self, record: dict) -> None:
        key = str(record.get("key") or "")
        with self._lock:
            score = self._score_state(key)
            if score is not None:
                score.status = "running"
                score.total = int(record.get("total") or score.total)
            self.state.current = key or None

    def _on_scoring_complete(self, record: dict) -> None:
        mode = record.get("mode")
        with self._lock:
            if isinstance(mode, str) and mode:
                self.state.scoring_mode = mode

    def _on_score_skipped(self, record: dict) -> None:
        with self._lock:
            score = self._score_state(str(record.get("key") or ""))
            if score is not None:
                score.status = "skipped"
                score.error = str(record.get("reason") or "skipped")

    def _on_score_failed(self, record: dict) -> None:
        with self._lock:
            score = self._score_state(str(record.get("key") or ""))
            if score is not None:
                score.status = "failed"
                score.error = str(record.get("error") or "scoring failed")

    def _on_load_failed(self, record: dict) -> None:
        self._fail(str(record.get("error") or "the model could not be loaded"))

    def _on_graded(self, record: dict) -> None:
        key = str(record.get("key") or "")
        with self._lock:
            score = self._score_state(key)
            if score is not None:
                score.status = "running"

    def _on_device_summary(self, record: dict) -> None:
        name = record.get("device_name")
        if name:
            self._append_log(f"device: {name}")

    def _on_model_loaded(self, record: dict) -> None:
        if not record.get("chat_template"):
            self._append_log("this model has no chat template; it is being scored on raw text")

    # -- state ------------------------------------------------------------
    def _set_phase(self, phase: str, message: str) -> None:
        with self._lock:
            self.state.phase = phase
            self.state.message = message

    def _set_progress(self, fraction: float) -> None:
        with self._lock:
            self.state.progress_percent = max(0.0, min(100.0, round(fraction * 100.0, 1)))

    def _fail(self, message: str) -> None:
        self._finish("error", message)

    def _finish(self, status: str, message: str) -> None:
        with self._lock:
            self.state.status = status
            self.state.phase = "done" if status == "completed" else "idle"
            self.state.message = message
            self.state.current = None
            if status == "error":
                self.state.error = message
            if status == "completed":
                self.state.progress_percent = 100.0
        self._stop_file.unlink(missing_ok = True)
        if status == "completed":
            self._persist()

    def _persist(self) -> None:
        """Write the run to the database so the leaderboard can rank it.

        A failure here is logged rather than raised: the scores are real and on
        disk, and turning a completed run into an error because the write failed
        would discard a result the user waited an hour for.
        """
        try:
            from storage import benchmark_runs_db
            benchmark_runs_db.save_run(self.state.to_dict())
        except Exception:
            logger.exception("could not persist benchmark run %s", self.run_id)


def _load_jsonl(path: Path) -> list[dict]:
    """Rows from a JSONL file, skipping any that will not parse.

    Tolerant because these files are written by a pass that may have been killed
    mid-write, and a truncated last line should cost one record rather than the
    whole comparison.
    """
    rows: list[dict] = []
    try:
        with Path(path).open("r", encoding = "utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
    except OSError:
        return []
    return rows


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # NaN and infinity have no JSON form; a score is a fraction, so anything
    # outside 0..1 is a bug upstream and is dropped rather than stored.
    if result != result or result in (float("inf"), float("-inf")):
        return None
    if result < 0.0 or result > 1.0:
        return None
    return result


def _is_truncated(max_problems: Optional[int], total: int) -> bool:
    """Whether a benchmark was cut short by the run's problem cap.

    Decided from the cap and the count, not from the benchmark declaring a total:
    a truncated score is an estimate from a sample, and the leaderboard has to be
    able to say so.
    """
    return bool(max_problems and total and total >= max_problems)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pass_path(name: str) -> Path:
    """The path to one of the passes, resolved from this file.

    Absolute rather than relative so the child does not depend on a working
    directory, and derived from ``__file__`` so a relocated install still finds
    them.
    """
    return Path(__file__).resolve().parent / name


class BenchmarkRunManager:
    """The single active run, and the lock that keeps the GPU to itself.

    One run at a time, and mutually exclusive with LABZ's own training and
    nanochat's: all three want the whole GPU, and running two would have one of
    them fail out of memory at an unpredictable point.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: Optional[BenchmarkRun] = None

    def current(self) -> Optional[BenchmarkRun]:
        with self._lock:
            return self._current

    def is_active(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.is_active()

    def start(self, config: BenchmarkConfig) -> BenchmarkRun:
        from core.training.lifecycle import training_lifecycle_guard

        if not config.benchmarks:
            raise BenchmarkUnavailable("Select at least one benchmark")
        if len(config.benchmarks) > _MAX_BENCHMARKS_PER_RUN:
            raise BenchmarkUnavailable("Too many benchmarks in one run")

        blocked = catalogue.unsupported_keys(config.benchmarks)
        if blocked:
            reasons = "; ".join(f"{key} ({reason})" for key, reason in blocked.items())
            raise BenchmarkUnavailable(f"Cannot run on this host: {reasons}")

        with self._lock:
            if self._current is not None and self._current.is_active():
                raise BenchmarkBusy("A benchmark run is already in progress")
            with training_lifecycle_guard():
                from core.training.training import get_training_backend

                backend = get_training_backend()
                if backend.is_training_active():
                    raise BenchmarkBusy("LABZ training is running; stop it first")

                from core.nanochat.orchestrator import get_run_manager as nanochat_runs

                if nanochat_runs().is_active():
                    raise BenchmarkBusy("A nanochat run is in progress; stop it first")

                run = BenchmarkRun(run_id = uuid.uuid4().hex[:12], config = config)
                self._current = run

        run.start()
        return run

    def stop(self) -> dict:
        with self._lock:
            run = self._current
        if run is None:
            return {"status": "idle", "message": "No benchmark run"}
        return run.stop()

    def status(self) -> RunState:
        with self._lock:
            run = self._current
        if run is None:
            return RunState(run_id = "")
        return run.state

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Stop the active run, killing if it does not comply.

        Called on backend shutdown. A scoring process left running would outlive
        the app holding VRAM, so it is stopped unconditionally.
        """
        with self._lock:
            run = self._current
        if run is None or not run.is_active():
            return True
        run.stop()
        return run.wait(timeout)


_manager: Optional[BenchmarkRunManager] = None
_manager_lock = threading.Lock()


def get_run_manager() -> BenchmarkRunManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = BenchmarkRunManager()
        return _manager
