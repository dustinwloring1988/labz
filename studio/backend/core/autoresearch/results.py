# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""results.tsv: the experiment ledger, and the checkpoints that go with it.

autoresearch's whole record of a run is one tab-separated file the agent appends
to, with the header program.md specifies::

    commit  val_bpb  memory_gb  status  description

This module reads and writes that file, and keeps the per-experiment checkpoints
that ``analysis.ipynb`` and the chat tab both need.

Two things here are not in the project:

  - **A crash row.** program.md says a run that produced no val_bpb is a crash,
    and the ledger records crashes as ``0.000000`` / ``0.0``. That is honest as a
    log but useless as data: a zero sorts as the best possible score, so any
    chart or "best experiment" computed over the file naively picks a crash. Rows
    written here carry the crash's real status so the two can be told apart.

  - **Per-experiment checkpoints.** train.py saves ``checkpoint_pre_eval.pt`` into
    its working directory and overwrites it every run, so after two experiments
    the model that scored best is gone. Each experiment's weights are copied out
    under ``.checkpoints/`` with the score that earned them, which is what makes
    "chat with the model from experiment 7" a thing that can be asked.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# The project's own columns, in the order program.md writes them. Kept verbatim:
# analysis.ipynb reads this file positionally, and the agent appends to it, so a
# different column order would silently break the project's own analysis.
RESULTS_FILENAME = "results.tsv"
RESULTS_COLUMNS = ("commit", "val_bpb", "memory_gb", "status", "description")
RESULTS_HEADER = "\t".join(RESULTS_COLUMNS)

# train.py's checkpoint name, relative to the checkout root.
TRAIN_CHECKPOINT_NAME = "checkpoint_pre_eval.pt"
TRAIN_LOG_NAME = "run.log"
# Where per-experiment weights are kept, relative to the checkout root. Gitignored
# by environment.install() so a commit never stages a 200MB file.
CHECKPOINT_DIR_NAME = ".checkpoints"

STATUS_KEEP = "keep"
STATUS_DISCARD = "discard"
STATUS_CRASH = "crash"

# The value the project's own instructions say to write for a run that produced
# no score. Recognised on read so a crash written by the agent is shown as a
# crash rather than as a suspiciously good result.
_PLACEHOLDER_VALUES = {"0.000000", "0.0", "0", "0.00"}


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


@dataclass
class ExperimentResult:
    """One row of the ledger, plus the fields the studio adds."""

    index: int
    commit: str = ""
    val_bpb: Optional[float] = None
    memory_gb: Optional[float] = None
    status: str = STATUS_CRASH
    description: str = ""
    # Set when the row is a crash, so the UI can say why rather than showing a
    # zero as if it were a score.
    error: Optional[str] = None
    # True when this row is the best score so far. The running frontier, which is
    # the thing worth plotting rather than every experiment.
    is_best: bool = False
    checkpoint: Optional[str] = None
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "commit": self.commit,
            "val_bpb": self.val_bpb,
            "memory_gb": self.memory_gb,
            "status": self.status,
            "description": self.description,
            "error": self.error,
            "is_best": self.is_best,
            "checkpoint": self.checkpoint,
            "ts": self.ts,
        }


@dataclass
class ResultsSummary:
    rows: list[ExperimentResult] = field(default_factory=list)
    exists: bool = False

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def kept(self) -> int:
        return sum(1 for row in self.rows if row.status == STATUS_KEEP)

    @property
    def discarded(self) -> int:
        return sum(1 for row in self.rows if row.status == STATUS_DISCARD)

    @property
    def crashed(self) -> int:
        return sum(1 for row in self.rows if row.status == STATUS_CRASH)

    def scored(self) -> list[ExperimentResult]:
        """Rows with a real score, best first.

        Crashes are excluded rather than included as zeros: a zero is not a
        score, and letting one into a min() invents a best experiment that never
        happened.
        """
        return sorted(
            (row for row in self.rows if row.val_bpb is not None and not _is_placeholder(str(row.val_bpb))),
            key=lambda row: row.val_bpb,  # type: ignore[arg-type]
        )

    def best(self) -> Optional[ExperimentResult]:
        rows = self.scored()
        return rows[0] if rows else None

    def baseline(self) -> Optional[ExperimentResult]:
        """The first experiment that produced a score.

        This is what the project's own analysis treats as the baseline, because
        a crash on the first attempt leaves nothing to measure against.
        """
        for row in self.rows:
            if row.val_bpb is not None and not _is_placeholder(str(row.val_bpb)):
                return row
        return None

    def to_dict(self) -> dict:
        best = self.best()
        baseline = self.baseline()
        return {
            "rows": [row.to_dict() for row in self.rows],
            "exists": self.exists,
            "total": self.total,
            "kept": self.kept,
            "discarded": self.discarded,
            "crashed": self.crashed,
            "best_val_bpb": best.val_bpb if best else None,
            "best_index": best.index if best else None,
            "best_commit": best.commit if best else None,
            "baseline_val_bpb": baseline.val_bpb if baseline else None,
        }


# -----------------------------------------------------------------------------
# Reading
# -----------------------------------------------------------------------------

def results_path(root: Path) -> Path:
    return root / RESULTS_FILENAME


def checkpoint_dir(root: Path) -> Path:
    return root / CHECKPOINT_DIR_NAME


def _parse_row(line: str, index: int) -> Optional[ExperimentResult]:
    """One TSV line into a row.

    Tolerant by design. The file is written by an LLM, so a row can have too few
    columns, a trailing tab, or a description containing a stray tab. A row that
    cannot be read is dropped rather than raising, because one malformed line
    should not blank the whole history in the UI.
    """
    if not line.strip() or line.lstrip().startswith("#"):
        return None
    parts = line.rstrip("\n").rstrip("\r").split("\t")
    if len(parts) < len(RESULTS_COLUMNS):
        # A short row is still worth showing if it at least named a status.
        if len(parts) < 2:
            return None
        parts = parts + [""] * (len(RESULTS_COLUMNS) - len(parts))

    commit, raw_bpb, raw_memory, status, description = parts[:5]
    status = status.strip().lower() or STATUS_CRASH
    if status not in (STATUS_KEEP, STATUS_DISCARD, STATUS_CRASH):
        status = STATUS_CRASH

    val_bpb = _to_float(raw_bpb)
    memory_gb = _to_float(raw_memory)

    # A zero in either column is the project's placeholder for "this run did not
    # finish", so a row carrying one is a crash whatever its status column says.
    crashed = status == STATUS_CRASH or (
        val_bpb is not None and _is_placeholder(str(val_bpb))
    )
    if crashed:
        val_bpb = None
        error = "the run did not finish, so it has no score"
    else:
        error = None

    return ExperimentResult(
        index=index,
        commit=commit.strip(),
        val_bpb=val_bpb,
        memory_gb=memory_gb,
        status=STATUS_CRASH if crashed else status,
        description=description.strip(),
        error=error,
        ts=0.0,
    )


def _to_float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def read_results(root: Path) -> ResultsSummary:
    """The whole ledger, with the running best marked.

    Read from the file rather than from run memory so the history survives a
    restart and is shared with analysis.ipynb, which reads the same file.
    """
    path = results_path(root)
    summary = ResultsSummary(exists=path.exists())
    if not summary.exists:
        return summary

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return summary

    checkpoints = _read_checkpoint_index(root)
    index = 0
    best: Optional[float] = None
    best_row: Optional[ExperimentResult] = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # The header, and any header the agent rewrote with different casing or
        # spacing. Checked before the index so it never consumes one.
        if _looks_like_header(stripped):
            continue
        index += 1
        row = _parse_row(stripped, index)
        if row is None:
            index -= 1
            continue
        if row.commit:
            row.checkpoint = checkpoints.get(row.commit)
        if row.val_bpb is not None and (best is None or row.val_bpb < best):
            # The old frontier is demoted rather than left marked, so the flag
            # always means "best so far" and never "was best at some point".
            # Two rows reading as the best is how a chart grows two frontiers.
            if best_row is not None:
                best_row.is_best = False
            best = row.val_bpb
            best_row = row
            row.is_best = True
        summary.rows.append(row)
    return summary


def _looks_like_header(line: str) -> bool:
    """Whether a line is a header rather than data.

    Compared with punctuation removed and case folded, because the file is
    written by an LLM: a header rewritten as ``Val BPB`` rather than
    ``val_bpb`` is still a header, and reading it as data would add a bogus
    first row that renumbers every real experiment.
    """
    cells = {
        re.sub(r"[^a-z0-9]", "", cell.lower())
        for cell in line.split("\t")
    }
    return "valbpb" in cells and "status" in cells


# -----------------------------------------------------------------------------
# Writing
# -----------------------------------------------------------------------------

def ensure_results_file(root: Path) -> None:
    """Create the ledger with the project's header if it is not there.

    The agent is told to append to this file and program.md assumes it exists.
    Writing the header here means the very first experiment has somewhere to go.
    """
    path = results_path(root)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(RESULTS_HEADER + "\n")


def append_result(root: Path, *, commit: str, val_bpb: Optional[float],
                  memory_gb: Optional[float], status: str, description: str) -> None:
    """Append one row, in the column order analysis.ipynb expects.

    The agent normally writes this row itself. It is here for the case where the
    agent finished its work but did not get to the ledger, and for the studio's
    own crash rows, which have to be recorded whatever the agent managed to do.
    """
    ensure_results_file(root)
    status = status if status in (STATUS_KEEP, STATUS_DISCARD, STATUS_CRASH) else STATUS_CRASH
    bpb = f"{val_bpb:.6f}" if val_bpb is not None else "0.000000"
    memory = f"{memory_gb:.4f}" if memory_gb is not None else "0.0"
    # A tab in a description would move every later column, so it is collapsed.
    clean = " ".join((description or "").replace("\t", " ").split())
    row = "\t".join([commit.strip(), bpb, memory, status, clean])
    with results_path(root).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(row + "\n")


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------

@dataclass
class ExperimentCheckpoint:
    """One experiment's weights, kept because they scored."""

    name: str
    path: str
    experiment: int
    commit: str
    val_bpb: Optional[float]
    bytes: int
    modified: float
    num_params_m: Optional[float] = None
    depth: Optional[int] = None

    @property
    def model_id(self) -> str:
        return f"autoresearch/{self.experiment:03d}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "experiment": self.experiment,
            "commit": self.commit,
            "val_bpb": self.val_bpb,
            "bytes": self.bytes,
            "modified": self.modified,
            "num_params_m": self.num_params_m,
            "depth": self.depth,
            "model_id": self.model_id,
        }


_SIDECAR_SUFFIX = ".json"
_checkpoint_lock = threading.RLock()


def _sidecar_path(root: Path, name: str) -> Path:
    return checkpoint_dir(root) / f"{name}{_SIDECAR_SUFFIX}"


def capture_checkpoint(root: Path, *, experiment: int, commit: str,
                       val_bpb: Optional[float],
                       num_params_m: Optional[float] = None,
                       depth: Optional[int] = None) -> Optional[ExperimentCheckpoint]:
    """Copy this experiment's weights out before the next run overwrites them.

    train.py writes one fixed filename and overwrites it, so the only moment the
    weights of experiment N exist is right after N finished. Skipped when the
    file is missing, which happens when the run crashed before its save.
    """
    source = root / TRAIN_CHECKPOINT_NAME
    if not source.exists():
        return None
    name = f"exp_{experiment:03d}"
    target = checkpoint_dir(root) / f"{name}.pt"
    with _checkpoint_lock:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError:
            return None
        meta = {
            "experiment": experiment,
            "commit": commit,
            "val_bpb": val_bpb,
            "num_params_m": num_params_m,
            "depth": depth,
            "captured_at": time.time(),
        }
        try:
            with _sidecar_path(root, name).open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(meta, handle, indent=2)
        except OSError:
            # The weights are the valuable part; metadata that failed to write
            # costs the UI a label, not a model.
            pass
        try:
            stat = target.stat()
            size, modified = stat.st_size, stat.st_mtime
        except OSError:
            return None
    return ExperimentCheckpoint(
        name=name, path=str(target), experiment=experiment, commit=commit,
        val_bpb=val_bpb, bytes=size, modified=modified,
        num_params_m=num_params_m, depth=depth,
    )


def _read_checkpoint_index(root: Path) -> dict[str, str]:
    """Commit -> checkpoint name, so a ledger row can name its own weights."""
    out: dict[str, str] = {}
    directory = checkpoint_dir(root)
    if not directory.is_dir():
        return out
    for sidecar in directory.glob(f"*{_SIDECAR_SUFFIX}"):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        commit = str(meta.get("commit") or "").strip()
        name = sidecar.name[: -len(_SIDECAR_SUFFIX)]
        if commit:
            out[commit] = name
    return out


def list_checkpoints(root: Path) -> list[ExperimentCheckpoint]:
    """Every kept experiment, best score first.

    A weight file without its sidecar is still listed: the run wrote it, and
    refusing to show it would hide the one model the user may want to chat with.
    """
    directory = checkpoint_dir(root)
    if not directory.is_dir():
        return []
    out: list[ExperimentCheckpoint] = []
    for weights in directory.glob("*.pt"):
        name = weights.stem
        meta: dict = {}
        sidecar = _sidecar_path(root, name)
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        experiment = _int_or(meta.get("experiment"), _experiment_from_name(name))
        try:
            stat = weights.stat()
            size, modified = stat.st_size, stat.st_mtime
        except OSError:
            size, modified = 0, 0.0
        out.append(ExperimentCheckpoint(
            name=name,
            path=str(weights),
            experiment=experiment,
            commit=str(meta.get("commit") or ""),
            val_bpb=_to_float(str(meta["val_bpb"])) if meta.get("val_bpb") is not None else None,
            bytes=size,
            modified=modified,
            num_params_m=_to_float(str(meta["num_params_m"])) if meta.get("num_params_m") is not None else None,
            depth=_int_or(meta.get("depth"), None),
        ))
    out.sort(key=lambda c: (c.val_bpb is None, c.val_bpb if c.val_bpb is not None else 0.0, -c.experiment))
    return out


_EXPERIMENT_RE = re.compile(r"(\d+)")


def _experiment_from_name(name: str) -> int:
    match = _EXPERIMENT_RE.search(name)
    return int(match.group(1)) if match else 0


def _int_or(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
