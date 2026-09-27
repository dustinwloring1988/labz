# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chatting with a model an autoresearch experiment produced.

The same arrangement as nanochat's chat sidecar, and for the same reasons. An
autoresearch checkpoint is a raw ``state_dict`` plus a tokenizer that only
autoresearch's own code in autoresearch's own virtualenv can read, so generation
is a subprocess: one long-lived worker that loads a checkpoint once and then
answers prompts. Loading takes seconds, and a chat that stalls before every reply
feels broken, so the worker outlives the request.

Concurrency is one request at a time. Two of these on one GPU would need two
copies of the weights, and the whole point of this feature is that the GPU is
also where the experiments run.

Where it differs from nanochat, and why:

  - **A checkpoint per experiment, named by index.** autoresearch overwrites one
    fixed file every run, so there is no "step" to address a checkpoint by. The
    index is the identity, and it is also what the user recognises from the
    results table.

  - **A base model, not a chat model.** There is no chat stage in the project, so
    there is no chat template and no system prompt. What it does is continue
    text, and the UI is told so, because a model asked a question and shown a
    rambling continuation has not malfunctioned.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

from core.autoresearch import environment, results
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    is_process_shutting_down,
    terminate_pid,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# A checkpoint that takes longer than this to load is treated as failed rather
# than left hanging, so the UI gets an error instead of an endless spinner.
LOAD_TIMEOUT_SECONDS = 300
# A generation with no token for this long is considered dead.
GENERATE_TIMEOUT_SECONDS = 600


class AutoresearchChatError(RuntimeError):
    """Loading or driving an autoresearch checkpoint failed."""


# -----------------------------------------------------------------------------
# Finding checkpoints
# -----------------------------------------------------------------------------

def autoresearch_root() -> Path:
    """Where the copied project lives."""
    return environment.checkout_root()


@dataclass
class CheckpointInfo:
    """One experiment's weights, as the chat model picker sees it."""

    name: str
    path: str
    experiment: int
    commit: str
    val_bpb: Optional[float]
    bytes: int
    modified: float
    num_params_m: Optional[float] = None
    depth: Optional[int] = None
    # Read from what is on disk, not from the run that produced it. Without a
    # tokenizer there is no vocabulary to decode with, so the weights are inert.
    loadable: bool = True
    reason: Optional[str] = None

    @property
    def model_id(self) -> str:
        # Qualified by experiment because the project saves every experiment to
        # the same filename, and picking the wrong one would silently chat with a
        # model that lost three experiments ago.
        return f"autoresearch/{self.experiment:03d}"

    def display_name(self) -> str:
        score = f"val bpb {self.val_bpb:.4f}" if self.val_bpb is not None else "no score"
        return f"Experiment {self.experiment} · {score}"


def list_checkpoints() -> list[CheckpointInfo]:
    """Every experiment whose weights survived, best score first.

    Read from the filesystem rather than from run memory, so checkpoints are
    still offered after a restart and a run being cleared does not lose them.

    A checkpoint needs the tokenizer to be loadable at all, and it needs the
    ``train.py`` that produced it, which is why a shape change upstream makes an
    old one unloadable. The first is checked here so the UI can say so before the
    user clicks; the second cannot be known without trying.
    """
    tokenizer_present = environment.is_data_prepared()
    out: list[CheckpointInfo] = []
    for entry in results.list_checkpoints(autoresearch_root()):
        reason = None
        loadable = True
        if not tokenizer_present:
            loadable = False
            reason = (
                "No tokenizer on disk. Run data preparation, or the checkpoint "
                "cannot be loaded: it is a raw state_dict with no vocabulary of "
                "its own."
            )
        elif not Path(entry.path).exists():
            loadable = False
            reason = "The weight file is gone."
        out.append(CheckpointInfo(
            name=entry.name, path=entry.path, experiment=entry.experiment,
            commit=entry.commit, val_bpb=entry.val_bpb, bytes=entry.bytes,
            modified=entry.modified, num_params_m=entry.num_params_m,
            depth=entry.depth, loadable=loadable, reason=reason,
        ))
    return out


def _describe(info: CheckpointInfo) -> dict:
    """The picker's view of a checkpoint, including why it may be unusable."""
    return {
        "model_id": info.model_id,
        "name": info.display_name(),
        "backend": "autoresearch",
        "experiment": info.experiment,
        "commit": info.commit,
        "bytes": info.bytes,
        "modified": info.modified,
        "val_bpb": info.val_bpb,
        "num_params_m": info.num_params_m,
        "depth": info.depth,
        "loadable": info.loadable,
        "reason": info.reason,
        # There is no chat stage in autoresearch, so this model continues text.
        # Stated up front so the UI can label it rather than the user
        # discovering it by getting a paragraph of TinyStories back.
        "kind": "base",
        "hint": "Continues text. autoresearch has no chat stage, so this model "
                "predicts what comes next rather than replying.",
    }


def resolve_model_id(model_id: str) -> CheckpointInfo:
    """Parse ``autoresearch/<experiment>`` back to a checkpoint on disk.

    The id is what the picker stores, so it is re-resolved against the
    filesystem on use: a checkpoint can be deleted between being listed and
    being picked.
    """
    parts = str(model_id).split("/")
    if len(parts) != 2 or parts[0] != "autoresearch":
        raise AutoresearchChatError(f"not an autoresearch model id: {model_id!r}")
    try:
        experiment = int(parts[1])
    except ValueError as exc:
        raise AutoresearchChatError(f"not an experiment number: {parts[1]!r}") from exc

    for info in list_checkpoints():
        if info.experiment == experiment:
            if not info.loadable:
                raise AutoresearchChatError(info.reason or "checkpoint is not loadable")
            return info
    raise AutoresearchChatError(f"no checkpoint was kept for experiment {experiment}")


# -----------------------------------------------------------------------------
# The worker
# -----------------------------------------------------------------------------

class _ChatWorker:
    """A long-lived autoresearch process that loads one checkpoint and answers prompts.

    Communication is newline-delimited JSON on stdin/stdout. A request gets an id
    and the matching reply is picked out of the read stream, so a single pipe
    carries both directions without interleaving answers.
    """

    def __init__(self, checkpoint: CheckpointInfo) -> None:
        self.checkpoint = checkpoint
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._request_id = 0
        self.params: Optional[int] = None
        # Bounded stderr tail, so a failure can be explained with the worker's
        # own words rather than "it exited".
        self._stderr_lock = threading.Lock()
        self._stderr_lines: list[str] = []

    def start(self) -> None:
        root = autoresearch_root()
        if not environment.venv_python().exists():
            raise AutoresearchChatError(
                environment.environment_status().blocking_reason
                or "autoresearch is not installed"
            )
        from core.autoresearch.chat_server import write_chat_server

        # The worker is generated, so it is written on load rather than at
        # install: a checkout created by an older studio, or one refreshed since,
        # would otherwise be missing it.
        write_chat_server(root)

        command = [
            str(environment.venv_python()),
            str(root / "chat_server.py"),
            "-w", self.checkpoint.path,
        ]
        env = environment.experiment_environment()
        # The worker talks a protocol on stdout, so nothing may print there.
        # train.py is imported for its model class and does not print at import,
        # but the tokenizer and torch both can on some paths, and a stray line
        # would be read as a protocol frame.
        env["PYTHONWARNINGS"] = "ignore"

        try:
            self.proc = subprocess.Popen(
                command,
                cwd=str(root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **child_popen_kwargs(),
                **windows_hidden_subprocess_kwargs(),
            )
        except Exception as exc:
            raise AutoresearchChatError(
                f"could not start the autoresearch chat worker: {exc}"
            ) from exc

        adopt_pid(self.proc.pid)

        # A drain thread for stderr. Without one the worker's own log lines fill
        # the pipe buffer and the process blocks forever partway through a load,
        # which presents as a load that never finishes.
        threading.Thread(
            target=self._drain_stderr, name="autoresearch-chat-stderr", daemon=True
        ).start()

        # Wait for the ready handshake rather than sleeping: loading a
        # checkpoint takes seconds to tens of seconds depending on depth, and a
        # fixed wait is either slow or flaky.
        deadline = time.time() + LOAD_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AutoresearchChatError(
                    f"autoresearch exited while loading the checkpoint "
                    f"(code {self.proc.returncode}): {self._stderr_tail()}"
                )
            line = self._readline_with_timeout(1.0)
            if line is None:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                # A log line, not protocol. The worker routes its own logs to
                # stderr, so this should not happen; ignore rather than fail.
                continue
            if message.get("ready"):
                self.params = message.get("params")
                return
            if message.get("error"):
                raise AutoresearchChatError(str(message["error"]))

        raise AutoresearchChatError(
            f"loading the checkpoint took longer than {LOAD_TIMEOUT_SECONDS}s"
        )

    def _stderr_tail(self, limit: int = 800) -> str:
        """Whatever the worker said on its way out.

        Read from the buffer the drain thread fills, not from the pipe: the pipe
        is already being consumed and a second read would block.
        """
        with self._stderr_lock:
            text = "".join(self._stderr_lines[-40:])
        return text.strip()[-limit:] or "no output on stderr"

    def _drain_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                with self._stderr_lock:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 200:
                        del self._stderr_lines[:-200]
        except Exception:
            pass

    def _readline_with_timeout(self, timeout: float) -> Optional[str]:
        """One line from stdout, or None if nothing arrived in time.

        A blocking readline cannot be interrupted, which would make the readiness
        wait unable to notice a process that died. A reader thread per call is
        too expensive, so a short poll on the buffered reader is used instead.
        """
        import select

        proc = self.proc
        if proc is None or proc.stdout is None:
            return None
        if proc.poll() is not None:
            return None
        if os.name == "nt":
            # Windows has no select() on pipes. The process-exit check covers
            # the common failure, so a plain read is acceptable here: the worker
            # writes its handshake as soon as it is ready, or not at all.
            try:
                return proc.stdout.readline()
            except Exception:
                return None
        try:
            ready, _, _ = select.select([proc.stdout], [], [], timeout)  # type: ignore[arg-type]
        except Exception:
            return None
        if not ready:
            return None
        try:
            return proc.stdout.readline()
        except Exception:
            return None

    def generate(self, prompt: str, *, max_new_tokens: int = 256,
                 temperature: float = 0.8, top_k: int = 50,
                 cancel_event: Optional[threading.Event] = None,
                 ) -> Generator[str, None, None]:
        """Yield response text as it is produced.

        Yields cumulative text, matching the convention every other backend in
        this codebase uses, so callers diff it themselves.
        """
        proc = self.proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise AutoresearchChatError("the autoresearch chat worker is not running")
        if proc.poll() is not None:
            raise AutoresearchChatError("the autoresearch chat worker has exited")

        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            request = {
                "id": request_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
            }
            try:
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise AutoresearchChatError(
                    "the autoresearch chat worker closed its input"
                ) from exc

            text = ""
            deadline = time.time() + GENERATE_TIMEOUT_SECONDS
            while time.time() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    self._send({"id": request_id, "abort": True})
                    return
                try:
                    line = proc.stdout.readline()
                except Exception as exc:
                    raise AutoresearchChatError(
                        f"the autoresearch chat worker stopped responding: {exc}"
                    ) from exc
                if not line:
                    if proc.poll() is not None:
                        raise AutoresearchChatError(
                            f"the autoresearch chat worker exited mid-generation "
                            f"({self._stderr_tail(300)})"
                        )
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    # A log line, not protocol.
                    continue
                if message.get("id") != request_id:
                    continue
                if message.get("error"):
                    raise AutoresearchChatError(str(message["error"]))
                if message.get("done"):
                    return
                delta = message.get("text")
                if delta:
                    text += str(delta)
                    yield text

    def _send(self, payload: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except Exception:
            pass

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                terminate_pid(proc.pid, timeout=5.0, owner_verified=True)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


_worker: Optional[_ChatWorker] = None
_worker_lock = threading.RLock()


def get_worker() -> Optional[_ChatWorker]:
    with _worker_lock:
        return _worker


def load(checkpoint: CheckpointInfo) -> _ChatWorker:
    """Load a checkpoint for chatting, replacing whatever was loaded.

    The previous worker is closed first: two copies of the weights on one GPU is
    the failure mode this feature is most likely to hit, and the experiments
    need that GPU too.
    """
    global _worker
    with _worker_lock:
        if _worker is not None:
            if _worker.checkpoint.model_id == checkpoint.model_id:
                return _worker
            _worker.close()
            _worker = None
        if is_process_shutting_down():
            raise AutoresearchChatError("Unsloth is shutting down")
        worker = _ChatWorker(checkpoint)
        worker.start()
        _worker = worker
        return worker


def unload() -> bool:
    global _worker
    with _worker_lock:
        if _worker is None:
            return False
        _worker.close()
        _worker = None
        return True


def resident() -> Optional[dict]:
    """What is loaded, for the VRAM arbiter and the UI."""
    with _worker_lock:
        if _worker is None or _worker.proc is None:
            return None
        if _worker.proc.poll() is not None:
            return None
        return {
            "backend": "autoresearch",
            "model": _worker.checkpoint.model_id,
            "experiment": _worker.checkpoint.experiment,
            "val_bpb": _worker.checkpoint.val_bpb,
        }


def free_for_training(reason: str) -> bool:
    """Release the checkpoint so an experiment has the whole GPU."""
    if resident() is None:
        return False
    logger.info("unloading the autoresearch chat checkpoint to free VRAM (%s)", reason)
    return unload()


# -----------------------------------------------------------------------------
# Generation entry point
# -----------------------------------------------------------------------------

def generate_chat(model_id: str, prompt: str, *, max_new_tokens: int = 256,
                  temperature: float = 0.8, top_k: int = 50,
                  cancel_event: Optional[threading.Event] = None,
                  ) -> Generator[str, None, None]:
    """Stream a reply for one prompt. The shape the chat routes expect."""
    checkpoint = resolve_model_id(model_id)
    worker = load(checkpoint)
    yield from worker.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        cancel_event=cancel_event,
    )


def model_estimate(model_id: str) -> dict:
    """Size and shape of a checkpoint, for the model picker's metadata."""
    info = resolve_model_id(model_id)
    return {
        "model_id": info.model_id,
        "backend": "autoresearch",
        "experiment": info.experiment,
        "bytes": info.bytes,
        "val_bpb": info.val_bpb,
        "num_params_m": info.num_params_m,
        "depth": info.depth,
        # A raw state_dict, so it is not a GGUF and not a HuggingFace repo.
        "format": "autoresearch-checkpoint",
    }
