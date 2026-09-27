# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chatting with a nanochat checkpoint from the normal Chat tab.

A nanochat checkpoint is a raw ``state_dict`` plus a tokenizer, not a HuggingFace
model directory, and it can only be loaded by nanochat's own code in nanochat's own
virtualenv. So generation is a subprocess: one long-lived worker that loads a
checkpoint once and then answers prompts, which keeps token streaming cheap
without paying the load cost per message.

Why a persistent worker rather than one process per message: loading a checkpoint
takes seconds, and a chat UI that stalls for several seconds before each reply
feels broken.

Concurrency is one request at a time. Two of these sidecars on one GPU would need
two copies of the weights, and on the consumer cards this feature targets that is
the difference between working and OOM. A second request waits rather than
spawning a competing worker.
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
from typing import Any, Callable, Generator, Iterable, Optional

from core.nanochat import environment, presets
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

# Checkpoint source -> how nanochat names the directory it writes into.
SOURCE_DIRS = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}


class NanochatChatError(RuntimeError):
    """Loading or driving a nanochat checkpoint failed."""


# -----------------------------------------------------------------------------
# Finding checkpoints
# -----------------------------------------------------------------------------

def nanochat_base_dir() -> Path:
    """Where nanochat keeps its checkpoints.

    Matches the orchestrator: inside the account workspace, not the user's real
    ``~/.cache/nanochat``, so a run started from either path finds the same
    checkpoints and accounts stay isolated.
    """
    from utils.paths.storage_roots import account_path

    return account_path("outputs/nanochat/base")


def _step_of(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


@dataclass
class CheckpointInfo:
    source: str
    model_tag: str
    step: int
    path: str
    bytes: int
    modified: float
    num_params: Optional[int] = None
    val_bpb: Optional[float] = None
    max_seq_len: Optional[int] = None
    # Read from the checkpoint metadata when it is there. Without a tokenizer on
    # disk the checkpoint cannot be loaded at all.
    loadable: bool = False
    reason: Optional[str] = None

    def model_id(self) -> str:
        """Identifier the chat model picker shows.

        Qualified by source and step because ``d12`` exists three times over (base,
        sft, rl) and picking the wrong one would silently chat with an
        un-fine-tuned base model.
        """
        return f"nanochat/{self.source}/{self.model_tag}/{self.step}"


def list_checkpoints() -> list[CheckpointInfo]:
    """Every nanochat checkpoint on disk, newest first within each source.

    Read from the filesystem rather than from a run record, so checkpoints
    survive a run being cleared and are still offered after a restart.
    """
    base = nanochat_base_dir()
    tokenizer_present = (base / "tokenizer" / "tokenizer.pkl").exists()
    out: list[CheckpointInfo] = []

    for source, folder in SOURCE_DIRS.items():
        root = base / folder
        if not root.is_dir():
            continue
        for tag_dir in sorted(root.iterdir()):
            if not tag_dir.is_dir():
                continue
            for model_file in sorted(tag_dir.glob("model_*.pt"), key=_step_of):
                step = _step_of(model_file)
                if step < 0:
                    continue
                meta_path = tag_dir / f"meta_{step:06d}.json"
                num_params = None
                val_bpb = None
                max_seq_len = None
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        model_config = meta.get("model_config") or {}
                        num_params = model_config and None
                        num_params = meta.get("num_params")
                        val_bpb = meta.get("val_bpb")
                        max_seq_len = model_config.get("sequence_len")
                    except Exception:
                        logger.debug("could not read %s", meta_path, exc_info=True)
                try:
                    stat = model_file.stat()
                    size, modified = stat.st_size, stat.st_mtime
                except OSError:
                    size, modified = 0, 0.0

                loadable = tokenizer_present
                reason = None if tokenizer_present else (
                    "No tokenizer on disk. Run the tokenizer stage, or retrain a "
                    "tokenizer, before this checkpoint can be loaded."
                )
                out.append(CheckpointInfo(
                    source=source,
                    model_tag=tag_dir.name,
                    step=step,
                    path=str(model_file),
                    bytes=size,
                    modified=modified,
                    num_params=num_params,
                    val_bpb=val_bpb,
                    max_seq_len=max_seq_len,
                    loadable=loadable,
                    reason=reason,
                ))

    # Newest first, so the checkpoint a just-finished run produced is at the top.
    out.sort(key=lambda c: (c.source, c.model_tag, c.step), reverse=True)
    return out


def resolve_model_id(model_id: str) -> CheckpointInfo:
    """Parse ``nanochat/<source>/<tag>/<step>`` back to a checkpoint on disk.

    The id is what the chat model picker stores, so it is re-resolved against the
    filesystem on use rather than trusted: a checkpoint can be deleted between
    being listed and being picked.
    """
    parts = model_id.split("/")
    if len(parts) != 4 or parts[0] != "nanochat":
        raise NanochatChatError(f"not a nanochat model id: {model_id!r}")
    _prefix, source, tag, step_text = parts
    if source not in SOURCE_DIRS:
        raise NanochatChatError(f"unknown checkpoint source: {source!r}")
    try:
        step = int(step_text)
    except ValueError as exc:
        raise NanochatChatError(f"not a step number: {step_text!r}") from exc

    path = nanochat_base_dir() / SOURCE_DIRS[source] / tag / f"model_{step:06d}.pt"
    if not path.exists():
        raise NanochatChatError(f"checkpoint is gone: {path}")
    info = next(
        (c for c in list_checkpoints() if c.source == source and c.model_tag == tag and c.step == step),
        None,
    )
    if info is None:
        raise NanochatChatError(f"checkpoint metadata is unreadable: {path}")
    if not info.loadable:
        raise NanochatChatError(info.reason or "checkpoint is not loadable")
    return info


# -----------------------------------------------------------------------------
# The worker
# -----------------------------------------------------------------------------

class _ChatWorker:
    """A long-lived nanochat process that loads one checkpoint and answers prompts.

    Communication is newline-delimited JSON on stdin/stdout. A request gets an id
    and the matching reply is picked out of the read stream, so a single pipe
    carries both directions without interleaving answers.
    """

    def __init__(self, checkpoint: CheckpointInfo) -> None:
        self.checkpoint = checkpoint
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self._request_id = 0

    def start(self) -> None:
        status = environment.environment_status()
        if not status.venv_present:
            raise NanochatChatError(status.blocking_reason or "nanochat is not installed")

        command = [
            str(environment.venv_python()), "-m", "scripts.chat_server",
            "-i", self.checkpoint.source,
            "-g", self.checkpoint.model_tag,
            "-s", str(self.checkpoint.step),
        ]
        env = environment.stage_environment(
            # The worker answers on stdout as a protocol, so no event file is
            # wanted: nanochat's print0 would interleave log lines into the reply
            # stream. chat_server routes its own logs to stderr instead.
            event_file = environment.checkout_root() / ".chat_worker_no_events.jsonl",
            base_dir = nanochat_base_dir(),
        )
        env.pop("NANOCHAT_EVENT_FILE", None)

        try:
            self.proc = subprocess.Popen(
                command,
                cwd = str(environment.checkout_root()),
                env = env,
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                bufsize = 1,
                **child_popen_kwargs(),
                **windows_hidden_subprocess_kwargs(),
            )
        except Exception as exc:
            raise NanochatChatError(f"could not start the nanochat chat worker: {exc}") from exc

        adopt_pid(self.proc.pid)

        # Wait for the worker's ready handshake rather than sleeping: loading a
        # checkpoint takes seconds to minutes depending on depth, and a fixed wait
        # is either slow or flaky. chat_server emits {"ready": true} once the
        # model is resident.
        deadline = time.time() + LOAD_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self.proc.poll() is not None:
                stderr = ""
                if self.proc.stderr is not None:
                    try:
                        stderr = self.proc.stderr.read() or ""
                    except Exception:
                        pass
                raise NanochatChatError(
                    f"nanochat exited while loading (code {self.proc.returncode}): "
                    f"{stderr.strip()[-600:] or 'no output'}"
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
                # nanochat's own output, which chat_server sends to stderr. It
                # should not appear here; ignore it rather than fail.
                continue
            if message.get("ready"):
                return
            if message.get("error"):
                raise NanochatChatError(str(message["error"]))

        raise NanochatChatError(
            f"loading the checkpoint took longer than {LOAD_TIMEOUT_SECONDS}s"
        )

    def _readline_with_timeout(self, timeout: float) -> Optional[str]:
        """One line from stdout, or None if nothing arrived in time.

        A blocking readline cannot be interrupted, which would make the readiness
        wait unable to notice a process that died. A reader thread per call is too
        expensive, so a short poll on the buffered reader is used instead: the
        worker only writes the handshake once, so blocking there is acceptable
        and the death check runs on the way in.
        """
        import select

        if os.name == "nt":
            # Windows has no select() on pipes. The process-exit check above
            # covers the common failure, so a plain read is acceptable here.
            try:
                return self.proc.stdout.readline()  # type: ignore[union-attr]
            except Exception:
                return None
        try:
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout)  # type: ignore[arg-type]
        except Exception:
            return None
        if not ready:
            return None
        try:
            return self.proc.stdout.readline()  # type: ignore[union-attr]
        except Exception:
            return None

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 50,
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Yield response text as it is produced.

        Yields cumulative text, matching the convention every other backend in
        this codebase uses, so callers diff it themselves.
        """
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise NanochatChatError("the nanochat chat worker is not running")
        if self.proc.poll() is not None:
            raise NanochatChatError("the nanochat chat worker has exited")

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
                self.proc.stdin.write(json.dumps(request) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise NanochatChatError("the nanochat chat worker closed its input") from exc

            text = ""
            deadline = time.time() + GENERATE_TIMEOUT_SECONDS
            while time.time() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    self._send({"id": request_id, "abort": True})
                    return
                line = self.proc.stdout.readline()
                if not line:
                    if self.proc.poll() is not None:
                        raise NanochatChatError("the nanochat chat worker exited mid-generation")
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
                    raise NanochatChatError(str(message["error"]))
                if message.get("done"):
                    return
                delta = message.get("text")
                if delta:
                    text += str(delta)
                    yield text

    def _send(self, payload: dict) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
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
    the failure mode this feature is most likely to hit.
    """
    global _worker
    with _worker_lock:
        if _worker is not None:
            if _worker.checkpoint.model_id() == checkpoint.model_id():
                return _worker
            _worker.close()
            _worker = None
        if is_process_shutting_down():
            raise NanochatChatError("LABZ is shutting down")
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
            "backend": "nanochat",
            "model": _worker.checkpoint.model_id(),
            "source": _worker.checkpoint.source,
            "step": _worker.checkpoint.step,
        }


def free_for_training(reason: str) -> bool:
    """Release the checkpoint so a training run has the whole GPU."""
    if resident() is None:
        return False
    logger.info("unloading the nanochat chat checkpoint to free VRAM (%s)", reason)
    return unload()


# -----------------------------------------------------------------------------
# Generation entry point
# -----------------------------------------------------------------------------

def generate_chat(
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_k: int = 50,
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
    depth = 0
    for part in info.model_tag.split("d"):
        if part.isdigit():
            depth = int(part)
    counts = presets.param_breakdown(depth) if depth else None
    return {
        "model_id": info.model_id(),
        "backend": "nanochat",
        "source": info.source,
        "step": info.step,
        "bytes": info.bytes,
        "val_bpb": info.val_bpb,
        "num_params": info.num_params or (counts["total"] if counts else None),
        "n_embd": counts["n_embd"] if counts else None,
        "n_head": counts["n_head"] if counts else None,
        # A raw state_dict, so it is not a GGUF and not a HuggingFace repo.
        "format": "nanochat-checkpoint",
    }
