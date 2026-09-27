# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP surface for the nanochat tab.

The live view is driven by an SSE stream on ``/progress``, mirroring the shape
the LABZ train tab already uses so the frontend can reuse its stream
plumbing: ``Last-Event-ID`` resume, a heartbeat while a stage is preparing, and
an ``event_id`` that is a monotonically increasing integer.

The stream polls the run state rather than being pushed to, because the state is
owned by a reader thread in a different process boundary (the nanochat child
writes a file, not a queue).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from auth.authentication import get_current_subject
from core.nanochat import catalogue, environment, presets
from core.nanochat.orchestrator import (
    NanochatBusy,
    NanochatNotInstalled,
    get_run_manager,
)
from models.nanochat import (
    NanochatBenchmarksResponse,
    NanochatCatalogueResponse,
    NanochatEnvironmentResponse,
    NanochatFitResponse,
    NanochatLogResponse,
    NanochatMetricsResponse,
    NanochatPresetsResponse,
    NanochatSamplesResponse,
    NanochatStartRequest,
    NanochatStartResponse,
    NanochatStatusResponse,
    NanochatStopRequest,
    NanochatStopResponse,
    NanochatValidateRequest,
    NanochatValidateResponse,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

router = APIRouter()

# How often the progress stream samples the run state.
_STREAM_POLL_SECONDS = 1.0
# Frames sent while a stage is running but has not yet emitted a metric (model
# build, data load, torch.compile). Those can take minutes with nothing to say.
_PREP_HEARTBEAT_EVERY_POLLS = 5
# A stage that produces no step change for this long is reported as an error.
# Generous because a stage can legitimately go quiet: compiling, or the first
# optimizer step on a large model is slow.
_STALL_TIMEOUT_POLLS = 1800
# Console lines a single progress frame may carry. Caps the frame size when a
# stage is printing hard; the remainder rides along on the next tick.
_MAX_LOG_LINES_PER_FRAME = 200


def _manager():
    return get_run_manager()


def _require_run(expected: Optional[str] = None):
    """The active run, or a 404.

    ``expected`` guards against a stale UI acting on a run that has already been
    replaced: a poll that arrives after a new run started must not report the new
    run's state to the old tab.
    """
    run = _manager().current()
    if run is None:
        raise HTTPException(status_code=404, detail="No nanochat run")
    if expected and run.run_id != expected:
        raise HTTPException(status_code=409, detail="nanochat run was superseded")
    return run


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

@router.get("/environment", response_model=NanochatEnvironmentResponse)
async def get_environment(current_subject: str = Depends(get_current_subject)) -> NanochatEnvironmentResponse:
    """Whether nanochat is usable yet, and what is missing if not."""
    status = environment.environment_status()
    return NanochatEnvironmentResponse(**vars(status), ready=status.ready())


@router.post("/environment/install", response_model=NanochatEnvironmentResponse)
async def install_environment(current_subject: str = Depends(get_current_subject)) -> NanochatEnvironmentResponse:
    """Start the clone and dependency install in the background.

    Returns immediately: this pulls a multi-gigabyte torch wheel, and a request
    that waited for it would time out. The UI polls /environment for progress.
    """
    environment.install_async()
    status = environment.environment_status()
    return NanochatEnvironmentResponse(**vars(status), ready=status.ready())


@router.post("/environment/check", response_model=dict)
async def check_registry(current_subject: str = Depends(get_current_subject)) -> dict:
    """Ask nanochat to validate its own tables."""
    status = environment.environment_status()
    if not status.venv_present:
        raise HTTPException(status_code=409, detail=status.blocking_reason or "not installed")
    import subprocess

    proc = subprocess.run(
        [str(environment.venv_python()), "-m", "nanochat.registry", "--check"],
        cwd=str(environment.checkout_root()),
        capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
        **windows_hidden_subprocess_kwargs(),
    )
    return {
        "ok": proc.returncode == 0,
        "output": (proc.stdout or "") + (proc.stderr or ""),
    }


# -----------------------------------------------------------------------------
# Catalogue, presets, fit
# -----------------------------------------------------------------------------

@router.get("/catalogue", response_model=NanochatCatalogueResponse)
async def get_catalogue(force: bool = Query(default=False),
    current_subject: str = Depends(get_current_subject),
) -> NanochatCatalogueResponse:
    """Datasets, SFT tasks, benchmarks and base evals, straight from nanochat."""
    payload = catalogue.get_catalogue(force=force)
    return NanochatCatalogueResponse(**payload)


@router.get("/presets", response_model=NanochatPresetsResponse)
async def get_presets(current_subject: str = Depends(get_current_subject)) -> NanochatPresetsResponse:
    """Depth presets with their real shapes, plus the pipeline and the defaults."""
    return NanochatPresetsResponse(
        presets=presets.describe_presets(),
        default_depth=presets.DEFAULT_DEPTH,
        stages=catalogue.stages(),
        defaults=catalogue.default_selection(),
    )


def _fit_response(depth: int, **kwargs) -> NanochatFitResponse:
    estimate = presets.estimate_fit(depth, **kwargs)
    data = vars(estimate).copy()
    counts = presets.param_breakdown(depth)
    data["transformer_params"] = counts["transformer_matrices"]
    return NanochatFitResponse(**data)


@router.get("/presets/{depth}/fit", response_model=NanochatFitResponse)
async def get_preset_fit(
    depth: int,
    num_iterations: int = Query(default=1000, ge=1),
    total_batch_size: int = Query(default=32768, ge=1),
    max_seq_len: int = Query(default=2048, ge=128),
    current_subject: str = Depends(get_current_subject),
) -> NanochatFitResponse:
    """What a depth costs on this machine, without starting a run."""
    if depth < 1 or depth > 64:
        raise HTTPException(status_code=400, detail="depth must be between 1 and 64")
    return _fit_response(
        depth,
        num_iterations=num_iterations,
        total_batch_size=total_batch_size,
        max_seq_len=max_seq_len,
    )


@router.post("/validate", response_model=NanochatValidateResponse)
async def validate_config(
    request: NanochatValidateRequest,
    current_subject: str = Depends(get_current_subject),
) -> NanochatValidateResponse:
    """Check a configuration and price it, without starting anything.

    A configuration that cannot be built at all still has to be describable, or
    this endpoint could not report *why*. So a request that fails validation is
    reported rather than rejected: ``config_errors`` carries the reason.
    """
    config = request.config
    errors: list[str] = []
    warnings: list[str] = []

    try:
        resolved = config.to_config()
    except (ValidationError, TypeError, ValueError) as exc:
        return NanochatValidateResponse(ok=False, errors=[], config_errors=[str(exc)])

    fit = _fit_response(
        config.depth,
        num_iterations=config.num_iterations,
        total_batch_size=config.total_batch_size,
        max_seq_len=config.max_seq_len,
    )
    errors.extend(fit.reasons)
    warnings.extend(fit.suggestions)

    # Cross-check the selection against nanochat's own tables, so a renamed task
    # or a typo'd benchmark is caught before a run downloads gigabytes.
    cat = catalogue.get_catalogue()
    if not cat.get("unavailable"):
        known_datasets = cat.get("datasets") or {}
        if known_datasets and config.dataset not in known_datasets:
            errors.append(
                f"Unknown dataset {config.dataset!r}. Available: "
                f"{', '.join(sorted(known_datasets))}"
            )
        known_tasks = cat.get("training_tasks") or {}
        unknown = [t for t in config.sft_tasks if known_tasks and t not in known_tasks]
        if unknown:
            errors.append(
                f"Unknown SFT task(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(known_tasks))}"
            )
        known_benchmarks = cat.get("benchmarks") or {}
        unknown = [b for b in config.chat_benchmarks if known_benchmarks and b not in known_benchmarks]
        if unknown:
            errors.append(
                f"Unknown benchmark(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(known_benchmarks))}"
            )
        known_evals = cat.get("base_evals") or {}
        unknown = [b for b in config.base_benchmarks if known_evals and b not in known_evals]
        if unknown:
            errors.append(
                f"Unknown base evaluation(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(known_evals))}"
            )

        # A benchmark also in the SFT mixture scores format learning as much as
        # capability. Worth saying, because the number looks like a capability.
        for key in config.chat_benchmarks:
            if key in (cat.get("trained_on_by_default") or []):
                warnings.append(
                    f"{key} is also part of the SFT mixture, so its score reflects "
                    f"format learning as well as capability."
                )

    if not config.sft_tasks and "sft" in config.stages:
        warnings.append("The SFT stage will run with no training tasks selected.")
    if config.fp8:
        from core.nanochat.environment import _device_peak_flops
        _name, _flops, device_type = _device_peak_flops()
        if device_type != "cuda":
            errors.append("FP8 training requires a CUDA GPU.")
        else:
            warnings.append(
                "FP8 needs a GPU with tensor cores for the scaled dtype (Hopper or "
                "newer). On older cards it is ignored and training is slower."
            )
    if "rl" in config.stages and config.rl_num_samples * config.rl_examples_per_step > 512:
        warnings.append(
            "That many rollout sequences per RL step will be slow and memory "
            "hungry on a single GPU."
        )

    return NanochatValidateResponse(
        ok=not errors,
        fit=fit,
        errors=errors,
        warnings=warnings,
    )


# -----------------------------------------------------------------------------
# Run lifecycle
# -----------------------------------------------------------------------------

@router.post("/start", response_model=NanochatStartResponse)
async def start_run(
    request: NanochatStartRequest,
    current_subject: str = Depends(get_current_subject),
) -> NanochatStartResponse:
    status = environment.environment_status()
    if not status.ready():
        raise HTTPException(
            status_code=409,
            detail=status.blocking_reason or "nanochat is not set up yet",
        )

    config = request.to_config()
    fit = _fit_response(
        config.depth,
        num_iterations=config.num_iterations,
        total_batch_size=config.total_batch_size,
        max_seq_len=config.max_seq_len,
    )

    try:
        run = _manager().start(config)
    except NanochatBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NanochatNotInstalled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    message = "nanochat run started"
    if not fit.fits:
        # Started anyway, but the estimate is passed back so the UI can say why
        # this may end in an out-of-memory error. Refusing outright would block a
        # configuration that might still work on a card with more free VRAM than
        # the probe saw.
        message = "Started, but the estimate says this may not fit: " + "; ".join(fit.reasons)

    return NanochatStartResponse(run_id=run.run_id, status=run.state.status,
                                 message=message, fit=fit)


@router.post("/stop", response_model=NanochatStopResponse)
async def stop_run(
    request: NanochatStopRequest,
    current_subject: str = Depends(get_current_subject),
) -> NanochatStopResponse:
    result = _manager().stop(save=request.save)
    return NanochatStopResponse(**result)


@router.get("/status", response_model=NanochatStatusResponse)
async def get_status(
    expected_run_id: Optional[str] = Query(default=None),
    current_subject: str = Depends(get_current_subject),
) -> NanochatStatusResponse:
    # No run yet is a state, not a failure: a tab opened before the first run
    # hydrates from this on every mount, and a 404 there would be noise. A run
    # that was replaced still 409s, because that poll is about a run the caller
    # believed in and is no longer.
    run = _manager().current()
    if run is None:
        return NanochatStatusResponse(run_id="", status="idle", phase="idle")
    if expected_run_id and run.run_id != expected_run_id:
        raise HTTPException(status_code=409, detail="nanochat run was superseded")
    state = run.state
    data = state.to_dict()
    # to_dict() carries stages as raw dataclass dicts; reshape to the response
    # model, adding the derived duration each stage needs.
    data["stages"] = [
        {
            "key": stage.key,
            "status": stage.status,
            "message": stage.message,
            "error": stage.error,
            "started_at": stage.started_at,
            "ended_at": stage.ended_at,
            "duration_seconds": stage.duration_seconds,
        }
        for stage in state.stages.values()
    ]
    return NanochatStatusResponse(**data)


@router.get("/metrics", response_model=NanochatMetricsResponse)
async def get_metrics(expected_run_id: Optional[str] = Query(default=None)) -> NanochatMetricsResponse:
    run = _require_run(expected_run_id)
    return NanochatMetricsResponse(run_id=run.run_id, series=run.metrics())


@router.get("/samples", response_model=NanochatSamplesResponse)
async def get_samples(
    limit: int = Query(default=100, ge=1, le=500),
    expected_run_id: Optional[str] = Query(default=None),
    current_subject: str = Depends(get_current_subject),
) -> NanochatSamplesResponse:
    """The sample feed: base completions, SFT chat turns, RL rollouts."""
    run = _require_run(expected_run_id)
    return NanochatSamplesResponse(run_id=run.run_id, samples=run.samples(limit=limit))


@router.get("/benchmarks", response_model=NanochatBenchmarksResponse)
async def get_benchmarks(expected_run_id: Optional[str] = Query(default=None)) -> NanochatBenchmarksResponse:
    run = _require_run(expected_run_id)
    return NanochatBenchmarksResponse(run_id=run.run_id, benchmarks=run.benchmarks())


@router.get("/log", response_model=NanochatLogResponse)
async def get_log(
    limit: int = Query(default=200, ge=1, le=2000),
    since: int = Query(default=0, ge=0),
    expected_run_id: Optional[str] = Query(default=None),
    current_subject: str = Depends(get_current_subject),
) -> NanochatLogResponse:
    """Console output for a run, as an incremental feed.

    ``since`` is the last sequence number the caller rendered. Without it this
    returns the tail, which is what a console opening mid-run wants; with it, only
    what is new.
    """
    run = _require_run(expected_run_id)
    if since > 0:
        lines, cursor = run.logs_since(since, limit=limit)
    else:
        tail = run.log_tail(limit=limit)
        lines = tail
        cursor = tail[-1]["seq"] if tail else 0
    return NanochatLogResponse(run_id=run.run_id, lines=lines, seq=cursor)


# -----------------------------------------------------------------------------
# Progress stream
# -----------------------------------------------------------------------------

def _format_sse(data: str, event: str = "progress", event_id: Optional[int] = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {data}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


@router.post("/progress")
@router.get("/progress", include_in_schema=False)
async def stream_progress(
    request: Request,
    expected_run_id: Optional[str] = Query(default=None),
    current_subject: str = Depends(get_current_subject),
):
    """Server-sent progress for the live run view.

    Polls the run state once a second and emits only when something changed, with
    a heartbeat otherwise. Reconnect resumes from the last event id the client
    saw, so a dropped connection does not leave the progress bar behind.
    """
    run = _require_run(expected_run_id)
    run_id = run.run_id

    last_event_id: Optional[int] = None
    header = request.headers.get("last-event-id")
    if header is not None:
        try:
            last_event_id = int(header)
        except ValueError:
            logger.warning("invalid Last-Event-ID: %s", header)

    async def event_generator():
        # Match the train tab's reconnect delay so the client's default backoff
        # is already the right order of magnitude.
        yield "retry: 3000\n\n"

        last_signature: Optional[str] = None
        no_update_polls = 0
        polls_since_step_change = 0
        seen_a_step = False
        counter = last_event_id or 0
        # Per-connection console cursor. The stream carries log lines alongside
        # progress rather than over a second connection: a run's output is the
        # answer to "what is it doing right now", and it is only useful alongside
        # the state that explains it. A client that reconnects starts from zero and
        # gets the tail, which is what a console opening mid-run should show.
        log_cursor = 0

        while True:
            if await request.is_disconnected():
                return
            current = _manager().current()
            if current is None or current.run_id != run_id:
                # The run was replaced or cleared underneath us. Say so rather
                # than silently going quiet.
                yield _format_sse(json.dumps({"reason": "run_ended"}), event="error", event_id=counter)
                return

            state = current.state
            active = current.is_active()
            fresh_logs, next_log_cursor = current.logs_since(log_cursor)

            payload = {
                "run_id": run_id,
                "status": state.status,
                "phase": state.phase,
                "current_stage": state.current_stage,
                "message": state.message,
                "step": state.step,
                "total_steps": state.total_steps,
                "progress_percent": state.progress_percent,
                "loss": state.loss,
                "val_bpb": state.val_bpb,
                "chatcore": state.chatcore,
                "reward": state.reward,
                "tok_per_sec": state.tok_per_sec,
                "mfu": state.mfu,
                "eta_seconds": state.eta_seconds,
                "elapsed_seconds": state.elapsed_seconds,
                "peak_memory_bytes": state.peak_memory_bytes,
                "num_params": state.num_params,
                "warnings": list(state.warnings[-5:]),
                "stages": [
                    {"key": s.key, "status": s.status, "error": s.error}
                    for s in state.stages.values()
                ],
                "logs": fresh_logs,
            }
            signature = json.dumps(payload, sort_keys=True)

            if signature != last_signature:
                counter += 1
                yield _format_sse(signature, event="progress", event_id=counter)
                last_signature = signature
                # Only advance once the lines are actually on the wire, so a
                # connection that dies mid-frame resends them instead of losing
                # them. `min` guards the case where the buffer rolled over past the
                # cursor: a line evicted before it was ever sent is not recoverable,
                # and stalling forever on a seq that will never arrive is worse.
                log_cursor = max(log_cursor, min(next_log_cursor, log_cursor + _MAX_LOG_LINES_PER_FRAME))
                no_update_polls = 0
                if state.step:
                    seen_a_step = True
                    polls_since_step_change = 0
            else:
                no_update_polls += 1
                polls_since_step_change += 1
                # A heartbeat keeps proxies from timing the connection out, and
                # is more frequent before the first step (a long model build) than
                # during training.
                every = _PREP_HEARTBEAT_EVERY_POLLS if not seen_a_step else 20
                if no_update_polls % every == 0:
                    yield _format_sse(
                        json.dumps({"run_id": run_id, "phase": state.phase,
                                    "message": state.message}),
                        event="heartbeat", event_id=counter,
                    )

            if not active:
                yield _format_sse(
                    json.dumps({"run_id": run_id, "status": state.status,
                                "message": state.message, "error": state.error}),
                    event="complete", event_id=counter,
                )
                return

            # Only count a stall once stepping has started: a stage that is
            # legitimately building a model for ten minutes is not stalled.
            if seen_a_step and polls_since_step_change > _STALL_TIMEOUT_POLLS:
                yield _format_sse(
                    json.dumps({"run_id": run_id, "error": "no progress for 30 minutes"}),
                    event="error", event_id=counter,
                )
                return

            await asyncio.sleep(_STREAM_POLL_SECONDS)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -----------------------------------------------------------------------------
# Chatting with a checkpoint from the normal Chat tab
# -----------------------------------------------------------------------------

@router.get("/chat/models")
async def list_chat_models(
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """nanochat checkpoints, as model-picker entries.

    Exposed as its own endpoint rather than folded into the generic model list
    because a nanochat checkpoint is not a HuggingFace repo or a GGUF file: it can
    only be loaded by nanochat's own loader in its own virtualenv, and offering it
    anywhere that would try to read safetensors would break.
    """
    from core.nanochat import chat_sidecar

    entries = []
    for info in chat_sidecar.list_checkpoints():
        entries.append({
            "model_id": info.model_id(),
            "name": f"{info.source.upper()} {info.model_tag} step {info.step}",
            "backend": "nanochat",
            "source": info.source,
            "model_tag": info.model_tag,
            "step": info.step,
            "bytes": info.bytes,
            "modified": info.modified,
            "num_params": info.num_params,
            "val_bpb": info.val_bpb,
            "max_seq_len": info.max_seq_len,
            "loadable": info.loadable,
            "reason": info.reason,
        })
    return {"models": entries, "resident": chat_sidecar.resident()}


@router.post("/chat/load")
async def load_chat_model(
    model_id: str = Query(...),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Load a checkpoint into the chat worker.

    Loading is separate from generation because it is slow (seconds to minutes)
    and the UI wants to show that separately. Also refuses while a run is active:
    the checkpoint and the run cannot both have the GPU.
    """
    from core.nanochat import chat_sidecar

    manager = _manager()
    if manager.is_active():
        raise HTTPException(
            status_code=409,
            detail="A nanochat run is in progress; stop it before loading a checkpoint for chat.",
        )
    try:
        info = chat_sidecar.resolve_model_id(model_id)
        chat_sidecar.load(info)
    except chat_sidecar.NanochatChatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "loaded", "model_id": model_id, "resident": chat_sidecar.resident()}


@router.post("/chat/unload")
async def unload_chat_model(
    current_subject: str = Depends(get_current_subject),
) -> dict:
    from core.nanochat import chat_sidecar

    return {"status": "unloaded" if chat_sidecar.unload() else "not_loaded"}


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    model_id: str = Query(...),
    prompt: str = Query(..., min_length=1),
    max_new_tokens: int = Query(default=256, ge=1, le=8192),
    temperature: float = Query(default=0.6, ge=0.0, le=2.0),
    top_k: int = Query(default=50, ge=0, le=1000),
    current_subject: str = Depends(get_current_subject),
):
    """Stream a reply, in the same cumulative-text shape the chat routes use.

    Every other backend here yields cumulative text and each caller diffs it
    itself, so matching that keeps the frontend's streaming path unchanged.
    """
    from core.nanochat import chat_sidecar

    cancel_event = threading.Event()

    async def event_generator():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def produce() -> None:
            try:
                for cumulative in chat_sidecar.generate_chat(
                    model_id, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    cancel_event=cancel_event,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, cumulative)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        thread = threading.Thread(target=produce, name="nanochat-chat", daemon=True)
        thread.start()

        previous = ""
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Keepalive: an idle generation behind a slow proxy would
                    # otherwise look like a dead connection.
                    yield ": keepalive\n\n"
                    continue
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    yield _format_sse(
                        json.dumps({"error": str(item)}), event="error"
                    )
                    return
                delta = item[len(previous):]
                previous = item
                if delta:
                    yield _format_sse(json.dumps({"text": delta}), event="token")
            yield _format_sse(json.dumps({"done": True}), event="done")
        finally:
            cancel_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------

@router.get("/checkpoints")
async def list_checkpoints(
    model_tag: Optional[str] = Query(default=None),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Checkpoints on disk, newest step first, per source."""
    from core.nanochat.orchestrator import NanochatConfig, NanochatRun

    run = NanochatRun(run_id="probe", config=NanochatConfig(model_tag=model_tag))
    out = []
    for source in ("base", "sft", "rl"):
        directory = run.checkpoints_dir(source)
        entries = []
        if directory.exists():
            for path in sorted(directory.glob("model_*.pt")):
                try:
                    step = int(path.stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                entries.append({
                    "step": step,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                })
        out.append({
            "source": source,
            "model_tag": model_tag or f"d{run.config.depth}",
            "path": str(directory),
            "exists": directory.exists(),
            "checkpoints": sorted(entries, key=lambda e: e["step"], reverse=True),
        })
    return {"checkpoints": out}
