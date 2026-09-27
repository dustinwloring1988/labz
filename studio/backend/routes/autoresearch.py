# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP surface for the autoresearch tab.

autoresearch runs nanochat's from-scratch pipeline under an agent loop: it edits
the training script, trains for a fixed budget, keeps the change if the
validation metric improved, and repeats. It is a third path alongside Train
(fine-tunes an existing model) and nanochat (one pipeline, once), so it gets its
own routes rather than a mode inside either.

The live view is driven by an SSE stream on ``/progress``, mirroring the shape
the nanochat tab already uses so the frontend can reuse that plumbing:
``Last-Event-ID`` resume, a heartbeat while an experiment is being set up, and an
``event_id`` that is a monotonically increasing integer. The stream polls the run
state rather than being pushed to, because the state is owned by a sequencer
thread in a different process boundary.

Two design points worth stating outright:

  - **The score in the table is the project's, not the agent's.** Every number the
    results tab shows is parsed out of ``run.log``, which is what program.md
    names as the source of truth. The agent's closing message is kept as prose,
    but it cannot move a score. An agent that misreads its own log must not be
    able to improve its recorded result.

  - **A 404 from ``/status`` is a state, not a failure.** The tab is opened before
    the first loop, so "no run" is the normal first answer and is reported as an
    empty status rather than an error the UI has to special-case.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from auth.authentication import get_current_subject
from core.autoresearch import agent as agent_mod
from core.autoresearch import chat_sidecar, environment, report, results
from core.autoresearch.orchestrator import (
    AutoresearchBusy,
    AutoresearchNotInstalled,
    estimate_duration_seconds,
    get_run_manager,
)
from models.autoresearch import (
    AutoresearchAgentInfo,
    AutoresearchAgentsResponse,
    AutoresearchChatModelsResponse,
    AutoresearchCheckpointInfo,
    AutoresearchCheckpointsResponse,
    AutoresearchEnvironmentResponse,
    AutoresearchLogResponse,
    AutoresearchMetricPoint,
    AutoresearchMetricsResponse,
    AutoresearchReportRequest,
    AutoresearchReportsResponse,
    AutoresearchResultsResponse,
    AutoresearchStartRequest,
    AutoresearchStartResponse,
    AutoresearchStatusResponse,
    AutoresearchStopRequest,
    AutoresearchStopResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# How often the progress stream samples the run state.
_STREAM_POLL_SECONDS = 1.0
# A heartbeat cadence before the first experiment has produced a step. The setup
# phases (the agent reading program.md, deciding what to change) can run for
# minutes with nothing to show, and a silent stream reads as a dead one.
_PREP_HEARTBEAT_EVERY_POLLS = 10
# After a step has been seen, a stall this long in polls is treated as a hang.
_STALL_TIMEOUT_POLLS = 3600  # an hour with no change
_MAX_LOG_LINES_PER_FRAME = 200


def _format_sse(data: str, event: str = "progress", event_id: Optional[int] = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {data}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


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
        raise HTTPException(status_code=404, detail="No autoresearch run")
    if expected and run.run_id != expected:
        raise HTTPException(status_code=409, detail="autoresearch run was superseded")
    return run


def _status_response(run, estimated_seconds: int = 0) -> AutoresearchStatusResponse:
    payload = run.state().to_dict()
    payload["estimated_seconds"] = estimated_seconds
    return AutoresearchStatusResponse(**payload)


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

@router.get("/environment", response_model=AutoresearchEnvironmentResponse)
async def get_environment(
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchEnvironmentResponse:
    """What is installed, what is missing, and what the tab can do about it."""
    status = environment.environment_status()
    return AutoresearchEnvironmentResponse(**vars(status), ready=status.ready())


@router.post("/environment/install", response_model=AutoresearchEnvironmentResponse)
async def install_environment(
    prepare_data: bool = Query(default=True),
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchEnvironmentResponse:
    """Copy the project in, build the venv, and prepare the data.

    Returns immediately and does the work on a thread: this is a multi-gigabyte
    torch download and a dataset download, and a request that waited for it
    would sit behind a proxy timeout long before it finished.
    """
    environment.install_async(prepare_data=prepare_data)
    status = environment.environment_status()
    return AutoresearchEnvironmentResponse(**vars(status), ready=status.ready())


@router.post("/environment/prepare-data", response_model=AutoresearchEnvironmentResponse)
async def prepare_data(
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchEnvironmentResponse:
    """Run ``prepare.py`` alone, for a workspace that already has a venv."""
    environment.prepare_data_async()
    status = environment.environment_status()
    return AutoresearchEnvironmentResponse(**vars(status), ready=status.ready())


@router.get("/agents", response_model=AutoresearchAgentsResponse)
async def list_agents(
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchAgentsResponse:
    """The agent CLIs this machine can run, and the state of the experiment repo.

    Reported rather than filtered: "Codex is not on PATH" and "Codex is on PATH
    but will not run" are different problems, and only one is fixed by
    installing something. The git state comes back with it because a dirty tree
    means a previous run was interrupted mid-edit, which the user should know
    before starting another.
    """
    agents = agent_mod.detect_agents()
    git = agent_mod.git_state(str(environment.checkout_root()))
    return AutoresearchAgentsResponse(
        agents=[
            AutoresearchAgentInfo(
                key=a.key, label=a.label, path=a.path, version=a.version,
                available=a.available, reason=a.reason, notes=a.notes,
            )
            for a in agents
        ],
        recommended=agent_mod.first_available_agent(),
        git_branch=git.branch,
        git_head=git.head,
        git_dirty=git.dirty,
        git_commits=git.commit_count,
        git_error=git.error,
    )


# -----------------------------------------------------------------------------
# Run lifecycle
# -----------------------------------------------------------------------------

@router.post("/start", response_model=AutoresearchStartResponse)
async def start_run(
    request: AutoresearchStartRequest,
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchStartResponse:
    """Start a loop of ``num_experiments`` agent runs.

    409 rather than 422 when one is already going: the request is well formed,
    and the only thing wrong is the machine's state, which the UI shows as a
    disabled Start button rather than a form error.
    """
    config = request.to_config()
    try:
        run = _manager().start(config)
    except AutoresearchBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AutoresearchNotInstalled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AutoresearchStartResponse(
        run_id=run.run_id,
        status="running",
        message=f"Running {config.num_experiments} experiments",
        estimated_seconds=estimate_duration_seconds(config),
    )


@router.post("/stop", response_model=AutoresearchStopResponse)
async def stop_run(
    request: AutoresearchStopRequest | None = None,
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchStopResponse:
    """Stop after the experiment in flight.

    Deliberately not a kill. The agent is partway through a file edit and a git
    commit, and interrupting it at an arbitrary moment can leave ``train.py``
    changed with no record of the change, which the next experiment would then
    build on. So the run is asked to finish, and only killed if it does not.
    """
    payload = request or AutoresearchStopRequest()
    result = _manager().stop()
    return AutoresearchStopResponse(
        status=str(result.get("status", "idle")),
        run_id=result.get("run_id"),
    )


@router.get("/status", response_model=AutoresearchStatusResponse)
async def run_status(
    expected_run_id: Optional[str] = Query(default=None),
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchStatusResponse:
    """The run's state, or an empty one if nothing has run yet.

    The empty case is a 200 with ``status="idle"`` rather than a 404: the tab is
    opened before the first loop, so "no run" is the normal first answer and the
    UI should not have to treat it as an error.
    """
    run = _manager().current()
    if run is None:
        return AutoresearchStatusResponse(
            run_id="", status="idle", phase="idle",
            message="No experiments have run yet",
        )
    if expected_run_id and run.run_id != expected_run_id:
        raise HTTPException(status_code=409, detail="autoresearch run was superseded")
    return _status_response(run)


# -----------------------------------------------------------------------------
# Progress stream
# -----------------------------------------------------------------------------

@router.post("/progress")
@router.get("/progress", include_in_schema=False)
async def stream_progress(
    request: Request,
    expected_run_id: Optional[str] = Query(default=None),
    current_subject: str = Depends(get_current_subject),
):
    """The live run, as SSE.

    Polls the run state once a second and emits a frame only when the payload
    changes. Emitting on a timer regardless would put a frame on the wire every
    second for a twelve-hour run that reports a step every few hundred
    milliseconds of work and nothing at all while the agent thinks.
    """
    run = _require_run(expected_run_id)
    run_id = run.run_id
    header = request.headers.get("last-event-id")
    last_event_id: Optional[int] = None
    if header is not None:
        try:
            last_event_id = int(header)
        except ValueError:
            last_event_id = None

    async def event_generator():
        yield "retry: 3000\n\n"
        counter = last_event_id or 0
        log_cursor = 0
        last_signature: Optional[str] = None
        seen_a_step = False
        polls_since_step_change = 0
        no_update_polls = 0

        while True:
            if await request.is_disconnected():
                return
            current = _manager().current()
            if current is None or current.run_id != run_id:
                yield _format_sse(
                    json.dumps({"reason": "run_ended"}), event="error", event_id=counter
                )
                return

            state = current.state()
            active = current.is_active()
            fresh_logs, next_log_cursor = current.logs_since(log_cursor)
            payload = state.to_dict()
            payload["logs"] = fresh_logs
            signature = json.dumps(payload, sort_keys=True, default=str)

            if signature != last_signature:
                counter += 1
                yield _format_sse(signature, event="progress", event_id=counter)
                last_signature = signature
                log_cursor = max(
                    log_cursor,
                    min(next_log_cursor, log_cursor + _MAX_LOG_LINES_PER_FRAME),
                )
                no_update_polls = 0
                if state.step:
                    seen_a_step = True
                    polls_since_step_change = 0
            else:
                no_update_polls += 1
                polls_since_step_change += 1
                every = _PREP_HEARTBEAT_EVERY_POLLS if not seen_a_step else 30
                if no_update_polls % every == 0:
                    yield _format_sse(
                        json.dumps({
                            "run_id": run_id,
                            "phase": state.phase,
                            "message": state.message,
                        }),
                        event="heartbeat",
                        event_id=counter,
                    )

            if not active:
                yield _format_sse(
                    json.dumps({
                        "run_id": run_id,
                        "status": state.status,
                        "error": state.error,
                        "message": state.message,
                        "experiments": [e.to_dict() for e in state.experiments],
                    }),
                    event="complete",
                    event_id=counter,
                )
                return

            if seen_a_step and polls_since_step_change > _STALL_TIMEOUT_POLLS:
                yield _format_sse(
                    json.dumps({
                        "run_id": run_id,
                        "error": "no progress for an hour",
                    }),
                    event="error",
                    event_id=counter,
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
# Logs
# -----------------------------------------------------------------------------

@router.get("/log", response_model=AutoresearchLogResponse)
async def fetch_logs(
    limit: int = Query(default=400, ge=1, le=2000),
    since: int = Query(default=0, ge=0),
    expected_run_id: Optional[str] = Query(default=None),
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchLogResponse:
    """Console lines, by cursor.

    The cursor is what makes the stream and this poll safe to run at the same
    time: both deliver, and the client drops anything at or below the last seq
    it drew, so a reconnect resumes instead of re-printing the tail.
    """
    run = _require_run(expected_run_id)
    if since > 0:
        logs, next_seq = run.logs_since(since, limit=limit)
    else:
        logs = run.log_tail(limit=limit)
        next_seq = logs[-1]["seq"] if logs else since
    return AutoresearchLogResponse(
        run_id=run.run_id,
        logs=logs,
        next_seq=next_seq,
    )


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

@router.get("/results", response_model=AutoresearchResultsResponse)
async def fetch_results(
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchResultsResponse:
    """The ledger, read from ``results.tsv``.

    Read from the file rather than from run memory so the history is the same
    one analysis.ipynb reads, survives a restart, and includes rows the agent
    added by hand.
    """
    summary = results.read_results(environment.checkout_root())
    return AutoresearchResultsResponse(**summary.to_dict())


@router.get("/metrics", response_model=AutoresearchMetricsResponse)
async def fetch_metrics(
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchMetricsResponse:
    """One point per experiment, for the progress chart.

    Crashes are included as points with no value rather than dropped, so the
    chart's x-axis is the experiment index and a gap means "this one produced
    nothing" instead of silently renumbering the run.
    """
    summary = results.read_results(environment.checkout_root())
    best = summary.best()
    baseline = summary.baseline()
    return AutoresearchMetricsResponse(
        series=[
            AutoresearchMetricPoint(
                index=row.index,
                val_bpb=row.val_bpb,
                memory_gb=row.memory_gb,
                is_best=row.is_best,
                status=row.status,
                description=row.description,
            )
            for row in summary.rows
        ],
        baseline_val_bpb=baseline.val_bpb if baseline else None,
        best_val_bpb=best.val_bpb if best else None,
    )


# -----------------------------------------------------------------------------
# Checkpoints and chat
# -----------------------------------------------------------------------------

@router.get("/checkpoints", response_model=AutoresearchCheckpointsResponse)
async def list_checkpoints(
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchCheckpointsResponse:
    """Every experiment whose weights were kept, best score first."""
    return AutoresearchCheckpointsResponse(
        checkpoints=[
            AutoresearchCheckpointInfo(**chat_sidecar._describe(info))
            for info in chat_sidecar.list_checkpoints()
        ]
    )


@router.get("/chat/models", response_model=AutoresearchChatModelsResponse)
async def list_chat_models(
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchChatModelsResponse:
    """Checkpoints, as chat model-picker entries.

    Its own endpoint rather than folded into the generic model list, because a
    checkpoint here is not a HuggingFace repo or a GGUF: it is a raw state_dict
    loadable only by the generated worker in the experiment workspace, and
    offering it anywhere that would read safetensors would break.
    """
    from models.autoresearch import AutoresearchCheckpointInfo

    return AutoresearchChatModelsResponse(
        models=[
            AutoresearchCheckpointInfo(**chat_sidecar._describe(info))
            for info in chat_sidecar.list_checkpoints()
        ],
        resident=chat_sidecar.resident(),
    )


@router.post("/chat/load")
async def load_chat_model(
    model_id: str = Query(...),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Load a checkpoint into the chat worker.

    Loading is separate from generation because it is slow and the UI wants to
    show that separately. Also refuses while a loop is running: the checkpoint
    and the experiments cannot both have the GPU, and an OOM at step 0 of an
    unattended run is a much worse outcome than a refused click.
    """
    if _manager().is_active():
        raise HTTPException(
            status_code=409,
            detail="An autoresearch loop is running; stop it before loading a "
                   "checkpoint for chat.",
        )
    try:
        info = chat_sidecar.resolve_model_id(model_id)
        chat_sidecar.load(info)
    except chat_sidecar.AutoresearchChatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "loaded", "model_id": model_id, "resident": chat_sidecar.resident()}


@router.post("/chat/unload")
async def unload_chat_model(
    current_subject: str = Depends(get_current_subject),
) -> dict:
    from core.autoresearch import chat_sidecar as sidecar

    return {"status": "unloaded" if sidecar.unload() else "not_loaded"}


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    model_id: str = Query(...),
    prompt: str = Query(..., min_length=1),
    max_new_tokens: int = Query(default=256, ge=1, le=8192),
    temperature: float = Query(default=0.8, ge=0.0, le=2.0),
    top_k: int = Query(default=50, ge=0, le=1000),
    current_subject: str = Depends(get_current_subject),
):
    """Stream a continuation, in the same cumulative-text shape the chat routes use.

    Every other backend here yields cumulative text and each caller diffs it
    itself, so matching that keeps the frontend's streaming path unchanged.
    """
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

        threading.Thread(
            target=produce, name="autoresearch-chat", daemon=True
        ).start()

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
                    yield _format_sse(json.dumps({"error": str(item)}), event="error")
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
# Report
# -----------------------------------------------------------------------------

@router.get("/reports", response_model=AutoresearchReportsResponse)
async def list_reports(
    current_subject: str = Depends(get_current_subject),
) -> AutoresearchReportsResponse:
    """Reports written so far, newest first."""
    return AutoresearchReportsResponse(
        reports=report.list_reports(environment.checkout_root())
    )


@router.post("/report")
async def generate_report(
    body: AutoresearchReportRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Stream a report about the search, written by a chosen model.

    The model is the user's choice between the catalog models the studio can
    serve and this search's own checkpoints, because they are good for different
    things: a report is a summarisation task over a table of numbers, and a
    50M-parameter model trained on TinyStories for five minutes will produce
    fluent nonsense about it. The second option exists for when nothing else is
    available, and the model is told in the prompt that it is reporting on its
    own training run so the output is not mistaken for an assessment of it.

    The request's own ``Request`` is handed to the completions path rather than a
    fabricated one, so a disconnect cancels the generation instead of leaving it
    running against a client that has gone.
    """
    run = _manager().current()
    run_id = run.run_id if run is not None and run.run_id else "latest"
    if body.run_id != "latest":
        run_id = body.run_id
    if run is not None and run.is_active():
        raise HTTPException(
            status_code=409,
            detail="The loop is still running. Stop it first: a report over a "
                   "partial search describes experiments that are still in flight.",
        )

    payload = report.ReportRequest(
        model_id=body.model_id,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        source=body.source,
        agent_key=body.agent_key,
        agent_args=list(body.agent_args),
    )
    cancel_event = threading.Event()

    async def event_generator():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async def drain() -> None:
            try:
                async for cumulative in report.generate_report(
                    payload,
                    fastapi_request=request,
                    owner=current_subject,
                    cancel_event=cancel_event,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, cumulative)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        # The generation is async and this SSE loop is already on the server's
        # loop, so it gets a thread with its own loop rather than nesting one
        # here. Everything crossing back goes through call_soon_threadsafe, which
        # is what makes the asyncio.Queue safe to share across the two.
        def produce() -> None:
            try:
                asyncio.run(drain())
            except Exception:  # noqa: BLE001 - drain already reported it
                logger.debug("autoresearch report producer ended", exc_info=True)

        producer = threading.Thread(
            target=produce, name="autoresearch-report", daemon=True
        )
        producer.start()

        accumulated = ""
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    detail = str(item)
                    yield _format_sse(json.dumps({"error": detail}), event="error")
                    return
                delta = item[len(accumulated):]
                accumulated = item
                if delta:
                    yield _format_sse(json.dumps({"text": delta}), event="token")

            saved = None
            if body.save and accumulated.strip():
                try:
                    saved = report.save_report(
                        environment.checkout_root(), run_id, accumulated
                    )
                except report.ReportError as exc:
                    # The report itself is fine; failing to file it must not
                    # discard several minutes of generation the user is reading.
                    yield _format_sse(
                        json.dumps({"warning": str(exc)}), event="warning"
                    )
            yield _format_sse(
                json.dumps({"done": True, "saved_to": saved, "run_id": run_id}),
                event="done",
            )
        finally:
            cancel_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


@router.get("/report", response_model=AutoresearchReportsResponse)
async def load_report(
    run_id: str = Query(default="latest"),
    current_subject: str = Depends(get_current_subject),
):
    """A previously written report, so the tab can reopen one it generated."""
    root = environment.checkout_root()
    if run_id == "latest":
        saved = report.list_reports(root)
        if not saved:
            raise HTTPException(status_code=404, detail="No report has been written yet")
        run_id = saved[0]["run_id"]
    text = report.load_report(root, run_id)
    if text is None:
        raise HTTPException(status_code=404, detail="No such report")
    return {"run_id": run_id, "text": text}
