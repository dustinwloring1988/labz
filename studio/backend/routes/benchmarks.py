# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP surface for the Benchmark tab.

The live view is driven by an SSE stream on ``/progress``, mirroring the shape
the nanochat tab already uses so the frontend can reuse its stream plumbing:
``Last-Event-ID`` resume, a heartbeat while a pass is quiet, and an ``event_id``
that increases monotonically.

The stream polls the run state rather than being pushed to, because the state is
owned by a reader thread in this process and the children report through a file
rather than a queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from auth.authentication import get_current_subject
from core.benchmarks import catalogue, reference_scores
from core.benchmarks.orchestrator import (
    BenchmarkBusy,
    BenchmarkConfig,
    BenchmarkUnavailable,
    get_run_manager,
)
from models.benchmarks import (
    BenchmarkCatalogueResponse,
    BenchmarkHistoryResponse,
    BenchmarkLeaderboardResponse,
    BenchmarkModelCandidate,
    BenchmarkModelListResponse,
    BenchmarkRunSummary,
    BenchmarkStartRequest,
    BenchmarkStartResponse,
    BenchmarkStatusResponse,
    BenchmarkStopResponse,
    LeaderboardRow,
)
from storage import benchmark_runs_db

logger = logging.getLogger(__name__)

router = APIRouter()

# How often the progress stream samples the run state.
_STREAM_POLL_SECONDS = 1.0
# Frames sent while a pass is running but has nothing new to say: a model load, a
# dataset download, a torch.compile. Those can take minutes.
_HEARTBEAT_EVERY_POLLS = 5
# Console lines a single frame may carry. Capped so one very chatty pass cannot
# blow up the stream buffer; the rest arrives on the next tick.
_MAX_LOG_LINES_PER_FRAME = 200


def _format_sse(
    data: str,
    event: str = "progress",
    event_id: Optional[int] = None,
) -> str:
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

    ``expected`` guards against a stale tab acting on a run that has already been
    replaced: a poll arriving after a new run started must not report the new
    run's progress to the old one.
    """
    run = _manager().current()
    if run is None:
        raise HTTPException(status_code = 404, detail = "No benchmark run")
    if expected and run.run_id != expected:
        raise HTTPException(status_code = 409, detail = "benchmark run was superseded")
    return run


# -----------------------------------------------------------------------------
# Catalogue
# -----------------------------------------------------------------------------


@router.get("/benchmarks", response_model = BenchmarkCatalogueResponse)
async def list_benchmarks(
    current_subject: str = Depends(get_current_subject),
) -> BenchmarkCatalogueResponse:
    """The benchmark picker.

    The table is nanochat's, read at request time, so a benchmark added upstream
    appears here without a change on this side.
    """
    return BenchmarkCatalogueResponse(**catalogue.catalogue_payload())


@router.get("/models", response_model = BenchmarkModelListResponse)
async def list_benchmark_models(
    current_subject: str = Depends(get_current_subject),
) -> BenchmarkModelListResponse:
    """Everything on this machine that can be benchmarked.

    Composed from the app's own listings rather than a second filesystem scan, so
    the picker cannot offer a model the rest of the app cannot see, a model
    downloaded after startup shows up here the same way it does in Chat, and the
    path allowlist and managed-account filtering those routes already enforce
    apply here too rather than being restated.

    Each source is optional. A host with no training outputs has no checkpoints
    and no adapters, which is a normal state rather than a failure, so a source
    that cannot be read is skipped and the others still render.
    """
    candidates: dict[str, BenchmarkModelCandidate] = {}

    def _add(candidate: BenchmarkModelCandidate) -> None:
        # Keyed by (id, path) so an adapter and the base it adapts are both
        # present and neither is folded into the other.
        candidates.setdefault(f"{candidate.id}\u0000{candidate.path}", candidate)

    for source, loader in (
        ("models", _local_models),
        ("checkpoints", _checkpoints),
        ("loras", _adapters),
    ):
        try:
            for candidate in await loader(current_subject):
                _add(candidate)
        except Exception as exc:  # noqa: BLE001
            # Logged and skipped rather than raised: the tab is more useful with
            # three of four sources than with an error page.
            logger.warning("could not list benchmark %s: %s", source, exc)

    ordered = sorted(candidates.values(), key = lambda row: (row.source, row.label))
    return BenchmarkModelListResponse(
        models = ordered,
        empty_reason = None
        if ordered
        else "No local models were found. Download one from the Model hub first.",
    )


async def _local_models(current_subject: str) -> list[BenchmarkModelCandidate]:
    """Downloaded and cached models, from the app's own local inventory."""
    from core.inference.local_model_resolver import local_load_dir
    from routes.models import list_local_models

    # via_api_key=False: this is a UI session, and the picker has to hand the
    # scorer a real path. The redaction that route applies is keyed off that
    # flag precisely because a UI operator is allowed to see local paths.
    payload = await list_local_models(
        models_dir = "./models",
        current_subject = current_subject,
        via_api_key = False,
    )
    rows: list[BenchmarkModelCandidate] = []
    for info in getattr(payload, "models", []) or []:
        raw = str(getattr(info, "path", "") or "")
        if not raw:
            continue
        # The inventory reports where a model *lives*, which for an HF cache entry
        # is the repo directory -- and a repo directory has no config.json, only
        # its snapshot does. Handing that straight to the loader fails with
        # "Unrecognized model ... should have a model_type key", which is the
        # scanner's path rather than a loadable one. local_load_dir is the
        # resolver the orchestrator itself uses to turn one into the other, so a
        # model the Chat tab can load is a model this tab can benchmark.
        path = local_load_dir(raw) or raw
        if not _is_loadable(path):
            # Offering an entry that cannot be loaded turns the failure into a
            # button press instead of a list entry, so it is dropped here.
            continue
        rows.append(
            BenchmarkModelCandidate(
                id = str(info.id),
                label = str(getattr(info, "display_name", None) or info.id),
                path = path,
                source = str(getattr(info, "source", "local") or "local"),
                # The inventory already knows the format, and it decides the
                # scoring path: a GGUF is scored by asking the running
                # llama-server rather than from local logits, so the picker has
                # to be able to say so before the user picks it.
                format = "gguf"
                if str(getattr(info, "model_format", "") or "").lower() == "gguf"
                else "safetensors",
            )
        )
    return rows


def _is_loadable(path: str) -> bool:
    """Whether a directory holds something a loader can open.

    One stat for the two manifests a candidate can be: a model config, or an
    adapter config. Deliberately shallow -- this runs per inventory row on a page
    that lists every model on the machine, and a full weight-file scan would make
    opening the tab slow for a check the loader will repeat anyway.
    """
    from pathlib import Path
    try:
        base = Path(path)
        return (base / "config.json").is_file() or (base / "adapter_config.json").is_file()
    except OSError:
        return False


async def _checkpoints(current_subject: str) -> list[BenchmarkModelCandidate]:
    """Training checkpoints, which are the main reason to have this tab."""
    from routes.models import list_checkpoints

    payload = await list_checkpoints(current_subject = current_subject)
    rows: list[BenchmarkModelCandidate] = []
    for model in getattr(payload, "models", []) or []:
        for checkpoint in getattr(model, "checkpoints", []) or []:
            path = str(getattr(checkpoint, "path", "") or "")
            if not path:
                continue
            name = str(getattr(checkpoint, "display_name", "") or Path(path).name)
            rows.append(
                BenchmarkModelCandidate(
                    id = f"checkpoint:{name}",
                    label = name,
                    path = path,
                    source = "checkpoint",
                    format = "safetensors",
                )
            )
    return rows


async def _adapters(current_subject: str) -> list[BenchmarkModelCandidate]:
    """Trained LoRA adapters, offered as a model to benchmark in their own right.

    A LoRA's path is what the scorer attaches, and its own base is discovered from
    the adapter's metadata, so the candidate carries the adapter path rather than
    a merged model that would have to be written out first.
    """
    from routes.models import scan_loras

    payload = await scan_loras(current_subject = current_subject)
    rows: list[BenchmarkModelCandidate] = []
    for row in _rows_of(payload):
        path = str(row.get("path") or row.get("lora_path") or "")
        if not path:
            continue
        label = str(row.get("name") or row.get("lora_name") or Path(path).name)
        rows.append(
            BenchmarkModelCandidate(
                id = f"lora:{label}",
                label = label,
                path = path,
                source = "lora",
                format = "safetensors",
                lora = True,
                base_model = row.get("base_model"),
            )
        )
    return rows


def _rows_of(payload) -> list[dict]:
    """Best-effort rows out of whatever shape a listing endpoint returned."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    for attribute in ("loras", "models", "data", "items"):
        value = getattr(payload, attribute, None)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    if isinstance(payload, dict):
        for attribute in ("loras", "models", "data", "items"):
            value = payload.get(attribute)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


# -----------------------------------------------------------------------------
# Run lifecycle
# -----------------------------------------------------------------------------


@router.post("/start", response_model = BenchmarkStartResponse)
async def start_run(
    payload: BenchmarkStartRequest, current_subject: str = Depends(get_current_subject)
) -> BenchmarkStartResponse:
    """Start benchmarking a model.

    Returns as soon as the run is launched. A suite takes minutes to hours, and a
    request that waited for it would time out long before the interesting part.
    """
    known, missing = catalogue.resolve_specs(payload.benchmarks)
    if missing:
        raise HTTPException(
            status_code = 400,
            detail = (
                f"Unknown benchmark(s): {', '.join(missing)}. "
                f"Available: {', '.join(sorted(row['key'] for row in catalogue.benchmarks()))}"
            ),
        )
    blocked = catalogue.unsupported_keys(payload.benchmarks)
    if blocked:
        detail = "; ".join(f"{key}: {reason}" for key, reason in blocked.items())
        raise HTTPException(status_code = 400, detail = f"Cannot run on this host. {detail}")

    config = BenchmarkConfig(
        model_id = payload.model_id,
        # The label a user recognises rather than the path they picked, so the
        # leaderboard says "qwen3-4b-step400" and not a temp directory.
        model_label = payload.model_label or Path(payload.model_path).name or payload.model_id,
        model_path = payload.model_path,
        format = payload.format,
        lora_path = payload.lora_path,
        load_in_4bit = payload.load_in_4bit,
        benchmarks = list(payload.benchmarks),
        max_problems = payload.max_problems,
        max_new_tokens = payload.max_new_tokens,
        batch_size = payload.batch_size,
        max_seq_length = payload.max_seq_length,
        trust_remote_code = payload.trust_remote_code,
        hf_token = payload.hf_token,
        judge_model_id = payload.judge_model_id,
        judge_model_label = payload.judge_model_label,
        judge_model_path = payload.judge_model_path,
    )

    try:
        run = _manager().start(config)
    except BenchmarkBusy as exc:
        raise HTTPException(status_code = 409, detail = str(exc))
    except BenchmarkUnavailable as exc:
        raise HTTPException(status_code = 409, detail = str(exc))

    return BenchmarkStartResponse(
        run_id = run.run_id,
        status = run.state.status,
        scoring_mode = config.scoring_mode_hint(),
    )


@router.post("/stop", response_model = BenchmarkStopResponse)
async def stop_run(current_subject: str = Depends(get_current_subject)) -> BenchmarkStopResponse:
    """Ask the run to stop. Co-operative, escalating to a kill if ignored."""
    result = _manager().stop()
    return BenchmarkStopResponse(
        status = str(result.get("status") or "idle"),
        message = str(result.get("message") or ""),
    )


@router.get("/status", response_model = BenchmarkStatusResponse)
async def run_status(
    expected_run_id: Optional[str] = Query(default = None),
    log_cursor: int = Query(default = 0),
    current_subject: str = Depends(get_current_subject),
) -> BenchmarkStatusResponse:
    """The run's state, for a poll or a tab that has just opened.

    A 404 rather than an empty state when a run is expected and gone, so the UI
    can tell "superseded" from "never started".
    """
    run = _require_run(expected_run_id)
    logs, _cursor = run.logs_since(log_cursor, _MAX_LOG_LINES_PER_FRAME)
    return BenchmarkStatusResponse(**run.state.to_dict(), logs = logs)


@router.post("/progress")
@router.get("/progress", include_in_schema = False)
async def stream_progress(
    request: Request,
    expected_run_id: Optional[str] = Query(default = None),
    current_subject: str = Depends(get_current_subject),
):
    """Server-sent progress for the live run view.

    The stream polls the run state and only emits when the payload actually
    changed, so an idle run costs a poll a second rather than a frame a second,
    and a reconnect resumes from the last event id rather than replaying the run.
    """
    run = _require_run(expected_run_id)
    run_id = run.run_id

    last_event_id = request.headers.get("last-event-id")
    counter = 0
    log_cursor = 0
    if last_event_id:
        try:
            counter = int(last_event_id)
        except ValueError:
            counter = 0

    async def event_generator():
        nonlocal counter, log_cursor
        # Tells the browser how long to wait before reconnecting when the
        # backend goes away mid-run, which it does on every update.
        yield "retry: 3000\n\n"
        last_signature = ""
        idle_polls = 0

        while True:
            if await request.is_disconnected():
                return
            current = _manager().current()
            if current is None or current.run_id != run_id:
                yield _format_sse(
                    json.dumps({"reason": "run_ended"}),
                    event = "error",
                    event_id = counter,
                )
                return

            fresh_logs, log_cursor = current.logs_since(log_cursor, _MAX_LOG_LINES_PER_FRAME)
            payload = current.state.to_dict()
            payload["logs"] = fresh_logs

            signature = json.dumps(payload, sort_keys = True)
            if signature != last_signature:
                counter += 1
                last_signature = signature
                yield _format_sse(signature, event = "progress", event_id = counter)
                idle_polls = 0
            else:
                # Something is running but has said nothing new, which for a
                # model load or a download is the normal case for minutes.
                idle_polls += 1
                if idle_polls % _HEARTBEAT_EVERY_POLLS == 0:
                    yield _format_sse(
                        json.dumps({"reason": "heartbeat", "phase": payload["phase"]}),
                        event = "heartbeat",
                        event_id = counter,
                    )

            if not current.is_active():
                yield _format_sse(
                    json.dumps({"status": current.state.status}),
                    event = "complete",
                    event_id = counter,
                )
                return

            await asyncio.sleep(_STREAM_POLL_SECONDS)

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -----------------------------------------------------------------------------
# History and leaderboard
# -----------------------------------------------------------------------------


def _summary(run: dict) -> BenchmarkRunSummary:
    return BenchmarkRunSummary(
        run_id = str(run.get("run_id") or ""),
        model_id = str(run.get("model_id") or ""),
        model_label = str(run.get("model_label") or ""),
        format = str(run.get("format") or "safetensors"),
        lora_path = run.get("lora_path"),
        status = str(run.get("status") or "completed"),
        scoring_mode = run.get("scoring_mode"),
        scores = run.get("scores") or [],
        composite = run.get("composite"),
        benchmarks_run = run.get("benchmarks_run") or [],
        max_problems = run.get("max_problems"),
        created_at = float(run.get("created_at") or 0.0),
        duration_seconds = float(run.get("duration_seconds") or 0.0),
    )


@router.get("/history", response_model = BenchmarkHistoryResponse)
async def history(
    limit: int = Query(default = 50, ge = 1, le = 500),
    current_subject: str = Depends(get_current_subject),
) -> BenchmarkHistoryResponse:
    """Stored runs, newest first."""
    runs = await asyncio.to_thread(
        benchmark_runs_db.list_runs, limit = limit, owner_subject = current_subject
    )
    return BenchmarkHistoryResponse(runs = [_summary(run) for run in runs])


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    """Remove a run from the leaderboard.

    Only a finished run can be removed. A running one is owned by the
    orchestrator, and deleting its row underneath it would leave the tab writing
    scores to a run the board no longer has.
    """
    active = _manager().current()
    if active is not None and active.run_id == run_id and active.is_active():
        raise HTTPException(status_code = 409, detail = "That run is still going")

    removed = await asyncio.to_thread(
        benchmark_runs_db.delete_run, run_id, owner_subject = current_subject
    )
    if not removed:
        raise HTTPException(status_code = 404, detail = "No such run")
    return {"status": "ok", "run_id": run_id}


@router.get("/leaderboard", response_model = BenchmarkLeaderboardResponse)
async def leaderboard(
    benchmarks: Optional[str] = Query(
        default = None,
        description = "Comma-separated benchmark keys to rank on. Defaults to all.",
    ),
    include_references: bool = Query(default = True),
    current_subject: str = Depends(get_current_subject),
) -> BenchmarkLeaderboardResponse:
    """Rank stored runs, alongside published reference scores.

    The two populations are ranked in one list because the comparison is the
    point, and labelled because they are not the same kind of number: a reference
    row was measured by somebody else's harness, on a machine that is not this
    one, and says nothing about what this machine would measure. See
    ``core/benchmarks/reference_scores.py`` for why that is stated rather than
    smoothed over.
    """
    if benchmarks:
        keys = [key.strip() for key in benchmarks.split(",") if key.strip()]
    else:
        keys = [row["key"] for row in catalogue.benchmarks()]

    runs = await asyncio.to_thread(
        benchmark_runs_db.best_per_model, keys, owner_subject = current_subject
    )

    rows: list[LeaderboardRow] = []
    for run in runs:
        by_key = {
            str(score.get("key")): score
            for score in run.get("scores") or []
            if score.get("status") == "complete"
        }
        # Only the benchmarks the board is ranked on, so a run that also did
        # something else does not gain columns the reader did not ask for.
        projected = {key: (by_key[key].get("accuracy") if key in by_key else None) for key in keys}
        if not any(value is not None for value in projected.values()):
            continue
        rows.append(
            LeaderboardRow(
                run_id = str(run.get("run_id") or ""),
                model_label = str(run.get("model_label") or ""),
                model_id = str(run.get("model_id") or ""),
                format = run.get("format"),
                source = "local",
                scoring_mode = run.get("scoring_mode"),
                scores = projected,
                truncated_keys = sorted(
                    key
                    for key, score in by_key.items()
                    if score.get("truncated") and key in projected
                ),
                composite = run.get("composite"),
                benchmarks_count = sum(1 for value in projected.values() if value is not None),
                created_at = float(run.get("created_at") or 0.0),
                duration_seconds = float(run.get("duration_seconds") or 0.0),
                max_problems = run.get("max_problems"),
                # Named so a judged cell is attributable. "50%" from a judge is not
                # the same claim as "50%" from exact matching, and a row that
                # mixed the two without saying which produced each would be
                # asking the reader to trust a number with no author.
                judge_model = next(
                    (
                        str(score.get("judge_model"))
                        for score in run.get("scores") or []
                        if score.get("judge_model")
                    ),
                    None,
                ),
            )
        )

    if include_references:
        for entry in reference_scores.references_for(keys):
            rows.append(
                LeaderboardRow(
                    model_label = str(entry["model_label"]),
                    model_id = str(entry["model_id"]),
                    source = "reference",
                    scores = {key: entry["scores"].get(key) for key in keys},
                    composite = _reference_composite(entry["scores"], keys),
                    benchmarks_count = len(entry["scores"]),
                    source_note = entry.get("source_note"),
                )
            )

    rows.sort(key = _rank_key)
    return BenchmarkLeaderboardResponse(
        rows = rows,
        sort_keys = keys,
        include_references = include_references,
        composite = rows[0].composite if rows else None,
    )


def _rank_key(row: LeaderboardRow):
    """Rank: measurement group first, then composite, with nulls last.

    Grouped by scoring mode, because a logits run, a generated run and a judged
    run are three different measurements. A judged run scoring higher does not
    mean the model is better; it means a second model's opinion was substituted
    for the definition, and ranking them in one column would let the method decide
    the ranking.

    Within a group, local rows lead so a reader sees their own measurement at the
    top of their own board, and a published reference never outranks a
    measurement taken on this machine.
    """
    return (
        0 if row.source == "local" else 1,
        _MODE_ORDER.get(row.scoring_mode or "", 3),
        row.composite is None,
        -(row.composite or 0.0),
    )


# Exact first, then the two weaker generative measurements, then anything a
# newer scoring path introduces. A new mode lands at the end rather than
# interleaving with the ones a reader already has a feel for.
_MODE_ORDER = {"logits": 0, "generated": 1, "judged": 2}


def _reference_composite(scores: dict, keys: list[str]) -> Optional[float]:
    """A published row's composite, on the same centred scale as a local run.

    Computed with the same centring the local runs use so the two are on one
    scale. A reference score carries no ``baseline`` of its own, so the baseline
    is taken from this host's registry; the two disagree only if nanochat's
    registry changed, and in that case a centred comparison would be wrong
    anyway.
    """
    values: list[float] = []
    for key in keys:
        accuracy = scores.get(key)
        if accuracy is None:
            continue
        rows, _missing = catalogue.resolve_specs([key])
        baseline = float(rows[0].get("baseline") or 0.0) if rows else 0.0
        room = 1.0 - baseline
        if room > 0:
            values.append((float(accuracy) - baseline) / room)
    if not values:
        return None
    return sum(values) / len(values)
