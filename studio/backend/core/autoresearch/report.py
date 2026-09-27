# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Writing a report about a finished autoresearch search.

The ledger says what happened: a number per experiment, and whether it was kept.
It does not say what the search *found*, which is the question someone actually
has the next morning. That is a language task, so it is given to a language
model.

Which model is the user's choice, and the choice matters:

  - **A catalog model.** Usually the right answer. A report is a summarisation
    task over a table of numbers, and a 50M-parameter model trained on
    TinyStories for five minutes will produce fluent nonsense about it. The
    models the Chat tab already knows about are the ones that can read a table.

  - **One of the autoresearch checkpoints.** Offered because it is the honest
    option when nothing else is available, and because seeing what a
    five-minute TinyStories model says about its own training run is
    informative in its own right. It is labelled as what it is in the prompt and
    in the UI, because a report written by the experiment's own subject matter
    should not be mistaken for an assessment of it.

The prompt is built here rather than in the route so the same text is produced
whichever model answers, and so it can be unit-tested without a GPU.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from core.autoresearch import environment, results

if TYPE_CHECKING:
    from typing import AsyncIterator

logger = logging.getLogger(__name__)

# A report over a hundred experiments is a table nobody reads. This is the
# number of rows sent; beyond it, the frontier and the headline rows are sent and
# the middle is summarised, because the middle is discard noise by construction.
MAX_ROWS_IN_PROMPT = 60

REPORT_FILENAME = "report.md"


class ReportError(RuntimeError):
    """Generating a report failed."""


# -----------------------------------------------------------------------------
# The prompt
# -----------------------------------------------------------------------------

_INSTRUCTIONS = """\
You are writing the report for a finished autoresearch search.

autoresearch trains a small language model from scratch on TinyStories for a
fixed wall-clock budget per attempt, repeatedly. Each experiment is one change to
the training script; the search keeps a change if it lowers `val_bpb`
(validation bits per byte, lower is better, and comparable across vocabulary
sizes) and reverts it if it does not. `memory_gb` is peak VRAM, and `status` is
keep, discard, or crash.

Write a report in Markdown with these sections, in this order:

## What was searched
One paragraph: how many experiments ran, over what baseline, and what the final
score was against it. Give the improvement as both an absolute change in val_bpb
and a percentage.

## The frontier
The experiments that moved the score, in the order they did. For each, what
changed in the training script and what it was worth. Be specific about the
change — the descriptions in the data are the agent's own words for what it
edited.

## What did not work
The discarded attempts, grouped by what they had in common. This is the most
useful part of the report and the easiest to skip: a pattern in the failures is
worth more than the score itself. Name the pattern and give two or three
examples.

## Crashes
If any experiment failed to finish, say what the failure was and whether it looks
like a bad idea or a bad run. If none did, say so in one line.

## What to try next
Three to five concrete, specific next experiments, each naming the file and the
constant or function to change. They should follow from the frontier and the
failures above, not from general advice about hyperparameter search.

Rules:

- Ground every claim in the data below. Do not invent an experiment, a number, or
  a mechanism.
- If a description is too vague to explain a result, say it was vague rather than
  inventing a mechanism for it.
- Do not open with a restatement of what autoresearch is.
- Plain Markdown. No front matter, no code fences around the whole thing.
"""


def _format_rows(summary: results.ResultsSummary) -> str:
    """The ledger as a table, with the frontier marked.

    A table rather than CSV because the report is written by a language model and
    a header row plus aligned columns is the form that survives that best.
    """
    rows = summary.rows
    if len(rows) > MAX_ROWS_IN_PROMPT:
        # Keep the frontier and the rows either side of it, then say what was
        # left out. The middle of a search is discarded attempts, and a hundred
        # of them in the prompt crowds out the handful that mattered.
        head = rows[:20]
        tail = rows[-20:]
        omitted = len(rows) - len(head) - len(tail)
        rows = head + tail
    else:
        omitted = 0

    lines = [
        "| # | val_bpb | memory_gb | status | description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        bpb = f"{row.val_bpb:.6f}" if row.val_bpb is not None else "-"
        memory = f"{row.memory_gb:.2f}" if row.memory_gb is not None else "-"
        marker = " **(kept)**" if row.is_best else ""
        description = (row.description or row.error or "").replace("|", "\\|")
        lines.append(
            f"| {row.index} | {bpb}{marker} | {memory} | {row.status} | {description} |"
        )
    if omitted:
        lines.append("")
        lines.append(
            f"({omitted} experiments in the middle of the search are omitted here. "
            f"All of them were discarded or crashed, since only improvements are "
            f"marked kept.)"
        )
    return "\n".join(lines)


def _headline_block(summary: results.ResultsSummary) -> str:
    best = summary.best()
    baseline = summary.baseline()
    lines = [f"- Experiments run: {summary.total}"]
    lines.append(
        f"- Kept: {summary.kept}, discarded: {summary.discarded}, crashed: {summary.crashed}"
    )
    if baseline is not None:
        lines.append(f"- Baseline val_bpb: {baseline.val_bpb:.6f} (experiment {baseline.index})")
    if best is not None:
        lines.append(f"- Best val_bpb: {best.val_bpb:.6f} (experiment {best.index})")
        if baseline is not None and baseline.val_bpb:
            delta = baseline.val_bpb - best.val_bpb
            percent = delta / baseline.val_bpb * 100.0
            lines.append(
                f"- Improvement over baseline: {delta:.6f} val_bpb ({percent:.2f}%)"
            )
        lines.append(f"- Winning change: {best.description or 'not described'}")
    memories = [r.memory_gb for r in summary.rows if r.memory_gb]
    if memories:
        lines.append(
            f"- Peak VRAM across experiments: {min(memories):.2f} to {max(memories):.2f} GB"
        )
    return "\n".join(lines)


def build_report_prompt(summary: results.ResultsSummary, *, subject_note: str = "") -> str:
    """The whole prompt: instructions, the headline numbers, then the table."""
    parts = [_INSTRUCTIONS]
    if subject_note:
        parts.append(subject_note)
    parts.append("Here is the search.\n")
    parts.append(_headline_block(summary))
    parts.append("")
    parts.append(_format_rows(summary))
    parts.append("")
    parts.append(
        f"The table is `results.tsv` in the experiment workspace, with one row per "
        f"attempt. `program.md` there is the agent's instruction file and explains "
        f"the protocol if you need it."
    )
    return "\n".join(parts)


def subject_note_for(checkpoint) -> str:
    """What to tell a model that is one of the checkpoints it is reporting on.

    Without this the model writes a confident report and the reader has no way to
    know it came from a 50M-parameter model that has only seen TinyStories. Being
    explicit costs a sentence and buys the reader the ability to weigh the output.
    """
    return (
        "Note on you: you are one of the models this search produced, not a "
        "separate assistant. You are writing about your own training run. You have "
        "no access to anything outside the data below, and no ability to check it. "
        "Write the report as the experiment's own summary of itself, and do not "
        "claim authority you do not have — where the data does not explain a "
        "result, say so."
    )


# -----------------------------------------------------------------------------
# Calling a model
# -----------------------------------------------------------------------------

@dataclass
class ReportRequest:
    # "catalog": a model the studio can serve. "autoresearch": one of this
    # search's own checkpoints. "agent": a coding-agent CLI, which is how a user
    # reaches a local model they have configured in that tool rather than in the
    # studio.
    model_id: str
    max_tokens: int = 2000
    temperature: float = 0.7
    source: str = "catalog"
    experiment: Optional[int] = None
    # Which CLI, when source is "agent". Kept separate from model_id because the
    # id here is an agent key rather than a model name.
    agent_key: str = ""
    # The same passthrough the experiment loop takes, so a report can be written
    # by the same local model the experiments were.
    agent_args: list = field(default_factory=list)


def _strip_fences(text: str) -> str:
    """Remove a whole-document code fence if the model wrapped the report in one.

    Models asked for Markdown will sometimes return it fenced, especially when
    the prompt is long. Stripping the fence is right here because the report is
    always the entire response, so a fence around all of it is never content.
    """
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    first = lines[0].strip()
    if not (first.startswith("```markdown") or first == "```"):
        return stripped
    if lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


@dataclass
class ReportResult:
    text: str
    model_id: str
    source: str
    saved_to: Optional[str] = None
    seconds: float = 0.0


async def generate_report(request: ReportRequest, *, fastapi_request,
                          owner: str, cancel_event: Optional[threading.Event] = None,
                          ) -> "AsyncIterator[str]":
    """Stream a report. Yields cumulative text, like every other backend here.

    Two sources, one shape:

      - ``source="catalog"`` goes through the studio's own model stack, so "the
        models we have" means exactly what it means in the Chat tab: anything
        servable locally, a GGUF, or a configured external provider.

      - ``source="autoresearch"`` goes through the chat sidecar, which is the
        only way to talk to a checkpoint that exists solely as a state_dict in
        the experiment workspace.

    Async because the catalog path is.

    ``background`` marks the case where there is no HTTP client: the report is
    the loop's own last phase rather than something a browser is waiting on.
    That distinction is not cosmetic — see ``_stream_catalog_model``.
    """
    background = fastapi_request is None
    summary = results.read_results(environment.checkout_root())
    if summary.total == 0:
        raise ReportError(
            "There are no experiments to report on yet. Run the loop first, or "
            "check that the agent wrote rows to results.tsv."
        )

    if request.source == "autoresearch":
        note = subject_note_for(_autoresearch_checkpoint_info(request.model_id))
    elif request.source == "agent":
        # An agent CLI reached through a provider is often the user's own local
        # model. Told the same thing, because "a model that is configured
        # somewhere else" is exactly the situation where a confident report reads
        # as an assessment when it is a self-report.
        note = subject_note_for(None)
    else:
        note = ""
    prompt = build_report_prompt(summary, subject_note=note)

    if request.source == "autoresearch":
        async for chunk in _stream_checkpoint(
            request, prompt, cancel_event=cancel_event
        ):
            yield chunk
        return

    if request.source == "agent":
        async for chunk in _stream_agent_source(
            request, prompt, cancel_event=cancel_event
        ):
            yield chunk
        return

    async for chunk in _stream_catalog_model(
        request, prompt, fastapi_request=fastapi_request, owner=owner,
        cancel_event=cancel_event, background=background,
    ):
        yield chunk


async def _stream_agent_source(request: ReportRequest, prompt: str, *,
                                cancel_event: Optional[threading.Event] = None,
                                ) -> "AsyncIterator[str]":
    """Write the report with a coding-agent CLI, in one shot.

    The third source, and the one that reaches a local model most often. A user
    who has a model running under opencode or Codex has configured it there, not
    in the studio, and asking them to re-describe it in a second place is the
    wrong default when the same tool that runs the experiments can also write the
    report.

    One shot, so there is nothing to stream: these CLIs emit their prose when it
    is done rather than token by token in a form this can read. Yielded once,
    which the SSE wrapper turns into a single frame. That is a real difference
    from the other two sources and is why the Report tab says a report "arrives
    as it is written" rather than promising progress it cannot show here.
    """
    from core.autoresearch import agent as agent_mod

    outbox: "asyncio.Queue" = asyncio.Queue()
    sentinel = object()

    def produce() -> None:
        try:
            text = agent_mod.run_one_shot(
                agent_key=request.agent_key or request.model_id,
                prompt=prompt,
                cwd=str(environment.checkout_root()),
                extra_args=request.agent_args,
            )
            outbox.put_nowait(text)
        except Exception as exc:  # noqa: BLE001
            outbox.put_nowait(exc)
        finally:
            outbox.put_nowait(sentinel)

    threading.Thread(
        target=produce, name="autoresearch-report-agent", daemon=True
    ).start()

    while True:
        item = await outbox.get()
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise ReportError(str(item)) from item
        yield item


async def _stream_checkpoint(request: ReportRequest, prompt: str, *,
                             cancel_event: Optional[threading.Event] = None,
                             ) -> "AsyncIterator[str]":
    """Run the prompt on one of this search's own checkpoints.

    The sidecar's generator is synchronous and blocking (it reads a subprocess),
    so it is drained on a worker thread and handed to the event loop. Yielded
    text is cumulative, which is the convention the whole codebase uses.
    """
    from core.autoresearch import chat_sidecar

    queue: "asyncio.Queue" = asyncio.Queue()
    SENTINEL = object()
    loop = asyncio.get_running_loop()

    # The sidecar's generator is synchronous and blocking, so it runs on its own
    # thread and marshals through the loop: asyncio.Queue is not thread-safe.
    def hand_off(item) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def produce_threaded() -> None:
        try:
            for cumulative in chat_sidecar.generate_chat(
                request.model_id, prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=50,
                cancel_event=cancel_event,
            ):
                hand_off(cumulative)
        except Exception as exc:  # noqa: BLE001
            hand_off(exc)
        finally:
            hand_off(SENTINEL)

    threading.Thread(
        target=produce_threaded, name="autoresearch-report", daemon=True
    ).start()

    while True:
        item = await queue.get()
        if item is SENTINEL:
            return
        if isinstance(item, Exception):
            raise ReportError(str(item)) from item
        yield item


async def _stream_catalog_model(request: ReportRequest, prompt: str, *,
                                fastapi_request, owner: str,
                                cancel_event: Optional[threading.Event] = None,
                                background: bool = False,
                                ) -> "AsyncIterator[str]":
    """Run the prompt on one of the studio's own models.

    Two paths, because the caller is different.

    **Interactive** (a browser is waiting): through the same OpenAI-compatible
    completions function the Chat tab uses, so the model may be a local
    safetensors model, a GGUF on llama.cpp, or a configured external provider,
    and the request may switch or load models. All of that machinery is the point
    of going through it.

    **Background** (the loop's own last phase, no client): straight at whichever
    backend is resident.

    The background path exists because the interactive one is built around an
    HTTP request. It reads ``request.url.path`` for its audit log and
    ``request.headers`` for auth, so it needs a real Request, and there is no
    real client to give it. Fabricating one is how this ends up failing deep in
    the inference stack on a header that was a plain function. What the request
    was carrying -- a client that might go away -- does not exist here, and the
    loop's own cancel event is the only thing that can stop the work.

    The one thing the background path gives up is loading a model that is not
    already resident. That is stated in the error rather than discovered by a
    failed request, because the fix is "load it in Chat first" and a user should
    not have to infer that from a stack trace.
    """
    if background:
        async for chunk in _stream_resident_model(
            request, prompt, cancel_event=cancel_event
        ):
            yield chunk
        return

    from models.inference import ChatCompletionRequest
    from routes.inference import produce_openai_chat_completions

    try:
        payload = ChatCompletionRequest.model_validate({
            "model": request.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        })
    except Exception as exc:  # noqa: BLE001
        raise ReportError(f"could not build the report request: {exc}") from exc

    try:
        response = await produce_openai_chat_completions(
            payload, fastapi_request, owner, cancel_on_disconnect=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise ReportError(f"could not reach {request.model_id}: {exc}") from exc

    status = int(getattr(response, "status_code", 200) or 200)
    if status >= 400:
        detail = await _response_detail(response)
        raise ReportError(detail or f"{request.model_id} refused the request (HTTP {status})")

    iterator = getattr(response, "body_iterator", None)
    if iterator is None:
        raise ReportError("the completions path did not return a stream")

    text = ""
    async for raw in iterator:
        if cancel_event is not None and cancel_event.is_set():
            return
        for delta, error in _deltas_from_sse(_as_text(raw)):
            if error:
                raise ReportError(error)
            if not delta:
                continue
            text += delta
            yield text
    if not text.strip():
        raise ReportError(
            f"{request.model_id} returned nothing. It may not be loaded, or the "
            f"model may not support the context length this report needs."
        )


async def _stream_resident_model(request: ReportRequest, prompt: str, *,
                                 cancel_event: Optional[threading.Event] = None,
                                 ) -> "AsyncIterator[str]":
    """Stream from whichever backend is already resident.

    Both backends yield cumulative text and interleave dict control frames, so the
    reader skips dicts and keeps the last snapshot -- the same two rules
    ``utils.datasets.llm_assist`` follows when it runs a completion of its own.
    """
    from core.inference import get_inference_backend
    from routes.inference import get_llama_cpp_backend

    messages = [{"role": "user", "content": prompt}]
    llama = get_llama_cpp_backend()
    if getattr(llama, "is_loaded", False):
        def generate():
            return llama.generate_chat_completion(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        name = "the loaded GGUF"
    else:
        backend = get_inference_backend()
        if not getattr(backend, "active_model_name", None):
            raise ReportError(
                "No model is loaded, so the report cannot be written in the "
                "background. Load the model in Chat first, untick writing the "
                "report when the loop finishes, or write one by hand from the "
                "Report tab once the experiments are done."
            )

        def generate():
            return backend.generate_chat_response(
                messages=messages,
                temperature=request.temperature,
                max_new_tokens=request.max_tokens,
            )
        name = backend.active_model_name

    outbox: "asyncio.Queue" = asyncio.Queue()
    sentinel = object()

    def produce() -> None:
        try:
            for chunk in generate():
                # Control frames arrive in-band on both backends; a dict is not
                # text and assigning one to the accumulator would corrupt it.
                if isinstance(chunk, dict):
                    continue
                outbox.put_nowait(chunk)
        except Exception as exc:  # noqa: BLE001
            outbox.put_nowait(exc)
        finally:
            outbox.put_nowait(sentinel)

    threading.Thread(
        target=produce, name="autoresearch-report-gen", daemon=True
    ).start()

    while True:
        item = await outbox.get()
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise ReportError(f"{name} stopped early: {item}") from item
        yield item


def _as_text(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw or "")


def _deltas_from_sse(text: str) -> Iterator[tuple[str, Optional[str]]]:
    """Pull content deltas out of one SSE chunk, plus any error frame.

    Yields ``(delta, None)`` for content and ``("", message)`` for an error, so
    the caller does not have to know the frame layout. An unparseable frame is
    skipped rather than raised: a keepalive comment or a future event type is
    not a reason to fail a report that is otherwise arriving.
    """
    for block in text.replace("\r\n", "\n").split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            error = payload.get("error")
            if error:
                message = error.get("message") if isinstance(error, dict) else str(error)
                yield "", str(message)
                continue
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content, None


async def _response_detail(response) -> str:
    """The body of a non-2xx completions response, as text.

    The status alone is useless here: "HTTP 400" does not say which model was
    refused or why, and the reason is almost always in the body.
    """
    body = getattr(response, "body", None)
    if body is None:
        return ""
    try:
        raw = body() if callable(body) else body
        if hasattr(raw, "__await__"):
            raw = await raw
        return _as_text(raw)[:600]
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def report_path(checkout: Path, run_id: str) -> Path:
    return checkout / "reports" / f"{run_id}.md"


def save_report(checkout: Path, run_id: str, text: str) -> str:
    """Write the report next to the results, so it survives the tab being closed.

    Markdown rather than HTML: the ledger it is derived from is a TSV, the
    project ships a notebook that plots it, and a report someone will paste into
    a commit message or a PR should not be a proprietary format.
    """
    path = report_path(checkout, run_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = _strip_fences(text)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(body.rstrip() + "\n")
    except OSError as exc:
        raise ReportError(f"could not save the report: {exc}") from exc
    return str(path)


def load_report(checkout: Path, run_id: str) -> Optional[str]:
    path = report_path(checkout, run_id)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def list_reports(checkout: Path) -> list[dict]:
    directory = checkout / "reports"
    if not directory.is_dir():
        return []
    out: list[dict] = []
    for path in directory.glob("*.md"):
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append({
            "run_id": path.stem,
            "path": str(path),
            "bytes": stat.st_size,
            "modified": stat.st_mtime,
        })
    out.sort(key=lambda entry: entry["modified"], reverse=True)
    return out
