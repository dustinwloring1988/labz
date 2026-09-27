# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The agent that runs each autoresearch experiment.

autoresearch is not a sweep. Its whole premise is that an LLM reads
``program.md``, changes ``train.py``, trains for a fixed budget, reads one
number back, and decides whether the change was worth keeping. That decision is
the experiment, so the studio cannot make it: it can only supply the agent, the
loop, and somewhere to show the outcome.

This module owns the agent side of that:

  - which agent CLIs are installed, and how to invoke each one headlessly
  - the per-experiment prompt, built from the project's own program.md
  - parsing what the agent says back, so the UI can show what it was doing
  - reading the one number that decides keep or discard out of the run log

Why a CLI subprocess and not a model call: the agent has to edit files, run
``uv run train.py``, read the log, and commit. That is a coding agent doing its
job in a real repository, which is exactly what these CLIs are for. Reimplementing
a file-editing loop over a chat completion would be a worse version of one of
them.

A note on unattended permissions. Each experiment asks the agent to run a
command that takes about five minutes, edit one file, and commit. That cannot be
answered by a prompt, so the agent has to be allowed to act without asking. The
loop runs in a copy of the project inside the app's own workspace, and
``skip_permissions`` is surfaced in the UI rather than buried, because a setting
that decides whether an LLM can run shell commands unattended should be visible
to the person who turned it on.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from utils.subprocess_compat import windows_hidden_subprocess_kwargs

# How long an agent gets for one experiment, on top of the training time.
# The project's own guidance is to kill a run past ten minutes, so this is set
# well above the worst case rather than at it: cutting an agent off while it is
# mid-commit loses the experiment's work, which is the one thing the loop cannot
# recover.
DEFAULT_AGENT_TIMEOUT_SECONDS = 45 * 60
# The hard floor under the timeout above. Shorter than this and a legitimate
# five-minute training pass plus a commit gets killed.
MIN_AGENT_TIMEOUT_SECONDS = 15 * 60
MAX_AGENT_TIMEOUT_SECONDS = 6 * 60 * 60

# One training pass in autoresearch is a fixed TIME_BUDGET of 300 seconds in
# prepare.py, plus a 21M-token evaluation and interpreter start-up. Used only to
# tell the user what "12 experiments" costs before they commit to it.
SECONDS_PER_EXPERIMENT_ESTIMATE = 6 * 60


@dataclass(frozen = True)
class AgentSpec:
    """One supported agent CLI and how to drive it headlessly."""

    key: str
    label: str
    executable: str
    # Version probe. A CLI that answers this is installed and runnable.
    version_args: tuple[str, ...]
    # Everything before the prompt.
    base_args: tuple[str, ...]
    # True when the agent's edits are accepted without prompting, and the flag
    # that does it. The key exists so the UI can warn about exactly this.
    auto_flag: str | None = None
    # Set when the CLI wants the working directory passed explicitly rather
    # than inherited, and the flag that says so.
    dir_flag: str | None = None
    # True when the agent's model is resolved by a cloud subscription whose
    # endpoint only serves that provider's own model names, so a name from
    # anywhere else is refused by the service before the agent does any work.
    subscription_backed: bool = False
    notes: str = ""


AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        key = "codex",
        label = "Codex CLI",
        executable = "codex",
        version_args = ("--version",),
        base_args = ("exec", "--skip-git-repo-check", "--color", "never"),
        # Matches codex's own wording. Without it codex prompts for approval on
        # every shell command, and a headless run hangs on the first `git`.
        auto_flag = "--dangerously-bypass-approvals-and-sandbox",
        dir_flag = "-C",
        subscription_backed = True,
        notes = "Reads your ~/.codex/config.toml, so its model and approvals apply.",
    ),
    AgentSpec(
        key = "opencode",
        label = "opencode",
        executable = "opencode",
        version_args = ("--version",),
        base_args = ("run",),
        auto_flag = "--auto",
        dir_flag = "--dir",
        notes = "Uses whichever provider opencode is configured with.",
    ),
    AgentSpec(
        key = "claude",
        label = "Claude Code",
        executable = "claude",
        version_args = ("--version",),
        base_args = ("-p", "--output-format", "text"),
        auto_flag = "--dangerously-skip-permissions",
        notes = "Needs `claude` on PATH and already authenticated.",
    ),
)

# The order the UI lists them in, which is also the order auto-detection walks.
#
# opencode first. It is the one of the three that is most often pointed at a
# local model, because its provider configuration is where a user sets one up, and
# an experiment loop that runs on the user's own weights rather than a cloud
# account is the default this feature should have.
PREFERRED_AGENT_ORDER = ("opencode", "codex", "claude")

_BY_KEY = {spec.key: spec for spec in AGENT_SPECS}


def get_spec(key: str) -> Optional[AgentSpec]:
    return _BY_KEY.get(key)


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

_VERSION_TTL_SECONDS = 60.0
_version_lock = threading.Lock()
_version_cache: dict[str, tuple[float, Optional[str]]] = {}
_resolved_cache: dict[str, tuple[float, Optional[tuple[str, str]]]] = {}


# Windows extensions CreateProcess will actually run, because it resolves a bare
# name through PATH and appends .exe. npm installs these CLIs as .cmd shims next
# to the real entry point, and those are the common case by far.
#
# Everything not listed here, notably .ps1, is rejected: PowerShell resolves
# those but CreateProcess does not, so a CLI that exists only as a .ps1 is one
# the app cannot launch however willing PowerShell would be.
_NOT_EXECUTABLE_SUFFIXES = (".ps1", ".psm1", ".vbs", ".js", ".mjs", ".py", ".pyw")
# A batch file is a script the OS runs through cmd.exe, so an argument reaching
# it has already been through a command interpreter.
_BATCH_SUFFIXES = (".cmd", ".bat")

# How deep below its launcher to look for a real binary. Generous because the
# two npm layouts differ sharply in depth, and bounded because a node_modules
# tree is enormous.
_REAL_BINARY_SEARCH_DEPTH = 8
# A cap on how many directory entries are walked, so a pathological install
# cannot turn detection into a long scan.
_REAL_BINARY_SCAN_LIMIT = 20_000


def _looks_launchable(path: str) -> bool:
    return not path.lower().endswith(_NOT_EXECUTABLE_SUFFIXES)


def _nearby_real_binary(launcher: Path, name: str) -> Optional[str]:
    """A real executable for a CLI whose launcher on PATH is a batch file.

    npm's .cmd shims run the actual program, which for both agents here is a
    native binary somewhere under node_modules. Finding it matters for two
    reasons: argv then reaches the program verbatim instead of passing through
    cmd.exe, and a batch file's own error handling stops being in the way.

    Breadth-first with a hard scan budget. The depth limit is not tight because
    the layouts differ sharply: opencode sits at
    ``node_modules/<pkg>/bin/<name>.exe`` while codex adds a platform package and
    a vendor directory below that, so a limit tuned for one misses the other.

    Best effort. A CLI whose binary cannot be found still works through its shim
    once the prompt is delivered by file rather than on the command line, which
    is why the caller does not treat a miss as fatal.
    """
    queue: list[Path] = [launcher.parent]
    visited = 0
    while queue:
        directory = queue.pop(0)
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        visited += len(entries)
        if visited > _REAL_BINARY_SCAN_LIMIT:
            return None
        for entry in entries:
            if entry.is_file():
                if entry.suffix.lower() in (".exe", ".com") and entry.stem.lower() == name.lower():
                    return str(entry)
            elif (
                entry.is_dir()
                and len(entry.parts) - len(launcher.parts) <= _REAL_BINARY_SEARCH_DEPTH
            ):
                queue.append(entry)
    return None


def _candidate_executables(spec: AgentSpec) -> list[str]:
    """Every way this CLI could be launched, best first.

    A real binary first, then whatever is on PATH. Not used to build a command
    with an absolute path when a relative one would do: the command carries the
    bare name, because that is what Popen resolves through PATH, and a
    hard-coded absolute path would keep pointing at an old install after the user
    upgrades the CLI.
    """
    candidates: list[str] = []
    found = shutil.which(spec.executable)
    if found:
        if found.lower().endswith(_BATCH_SUFFIXES):
            real = _nearby_real_binary(Path(found), spec.executable)
            if real:
                candidates.append(real)
        if _looks_launchable(found):
            candidates.append(found)
    return candidates


def _probe_version(executable: str, spec: AgentSpec) -> Optional[str]:
    """The CLI's version string, or None if it will not run.

    A CLI on PATH that cannot print its version is a broken install, and offering
    it in a picker would only move the failure to the first experiment.
    """
    key = f"{spec.key}:{executable}"
    now = time.monotonic()
    with _version_lock:
        cached = _version_cache.get(key)
    if cached is not None and now - cached[0] < _VERSION_TTL_SECONDS:
        return cached[1]
    version: Optional[str] = None
    try:
        proc = subprocess.run(
            [executable, *spec.version_args],
            capture_output = True,
            text = True,
            timeout = 45,
            encoding = "utf-8",
            errors = "replace",
            **windows_hidden_subprocess_kwargs(),
        )
        if proc.returncode == 0:
            line = (proc.stdout or proc.stderr or "").strip().splitlines()
            if line:
                version = line[0].strip()[:120]
    except Exception:
        version = None
    with _version_lock:
        _version_cache[key] = (time.monotonic(), version)
    return version


def resolve_executable(spec: AgentSpec) -> Optional[tuple[str, str]]:
    """The first candidate that actually runs, with its version.

    Probing rather than trusting the file extension, because the extension only
    says what the OS might do: an npm shim can name a binary that was moved by a
    partial upgrade, and that failure belongs here rather than halfway through
    the first experiment.
    """
    key = f"{spec.key}:resolve"
    now = time.monotonic()
    with _version_lock:
        cached = _resolved_cache.get(key)
    if cached is not None and now - cached[0] < _VERSION_TTL_SECONDS:
        return cached[1]

    resolved: Optional[tuple[str, str]] = None
    for candidate in _candidate_executables(spec):
        version = _probe_version(candidate, spec)
        if version is not None:
            resolved = (candidate, version)
            break
    with _version_lock:
        _resolved_cache[key] = (time.monotonic(), resolved)
    return resolved


@dataclass
class AgentAvailability:
    key: str
    label: str
    path: Optional[str]
    version: Optional[str]
    available: bool
    reason: Optional[str] = None
    notes: str = ""


def detect_agents() -> list[AgentAvailability]:
    """Every supported agent, in preference order, with why the missing ones are.

    Reported rather than filtered: "Codex CLI is not on PATH" is a different
    problem from "Codex CLI is on PATH but will not run", and only one of those
    is fixed by installing something.
    """
    ordered = sorted(AGENT_SPECS, key = lambda s: PREFERRED_AGENT_ORDER.index(s.key))
    out: list[AgentAvailability] = []
    for spec in ordered:
        candidates = _candidate_executables(spec)
        if not candidates:
            found_anywhere = shutil.which(spec.executable)
            reason = (
                "only a PowerShell script was found, which a subprocess cannot "
                "launch; reinstall it so a .cmd or .exe launcher is on PATH"
                if found_anywhere
                else "not found on PATH"
            )
            out.append(
                AgentAvailability(
                    key = spec.key,
                    label = spec.label,
                    path = None,
                    version = None,
                    available = False,
                    reason = reason,
                    notes = spec.notes,
                )
            )
            continue
        resolved = resolve_executable(spec)
        if resolved is None:
            out.append(
                AgentAvailability(
                    key = spec.key,
                    label = spec.label,
                    path = candidates[0],
                    version = None,
                    available = False,
                    reason = "found on PATH but did not respond to --version",
                    notes = spec.notes,
                )
            )
            continue
        path, version = resolved
        out.append(
            AgentAvailability(
                key = spec.key,
                label = spec.label,
                path = path,
                version = version,
                available = True,
                notes = spec.notes,
            )
        )
    return out


def first_available_agent() -> Optional[str]:
    for agent in detect_agents():
        if agent.available:
            return agent.key
    return None


# -----------------------------------------------------------------------------
# The prompt
# -----------------------------------------------------------------------------

# What the studio adds on top of the project's own program.md.
#
# program.md already tells the agent the whole protocol, including the loop and
# the keep/discard rule. Repeating it here would be a second copy to keep in
# step with the project. So this only does the three things the project's file
# cannot know: which experiment number this is, how many are left, and what the
# studio will do with the result.
_EXPERIMENT_HEADER = """\
You are running experiment {index} of {total} in an autoresearch loop.

Everything else you need is in program.md in this repository. Read it first and
follow it. The short version, so you know what is expected of this run:

  - Change train.py to test one idea. Only train.py is yours to edit;
    prepare.py, program.md and the results file are not.
  - Train by running `uv run train.py`, redirecting output to run.log.
  - Read val_bpb out of run.log. Lower is better.
  - Append exactly one row to results.tsv, then commit the change if it helped
    and undo it with git if it did not.

{state_block}\
After you finish, this run is over: do not start another experiment, and do not
edit train.py again. The loop will invoke you again for the next one.

Your final message is shown to the user as a one-line summary of what you did
and what it scored. Keep it to one or two sentences, and lead with the number.
"""


def _state_block(*, best_bpb: float | None, completed: int, baseline_bpb: float | None) -> str:
    """Where the search stands, in the units program.md uses.

    Given as a fact rather than a hint: the point of telling an agent the
    running best is that it should try to beat it, and a viterbi-smoothed
    sentence about "considering previous results" reliably produces an agent that
    ignores them.
    """
    lines = ["Current state of the search:", ""]
    if completed <= 0:
        lines.append("  - This is the first experiment. There is no baseline to beat yet,")
        lines.append("    so the number you produce becomes the baseline.")
    else:
        lines.append(f"  - {completed} experiment(s) are already recorded in results.tsv.")
    if baseline_bpb is not None:
        lines.append(f"  - The baseline val_bpb is {baseline_bpb:.6f}.")
    if best_bpb is not None:
        lines.append(f"  - The best val_bpb so far is {best_bpb:.6f}. Beat it, or try something")
        lines.append(
            "    a different kind of change rather than a smaller version of the last one."
        )
    lines.append("")
    return "\n".join(lines)


def build_experiment_prompt(
    *, index: int, total: int, best_bpb: float | None, completed: int, baseline_bpb: float | None
) -> str:
    return _EXPERIMENT_HEADER.format(
        index = index,
        total = total,
        state_block = _state_block(
            best_bpb = best_bpb,
            completed = completed,
            baseline_bpb = baseline_bpb,
        ),
    )


# -----------------------------------------------------------------------------
# Invocation
# -----------------------------------------------------------------------------

# The prompt is delivered in a file, not on the command line.
#
# Three reasons, and the first is the one that would otherwise be a bug: on
# Windows these CLIs are npm shims, and a .cmd is executed by cmd.exe, so
# anything in an argument has already been through a command interpreter. The
# prompt quotes file names and contains punctuation that cmd treats specially. A
# mangled prompt does not fail loudly, it produces an agent that misunderstands
# its instructions.
#
# The others: the prompt grows with the state block and would eventually meet
# the 8191-character command-line limit, and a file on disk is something the
# user can read afterwards to see exactly what each experiment was told.
PROMPT_DIR_NAME = ".autoresearch"
PROMPT_FILENAME = "experiment.md"

# What actually goes on the command line: the same short sentence every time.
_EXPERIMENT_INSTRUCTION = (
    f"Read {PROMPT_DIR_NAME}/{PROMPT_FILENAME} and do exactly what it says. "
    f"That file is the complete brief for this experiment."
)
# Kept as the historical name, because the tests and the logs refer to it.
_PROMPT_INSTRUCTION = _EXPERIMENT_INSTRUCTION

# What may not be passed through ``extra_args``.
#
# The brief is a file reference and the working directory is set by the caller,
# and a passthrough that could replace either would let a UI field change what
# the run actually does. Everything else the CLI accepts is fair game, which is
# the point: an agent CLI's own flags are how a user points it at a local model
# (``--model``, a provider flag, a base URL) and this is the only place that can
# be done without patching the backend.
_FORBIDDEN_EXTRA_ARGS = ("-C", "--cd", "--dir", "--directory", "-w", "--cwd")
# Public alias: the request model validates against this list rather than
# re-deriving it, so adding a flag here is enough to close it in both places.
FORBIDDEN_EXTRA_ARGS = _FORBIDDEN_EXTRA_ARGS

# The flags the supported CLIs spell model selection with. A passthrough field is
# opaque by design -- each CLI has its own spelling and refusing the ones we did
# not recognise would defeat the field -- but reading these is what lets a run be
# stopped before it starts rather than forty minutes in.
_MODEL_FLAGS = ("--model", "-m")

# The suffix the studio gives its own quantized builds, and the convention
# `routes/models.py` uses to mark one. A name ending in it is one of this machine's
# weights, which is the clearest possible signal that it cannot be a cloud slug.
_GGUF_SUFFIX = "-GGUF"


def _model_from_extra_args(extra_args: Sequence[str] | None) -> str | None:
    """The model named in a passthrough, or None if it names none.

    Reads the same argv `build_command` will build, and only the two shapes a
    model flag actually takes: ``--model x`` and ``--model=x``. Anything else --
    a bare flag with no value, a different flag entirely -- is left alone rather
    than guessed at, because a wrong guess here would refuse a working setup.
    """
    items = [str(item) for item in (extra_args or [])]
    for position, item in enumerate(items):
        for flag in _MODEL_FLAGS:
            if item == flag:
                following = items[position + 1] if position + 1 < len(items) else ""
                return following or None
            if item.startswith(f"{flag}="):
                return item.split("=", 1)[1] or None
    return None


def unusable_model_reason(agent_key: str, extra_args: Sequence[str] | None) -> str | None:
    """Why this agent cannot use the model these arguments name, or None.

    Codex authenticates against a ChatGPT subscription and its endpoint serves
    that subscription's model names and nothing else, so a name from anywhere
    else comes back as a 400 before the agent reads the brief. That is a
    guaranteed failure, and it used to be discovered by paying for it: a loop
    pointed at ``--model unsloth/gemma-4-E2B-it-GGUF`` burned six experiments of
    about eleven minutes each and produced six identical crash rows.

    Only the structurally impossible cases are refused here, and only for an agent
    that is genuinely subscription-backed. A name that is a real slug this
    particular plan cannot reach is a different problem, and it is caught by
    `_explain_repeat` stopping the run on the second identical crash rather than
    by a guess made before anything has run.

    opencode and Claude are not flagged: pointing those at a local model is the
    documented way to run the loop on your own weights, and refusing it would
    refuse the feature's preferred agent.
    """
    spec = get_spec(agent_key)
    if spec is None or not spec.subscription_backed:
        return None
    model = _model_from_extra_args(extra_args)
    if not model:
        return None
    if "/" in model or model.upper().endswith(_GGUF_SUFFIX):
        return (
            f"{spec.label} serves its ChatGPT subscription's own models, so it cannot "
            f"use {model!r}, which names a local or repository model. Pick an agent "
            f"pointed at a local server -- opencode reads its provider config -- or "
            f"drop the model flag to use your plan's."
        )
    return None


def _draft_filename(index: int, total: int) -> str:
    return f"brief-{index:03d}of{total:03d}.md"


def write_prompt(
    root,
    prompt: str,
    *,
    index: int = 0,
    total: int = 1,
) -> Path:
    """Put a brief where the agent will find it, under a per-experiment name.

    Written under a dot-directory that ``environment.install()`` adds to the copy's
    .gitignore, so the brief is never staged by the agent's own commits.

    Named per experiment rather than reused, so a brief left over from an
    interrupted run is still on disk to be read afterwards instead of being
    overwritten by the next one -- which is the only record of what the loop
    actually asked for.
    """
    directory = Path(root) / PROMPT_DIR_NAME
    path = directory / _draft_filename(index, total)
    try:
        directory.mkdir(parents = True, exist_ok = True)
        path.write_text(prompt, encoding = "utf-8", newline = "\n")
    except OSError as exc:
        raise RuntimeError(f"could not write the experiment brief to {path}: {exc}") from exc
    return path


# Kept for the report path, which writes one brief rather than a numbered series.
REPORT_PROMPT_FILENAME = "report-brief.md"
# What goes on the command line when a brief is being referred to. Short, ASCII,
# and identical every time, so it survives any shell without metacharacters.
_BRIEF_INSTRUCTION = (
    f"Read {PROMPT_DIR_NAME}/{REPORT_PROMPT_FILENAME} and do exactly what it says. "
    f"That file is the complete brief."
)


def write_report_prompt(root, prompt: str) -> Path:
    """Put the report brief where a one-shot agent invocation will find it."""
    directory = Path(root) / PROMPT_DIR_NAME
    path = directory / REPORT_PROMPT_FILENAME
    try:
        directory.mkdir(parents = True, exist_ok = True)
        path.write_text(prompt, encoding = "utf-8", newline = "\n")
    except OSError as exc:
        raise RuntimeError(f"could not write the report brief to {path}: {exc}") from exc
    return path


@dataclass
class AgentCommand:
    argv: list[str]
    prompt: str
    cwd: str
    timeout_seconds: int


def build_command(
    *,
    agent_key: str,
    prompt: str,
    cwd: str,
    skip_permissions: bool,
    timeout_seconds: int,
    extra_args: Optional[Iterable[str]] = None,
    instruction: str = _EXPERIMENT_INSTRUCTION,
) -> AgentCommand:
    """The exact argv for one agent run.

    The prompt is a file reference, not an argument. See
    ``_EXPERIMENT_INSTRUCTION`` for why that is not merely tidiness.

    ``extra_args`` is the passthrough that makes the agent's own configuration
    reachable from the UI. An agent CLI is a separate program with its own model
    setting, and the studio cannot know how someone has configured theirs -- so
    this is how a user points it at a local model, a different provider, or a
    specific reasoning effort without patching the backend. The arguments that
    decide *what the loop does* are refused, because those are the studio's, not
    the field's.

    ``instruction`` is the short sentence that points at the brief. It is a
    parameter because the report path reuses this whole builder against a
    different file, and rebuilding the argv afterwards would be a chance to
    silently drop an argument.
    """
    spec = _BY_KEY.get(agent_key)
    if spec is None:
        raise ValueError(f"unknown agent: {agent_key!r}")

    resolved = resolve_executable(spec)
    if resolved is None:
        raise RuntimeError(
            f"{spec.label} is not runnable: it is on PATH but did not answer "
            f"--version. Reinstall it, or pick another agent."
        )
    executable = resolved[0]

    argv = [executable, *spec.base_args]
    if skip_permissions and spec.auto_flag:
        argv.append(spec.auto_flag)
    for extra in extra_args or ():
        text = str(extra)
        if not text.strip():
            continue
        if any(text == flag or text.startswith(f"{flag}=") for flag in _FORBIDDEN_EXTRA_ARGS):
            raise ValueError(
                f"the agent's working directory and brief are set by the loop, so "
                f"{text!r} cannot be passed through"
            )
        argv.append(text)
    if spec.dir_flag:
        argv.extend([spec.dir_flag, cwd])
    argv.append(instruction)
    return AgentCommand(
        argv = argv,
        prompt = prompt,
        cwd = cwd,
        timeout_seconds = max(
            MIN_AGENT_TIMEOUT_SECONDS, min(MAX_AGENT_TIMEOUT_SECONDS, timeout_seconds)
        ),
    )


def run_one_shot(
    *,
    agent_key: str,
    prompt: str,
    cwd: str,
    extra_args: Optional[Iterable[str]] = None,
    timeout_seconds: int = 900,
) -> str:
    """Run an agent once for a non-interactive job and return what it said.

    Used for writing a report: the same CLIs that edit ``train.py`` can also be
    asked for prose, and a user whose local model is reachable through one of them
    would reasonably expect to write the report with that same model.

    The brief goes in a file and the argv carries the short reference, for the same
    reason the experiment loop does it: a .cmd shim runs through cmd.exe, and a
    prompt with the report's table in it has punctuation that would be eaten.
    """
    from core.autoresearch import environment

    write_report_prompt(cwd, prompt)
    command = build_command(
        agent_key = agent_key,
        prompt = prompt,
        cwd = cwd,
        # The brief is fully specified and the job is read-only prose, so this is
        # a case where unattended permissions are not needed. Left off, because an
        # agent that cannot ask is an agent that hangs.
        skip_permissions = False,
        timeout_seconds = timeout_seconds,
        extra_args = extra_args,
        instruction = _BRIEF_INSTRUCTION,
    )

    from utils.subprocess_compat import windows_hidden_subprocess_kwargs

    env = environment.experiment_environment()
    try:
        proc = subprocess.run(
            command.argv,
            cwd = command.cwd,
            env = env,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = command.timeout_seconds,
            stdin = subprocess.DEVNULL,
            **windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"the agent did not finish within {command.timeout_seconds // 60} minutes"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not run the agent: {exc}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        raise RuntimeError(
            f"the agent exited with code {proc.returncode}" + (f": {tail}" if tail else "")
        )
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError("the agent produced no output")
    return summarise_agent_output(out, max_chars = 100_000)


def summarise_agent_output(text: str, *, max_chars: int = 400) -> str:
    """The agent's closing message, trimmed to something a row can show.

    These CLMs end with a paragraph of explanation, often wrapped across lines
    and often with a bulleted recap. The first non-empty paragraph is the
    summary; anything after it is the agent talking to itself.
    """
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return ""
    paragraphs = [block.strip() for block in cleaned.split("\n\n") if block.strip()]
    if not paragraphs:
        return ""
    first = paragraphs[0]
    # A single long paragraph that is really a wrapped list: keep the first few
    # lines rather than the whole thing.
    lines = [line.strip() for line in first.split("\n") if line.strip()]
    joined = " ".join(lines)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1].rstrip() + "…"
    return joined


# -----------------------------------------------------------------------------
# Reading the result
# -----------------------------------------------------------------------------

# The summary block train.py prints at the end. Every key here is one program.md
# tells the agent to read, so the studio reads the same ones and cannot disagree
# with the agent about what happened.
_SUMMARY_KEYS = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
    "dataset",
    "train_batch_size",
    "eval_batch_size",
)

# Case is allowed through because two of the keys are mixed-case on purpose:
# train.py prints total_tokens_M and num_params_M, matching the units suffix
# style it uses elsewhere. A lowercase-only pattern silently drops both, and
# with them the parameter count the model picker shows.
_SUMMARY_RE = re.compile(r"^(?P<key>[A-Za-z_0-9]+):\s*(?P<value>\S+)\s*$", re.MULTILINE)


@dataclass
class RunOutcome:
    """What one ``uv run train.py`` produced, as far as the studio cares."""

    ok: bool
    val_bpb: Optional[float] = None
    summary: dict[str, str] = field(default_factory = dict)
    error: Optional[str] = None
    # The tail of the log, for the UI when a run failed. A train.py crash
    # reports nothing structured, so this is the only place the reason exists.
    tail: list[str] = field(default_factory = list)


def parse_run_log(text: str, *, tail_lines: int = 60) -> RunOutcome:
    """Pull the summary block out of a run log.

    Returns ok=False when there is no val_bpb, which is how a crash announces
    itself: train.py exits non-zero without printing the block, and program.md's
    own rule is that an empty grep means the run failed.
    """
    summary: dict[str, str] = {}
    for match in _SUMMARY_RE.finditer(text or ""):
        key = match.group("key")
        if key in _SUMMARY_KEYS:
            summary[key] = match.group("value")

    val_bpb: Optional[float] = None
    if "val_bpb" in summary:
        try:
            val_bpb = float(summary["val_bpb"])
        except ValueError:
            val_bpb = None

    lines = [line for line in (text or "").splitlines() if line.strip()]
    tail = lines[-tail_lines:]

    if val_bpb is None:
        return RunOutcome(
            ok = False,
            summary = summary,
            error = "the run did not print a val_bpb, so it failed before evaluation",
            tail = tail,
        )
    return RunOutcome(ok = True, val_bpb = val_bpb, summary = summary, tail = tail)


# -----------------------------------------------------------------------------
# Git state, for the UI
# -----------------------------------------------------------------------------


def _git(
    root: str,
    args: list[str],
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd = root,
        capture_output = True,
        text = True,
        timeout = timeout,
        encoding = "utf-8",
        errors = "replace",
        **windows_hidden_subprocess_kwargs(),
    )


@dataclass
class GitState:
    available: bool = False
    branch: str = ""
    head: str = ""
    dirty: bool = False
    commit_count: int = 0
    error: Optional[str] = None


def git_state(root: str) -> GitState:
    """The experiment repository's state, for the Configure tab.

    A dirty tree is worth surfacing before a loop starts: it means the last run
    was interrupted between editing train.py and committing, so the next
    experiment would build on an unrecorded change.
    """
    if not os.path.isdir(os.path.join(root, ".git")):
        return GitState(available = False, error = "the experiment repository is not initialised yet")
    try:
        branch = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
        head = _git(root, ["rev-parse", "--short", "HEAD"])
        status = _git(root, ["status", "--porcelain"])
        count = _git(root, ["rev-list", "--count", "HEAD"])
    except Exception as exc:
        return GitState(available = False, error = str(exc))
    if branch.returncode != 0:
        return GitState(
            available = False, error = (branch.stderr or "git rev-parse failed").strip()[-200:]
        )
    return GitState(
        available = True,
        branch = (branch.stdout or "").strip(),
        head = (head.stdout or "").strip(),
        dirty = bool((status.stdout or "").strip()),
        commit_count = int((count.stdout or "0").strip() or 0) if count.returncode == 0 else 0,
    )
