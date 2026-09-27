# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request and response schemas for the autoresearch endpoints.

Validation here is the boundary that stops a bad request becoming a twelve-hour
unattended run. The rules that matter:

  - an experiment count is bounded, because each one is a real agent invocation
    that trains a model and costs real time, and an unbounded field is a typo
    away from a weekend
  - the agent must be one this build knows how to drive, resolved at request time
    so a renamed CLI is a 400 rather than a subprocess failure an hour in
  - the report's model must name a real source, because the two sources take
    different code paths and a typo in one would otherwise be discovered halfway
    through a generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.autoresearch import agent as agent_mod

# One experiment is a coding agent editing a file, a five-minute training run and
# a git commit. The ceiling is a day of unattended work, which is the longest
# anyone should start without watching it.
MAX_EXPERIMENTS = 144

# Characters a shell treats as syntax, plus the POSIX ones. These reach a command
# line, and on Windows the agent's .cmd shim hands them to cmd.exe first, so this
# set is what stands between a user naming a local model and a user injecting a
# command into an unattended loop. A model name, a provider id and a base URL
# contain none of them, so nothing legitimate is lost.
_SHELL_SYNTAX = set("&|<>^%!`\"';$()\n\r\t")

# Long enough for a base URL or a config key=value, short enough that a field
# cannot be used to build an enormous command line.
_MAX_AGENT_ARG_CHARS = 200


def _clean_agent_args(value: List[str]) -> List[str]:
    """Validate a passthrough of agent CLI arguments.

    Shared by the loop and the report, because both land on a command line and a
    rule that existed in only one of them would be a rule with a hole in it.
    """
    cleaned: List[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if len(text) > _MAX_AGENT_ARG_CHARS:
            raise ValueError(f"agent argument is too long: {text[:40]}…")
        bad = _SHELL_SYNTAX.intersection(text)
        if bad:
            raise ValueError(
                f"agent argument {text!r} contains a character a shell treats as "
                f"syntax: {''.join(sorted(bad))}"
            )
        # Only the flags that decide what the loop does are refused. Every other
        # flag is the point of the field: `--model` and a provider or base-url flag
        # are how the loop is pointed at a local model, and refusing all flags
        # would leave no way to do that.
        head = text.split("=", 1)[0]
        if head in agent_mod.FORBIDDEN_EXTRA_ARGS:
            raise ValueError(
                f"{head!r} cannot be passed through: the loop sets the agent's "
                f"working directory and brief itself."
            )
        cleaned.append(text)
    return cleaned


class AutoresearchStartRequest(BaseModel):
    """Start a loop. Mirrors AutoresearchConfig, with the validation that matters."""

    model_config = ConfigDict(extra = "forbid")

    num_experiments: int = Field(default = 8, ge = 1, le = MAX_EXPERIMENTS)
    # opencode, matching the frontend default and agent.PREFERRED_AGENT_ORDER.
    # See AutoresearchConfig.agent for why that one leads.
    agent: str = "opencode"
    # Extra arguments for the agent CLI, passed through verbatim.
    #
    # The lever for running the loop on a local model: each agent CLI spells this
    # differently, so the field takes whatever that CLI accepts. Bounded and
    # character-filtered rather than free text, because it lands on a command
    # line, and the characters that matter there are the ones a shell treats
    # specially.
    agent_args: List[str] = Field(default_factory = list, max_length = 16)
    # Whether the agent may edit files and run commands without being asked. It
    # has to be on for a loop to run unattended at all, and it is surfaced in the
    # UI rather than assumed, because this is the setting that decides whether an
    # LLM can run shell commands on this machine.
    skip_permissions: bool = True
    agent_timeout_seconds: int = Field(
        default = agent_mod.DEFAULT_AGENT_TIMEOUT_SECONDS,
        ge = agent_mod.MIN_AGENT_TIMEOUT_SECONDS,
        le = agent_mod.MAX_AGENT_TIMEOUT_SECONDS,
    )
    # Zero means never give up, which is the honest default: a crash is
    # information, and a loop that gives up after two cannot tell a bad idea
    # from a bad machine.
    max_consecutive_failures: int = Field(default = 0, ge = 0, le = MAX_EXPERIMENTS)
    prepare_data: bool = True
    # A one-line steer, prepended to program.md's instructions.
    focus: str = Field(default = "", max_length = 2000)
    continue_on_failure: bool = True
    # Write the report as the run's last phase, with this model. On one GPU the
    # report cannot be written while the experiments hold the card, and the moment
    # it becomes possible is when the queue drains -- so it is offered as part of
    # starting the loop rather than as something to remember afterwards.
    report_after_finish: bool = False
    report_model_id: str = Field(default = "", max_length = 512)
    report_source: Literal["catalog", "autoresearch", "agent"] = "catalog"
    # Which CLI, when the report source is an agent. Blank means the loop's own
    # agent, so "write it with the same thing" needs no second decision.
    report_agent_key: str = Field(default = "", max_length = 64)
    # The same passthrough, so the report can be written by the same local model
    # the experiments were. Blank means the loop's own arguments.
    report_agent_args: List[str] = Field(default_factory = list, max_length = 16)

    @field_validator("report_agent_args")
    @classmethod
    def _check_report_agent_args(cls, value: List[str]) -> List[str]:
        return _clean_agent_args(value)

    @model_validator(mode = "after")
    def _check_report(self) -> "AutoresearchStartRequest":
        # Only checked when asked for. A blank model with the box unticked is the
        # default state of a form nobody filled in, not a mistake.
        if self.report_after_finish and not self.report_model_id.strip():
            raise ValueError(
                "Choose a model to write the report, or untick writing it when the loop finishes."
            )
        return self

    @model_validator(mode = "after")
    def _check_agent_model(self) -> "AutoresearchStartRequest":
        # A cross-field check, so it needs both fields and cannot be a field
        # validator. The arguments are deliberately opaque -- each agent CLI
        # spells its flags differently -- but a model name the selected agent's
        # endpoint can never serve is not a matter of spelling, and finding that
        # out costs an experiment. Roughly eleven minutes each, and the failure
        # repeats: a loop pointed at a local GGUF through a ChatGPT-backed agent
        # produced six identical crash rows before anyone saw why.
        reason = agent_mod.unusable_model_reason(self.agent, self.agent_args)
        if reason:
            raise ValueError(reason)
        return self

    @field_validator("agent")
    @classmethod
    def _check_agent(cls, value: str) -> str:
        if agent_mod.get_spec(value) is None:
            supported = ", ".join(spec.key for spec in agent_mod.AGENT_SPECS)
            raise ValueError(f"agent must be one of: {supported}")
        return value

    @field_validator("agent_args")
    @classmethod
    def _check_agent_args(cls, value: List[str]) -> List[str]:
        return _clean_agent_args(value)

    @field_validator("focus")
    @classmethod
    def _flatten_focus(cls, value: str) -> str:
        # Collapsed to one line because it is spliced into a prompt; a newline in
        # the middle of it would read as the end of the instruction block.
        return " ".join(value.split())

    def to_config(self):
        from core.autoresearch.orchestrator import AutoresearchConfig

        fields = set(AutoresearchConfig.__dataclass_fields__)
        payload = {key: value for key, value in self.model_dump().items() if key in fields}
        return AutoresearchConfig(**payload)


class AutoresearchStopRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    # Kept for symmetry with nanochat's stop request, which offers a save
    # toggle. autoresearch has nothing to save: the agent has already committed
    # or reverted by the time the loop notices a stop, and its weights are
    # captured per experiment either way.
    save: bool = True


class AutoresearchReportRequest(BaseModel):
    """Which model writes the report, and how long it may take."""

    model_config = ConfigDict(extra = "forbid")

    # Three sources, because there are three places a model can live:
    #   - "catalog": a model the studio can load and serve.
    #   - "autoresearch": one of this search's own checkpoints.
    #   - "agent": a coding-agent CLI, which is how a model the user has
    #     configured in that tool -- typically a local one -- is reached.
    model_id: str = Field(min_length = 1, max_length = 512)
    source: Literal["catalog", "autoresearch", "agent"] = "catalog"
    # Which CLI, when source is "agent".
    agent_key: str = Field(default = "", max_length = 64)
    # The same passthrough the experiment loop takes, so the report can be written
    # by the same local model the experiments were.
    agent_args: List[str] = Field(default_factory = list, max_length = 16)
    max_tokens: int = Field(default = 2000, ge = 128, le = 16384)
    temperature: float = Field(default = 0.7, ge = 0.0, le = 2.0)
    # Where to file the report. Optional: an unwritten report can still be copied.
    save: bool = True
    run_id: str = Field(default = "latest", max_length = 128, pattern = r"^[A-Za-z0-9._-]+$")

    @field_validator("agent_args")
    @classmethod
    def _check_report_agent_args(cls, value: List[str]) -> List[str]:
        # Same rule as the loop's passthrough, and for the same reason: this
        # reaches a command line that a .cmd shim hands to cmd.exe.
        return _clean_agent_args(value)


# --- responses ---


class AutoresearchEnvironmentResponse(BaseModel):
    source_present: bool
    source_path: str
    checkout_present: bool
    checkout_path: str
    venv_present: bool
    python_path: str
    uv_available: bool
    torch_installed: bool
    torch_version: Optional[str] = None
    cuda_available: bool
    device_name: Optional[str] = None
    data_prepared: bool
    git_ready: bool
    blocking_reason: Optional[str] = None
    install_state: str
    install_progress: float
    install_message: str
    ready: bool


class AutoresearchAgentInfo(BaseModel):
    key: str
    label: str
    path: Optional[str] = None
    version: Optional[str] = None
    available: bool
    reason: Optional[str] = None
    notes: str = ""


class AutoresearchAgentsResponse(BaseModel):
    agents: List[AutoresearchAgentInfo] = Field(default_factory = list)
    recommended: Optional[str] = None
    git_branch: str = ""
    git_head: str = ""
    git_dirty: bool = False
    git_commits: int = 0
    git_error: Optional[str] = None


class AutoresearchExperimentInfo(BaseModel):
    index: int
    status: str
    message: str = ""
    error: Optional[str] = None
    val_bpb: Optional[float] = None
    memory_gb: Optional[float] = None
    commit: str = ""
    description: str = ""
    summary: str = ""
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    checkpoint: Optional[str] = None


class AutoresearchStatusResponse(BaseModel):
    run_id: str
    status: str
    phase: str
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_seconds: float = 0.0

    experiment: int = 0
    total_experiments: int = 0
    progress_percent: float = 0.0

    step: int = 0
    loss: Optional[float] = None
    tok_per_sec: Optional[int] = None
    mfu: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None

    best_val_bpb: Optional[float] = None
    baseline_val_bpb: Optional[float] = None
    kept: int = 0
    discarded: int = 0
    crashed: int = 0

    experiments: List[AutoresearchExperimentInfo] = Field(default_factory = list)
    warnings: List[str] = Field(default_factory = list)

    # The report the loop wrote as its last phase, if it was asked to. Carried on
    # the status rather than only on disk so the tab is already showing it when
    # the run's last frame arrives.
    report_text: str = ""
    report_model: str = ""
    report_saved_to: Optional[str] = None
    report_error: Optional[str] = None
    report_running: bool = False

    resolved_config: Dict[str, Any] = Field(default_factory = dict)
    git_branch: str = ""
    git_head: str = ""
    stop_requested: bool = False

    # What a run of this size costs, so the UI can say it before the user starts
    # rather than after.
    estimated_seconds: int = 0


class AutoresearchStartResponse(BaseModel):
    run_id: str
    status: str
    message: str = ""
    estimated_seconds: int = 0


class AutoresearchStopResponse(BaseModel):
    status: str
    run_id: Optional[str] = None


class AutoresearchLogLine(BaseModel):
    seq: int
    stream: str
    line: str
    ts: float = 0.0


class AutoresearchLogResponse(BaseModel):
    run_id: str
    logs: List[AutoresearchLogLine] = Field(default_factory = list)
    next_seq: int = 0


class AutoresearchResultRow(BaseModel):
    index: int
    commit: str = ""
    val_bpb: Optional[float] = None
    memory_gb: Optional[float] = None
    status: str
    description: str = ""
    error: Optional[str] = None
    is_best: bool = False
    checkpoint: Optional[str] = None


class AutoresearchResultsResponse(BaseModel):
    rows: List[AutoresearchResultRow] = Field(default_factory = list)
    exists: bool = False
    total: int = 0
    kept: int = 0
    discarded: int = 0
    crashed: int = 0
    best_val_bpb: Optional[float] = None
    best_index: Optional[int] = None
    best_commit: Optional[str] = None
    baseline_val_bpb: Optional[float] = None


class AutoresearchMetricPoint(BaseModel):
    index: int
    val_bpb: Optional[float] = None
    memory_gb: Optional[float] = None
    is_best: bool = False
    status: str = ""
    # Carried so the frontier chart's tooltip can say what the experiment
    # actually changed. Without it the chart can show that a point improved and
    # not why, which is the half of the question the ledger exists to answer.
    description: str = ""


class AutoresearchMetricsResponse(BaseModel):
    # One point per experiment. The series is a step function rather than a line:
    # the score only changes when an experiment is recorded, and drawing it as a
    # line would imply a value between two experiments that never existed.
    series: List[AutoresearchMetricPoint] = Field(default_factory = list)
    baseline_val_bpb: Optional[float] = None
    best_val_bpb: Optional[float] = None


class AutoresearchCheckpointInfo(BaseModel):
    model_id: str
    name: str
    backend: str
    experiment: int
    commit: str = ""
    bytes: int = 0
    modified: float = 0.0
    val_bpb: Optional[float] = None
    num_params_m: Optional[float] = None
    depth: Optional[int] = None
    loadable: bool = True
    reason: Optional[str] = None
    # "base", because autoresearch has no chat stage. The UI labels it rather
    # than pretending the model will answer a question.
    kind: str = "base"
    hint: str = ""


class AutoresearchChatModelsResponse(BaseModel):
    models: List[AutoresearchCheckpointInfo] = Field(default_factory = list)
    resident: Optional[dict] = None


class AutoresearchCheckpointsResponse(BaseModel):
    checkpoints: List[AutoresearchCheckpointInfo] = Field(default_factory = list)


class AutoresearchReportFile(BaseModel):
    run_id: str
    path: str
    bytes: int
    modified: float


class AutoresearchReportsResponse(BaseModel):
    reports: List[AutoresearchReportFile] = Field(default_factory = list)
