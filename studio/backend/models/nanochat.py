# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request and response schemas for the nanochat endpoints.

Validation here is the boundary that stops a bad configuration becoming a
multi-hour run that fails at step 40,000. The rules that matter:

  - a depth must produce a model, and must be one the arithmetic in
    ``core.nanochat.presets`` can size
  - a batch size must be a multiple of what one micro-batch produces, because
    nanochat asserts this and dies immediately otherwise
  - benchmark and task names must exist in nanochat's own catalogue, resolved at
    request time so a renamed task is a 400 rather than a subprocess failure
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.nanochat import presets

# The nanochat CLI names. A value outside this set is a client bug.
VALID_STAGES = (
    "dataset", "tokenizer", "pretrain", "base_eval", "sft", "chat_eval", "rl",
)

# nanochat's window_pattern tiles L (full) and S (half) attention across layers.
# S reduces FLOPs on long context at the cost of quality; the speedrun default is
# "SSSL", but SDPA cannot do sliding-window attention at all, and consumer cards
# usually fall back to SDPA, so the orchestrator defaults to "L".
VALID_WINDOW_PATTERNS = ("L", "S", "SL", "LS", "SSL", "SLS", "LSS", "SSSL", "SLSS", "LSSL", "LL", "SS")


class NanochatStartRequest(BaseModel):
    """Start a run. Mirrors NanochatConfig, with the validation that matters."""

    model_config = ConfigDict(extra="forbid")

    # --- model ---
    depth: int = Field(default=presets.DEFAULT_DEPTH, ge=1, le=64)
    aspect_ratio: int = Field(default=presets.DEFAULT_ASPECT_RATIO, ge=1, le=512)
    head_dim: int = Field(default=presets.DEFAULT_HEAD_DIM, ge=8, le=512)
    max_seq_len: int = Field(default=2048, ge=128, le=32768)
    window_pattern: str = Field(default="L")
    model_tag: Optional[str] = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")

    # --- data ---
    dataset: str = "climbmix-400b"
    num_shards: int = Field(default=32, ge=1, le=7000)
    vocab_size: int = Field(default=presets.DEFAULT_VOCAB_SIZE, ge=256, le=131072)
    train_tokenizer: bool = True

    # --- horizon ---
    num_iterations: int = Field(default=1000, ge=1, le=10_000_000)
    param_data_ratio: float = Field(default=12.0, ge=0.0, le=200.0)
    total_batch_size: int = Field(default=32768, ge=1, le=67_108_864)
    device_batch_size: int = Field(default=4, ge=1, le=1024)
    fp8: bool = False

    # --- learning rates ---
    embedding_lr: float = Field(default=0.3, gt=0.0, le=10.0)
    unembedding_lr: float = Field(default=0.008, gt=0.0, le=10.0)
    matrix_lr: float = Field(default=0.02, gt=0.0, le=10.0)
    scalar_lr: float = Field(default=0.5, gt=0.0, le=10.0)
    weight_decay: float = Field(default=0.28, ge=0.0, le=10.0)
    warmup_steps: int = Field(default=40, ge=0, le=100_000)
    warmdown_ratio: float = Field(default=0.65, ge=0.0, le=1.0)
    final_lr_frac: float = Field(default=0.05, ge=0.0, le=1.0)

    # --- cadence ---
    eval_every: int = Field(default=100, ge=-1, le=1_000_000)
    eval_tokens: int = Field(default=2 * 1024 * 1024, ge=1024, le=10_000_000_000)
    core_metric_every: int = Field(default=500, ge=-1, le=1_000_000)
    sample_every: int = Field(default=200, ge=-1, le=1_000_000)
    save_every: int = Field(default=-1, ge=-1, le=1_000_000)
    core_metric_max_per_task: int = Field(default=500, ge=1, le=100_000)

    # --- SFT ---
    sft_tasks: List[str] = Field(default_factory=lambda: ["smoltalk", "mmlu", "gsm8k"])
    mmlu_epochs: int = Field(default=3, ge=0, le=100)
    gsm8k_epochs: int = Field(default=4, ge=0, le=100)
    sft_iterations: int = Field(default=400, ge=1, le=10_000_000)
    sft_sample_every: int = Field(default=100, ge=-1, le=1_000_000)

    # --- RL ---
    run_rl: bool = False
    rl_epochs: int = Field(default=1, ge=1, le=1000)
    rl_examples_per_step: int = Field(default=16, ge=1, le=10_000)
    rl_num_samples: int = Field(default=16, ge=1, le=1024)
    rl_max_new_tokens: int = Field(default=256, ge=16, le=8192)
    rl_eval_every: int = Field(default=60, ge=1, le=1_000_000)
    rl_save_every: int = Field(default=60, ge=1, le=1_000_000)

    # --- benchmarks ---
    base_benchmarks: List[str] = Field(default_factory=lambda: ["core", "bpb", "sample"])
    chat_benchmarks: List[str] = Field(
        default_factory=lambda: ["arc-easy", "arc-challenge", "mmlu", "gsm8k", "humaneval"]
    )

    # --- stages ---
    stages: List[str] = Field(
        default_factory=lambda: list(VALID_STAGES)
    )

    @field_validator("window_pattern")
    @classmethod
    def _check_window_pattern(cls, value: str) -> str:
        if value not in VALID_WINDOW_PATTERNS:
            raise ValueError(
                f"window_pattern must be one of {', '.join(VALID_WINDOW_PATTERNS)}"
            )
        return value

    @field_validator("stages")
    @classmethod
    def _check_stages(cls, value: List[str]) -> List[str]:
        unknown = [s for s in value if s not in VALID_STAGES]
        if unknown:
            raise ValueError(f"unknown stages: {', '.join(unknown)}")
        # Preserve order but drop repeats, so a double-clicked checkbox does not
        # run a stage twice.
        seen: set[str] = set()
        ordered = []
        for stage in value:
            if stage not in seen:
                seen.add(stage)
                ordered.append(stage)
        return ordered

    @field_validator("sft_tasks", "chat_benchmarks", "base_benchmarks")
    @classmethod
    def _dedupe(cls, value: List[str]) -> List[str]:
        seen: set[str] = set()
        ordered = []
        for item in value:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    @model_validator(mode="after")
    def _check_consistency(self) -> "NanochatStartRequest":
        # nanochat asserts total_batch_size is a multiple of
        # device_batch_size * max_seq_len * world_size. Violating it kills the
        # stage seconds after a multi-gigabyte model has already been built, so
        # it is caught here instead.
        micro = self.device_batch_size * self.max_seq_len
        if self.total_batch_size % micro != 0:
            raise ValueError(
                f"total_batch_size ({self.total_batch_size}) must be a multiple of "
                f"device_batch_size * max_seq_len ({self.device_batch_size} * "
                f"{self.max_seq_len} = {micro})"
            )
        if "rl" in self.stages and not self.run_rl:
            # Harmless, but it means the UI and the request disagree about what
            # will happen, which is worse than a clear default.
            object.__setattr__(self, "run_rl", True)
        if "pretrain" in self.stages and self.num_iterations < 1:
            raise ValueError("num_iterations must be at least 1 when pretraining")
        return self

    def to_config(self):
        """The orchestrator's config object."""
        from core.nanochat.orchestrator import NanochatConfig

        fields = set(NanochatConfig.__dataclass_fields__)
        payload = {
            key: value for key, value in self.model_dump().items() if key in fields
        }
        return NanochatConfig(**payload)


class NanochatStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save: bool = Field(
        default=True,
        description=(
            "Save a checkpoint before stopping. False kills the stage, which can "
            "leave the last checkpoint short of the current step."
        ),
    )


class NanochatValidateRequest(BaseModel):
    """Ask what a configuration would cost, without starting it."""

    model_config = ConfigDict(extra="forbid")

    config: NanochatStartRequest = Field(default_factory=NanochatStartRequest)
    # Not enforced: the point of this endpoint is to find out whether the run
    # would fit, so a too-large configuration has to be describable.
    strict: bool = False


# --- responses ---

class NanochatEnvironmentResponse(BaseModel):
    checkout_present: bool
    checkout_path: str
    checkout_ref: str
    venv_present: bool
    python_path: str
    uv_available: bool
    torch_installed: bool
    torch_version: Optional[str] = None
    cuda_available: bool
    device_name: Optional[str] = None
    blocking_reason: Optional[str] = None
    install_state: str
    install_progress: float
    install_message: str
    ready: bool


class NanochatStageInfo(BaseModel):
    key: str
    status: str
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_seconds: Optional[float] = None


class NanochatCheckpointInfo(BaseModel):
    stage: Optional[str] = None
    source: Optional[str] = None
    step: Optional[int] = None
    path: Optional[str] = None
    model_tag: Optional[str] = None
    num_params: Optional[int] = None
    ts: float = 0.0


class NanochatStatusResponse(BaseModel):
    run_id: str
    status: str
    phase: str
    current_stage: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_seconds: float = 0.0

    step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0

    loss: Optional[float] = None
    val_bpb: Optional[float] = None
    chatcore: Optional[float] = None
    reward: Optional[float] = None
    tok_per_sec: Optional[int] = None
    mfu: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    eta_seconds: Optional[float] = None
    elapsed_seconds: float = 0.0

    resolved_config: Optional[dict] = None
    num_params: Optional[int] = None

    stages: List[NanochatStageInfo] = Field(default_factory=list)
    checkpoints: List[NanochatCheckpointInfo] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class NanochatMetricPoint(BaseModel):
    step: int
    value: Optional[float] = None
    total_steps: Optional[int] = None


class NanochatMetricsResponse(BaseModel):
    run_id: str
    # Keyed by what the number measures, not by stage: loss and reward share an
    # event shape but not a scale, and plotting them on one axis would read as a
    # divergence.
    series: Dict[str, List[NanochatMetricPoint]] = Field(default_factory=dict)


class NanochatSample(BaseModel):
    stage: str
    step: int
    # completion | chat | rollout. A base model continues text rather than
    # replying, and the feed has to label that difference.
    mode: str
    prompt: str
    completion: str
    reward: Optional[float] = None
    advantage: Optional[float] = None
    ts: float = 0.0


class NanochatBenchmark(BaseModel):
    stage: str
    name: str
    accuracy: Optional[float] = None
    baseline: Optional[float] = None
    centered: Optional[float] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    is_partial: bool = False
    ts: float = 0.0


class NanochatSamplesResponse(BaseModel):
    run_id: str
    samples: List[NanochatSample] = Field(default_factory=list)


class NanochatBenchmarksResponse(BaseModel):
    run_id: str
    benchmarks: List[NanochatBenchmark] = Field(default_factory=list)


class NanochatFitResponse(BaseModel):
    depth: int
    num_params: int
    transformer_params: int = 0
    n_embd: int
    n_head: int
    model_memory_bytes: int
    training_memory_bytes: int
    tokens: int
    total_flops: float
    est_seconds: Optional[float] = None
    fits: bool = True
    reasons: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    device_name: Optional[str] = None
    peak_flops_bf16: Optional[float] = None
    device_type: str = "cpu"
    max_batch_size: Optional[int] = None


class NanochatValidateResponse(BaseModel):
    ok: bool
    fit: Optional[NanochatFitResponse] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    # Populated when the request could not even be built.
    config_errors: List[str] = Field(default_factory=list)


class NanochatPresetsResponse(BaseModel):
    presets: List[dict]
    default_depth: int
    stages: List[dict]
    defaults: dict


class NanochatCatalogueResponse(BaseModel):
    schema_version: int
    unavailable: bool = False
    reason: Optional[str] = None
    datasets: Dict[str, Any] = Field(default_factory=dict)
    default_dataset: Optional[str] = None
    training_tasks: Dict[str, Any] = Field(default_factory=dict)
    default_training_tasks: List[str] = Field(default_factory=list)
    benchmarks: Dict[str, Any] = Field(default_factory=dict)
    default_chat_benchmarks: List[str] = Field(default_factory=list)
    base_evals: Dict[str, Any] = Field(default_factory=dict)
    default_base_evals: List[str] = Field(default_factory=list)
    trained_on_by_default: List[str] = Field(default_factory=list)
    registry_problems: List[str] = Field(default_factory=list)


class NanochatStartResponse(BaseModel):
    run_id: str
    status: str
    message: str
    # Carried back so the UI can render the estimate it showed, and so a later
    # "why did it change?" question has an answer.
    fit: Optional[NanochatFitResponse] = None


class NanochatStopResponse(BaseModel):
    status: str
    message: str


class NanochatLogLine(BaseModel):
    """One line of console output from a run.

    ``seq`` is monotonic per run and is the cursor: a client asks for everything
    after the last ``seq`` it rendered, so a reconnect resumes instead of
    replaying the tail it already has.
    """

    seq: int
    stream: str = "stdout"
    stage: Optional[str] = None
    line: str
    ts: float


class NanochatLogResponse(BaseModel):
    run_id: str
    lines: List[NanochatLogLine] = Field(default_factory=list)
    # The cursor to send as ``since`` next time. Echoed back so a client that
    # connected late and got an empty tail still has somewhere to resume from.
    seq: int = 0
