# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic contracts for the Benchmark tab.

The benchmark table itself is NOT declared here. It is read from nanochat's own
registry at runtime (see ``core/benchmarks/catalogue.py``) so that a benchmark
added upstream shows up in the picker without a change on this side, and so the
two cannot drift. These models describe the shape that registry already has,
plus the fields only this app can fill in: whether a benchmark can run on this
host, and what kind of scoring pass it needs.

Scoring happens in three passes, each in the interpreter that owns the
dependencies it needs (see ``core/benchmarks/orchestrator.py`` for why):

    materialize  nanochat venv   tasks -> items on disk
    score        backend venv    items -> predictions, using the model under test
    grade        nanochat venv   predictions -> accuracy, via task.evaluate()

The three ``scoring_mode`` values below are the observable consequence of that
split, and the reason every score records which one produced it. Numbers from
different modes are not interchangeable, so a leaderboard that mixed them
silently would be lying.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# A categorical benchmark is answered by comparing one letter against the others,
# so the model under test only has to score the next token. A generative one has
# to write an answer that is then graded against a reference.
#
# GGUF cannot be loaded into the scoring process at all (there is no safetensors
# checkpoint behind it), so it is scored by asking the running llama-server to
# generate and reading the letter off the text. That is a weaker measurement of
# the same thing, which is why it gets its own value rather than being folded
# into the exact one.
#
# `judged` is a generative score a second model decided. It is kept apart from the
# exact value for the same reason: it is a different measurement, by a model whose
# biases are its own, and a board that ranked the two together would be letting
# the method decide the ranking.
ScoringMode = Literal["logits", "generated", "mixed", "judged"]

# How a prediction is compared to the gold answer.
BenchmarkKind = Literal["categorical", "generative"]

ModelFormat = Literal["safetensors", "gguf"]

BenchmarkRunStatus = Literal["idle", "running", "completed", "error", "stopped"]

BenchmarkScoreStatus = Literal["pending", "running", "complete", "skipped", "failed"]


class BenchmarkSpec(BaseModel):
    """One selectable benchmark, as nanochat's registry describes it.

    ``baseline`` is the accuracy of guessing (0.25 for four-way multiple choice,
    0.0 for a free-form answer). It is not decoration: without it a 57-subject
    MMLU and a GSM8K would average into a number that means nothing, so every
    aggregate this app reports is centred on the baseline.
    """

    key: str = Field(..., min_length = 1, max_length = 64)
    label: str = Field(..., min_length = 1, max_length = 120)
    task_name: str = Field(..., min_length = 1, max_length = 120)
    notes: str = ""
    baseline: float = Field(default = 0.0, ge = 0.0, le = 1.0)
    kind: BenchmarkKind = "categorical"
    requires_posix_sandbox: bool = False
    max_problems: Optional[int] = None
    is_default: bool = False
    # Filled in by this app, not by nanochat: whether this benchmark can actually
    # be run on this host, and why not when it cannot.
    supported: bool = True
    unavailable_reason: Optional[str] = None


class BenchmarkCatalogueResponse(BaseModel):
    """The benchmark picker, plus whether anything can run yet."""

    benchmarks: List[BenchmarkSpec] = Field(default_factory = list)
    default_benchmarks: List[str] = Field(default_factory = list)
    # False when the nanochat environment that owns the datasets is not set up.
    # The tab renders its install affordance rather than an empty picker.
    available: bool = True
    reason: Optional[str] = None
    # HumanEval executes model-written code behind a POSIX `resource` cap. On a
    # host without it, running the benchmark anyway would mean executing
    # untrusted generated code unisolated, so it is reported unsupported instead.
    posix_sandbox: bool = True


class BenchmarkModelCandidate(BaseModel):
    """A model the tab can benchmark, flattened from wherever it was discovered.

    ``format`` decides which scoring pass runs: safetensors gets the exact
    logit comparison, GGUF the generated one.
    """

    id: str = Field(..., min_length = 1, max_length = 400)
    label: str = Field(..., min_length = 1, max_length = 200)
    path: str = Field(..., min_length = 1, max_length = 1000)
    source: str = Field(..., max_length = 40)
    format: ModelFormat = "safetensors"
    # Set when the candidate is an adapter that needs a base model alongside it.
    lora: bool = False
    base_model: Optional[str] = Field(default = None, max_length = 400)


class BenchmarkModelListResponse(BaseModel):
    models: List[BenchmarkModelCandidate] = Field(default_factory = list)
    # True when nothing was found, so the picker can explain rather than render
    # an empty list the user cannot interpret.
    empty_reason: Optional[str] = None


class BenchmarkStartRequest(BaseModel):
    """Start a benchmark run.

    extra="forbid" so a stale client cannot silently send a field this side no
    longer honours, which would otherwise look like it took effect.
    """

    model_config = ConfigDict(extra = "forbid")

    model_id: str = Field(..., min_length = 1, max_length = 400)
    model_label: Optional[str] = Field(default = None, max_length = 200)
    model_path: str = Field(..., min_length = 1, max_length = 1000)
    format: ModelFormat = "safetensors"
    # An adapter to evaluate on top of model_path. The base is loaded and the
    # adapter attached, so a LoRA run measures the merged behaviour.
    lora_path: Optional[str] = Field(default = None, max_length = 1000)
    load_in_4bit: bool = False

    benchmarks: List[str] = Field(..., min_length = 1, max_length = 32)
    # None means "every problem the benchmark has", which for MMLU is 14k and
    # takes hours. The UI offers a bounded default and says what it is.
    max_problems: Optional[int] = Field(default = 400, ge = 8, le = 20000)
    max_new_tokens: int = Field(default = 512, ge = 16, le = 8192)
    batch_size: int = Field(default = 8, ge = 1, le = 256)
    max_seq_length: int = Field(default = 2048, ge = 256, le = 131072)
    trust_remote_code: bool = False
    hf_token: Optional[str] = Field(default = None, max_length = 400)

    # A second model that scores the generative answers, after the model under
    # test has answered. All three or none: a label without a path is a
    # half-filled form rather than a judge, and the backend ignores it.
    #
    # Only the generative benchmarks are judged. A categorical answer is a single
    # letter that exact logit matching already measures, so a second model there
    # could only add noise -- the run records that it declined rather than
    # pretending to have scored it.
    judge_model_id: Optional[str] = Field(default = None, max_length = 400)
    judge_model_label: Optional[str] = Field(default = None, max_length = 200)
    judge_model_path: Optional[str] = Field(default = None, max_length = 1000)

    @model_validator(mode = "after")
    def _judge_is_whole_or_nothing(self):
        # Without a path there is no model to load, so the id and label left over
        # from a half-filled form are dropped rather than carried into a run that
        # would report a judge it never had.
        if not self.judge_model_path:
            if self.judge_model_id or self.judge_model_label:
                logger.warning("benchmark start named a judge but no path; ignoring the judge")
            self.judge_model_id = None
            self.judge_model_label = None
        return self

    @field_validator("benchmarks")
    @classmethod
    def _check_benchmarks(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()
        for key in value:
            key = key.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        if not cleaned:
            raise ValueError("select at least one benchmark")
        return cleaned


class BenchmarkScore(BaseModel):
    """One benchmark's outcome within a run."""

    key: str
    label: str = ""
    kind: BenchmarkKind = "categorical"
    status: BenchmarkScoreStatus = "pending"
    accuracy: Optional[float] = None
    baseline: float = 0.0
    # (accuracy - baseline) / (1 - baseline), i.e. the share of the gap between
    # guessing and a perfect score that was closed. This is the only figure that
    # is comparable across benchmarks, so it is what the leaderboard sorts on.
    centered: Optional[float] = None
    correct: int = 0
    total: int = 0
    elapsed_seconds: float = 0.0
    # True when max_problems cut the benchmark short, so the number is an
    # estimate from a sample rather than the published-set score.
    truncated: bool = False
    scoring_mode: Optional[ScoringMode] = None
    error: Optional[str] = None

    # What a judge model concluded, kept beside nanochat's own verdict.
    #
    # `exact_accuracy` is always kept on a judged row, because it is the
    # definition and it is deterministic. `accuracy` carries whichever was
    # reported: the judge's, for a generative benchmark a judge was asked about.
    # A reader who wants the definition rather than the opinion has it on the same
    # row, and the leaderboard says which of the two it is ranking.
    judge_model: Optional[str] = None
    exact_accuracy: Optional[float] = None
    judge_accuracy: Optional[float] = None
    # Answers the judge actually ruled on. Below `total` when some verdicts could
    # not be parsed: an unreadable verdict is left out of the judged rate rather
    # than counted wrong, because charging the model under test for the judge's
    # formatting is the one outcome that would make the number meaningless.
    judge_judged: int = 0
    judge_unreadable: int = 0
    # Items where the judge and the exact grader differed, and how many of those
    # the judge accepted that exact marking rejected.
    disagreements: int = 0
    judge_raised: int = 0


class BenchmarkStatusResponse(BaseModel):
    """Everything the live view reads about the active run."""

    run_id: str = ""
    status: BenchmarkRunStatus = "idle"
    phase: str = "idle"
    message: str = ""
    error: Optional[str] = None

    model_id: str = ""
    model_label: str = ""
    format: ModelFormat = "safetensors"
    lora_path: Optional[str] = None
    load_in_4bit: bool = False

    # The judge, when one was configured. Named on the run so a leaderboard row
    # from six weeks ago says which model produced the generative figures rather
    # than leaving a number with no author.
    judge_model_id: Optional[str] = None
    judge_model_label: Optional[str] = None
    # Which pass is going, including the judge. The live view names it so a run
    # sitting in "Judging" is not read as stalled.
    phase: str = "idle"

    # Requested keys, in the order they will run.
    requested: List[str] = Field(default_factory = list)
    current: Optional[str] = None
    scores: List[BenchmarkScore] = Field(default_factory = list)
    # Mean of the centred scores of the benchmarks that finished. Null until at
    # least one has, so the UI never shows a composite over nothing.
    composite: Optional[float] = None
    scoring_mode: Optional[ScoringMode] = None

    progress_percent: float = 0.0
    elapsed_seconds: float = 0.0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    logs: List[dict] = Field(default_factory = list)


class BenchmarkStartResponse(BaseModel):
    run_id: str
    status: BenchmarkRunStatus = "running"
    # What the run is expected to use, before anything has been scored. The
    # recorded mode comes back on the status and is authoritative; this is only
    # so the UI can label the run while it is still queued.
    scoring_mode: Optional[ScoringMode] = None


class BenchmarkStopResponse(BaseModel):
    status: str
    message: str = ""


class BenchmarkRunSummary(BaseModel):
    """A stored run, as the history list and the leaderboard rows read it."""

    run_id: str
    model_id: str
    model_label: str
    format: ModelFormat = "safetensors"
    lora_path: Optional[str] = None
    status: BenchmarkRunStatus = "completed"
    scoring_mode: Optional[ScoringMode] = None
    scores: List[BenchmarkScore] = Field(default_factory = list)
    composite: Optional[float] = None
    benchmarks_run: List[str] = Field(default_factory = list)
    max_problems: Optional[int] = None
    created_at: float = 0.0
    duration_seconds: float = 0.0


class BenchmarkHistoryResponse(BaseModel):
    runs: List[BenchmarkRunSummary] = Field(default_factory = list)


class LeaderboardRow(BaseModel):
    """One ranked line.

    ``source`` separates the two populations on the board. A local run is this
    machine's own measurement; a reference score is published and carries no
    claim about what this machine would measure. They are ranked together
    because that is the comparison worth making, and labelled because they are
    not the same kind of number.
    """

    run_id: Optional[str] = None
    model_label: str
    model_id: str
    format: Optional[ModelFormat] = None
    source: Literal["local", "reference"] = "local"
    scoring_mode: Optional[ScoringMode] = None
    scores: dict[str, Optional[float]] = Field(default_factory = dict)
    # Which of those scores came from a run that was cut short by the problem cap.
    # They are shown rather than hidden -- a 400-problem sample is a real
    # measurement, and hiding it would leave a user's only run invisible -- but
    # they are marked, and the row carries no composite, so a cheap sample can
    # never outrank a full-set run.
    truncated_keys: List[str] = Field(default_factory = list)
    composite: Optional[float] = None
    benchmarks_count: int = 0
    created_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    max_problems: Optional[int] = None
    # The judge behind any judged figure on this row. A judged score with no
    # author is a number nobody can compare against a new one.
    judge_model: Optional[str] = None
    # Where a published number came from, shown on the row so a reader can tell
    # a measurement of a different model family from an error.
    source_note: Optional[str] = None


class BenchmarkLeaderboardResponse(BaseModel):
    rows: List[LeaderboardRow] = Field(default_factory = list)
    # The benchmark keys the board is currently sorted on, and the composite is
    # only meaningful across rows sharing them, so the UI states both.
    sort_keys: List[str] = Field(default_factory = list)
    include_references: bool = True
    composite: Optional[float] = None
