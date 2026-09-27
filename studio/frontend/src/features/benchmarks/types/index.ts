// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Benchmark shapes, matching `models/benchmarks.py` field for field.
 *
 * The backend speaks snake_case and these are declared that way on purpose, so
 * the API response is assignable without a mapping layer. The one place a
 * conversion exists is the start request, because the form state is camelCase.
 */

/**
 * How a prediction was obtained. Not interchangeable: see the note on `scoring_mode`.
 *
 * `judged` is a generative score produced by a second model rather than by the
 * grader, so it is ranked below the exact modes. Two measurements that agree are
 * worth more than either alone, and a judged row is a weaker claim about the same
 * answer: it depends on a model being right about being right.
 */
export type BenchmarkScoringMode = "logits" | "generated" | "judged" | "mixed";

/** What kind of question it is, which decides how it is answered and graded. */
export type BenchmarkKind = "categorical" | "generative";

export type BenchmarkRunStatus =
  | "idle"
  | "running"
  | "completed"
  | "error"
  | "stopped";

export type BenchmarkScoreStatus =
  | "pending"
  | "running"
  | "complete"
  | "skipped"
  | "failed";

export type BenchmarkModelFormat = "safetensors" | "gguf";

export type BenchmarkSpec = {
  key: string;
  label: string;
  task_name: string;
  notes: string;
  /** Accuracy of guessing. What every aggregate is centred on. */
  baseline: number;
  kind: BenchmarkKind;
  requires_posix_sandbox: boolean;
  max_problems: number | null;
  is_default: boolean;
  /** False when running it here would be wrong rather than slow. */
  supported: boolean;
  unavailable_reason: string | null;
};

export type BenchmarkCatalogue = {
  benchmarks: BenchmarkSpec[];
  default_benchmarks: string[];
  /** False when the datasets cannot be fetched yet. */
  available: boolean;
  reason: string | null;
  posix_sandbox: boolean;
};

export type BenchmarkModelCandidate = {
  id: string;
  label: string;
  path: string;
  source: string;
  format: BenchmarkModelFormat;
  lora: boolean;
  base_model: string | null;
};

export type BenchmarkModelList = {
  models: BenchmarkModelCandidate[];
  empty_reason: string | null;
};

export type BenchmarkScore = {
  key: string;
  label: string;
  kind: BenchmarkKind;
  status: BenchmarkScoreStatus;
  accuracy: number | null;
  baseline: number;
  /** Share of the guess-to-perfect gap closed. The only cross-benchmark figure. */
  centered: number | null;
  correct: number;
  total: number;
  elapsed_seconds: number;
  /** True when the problem cap cut the benchmark short, so this is a sample. */
  truncated: boolean;
  scoring_mode: BenchmarkScoringMode | null;
  error: string | null;

  /**
   * The exact grade, once a judge has also run.
   *
   * `accuracy` is the judge's verdict when there is one, so this is what the
   * benchmark's own grader said about the same answers. The two disagreeing is the
   * interesting result, not an inconsistency to be resolved.
   */
  exact_accuracy: number | null;
  judge_accuracy: number | null;
  judge_judged: number;
  /** Answers the judge could not return a readable verdict for. Not counted wrong. */
  judge_unreadable: number;
  /** Where the judge and the exact grader disagreed, and which one was higher. */
  disagreements: number;
  judge_raised: number;
  judge_model: string | null;
};

export type BenchmarkLogLine = {
  seq: number;
  line: string;
  stream: "stdout" | "stderr";
  phase: string | null;
  ts: number;
};

export type BenchmarkStatus = {
  run_id: string;
  status: BenchmarkRunStatus;
  /** Which pass is running: materialize, score, grade, judge. */
  phase: string;
  message: string;
  error: string | null;

  model_id: string;
  model_label: string;
  format: BenchmarkModelFormat;
  lora_path: string | null;
  load_in_4bit: boolean;

  /** Null unless a judge is configured, which is why these are nullable. */
  judge_model_id: string | null;
  judge_model_label: string | null;

  requested: string[];
  current: string | null;
  scores: BenchmarkScore[];
  composite: number | null;
  scoring_mode: BenchmarkScoringMode | null;

  progress_percent: number;
  elapsed_seconds: number;
  started_at: number | null;
  ended_at: number | null;
  logs: BenchmarkLogLine[];
};

export type BenchmarkStartResponse = {
  run_id: string;
  status: BenchmarkRunStatus;
  scoring_mode: BenchmarkScoringMode | null;
};

export type BenchmarkRunSummary = {
  run_id: string;
  model_id: string;
  model_label: string;
  format: BenchmarkModelFormat;
  lora_path: string | null;
  status: BenchmarkRunStatus;
  scoring_mode: BenchmarkScoringMode | null;
  scores: BenchmarkScore[];
  composite: number | null;
  benchmarks_run: string[];
  max_problems: number | null;
  created_at: number;
  duration_seconds: number;
};

export type BenchmarkHistory = {
  runs: BenchmarkRunSummary[];
};

/** A benchmark the run is configured to evaluate. */
export type BenchmarkKey = {
  key: string;
  label: string;
  kind: BenchmarkKind;
  baseline: number;
  notes: string;
  supported: boolean;
  unavailable_reason: string | null;
  requires_posix_sandbox: boolean;
};

/**
 * One ranked line on the board.
 *
 * `source` separates the two populations on it. A local row is this machine's own
 * measurement; a reference row is a published figure that was produced by somebody
 * else's harness on a machine that is not this one. They are ranked together
 * because the comparison is the point, and separated because they are not the same
 * kind of number.
 */
export type LeaderboardRow = {
  run_id: string | null;
  model_label: string;
  model_id: string;
  format: BenchmarkModelFormat | null;
  source: "local" | "reference";
  scoring_mode: BenchmarkScoringMode | null;
  /** Accuracy per benchmark key. Null where that row has no figure. */
  scores: Record<string, number | null>;
  /** Which of those figures came from a run cut short by the problem cap. */
  truncated_keys: string[];
  /** Null for a truncated run: a sample is an estimate, not a measurement. */
  composite: number | null;
  benchmarks_count: number;
  created_at: number | null;
  duration_seconds: number | null;
  max_problems: number | null;
  source_note: string | null;
};

export type BenchmarkLeaderboard = {
  rows: LeaderboardRow[];
  /** The keys the board is currently projected onto. */
  sort_keys: string[];
  include_references: boolean;
  composite: number | null;
};
