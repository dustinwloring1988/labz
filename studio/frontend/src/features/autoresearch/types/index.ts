// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Shapes returned by `/api/autoresearch/*`.
 *
 * snake_case throughout, matching the wire. Converting to camelCase happens once
 * in the runtime store's setters, so the live view reads camelCase and the
 * network layer stays a faithful mirror of the backend.
 */

export type AutoresearchEnvironmentStatus = {
  source_present: boolean;
  source_path: string;
  checkout_present: boolean;
  checkout_path: string;
  venv_present: boolean;
  python_path: string;
  uv_available: boolean;
  torch_installed: boolean;
  torch_version: string | null;
  cuda_available: boolean;
  device_name: string | null;
  data_prepared: boolean;
  git_ready: boolean;
  blocking_reason: string | null;
  install_state: "absent" | "installing" | "ready" | "failed";
  install_progress: number;
  install_message: string;
  ready: boolean;
};

/**
 * One agent CLI and whether this machine can actually launch it.
 *
 * `reason` is reported rather than filtered, because "not on PATH" and "on PATH
 * but will not run" are different problems and only one is fixed by installing
 * something.
 */
export type AutoresearchAgentInfo = {
  key: string;
  label: string;
  path: string | null;
  version: string | null;
  available: boolean;
  reason: string | null;
  notes: string;
};

export type AutoresearchAgents = {
  agents: AutoresearchAgentInfo[];
  recommended: string | null;
  /** The experiment repository, because its state changes what a run will do. */
  git_branch: string;
  git_head: string;
  git_dirty: boolean;
  git_commits: number;
  git_error: string | null;
};

export type AutoresearchExperimentInfo = {
  index: number;
  status:
    | "pending"
    | "running"
    | "kept"
    | "discarded"
    | "crashed"
    | "stopped";
  message: string;
  error: string | null;
  val_bpb: number | null;
  memory_gb: number | null;
  commit: string;
  description: string;
  summary: string;
  started_at: number | null;
  ended_at: number | null;
  duration_seconds: number | null;
  checkpoint: string | null;
};

export type AutoresearchStatus = {
  run_id: string;
  status: "idle" | "running" | "completed" | "error" | "stopped";
  phase: "idle" | "preparing" | "experiment" | "finished";
  message: string;
  error: string | null;
  started_at: number | null;
  ended_at: number | null;
  duration_seconds: number;

  experiment: number;
  total_experiments: number;
  progress_percent: number;

  /** Read out of run.log, which is the only place a live number exists. */
  step: number;
  loss: number | null;
  tok_per_sec: number | null;
  mfu: number | null;
  peak_memory_gb: number | null;
  elapsed_seconds: number;
  eta_seconds: number | null;

  best_val_bpb: number | null;
  baseline_val_bpb: number | null;
  kept: number;
  discarded: number;
  crashed: number;

  experiments: AutoresearchExperimentInfo[];
  warnings: string[];

  /** The report the loop wrote as its last phase, if it was asked to. */
  report_text: string;
  report_model: string;
  report_saved_to: string | null;
  report_error: string | null;
  report_running: boolean;

  resolved_config: Record<string, unknown>;
  git_branch: string;
  git_head: string;
  stop_requested: boolean;

  /** What a run of this size costs, so it can be said before starting one. */
  estimated_seconds: number;
};

export type AutoresearchStartResponse = {
  run_id: string;
  status: string;
  message: string;
  estimated_seconds: number;
};

/**
 * One line of a run's console output.
 *
 * `seq` is monotonic per run and is the cursor. The console asks for everything
 * after the last `seq` it drew, so a reconnect resumes instead of replaying the
 * tail it already has — otherwise every dropped connection would re-print lines
 * the user is still looking at.
 */
export type AutoresearchLogLine = {
  seq: number;
  stream: "stdout" | "stderr";
  line: string;
  ts: number;
};

export type AutoresearchResultRow = {
  index: number;
  commit: string;
  val_bpb: number | null;
  memory_gb: number | null;
  status: "keep" | "discard" | "crash";
  description: string;
  error: string | null;
  is_best: boolean;
  checkpoint: string | null;
};

export type AutoresearchResults = {
  rows: AutoresearchResultRow[];
  exists: boolean;
  total: number;
  kept: number;
  discarded: number;
  crashed: number;
  best_val_bpb: number | null;
  best_index: number | null;
  best_commit: string | null;
  baseline_val_bpb: number | null;
};

export type AutoresearchMetricPoint = {
  index: number;
  val_bpb: number | null;
  memory_gb: number | null;
  is_best: boolean;
  status: string;
  /** So the chart's tooltip can say what the experiment actually changed. */
  description: string;
};

export type AutoresearchMetrics = {
  /** One point per experiment. A step function, not a line: the score only
   *  changes when an experiment is recorded, and a line would imply values
   *  between two experiments that never existed. */
  series: AutoresearchMetricPoint[];
  baseline_val_bpb: number | null;
  best_val_bpb: number | null;
};

export type AutoresearchCheckpointInfo = {
  model_id: string;
  name: string;
  backend: string;
  experiment: number;
  commit: string;
  bytes: number;
  modified: number;
  val_bpb: number | null;
  num_params_m: number | null;
  depth: number | null;
  loadable: boolean;
  reason: string | null;
  /** Always "base". autoresearch has no chat stage, so these models continue
   *  text rather than reply, and the UI labels them rather than letting the
   *  user discover it. */
  kind: "base";
  hint: string;
};

export type AutoresearchChatModels = {
  models: AutoresearchCheckpointInfo[];
  resident: { backend: string; model: string; experiment: number } | null;
};

export type AutoresearchReportFile = {
  run_id: string;
  path: string;
  bytes: number;
  modified: number;
};

/**
 * One model the report can be written by, and where it lives.
 *
 * Three sources, because a model can live in three places:
 *  - an agent CLI the user has configured, which is where a local model usually is
 *  - the studio's own catalog
 *  - one of this search's own checkpoints
 */
export type AutoresearchReportModel = {
  id: string;
  label: string;
  source: "catalog" | "autoresearch" | "agent";
  /** A note the picker shows, because the three are not interchangeable. */
  note: string;
  /** For an agent source, which CLI it is. */
  agentKey?: string;
};

export type ProgressEvent = {
  run_id: string;
  status: AutoresearchStatus["status"];
  phase: AutoresearchStatus["phase"];
  message: string;
  error: string | null;
  started_at: number | null;
  ended_at: number | null;
  duration_seconds: number;

  experiment: number;
  total_experiments: number;
  progress_percent: number;

  step: number;
  loss: number | null;
  tok_per_sec: number | null;
  mfu: number | null;
  peak_memory_gb: number | null;
  elapsed_seconds: number;
  eta_seconds: number | null;

  best_val_bpb: number | null;
  baseline_val_bpb: number | null;
  kept: number;
  discarded: number;
  crashed: number;

  experiments: AutoresearchExperimentInfo[];
  warnings: string[];

  resolved_config: Record<string, unknown>;
  git_branch: string;
  git_head: string;
  stop_requested: boolean;
  estimated_seconds: number;

  report_text: string;
  report_model: string;
  report_saved_to: string | null;
  report_error: string | null;
  report_running: boolean;

  /** Console lines carried on this frame, if any. */
  logs?: AutoresearchLogLine[];
};
