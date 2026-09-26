// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Shapes returned by `/api/nanochat/*`.
 *
 * snake_case throughout, matching the wire. Converting to camelCase happens once
 * in the runtime store's setters, so the live view reads camelCase and the
 * network layer stays a faithful mirror of the backend.
 */

export type NanochatEnvironmentStatus = {
  checkout_present: boolean;
  checkout_path: string;
  checkout_ref: string;
  venv_present: boolean;
  python_path: string;
  uv_available: boolean;
  torch_installed: boolean;
  torch_version: string | null;
  cuda_available: boolean;
  device_name: string | null;
  blocking_reason: string | null;
  install_state: "absent" | "installing" | "ready" | "failed";
  install_progress: number;
  install_message: string;
  ready: boolean;
};

export type NanochatDataset = {
  key: string;
  label: string;
  repo: string;
  base_url: string;
  data_dir: string;
  max_shard: number | null;
  text_column: string;
  notes: string;
  local_dir: string;
  shards_present: number;
  on_disk: boolean;
  is_default: boolean;
};

export type NanochatTrainingTask = {
  key: string;
  label: string;
  rows: number;
  teaches: string;
  default_epochs: number;
  is_default: boolean;
};

export type NanochatBenchmarkSpec = {
  key: string;
  label: string;
  task_name: string;
  baseline: number;
  categorical: boolean;
  max_problems: number | null;
  stages: string[];
  notes: string;
  is_default: boolean;
  requires_posix_sandbox?: boolean;
};

export type NanochatBaseEvalSpec = {
  label: string;
  flag: string;
  notes: string;
  default: boolean;
};

export type NanochatStageDescriptor = {
  key: string;
  group: "prepare" | "train" | "evaluate";
  title: string;
};

export type NanochatCatalogue = {
  schema_version: number;
  unavailable: boolean;
  reason: string | null;
  datasets: Record<string, NanochatDataset>;
  default_dataset: string | null;
  training_tasks: Record<string, NanochatTrainingTask>;
  default_training_tasks: string[];
  benchmarks: Record<string, NanochatBenchmarkSpec>;
  default_chat_benchmarks: string[];
  base_evals: Record<string, NanochatBaseEvalSpec>;
  default_base_evals: string[];
  trained_on_by_default: string[];
  registry_problems: string[];
  stages: NanochatStageDescriptor[];
};

export type NanochatDepthPreset = {
  key: string;
  depth: number;
  label: string;
  tier: "tiny" | "small" | "medium" | "large" | "xlarge";
  summary: string;
  good_for: string;
  caveat: string | null;
  num_params: number;
  transformer_params: number;
  embedding_params: number;
  n_embd: number;
  n_head: number;
  kv_cache_bytes_per_token: number;
  model_tag: string;
  is_default: boolean;
};

export type NanochatPresetResponse = {
  presets: NanochatDepthPreset[];
  default_depth: number;
  stages: NanochatStageDescriptor[];
  defaults: {
    dataset: string | null;
    sft_tasks: string[];
    chat_benchmarks: string[];
    base_benchmarks: string[];
  };
};

export type NanochatFit = {
  depth: number;
  num_params: number;
  transformer_params: number;
  n_embd: number;
  n_head: number;
  model_memory_bytes: number;
  training_memory_bytes: number;
  tokens: number;
  total_flops: number;
  est_seconds: number | null;
  fits: boolean;
  reasons: string[];
  suggestions: string[];
  device_name: string | null;
  peak_flops_bf16: number | null;
  device_type: string;
  max_batch_size: number | null;
};

export type NanochatValidateResult = {
  ok: boolean;
  fit: NanochatFit | null;
  errors: string[];
  warnings: string[];
  config_errors: string[];
};

export type NanochatStageInfo = {
  key: string;
  status: "pending" | "running" | "ok" | "failed" | "skipped" | "stopped";
  message: string;
  error: string | null;
  started_at: number | null;
  ended_at: number | null;
  duration_seconds: number | null;
};

export type NanochatCheckpointInfo = {
  stage: string | null;
  source: string | null;
  step: number | null;
  path: string | null;
  model_tag: string | null;
  num_params: number | null;
  ts: number;
};

export type NanochatStatus = {
  run_id: string;
  status: "idle" | "running" | "completed" | "error" | "stopped";
  phase: string;
  current_stage: string | null;
  message: string;
  error: string | null;
  started_at: number | null;
  ended_at: number | null;
  duration_seconds: number;
  step: number;
  total_steps: number;
  progress_percent: number;
  loss: number | null;
  val_bpb: number | null;
  chatcore: number | null;
  reward: number | null;
  tok_per_sec: number | null;
  mfu: number | null;
  peak_memory_bytes: number | null;
  eta_seconds: number | null;
  elapsed_seconds: number;
  resolved_config: Record<string, unknown> | null;
  num_params: number | null;
  stages: NanochatStageInfo[];
  checkpoints: NanochatCheckpointInfo[];
  warnings: string[];
};

/**
 * One generation from a running stage.
 *
 * `mode` is what makes the feed renderable: a base model continues text, an SFT
 * model replies, and an RL model produces rollouts with a reward attached. Drawing
 * all three as the same conversation turn would misrepresent two of them.
 */
export type NanochatSample = {
  stage: string;
  step: number;
  mode: "completion" | "chat" | "rollout" | "unconditioned";
  prompt: string;
  completion: string;
  reward?: number | null;
  advantage?: number | null;
  ts: number;
};

export type NanochatBenchmark = {
  stage: string;
  name: string;
  accuracy: number | null;
  baseline: number | null;
  centered: number | null;
  metric: string | null;
  value: number | null;
  is_partial: boolean;
  ts: number;
};

/**
 * One line of a run's console output.
 *
 * `seq` is monotonic per run and is the cursor. The console asks for everything
 * after the last `seq` it drew, so a reconnect resumes instead of replaying the
 * tail it already has — otherwise every dropped connection would re-print lines
 * the user is still looking at.
 */
export type NanochatLogLine = {
  seq: number;
  stream: "stdout" | "stderr";
  stage: string | null;
  line: string;
  ts: number;
};
