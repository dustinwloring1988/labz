// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Benchmark test fixtures.
 *
 * A shared factory rather than object literals in each test, because
 * `BenchmarkStatus` and `BenchmarkScore` have grown fields twice already and every
 * literal is a compile error waiting to happen the next time they do. Defaults here
 * describe an unremarkable run; a test overrides only the field it is about.
 */

import type {
  BenchmarkScore,
  BenchmarkScoringMode,
  BenchmarkSpec,
  BenchmarkStatus,
} from "../../src/features/benchmarks/types/index.ts";

export function makeScore(overrides: Partial<BenchmarkScore> = {}): BenchmarkScore {
  return {
    key: "mmlu",
    label: "MMLU",
    kind: "categorical",
    status: "complete",
    accuracy: 0.5,
    baseline: 0.25,
    centered: 0.333,
    correct: 5,
    total: 10,
    elapsed_seconds: 12,
    truncated: false,
    scoring_mode: "logits",
    error: null,
    // No judge has run, which is the state every unjudged row is in.
    exact_accuracy: null,
    judge_accuracy: null,
    judge_judged: 0,
    judge_unreadable: 0,
    disagreements: 0,
    judge_raised: 0,
    judge_model: null,
    ...overrides,
  };
}

export function makeStatus(overrides: Partial<BenchmarkStatus> = {}): BenchmarkStatus {
  return {
    run_id: "r1",
    status: "running",
    phase: "score",
    message: "Scoring",
    error: null,
    model_id: "me/m",
    model_label: "my model",
    format: "safetensors",
    lora_path: null,
    load_in_4bit: false,
    judge_model_id: null,
    judge_model_label: null,
    requested: ["mmlu"],
    current: "mmlu",
    scores: [],
    composite: null,
    scoring_mode: "logits",
    progress_percent: 10,
    elapsed_seconds: 1,
    started_at: 1,
    ended_at: null,
    logs: [],
    ...overrides,
  };
}

export function makeSpec(overrides: Partial<BenchmarkSpec> = {}): BenchmarkSpec {
  return {
    key: "mmlu",
    label: "MMLU",
    task_name: "mmlu",
    notes: "",
    baseline: 0.25,
    kind: "categorical",
    requires_posix_sandbox: false,
    max_problems: null,
    is_default: true,
    supported: true,
    unavailable_reason: null,
    ...overrides,
  };
}

export const MODE_ORDER: readonly BenchmarkScoringMode[] = [
  "logits",
  "generated",
  "judged",
  "mixed",
];
