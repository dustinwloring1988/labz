// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Benchmarks: measuring a model against common benchmarks, and ranking the
 * results.
 *
 * Separate from the Train tab because it is a different thing. Train changes a
 * model; this measures one and leaves a number behind. The benchmarks and their
 * grading come from nanochat's own task definitions (see the backend's
 * `core/benchmarks/__init__.py` for why the work is split across two
 * interpreters), so a score here is the same measurement nanochat publishes, run
 * against a model of this app's choosing rather than one nanochat trained.
 */

export { BenchmarkPage, type BenchmarkTab } from "./benchmarks-page";

export {
  useBenchmarkRuntimeStore,
  isBenchmarkRunActive,
  isBenchmarkRunFinished,
  hasBenchmarkRun,
  useBenchmarkRunActive,
  useBenchmarkRunFinished,
  type BenchmarkRuntimeState,
} from "./stores/benchmark-runtime-store";

export {
  useBenchmarkConfigStore,
  type BenchmarkConfigStore,
} from "./stores/benchmark-config-store";

export {
  DEFAULT_BENCHMARK_FORM,
  DEFAULT_MAX_PROBLEMS,
  MAX_PROBLEMS_CEILING,
  MIN_PROBLEMS,
  canStartBenchmarkRun,
  clampMaxProblems,
  resolveSelectedKeys,
  scoringModeForFormat,
  toggleBenchmarkKey,
  validateBenchmarkForm,
  type BenchmarkFormProblem,
  type BenchmarkFormState,
} from "./stores/benchmark-config-policy";

export {
  benchmarkLabel,
  formatAccuracy,
  formatCentered,
  formatDuration,
  isExactScoringMode,
  scoringModeLabel,
} from "./lib/format";

export * from "./api/benchmarks-api";
export * from "./types";
