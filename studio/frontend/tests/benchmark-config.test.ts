// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The rules that decide what a benchmark run does, and that decide which numbers
// are allowed to be compared. Both are pure, so both are tested here rather than
// through a rendered page.

import assert from "node:assert/strict";
import test from "node:test";
import {
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
  type BenchmarkFormState,
} from "../src/features/benchmarks/stores/benchmark-config-policy.ts";
import {
  benchmarkLabel,
  formatAccuracy,
  formatCentered,
  formatDuration,
  isExactScoringMode,
  scoringModeLabel,
} from "../src/features/benchmarks/lib/format.ts";

const ORDER = ["mmlu", "arc-easy", "arc-challenge", "gsm8k", "humaneval"];
const ALL = ORDER;

function form(overrides: Partial<BenchmarkFormState> = {}): BenchmarkFormState {
  return {
    ...DEFAULT_BENCHMARK_FORM,
    modelPath: "C:/models/thing",
    benchmarkKeys: ["mmlu"],
    ...overrides,
  };
}

// --- what can be run ---

test("a form with a model and one benchmark can start", () => {
  assert.equal(canStartBenchmarkRun(form(), ALL), true);
});

test("a form with no model cannot start", () => {
  const problems = validateBenchmarkForm(form({ modelPath: null }), ALL);
  assert.deepEqual(
    problems.map((problem) => problem.field),
    ["model"],
  );
});

test("a form with no benchmark cannot start", () => {
  const problems = validateBenchmarkForm(form({ benchmarkKeys: [] }), ALL);
  assert.deepEqual(
    problems.map((problem) => problem.field),
    ["benchmarks"],
  );
});

test("every problem is reported at once, not just the first", () => {
  // A form that can be wrong in three ways should say so three times: reporting
  // one at a time makes the user fix them in three passes.
  const problems = validateBenchmarkForm(
    { ...DEFAULT_BENCHMARK_FORM, maxProblems: 0, batchSize: 0 },
    ALL,
  );
  assert.deepEqual(problems.map((problem) => problem.field).sort(), [
    "batchSize",
    "benchmarks",
    "maxProblems",
    "model",
  ]);
});

test("a benchmark this host cannot run is not counted as a selection", () => {
  // HumanEval is unavailable on a host with no POSIX sandbox, so it is not in the
  // supported set. Selecting only it must not read as a runnable suite.
  const supported = ORDER.filter((key) => key !== "humaneval");
  assert.equal(
    canStartBenchmarkRun(form({ benchmarkKeys: ["humaneval"] }), supported),
    false,
  );
  assert.equal(
    canStartBenchmarkRun(
      form({ benchmarkKeys: ["humaneval", "mmlu"] }),
      supported,
    ),
    true,
  );
});

test("a problem cap outside its range is rejected rather than silently clamped", () => {
  assert.equal(
    canStartBenchmarkRun(form({ maxProblems: MIN_PROBLEMS - 1 }), ALL),
    false,
  );
  assert.equal(
    canStartBenchmarkRun(form({ maxProblems: MAX_PROBLEMS_CEILING + 1 }), ALL),
    false,
  );
  // A fractional cap is rejected too: the backend would floor it, and a form that
  // says 400.5 while the run does 400 is a small lie the user cannot see.
  assert.equal(canStartBenchmarkRun(form({ maxProblems: 400.5 }), ALL), false);
});

test("the default cap is a whole number inside its own range", () => {
  // The backend is told this value, and an out-of-range one 422s the start.
  assert.ok(Number.isInteger(DEFAULT_MAX_PROBLEMS));
  assert.ok(DEFAULT_MAX_PROBLEMS >= MIN_PROBLEMS);
  assert.ok(DEFAULT_MAX_PROBLEMS <= MAX_PROBLEMS_CEILING);
});

test("the cap clamps rather than refusing to move", () => {
  assert.equal(clampMaxProblems(0), MIN_PROBLEMS);
  assert.equal(clampMaxProblems(10_000_000), MAX_PROBLEMS_CEILING);
  assert.equal(clampMaxProblems(400.4), 400);
  assert.equal(clampMaxProblems(Number.NaN), DEFAULT_MAX_PROBLEMS);
});

// --- selection order ---

test("selection follows the catalogue order, not the click order", () => {
  // A suite should read the same way every time it is assembled, so the order is
  // the catalogue's rather than however fast the user clicked.
  let selected = toggleBenchmarkKey([], "gsm8k", ORDER);
  selected = toggleBenchmarkKey(selected, "mmlu", ORDER);
  selected = toggleBenchmarkKey(selected, "arc-easy", ORDER);
  assert.deepEqual(selected, ["mmlu", "arc-easy", "gsm8k"]);
});

test("toggling an already-selected benchmark removes it", () => {
  const selected = toggleBenchmarkKey(["mmlu", "gsm8k"], "mmlu", ORDER);
  assert.deepEqual(selected, ["gsm8k"]);
});

test("a stored selection is filtered through the catalogue on load", () => {
  // A key that has disappeared upstream must not be resent, or the start request
  // 400s on something the user can neither see nor remove.
  assert.deepEqual(resolveSelectedKeys(["gsm8k", "gone", "mmlu"], ORDER), [
    "mmlu",
    "gsm8k",
  ]);
});

// --- how a model is measured ---

test("a GGUF is measured by generation and safetensors from logits", () => {
  // The two are not the same measurement, so which one applies is a property of
  // the model the user picked rather than an implementation detail.
  assert.equal(scoringModeForFormat("gguf"), "generated");
  assert.equal(scoringModeForFormat("safetensors"), "logits");
});

// --- formatting ---

test("an absent score is a dash, never zero", () => {
  // 0% reads as a model that got everything wrong. A benchmark that did not run,
  // was skipped, or produced nothing must not look like that.
  for (const format of [formatAccuracy, formatCentered]) {
    assert.equal(format(null), "—");
    assert.equal(format(undefined), "—");
    assert.equal(format(Number.NaN), "—");
  }
  assert.equal(formatAccuracy(0), "0.0%");
  assert.equal(formatCentered(0), "0%");
});

test("scores are shown as percentages", () => {
  assert.equal(formatAccuracy(0.6789), "67.9%");
  assert.equal(formatAccuracy(1), "100.0%");
  assert.equal(formatCentered(0.5), "50%");
});

test("durations are coarse, because a column of them is being scanned", () => {
  assert.equal(formatDuration(9), "9s");
  assert.equal(formatDuration(69), "1m 09s");
  assert.equal(formatDuration(3600), "1h 00m");
  assert.equal(formatDuration(null), "—");
  assert.equal(formatDuration(-5), "0s");
});

test("a scoring mode is named, and only the exact one is called exact", () => {
  assert.equal(scoringModeLabel("logits"), "exact (logits)");
  assert.equal(isExactScoringMode("logits"), true);
  // The generated path measures the same question less precisely, so calling it
  // exact would overstate it.
  assert.equal(isExactScoringMode("generated"), false);
  assert.equal(isExactScoringMode("mixed"), false);
  assert.equal(scoringModeLabel(null), "unknown");
});

test("a benchmark key falls back to a readable name", () => {
  assert.equal(benchmarkLabel("mmlu"), "MMLU");
  assert.equal(benchmarkLabel("arc-easy"), "ARC-Easy");
  // An unrecognised key is shown as itself rather than as an empty cell, so a
  // benchmark added upstream is still identifiable before this side learns its name.
  assert.equal(benchmarkLabel("brand-new-benchmark"), "brand-new-benchmark");
  assert.equal(benchmarkLabel("mmlu", "From the server"), "From the server");
});
