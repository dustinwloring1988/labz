// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The judge, from the form to the board. Three things have to hold and each has
// a way of being quietly wrong: the request has to carry a judge, a judge has to
// stay out of the way when nothing would use it, and a judged score has to reach
// the reader with the exact figure and its coverage attached.

import assert from "node:assert/strict";
import test from "node:test";
import { register } from "node:module";

// The API module reaches for the `@/` alias, which node does not resolve on its
// own. Registered, then imported dynamically: a static import would be hoisted
// above the register() call and would fail to resolve before the hook exists.
register("./helpers/benchmarks-api-resolver.mjs", import.meta.url);

const { buildBenchmarkStartPayload } = await import(
  "../src/features/benchmarks/api/benchmarks-api.ts"
);

import {
  DEFAULT_BENCHMARK_FORM,
  judgeApplies,
  type BenchmarkFormState,
} from "../src/features/benchmarks/stores/benchmark-config-policy.ts";
import {
  formatJudgeCoverage,
  scoringModeLabel,
} from "../src/features/benchmarks/lib/format.ts";
import type {
  BenchmarkModelCandidate,
} from "../src/features/benchmarks/types/index.ts";

const GENERATIVE = ["gsm8k", "humaneval"];
const CATEGORICAL = ["mmlu", "arc-easy"];

function model(
  overrides: Partial<BenchmarkModelCandidate> = {},
): BenchmarkModelCandidate {
  return {
    id: "me/m",
    label: "my model",
    path: "C:/models/m",
    source: "downloaded",
    format: "safetensors",
    lora: false,
    base_model: null,
    ...overrides,
  };
}

function form(overrides: Partial<BenchmarkFormState> = {}): BenchmarkFormState {
  return {
    ...DEFAULT_BENCHMARK_FORM,
    modelPath: "C:/models/m",
    benchmarkKeys: ["gsm8k"],
    ...overrides,
  };
}

function payload(overrides: {
  judgeModel?: BenchmarkModelCandidate | null;
  benchmarkKeys?: string[];
} = {}) {
  return buildBenchmarkStartPayload({
    model: model(),
    benchmarks: overrides.benchmarkKeys ?? ["gsm8k"],
    maxProblems: 400,
    batchSize: 8,
    maxNewTokens: 512,
    maxSeqLength: 2048,
    loadIn4bit: false,
    judgeModel: overrides.judgeModel ?? null,
  });
}

// --- the request carries the judge, or explicitly does not ---

test("a chosen judge is sent with the run", () => {
  const judge = model({
    id: "judge/7b",
    label: "the judge",
    path: "C:/models/judge",
  });
  const body = payload({ judgeModel: judge });
  assert.equal(body.judge_model_id, "judge/7b");
  assert.equal(body.judge_model_label, "the judge");
  assert.equal(body.judge_model_path, "C:/models/judge");
});

test("no judge is sent as three nulls rather than omitted", () => {
  // The backend treats the three as one all-or-nothing setting, so a payload that
  // left them out and one that set them inconsistently would be indistinguishable
  // from each other at the point it matters. Sending nulls says "no judge" out loud.
  const body = payload();
  assert.equal(body.judge_model_id, null);
  assert.equal(body.judge_model_label, null);
  assert.equal(body.judge_model_path, null);
});

test("the model being benchmarked is unaffected by who judges it", () => {
  // The same checkpoint can be its own judge; that is a smoke test, not a
  // mistake, so nothing about the model under test may change.
  const plain = payload();
  const selfJudged = payload({ judgeModel: model() });
  assert.equal(plain.model_id, selfJudged.model_id);
  assert.equal(plain.model_path, selfJudged.model_path);
  assert.equal(selfJudged.judge_model_path, "C:/models/m");
});

// --- a judge is only offered when something would use it ---

test("a judge applies when a generative benchmark is selected", () => {
  assert.equal(judgeApplies(["gsm8k"], GENERATIVE), true);
  assert.equal(judgeApplies(["mmlu", "gsm8k"], GENERATIVE), true);
});

test("a judge does not apply to a suite of only categorical benchmarks", () => {
  // MMLU and ARC are one letter compared against one letter of logits. Asking a
  // second model "did it say A when gold is A" can only add noise, so the picker
  // is disabled rather than the run being refused.
  assert.equal(judgeApplies(["mmlu", "arc-easy"], GENERATIVE), false);
  assert.equal(judgeApplies([], GENERATIVE), false);
  // And the converse: a categorical key never makes a judge useful.
  assert.equal(judgeApplies(CATEGORICAL, GENERATIVE), false);
});

test("a judge left over from another suite does not stop a run", () => {
  // Switching back to a categorical-only suite leaves the judge set. That is not
  // a mistake the user has to clear, so it must not be validation failure.
  const state = form({
    benchmarkKeys: ["mmlu"],
    judgeModelPath: "C:/models/judge",
  });
  assert.equal(judgeApplies(state.benchmarkKeys, GENERATIVE), false);
  assert.ok(state.judgeModelPath, "the stale selection is still there, harmlessly");
});

// --- the score has to arrive readable ---

test("judge coverage names the answers that went unread", () => {
  // The number on its own would read as a clean sweep. "6/8, 2 unreadable" cannot.
  assert.equal(formatJudgeCoverage(8, 0), "judged 8/8");
  assert.equal(formatJudgeCoverage(6, 2), "judged 6/8 · 2 unreadable");
  assert.equal(formatJudgeCoverage(1, 7), "judged 1/8 · 7 unreadable");
});

test("coverage is absent when there is no judge, not zero", () => {
  // An unjudged row must not grow a "judged 0/0" line, which would imply a judge
  // ran and failed rather than that none was configured.
  assert.equal(formatJudgeCoverage(null, null), null);
  assert.equal(formatJudgeCoverage(0, 0), null);
});

test("a judged score is named as judged, and is not called exact", () => {
  assert.equal(scoringModeLabel("judged"), "judged answer");
  // The one word that would overclaim. `isExactScoringMode` gates the board's
  // comparability warning, and a judged row must fall on the wrong side of it.
  assert.notEqual(scoringModeLabel("judged"), "exact (logits)");
});
