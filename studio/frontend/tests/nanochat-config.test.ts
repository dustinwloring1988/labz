// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_NANOCHAT_CONFIG,
  gradientAccumulationSteps,
  isValidTotalBatchSize,
  validTotalBatchSizes,
  type NanochatConfigState,
} from "../src/features/nanochat/stores/nanochat-config-policy.ts";
import { migrateNanochatConfig } from "../src/features/nanochat/stores/nanochat-config-persistence.ts";

// --- batch divisibility ---
//
// nanochat asserts total_batch_size % (device_batch_size * max_seq_len) == 0 and
// dies at the first forward pass otherwise. Catching it in the field is the whole
// point of these, because a run that dies at step 1 has already downloaded the
// corpus and built the model.

test("a batch that divides evenly is accepted", () => {
  // 32768 / (4 * 2048) = 4.
  assert.equal(isValidTotalBatchSize(32768, 4, 2048), true);
  assert.equal(isValidTotalBatchSize(524288, 8, 2048), true);
});

test("a batch that does not divide evenly is rejected", () => {
  // 33000 is not a multiple of 8192.
  assert.equal(isValidTotalBatchSize(33000, 4, 2048), false);
  // A micro-batch larger than the total batch is nonsense, not merely uneven.
  assert.equal(isValidTotalBatchSize(1024, 4, 2048), false);
});

test("a zero micro-batch is not a valid configuration", () => {
  // Guards a divide-by-zero in the field, where an empty input parses as 0.
  assert.equal(isValidTotalBatchSize(4096, 0, 2048), false);
  assert.equal(isValidTotalBatchSize(4096, 4, 0), false);
});

test("the offered batch sizes are filtered to the ones that divide", () => {
  // maxSeqLen 2048 with a device batch of 1 means a step of 2048 tokens.
  const offered = validTotalBatchSizes(1, 2048);
  assert.ok(offered.includes(32768));
  for (const size of offered) {
    assert.equal(size % 2048, 0, `${size} does not divide`);
  }
  // A device batch that does not divide the smallest offered size leaves nothing
  // valid, which the UI has to survive rather than render an empty list silently.
  assert.deepEqual(validTotalBatchSizes(3, 1000), []);
});

test("gradient accumulation is reported, not silently truncated", () => {
  assert.equal(gradientAccumulationSteps(32768, 4, 2048), 4);
  assert.equal(gradientAccumulationSteps(524288, 8, 2048), 32);
  // An uneven batch still reports something rather than Infinity or NaN.
  assert.equal(Number.isFinite(gradientAccumulationSteps(33000, 4, 2048)), true);
  assert.equal(gradientAccumulationSteps(4096, 0, 2048), 0);
});

// --- persistence migration ---

test("an absent payload keeps the defaults", () => {
  assert.deepEqual(
    migrateNanochatConfig(undefined, DEFAULT_NANOCHAT_CONFIG),
    DEFAULT_NANOCHAT_CONFIG,
  );
  assert.deepEqual(
    migrateNanochatConfig(null as never, DEFAULT_NANOCHAT_CONFIG),
    DEFAULT_NANOCHAT_CONFIG,
  );
});

test("a stored configuration survives a reload", () => {
  const stored: Partial<NanochatConfigState> = { depth: 20, numIterations: 4321, fp8: true };
  const migrated = migrateNanochatConfig(stored, DEFAULT_NANOCHAT_CONFIG);
  assert.equal(migrated.depth, 20);
  assert.equal(migrated.numIterations, 4321);
  assert.equal(migrated.fp8, true);
  // Everything the user did not set comes from the defaults.
  assert.equal(migrated.aspectRatio, DEFAULT_NANOCHAT_CONFIG.aspectRatio);
});

test("an unknown key is dropped rather than carried", () => {
  // A payload from a newer build must not leak fields this one does not know.
  const migrated = migrateNanochatConfig(
    { depth: 16, somethingFromTheFuture: true } as never,
    DEFAULT_NANOCHAT_CONFIG,
  );
  assert.equal("somethingFromTheFuture" in migrated, false);
  assert.equal(migrated.depth, 16);
});

test("a stored value outside the current bounds is clamped, not discarded", () => {
  // Dropping the whole payload would lose every other setting the user chose.
  const migrated = migrateNanochatConfig(
    { depth: 9999, numIterations: -5, maxSeqLen: 10 ** 9 } as never,
    DEFAULT_NANOCHAT_CONFIG,
  );
  assert.equal(migrated.depth, 64);
  assert.equal(migrated.numIterations, 1);
  assert.equal(migrated.maxSeqLen, 32768);
});

test("a stored batch that no longer divides is replaced", () => {
  // The one validity rule that would otherwise survive a reload and kill the run
  // at its first forward pass.
  const migrated = migrateNanochatConfig(
    { deviceBatchSize: 3, maxSeqLen: 1000, totalBatchSize: 32768 } as never,
    DEFAULT_NANOCHAT_CONFIG,
  );
  assert.equal(
    isValidTotalBatchSize(migrated.totalBatchSize, migrated.deviceBatchSize, migrated.maxSeqLen),
    true,
  );
});

test("an empty selection list is replaced with the default", () => {
  // An empty list reads as "nothing selected", which produces a run that trains
  // on no data. Better to fall back than to start one.
  const migrated = migrateNanochatConfig(
    { sftTasks: [], chatBenchmarks: [], stages: [] } as never,
    DEFAULT_NANOCHAT_CONFIG,
  );
  assert.deepEqual(migrated.sftTasks, DEFAULT_NANOCHAT_CONFIG.sftTasks);
  assert.deepEqual(migrated.chatBenchmarks, DEFAULT_NANOCHAT_CONFIG.chatBenchmarks);
  assert.deepEqual(migrated.stages, DEFAULT_NANOCHAT_CONFIG.stages);
});

test("a non-array selection list is replaced with the default", () => {
  const migrated = migrateNanochatConfig(
    { sftTasks: "gsm8k" } as never,
    DEFAULT_NANOCHAT_CONFIG,
  );
  assert.deepEqual(migrated.sftTasks, DEFAULT_NANOCHAT_CONFIG.sftTasks);
});

test("an unknown param mode falls back rather than persisting garbage", () => {
  const migrated = migrateNanochatConfig(
    { paramMode: "wizard" } as never,
    DEFAULT_NANOCHAT_CONFIG,
  );
  assert.equal(migrated.paramMode, "simple");
  const kept = migrateNanochatConfig(
    { paramMode: "advanced" } as never,
    DEFAULT_NANOCHAT_CONFIG,
  );
  assert.equal(kept.paramMode, "advanced");
});

test("a model tag that is not a string becomes null", () => {
  const migrated = migrateNanochatConfig({ modelTag: 42 } as never, DEFAULT_NANOCHAT_CONFIG);
  assert.equal(migrated.modelTag, null);
});

// --- defaults ---

test("the defaults describe a run that can actually be built", () => {
  const config = DEFAULT_NANOCHAT_CONFIG;
  assert.equal(
    isValidTotalBatchSize(config.totalBatchSize, config.deviceBatchSize, config.maxSeqLen),
    true,
    "the default configuration would fail nanochat's own batch assertion",
  );
});

test("the default window pattern is full attention", () => {
  // nanochat ships "SSSL", which needs Flash Attention 3. A consumer card falls
  // back to SDPA, which has no sliding-window support at all and warns that GPU
  // utilisation will be terrible, so the default is deliberately not nanochat's.
  assert.equal(DEFAULT_NANOCHAT_CONFIG.windowPattern, "L");
});

test("the default batch is far below nanochat's, and that is deliberate", () => {
  // nanochat auto-computes 524288 tokens, which needs roughly 137 GB for the fp32
  // logits alone at a 32768 vocab. A desktop default has to fit a 16 GB card.
  assert.ok(DEFAULT_NANOCHAT_CONFIG.totalBatchSize < 524288);
  assert.ok(DEFAULT_NANOCHAT_CONFIG.totalBatchSize >= 16384);
});

test("the default stages run the whole pipeline", () => {
  assert.deepEqual(DEFAULT_NANOCHAT_CONFIG.stages, [
    "dataset",
    "tokenizer",
    "pretrain",
    "base_eval",
    "sft",
    "chat_eval",
    "rl",
  ]);
  // RL is opt-in: it is the slowest stage and the one most likely to be unwanted.
  assert.equal(DEFAULT_NANOCHAT_CONFIG.runRl, false);
});

test("the default benchmark selection matches nanochat's own", () => {
  assert.deepEqual(DEFAULT_NANOCHAT_CONFIG.baseBenchmarks, ["core", "bpb", "sample"]);
  assert.deepEqual(DEFAULT_NANOCHAT_CONFIG.chatBenchmarks, [
    "arc-easy",
    "arc-challenge",
    "mmlu",
    "gsm8k",
    "humaneval",
  ]);
});
