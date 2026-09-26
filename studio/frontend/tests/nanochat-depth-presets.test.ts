// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { FALLBACK_PRESETS, formatBytes, formatDuration, formatParams } from "../src/features/nanochat/lib/depth-presets.ts";

// The backend owns these numbers; this file is the frontend's copy so the picker
// renders something on first paint instead of an empty state. The vectors below
// are the ones tests/test_nanochat_presets.py asserts against
// nanochat/shapes.py, so a change on either side shows up as a failure here rather
// than as a stale number in the UI.

interface Vector {
  depth: number;
  n_embd: number;
  n_head: number;
  num_params: number;
  transformer: number;
  embedding: number;
  kv_per_token: number;
}

// All at the preset configuration: vocab 32768, aspect ratio 64, head dim 128.
// The figures come from nanochat/shapes.py, which tests/test_shapes.py asserts
// against the real GPT model, and which the backend mirrors in
// core/nanochat/presets.py. Anything that drifts fails on one of those sides too.
const VECTORS: Vector[] = [
  { depth: 4, n_embd: 256, n_head: 2, num_params: 36700242, transformer: 3145776, embedding: 33554432, kv_per_token: 4096 },
  { depth: 6, n_embd: 384, n_head: 3, num_params: 73531538, transformer: 10616940, embedding: 62914560, kv_per_token: 9216 },
  { depth: 8, n_embd: 512, n_head: 4, num_params: 125829354, transformer: 25166016, embedding: 100663296, kv_per_token: 16384 },
  { depth: 10, n_embd: 640, n_head: 5, num_params: 195952986, transformer: 49152300, embedding: 146800640, kv_per_token: 25600 },
  { depth: 12, n_embd: 768, n_head: 6, num_params: 286261730, transformer: 84935088, embedding: 201326592, kv_per_token: 36864 },
  { depth: 16, n_embd: 1024, n_head: 8, num_params: 536871738, transformer: 201327360, embedding: 335544320, kv_per_token: 65536 },
  { depth: 20, n_embd: 1280, n_head: 10, num_params: 896533746, transformer: 393217200, embedding: 503316480, kv_per_token: 102400 },
  { depth: 24, n_embd: 1536, n_head: 12, num_params: 1384122122, transformer: 679478976, embedding: 704643072, kv_per_token: 147456 },
  { depth: 32, n_embd: 2048, n_head: 16, num_params: 2818575450, transformer: 1610615808, embedding: 1207959552, kv_per_token: 262144 },
];

function presetFor(depth: number) {
  const found = FALLBACK_PRESETS.find((preset) => preset.depth === depth);
  assert.ok(found, `no fallback preset for depth ${depth}`);
  return found;
}

test("the fallback shapes match the verified vectors", () => {
  for (const vector of VECTORS) {
    const preset = presetFor(vector.depth);
    assert.equal(preset.nEmbd, vector.n_embd, `d${vector.depth} n_embd`);
    assert.equal(preset.nHead, vector.n_head, `d${vector.depth} n_head`);
  }
});

test("the fallback parameter counts match the verified vectors", () => {
  for (const vector of VECTORS) {
    const preset = presetFor(vector.depth);
    assert.equal(preset.numParams, vector.num_params, `d${vector.depth} total`);
    assert.equal(preset.transformerParams, vector.transformer, `d${vector.depth} transformer`);
    assert.equal(preset.embeddingParams, vector.embedding, `d${vector.depth} embeddings`);
  }
});

test("the parts add up to the total, less the scalar lambdas", () => {
  for (const vector of VECTORS) {
    const preset = presetFor(vector.depth);
    // gpt.py counts two scalars per layer (resid and x0 lambdas), a 24-wide smear
    // gate, and two more. Anything else unaccounted for means a bucket is wrong.
    const scalars = 2 * vector.depth + 24 + 2;
    assert.equal(
      preset.transformerParams + preset.embeddingParams + scalars,
      preset.numParams,
      `d${vector.depth} parts do not sum to the total`,
    );
  }
});
test("a deeper model is never a smaller one", () => {
  const totals = FALLBACK_PRESETS.map((preset) => preset.numParams);
  assert.deepEqual(totals, [...totals].sort((a, b) => a - b));
  assert.equal(new Set(totals).size, totals.length);
});

test("the offered depths are the ones the backend serves", () => {
  assert.deepEqual(
    FALLBACK_PRESETS.map((preset) => preset.depth),
    [4, 6, 8, 10, 12, 16, 20, 24, 32],
  );
});

test("every preset is identified the way the backend identifies it", () => {
  // The picker keys on `key`, and the backend sends `d<depth>`. A mismatch would
  // make the React list key collide or the aria state wrong.
  for (const preset of FALLBACK_PRESETS) {
    assert.equal(preset.key, `d${preset.depth}`);
    assert.equal(preset.label, `d${preset.depth}`);
  }
});

test("every preset says what it is for and carries a caveat when it needs one", () => {
  for (const preset of FALLBACK_PRESETS) {
    assert.ok(preset.goodFor.length > 0, `d${preset.depth} has no guidance`);
    assert.ok(preset.summary.length > 0, `d${preset.depth} has no summary`);
    assert.ok(["tiny", "small", "medium", "large", "xlarge"].includes(preset.tier));
  }
  // The two that are not practical on a desktop have to say so.
  assert.ok(presetFor(24).caveat, "d24 needs a caveat about the multi-day cost");
  assert.ok(presetFor(32).caveat, "d32 needs a caveat about being impractical here");
});

test("the embedding-dominance claim holds for every consumer-realistic depth", () => {
  // At a 32768 vocab the embedding tables outnumber the transformer matrices until
  // about depth 25, so quoting the total alone would overstate how much reasoning
  // capacity a shallow model has. d4 is 37M parameters but 3.1M transformer.
  for (const preset of FALLBACK_PRESETS.filter((p) => p.depth <= 24)) {
    assert.ok(
      preset.embeddingParams > preset.transformerParams,
      `d${preset.depth} is expected to be embedding-dominated`,
    );
  }
  // Past the crossover it flips, which is why the picker shows both numbers.
  assert.ok(presetFor(32).transformerParams > presetFor(32).embeddingParams);
});

test("every embedding-dominated preset qualifies its parameter count", () => {
  for (const preset of FALLBACK_PRESETS.filter((p) => p.embeddingParams > p.transformerParams)) {
    assert.match(
      preset.summary,
      /transformer/,
      `d${preset.depth} is embedding-dominated, so its summary must say so: ${preset.summary}`,
    );
  }
});

test("the summary quotes the real parameter count", () => {
  for (const preset of FALLBACK_PRESETS) {
    const millions = preset.numParams / 1e6;
    // d32 is written "2.8B" rather than "2.82B": a three-sigma figure on a model
    // nobody can train here would be false precision, and the backend's own
    // summary test skips it for the same reason.
    const expected =
      preset.depth >= 32
        ? `${(millions / 1000).toFixed(1)}B`
        : millions < 1000
          ? `${Math.round(millions)}M`
          : `${(millions / 1000).toFixed(2)}B`;
    assert.ok(
      preset.summary.includes(expected),
      `d${preset.depth} is ${preset.numParams.toLocaleString()} params but its summary says ${preset.summary}`,
    );
  }
});

test("the KV cache figure is the two-tensors-per-layer one", () => {
  for (const vector of VECTORS) {
    const preset = presetFor(vector.depth);
    assert.equal(
      preset.kvCacheBytesPerToken,
      vector.depth * 2 * vector.n_head * 128 * 2,
      `d${vector.depth} kv cache per token`,
    );
  }
});

// --- formatting ---

test("parameter counts are readable at every scale", () => {
  assert.equal(formatParams(589866), "590K");
  assert.equal(formatParams(125829354), "126M");
  assert.equal(formatParams(1384122122), "1.38B");
  assert.equal(formatParams(2818575450), "2.82B");
  assert.equal(formatParams(512), "512");
});

test("byte counts are readable at every scale", () => {
  assert.equal(formatBytes(344635904), "345 MB");
  assert.equal(formatBytes(13_800_000_000), "13.8 GB");
  assert.equal(formatBytes(2_400_000_000_000), "2.40 TB");
  // Missing is a dash, not "0 B": a probe that could not run is not a zero.
  assert.equal(formatBytes(null), "—");
  assert.equal(formatBytes(undefined), "—");
  assert.equal(formatBytes(Number.NaN), "—");
});

test("durations read as a wait, and an unknown one is a dash", () => {
  assert.equal(formatDuration(45), "45s");
  assert.equal(formatDuration(600), "10m");
  assert.equal(formatDuration(7200), "2.0h");
  assert.equal(formatDuration(180000), "2.1d");
  assert.equal(formatDuration(null), "—");
  assert.equal(formatDuration(-1), "—");
});
