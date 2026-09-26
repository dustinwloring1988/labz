// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Shape and defaults for the nanochat run configuration.
 *
 * Kept apart from the store so the defaults, the invariants and the types can be
 * read (and tested) without pulling in zustand or localStorage.
 *
 * Every value here is nanochat's own default or a consumer-friendly adjustment of
 * it. Two are deliberately not nanochat's defaults:
 *
 *   window_pattern "L"  nanochat ships "SSSL", which alternates sliding-window
 *                       attention. That needs Flash Attention 3, and a consumer
 *                       card falls back to SDPA, which has no sliding-window
 *                       support at all and warns that GPU utilisation will be
 *                       terrible. "L" is full attention everywhere and is the
 *                       right default for a desktop.
 *
 *   total_batch_size 32768  nanochat's auto-computed batch is 524288 tokens,
 *                       which needs roughly 137 GB just for the fp32 logits at a
 *                       32768 vocab. On a 16 GB card that is a certain OOM. The
 *                       depth picker re-derives a batch that fits, so this is
 *                       only the starting point.
 */
import { PRESET_DEFAULT_DEPTH } from "../lib/depth-presets.ts";

export type NanochatStageKey =
  | "dataset"
  | "tokenizer"
  | "pretrain"
  | "base_eval"
  | "sft"
  | "chat_eval"
  | "rl";

export type NanochatParamMode = "simple" | "advanced";

export type NanochatConfigState = {
  // --- model ---
  depth: number;
  aspectRatio: number;
  headDim: number;
  maxSeqLen: number;
  windowPattern: string;
  modelTag: string | null;

  // --- data ---
  dataset: string;
  numShards: number;
  vocabSize: number;
  trainTokenizer: boolean;

  // --- horizon ---
  numIterations: number;
  paramDataRatio: number;
  totalBatchSize: number;
  deviceBatchSize: number;
  fp8: boolean;

  // --- learning rates ---
  embeddingLr: number;
  unembeddingLr: number;
  matrixLr: number;
  scalarLr: number;
  weightDecay: number;
  warmupSteps: number;
  warmdownRatio: number;
  finalLrFrac: number;

  // --- cadence ---
  evalEvery: number;
  evalTokens: number;
  coreMetricEvery: number;
  sampleEvery: number;
  saveEvery: number;
  coreMetricMaxPerTask: number;

  // --- SFT ---
  sftTasks: string[];
  mmluEpochs: number;
  gsm8kEpochs: number;
  sftIterations: number;
  sftSampleEvery: number;

  // --- RL ---
  runRl: boolean;
  rlEpochs: number;
  rlExamplesPerStep: number;
  rlNumSamples: number;
  rlMaxNewTokens: number;
  rlEvalEvery: number;
  rlSaveEvery: number;

  // --- benchmarks ---
  baseBenchmarks: string[];
  chatBenchmarks: string[];

  // --- stages ---
  stages: NanochatStageKey[];

  // --- ui ---
  paramMode: NanochatParamMode;
};

export const DEFAULT_NANOCHAT_CONFIG: NanochatConfigState = {
  depth: PRESET_DEFAULT_DEPTH,
  aspectRatio: 64,
  headDim: 128,
  maxSeqLen: 2048,
  windowPattern: "L",
  modelTag: null,

  dataset: "climbmix-400b",
  numShards: 32,
  vocabSize: 32768,
  trainTokenizer: true,

  numIterations: 1000,
  paramDataRatio: 12,
  totalBatchSize: 32768,
  deviceBatchSize: 4,
  fp8: false,

  embeddingLr: 0.3,
  unembeddingLr: 0.008,
  matrixLr: 0.02,
  scalarLr: 0.5,
  weightDecay: 0.28,
  warmupSteps: 40,
  warmdownRatio: 0.65,
  finalLrFrac: 0.05,

  evalEvery: 100,
  evalTokens: 2 * 1024 * 1024,
  coreMetricEvery: 500,
  sampleEvery: 200,
  saveEvery: -1,
  coreMetricMaxPerTask: 500,

  sftTasks: ["smoltalk", "mmlu", "gsm8k"],
  mmluEpochs: 3,
  gsm8kEpochs: 4,
  sftIterations: 400,
  sftSampleEvery: 100,

  runRl: false,
  rlEpochs: 1,
  rlExamplesPerStep: 16,
  rlNumSamples: 16,
  rlMaxNewTokens: 256,
  rlEvalEvery: 60,
  rlSaveEvery: 60,

  baseBenchmarks: ["core", "bpb", "sample"],
  chatBenchmarks: ["arc-easy", "arc-challenge", "mmlu", "gsm8k", "humaneval"],

  stages: ["dataset", "tokenizer", "pretrain", "base_eval", "sft", "chat_eval", "rl"],

  paramMode: "simple",
};

/** Context lengths offered in the picker. */
export const NANOCHAT_CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192] as const;

/** Batch sizes offered, in tokens. The lower end is what a 16 GB card needs. */
export const NANOCHAT_BATCH_SIZES = [
  4096, 8192, 16384, 32768, 65536, 131072, 524288,
] as const;

/**
 * A batch size has to be a multiple of deviceBatchSize * maxSeqLen, because
 * nanochat asserts it and dies at the first forward pass otherwise.
 *
 * Snapping here means a user cannot type an invalid value into the field and
 * only find out when the run dies.
 */
export function validTotalBatchSizes(
  deviceBatchSize: number,
  maxSeqLen: number,
): number[] {
  const micro = deviceBatchSize * maxSeqLen;
  return NANOCHAT_BATCH_SIZES.filter((size) => size % micro === 0);
}

export function isValidTotalBatchSize(
  totalBatchSize: number,
  deviceBatchSize: number,
  maxSeqLen: number,
): boolean {
  const micro = deviceBatchSize * maxSeqLen;
  return micro > 0 && totalBatchSize % micro === 0;
}

/** The micro-batch a total batch size decomposes into, for the preview card. */
export function gradientAccumulationSteps(
  totalBatchSize: number,
  deviceBatchSize: number,
  maxSeqLen: number,
): number {
  const micro = deviceBatchSize * maxSeqLen;
  return micro > 0 ? Math.floor(totalBatchSize / micro) : 0;
}
