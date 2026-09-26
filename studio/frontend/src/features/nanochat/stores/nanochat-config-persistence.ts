// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DEFAULT_NANOCHAT_CONFIG, type NanochatConfigState } from "./nanochat-config-policy.ts";

export const NANOCHAT_CONFIG_PERSISTENCE_VERSION = 1;

/**
 * Merge a persisted configuration over the current defaults.
 *
 * Two rules, both about not silently changing what a run will do:
 *
 *  - An unknown key is dropped and a missing key takes the default. A
 *    downgrading user must not end up with `undefined` in a field the backend
 *    requires.
 *  - A value that is now invalid is replaced by the default rather than kept.
 *    A batch size that no longer divides evenly would otherwise survive a
 *    reload and kill the run at the first forward pass.
 */
export function migrateNanochatConfig(
  persisted: Partial<NanochatConfigState> | undefined,
  current: NanochatConfigState,
): NanochatConfigState {
  if (!persisted || typeof persisted !== "object") return current;

  const merged = { ...current } as Record<string, unknown>;
  for (const [key, value] of Object.entries(persisted)) {
    if (key in current) merged[key] = value;
  }

  const result = merged as unknown as NanochatConfigState;

  // Clamp rather than reject: a stored value outside the current bounds is
  // almost certainly from an older build with wider limits, and dropping the
  // whole configuration would lose every other setting the user chose.
  result.depth = clampInt(result.depth, 1, 64, current.depth);
  result.maxSeqLen = clampInt(result.maxSeqLen, 128, 32768, current.maxSeqLen);
  result.numIterations = clampInt(result.numIterations, 1, 10_000_000, current.numIterations);
  result.deviceBatchSize = clampInt(result.deviceBatchSize, 1, 1024, current.deviceBatchSize);
  result.numShards = clampInt(result.numShards, 1, 7000, current.numShards);
  result.vocabSize = clampInt(result.vocabSize, 256, 131072, current.vocabSize);

  if (!Array.isArray(result.sftTasks) || result.sftTasks.length === 0) {
    result.sftTasks = current.sftTasks;
  }
  if (!Array.isArray(result.chatBenchmarks) || result.chatBenchmarks.length === 0) {
    result.chatBenchmarks = current.chatBenchmarks;
  }
  if (!Array.isArray(result.baseBenchmarks) || result.baseBenchmarks.length === 0) {
    result.baseBenchmarks = current.baseBenchmarks;
  }
  if (!Array.isArray(result.stages) || result.stages.length === 0) {
    result.stages = current.stages;
  }
  if (result.paramMode !== "simple" && result.paramMode !== "advanced") {
    result.paramMode = current.paramMode;
  }
  if (typeof result.windowPattern !== "string") {
    result.windowPattern = current.windowPattern;
  }
  if (result.modelTag !== null && typeof result.modelTag !== "string") {
    result.modelTag = current.modelTag;
  }

  // The one validity rule that would otherwise surface as a crash hours in.
  const micro = result.deviceBatchSize * result.maxSeqLen;
  if (micro <= 0 || result.totalBatchSize % micro !== 0) {
    const largest = [...Object.values(DEFAULT_NANOCHAT_CONFIG)]
      .filter((v): v is number => typeof v === "number")
      .reduce((best, v) => (v % micro === 0 && v > best ? v : best), 0);
    result.totalBatchSize = largest || micro;
  }

  return result;
}

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}
