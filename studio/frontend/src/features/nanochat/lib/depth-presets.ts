// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Fallback depth presets, used before the backend answers and if it cannot.
 *
 * The numbers are the same ones the backend computes, and they are asserted
 * against it in `depth-presets.test.ts`, so a change on either side fails a test
 * rather than leaving the picker quoting a stale figure. The backend remains the
 * source of truth; this exists so the tab renders something meaningful on first
 * paint instead of an empty state.
 */

export const PRESET_DEFAULT_DEPTH = 12;

export type FallbackPreset = {
  /** Matches the backend's key, so the two sources are interchangeable. */
  key: string;
  depth: number;
  label: string;
  tier: "tiny" | "small" | "medium" | "large" | "xlarge";
  summary: string;
  goodFor: string;
  caveat?: string;
  numParams: number;
  transformerParams: number;
  embeddingParams: number;
  nEmbd: number;
  nHead: number;
  kvCacheBytesPerToken: number;
};

export const FALLBACK_PRESETS: readonly FallbackPreset[] = [
  {
    key: "d4", depth: 4, label: "d4", tier: "tiny",
    summary: "37M params, but only 3M of them transformer",
    goodFor: "Seeing the whole pipeline end to end. Not a usable model.",
    caveat: "Almost all of it is the token embedding table. Far too small to produce coherent text.",
    numParams: 36_700_242, transformerParams: 3_145_776, embeddingParams: 33_554_432,
    nEmbd: 256, nHead: 2, kvCacheBytesPerToken: 4_096,
  },
  {
    key: "d6", depth: 6, label: "d6", tier: "tiny",
    summary: "74M params, 11M transformer",
    goodFor: "A working toy model you can watch learn a small domain.",
    caveat: "Output is mostly word salad.",
    numParams: 73_531_538, transformerParams: 10_616_940, embeddingParams: 62_914_560,
    nEmbd: 384, nHead: 3, kvCacheBytesPerToken: 9_216,
  },
  {
    key: "d8", depth: 8, label: "d8", tier: "small",
    summary: "126M params, 25M transformer, minutes on a modern GPU",
    goodFor: "A first model that completes short prompts recognisably.",
    numParams: 125_829_354, transformerParams: 25_166_016, embeddingParams: 100_663_296,
    nEmbd: 512, nHead: 4, kvCacheBytesPerToken: 16_384,
  },
  {
    key: "d10", depth: 10, label: "d10", tier: "small",
    summary: "196M params, 49M transformer",
    goodFor: "Follows simple instructions. A sensible first real run.",
    numParams: 195_952_986, transformerParams: 49_152_300, embeddingParams: 146_800_640,
    nEmbd: 640, nHead: 5, kvCacheBytesPerToken: 25_600,
  },
  {
    key: "d12", depth: 12, label: "d12", tier: "medium",
    summary: "286M params, 85M transformer, the smallest nanochat miniseries depth",
    goodFor: "Coherent short answers, roughly GPT-2-small quality.",
    numParams: 286_261_730, transformerParams: 84_935_088, embeddingParams: 201_326_592,
    nEmbd: 768, nHead: 6, kvCacheBytesPerToken: 36_864,
  },
  {
    key: "d16", depth: 16, label: "d16", tier: "medium",
    summary: "537M params, 201M transformer",
    goodFor: "Noticeably better reasoning. Wants 12GB+ of VRAM.",
    numParams: 536_871_738, transformerParams: 201_327_360, embeddingParams: 335_544_320,
    nEmbd: 1024, nHead: 8, kvCacheBytesPerToken: 65_536,
  },
  {
    key: "d20", depth: 20, label: "d20", tier: "large",
    summary: "897M params, 393M transformer, nanochat's default depth",
    goodFor: "Chattier and more accurate. Expect a multi-hour run on one GPU.",
    numParams: 896_533_746, transformerParams: 393_217_200, embeddingParams: 503_316_480,
    nEmbd: 1280, nHead: 10, kvCacheBytesPerToken: 102_400,
  },
  {
    key: "d24", depth: 24, label: "d24", tier: "large",
    summary: "1.38B params, 679M transformer, the depth of the published speedrun",
    goodFor: "GPT-2 class quality.",
    caveat: "The speedrun reached this on 8xH100 in about 90 minutes. On one consumer GPU it is a multi-day commitment.",
    numParams: 1_384_122_122, transformerParams: 679_478_976, embeddingParams: 704_643_072,
    nEmbd: 1536, nHead: 12, kvCacheBytesPerToken: 147_456,
  },
  {
    key: "d32", depth: 32, label: "d32", tier: "xlarge",
    summary: "2.8B params, 1.6B transformer",
    goodFor: "Scaling-law experiments.",
    caveat: "Weeks on one consumer GPU. Included so the scaling curve is explorable, not because it is practical here.",
    numParams: 2_818_575_450, transformerParams: 1_610_615_808, embeddingParams: 1_207_959_552,
    nEmbd: 2048, nHead: 16, kvCacheBytesPerToken: 262_144,
  },
];

/** Compact counts for the picker, e.g. "286M". */
export function formatParams(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(count >= 1e10 ? 0 : 2)}B`;
  if (count >= 1e6) return `${Math.round(count / 1e6)}M`;
  if (count >= 1e3) return `${Math.round(count / 1e3)}K`;
  return String(count);
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "—";
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

/** A duration from seconds, phrased for a wait. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) {
    return "—";
  }
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86_400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86_400).toFixed(1)}d`;
}
