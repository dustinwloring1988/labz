// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What a benchmark run is configured to do, and the rules for what is valid.
 *
 * Kept apart from the store so the rules are pure functions and testable without a
 * React tree, which is the same split the nanochat config uses.
 */

/** The default problem cap.
 *
 * Not a round number chosen for looks: MMLU's test split is 14,042 problems and
 * ARC-Easy's is 2,376, so running a suite unbounded is an overnight job with no
 * warning. 400 is roughly ten minutes per categorical benchmark on a modern GPU,
 * which is enough for the number to have stopped moving, and the run preview
 * states the cap rather than applying it silently.
 */
export const DEFAULT_MAX_PROBLEMS = 400;

export const MAX_PROBLEMS_CEILING = 20_000;

export const MIN_PROBLEMS = 8;

export const DEFAULT_BATCH_SIZE = 8;

export const DEFAULT_MAX_SEQ_LENGTH = 2048;

/** Generative answers get far more room than a letter does. */
export const DEFAULT_MAX_NEW_TOKENS = 512;

export type BenchmarkFormState = {
  /** The chosen model's path, which is how a model is identified across reloads:
   *  a display label is not unique and a hub id is not always known. */
  modelPath: string | null;
  /** The second opinion's path, or null to run without a judge. Identified the same
   *  way as `modelPath`, for the same reason. */
  judgeModelPath: string | null;
  maxProblems: number;
  batchSize: number;
  maxNewTokens: number;
  maxSeqLength: number;
  loadIn4bit: boolean;
  benchmarkKeys: string[];
};

export const DEFAULT_BENCHMARK_FORM: BenchmarkFormState = {
  modelPath: null,
  judgeModelPath: null,
  maxProblems: DEFAULT_MAX_PROBLEMS,
  batchSize: DEFAULT_BATCH_SIZE,
  maxNewTokens: DEFAULT_MAX_NEW_TOKENS,
  maxSeqLength: DEFAULT_MAX_SEQ_LENGTH,
  loadIn4bit: false,
  benchmarkKeys: [],
};

export type BenchmarkFormProblem = {
  field:
    | "model"
    | "judgeModel"
    | "benchmarks"
    | "maxProblems"
    | "batchSize";
  message: string;
};

function isIntegerInRange(value: number, min: number, max: number): boolean {
  return Number.isInteger(value) && value >= min && value <= max;
}

export function clampMaxProblems(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_MAX_PROBLEMS;
  return Math.min(
    MAX_PROBLEMS_CEILING,
    Math.max(MIN_PROBLEMS, Math.round(value)),
  );
}

/**
 * What is wrong with the form, or an empty list when it can be run.
 *
 * Returns every problem rather than the first, so the page can mark each field
 * that needs attention instead of making the user fix them one at a time.
 */
export function validateBenchmarkForm(
  form: BenchmarkFormState,
  availableKeys: readonly string[],
): BenchmarkFormProblem[] {
  const problems: BenchmarkFormProblem[] = [];

  if (!form.modelPath) {
    problems.push({ field: "model", message: "Choose a model to benchmark." });
  }

  const usable = form.benchmarkKeys.filter((key) =>
    availableKeys.includes(key),
  );
  if (usable.length === 0) {
    problems.push({
      field: "benchmarks",
      message: "Select at least one benchmark.",
    });
  }

  if (!isIntegerInRange(form.maxProblems, MIN_PROBLEMS, MAX_PROBLEMS_CEILING)) {
    problems.push({
      field: "maxProblems",
      message: `Problems per benchmark must be a whole number between ${MIN_PROBLEMS} and ${MAX_PROBLEMS_CEILING}.`,
    });
  }

  if (!isIntegerInRange(form.batchSize, 1, 256)) {
    problems.push({
      field: "batchSize",
      message: "Batch size must be between 1 and 256.",
    });
  }

  return problems;
}

export function canStartBenchmarkRun(
  form: BenchmarkFormState,
  availableKeys: readonly string[],
): boolean {
  return validateBenchmarkForm(form, availableKeys).length === 0;
}

/** Add or remove a key, preserving the catalogue's order rather than click order.
 *
 * The picker renders the catalogue's order and the run follows it, so a suite
 * reads the same way every time regardless of the order it was assembled in.
 */
export function toggleBenchmarkKey(
  selected: readonly string[],
  key: string,
  catalogueOrder: readonly string[],
): string[] {
  const next = selected.includes(key)
    ? selected.filter((entry) => entry !== key)
    : [...selected, key];
  return catalogueOrder.filter((entry) => next.includes(entry));
}

/** The benchmarks a run will actually attempt, in catalogue order. */
export function resolveSelectedKeys(
  selected: readonly string[],
  catalogueOrder: readonly string[],
): string[] {
  const chosen = new Set(selected);
  return catalogueOrder.filter((key) => chosen.has(key));
}

/** Whether the chosen model forces the weaker scoring path.
 *
 * A GGUF has no safetensors checkpoint behind it, so it cannot be loaded into the
 * scoring process and is measured by asking the running llama-server instead. The
 * run preview says so rather than letting the user assume the exact measurement.
 */
export function scoringModeForFormat(
  format: "safetensors" | "gguf",
): "logits" | "generated" {
  return format === "gguf" ? "generated" : "logits";
}

/**
 * Whether a judge would do anything for this selection.
 *
 * Only generative benchmarks are judged. A categorical one is answered with a
 * single letter compared against single-letter logits, and asking a second model
 * "did it say A when gold is A" can only add noise -- so the picker is disabled
 * rather than the run being rejected, since a judge left over from an earlier
 * selection is not a mistake the user needs to fix.
 */
export function judgeApplies(
  selected: readonly string[],
  generativeKeys: readonly string[],
): boolean {
  const chosen = new Set(selected);
  return generativeKeys.some((key) => chosen.has(key));
}
