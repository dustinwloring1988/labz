// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Shared formatting for benchmark numbers.
 *
 * A score is a fraction, and the two places it appears want it formatted
 * differently: a run's own score wants the precision that makes two runs
 * distinguishable, and a leaderboard cell wants it compact enough that a column
 * of them can be scanned. Both are here so the rounding rule lives in one place.
 */

/** A benchmark key as a readable name, for the cases where only the key is to hand. */
const KEY_LABELS: Record<string, string> = {
  mmlu: "MMLU",
  "arc-easy": "ARC-Easy",
  "arc-challenge": "ARC-Challenge",
  gsm8k: "GSM8K",
  humaneval: "HumanEval",
};

export function benchmarkLabel(key: string, label?: string | null): string {
  if (label && label.length > 0) return label;
  return KEY_LABELS[key] ?? key;
}

/** An accuracy as a percentage, or an em dash when there is none.
 *
 * The em dash rather than 0% is the point: a benchmark that did not run, was
 * skipped, or produced nothing must never read as a score of zero, which would
 * look like a model that got everything wrong rather than one that was not
 * measured.
 */
export function formatAccuracy(accuracy: number | null | undefined): string {
  if (accuracy === null || accuracy === undefined || !Number.isFinite(accuracy))
    return "—";
  return `${(accuracy * 100).toFixed(1)}%`;
}

/** A centred score as a percentage. See `formatAccuracy` on the em dash. */
export function formatCentered(centered: number | null | undefined): string {
  if (centered === null || centered === undefined || !Number.isFinite(centered))
    return "—";
  return `${(centered * 100).toFixed(0)}%`;
}

/** A duration as `4m 09s` / `1h 04m`. Coarse on purpose: this is a column, not a metric. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds))
    return "—";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${String(secs).padStart(2, "0")}s`;
  return `${secs}s`;
}

/** How a score was measured, in words a reader can act on. */
export function scoringModeLabel(mode: string | null | undefined): string {
  if (mode === "logits") return "exact (logits)";
  if (mode === "generated") return "generated answer";
  if (mode === "judged") return "judged answer";
  if (mode === "mixed") return "mixed methods";
  return "unknown";
}

/** Whether a scoring mode is the exact one.
 *
 * The leaderboard uses this to warn about a board that mixes methods, because
 * the numbers are not comparable and the reader is the only one who can tell
 * which rows they are looking at.
 */
export function isExactScoringMode(mode: string | null | undefined): boolean {
  return mode === "logits";
}

/** How much of a suite a judge actually returned a verdict for.
 *
 * Null when there is no judge on the row. The two counts are kept rather than
 * collapsed into one number because the distinction is the point: an answer the
 * judge could not parse is one it never graded, and counting it as wrong would
 * report a score for a question nobody answered.
 */
export function formatJudgeCoverage(
  judged: number | null | undefined,
  unreadable: number | null | undefined,
): string | null {
  if (judged === null || judged === undefined) return null;
  const read = judged + (unreadable ?? 0);
  if (read <= 0) return null;
  if (!unreadable) return `judged ${read}/${read}`;
  return `judged ${judged}/${read} · ${unreadable} unreadable`;
}
