// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Number and label formatting shared by the autoresearch views. */

/** "1.2 GB". Binary units, because that is what a VRAM figure is quoted in. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "—";
  if (bytes === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = value >= 100 || unit === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

/**
 * A duration, in words.
 *
 * Words rather than a clock because nobody is waiting for a stopwatch: what the
 * user wants to know is whether a run is five minutes or all afternoon.
 */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const secondsLeft = total % 60;
  if (minutes < 60) {
    return secondsLeft === 0 ? `${minutes}m` : `${minutes}m ${secondsLeft}s`;
  }
  const hours = Math.floor(minutes / 60);
  const minutesLeft = minutes % 60;
  if (hours < 24) {
    return minutesLeft === 0 ? `${hours}h` : `${hours}h ${minutesLeft}m`;
  }
  const days = Math.floor(hours / 24);
  const hoursLeft = hours % 24;
  return hoursLeft === 0 ? `${days}d` : `${days}d ${hoursLeft}h`;
}

/**
 * val_bpb, at the precision that matters.
 *
 * Six decimals is the project's own precision and the difference between two
 * real attempts. Three would show several neighbouring experiments as identical,
 * which is the one thing this number is for.
 */
export function formatBpb(value: number | null | undefined, digits = 6): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

/** The change from one score to another, as a signed number. */
export function formatDelta(value: number | null | undefined, digits = 6): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

/** A large count, with separators. */
export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Math.round(value).toLocaleString();
}

/** Throughput, from tok/s. */
export function formatThroughput(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k tok/s`;
  return `${Math.round(value)} tok/s`;
}

/** Model parameters, from a parameter count in millions. */
export function formatParamsM(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value.toFixed(1)}M params`;
}

/** Loss and MFU, both to a sensible number of digits. */
export function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export const EXPERIMENT_STATUS_LABELS: Record<string, string> = {
  pending: "Not started",
  running: "Running",
  kept: "Kept",
  discarded: "Discarded",
  crashed: "Crashed",
  stopped: "Stopped",
};
