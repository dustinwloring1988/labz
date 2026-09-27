// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DEFAULT_AUTORESEARCH_CONFIG,
  MAX_EXPERIMENTS,
  MAX_TIMEOUT_MINUTES,
  MIN_EXPERIMENTS,
  MIN_TIMEOUT_MINUTES,
  type AutoresearchConfigState,
} from "./autoresearch-config-policy.ts";

export const AUTORESEARCH_CONFIG_PERSISTENCE_VERSION = 1;

/**
 * Merge a persisted configuration over the current defaults.
 *
 * Two rules, both about not silently changing what a run will do:
 *
 *  - An unknown key is dropped and a missing key takes the default. A
 *    downgrading user must not end up with `undefined` in a field the backend
 *    requires.
 *  - A value that is now out of bounds is clamped rather than kept. An
 *    experiment count is a real number of real training runs, so a stored value
 *    above the current ceiling has to come down rather than survive a reload.
 */
export function migrateAutoresearchConfig(
  persisted: Partial<AutoresearchConfigState> | undefined,
  current: AutoresearchConfigState,
): AutoresearchConfigState {
  if (!persisted || typeof persisted !== "object") return current;

  const merged = { ...current } as Record<string, unknown>;
  for (const [key, value] of Object.entries(persisted)) {
    if (key in current) merged[key] = value;
  }

  const result = merged as unknown as AutoresearchConfigState;

  result.numExperiments = clampInt(
    result.numExperiments,
    MIN_EXPERIMENTS,
    MAX_EXPERIMENTS,
    current.numExperiments,
  );
  result.agentTimeoutMinutes = clampInt(
    result.agentTimeoutMinutes,
    MIN_TIMEOUT_MINUTES,
    MAX_TIMEOUT_MINUTES,
    current.agentTimeoutMinutes,
  );
  result.maxConsecutiveFailures = clampInt(
    result.maxConsecutiveFailures,
    0,
    MAX_EXPERIMENTS,
    current.maxConsecutiveFailures,
  );

  if (typeof result.agent !== "string" || !result.agent) {
    result.agent = current.agent;
  }
  if (typeof result.agentArgs !== "string") {
    result.agentArgs = current.agentArgs;
  } else {
    // Whitespace-collapsed so a value that wrapped across lines in a textarea
    // still reads as one argument list.
    result.agentArgs = result.agentArgs.replace(/\s*\n\s*/g, " ").trim();
  }
  if (typeof result.focus !== "string") {
    result.focus = current.focus;
  } else {
    // Collapsed on the way in as well as out: the backend flattens it too,
    // because it is spliced into a prompt.
    result.focus = result.focus.replace(/\s*\n\s*/g, " ").trim();
  }
  for (const flag of [
    "skipPermissions",
    "prepareData",
    "reportAfterFinish",
  ] as const) {
    if (typeof result[flag] !== "boolean") {
      result[flag] = current[flag];
    }
  }
  if (typeof result.reportModelId !== "string") {
    result.reportModelId = current.reportModelId;
  }
  if (result.reportSource !== "catalog" && result.reportSource !== "autoresearch"
      && result.reportSource !== "agent") {
    result.reportSource = current.reportSource;
  }
  if (typeof result.reportAgentKey !== "string") {
    result.reportAgentKey = current.reportAgentKey;
  }

  return result;
}

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}

export { DEFAULT_AUTORESEARCH_CONFIG };
