// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { AutoresearchPage } from "./autoresearch-page";
export type { AutoresearchTab } from "./autoresearch-navigation";
export {
  useAutoresearchRuntimeStore,
  isAutoresearchRunActive,
  isAutoresearchRunFinished,
  hasAutoresearchHistory,
  activeAutoresearchIndex,
  type AutoresearchRuntimeState,
  type AutoresearchRuntimeActions,
  type ProgressEvent,
} from "./stores/autoresearch-runtime-store";
export {
  useAutoresearchConfigStore,
  AUTORESEARCH_CONFIG_PERSISTENCE_NAME,
  type AutoresearchConfigStore,
} from "./stores/autoresearch-config-store";
export {
  DEFAULT_AUTORESEARCH_CONFIG,
  validateAutoresearchConfig,
  estimateRunSeconds,
  agentLabel,
  describeAgentArgs,
  parseAgentArgs,
  AGENT_OWNED_FLAGS,
  MIN_EXPERIMENTS,
  MAX_EXPERIMENTS,
  MIN_TIMEOUT_MINUTES,
  MAX_TIMEOUT_MINUTES,
  SECONDS_PER_EXPERIMENT_ESTIMATE,
  type AutoresearchConfigState,
} from "./stores/autoresearch-config-policy";
export * from "./api/autoresearch-api";
export * from "./types";
