// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * nanochat: training a language model from scratch, locally.
 *
 * Separate from the Train tab because it is a different thing: Train fine-tunes an
 * existing model, while nanochat pretrains one from random weights through its own
 * pipeline (data, tokenizer, pretrain, evaluate, fine-tune, evaluate, RL) in its own
 * virtualenv.
 */

export { NanochatPage } from "./nanochat-page";

export {
  useNanochatRuntimeStore,
  isNanochatRunActive,
  isNanochatRunFinished,
  type NanochatRuntimeState,
  type ProgressEvent,
} from "./stores/nanochat-runtime-store";

export {
  useNanochatConfigStore,
  type NanochatConfigStore,
} from "./stores/nanochat-config-store";

export {
  DEFAULT_NANOCHAT_CONFIG,
  isValidTotalBatchSize,
  gradientAccumulationSteps,
  validTotalBatchSizes,
  type NanochatConfigState,
  type NanochatStageKey,
  type NanochatParamMode,
} from "./stores/nanochat-config-policy";

export * from "./api/nanochat-api";
export * from "./types";
