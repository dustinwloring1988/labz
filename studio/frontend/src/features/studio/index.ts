// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Train feature's public surface.
 *
 * The chart cards and their scales are shared with the nanochat tab, which is
 * the point: one loss chart, one evaluation chart, one set of formatters, so the
 * two training tabs cannot drift apart visually. Everything re-exported here is
 * what another feature is allowed to build on; the rest of the Train internals
 * stay private.
 */
export { EvalLossChartCard } from "./sections/charts/eval-loss-chart-card";
export { TrainingLossChartCard } from "./sections/charts/training-loss-chart-card";
export {
  CHART_CONTAINER_CLASS,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  MAX_RENDER_POINTS,
  buildStepTicks,
  buildYDomain,
  compressSeries,
  ema,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
} from "./sections/charts/utils";
