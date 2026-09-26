// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The nanochat charts, in the same 2x2 grid as Train's.
 *
 * Train's loss and evaluation cards are used as they are: same component, same
 * smoothing, same average line, same placeholder. The other two are nanochat's
 * own measurements, drawn with the same primitives and the same imported
 * constants. Reimplementing the first two here would be the one way to guarantee
 * they stop matching.
 */
import { type ReactElement, useMemo } from "react";

import {
  EvalLossChartCard,
  MAX_RENDER_POINTS,
  TrainingLossChartCard,
  buildStepTicks,
  buildYDomain,
  compressSeries,
  ema,
} from "@/features/studio";

import { RewardChartCard } from "./charts/reward-chart-card";
import { ThroughputChartCard } from "./charts/throughput-chart-card";

export interface NanochatChartSeries {
  step: number;
  value: number | null;
  totalSteps: number | null;
}

export interface NanochatChartsProps {
  loss: NanochatChartSeries[];
  valBpb: NanochatChartSeries[];
  tokPerSec: NanochatChartSeries[];
  mfu: NanochatChartSeries[];
  reward: NanochatChartSeries[];
  chatcore: NanochatChartSeries[];
  isRunning: boolean;
}

// Train's chart types name their field `loss` whatever the metric is, because
// they were written for one; nanochat's wire shape says `value`. Bridged once
// here rather than at every call site.
function finite(points: NanochatChartSeries[]): { step: number; loss: number }[] {
  return points
    .filter((point) => point.value !== null && Number.isFinite(point.value))
    .map((point) => ({ step: point.step, loss: point.value as number }));
}

export function NanochatChartsContent({
  loss,
  valBpb,
  tokPerSec,
  mfu,
  reward,
  chatcore,
  isRunning,
}: NanochatChartsProps): ReactElement | null {
  const lossPoints = useMemo(() => finite(loss), [loss]);
  const evalPoints = useMemo(() => finite(valBpb), [valBpb]);

  const smoothedLoss = useMemo(
    () => (lossPoints.length > 0 ? ema(lossPoints, 0.6) : []),
    [lossPoints],
  );
  const reducedLoss = useMemo(
    () => compressSeries(smoothedLoss, MAX_RENDER_POINTS),
    [smoothedLoss],
  );
  const reducedEval = useMemo(
    () => compressSeries(evalPoints, MAX_RENDER_POINTS),
    [evalPoints],
  );

  // Throughput and MFU are sampled on the same events as loss, so one domain for
  // both keeps the two cards brushable together. They share an axis here because
  // both are "how hard is the card working" readings, not two unrelated measures.
  const throughputPoints = useMemo(() => finite(tokPerSec), [tokPerSec]);
  const mfuByStep = useMemo(() => {
    const map = new Map<number, number>();
    for (const point of finite(mfu)) {
      map.set(point.step, point.loss);
    }
    return map;
  }, [mfu]);
  const reducedThroughput = useMemo(
    () =>
      compressSeries(throughputPoints, MAX_RENDER_POINTS).map((point) => ({
        step: point.step,
        tokPerSec: point.loss,
        mfu: mfuByStep.get(point.step) ?? 0,
      })),
    [mfuByStep, throughputPoints],
  );

  // Reward once RL runs, ChatCORE before that: whichever exists is the one the
  // run is currently optimising, and neither is comparable to loss.
  const rewardPoints = useMemo(() => finite(reward), [reward]);
  const chatcorePoints = useMemo(() => finite(chatcore), [chatcore]);
  const postTrainingPoints = rewardPoints.length > 0 ? rewardPoints : chatcorePoints;
  const postTrainingTitle = rewardPoints.length > 0 ? "Mean reward" : "ChatCORE";
  const reducedPostTraining = useMemo(
    () =>
      compressSeries(postTrainingPoints, MAX_RENDER_POINTS).map((point) => ({
        step: point.step,
        value: point.loss,
      })),
    [postTrainingPoints],
  );

  // Every step the run has reported, so all four cards share one x domain and
  // brush together the way Train's do.
  const allSteps = useMemo(() => {
    const set = new Set<number>();
    for (const points of [lossPoints, throughputPoints, postTrainingPoints]) {
      for (const point of points) {
        set.add(point.step);
      }
    }
    return Array.from(set).sort((a, b) => a - b);
  }, [lossPoints, postTrainingPoints, throughputPoints]);

  const visibleStepDomain = useMemo<[number, number]>(() => {
    if (allSteps.length === 0) {
      return [0, 1];
    }
    const first = allSteps[0];
    const last = allSteps[allSteps.length - 1];
    if (first === last) {
      return [first, first + 4];
    }
    if (last - first < 6) {
      return [Math.max(0, last - 6), last];
    }
    return [first, last];
  }, [allSteps]);

  const xAxisTicks = useMemo(
    () => buildStepTicks(visibleStepDomain[0], visibleStepDomain[1]),
    [visibleStepDomain],
  );

  const lossDomain = useMemo(
    () => buildYDomain(reducedLoss.flatMap((point) => [point.loss, point.smoothed])),
    [reducedLoss],
  );
  const throughputDomain = useMemo(
    () => buildYDomain(reducedThroughput.map((point) => point.tokPerSec)),
    [reducedThroughput],
  );
  const postTrainingDomain = useMemo(
    () => buildYDomain(reducedPostTraining.map((point) => point.value)),
    [reducedPostTraining],
  );
  const evalDomain = useMemo(
    () => buildYDomain(reducedEval.map((point) => point.loss)),
    [reducedEval],
  );
  const evalTicks = useMemo(() => {
    if (reducedEval.length < 2) {
      return undefined;
    }
    return buildStepTicks(
      reducedEval[0].step,
      reducedEval[reducedEval.length - 1].step,
    );
  }, [reducedEval]);

  const avgRaw =
    lossPoints.length > 0
      ? +(lossPoints.reduce((sum, point) => sum + point.loss, 0) / lossPoints.length).toFixed(4)
      : 0;

  if (lossPoints.length === 0 && postTrainingPoints.length === 0) {
    return (
      <p className="rounded-2xl border border-border/60 px-4 py-10 text-center text-sm text-muted-foreground">
        {isRunning
          ? "Metrics appear once the first training step reports."
          : "No metrics yet. Start a run to see loss, validation and throughput here."}
      </p>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      {lossPoints.length > 0 ? (
        <TrainingLossChartCard
          data={reducedLoss.map((point) => ({
            ...point,
            displayLoss: point.loss,
            displaySmoothed: point.smoothed,
          }))}
          domain={lossDomain}
          visibleStepDomain={visibleStepDomain}
          xAxisTicks={xAxisTicks}
          avgRaw={avgRaw}
          avgDisplay={avgRaw}
          showRaw={true}
          showSmoothed={true}
          showAvgLine={true}
          scale="linear"
        />
      ) : null}
      <RewardChartCard
        data={reducedPostTraining.map((point) => ({
          ...point,
          displayValue: point.value,
        }))}
        domain={postTrainingDomain}
        visibleStepDomain={visibleStepDomain}
        xAxisTicks={xAxisTicks}
        title={postTrainingTitle}
        emptyHint={
          postTrainingTitle === "Mean reward"
            ? "Reinforcement learning has not reported a reward yet."
            : "ChatCORE is measured after the first evaluation of the chat model."
        }
      />
      <ThroughputChartCard
        data={reducedThroughput}
        domain={throughputDomain}
        visibleStepDomain={visibleStepDomain}
        xAxisTicks={xAxisTicks}
        emptyHint="Throughput is reported alongside the first training step."
      />
      <EvalLossChartCard
        data={reducedEval}
        domain={evalDomain}
        ticks={evalTicks}
        isTraining={isRunning}
        evalEnabled={true}
      />
    </div>
  );
}
