// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The search's two charts: the frontier, and what each experiment cost in VRAM.
 *
 * Built from the same primitives as the Train and nanochat charts rather than
 * by reaching for a charting library directly, which is what makes a card on this
 * tab look like a card on the others. The data is autoresearch's own: one point
 * per experiment rather than per step, because an experiment is the unit the
 * search reasons in.
 *
 * Two choices worth stating, because both are ways this chart could lie:

 *  - **Crashes are drawn as gaps, not as zeros.** autoresearch's own instructions
 *    record a failed run as `0.000000`, which sorts as a perfect score. Plotting
 *    that would draw a crashed experiment as the biggest win of the search. The
 *    zero is recognised and left out, so a gap means "this produced nothing".

 *  - **The frontier is a step function.** The score only changes when an
 *    experiment is recorded, and joining the points with a straight line would
 *    draw values between two experiments that never existed.
 */
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import {
  CHART_CONTAINER_CLASS,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
} from "@/features/studio";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { formatBpb } from "../lib/format";
import type { AutoresearchMetricPoint } from "../types/index.ts";

type Point = AutoresearchMetricPoint & {
  /** The running minimum at this point, or null before the first score. */
  frontier: number | null;
  /** A point that finished, as opposed to one that is a gap. */
  scored: number | null;
};

function buildSeries(series: AutoresearchMetricPoint[]): Point[] {
  let best: number | null = null;
  return series.map((row) => {
    if (row.val_bpb === null || !Number.isFinite(row.val_bpb)) {
      return { ...row, frontier: best, scored: null };
    }
    if (best === null || row.val_bpb < best) best = row.val_bpb;
    return { ...row, frontier: best, scored: row.val_bpb };
  });
}

/** Enough ticks to be readable and few enough not to collide. */
function tickIndices(max: number): number[] {
  if (max <= 10) return Array.from({ length: max }, (_unused, index) => index + 1);
  const stepSize = Math.ceil(max / 8);
  const ticks: number[] = [];
  for (let index = 1; index <= max; index += stepSize) ticks.push(index);
  if (ticks[ticks.length - 1] !== max) ticks.push(max);
  return ticks;
}

const frontierConfig = {
  frontier: { label: "Best val bpb", color: "#10b981" },
  scored: { label: "This experiment", color: "#94a3b8" },
  kept: { label: "Kept", color: "#10b981" },
} satisfies ChartConfig;

const memoryConfig = {
  memory_gb: { label: "Peak VRAM (GB)", color: "#f59e0b" },
} satisfies ChartConfig;

export function AutoresearchChartsContent({
  series,
  baselineValBpb,
  height = 220,
}: {
  series: AutoresearchMetricPoint[];
  baselineValBpb: number | null;
  bestValBpb: number | null;
  height?: number;
}) {
  const points = buildSeries(series);
  const ticks = tickIndices(Math.max(1, points.length));

  if (points.length === 0) {
    return (
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm">The frontier</CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className={`${CHART_CONTAINER_CLASS} flex items-center justify-center rounded-lg border border-dashed border-border/60 px-4 text-center text-sm text-muted-foreground`}
            style={{ height }}
          >
            No experiments have finished yet. The frontier appears once the first one
            has a val_bpb.
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm">The frontier</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={frontierConfig}
            className={CHART_CONTAINER_CLASS}
            style={{ height }}
          >
            <LineChart
              data={points}
              accessibilityLayer
              margin={DEFAULT_CHART_MARGIN}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="index"
                type="number"
                ticks={ticks}
                allowDataOverflow
                allowDecimals={false}
                minTickGap={24}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                fontSize={10}
              />
              <YAxis
                // Reversed: lower val_bpb is better, so a line that goes down is
                // a line that goes up in quality. A conventional axis would make
                // the best result look like the worst.
                reversed
                domain={["dataMin", "dataMax"]}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickCount={5}
                fontSize={10}
                width={DEFAULT_Y_AXIS_WIDTH}
                tickFormatter={(value) => Number(value).toFixed(3)}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_value, payload) => {
                      const row = payload?.[0]?.payload as Point | undefined;
                      if (!row) return "";
                      return row.scored === null
                        ? `Experiment ${row.index} (did not finish)`
                        : `Experiment ${row.index} · val bpb ${formatBpb(row.val_bpb)}`;
                    }}
                    formatter={(_value, name, item) => {
                      const row = item?.payload as Point | undefined;
                      if (!row) return [null, name];
                      if (name === "kept") {
                        return [row.description || "kept this change", "Kept"];
                      }
                      if (name === "frontier") {
                        return [
                          row.frontier === null ? "—" : formatBpb(row.frontier),
                          "Best so far",
                        ];
                      }
                      return [row.scored === null ? "crashed" : formatBpb(row.scored), name];
                    }}
                  />
                }
              />
              {baselineValBpb !== null ? (
                <ReferenceLine
                  y={baselineValBpb}
                  stroke="var(--color-kept)"
                  strokeDasharray="4 3"
                  strokeOpacity={0.5}
                  label={{
                    value: "baseline",
                    position: "insideTopRight",
                    fontSize: 9,
                    fill: "currentColor",
                    opacity: 0.6,
                  }}
                />
              ) : null}
              {/* Every experiment that finished, so a discarded one is visible
                  as a point rather than only as a gap in the frontier. */}
              <Line
                type="linear"
                dataKey="scored"
                stroke="var(--color-scored)"
                strokeWidth={1}
                strokeOpacity={0.5}
                dot={{ r: 2, strokeWidth: 0, fill: "var(--color-scored)" }}
                connectNulls={false}
                isAnimationActive={false}
              />
              <Line
                type="stepAfter"
                dataKey="frontier"
                stroke="var(--color-frontier)"
                strokeWidth={2}
                dot={false}
                connectNulls
                strokeLinecap="round"
                strokeLinejoin="round"
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm">What each experiment cost</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={memoryConfig}
            className={CHART_CONTAINER_CLASS}
            style={{ height }}
          >
            <ScatterChart
              accessibilityLayer
              margin={DEFAULT_CHART_MARGIN}
              data={points}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="index"
                type="number"
                ticks={ticks}
                allowDataOverflow
                allowDecimals={false}
                minTickGap={24}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                fontSize={10}
                label={{ value: "experiment", position: "insideBottom", offset: -2, fontSize: 9 }}
              />
              <YAxis
                dataKey="memory_gb"
                type="number"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickCount={5}
                fontSize={10}
                width={DEFAULT_Y_AXIS_WIDTH}
                unit=" GB"
              />
              <ZAxis range={[60, 60]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const row = payload[0].payload as Point;
                  return (
                    <div className="rounded-lg border border-border bg-popover px-2.5 py-2 text-xs shadow-md">
                      <p className="font-medium">Experiment {row.index}</p>
                      {row.memory_gb !== null ? (
                        <p className="text-muted-foreground">
                          Peak VRAM {row.memory_gb.toFixed(2)} GB
                        </p>
                      ) : null}
                      {row.val_bpb !== null ? (
                        <p className="text-muted-foreground">
                          val bpb {formatBpb(row.val_bpb)}
                        </p>
                      ) : (
                        <p className="text-muted-foreground">did not finish</p>
                      )}
                    </div>
                  );
                }}
              />
              <Scatter
                dataKey="memory_gb"
                fill="var(--color-memory_gb)"
                isAnimationActive={false}
              />
            </ScatterChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  );
}
