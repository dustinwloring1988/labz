// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Reward / ChatCORE: the quantity the post-training stages are optimising.
 *
 * One card for both, because they are the same measurement at different points
 * in the pipeline: RL reports mean reward per step, SFT reports ChatCORE at
 * evaluation time, and neither is comparable to loss. Whichever the run has
 * emitted is the one that gets drawn.
 */
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import {
  CHART_CONTAINER_CLASS,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
} from "@/features/studio";
import { useT } from "@/i18n";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { NANOCHAT_CHART_SYNC_ID } from "./utils";

export interface RewardChartPoint {
  step: number;
  value: number;
  displayValue: number;
}

export function RewardChartCard({
  data,
  domain,
  visibleStepDomain,
  xAxisTicks,
  title,
  emptyHint,
}: {
  data: RewardChartPoint[];
  domain: [number, number];
  visibleStepDomain: [number, number];
  xAxisTicks: number[];
  title: string;
  emptyHint: string;
}): ReactElement {
  const t = useT();
  const rewardConfig = {
    displayValue: { label: title, color: "#a855f7" },
  } satisfies ChartConfig;
  const showPoint = data.length <= 1 ? { r: 3, strokeWidth: 0 } : false;

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <div
            className={`${CHART_CONTAINER_CLASS} flex items-center justify-center rounded-lg border border-dashed border-border/60 px-4 text-center text-sm text-muted-foreground`}
          >
            {emptyHint}
          </div>
        ) : (
          <ChartContainer config={rewardConfig} className={CHART_CONTAINER_CLASS}>
            <LineChart
              data={data}
              syncId={NANOCHAT_CHART_SYNC_ID}
              syncMethod="value"
              accessibilityLayer={true}
              margin={DEFAULT_CHART_MARGIN}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="step"
                type="number"
                domain={visibleStepDomain}
                ticks={xAxisTicks}
                allowDataOverflow={true}
                allowDecimals={false}
                minTickGap={28}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                fontSize={10}
                tickFormatter={(value) => formatStepTick(Number(value))}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={domain}
                allowDataOverflow={true}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickCount={5}
                fontSize={10}
                width={DEFAULT_Y_AXIS_WIDTH}
                tickFormatter={(value) => formatAxisMetric(Number(value))}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_value, payload) =>
                      t("studio.charts.step", {
                        step: payload?.[0]?.payload?.step ?? "",
                      })
                    }
                    formatter={(_value, _name, item) => [
                      formatMetric(Number(item?.payload?.value)),
                      title,
                    ]}
                  />
                }
              />
              <Line
                type="linear"
                dataKey="displayValue"
                stroke="var(--color-displayValue)"
                strokeWidth={2}
                dot={showPoint}
                activeDot={{ r: 3, strokeWidth: 0 }}
                connectNulls={true}
                strokeLinecap="round"
                strokeLinejoin="round"
                isAnimationActive={false}
              />
              <ChartLegend content={<ChartLegendContent />} />
            </LineChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}
