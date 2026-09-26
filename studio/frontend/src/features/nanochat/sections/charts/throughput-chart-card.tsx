// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Throughput: tokens per second, with the model's share of the card's peak
 * floats alongside it.
 *
 * Built from the Train chart primitives rather than by reusing one of Train's
 * cards, because the shape of the data is nanochat's own. What makes it look
 * like a Train chart is that every constant, formatter and class here is Train's,
 * imported from the module Train uses, so the two tabs cannot drift apart.
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

export interface ThroughputChartPoint {
  step: number;
  tokPerSec: number;
  mfu: number;
}

export function ThroughputChartCard({
  data,
  domain,
  visibleStepDomain,
  xAxisTicks,
  emptyHint,
}: {
  data: ThroughputChartPoint[];
  domain: [number, number];
  visibleStepDomain: [number, number];
  xAxisTicks: number[];
  emptyHint: string;
}): ReactElement {
  const t = useT();
  const throughputConfig = {
    displayTokPerSec: { label: "tokens/sec", color: "#10b981" },
    displayMfu: { label: "MFU %", color: "#0ea5e9" },
  } satisfies ChartConfig;
  const showPoint = data.length <= 1 ? { r: 3, strokeWidth: 0 } : false;

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Throughput</CardTitle>
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <div
            className={`${CHART_CONTAINER_CLASS} flex items-center justify-center rounded-lg border border-dashed border-border/60 px-4 text-center text-sm text-muted-foreground`}
          >
            {emptyHint}
          </div>
        ) : (
          <ChartContainer config={throughputConfig} className={CHART_CONTAINER_CLASS}>
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
                    formatter={(_value, name, item) => {
                      if (name === "displayMfu") {
                        return [`${formatMetric(Number(item?.payload?.mfu))}%`, "MFU"];
                      }
                      return [
                        formatMetric(Number(item?.payload?.tokPerSec)),
                        "tokens/sec",
                      ];
                    }}
                  />
                }
              />
              <Line
                type="linear"
                dataKey="displayTokPerSec"
                stroke="var(--color-displayTokPerSec)"
                strokeWidth={2}
                dot={showPoint}
                activeDot={{ r: 3, strokeWidth: 0 }}
                connectNulls={true}
                strokeLinecap="round"
                strokeLinejoin="round"
                isAnimationActive={false}
              />
              <Line
                type="linear"
                dataKey="displayMfu"
                stroke="var(--color-displayMfu)"
                strokeWidth={1.5}
                strokeDasharray="4 3"
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
