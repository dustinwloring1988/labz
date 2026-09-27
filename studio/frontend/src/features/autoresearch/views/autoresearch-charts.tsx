// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The chart of a search, and nothing else.
 *
 * Split out and lazily loaded for the same reason nanochat's charts are: the
 * charting library is the largest thing this feature imports, and most visits to
 * the tab are about the run rather than the history.
 *
 * What is plotted is the frontier, not the scores. Every experiment that landed
 * is a step, and the line that matters is the running minimum, because a
 * discarded experiment going nowhere is noise and a kept one is the answer to
 * "what did the search find".
 */
import { lazy, Suspense } from "react";

import type { AutoresearchMetricPoint } from "../types/index.ts";

const AutoresearchChartsContent = lazy(() =>
  import("./autoresearch-charts-content").then((module) => ({
    default: module.AutoresearchChartsContent,
  })),
);

// Enough cards to fill a row, past which the placeholder stops telling the user
// anything about what is coming.
const SKELETON_KEYS = ["frontier", "memory"];

export function AutoresearchCharts(props: {
  series: AutoresearchMetricPoint[];
  baselineValBpb: number | null;
  bestValBpb: number | null;
  height?: number;
}) {
  return (
    <Suspense
      fallback={
        <div className="grid gap-4 sm:grid-cols-2">
          {SKELETON_KEYS.map((key) => (
            <div
              key={key}
              className="h-48 animate-pulse rounded-2xl border border-border bg-muted/30"
            />
          ))}
        </div>
      }
    >
      <AutoresearchChartsContent {...props} />
    </Suspense>
  );
}
