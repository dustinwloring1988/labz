// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The charts section, lazy like Train's so recharts stays out of the first
 * paint of a page whose main content is a form.
 */
import { lazy, Suspense } from "react";

import type { NanochatChartsProps } from "./nanochat-charts-content";

const NanochatChartsContent = lazy(() =>
  import("./nanochat-charts-content").then((module) => ({
    default: module.NanochatChartsContent,
  })),
);

const SKELETON_KEYS = ["loss", "post-training", "throughput", "evaluation"];

export function NanochatCharts(props: NanochatChartsProps) {
  return (
    <Suspense
      fallback={
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {SKELETON_KEYS.map((key) => (
            <div
              key={key}
              className="h-[calc(280px*var(--ui-space-scale,1))] 4xl:h-[calc(360px*var(--ui-space-scale,1))] rounded-xl border bg-muted/30 animate-pulse"
            />
          ))}
        </div>
      }
    >
      <NanochatChartsContent {...props} />
    </Suspense>
  );
}
