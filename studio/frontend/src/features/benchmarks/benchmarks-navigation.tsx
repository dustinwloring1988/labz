// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";
import type { BenchmarkTab } from "./benchmarks-page";

/**
 * The benchmark sub-nav.
 *
 * The Train sub-nav, not a lookalike: same unstyled list, same hidden sliding
 * indicator, same hand-drawn underline. A tab strip is the one piece of chrome
 * every page shares, and a second style for it reads as a different application.
 */
export function BenchmarkSubNav({
  value,
  runActive,
  hasRun,
}: {
  value: BenchmarkTab;
  runActive: boolean;
  hasRun: boolean;
}): ReactElement {
  const items: ReadonlyArray<{
    value: BenchmarkTab;
    label: string;
    disabled: boolean;
  }> = [
    // A run owns the GPU, so the form is not editable underneath it. Disabled
    // rather than hidden: the user can still see what the run was.
    { value: "configure", label: "Configure", disabled: runActive },
    { value: "current-run", label: "Current Run", disabled: !hasRun },
    { value: "leaderboard", label: "Leaderboard", disabled: false },
  ];
  return (
    <TabsList
      unstyled={true}
      className="flex min-w-0 flex-1 flex-wrap items-center justify-start gap-3 pb-px text-ui-13 tracking-nav sm:gap-6"
    >
      {items.map((item) => {
        const active = value === item.value;
        return (
          <TabsTrigger
            key={item.value}
            value={item.value}
            disabled={item.disabled}
            indicatorClassName="hidden"
            className={cn(
              "relative h-9 flex-none select-none rounded-none border-0 px-0 py-0 text-ui-13 transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              "after:pointer-events-none after:absolute after:inset-x-0 after:bottom-0 after:h-[2px] after:rounded-full after:bg-foreground after:transition-opacity",
              active
                ? "font-semibold text-foreground after:opacity-100"
                : "text-muted-foreground hover:text-foreground after:opacity-0",
            )}
          >
            {item.label}
          </TabsTrigger>
        );
      })}
    </TabsList>
  );
}
