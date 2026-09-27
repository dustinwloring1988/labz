// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The sub-tab strip.
 *
 * Hand-drawn rather than the shadcn TabsList, for the same reason Train's and
 * nanochat's are: the underline has to sit on the control-accent colour and
 * follow the active palette, which is a few lines of CSS and considerably less
 * than convincing the component library to do it.
 */
import { TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

export type AutoresearchTab =
  | "configure"
  | "current-run"
  | "results"
  | "checkpoints"
  | "report";

export function AutoresearchSubNav({
  value,
  runActive,
  showRunView,
  hasResults,
  hasCheckpoints,
  hasReport,
}: {
  value: AutoresearchTab;
  runActive: boolean;
  showRunView: boolean;
  hasResults: boolean;
  hasCheckpoints: boolean;
  hasReport: boolean;
}) {
  // A tab that has nothing behind it is shown disabled rather than hidden, so
  // the shape of the feature is visible before the user has got there. The
  // exception is the ones that would be pure noise before any run exists: the
  // report needs a ledger, and the models need a loop.
  const items: ReadonlyArray<{
    value: AutoresearchTab;
    label: string;
    disabled: boolean;
  }> = [
    { value: "configure", label: "Configure", disabled: runActive },
    { value: "current-run", label: "Current Run", disabled: !showRunView },
    { value: "results", label: "Results", disabled: !hasResults },
    { value: "checkpoints", label: "Models", disabled: !hasCheckpoints },
    { value: "report", label: "Report", disabled: !hasReport },
  ];

  return (
    <TabsList unstyled className="flex min-w-0 items-center gap-1 overflow-x-auto">
      {items.map((item) => (
        <TabsTrigger
          key={item.value}
          value={item.value}
          disabled={item.disabled}
          className={cn(
            "relative shrink-0 px-3 py-2 text-ui-13 transition-colors",
            "after:absolute after:inset-x-2 after:-bottom-px after:h-0.5 after:rounded-full after:transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            value === item.value
              ? "font-medium text-foreground after:bg-control-accent"
              : "text-muted-foreground after:bg-transparent hover:text-foreground",
            item.disabled && "pointer-events-none opacity-40",
          )}
        >
          {item.label}
        </TabsTrigger>
      ))}
    </TabsList>
  );
}
