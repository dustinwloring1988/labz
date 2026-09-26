// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The sample feed: what the model is producing, live.
 *
 * This is the part that makes the run legible. Numbers say a loss is falling; a
 * sample says whether the model has learned anything, and during fine-tuning and
 * RL it is the only view that shows the thing being optimised.
 *
 * The three sample kinds are drawn differently, because they mean different
 * things:
 *
 *   completion  a base model continuing text. Not a conversation, and rendering
 *               it as one would invent a reply the model never gave.
 *   chat        a real turn after fine-tuning. Drawn as a conversation.
 *   rollout     an RL sample, drawn as a conversation with its reward, because
 *               during RL the reward is the signal being optimised.
 */
import { HugeiconsIcon } from "@hugeicons/react";
import { Alert02Icon, DiceIcon, Message01Icon } from "@hugeicons/core-free-icons";
import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import type { NanochatSample } from "../types";

const STAGE_LABELS: Record<string, string> = {
  pretrain: "Pretraining",
  base_eval: "Base evaluation",
  sft: "Fine-tuning",
  rl: "Reinforcement learning",
};

export function SampleFeed({
  samples,
  className,
  emptyHint,
}: {
  samples: NanochatSample[];
  className?: string;
  emptyHint?: string;
}) {
  // Default to whichever stage has samples, so the feed is never showing an empty
  // tab while another one is full.
  const stages = useMemo(() => {
    const present = Array.from(new Set(samples.map((sample) => sample.stage)));
    return present.length > 0 ? present : ["pretrain", "sft", "rl"];
  }, [samples]);

  const [stage, setStage] = useState<string | null>(null);
  const activeStage = stage && stages.includes(stage) ? stage : (stages[0] ?? "pretrain");
  const visible = useMemo(
    () => samples.filter((sample) => sample.stage === activeStage).slice(-60).reverse(),
    [samples, activeStage],
  );

  if (samples.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border p-8 text-center",
          className,
        )}
      >
        <p className="text-xs text-muted-foreground">
          {emptyHint ??
            "No samples yet. The model generates a few every so often so you can see what it is learning."}
        </p>
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {stages.length > 1 ? (
        <Tabs value={activeStage} onValueChange={setStage}>
          <TabsList>
            {stages.map((key) => (
              <TabsTrigger key={key} value={key}>
                {STAGE_LABELS[key] ?? key}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      ) : null}

      <div className="flex max-h-[calc(32rem*var(--ui-space-scale,1))] flex-col gap-2 overflow-y-auto pr-1">
        {visible.length === 0 ? (
          <p className="py-6 text-center text-xs text-muted-foreground">
            Nothing sampled yet in this stage.
          </p>
        ) : (
          visible.map((sample, index) => (
            <SampleCard key={`${sample.stage}-${sample.step}-${index}`} sample={sample} />
          ))
        )}
      </div>
    </div>
  );
}

function SampleCard({ sample }: { sample: NanochatSample }) {
  if (sample.mode === "unconditioned") {
    return (
      <div className="flex flex-col gap-1 rounded-lg border border-border bg-muted/20 p-2.5">
        <div className="flex items-center gap-1.5">
          <HugeiconsIcon icon={DiceIcon} className="size-3 text-muted-foreground" />
          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
            unconditioned
          </span>
          <span className="text-[10px] text-muted-foreground/60">step {sample.step}</span>
        </div>
        <p className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">
          {sample.completion}
        </p>
      </div>
    );
  }

  if (sample.mode === "completion") {
    return (
      <div className="flex flex-col gap-1 rounded-lg border border-border bg-muted/20 p-2.5">
        <div className="flex items-center gap-1.5">
          <Badge variant="outline" className="text-[10px]">
            base model
          </Badge>
          <span className="text-[10px] text-muted-foreground/60">step {sample.step}</span>
        </div>
        <p className="text-xs text-muted-foreground">{sample.prompt}</p>
        <p className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">
          {sample.completion}
        </p>
      </div>
    );
  }

  // chat and rollout both render as a conversation.
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border p-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        {sample.mode === "rollout" ? (
          <Badge
            variant="outline"
            className={cn(
              "text-[10px]",
              (sample.reward ?? 0) > 0
                ? "border-emerald-500/40 text-emerald-600 dark:text-emerald-400"
                : "text-muted-foreground",
            )}
          >
            reward {(sample.reward ?? 0).toFixed(2)}
          </Badge>
        ) : null}
        {sample.advantage !== null && sample.advantage !== undefined ? (
          <span className="text-[10px] text-muted-foreground/60">
            advantage {sample.advantage >= 0 ? "+" : ""}
            {sample.advantage.toFixed(3)}
          </span>
        ) : null}
        <span className="text-[10px] text-muted-foreground/60">step {sample.step}</span>
      </div>
      <div className="flex items-start gap-1.5 text-xs">
        <HugeiconsIcon icon={Message01Icon} className="mt-0.5 size-3 shrink-0 text-muted-foreground" />
        <span className="text-foreground/90">{sample.prompt}</span>
      </div>
      <div className="whitespace-pre-wrap break-words border-l-2 border-border pl-2.5 font-mono text-[11px] leading-relaxed">
        {sample.completion}
      </div>
    </div>
  );
}

/**
 * Benchmarks collected so far.
 *
 * A partial selection is labelled, because a centred mean over one task is not
 * the published ChatCORE number and reading it as one would be wrong in the
 * direction that flatters the model.
 */
export function BenchmarkResults({
  benchmarks,
}: {
  benchmarks: {
    stage: string;
    name: string;
    accuracy: number | null;
    baseline: number | null;
    centered: number | null;
    value: number | null;
    is_partial: boolean;
  }[];
}) {
  if (benchmarks.length === 0) {
    return (
      <p className="py-6 text-center text-xs text-muted-foreground">
        No benchmarks have run yet.
      </p>
    );
  }

  // Accuracy rows are one-per-task; scalar rows (val bpb, ChatCORE) are their own
  // thing and would read as noise mixed in with the per-task table.
  const accuracies = benchmarks.filter((row) => row.accuracy !== null);
  const scalars = benchmarks.filter((row) => row.accuracy === null);

  return (
    <div className="flex flex-col gap-4">
      {accuracies.length > 0 ? (
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="pb-1.5 font-medium">Benchmark</th>
              <th className="pb-1.5 text-right font-medium">Accuracy</th>
              <th className="pb-1.5 text-right font-medium">Chance</th>
              <th className="pb-1.5 text-right font-medium">Above chance</th>
            </tr>
          </thead>
          <tbody>
            {accuracies.map((row, index) => (
              <tr key={`${row.stage}-${row.name}-${index}`} className="border-b border-border/40">
                <td className="py-1.5">{row.name}</td>
                <td className="py-1.5 text-right font-mono">
                  {row.accuracy !== null ? `${(row.accuracy * 100).toFixed(2)}%` : "—"}
                </td>
                <td className="py-1.5 text-right font-mono text-muted-foreground">
                  {row.baseline !== null ? `${(row.baseline * 100).toFixed(0)}%` : "—"}
                </td>
                <td
                  className={cn(
                    "py-1.5 text-right font-mono",
                    (row.centered ?? 0) > 0
                      ? "text-emerald-600 dark:text-emerald-400"
                      : "text-muted-foreground",
                  )}
                >
                  {row.centered !== null ? row.centered.toFixed(3) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}

      {scalars.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          {scalars.slice(-8).map((row, index) => (
            <div
              key={`${row.name}-${index}`}
              className="flex items-center justify-between gap-2 text-xs"
            >
              <span className="flex items-center gap-1.5">
                {row.name}
                {row.is_partial ? (
                  <span className="flex items-center gap-1 text-[10px] text-amber-600 dark:text-amber-400">
                    <HugeiconsIcon icon={Alert02Icon} className="size-3" />
                    partial selection, not comparable to the full number
                  </span>
                ) : null}
              </span>
              <span className="font-mono">
                {row.value !== null ? row.value.toFixed(4) : "—"}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
