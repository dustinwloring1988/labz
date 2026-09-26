// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Current Run tab: stage progress, live metrics, charts, samples and
 * benchmarks.
 *
 * Laid out like Train's live view rather than like a nanochat-shaped thing: the
 * progress panel is a SectionCard, the numbers are Train's KPI row, and the
 * charts are Train's chart cards. The sample feed stays beside them because
 * during fine-tuning and RL the sample is the answer to "is this working?", and
 * making the user click to see it defeats the purpose.
 */
import { useMemo } from "react";

import { HugeiconsIcon } from "@hugeicons/react";
import {
  Alert02Icon,
  ChartAverageIcon,
  CheckmarkCircle01Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";

import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

import { stopNanochatRun } from "../api/nanochat-api";
import { formatBytes, formatDuration } from "../lib/depth-presets";
import { NanochatCharts } from "../sections/nanochat-charts";
import {
  isNanochatRunActive,
  useNanochatRuntimeStore,
} from "../stores/nanochat-runtime-store";
import { BenchmarkResults, SampleFeed } from "./sample-feed";
import { NanochatConsole } from "./nanochat-console";

const STAGE_TITLES: Record<string, string> = {
  dataset: "Download data",
  tokenizer: "Train tokenizer",
  pretrain: "Pretrain",
  base_eval: "Evaluate base model",
  sft: "Fine-tune",
  chat_eval: "Evaluate chat model",
  rl: "Reinforcement learning",
};

// Same vocabulary as Train's phase pills, so "what state is this in" reads the
// same on both tabs.
const STATUS_STYLES: Record<string, string> = {
  running: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
  ok: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
  failed: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  skipped: "bg-muted text-muted-foreground",
  pending: "bg-muted text-muted-foreground",
};

const RUN_STYLES: Record<string, string> = {
  running: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
  completed: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
  error: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  stopped: "bg-muted text-muted-foreground",
  idle: "bg-muted text-muted-foreground",
};

export function NanochatLiveRunView({ onConfigure }: { onConfigure: () => void }) {
  const status = useNanochatRuntimeStore((state) => state.status);
  const currentStage = useNanochatRuntimeStore((state) => state.currentStage);
  const message = useNanochatRuntimeStore((state) => state.message);
  const error = useNanochatRuntimeStore((state) => state.error);
  const step = useNanochatRuntimeStore((state) => state.step);
  const totalSteps = useNanochatRuntimeStore((state) => state.totalSteps);
  const progressPercent = useNanochatRuntimeStore((state) => state.progressPercent);
  const loss = useNanochatRuntimeStore((state) => state.loss);
  const valBpb = useNanochatRuntimeStore((state) => state.valBpb);
  const chatcore = useNanochatRuntimeStore((state) => state.chatcore);
  const reward = useNanochatRuntimeStore((state) => state.reward);
  const tokPerSec = useNanochatRuntimeStore((state) => state.tokPerSec);
  const mfu = useNanochatRuntimeStore((state) => state.mfu);
  const peakMemory = useNanochatRuntimeStore((state) => state.peakMemoryBytes);
  const etaSeconds = useNanochatRuntimeStore((state) => state.etaSeconds);
  const elapsed = useNanochatRuntimeStore((state) => state.elapsedSeconds);
  const numParams = useNanochatRuntimeStore((state) => state.numParams);
  const stages = useNanochatRuntimeStore((state) => state.stages);
  const samples = useNanochatRuntimeStore((state) => state.samples);
  const benchmarks = useNanochatRuntimeStore((state) => state.benchmarks);
  const warnings = useNanochatRuntimeStore((state) => state.warnings);
  const series = useNanochatRuntimeStore((state) => state.series);
  const stopRequested = useNanochatRuntimeStore((state) => state.stopRequested);
  const setStopRequested = useNanochatRuntimeStore((state) => state.setStopRequested);
  const active = useNanochatRuntimeStore(isNanochatRunActive);

  const failedStage = useMemo(
    () => stages.find((stage) => stage.status === "failed"),
    [stages],
  );
  const pct = Math.round(progressPercent);

  if (stages.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 py-16 text-center">
        <p className="text-sm text-muted-foreground">No run yet.</p>
        <Button onClick={onConfigure}>Configure a run</Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <SectionCard
        icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
        title={currentStage ? (STAGE_TITLES[currentStage] ?? currentStage) : "Idle"}
        description={message || "Live metrics"}
        accent="emerald"
        className="shadow-border border border-border/60 bg-card/90 ring-0 backdrop-blur-sm"
        headerAction={
          active ? (
            <div className="flex items-center gap-2">
              {stopRequested ? (
                <span className="text-ui-11 text-muted-foreground">Stopping after this step…</span>
              ) : null}
              <Button
                size="sm"
                variant="outline"
                className="h-8 gap-1.5 text-xs"
                disabled={stopRequested}
                onClick={() => {
                  setStopRequested(true);
                  void stopNanochatRun(true).catch(() => setStopRequested(false));
                }}
              >
                {stopRequested ? "Stopping…" : "Stop and save"}
              </Button>
            </div>
          ) : null
        }
      >
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={cn(
                "rounded-full px-2.5 py-1 text-ui-10 font-semibold",
                RUN_STYLES[status] ?? RUN_STYLES.idle,
              )}
            >
              {status}
            </span>
            <span className="rounded-full border border-border/60 px-2.5 py-1 text-ui-10 font-medium tabular-nums text-foreground/80">
              {pct}% complete
            </span>
            <StageStepper stages={stages} currentStage={currentStage} />
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                step {step.toLocaleString()} / {totalSteps > 0 ? totalSteps.toLocaleString() : "--"}
              </span>
              <span>{pct}%</span>
            </div>
            <Progress
              // An indeterminate bar when the step count is unknown: a stage that
              // has not emitted a config event yet (still building the model) has
              // no denominator, and a bar pinned at 0% would read as hung.
              value={totalSteps > 0 ? progressPercent : undefined}
              className="h-2 bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]"
            />
          </div>

          {failedStage || error ? (
            <p className="rounded-2xl border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs leading-relaxed text-red-500">
              <span className="font-medium">
                {failedStage
                  ? `${STAGE_TITLES[failedStage.key] ?? failedStage.key} failed`
                  : "Run failed"}
              </span>
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-muted-foreground">
                {(failedStage?.error ?? error ?? "").slice(-2000)}
              </pre>
            </p>
          ) : null}

          {warnings.length > 0 ? (
            <div
              aria-live="polite"
              className="flex gap-2 rounded-2xl border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs leading-relaxed text-amber-700 dark:text-amber-300"
            >
              <HugeiconsIcon icon={Alert02Icon} className="mt-0.5 size-4 shrink-0" />
              <ul className="min-w-0 space-y-1">
                {warnings.map((warning) => (
                  <li key={warning} className="break-words">
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="grid gap-x-4 gap-y-3 pt-1 sm:grid-cols-2 xl:grid-cols-6">
            <MetricStat label="Loss" valueClassName="text-2xl font-bold tracking-tight">
              {loss !== null ? loss.toFixed(4) : "--"}
            </MetricStat>
            <MetricStat label="Val bpb">{valBpb !== null ? valBpb.toFixed(4) : "--"}</MetricStat>
            <MetricStat label={reward !== null ? "Reward" : "ChatCORE"}>
              {reward !== null ? reward.toFixed(3) : chatcore !== null ? chatcore.toFixed(4) : "--"}
            </MetricStat>
            <MetricStat label="Throughput">
              {tokPerSec !== null ? `${(tokPerSec / 1000).toFixed(1)}k tok/s` : "--"}
            </MetricStat>
            <MetricStat label="Parameters" valueClassName="truncate">
              {numParams ? numParams.toLocaleString() : "--"}
            </MetricStat>
            <MetricStat label="Peak memory">{formatBytes(peakMemory)}</MetricStat>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>elapsed {formatDuration(elapsed)}</span>
            {etaSeconds !== null ? <span>eta {formatDuration(etaSeconds)}</span> : null}
            {mfu !== null ? <span>{mfu.toFixed(1)}% MFU</span> : null}
          </div>
        </div>
      </SectionCard>

      <NanochatConsole running={active} />

      <NanochatCharts
        loss={series.loss ?? []}
        valBpb={series.val_bpb ?? []}
        tokPerSec={series.tok_per_sec ?? []}
        mfu={series.mfu ?? []}
        reward={series.reward ?? []}
        chatcore={series.chatcore ?? []}
        isRunning={active}
      />

      <Tabs defaultValue="samples">
        <TabsList>
          <TabsTrigger value="samples">Samples</TabsTrigger>
          <TabsTrigger value="benchmarks">Benchmarks</TabsTrigger>
        </TabsList>

        <TabsContent value="samples" className="mt-3">
          <SampleFeed
            samples={samples}
            emptyHint={
              currentStage === "pretrain"
                ? "No samples yet. Pretraining samples a few prompts every so often; the first one takes a while because it happens after a checkpoint pass."
                : "No samples yet. A few are generated at the end of the first evaluation."
            }
          />
        </TabsContent>

        <TabsContent value="benchmarks" className="mt-3">
          <BenchmarkResults benchmarks={benchmarks} />
        </TabsContent>
      </Tabs>

      {stopRequested ? (
        <p className="text-xs text-muted-foreground">
          Stopping after the current step. The checkpoint is saved, so the run can be resumed.
        </p>
      ) : null}
    </div>
  );
}

function MetricStat({
  label,
  children,
  valueClassName,
}: {
  label: string;
  children: React.ReactNode;
  valueClassName?: string;
}) {
  return (
    <div className="min-w-0">
      <p className="text-ui-11 text-muted-foreground">{label}</p>
      <p className={cn("mt-1 text-base font-semibold tabular-nums", valueClassName)}>{children}</p>
    </div>
  );
}

function StageStepper({
  stages,
  currentStage,
}: {
  stages: { key: string; status: string; error: string | null }[];
  currentStage: string | null;
}) {
  return (
    <>
      {stages.map((stage) => {
        const isCurrent = stage.key === currentStage && stage.status === "running";
        return (
          <span
            key={stage.key}
            className={cn(
              "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-ui-10 font-medium",
              isCurrent
                ? cn("ring-1 ring-inset ring-current/30", STATUS_STYLES.running)
                : STATUS_STYLES[stage.status] ?? STATUS_STYLES.pending,
            )}
          >
            {stage.status === "ok" ? (
              <HugeiconsIcon icon={CheckmarkCircle01Icon} className="size-3" />
            ) : stage.status === "running" ? (
              <HugeiconsIcon icon={PlayIcon} className="size-3" />
            ) : (
              <span className="size-1.5 rounded-full bg-current opacity-40" />
            )}
            {STAGE_TITLES[stage.key] ?? stage.key}
          </span>
        );
      })}
    </>
  );
}
