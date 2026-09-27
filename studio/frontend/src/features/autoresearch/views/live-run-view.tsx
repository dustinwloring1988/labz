// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The run in progress, and the run that just finished.
 *
 * Laid out the way nanochat's live view is: a header with the stop control, a
 * progress bar, a row of numbers, the console, and the charts. The pieces differ
 * — there is no stage pipeline here, because the unit is an experiment rather
 * than a named stage — but "what state is this in" should read the same on both
 * tabs.
 */
import { useCallback, useMemo, useState } from "react";

import { ChartAverageIcon, StopCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { SectionCard } from "@/components/section-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";

import { stopAutoresearchRun } from "../api/autoresearch-api";
import {
  isAutoresearchRunActive,
  useAutoresearchRuntimeStore,
} from "../stores/autoresearch-runtime-store";
import { AutoresearchCharts } from "./autoresearch-charts";
import { AutoresearchConsole } from "./autoresearch-console";
import {
  EXPERIMENT_STATUS_LABELS,
  formatBpb,
  formatDelta,
  formatDuration,
  formatMetric,
  formatThroughput,
} from "../lib/format";
import type { AutoresearchExperimentInfo } from "../types/index.ts";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-muted text-muted-foreground",
  running: "bg-blue-500/15 text-blue-600 dark:text-blue-400",
  kept: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  discarded: "bg-muted text-muted-foreground",
  crashed: "bg-red-500/15 text-red-600 dark:text-red-400",
  stopped: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
};

export function AutoresearchLiveRunView({ onConfigure }: { onConfigure: () => void }) {
  const active = useAutoresearchRuntimeStore(isAutoresearchRunActive);
  const status = useAutoresearchRuntimeStore((state) => state.status);
  const phase = useAutoresearchRuntimeStore((state) => state.phase);
  const message = useAutoresearchRuntimeStore((state) => state.message);
  const error = useAutoresearchRuntimeStore((state) => state.error);
  const runId = useAutoresearchRuntimeStore((state) => state.runId);
  const experiment = useAutoresearchRuntimeStore((state) => state.experiment);
  const totalExperiments = useAutoresearchRuntimeStore((state) => state.totalExperiments);
  const progressPercent = useAutoresearchRuntimeStore((state) => state.progressPercent);
  const step = useAutoresearchRuntimeStore((state) => state.step);
  const loss = useAutoresearchRuntimeStore((state) => state.loss);
  const tokPerSec = useAutoresearchRuntimeStore((state) => state.tokPerSec);
  const mfu = useAutoresearchRuntimeStore((state) => state.mfu);
  const elapsedSeconds = useAutoresearchRuntimeStore((state) => state.elapsedSeconds);
  const etaSeconds = useAutoresearchRuntimeStore((state) => state.etaSeconds);
  const bestValBpb = useAutoresearchRuntimeStore((state) => state.bestValBpb);
  const baselineValBpb = useAutoresearchRuntimeStore((state) => state.baselineValBpb);
  const kept = useAutoresearchRuntimeStore((state) => state.kept);
  const discarded = useAutoresearchRuntimeStore((state) => state.discarded);
  const crashed = useAutoresearchRuntimeStore((state) => state.crashed);
  const experiments = useAutoresearchRuntimeStore((state) => state.experiments);
  const warnings = useAutoresearchRuntimeStore((state) => state.warnings);
  const series = useAutoresearchRuntimeStore((state) => state.series);
  const gitBranch = useAutoresearchRuntimeStore((state) => state.gitBranch);
  const setStopRequested = useAutoresearchRuntimeStore((state) => state.setStopRequested);

  const [stopping, setStopping] = useState(false);
  const [stopError, setStopError] = useState<string | null>(null);

  const handleStop = useCallback(async () => {
    setStopping(true);
    setStopError(null);
    // Marked before the request rather than after: the loop is cooperative, so
    // it may take a minute to actually stop, and a button that stays live that
    // long invites a second click.
    setStopRequested(true);
    try {
      await stopAutoresearchRun();
    } catch (caught) {
      setStopRequested(false);
      setStopError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setStopping(false);
    }
  }, [setStopRequested]);

  const improvement = useMemo(() => {
    if (bestValBpb === null || baselineValBpb === null) return null;
    return baselineValBpb - bestValBpb;
  }, [baselineValBpb, bestValBpb]);

  const nothingHappened =
    !runId && experiments.every((entry) => entry.status === "pending");

  if (nothingHappened) {
    return (
      <div className="flex flex-col items-center gap-3 py-16 text-center">
        <p className="text-sm text-muted-foreground">No run yet.</p>
        <p className="max-w-md text-xs text-muted-foreground/80">
          Set up a loop and it will appear here, live: which experiment is running,
          what it scored, and what the agent did.
        </p>
        <Button variant="outline" onClick={onConfigure}>
          Set one up
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 pb-8">
      <SectionCard
        icon={
          // A spinner in the icon slot rather than a static glyph: the card is
          // the one thing on the page that says whether work is happening, and
          // a still icon next to a moving progress bar reads as a stall.
          active ? (
            <Spinner className="size-5" />
          ) : (
            <HugeiconsIcon icon={ChartAverageIcon} className="size-5" />
          )
        }
        accent={active ? "blue" : "emerald"}
        title={
          active
            ? `Experiment ${experiment || "…"} of ${totalExperiments || "…"}`
            : status === "completed"
              ? "Run finished"
              : status === "stopped"
                ? "Run stopped"
                : "Run ended"
        }
        description={message || "Waiting for the loop to start"}
        badge={runId ? runId.replace(/^run_/, "") : undefined}
        headerAction={
          active ? (
            <Button
              variant="outline"
              size="sm"
              disabled={stopping}
              onClick={() => void handleStop()}
              className="gap-1.5"
            >
              <HugeiconsIcon icon={StopCircleIcon} className="size-4" />
              {stopping ? "Stopping…" : "Stop after this experiment"}
            </Button>
          ) : (
            <Button variant="secondary" size="sm" onClick={onConfigure}>
              Run another loop
            </Button>
          )
        }
      >
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Progress value={Math.round(progressPercent)} />
            <div className="flex flex-wrap justify-between gap-x-3 text-ui-11 text-muted-foreground">
              <span>
                {phase === "preparing"
                  ? "Preparing"
                  : `Experiment ${experiment} of ${totalExperiments}`}{" "}
                · {Math.round(progressPercent)}%
              </span>
              <span>
                {formatDuration(elapsedSeconds)} elapsed
                {etaSeconds !== null && active ? ` · about ${formatDuration(etaSeconds)} left in this experiment` : ""}
              </span>
            </div>
          </div>

          {stopError ? (
            <p className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-ui-12 text-destructive">
              {stopError}
            </p>
          ) : null}

          {error ? (
            <p className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-ui-12 text-destructive">
              {error}
            </p>
          ) : null}

          {warnings.length > 0 ? (
            <ul className="flex flex-col gap-1 rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-ui-11 text-amber-700 dark:text-amber-400">
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <MetricStat
              label="Best val bpb"
              value={formatBpb(bestValBpb)}
              hint={
                improvement !== null
                  ? `${formatDelta(-improvement)} against the baseline`
                  : "no scored experiment yet"
              }
              tone={improvement !== null && improvement > 0 ? "good" : undefined}
            />
            <MetricStat label="Kept" value={String(kept)} hint="lowered the score" />
            <MetricStat label="Discarded" value={String(discarded)} hint="no improvement" />
            <MetricStat label="Crashed" value={String(crashed)} hint="did not finish" />
            <MetricStat
              label="Loss"
              value={formatMetric(loss)}
              hint={step > 0 ? `step ${step}` : "waiting for a training pass"}
            />
            <MetricStat
              label="Throughput"
              value={formatThroughput(tokPerSec)}
              hint={mfu !== null ? `${formatMetric(mfu, 1)}% MFU` : "no training running"}
            />
          </div>

          {gitBranch ? (
            <p className="font-mono text-[10px] text-muted-foreground/70">
              experiment repository: {gitBranch}
            </p>
          ) : null}
        </div>
      </SectionCard>

      <AutoresearchConsole running={active} heightClassName="h-80" />

      <ExperimentStepper experiments={experiments} />

      <AutoresearchCharts
        series={series}
        baselineValBpb={baselineValBpb}
        bestValBpb={bestValBpb}
      />
    </div>
  );
}

function MetricStat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "good" | "bad";
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-xl border border-border p-3">
      <span className="text-ui-10 uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <span
        className={[
          "font-heading text-ui-20 font-semibold",
          tone === "good" ? "text-emerald-600 dark:text-emerald-400" : "",
        ].join(" ")}
      >
        {value}
      </span>
      {hint ? <span className="text-ui-10 text-muted-foreground/70">{hint}</span> : null}
    </div>
  );
}

/**
 * The experiment list.
 *
 * One row per experiment, in order, because the interesting thing about a search
 * is its sequence: which idea came after which, and what each was worth. Only
 * the running row animates, so a hundred experiments read as a list rather than
 * a hundred animations.
 */
function ExperimentStepper({
  experiments,
}: {
  experiments: AutoresearchExperimentInfo[];
}) {
  if (experiments.length === 0) return null;

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
      title="Experiments"
      description="What each one tried, and what it scored."
    >
      <ol className="flex flex-col gap-1.5">
        {experiments.map((entry) => (
          <li
            key={entry.index}
            className={[
              "flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border px-3 py-2 text-ui-12",
              entry.status === "running"
                ? "border-blue-500/40 bg-blue-500/5"
                : "border-border",
            ].join(" ")}
          >
            <span className="w-7 shrink-0 font-mono text-muted-foreground">
              {String(entry.index).padStart(2, "0")}
            </span>
            <Badge
              className={[
                "shrink-0 text-[10px]",
                STATUS_STYLES[entry.status] ?? STATUS_STYLES.pending,
              ].join(" ")}
            >
              {EXPERIMENT_STATUS_LABELS[entry.status] ?? entry.status}
            </Badge>
            <span className="min-w-0 flex-1 truncate">
              {entry.description ||
                entry.summary ||
                entry.error ||
                (entry.status === "pending" ? "not started" : entry.message)}
            </span>
            {entry.val_bpb !== null ? (
              <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
                {formatBpb(entry.val_bpb)}
              </span>
            ) : null}
            {entry.duration_seconds !== null ? (
              <span className="shrink-0 text-[10px] text-muted-foreground/70">
                {formatDuration(entry.duration_seconds)}
              </span>
            ) : null}
          </li>
        ))}
      </ol>
    </SectionCard>
  );
}
