// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AlertCircleIcon, StopCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { useEffect, useRef, type ReactElement } from "react";

import {
  formatAccuracy,
  formatCentered,
  formatJudgeCoverage,
  formatDuration,
  scoringModeLabel,
} from "../lib/format";
import type {
  BenchmarkLogLine,
  BenchmarkScore,
  BenchmarkStatus,
} from "../types";

/** The three passes, in the order they run, with what each is doing.
 *
 * Named rather than showing a raw phase string because "materialize" is an
 * implementation detail and "downloading the benchmark questions" is what is
 * actually happening for the several minutes it takes.
 */
const PHASES: ReadonlyArray<{ key: string; label: string; detail: string }> = [
  {
    key: "materialize",
    label: "Preparing",
    detail: "Fetching the benchmark questions. Cached after the first run.",
  },
  {
    key: "score",
    label: "Scoring",
    detail: "Running the model over each benchmark.",
  },
  {
    key: "grade",
    label: "Grading",
    detail: "Comparing each answer against the reference.",
  },
  {
    key: "judge",
    label: "Judging",
    detail:
      "A second model reading each generated answer. Runs only when a judge was chosen.",
  },
];

function phaseIndex(phase: string): number {
  return PHASES.findIndex((entry) => entry.key === phase);
}

/**
 * One line describing what the judge made of this benchmark, or null for a score
 * it never looked at.
 *
 * Written as prose rather than as two numbers because the interesting cases are
 * the ones a bare figure hides: that the judge disagreed with the exact grader
 * and in which direction, and that it only read part of the suite. A row reading
 * "12.5%" beside an exact "0%" is the finding, and it has to survive being
 * scanned quickly.
 */
function describeJudge(score: BenchmarkScore): string | null {
  if (score.judge_accuracy === null) return null;
  const parts: string[] = [];
  parts.push(`judged ${formatAccuracy(score.judge_accuracy)}`);

  // The exact grade is only worth naming when the two differ; otherwise it is
  // the same number twice.
  if (
    score.exact_accuracy !== null &&
    score.exact_accuracy !== score.judge_accuracy
  ) {
    const direction =
      score.judge_accuracy > score.exact_accuracy ? "higher" : "lower";
    parts.push(`exact grader said ${formatAccuracy(score.exact_accuracy)}`);
    if (score.disagreements > 0) {
      parts.push(
        `${score.disagreements} disagreement${score.disagreements === 1 ? "" : "s"}`,
      );
    }
    if (direction === "higher" && score.judge_raised > 0) {
      parts.push(
        `${score.judge_raised} raised by the judge, meaning the answer was right but not written as the grader looks for`,
      );
    }
  }

  const coverage = formatJudgeCoverage(score.judge_judged, score.judge_unreadable);
  if (coverage) parts.push(coverage);
  return parts.join(" · ");
}

function ScoreRow({
  score,
  active,
}: { score: BenchmarkScore; active: boolean }): ReactElement {
  const done = score.status === "complete";
  const failed = score.status === "failed" || score.status === "skipped";
  const judgeLine = describeJudge(score);

  return (
    <li
      className={cn(
        "flex items-center gap-4 rounded-2xl px-3.5 py-3",
        active ? "bg-muted/60" : "bg-muted/25",
      )}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-foreground">
            {score.label || score.key}
          </span>
          {active ? (
            <Badge variant="secondary" className="text-[10px]">
              running
            </Badge>
          ) : null}
          {score.truncated ? (
            // Said on the row rather than only in a tooltip: a score from a capped
            // run is an estimate, and a reader comparing it with a full-set run
            // needs to know without hunting for the caveat.
            <Badge variant="outline" className="text-[10px]">
              sampled
            </Badge>
          ) : null}
        </div>
        <p className="text-xs text-muted-foreground">
          {failed && score.error
            ? score.error
            : done
              ? `${score.correct} of ${score.total} correct · ${formatDuration(score.elapsed_seconds)}`
              : score.status === "pending"
                ? "Waiting"
                : "In progress"}
        </p>
        {judgeLine ? (
          // Both figures, on the row, rather than one figure and a tooltip. The
          // whole value of a judge is the comparison: where it disagrees with the
          // exact grader it is saying the answer was right but not written the
          // way the grader looks for, and hiding the exact number would hide the
          // finding.
          <p className="text-xs text-muted-foreground">
            {judgeLine}
          </p>
        ) : null}
      </div>
      <div className="flex shrink-0 flex-col items-end gap-0.5">
        <span
          className={cn(
            "font-mono text-ui-15 tabular-nums",
            done ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {formatAccuracy(score.accuracy)}
        </span>
        {/* The centred figure is the one that can be compared with another
            benchmark, so it is shown next to the raw one rather than instead of
            it: the raw number is what a reader recognises, the centred one is what
            can be averaged. */}
        {done && !score.truncated ? (
          <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
            {formatCentered(score.centered)} over guessing
          </span>
        ) : null}
      </div>
    </li>
  );
}

function Console({
  lines,
}: { lines: readonly BenchmarkLogLine[] }): ReactElement {
  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [lines.length]);

  return (
    <div className="max-h-64 overflow-y-auto rounded-2xl bg-muted/40 p-3 font-mono text-ui-11 leading-ui-15">
      {lines.length === 0 ? (
        <p className="text-muted-foreground">Waiting for output…</p>
      ) : (
        lines.map((line) => (
          <p
            key={line.seq}
            className={cn(
              "break-all whitespace-pre-wrap",
              line.stream === "stderr"
                ? "text-destructive"
                : "text-muted-foreground",
            )}
          >
            {line.line}
          </p>
        ))
      )}
      <div ref={endRef} />
    </div>
  );
}

export function BenchmarkLiveRunView({
  status,
  onStop,
  onConfigure,
}: {
  status: BenchmarkStatus;
  onStop: () => void;
  onConfigure: () => void;
}): ReactElement {
  const active = status.status === "running";
  // The judge step is listed only when this run has a judge. A run configured
  // without one would otherwise show four steps and finish after three, which
  // reads as a failure rather than as the configuration it was.
  const hasJudge = status.judge_model_label !== null;
  const visiblePhases = hasJudge ? PHASES : PHASES.filter((p) => p.key !== "judge");
  const current = phaseIndex(status.phase);
  const currentVisible = visiblePhases.findIndex((p) => p.key === status.phase);
  const completed = status.scores.filter(
    (score) => score.status === "complete",
  ).length;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-ui-15 font-semibold text-foreground">
            {status.model_label}
          </h2>
          <p className="text-sm text-muted-foreground">
            {active
              ? status.message || "Running"
              : status.status === "completed"
                ? `Finished in ${formatDuration(status.elapsed_seconds)}`
                : status.status === "stopped"
                  ? `Stopped after ${formatDuration(status.elapsed_seconds)}`
                  : status.error || "Not running"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {status.scoring_mode ? (
            <Badge variant="outline" className="text-[10px]">
              {scoringModeLabel(status.scoring_mode)}
            </Badge>
          ) : null}
          {hasJudge && status.judge_model_label ? (
            // Which model judged, on the run rather than buried in a per-benchmark
            // row: it is a property of the whole run, and a score without knowing
            // what read it is the thing this feature exists to make visible.
            <Badge variant="outline" className="max-w-56 truncate text-[10px]">
              judged by {status.judge_model_label}
            </Badge>
          ) : null}
          {status.composite !== null ? (
            <Badge variant="secondary" className="text-[10px]">
              {formatCentered(status.composite)} over guessing
            </Badge>
          ) : null}
          {active ? (
            <button
              type="button"
              onClick={onStop}
              className="flex h-8 items-center gap-1.5 rounded-full border border-border/70 px-3 text-ui-12 text-foreground transition-colors hover:bg-muted/60"
            >
              <HugeiconsIcon
                icon={StopCircleIcon}
                className="size-3.5"
                strokeWidth={1.75}
              />
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={onConfigure}
              className="flex h-8 items-center rounded-full border border-border/70 px-3 text-ui-12 text-foreground transition-colors hover:bg-muted/60"
            >
              Run another
            </button>
          )}
        </div>
      </div>

      {/* The passes, so a run that has been "Scoring" for eight minutes
          reads as progress through a known sequence rather than as a stall.
          The judging step is only listed when there is a judge to do it: a step
          that is always skipped would teach the reader to distrust the list. */}
      <ol className="flex flex-col gap-2">
        {visiblePhases.map((phase, index) => {
          const state =
            status.status === "completed" || current > index
              ? "done"
              : index === currentVisible
                ? "active"
                : "pending";          return (
            <li
              key={phase.key}
              className={cn(
                "flex items-start gap-3 rounded-2xl px-3.5 py-2.5",
                state === "active" ? "bg-muted/60" : "bg-muted/20",
              )}
            >
              <span
                className={cn(
                  "mt-1 size-2 shrink-0 rounded-full",
                  state === "done"
                    ? "bg-foreground"
                    : state === "active"
                      ? "animate-pulse bg-control-accent"
                      : "bg-muted-foreground/40",
                )}
              />
              <span className="flex min-w-0 flex-col gap-0.5">
                <span
                  className={cn(
                    "text-ui-13 font-medium",
                    state === "pending"
                      ? "text-muted-foreground"
                      : "text-foreground",
                  )}
                >
                  {phase.label}
                </span>
                <span className="text-xs text-muted-foreground">
                  {phase.detail}
                </span>
              </span>
            </li>
          );
        })}
      </ol>

      <div className="flex flex-col gap-2">
        <Progress value={status.progress_percent} className="h-1.5" />
        <div className="flex items-baseline justify-between gap-3 text-xs text-muted-foreground">
          <span>
            {completed} of {status.requested.length} benchmarks scored
          </span>
          <span className="font-mono tabular-nums">
            {formatDuration(status.elapsed_seconds)}
          </span>
        </div>
      </div>

      {status.error ? (
        <p className="flex items-start gap-2 rounded-2xl bg-destructive/10 px-3.5 py-3 text-sm text-destructive">
          <HugeiconsIcon
            icon={AlertCircleIcon}
            className="mt-px size-4 shrink-0"
          />
          {status.error}
        </p>
      ) : null}

      <div className="flex flex-col gap-6 @3xl:grid @3xl:grid-cols-2 @3xl:gap-8">
        <section className="flex min-w-0 flex-col gap-2.5">
          <h3 className="text-ui-13 font-semibold text-foreground">Scores</h3>
          {status.scores.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No benchmarks have run yet.
            </p>
          ) : (
            <ul className="flex flex-col gap-1.5">
              {status.scores.map((score) => (
                <ScoreRow
                  key={score.key}
                  score={score}
                  active={score.key === status.current}
                />
              ))}
            </ul>
          )}
        </section>

        <section className="flex min-w-0 flex-col gap-2.5">
          <h3 className="text-ui-13 font-semibold text-foreground">Console</h3>
          <Console lines={status.logs} />
        </section>
      </div>
    </div>
  );
}
