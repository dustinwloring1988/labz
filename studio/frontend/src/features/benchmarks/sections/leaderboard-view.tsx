// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertCircleIcon,
  Delete02Icon,
  InformationCircleIcon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactElement,
} from "react";

import {
  deleteBenchmarkRun,
  getBenchmarkLeaderboard,
} from "../api/benchmarks-api";
import {
  formatAccuracy,
  formatCentered,
  formatDuration,
  scoringModeLabel,
} from "../lib/format";
import type {
  BenchmarkLeaderboard,
  BenchmarkSpec,
  LeaderboardRow,
} from "../types";

/** Group headings, in the order the server ranked them.
 *
 * Named because the grouping is the point of the board rather than a detail of
 * its layout: a generated-answer score and an exact one are not comparable, and
 * a published figure is not this machine's measurement.
 */
function groupOf(
  row: LeaderboardRow,
): "local-exact" | "local-generated" | "local-judged" | "reference" {
  if (row.source === "reference") return "reference";
  if (row.scoring_mode === "logits") return "local-exact";
  // Judged rows get their own group rather than joining the generated ones.
  // Both are generative and both come from this machine, but a judged score is a
  // weaker claim about the same answers -- it depends on a model being right
  // about being right -- and putting the two side by side under one heading would
  // invite exactly the comparison the ranking avoids.
  if (row.scoring_mode === "judged") return "local-judged";
  return "local-generated";
}

const GROUP_LABELS: Record<
  ReturnType<typeof groupOf>,
  { title: string; note: string }
> = {
  "local-exact": {
    title: "Measured here",
    note: "Scored from the model's own logits, narrowed to the letters each question offers. The same checkpoint gives the same number twice.",
  },
  "local-generated": {
    title: "Measured here, by generated answer",
    note: "The model was asked to answer and the answer was read back. A weaker measurement of the same thing, so it is kept in its own group rather than ranked against the exact figures above.",
  },
  "local-judged": {
    title: "Measured here, judged by a second model",
    note: "A separate model read each generated answer and decided whether it was right, which forgives a correct answer the exact grader scored zero for not ending in '#### N'. It depends on that model being right about being right, so it sits below every exact figure.",
  },
  reference: {
    title: "Published reference scores",
    note: "Measured by somebody else's harness, on hardware that is not this machine, with their own prompt rendering and shot count. Useful as an anchor; not a like-for-like comparison.",
  },
};

function Cell({
  value,
  truncated,
}: {
  value: number | null;
  truncated: boolean;
}): ReactElement {
  if (value === null) {
    return <span className="text-muted-foreground/60">—</span>;
  }
  return (
    <span
      className={cn(
        "font-mono tabular-nums",
        truncated && "text-muted-foreground",
      )}
    >
      {formatAccuracy(value)}
      {/* A sampled figure is marked in place rather than only in the row, because
          the reader comparing two cells needs to know this one is from fewer
          problems than the other. */}
      {truncated ? <span className="ml-0.5 text-[10px]">*</span> : null}
    </span>
  );
}

function Row({
  row,
  sortKeys,
  rank,
  onDelete,
}: {
  row: LeaderboardRow;
  sortKeys: readonly string[];
  rank: number;
  onDelete: (runId: string) => void;
}): ReactElement {
  return (
    <tr className="border-b border-border/50 last:border-0">
      <td className="w-8 py-2.5 pr-2 text-ui-11 tabular-nums text-muted-foreground">
        {rank}
      </td>
      <td className="py-2.5 pr-3">
        <div className="flex min-w-0 flex-col gap-0.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <span
              className="truncate text-sm text-foreground"
              title={row.model_id}
            >
              {row.model_label}
            </span>
            {row.scoring_mode && row.scoring_mode !== "logits" ? (
              <Badge variant="outline" className="shrink-0 text-[10px]">
                {scoringModeLabel(row.scoring_mode)}
              </Badge>
            ) : null}
            {row.truncated_keys.length > 0 ? (
              <Badge variant="outline" className="shrink-0 text-[10px]">
                sampled
              </Badge>
            ) : null}
          </div>
          {row.source === "reference" && row.source_note ? (
            <span className="text-[11px] leading-tight text-muted-foreground">
              {row.source_note}
            </span>
          ) : row.source === "local" ? (
            <span className="text-[11px] text-muted-foreground">
              {row.benchmarks_count} of {sortKeys.length} benchmarks
              {row.max_problems
                ? ` · up to ${row.max_problems.toLocaleString()} problems each`
                : ""}
              {row.duration_seconds
                ? ` · ${formatDuration(row.duration_seconds)}`
                : ""}
            </span>
          ) : null}
        </div>
      </td>
      {sortKeys.map((key) => (
        <td key={key} className="px-2 py-2.5 text-right text-ui-12">
          <Cell
            value={row.scores[key] ?? null}
            truncated={row.truncated_keys.includes(key)}
          />
        </td>
      ))}
      <td className="px-2 py-2.5 text-right">
        <span className="font-mono text-ui-12 tabular-nums text-foreground">
          {formatCentered(row.composite)}
        </span>
      </td>
      <td className="w-10 py-2.5 pl-2 text-right">
        {row.source === "local" && row.run_id ? (
          <button
            type="button"
            aria-label={`Remove ${row.model_label} from the leaderboard`}
            onClick={() => onDelete(row.run_id as string)}
            className="rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive"
          >
            <HugeiconsIcon
              icon={Delete02Icon}
              className="size-3.5"
              strokeWidth={1.75}
            />
          </button>
        ) : null}
      </td>
    </tr>
  );
}

export function BenchmarkLeaderboardView({
  specs,
  onRunAgain,
}: {
  specs: readonly BenchmarkSpec[];
  onRunAgain: () => void;
}): ReactElement {
  const [board, setBoard] = useState<BenchmarkLeaderboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [includeReferences, setIncludeReferences] = useState(true);
  // Bumped to re-read the board. A counter rather than a reload function passed
  // through the effect's dependencies, so changing the toggle and deleting a run
  // both go through one path.
  const [revision, setRevision] = useState(0);

  // No synchronous setState: the board is absent until the first read lands, and
  // that absence is the loading state. Setting a flag first would mean a second
  // render before anything is on screen.
  useEffect(() => {
    let cancelled = false;
    void getBenchmarkLeaderboard({ includeReferences })
      .then((next) => {
        if (cancelled) return;
        setBoard(next);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => {
      cancelled = true;
    };
  }, [includeReferences, revision]);

  const handleDelete = useCallback(async (runId: string) => {
    try {
      await deleteBenchmarkRun(runId);
      setRevision((value) => value + 1);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  // The server projects onto every benchmark in its catalogue, so the columns are
  // whatever it decided the board is about rather than a second list to keep in
  // step here. The specs are the fallback for the frame before the first read.
  const sortKeys = useMemo(() => {
    if (board && board.sort_keys.length > 0) return board.sort_keys;
    return specs.map((spec) => spec.key);
  }, [board, specs]);

  const grouped = useMemo(() => {
    const rows = board?.rows ?? [];
    // The order here mirrors the server's ranking: exact, then the two weaker
    // generative measurements, then other people's numbers. Judged sits below
    // generated because it is the weaker of the two.
    const order: Array<ReturnType<typeof groupOf>> = [
      "local-exact",
      "local-generated",
      "local-judged",
      "reference",
    ];
    return order
      .map((key) => ({ key, rows: rows.filter((row) => groupOf(row) === key) }))
      .filter((group) => group.rows.length > 0);
  }, [board]);

  const hasLocal = (board?.rows ?? []).some((row) => row.source === "local");

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-ui-15 font-semibold text-foreground">
            Leaderboard
          </h2>
          <p className="text-sm text-muted-foreground">
            Every model you have benchmarked on this machine, best score per
            model.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-ui-12 text-muted-foreground">
            <Switch
              checked={includeReferences}
              onCheckedChange={setIncludeReferences}
            />
            Show published scores
          </label>
        </div>
      </div>

      {/* A board with one row is not a leaderboard, so the published rows are on
          by default and the reason they are marked is stated rather than left to
          be discovered. */}
      <p className="flex items-start gap-2 rounded-2xl bg-muted/40 px-3.5 py-2.5 text-xs text-muted-foreground">
        <HugeiconsIcon
          icon={InformationCircleIcon}
          className="mt-px size-3.5 shrink-0"
        />
        <span>
          &ldquo;Over guessing&rdquo; is the share of the gap between random
          guessing and a perfect score that was closed, which is the only figure
          comparable across benchmarks. An asterisk marks a score taken from
          fewer problems than the cap allows.
        </span>
      </p>

      {error ? (
        <p className="flex items-start gap-2 rounded-2xl bg-destructive/10 px-3.5 py-3 text-sm text-destructive">
          <HugeiconsIcon
            icon={AlertCircleIcon}
            className="mt-px size-4 shrink-0"
          />
          {error}
        </p>
      ) : null}

      {!board && !error ? (
        <p className="text-sm text-muted-foreground">
          Loading the leaderboard…
        </p>
      ) : grouped.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-3xl border border-dashed border-border/70 px-6 py-12 text-center">
          <HugeiconsIcon
            icon={StarIcon}
            className="size-8 text-muted-foreground/60"
            strokeWidth={1.5}
          />
          <div className="flex flex-col gap-1">
            <p className="text-sm font-medium text-foreground">
              Nothing benchmarked yet
            </p>
            <p className="max-w-md text-sm text-muted-foreground">
              {includeReferences
                ? "No published scores match these benchmarks either. Run a benchmark and the first result lands here."
                : "Run a benchmark and the first result lands here."}
            </p>
          </div>
          <button
            type="button"
            onClick={onRunAgain}
            className="mt-1 flex h-9 items-center rounded-full bg-foreground px-4 text-ui-13 font-medium text-background transition-opacity hover:opacity-90"
          >
            Configure a run
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-8">
          {grouped.map((group) => (
            <section key={group.key} className="flex flex-col gap-2">
              <div className="flex flex-col gap-0.5">
                <h3 className="text-ui-13 font-semibold text-foreground">
                  {GROUP_LABELS[group.key].title}
                </h3>
                <p className="text-xs text-muted-foreground">
                  {GROUP_LABELS[group.key].note}
                </p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b border-border/70">
                      <th className="w-8 py-2 pr-2" />
                      <th className="py-2 pr-3 text-left text-ui-11 font-medium text-muted-foreground">
                        Model
                      </th>
                      {sortKeys.map((key) => (
                        <th
                          key={key}
                          className="px-2 py-2 text-right text-ui-11 font-medium text-muted-foreground"
                        >
                          {specs.find((spec) => spec.key === key)?.label ?? key}
                        </th>
                      ))}
                      <th className="px-2 py-2 text-right text-ui-11 font-medium text-muted-foreground">
                        Over guessing
                      </th>
                      <th className="w-10 py-2 pl-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {group.rows.map((row, index) => (
                      <Row
                        key={row.run_id ?? row.model_id}
                        row={row}
                        sortKeys={sortKeys}
                        rank={index + 1}
                        onDelete={handleDelete}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ))}
        </div>
      )}

      {hasLocal ? (
        <p className="text-xs text-muted-foreground">
          A model appears once, with its best run. Re-running it with a larger
          problem cap replaces its row.
        </p>
      ) : null}
    </div>
  );
}
