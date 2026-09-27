// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The whole search, as the ledger records it.
 *
 * Read from `results.tsv` rather than from run memory, so this is the same
 * history analysis.ipynb reads, it survives a restart, and it includes rows the
 * agent added by hand. That is the point of the tab: a loop is one night, the
 * ledger is every night.
 */
import { useMemo } from "react";

import { CpuIcon, SparklesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { SectionCard } from "@/components/section-card";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";

import { useAutoresearchRuntimeStore } from "../stores/autoresearch-runtime-store";
import { AutoresearchCharts } from "./autoresearch-charts";
import { formatBpb, formatDelta } from "../lib/format";
import type { AutoresearchResultRow } from "../types/index.ts";

const STATUS_STYLES: Record<string, string> = {
  keep: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  discard: "bg-muted text-muted-foreground",
  crash: "bg-red-500/15 text-red-600 dark:text-red-400",
};

export function AutoresearchResultsView() {
  const rows = useAutoresearchRuntimeStore((state) => state.results);
  const loaded = useAutoresearchRuntimeStore((state) => state.resultsLoaded);
  const totals = useAutoresearchRuntimeStore((state) => state.resultsTotals);
  const series = useAutoresearchRuntimeStore((state) => state.series);

  const improvement = useMemo(() => {
    if (totals.bestValBpb === null || totals.baselineValBpb === null) return null;
    return totals.baselineValBpb - totals.bestValBpb;
  }, [totals.baselineValBpb, totals.bestValBpb]);

  if (!loaded) {
    return (
      <div className="flex justify-center py-12">
        <Spinner />
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-16 text-center">
        <p className="text-sm text-muted-foreground">No experiments recorded yet.</p>
        <p className="max-w-md text-xs text-muted-foreground/80">
          Every experiment the agent runs is appended to results.tsv in the experiment
          workspace, whether it helped, did not, or crashed. This is that file.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 pb-8">
      <SectionCard
        icon={<HugeiconsIcon icon={SparklesIcon} className="size-5" />}
        title={`${totals.total} experiment${totals.total === 1 ? "" : "s"}`}
        description={
          totals.bestValBpb === null
            ? "Nothing has produced a score yet."
            : `Best val bpb ${formatBpb(totals.bestValBpb)} from experiment ${totals.bestIndex}.`
        }
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Stat label="Baseline" value={formatBpb(totals.baselineValBpb)} />
          <Stat label="Best" value={formatBpb(totals.bestValBpb)} />
          <Stat
            label="Improvement"
            value={improvement === null ? "—" : formatDelta(-improvement)}
            tone={improvement !== null && improvement > 0 ? "good" : undefined}
          />
          <Stat label="Kept" value={String(totals.kept)} />
          <Stat label="Crashed" value={String(totals.crashed)} />
        </div>
      </SectionCard>

      <AutoresearchCharts
        series={series}
        baselineValBpb={totals.baselineValBpb}
        bestValBpb={totals.bestValBpb}
      />

      <SectionCard
        icon={<HugeiconsIcon icon={CpuIcon} className="size-5" />}
        title="The ledger"
        description="results.tsv, one row per experiment, in the order they ran."
      >
        <div className="-mx-2 overflow-x-auto px-2">
          {/* The ledger's own minimum: below this the description column starts
              wrapping mid-word and the commit column stops being scannable. A
              spacing-scale step rather than a hand-set length, so it tracks the
              UI font size instead of fighting it. */}
          <table className="w-full min-w-160 border-collapse text-ui-12">
            <thead>
              <tr className="border-b border-border text-left text-ui-10 uppercase tracking-wide text-muted-foreground">
                <th className="py-2 pr-3 font-medium">#</th>
                <th className="py-2 pr-3 font-medium">val bpb</th>
                <th className="py-2 pr-3 font-medium">VRAM</th>
                <th className="py-2 pr-3 font-medium">Status</th>
                <th className="py-2 pr-3 font-medium">Commit</th>
                <th className="py-2 font-medium">What changed</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <Row key={row.index} row={row} />
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "good";
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-xl border border-border p-3">
      <span className="text-ui-10 uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <span
        className={[
          "font-heading text-ui-18 font-semibold",
          tone === "good" ? "text-emerald-600 dark:text-emerald-400" : "",
        ].join(" ")}
      >
        {value}
      </span>
    </div>
  );
}

function Row({ row }: { row: AutoresearchResultRow }) {
  return (
    <tr
      className={[
        "border-b border-border/50 align-top",
        row.is_best ? "bg-emerald-500/5" : "",
      ].join(" ")}
    >
      <td className="py-2 pr-3 font-mono text-muted-foreground">{row.index}</td>
      <td className="py-2 pr-3 font-mono">
        {row.val_bpb === null ? (
          <span className="text-muted-foreground/60">—</span>
        ) : (
          <>
            {formatBpb(row.val_bpb)}
            {row.is_best ? (
              <Badge className="ml-1.5 bg-emerald-500/15 text-[10px] text-emerald-600 dark:text-emerald-400">
                best
              </Badge>
            ) : null}
          </>
        )}
      </td>
      <td className="py-2 pr-3 font-mono text-muted-foreground">
        {row.memory_gb === null ? "—" : `${row.memory_gb.toFixed(2)} GB`}
      </td>
      <td className="py-2 pr-3">
        <Badge
          className={[
            "text-[10px]",
            STATUS_STYLES[row.status] ?? STATUS_STYLES.discard,
          ].join(" ")}
        >
          {row.status}
        </Badge>
      </td>
      <td className="py-2 pr-3 font-mono text-[10px] text-muted-foreground/70">
        {row.commit || "—"}
      </td>
      <td className="py-2">
        {row.description || (
          <span className="text-muted-foreground/70">
            {row.error ?? "not described"}
          </span>
        )}
        {row.checkpoint ? (
          <span className="ml-1.5 font-mono text-[10px] text-muted-foreground/50">
            {row.checkpoint}.pt
          </span>
        ) : null}
      </td>
    </tr>
  );
}
