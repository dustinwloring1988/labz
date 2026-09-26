// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Depth picker.
 *
 * Two things this deliberately shows that a plain "parameters" number would not:
 *
 *  - transformer vs embedding parameters. At a 32768 vocab the embedding tables
 *    outnumber the transformer matrices until about depth 25, so quoting the total
 *    would overstate how much reasoning capacity a shallow model has. d4 is 37M
 *    parameters but only 3.1M of them are transformer.
 *  - the fit verdict for this machine, so a depth that cannot run here is
 *    rejected before the dataset downloads rather than hours into training.
 */
import { HugeiconsIcon } from "@hugeicons/react";
import { AlertCircleIcon, CheckmarkCircleIcon, CpuIcon } from "@hugeicons/core-free-icons";
import { useMemo } from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  FALLBACK_PRESETS,
  formatBytes,
  formatDuration,
  formatParams,
  type FallbackPreset,
} from "../lib/depth-presets";
import type { NanochatDepthPreset, NanochatFit } from "../types";

type Preset = NanochatDepthPreset | FallbackPreset;

function isBackendPreset(p: Preset): p is NanochatDepthPreset {
  return "num_params" in p;
}

const numParams = (p: Preset) => (isBackendPreset(p) ? p.num_params : p.numParams);
const transformerParams = (p: Preset) =>
  isBackendPreset(p) ? p.transformer_params : p.transformerParams;
const embeddingParams = (p: Preset) => (isBackendPreset(p) ? p.embedding_params : p.embeddingParams);
const nEmbd = (p: Preset) => (isBackendPreset(p) ? p.n_embd : p.nEmbd);
const nHead = (p: Preset) => (isBackendPreset(p) ? p.n_head : p.nHead);

export function DepthSelector({
  depth,
  onDepthChange,
  presets,
  fit,
  disabled,
}: {
  depth: number;
  onDepthChange: (depth: number) => void;
  presets?: NanochatDepthPreset[];
  fit: NanochatFit | null;
  disabled?: boolean;
}) {
  const rows = useMemo<Preset[]>(
    () => (presets && presets.length > 0 ? presets : [...FALLBACK_PRESETS]),
    [presets],
  );

  return (
    <div className="flex flex-col gap-2.5">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {rows.map((preset) => {
          const selected = preset.depth === depth;
          return (
            <button
              key={preset.key}
              type="button"
              disabled={disabled}
              aria-pressed={selected}
              onClick={() => onDepthChange(preset.depth)}
              className={cn(
                "group relative flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition-colors",
                "disabled:cursor-not-allowed disabled:opacity-50",
                // The outline stays on the `border` token so it follows the
                // contrast preference. Selection reads through the accent wash
                // rather than a hand-set border alpha, which would not.
                selected
                  ? "border-border bg-accent"
                  : "border-border hover:bg-accent/40",
              )}
            >
              <div className="flex w-full items-center justify-between gap-2">
                <span className="font-heading text-ui-15 font-semibold">
                  {preset.label}
                </span>
                <Badge variant="outline" className="shrink-0 text-[10px]">
                  {preset.tier}
                </Badge>
              </div>
              <span className="text-xs text-muted-foreground">
                {formatParams(numParams(preset))} params
              </span>
              <span className="text-[11px] text-muted-foreground/80">
                {formatParams(transformerParams(preset))} transformer ·{" "}
                {formatParams(embeddingParams(preset))} embedding
              </span>
              <span className="text-[11px] text-muted-foreground/70">
                d_model {nEmbd(preset)} · {nHead(preset)} heads
              </span>
            </button>
          );
        })}
      </div>

      <DepthDetail preset={rows.find((p) => p.depth === depth) ?? null} fit={fit} />

      <p className="text-[11px] leading-relaxed text-muted-foreground/80">
        Depth is nanochat&apos;s one architectural dial: everything else (model
        width, head count) is derived from it. At this vocabulary the embedding
        table is most of the model until roughly depth 25, so the transformer
        count is the number that reflects what depth actually buys.
      </p>
    </div>
  );
}

function DepthDetail({
  preset,
  fit,
}: {
  preset: Preset | null;
  fit: NanochatFit | null;
}) {
  if (!preset) return null;
  const good = isBackendPreset(preset) ? preset.good_for : preset.goodFor;
  const caveat = preset.caveat;

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-muted/30 p-3">
      <div className="flex items-start gap-2 text-xs text-muted-foreground">
        <HugeiconsIcon icon={CpuIcon} className="mt-px size-3.5 shrink-0" />
        <span>{good}</span>
      </div>

      {caveat ? (
        <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
          <HugeiconsIcon icon={AlertCircleIcon} className="mt-px size-3.5 shrink-0" />
          <span>{caveat}</span>
        </div>
      ) : null}

      {fit ? <FitVerdict fit={fit} /> : null}
    </div>
  );
}

function FitVerdict({ fit }: { fit: NanochatFit }) {
  if (fit.fits) {
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
          <HugeiconsIcon icon={CheckmarkCircleIcon} className="size-3.5" />
          Fits on {fit.device_name ?? "this device"}
        </span>
        {fit.est_seconds !== null ? (
          <span className="text-muted-foreground">
            about {formatDuration(fit.est_seconds)} of training
          </span>
        ) : null}
        <span className="text-muted-foreground">
          needs {formatBytes(fit.training_memory_bytes)}
        </span>
        {fit.peak_flops_bf16 ? (
          <span className="text-muted-foreground/70">
            {formatParams(fit.peak_flops_bf16)} peak bf16 FLOPS
          </span>
        ) : null}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      {fit.reasons.map((reason) => (
        <div
          key={reason}
          className="flex items-start gap-2 text-xs text-destructive"
        >
          <HugeiconsIcon icon={AlertCircleIcon} className="mt-px size-3.5 shrink-0" />
          <span>{reason}</span>
        </div>
      ))}
      {fit.suggestions.map((suggestion) => (
        <p key={suggestion} className="pl-5.5 text-[11px] text-muted-foreground">
          {suggestion}
        </p>
      ))}
    </div>
  );
}
