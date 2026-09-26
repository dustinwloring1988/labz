// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The run preview: what will happen, and the button that starts it.
 *
 * Start is blocked on a configuration that cannot run, but *not* on one that is
 * merely large. Refusing a big run would be wrong (the estimate could be
 * pessimistic, and the user may know something the probe does not), so a run that
 * does not fit is allowed with a clear warning attached. Refusing a run that
 * cannot be built at all — a batch size that does not divide, an unknown dataset
 * — is a different matter, because that one is guaranteed to die.
 */
import { Alert02Icon, InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { formatBytes, formatDuration, formatParams } from "../lib/depth-presets";
import type { NanochatFit } from "../types";
import type { NanochatConfigState } from "../stores/nanochat-config-policy";
import { isValidTotalBatchSize } from "../stores/nanochat-config-policy";

export function RunPreviewCard({
  config,
  fit,
  errors,
  warnings,
  isValidating,
  isStarting,
  canStart,
  onStart,
}: {
  config: NanochatConfigState;
  fit: NanochatFit | null;
  errors: string[];
  warnings: string[];
  isValidating: boolean;
  isStarting: boolean;
  canStart: boolean;
  onStart: () => void;
}) {
  const batchValid = isValidTotalBatchSize(
    config.totalBatchSize,
    config.deviceBatchSize,
    config.maxSeqLen,
  );
  const tokens = config.numIterations * config.totalBatchSize;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border p-3.5">
      <h3 className="font-heading text-ui-15 font-semibold">Run summary</h3>

      <dl className="flex flex-col gap-1.5 text-xs">
        <Row label="Model" value={`d${config.depth}`} />
        <Row label="Parameters" value={fit ? formatParams(fit.num_params) : "—"} />
        {fit ? (
          <Row
            label="of which transformer"
            value={formatParams(fit.transformer_params)}
            muted
          />
        ) : null}
        <Row label="Context" value={config.maxSeqLen.toLocaleString()} />
        <Row label="Total batch" value={`${config.totalBatchSize.toLocaleString()} tokens`} />
        <Row
          label="Tokens seen"
          value={`${(tokens / 1e9).toFixed(2)}B`}
        />
        {fit?.est_seconds ? (
          <Row label="Estimated time" value={formatDuration(fit.est_seconds)} />
        ) : null}
        {fit ? (
          <Row
            label="Training memory"
            value={formatBytes(fit.training_memory_bytes)}
          />
        ) : null}
        <Row
          label="Stages"
          value={`${config.stages.filter((s) => s !== "rl" || config.runRl).length} of ${config.stages.length}`}
        />
        {config.runRl ? <Row label="RL" value="enabled" /> : null}
      </dl>

      {!batchValid ? (
        <Callout tone="error">
          Total batch size must be a multiple of{" "}
          {(config.deviceBatchSize * config.maxSeqLen).toLocaleString()} (device batch ×
          context). nanochat would fail at the first step.
        </Callout>
      ) : null}

      {errors.length > 0 ? (
        <div className="flex flex-col gap-1">
          {errors.map((error) => (
            <Callout key={error} tone="error">
              {error}
            </Callout>
          ))}
        </div>
      ) : null}

      {fit && !fit.fits && fit.reasons.length > 0 ? (
        <div className="flex flex-col gap-1">
          {fit.reasons.map((reason) => (
            <Callout key={reason} tone="error">
              {reason}
            </Callout>
          ))}
          {fit.suggestions.map((suggestion) => (
            <p key={suggestion} className="pl-1 text-[11px] text-muted-foreground">
              {suggestion}
            </p>
          ))}
        </div>
      ) : null}

      {warnings.length > 0 ? (
        <div className="flex flex-col gap-1">
          {warnings.map((warning) => (
            <Callout key={warning} tone="warning">
              {warning}
            </Callout>
          ))}
        </div>
      ) : null}

      <Separator />

      <Button
        onClick={onStart}
        disabled={!canStart || isStarting || !batchValid || errors.length > 0}
        className="w-full"
      >
        {isStarting ? "Starting…" : "Start run"}
      </Button>

      {isValidating ? (
        <p className="text-center text-[11px] text-muted-foreground">Checking…</p>
      ) : fit && !fit.fits ? (
        <p className="text-center text-[11px] text-muted-foreground">
          You can still start this. It may run out of memory partway through.
        </p>
      ) : null}
    </div>
  );
}

function Row({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className={muted ? "pl-2 text-muted-foreground/80" : "text-muted-foreground"}>
        {label}
      </dt>
      <dd className="font-mono text-[11px]">{value}</dd>
    </div>
  );
}

function Callout({
  tone,
  children,
}: {
  tone: "error" | "warning";
  children: React.ReactNode;
}) {
  return (
    <div
      className={
        tone === "error"
          ? "flex items-start gap-1.5 rounded-md border border-destructive/30 bg-destructive/5 p-2 text-[11px] text-destructive"
          : "flex items-start gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-[11px] text-amber-700 dark:text-amber-400"
      }
    >
      <HugeiconsIcon
        icon={tone === "error" ? Alert02Icon : InformationCircleIcon}
        className="mt-px size-3 shrink-0"
      />
      <span>{children}</span>
    </div>
  );
}
