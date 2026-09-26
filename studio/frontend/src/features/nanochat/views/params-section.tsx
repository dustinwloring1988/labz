// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Run parameters, simple and advanced.
 *
 * Simple exposes the handful of numbers that actually change what a run is
 * (depth is elsewhere; this is batch, horizon, context, precision). Advanced
 * exposes the rest, each with a hint, because nanochat has 26+ knobs and a user
 * who wants one of them should be able to find it.
 *
 * The batch size field validates against deviceBatchSize * maxSeqLen, because
 * nanochat asserts that divisibility and dies at the first forward pass
 * otherwise. Catching it here turns a crash into a field that refuses the value.
 */
import { Alert02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs } from "@/components/ui/tabs";
import { SegmentedTabsList } from "@/components/segmented-tabs";
import { cn } from "@/lib/utils";
import {
  NANOCHAT_BATCH_SIZES,
  NANOCHAT_CONTEXT_LENGTHS,
  gradientAccumulationSteps,
  isValidTotalBatchSize,
  type NanochatConfigState,
  type NanochatParamMode,
} from "../stores/nanochat-config-policy";

// `as const` so the tuple satisfies SegmentedTabsList's at-least-two-options
// signature; a plain array widens `value` to string and no longer type-checks.
const PARAM_MODE_OPTIONS = [
  { value: "simple", label: "Simple" },
  { value: "advanced", label: "Advanced" },
] as const;

export function ParamsSection({
  config,
  set,
  paramMode,
  onParamModeChange,
  disabled,
}: {
  config: NanochatConfigState;
  set: (patch: Partial<NanochatConfigState>) => void;
  paramMode: NanochatParamMode;
  onParamModeChange: (mode: NanochatParamMode) => void;
  disabled?: boolean;
}) {
  const micro = config.deviceBatchSize * config.maxSeqLen;
  const batchValid = isValidTotalBatchSize(config.totalBatchSize, config.deviceBatchSize, config.maxSeqLen);
  const gradAccum = gradientAccumulationSteps(
    config.totalBatchSize,
    config.deviceBatchSize,
    config.maxSeqLen,
  );

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-heading text-ui-15 font-semibold">Run parameters</h3>
        {/* SegmentedTabsList has no change handler of its own: it reads the value
            from the Tabs context above it, so the Tabs wrapper is what makes the
            control interactive. */}
        <Tabs
          value={paramMode}
          onValueChange={(value) => onParamModeChange(value as NanochatParamMode)}
          className="contents"
        >
          <SegmentedTabsList
            value={paramMode}
            options={PARAM_MODE_OPTIONS}
            ariaLabel="Parameter detail"
            size="compact"
          />
        </Tabs>
      </div>

      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        <NumberField
          id="nanochat-iterations"
          label="Training steps"
          value={config.numIterations}
          onChange={(value) => set({ numIterations: value })}
          min={1}
          disabled={disabled}
          hint={`Pretraining runs this many optimizer steps. At depth ${config.depth} that is ${formatTokens(config.numIterations * config.totalBatchSize)} of text.`}
        />
        <NumberField
          id="nanochat-batch"
          label="Total batch size"
          value={config.totalBatchSize}
          onChange={(value) => set({ totalBatchSize: value })}
          min={1}
          disabled={disabled}
          invalid={!batchValid}
          hint={
            batchValid
              ? `${config.deviceBatchSize} × ${config.maxSeqLen} per micro-batch, so ${gradAccum} gradient accumulation steps.`
              : `Must be a multiple of ${micro.toLocaleString()} (device batch × context).`
          }
        />
        <NumberField
          id="nanochat-device-batch"
          label="Device batch size"
          value={config.deviceBatchSize}
          onChange={(value) => set({ deviceBatchSize: value })}
          min={1}
          disabled={disabled}
          hint="Sequences per forward pass. Lower this if a step runs out of memory."
        />
        <div className="flex flex-col gap-1">
          <Label htmlFor="nanochat-context" className="text-xs">
            Context length
          </Label>
          <Select
            value={String(config.maxSeqLen)}
            onValueChange={(value) => set({ maxSeqLen: Number(value) })}
            disabled={disabled}
          >
            <SelectTrigger id="nanochat-context" className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {NANOCHAT_CONTEXT_LENGTHS.map((len) => (
                <SelectItem key={len} value={String(len)}>
                  {len.toLocaleString()}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="nanochat-window" className="text-xs">
            Attention pattern
          </Label>
          <Select
            value={config.windowPattern}
            onValueChange={(value) => set({ windowPattern: value })}
            disabled={disabled}
          >
            <SelectTrigger id="nanochat-window" className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="L">L — full context everywhere</SelectItem>
              <SelectItem value="SSSL">SSSL — sliding window (nanochat&apos;s speedrun setting)</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-[11px] text-muted-foreground/80">
            Sliding window needs Flash Attention 3. Consumer cards fall back to
            SDPA, which cannot do it at all, so full attention is the safer
            default here.
          </p>
        </div>
        <div className="flex items-end pb-1">
          <label className="flex items-center gap-2 text-xs">
            <Switch
              checked={config.fp8}
              disabled={disabled}
              onCheckedChange={(checked) => set({ fp8: checked })}
            />
            FP8 training
            <span className="text-[11px] text-muted-foreground">(Hopper and newer)</span>
          </label>
        </div>
      </div>

      <p className="text-[11px] text-muted-foreground/70">
        Batch sizes that divide evenly:{" "}
        {NANOCHAT_BATCH_SIZES.filter(
          (size) => size % (config.deviceBatchSize * config.maxSeqLen) === 0,
        )
          .map((size) => size.toLocaleString())
          .join(", ")}
      </p>

      {paramMode === "advanced" ? (
        <AdvancedParams config={config} set={set} disabled={disabled} />
      ) : null}
    </section>
  );
}

function AdvancedParams({
  config,
  set,
  disabled,
}: {
  config: NanochatConfigState;
  set: (patch: Partial<NanochatConfigState>) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-4 border-t border-border pt-4">
      <div className="flex flex-col gap-2.5">
        <span className="text-xs font-medium text-muted-foreground">Horizon</span>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
          <NumberField
            id="nanochat-param-data-ratio"
            label="Data:param ratio"
            value={config.paramDataRatio}
            onChange={(value) => set({ paramDataRatio: value })}
            min={0}
            step={0.5}
            disabled={disabled}
            hint="Tokens per scaling parameter. 12 is nanochat's default; 20 is Chinchilla."
          />
          <NumberField
            id="nanochat-sft-iterations"
            label="Fine-tuning steps"
            value={config.sftIterations}
            onChange={(value) => set({ sftIterations: value })}
            min={1}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-aspect"
            label="Aspect ratio"
            value={config.aspectRatio}
            onChange={(value) => set({ aspectRatio: value })}
            min={1}
            disabled={disabled}
            hint="Model width is depth × this, rounded to a multiple of head dim."
          />
        </div>
      </div>

      <div className="flex flex-col gap-2.5">
        <span className="text-xs font-medium text-muted-foreground">Learning rates</span>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
          <NumberField
            id="nanochat-matrix-lr"
            label="Matrix LR (Muon)"
            value={config.matrixLr}
            onChange={(value) => set({ matrixLr: value })}
            min={0}
            step={0.001}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-embed-lr"
            label="Embedding LR"
            value={config.embeddingLr}
            onChange={(value) => set({ embeddingLr: value })}
            min={0}
            step={0.01}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-unembed-lr"
            label="Unembedding LR"
            value={config.unembeddingLr}
            onChange={(value) => set({ unembeddingLr: value })}
            min={0}
            step={0.001}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-scalar-lr"
            label="Scalar LR"
            value={config.scalarLr}
            onChange={(value) => set({ scalarLr: value })}
            min={0}
            step={0.01}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-weight-decay"
            label="Weight decay"
            value={config.weightDecay}
            onChange={(value) => set({ weightDecay: value })}
            min={0}
            step={0.01}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-warmup"
            label="Warmup steps"
            value={config.warmupSteps}
            onChange={(value) => set({ warmupSteps: value })}
            min={0}
            disabled={disabled}
          />
        </div>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          <NumberField
            id="nanochat-warmdown"
            label="Warmdown ratio"
            value={config.warmdownRatio}
            onChange={(value) => set({ warmdownRatio: value })}
            min={0}
            max={1}
            step={0.05}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-final-lr"
            label="Final LR fraction"
            value={config.finalLrFrac}
            onChange={(value) => set({ finalLrFrac: value })}
            min={0}
            max={1}
            step={0.01}
            disabled={disabled}
          />
        </div>
      </div>

      <div className="flex flex-col gap-2.5">
        <span className="text-xs font-medium text-muted-foreground">
          Evaluation cadence (−1 disables)
        </span>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          <NumberField
            id="nanochat-eval-every"
            label="Val bpb every"
            value={config.evalEvery}
            onChange={(value) => set({ evalEvery: value })}
            min={-1}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-core-every"
            label="CORE metric every"
            value={config.coreMetricEvery}
            onChange={(value) => set({ coreMetricEvery: value })}
            min={-1}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-sample-every"
            label="Text sample every"
            value={config.sampleEvery}
            onChange={(value) => set({ sampleEvery: value })}
            min={-1}
            disabled={disabled}
            hint="How often a sample is generated for the feed. This costs real time."
          />
          <NumberField
            id="nanochat-save-every"
            label="Checkpoint every"
            value={config.saveEvery}
            onChange={(value) => set({ saveEvery: value })}
            min={-1}
            disabled={disabled}
            hint="−1 saves only at the end, so a stopped run keeps nothing but its last checkpoint."
          />
        </div>
      </div>

      <div className="flex flex-col gap-2.5">
        <span className="text-xs font-medium text-muted-foreground">
          Reinforcement learning
        </span>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          <NumberField
            id="nanochat-rl-samples"
            label="Samples per question"
            value={config.rlNumSamples}
            onChange={(value) => set({ rlNumSamples: value })}
            min={1}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-rl-examples"
            label="Examples per step"
            value={config.rlExamplesPerStep}
            onChange={(value) => set({ rlExamplesPerStep: value })}
            min={1}
            disabled={disabled}
            hint="Sequences generated per step is this times samples per question."
          />
          <NumberField
            id="nanochat-rl-tokens"
            label="Max new tokens"
            value={config.rlMaxNewTokens}
            onChange={(value) => set({ rlMaxNewTokens: value })}
            min={16}
            disabled={disabled}
          />
          <NumberField
            id="nanochat-rl-epochs"
            label="RL epochs"
            value={config.rlEpochs}
            onChange={(value) => set({ rlEpochs: value })}
            min={1}
            disabled={disabled}
          />
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="nanochat-model-tag" className="text-xs">
          Model tag
        </Label>
        <Input
          id="nanochat-model-tag"
          value={config.modelTag ?? ""}
          disabled={disabled}
          placeholder={`d${config.depth}`}
          onChange={(event) =>
            set({ modelTag: event.target.value.trim() === "" ? null : event.target.value })
          }
          className="h-8"
        />
        <p className="text-[11px] text-muted-foreground/80">
          Names the checkpoint directory. Leave blank to use d{config.depth}.
          Checkpoints with the same tag overwrite each other.
        </p>
      </div>
    </div>
  );
}

export function NumberField({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
  hint,
  invalid,
}: {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  hint?: string;
  invalid?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Label htmlFor={id} className="text-xs">
        {label}
      </Label>
      <Input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        aria-invalid={invalid || undefined}
        onChange={(event) => {
          const next = Number(event.target.value);
          // An empty field parses as NaN; ignore it rather than writing NaN into
          // the config, which would then be sent to the backend as null.
          if (Number.isFinite(next)) {
            onChange(max !== undefined ? Math.min(max, next) : next);
          }
        }}
        className={cn("h-8", invalid && "border-destructive")}
      />
      {hint ? (
        invalid ? (
          <p className="flex items-start gap-1 text-[11px] text-destructive">
            <HugeiconsIcon icon={Alert02Icon} className="mt-px size-3 shrink-0" />
            <span>{hint}</span>
          </p>
        ) : (
          <p className="text-[11px] text-muted-foreground/80">{hint}</p>
        )
      ) : null}
    </div>
  );
}

function formatTokens(tokens: number): string {
  if (tokens >= 1e9) return `${(tokens / 1e9).toFixed(2)}B tokens`;
  if (tokens >= 1e6) return `${(tokens / 1e6).toFixed(1)}M tokens`;
  if (tokens >= 1e3) return `${(tokens / 1e3).toFixed(0)}K tokens`;
  return `${tokens} tokens`;
}
