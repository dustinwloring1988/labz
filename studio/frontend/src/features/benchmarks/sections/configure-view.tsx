// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertCircleIcon,
  CheckmarkCircle02Icon,
  RankingIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";

import {
  MAX_PROBLEMS_CEILING,
  MIN_PROBLEMS,
  type BenchmarkFormState,
} from "../stores/benchmark-config-policy";
import type { BenchmarkModelCandidate, BenchmarkSpec } from "../types";

const SOURCE_LABELS: Record<string, string> = {
  models_dir: "Downloaded",
  hf_cache: "Downloaded",
  lmstudio: "LM Studio",
  ollama: "Ollama",
  hermes: "Hermes",
  custom: "Custom folder",
  checkpoint: "Checkpoint",
  lora: "LoRA adapter",
  local: "Local",
};

/** Models grouped by where they came from, so a long list stays navigable. */
function groupCandidates(
  models: readonly BenchmarkModelCandidate[],
): { source: string; models: BenchmarkModelCandidate[] }[] {
  const groups = new Map<string, BenchmarkModelCandidate[]>();
  for (const model of models) {
    const bucket = groups.get(model.source);
    if (bucket) bucket.push(model);
    else groups.set(model.source, [model]);
  }
  // Checkpoints and adapters first: they are what this app produced and usually
  // what the user came here to compare.
  const order = [
    "checkpoint",
    "lora",
    "models_dir",
    "hf_cache",
    "custom",
    "lmstudio",
    "ollama",
    "hermes",
    "local",
  ];
  return [...groups.entries()]
    .map(([source, entries]) => ({
      source,
      models: [...entries].sort((a, b) => a.label.localeCompare(b.label)),
    }))
    .sort((a, b) => {
      const left = order.indexOf(a.source);
      const right = order.indexOf(b.source);
      return (
        (left === -1 ? order.length : left) -
        (right === -1 ? order.length : right)
      );
    });
}

function ModelPicker({
  models,
  value,
  onChange,
  emptyReason,
  disabled = false,
  triggerLabel = "Choose a model to benchmark",
  placeholder = "Choose a model to benchmark",
}: {
  models: readonly BenchmarkModelCandidate[];
  value: string | null;
  onChange: (path: string) => void;
  emptyReason: string | null;
  disabled?: boolean;
  triggerLabel?: string;
  placeholder?: string;
}): ReactElement {
  const groups = groupCandidates(models);
  const selected = models.find((model) => model.path === value) ?? null;

  if (models.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border/70 px-4 py-6 text-center">
        <p className="text-sm text-muted-foreground">
          {emptyReason ?? "No local models were found."}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <Select value={value ?? ""} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger className="w-full" aria-label={triggerLabel}>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {groups.map((group) => (
            <SelectGroup key={group.source}>
              <SelectLabel>
                {SOURCE_LABELS[group.source] ?? group.source}
              </SelectLabel>
              {group.models.map((model) => (
                <SelectItem
                  key={`${model.id}:${model.path}`}
                  value={model.path}
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <span className="truncate">{model.label}</span>
                    {model.format === "gguf" ? (
                      <Badge variant="outline" className="shrink-0 text-[10px]">
                        GGUF
                      </Badge>
                    ) : null}
                    {model.lora ? (
                      <Badge variant="outline" className="shrink-0 text-[10px]">
                        LoRA
                      </Badge>
                    ) : null}
                  </span>
                </SelectItem>
              ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
      {selected?.base_model ? (
        <p className="text-xs text-muted-foreground">
          Adapter on <span className="font-mono">{selected.base_model}</span>
        </p>
      ) : null}
    </div>
  );
}

function BenchmarkPicker({
  specs,
  selected,
  onToggle,
}: {
  specs: readonly BenchmarkSpec[];
  selected: readonly string[];
  onToggle: (key: string) => void;
}): ReactElement {
  return (
    <div className="flex flex-col gap-1.5">
      {specs.map((spec) => {
        const checked = selected.includes(spec.key);
        const id = `benchmark-${spec.key}`;
        return (
          <label
            key={spec.key}
            htmlFor={id}
            className={cn(
              "flex cursor-pointer items-start gap-3 rounded-2xl px-3 py-2.5 transition-colors",
              spec.supported
                ? "hover:bg-muted/50"
                : "cursor-not-allowed opacity-60",
            )}
          >
            <Checkbox
              id={id}
              checked={checked}
              // Unsupported is enforced on the server too, but disabling it here is
              // what lets the row say why rather than accepting a click that 400s.
              disabled={!spec.supported}
              onCheckedChange={() => onToggle(spec.key)}
              className="mt-0.5"
            />
            <span className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium text-foreground">
                  {spec.label}
                </span>
                <Badge variant="secondary" className="text-[10px]">
                  {spec.kind === "categorical"
                    ? "multiple choice"
                    : "generated answer"}
                </Badge>
                {spec.is_default ? (
                  <Badge variant="outline" className="text-[10px]">
                    default
                  </Badge>
                ) : null}
              </span>
              <span className="text-xs text-muted-foreground">
                {spec.supported
                  ? spec.notes
                  : `Cannot run here: ${spec.unavailable_reason ?? "unsupported on this host"}.`}
              </span>
            </span>
          </label>
        );
      })}
    </div>
  );
}

function NumberField({
  id,
  label,
  hint,
  value,
  min,
  max,
  step = 1,
  onChange,
  problem,
}: {
  id: string;
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  problem?: string;
}): ReactElement {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id} className="text-ui-13">
        {label}
      </Label>
      <Input
        id={id}
        type="number"
        inputMode="numeric"
        value={value}
        min={min}
        max={max}
        step={step}
        aria-invalid={problem ? true : undefined}
        onChange={(event) => {
          const next = Number(event.target.value);
          // An empty field parses as NaN. Held at the current value rather than
          // written, so clearing the box to retype does not snap the field to 0.
          if (Number.isFinite(next)) onChange(next);
        }}
      />
      <p
        className={cn(
          "text-xs text-muted-foreground",
          problem && "text-destructive",
        )}
      >
        {problem ?? hint}
      </p>
    </div>
  );
}

export function BenchmarkConfigureForm({
  form,
  specs,
  models,
  modelsEmptyReason,
  problems,
  onChange,
  onToggleBenchmark,
}: {
  form: BenchmarkFormState;
  specs: readonly BenchmarkSpec[];
  models: readonly BenchmarkModelCandidate[];
  modelsEmptyReason: string | null;
  problems: Readonly<Record<string, string>>;
  onChange: (patch: Partial<BenchmarkFormState>) => void;
  onToggleBenchmark: (key: string) => void;
}): ReactElement {
  const selectedCount = form.benchmarkKeys.length;
  const totalProblems = selectedCount * form.maxProblems;
  // Stated rather than left for the user to work out: a generative benchmark
  // generates an answer per problem, so its cost is not comparable to a
  // multiple-choice one, and a suite of them is a different job than it looks.
  const hasGenerative = specs.some(
    (spec) =>
      form.benchmarkKeys.includes(spec.key) && spec.kind === "generative",
  );

  return (
    <div className="@container/bench-configure flex flex-col gap-8">
      <div className="grid grid-cols-1 gap-8 @4xl/bench-configure:grid-cols-[minmax(0,1fr)_320px] @5xl/bench-configure:gap-10">
        <div className="flex min-w-0 flex-col gap-8">
          <section className="flex flex-col gap-3">
            <div className="flex flex-col gap-0.5">
              <h2 className="text-ui-15 font-semibold text-foreground">
                Model
              </h2>
              <p className="text-sm text-muted-foreground">
                Any model on this machine: something downloaded, a training
                checkpoint, or a LoRA adapter.
              </p>
            </div>
            <ModelPicker
              models={models}
              value={form.modelPath}
              emptyReason={modelsEmptyReason}
              onChange={(path) => onChange({ modelPath: path })}
            />
            {problems.model ? (
              <p className="flex items-center gap-1.5 text-xs text-destructive">
                <HugeiconsIcon
                  icon={AlertCircleIcon}
                  className="size-3.5 shrink-0"
                />
                {problems.model}
              </p>
            ) : null}
            <div className="flex items-center justify-between gap-4 rounded-2xl border border-border/60 px-3 py-2.5">
              <div className="flex min-w-0 flex-col gap-0.5">
                <Label htmlFor="bench-4bit" className="text-ui-13">
                  Load in 4-bit
                </Label>
                <p className="text-xs text-muted-foreground">
                  Fits a larger model on this GPU. Changes the number it
                  produces, so a 4-bit score is not directly comparable with a
                  16-bit one.
                </p>
              </div>
              <Switch
                id="bench-4bit"
                checked={form.loadIn4bit}
                onCheckedChange={(value) => onChange({ loadIn4bit: value })}
              />
            </div>
          </section>

          <section className="flex flex-col gap-3">
            <div className="flex flex-col gap-0.5">
              <h2 className="text-ui-15 font-semibold text-foreground">
                Benchmarks
              </h2>
              <p className="text-sm text-muted-foreground">
                Scored through nanochat&rsquo;s own task definitions, so these
                are the same benchmarks and the same grading its published
                numbers use.
              </p>
            </div>
            <BenchmarkPicker
              specs={specs}
              selected={form.benchmarkKeys}
              onToggle={onToggleBenchmark}
            />
            {problems.benchmarks ? (
              <p className="flex items-center gap-1.5 text-xs text-destructive">
                <HugeiconsIcon
                  icon={AlertCircleIcon}
                  className="size-3.5 shrink-0"
                />
                {problems.benchmarks}
              </p>
            ) : null}
          </section>

          <section className="flex flex-col gap-3">
            <div className="flex flex-col gap-0.5">
              <h2 className="text-ui-15 font-semibold text-foreground">
                Judge
                <span className="ml-2 align-middle text-xs font-normal text-muted-foreground">
                  optional
                </span>
              </h2>
              <p className="text-sm text-muted-foreground">
                A second model reads each generated answer and decides whether it
                is right. The exact grader wants{" "}
                <code className="font-mono text-xs">#### 18</code> and scores
                zero for a correct answer written as prose, so this measures
                getting it right rather than formatting it a particular way.
              </p>
            </div>
            <ModelPicker
              models={models}
              value={form.judgeModelPath}
              emptyReason={modelsEmptyReason}
              disabled={!hasGenerative}
              triggerLabel="Model to judge the answers"
              placeholder={
                hasGenerative
                  ? "No judge — score with the exact grader"
                  : "Select a generative benchmark to use a judge"
              }
              onChange={(path) => onChange({ judgeModelPath: path })}
            />
            <p className="text-xs text-muted-foreground">
              {hasGenerative ? (
                form.judgeModelPath === form.modelPath ? (
                  <>
                    The same model is selected for both. It is allowed, and a
                    self-judge is a reasonable smoke test, but a model grading its
                    own homework is a weaker measurement than a different one.
                  </>
                ) : (
                  <>
                    Runs after the benchmark, in its own pass, so only one model
                    is on the GPU at a time. Both figures are kept and the board
                    ranks judged runs below exact ones.
                  </>
                )
              ) : (
                <>
                  Only generative benchmarks are judged. A multiple-choice
                  benchmark is one letter compared against one letter of logits,
                  which a second model can only add noise to.
                </>
              )}
            </p>
          </section>

          <section className="flex flex-col gap-4">
            <div className="flex flex-col gap-0.5">
              <h2 className="text-ui-15 font-semibold text-foreground">
                Limits
              </h2>
              <p className="text-sm text-muted-foreground">
                A cap on problems per benchmark. Every benchmark here is larger
                than a quick check, so this is what keeps a suite from becoming
                an overnight job.
              </p>
            </div>

            <div className="flex flex-col gap-2.5">
              <div className="flex items-baseline justify-between gap-3">
                <Label htmlFor="bench-problems" className="text-ui-13">
                  Problems per benchmark
                </Label>
                <span className="font-mono text-ui-12 tabular-nums text-muted-foreground">
                  {form.maxProblems.toLocaleString()}
                </span>
              </div>
              <Slider
                id="bench-problems"
                min={MIN_PROBLEMS}
                max={2000}
                step={8}
                value={[form.maxProblems]}
                onValueChange={([next]) => {
                  if (typeof next === "number") onChange({ maxProblems: next });
                }}
              />
              <p className="text-xs text-muted-foreground">
                {selectedCount === 0
                  ? "Select a benchmark to see the total."
                  : `${totalProblems.toLocaleString()} problems in total.`}
                {form.maxProblems >= MAX_PROBLEMS_CEILING
                  ? " At the ceiling, so a benchmark may still be cut short."
                  : ""}
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 @sm/bench-configure:grid-cols-3">
              <NumberField
                id="bench-batch"
                label="Batch size"
                hint="Problems per forward pass. Higher is faster and uses more VRAM."
                value={form.batchSize}
                min={1}
                max={256}
                onChange={(value) => onChange({ batchSize: value })}
                problem={problems.batchSize}
              />
              <NumberField
                id="bench-new-tokens"
                label="Max new tokens"
                hint="Answer length for generated benchmarks."
                value={form.maxNewTokens}
                min={16}
                max={8192}
                step={16}
                onChange={(value) => onChange({ maxNewTokens: value })}
              />
              <NumberField
                id="bench-max-seq"
                label="Max sequence"
                hint="Prompt length cap. Raise it for long-context models."
                value={form.maxSeqLength}
                min={256}
                max={131072}
                step={256}
                onChange={(value) => onChange({ maxSeqLength: value })}
              />
            </div>
          </section>
        </div>

        <div className="@4xl/bench-configure:sticky @4xl/bench-configure:top-6 @4xl/bench-configure:self-start">
          <div className="flex flex-col gap-3 rounded-3xl bg-card p-5 ring-1 ring-border/60">
            <div className="flex items-center gap-2">
              <HugeiconsIcon
                icon={RankingIcon}
                className="size-4 shrink-0 text-muted-foreground"
                strokeWidth={1.75}
              />
              <h2 className="text-ui-14 font-semibold text-foreground">
                This run
              </h2>
            </div>
            <dl className="flex flex-col gap-2 text-ui-12">
              <Row label="Benchmarks" value={String(selectedCount)} />
              <Row
                label="Problems each"
                value={form.maxProblems.toLocaleString()}
              />
              <Row
                label="Total problems"
                value={totalProblems.toLocaleString()}
              />
              <Row
                label="Measurement"
                value={
                  form.modelPath &&
                  models.find((model) => model.path === form.modelPath)
                    ? models.find((model) => model.path === form.modelPath)
                        ?.format === "gguf"
                      ? "Generated answer"
                      : "Exact (logits)"
                    : "—"
                }
              />
            </dl>
            {hasGenerative ? (
              <p className="text-xs text-muted-foreground">
                A generated benchmark writes an answer per problem, so it takes
                far longer than a multiple-choice one at the same cap.
              </p>
            ) : null}
            {/* Stated here rather than discovered afterwards: a run stopped at the
                cap reports each benchmark's score but no suite composite, because
                averaging a sample with a full measurement would rank a lucky few
                hundred problems above a thorough run. The per-benchmark scores are
                the result; the composite is only for a run that measured whole
                benchmarks. */}
            {selectedCount > 0 ? (
              <p className="text-xs text-muted-foreground">
                At {form.maxProblems.toLocaleString()} problems each this run
                reports per-benchmark scores but no overall composite &mdash;
                the figures come from a sample of each benchmark, not the whole
                set. Raise the cap to the full size of a benchmark for a
                comparable composite.
              </p>
            ) : null}
            {form.loadIn4bit ? (
              <p className="text-xs text-muted-foreground">
                4-bit changes what the model computes. Not comparable with a
                16-bit score of the same model.
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }): ReactElement {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono tabular-nums text-foreground">{value}</dd>
    </div>
  );
}

/** The start control and the state it cannot be in. */
export function BenchmarkStartCta({
  canStart,
  isStarting,
  runActive,
  onStart,
  error,
}: {
  canStart: boolean;
  isStarting: boolean;
  runActive: boolean;
  onStart: () => void;
  error: string | null;
}): ReactElement {
  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        disabled={!canStart || isStarting || runActive}
        onClick={onStart}
        className={cn(
          "flex h-10 w-full items-center justify-center gap-2 rounded-full px-4 text-ui-14 font-medium transition-colors",
          "bg-foreground text-background hover:opacity-90",
          "disabled:cursor-not-allowed disabled:opacity-40",
        )}
      >
        {isStarting ? (
          "Starting…"
        ) : runActive ? (
          <>
            <HugeiconsIcon
              icon={CheckmarkCircle02Icon}
              className="size-4"
              strokeWidth={2}
            />
            Run in progress
          </>
        ) : (
          <>
            <HugeiconsIcon
              icon={RankingIcon}
              className="size-4"
              strokeWidth={1.75}
            />
            Run benchmarks
          </>
        )}
      </button>
      {error ? (
        <p className="flex items-start gap-1.5 text-xs text-destructive">
          <HugeiconsIcon
            icon={AlertCircleIcon}
            className="mt-px size-3.5 shrink-0"
          />
          {error}
        </p>
      ) : null}
    </div>
  );
}
