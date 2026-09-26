// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Dataset, SFT task and benchmark pickers.
 *
 * The options come from nanochat's own catalogue rather than a list kept here, so
 * a benchmark renamed upstream disappears from the picker instead of failing at
 * eval time. When the catalogue is unavailable the pickers render an explanation
 * rather than an empty box, because an empty checkbox list reads as "nothing is
 * selected" and silently produces a run that trains on no data.
 */
import { HugeiconsIcon } from "@hugeicons/react";
import { Alert02Icon, Database01Icon, Task01Icon } from "@hugeicons/core-free-icons";

import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { NanochatCatalogue } from "../types";

function CatalogueUnavailable({ catalogue }: { catalogue: NanochatCatalogue | null }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-muted-foreground">
      <HugeiconsIcon icon={Alert02Icon} className="mt-px size-3.5 shrink-0 text-amber-500" />
      <span>
        {catalogue?.reason
          ? `nanochat has not described its options yet (${catalogue.reason}).`
          : "nanochat has not described its options yet."}{" "}
        Set nanochat up to choose datasets and benchmarks. The defaults below are
        nanochat&apos;s own and are used as-is.
      </span>
    </div>
  );
}

export function DatasetSection({
  catalogue,
  dataset,
  onDatasetChange,
  numShards,
  onNumShardsChange,
  vocabSize,
  onVocabSizeChange,
  trainTokenizer,
  onTrainTokenizerChange,
  disabled,
}: {
  catalogue: NanochatCatalogue | null;
  dataset: string;
  onDatasetChange: (value: string) => void;
  numShards: number;
  onNumShardsChange: (value: number) => void;
  vocabSize: number;
  onVocabSizeChange: (value: number) => void;
  trainTokenizer: boolean;
  onTrainTokenizerChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  const datasets = catalogue?.datasets ?? {};
  const entries = Object.values(datasets);
  const active = dataset in datasets ? datasets[dataset] : null;

  return (
    <section className="flex flex-col gap-3">
      <header className="flex items-center gap-2">
        <HugeiconsIcon icon={Database01Icon} className="size-4 text-muted-foreground" />
        <h3 className="font-heading text-ui-15 font-semibold">Pretraining data</h3>
      </header>

      {catalogue && !catalogue.unavailable && entries.length === 0 ? (
        <CatalogueUnavailable catalogue={catalogue} />
      ) : null}

      {entries.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="nanochat-dataset">Corpus</Label>
          <Select value={dataset} onValueChange={onDatasetChange} disabled={disabled}>
            <SelectTrigger id="nanochat-dataset">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {entries.map((entry) => (
                <SelectItem key={entry.key} value={entry.key}>
                  {entry.label}
                  {entry.is_default ? " (default)" : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {active ? (
            <div className="flex flex-col gap-1 text-[11px] text-muted-foreground">
              <span>{active.notes}</span>
              <span className="text-muted-foreground/70">
                {entry_shards(active.shards_present)}
                {active.max_shard !== null
                  ? ` · up to ${active.max_shard.toLocaleString()} shards`
                  : ""}
              </span>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="flex flex-col gap-1.5">
        <div className="flex items-baseline justify-between gap-2">
          <Label htmlFor="nanochat-shards">Shards to download</Label>
          <span className="font-mono text-xs text-muted-foreground">{numShards}</span>
        </div>
        <Slider
          id="nanochat-shards"
          min={1}
          max={400}
          step={1}
          value={[numShards]}
          disabled={disabled}
          onValueChange={([value]) => onNumShardsChange(value ?? numShards)}
        />
        <p className="text-[11px] text-muted-foreground/80">
          One shard is roughly 100 MB. 32 is enough for a small run; the speedrun
          used 170. Downloading happens once and is reused by later runs.
        </p>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="nanochat-vocab">Tokenizer vocabulary</Label>
        <Select
          value={String(vocabSize)}
          onValueChange={(value) => onVocabSizeChange(Number(value))}
          disabled={disabled}
        >
          <SelectTrigger id="nanochat-vocab">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {[8192, 16384, 32768, 65536].map((size) => (
              <SelectItem key={size} value={String(size)}>
                {size.toLocaleString()}
                {size === 32768 ? " (default)" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <label className="flex items-start gap-2.5 text-xs">
        <Checkbox
          checked={trainTokenizer}
          disabled={disabled}
          onCheckedChange={(checked) => onTrainTokenizerChange(checked === true)}
          className="mt-0.5"
        />
        <span className="flex flex-col gap-0.5">
          <span>Train a tokenizer on this corpus</span>
          <span className="text-[11px] text-muted-foreground">
            Needed the first time, since the tokenizer has to be trained on the same
            text as the model. Training a new one invalidates existing checkpoints.
          </span>
        </span>
      </label>
    </section>
  );
}

function entry_shards(present: number): string {
  if (present <= 0) return "not downloaded yet";
  return `${present.toLocaleString()} shard${present === 1 ? "" : "s"} on disk`;
}

export function SftTaskSection({
  catalogue,
  selected,
  onChange,
  mmluEpochs,
  onMmluEpochsChange,
  gsm8kEpochs,
  onGsM8kEpochsChange,
  disabled,
}: {
  catalogue: NanochatCatalogue | null;
  selected: string[];
  onChange: (next: string[]) => void;
  mmluEpochs: number;
  onMmluEpochsChange: (value: number) => void;
  gsm8kEpochs: number;
  onGsM8kEpochsChange: (value: number) => void;
  disabled?: boolean;
}) {
  const tasks = catalogue?.training_tasks ?? {};
  const entries = Object.values(tasks);

  const toggle = (key: string) => {
    onChange(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  };

  return (
    <section className="flex flex-col gap-3">
      <header className="flex items-center gap-2">
        <HugeiconsIcon icon={Task01Icon} className="size-4 text-muted-foreground" />
        <h3 className="font-heading text-ui-15 font-semibold">
          Fine-tuning data
        </h3>
      </header>

      {entries.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Using nanochat&apos;s default mixture (SmolTalk, MMLU ×3, GSM8K ×4).
        </p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {entries.map((task) => (
            <label
              key={task.key}
              className="flex items-start gap-2.5 rounded-md border border-transparent p-1.5 text-xs transition-colors hover:bg-accent/40"
            >
              <Checkbox
                checked={selected.includes(task.key)}
                disabled={disabled}
                onCheckedChange={() => toggle(task.key)}
                className="mt-0.5"
              />
              <span className="flex flex-col gap-0.5">
                <span className="flex items-center gap-1.5">
                  {task.label}
                  {task.is_default ? (
                    <Badge variant="outline" className="text-[10px]">
                      default
                    </Badge>
                  ) : null}
                </span>
                <span className="text-[11px] text-muted-foreground">
                  {task.teaches}
                </span>
                <span className="text-[11px] text-muted-foreground/70">
                  ~{task.rows.toLocaleString()} rows per epoch
                </span>
              </span>
            </label>
          ))}
        </div>
      )}

      <div className="grid grid-cols-2 gap-2.5">
        <EpochField
          id="nanochat-mmlu-epochs"
          label="MMLU epochs"
          value={mmluEpochs}
          onChange={onMmluEpochsChange}
          disabled={disabled || !selected.includes("mmlu")}
        />
        <EpochField
          id="nanochat-gsm8k-epochs"
          label="GSM8K epochs"
          value={gsm8kEpochs}
          onChange={onGsM8kEpochsChange}
          disabled={disabled || !selected.includes("gsm8k")}
        />
      </div>
    </section>
  );
}

function EpochField({
  id,
  label,
  value,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Label htmlFor={id} className="text-xs">
        {label}
      </Label>
      <Input
        id={id}
        type="number"
        min={0}
        max={100}
        value={value}
        disabled={disabled}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) onChange(Math.max(0, Math.min(100, Math.round(next))));
        }}
        className="h-8"
      />
    </div>
  );
}

export function BenchmarkSection({
  catalogue,
  chatBenchmarks,
  onChatBenchmarksChange,
  baseBenchmarks,
  onBaseBenchmarksChange,
  disabled,
}: {
  catalogue: NanochatCatalogue | null;
  chatBenchmarks: string[];
  onChatBenchmarksChange: (next: string[]) => void;
  baseBenchmarks: string[];
  onBaseBenchmarksChange: (next: string[]) => void;
  disabled?: boolean;
}) {
  const benchmarks = catalogue?.benchmarks ?? {};
  const baseEvals = catalogue?.base_evals ?? {};
  const trainedOn = new Set(catalogue?.trained_on_by_default ?? []);

  const toggle = (list: string[], key: string, apply: (next: string[]) => void) => {
    apply(list.includes(key) ? list.filter((k) => k !== key) : [...list, key]);
  };

  const chatEntries = Object.values(benchmarks);
  const baseEntries = Object.entries(baseEvals);

  return (
    <section className="flex flex-col gap-3">
      <header>
        <h3 className="font-heading text-ui-15 font-semibold">Benchmarks</h3>
        <p className="text-[11px] text-muted-foreground">
          Evaluated during and after training. Nothing here is trained on, except
          where noted.
        </p>
      </header>

      {chatEntries.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium">After fine-tuning</span>
          {chatEntries.map((spec) => {
            const alsoTrained = trainedOn.has(spec.key);
            return (
              <label
                key={spec.key}
                className="flex items-start gap-2.5 rounded-md p-1.5 text-xs transition-colors hover:bg-accent/40"
              >
                <Checkbox
                  checked={chatBenchmarks.includes(spec.key)}
                  disabled={disabled}
                  onCheckedChange={() =>
                    toggle(chatBenchmarks, spec.key, onChatBenchmarksChange)
                  }
                  className="mt-0.5"
                />
                <span className="flex flex-col gap-0.5">
                  <span className="flex flex-wrap items-center gap-1.5">
                    {spec.label}
                    {alsoTrained ? (
                      <Badge
                        variant="outline"
                        className="border-amber-500/40 text-[10px] text-amber-600 dark:text-amber-400"
                      >
                        also in the SFT mixture
                      </Badge>
                    ) : null}
                    {spec.requires_posix_sandbox ? (
                      <Badge variant="outline" className="text-[10px]">
                        runs generated code
                      </Badge>
                    ) : null}
                  </span>
                  <span className="text-[11px] text-muted-foreground">
                    {spec.notes} Random guessing scores{" "}
                    {(spec.baseline * 100).toFixed(0)}%.
                  </span>
                </span>
              </label>
            );
          })}
          <p className="text-[11px] text-muted-foreground/80">
            Scoring all five is what the published ChatCORE number is. Selecting a
            subset still reports a centred mean, but it is labelled as partial
            because it is not comparable to the full number.
          </p>
        </div>
      ) : null}

      {baseEntries.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium">After pretraining</span>
          {baseEntries.map(([key, spec]) => (
            <label
              key={key}
              className="flex items-start gap-2.5 rounded-md p-1.5 text-xs transition-colors hover:bg-accent/40"
            >
              <Checkbox
                checked={baseBenchmarks.includes(key)}
                disabled={disabled}
                onCheckedChange={() =>
                  toggle(baseBenchmarks, key, onBaseBenchmarksChange)
                }
                className="mt-0.5"
              />
              <span className="flex flex-col gap-0.5">
                <span>{spec.label}</span>
                <span className="text-[11px] text-muted-foreground">{spec.notes}</span>
              </span>
            </label>
          ))}
        </div>
      ) : null}
    </section>
  );
}
