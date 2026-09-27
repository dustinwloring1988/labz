// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Having a model write up what the search found.
 *
 * The ledger says what happened: a number per experiment, kept or discarded. It
 * does not say what the search *found*, which is the question someone actually
 * has the next morning. So the report is a language task, and it is given to a
 * language model the user picks.
 *
 * The two sources are offered side by side because they are good for different
 * things, and pretending otherwise would be the whole failure mode of this
 * feature. A report is a summarisation task over a table of numbers, and a
 * 50M-parameter model that spent five minutes on TinyStories will write fluent,
 * confident nonsense about it. So the catalog models are the default, and the
 * checkpoints are there for when nothing else is available â€” labelled as what
 * they are, and told in the prompt that they are reporting on their own training
 * run, so the output is never mistaken for an assessment of it.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { Copy01Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";

import { copyToClipboard } from "@/lib/copy-to-clipboard";

import {
  loadAutoresearchReport,
  streamAutoresearchReport,
} from "../api/autoresearch-api";
import {
  reportRequestFor,
  reportSourceSuffix,
  useReportModels,
} from "../hooks/use-report-models";
import { parseAgentArgs } from "../stores/autoresearch-config-policy";
import { useAutoresearchConfigStore } from "../stores/autoresearch-config-store";
import { useAutoresearchRuntimeStore } from "../stores/autoresearch-runtime-store";
import { formatBpb } from "../lib/format";

export function AutoresearchReportView() {
  const resultsLoaded = useAutoresearchRuntimeStore((state) => state.resultsLoaded);
  const totals = useAutoresearchRuntimeStore((state) => state.resultsTotals);
  // The report the loop wrote as its last phase, if it was asked to. Shown
  // before anything is generated so the sequence a user asked for -- experiment,
  // then research -- lands on the finished thing rather than an empty tab.
  const autoReport = useAutoresearchRuntimeStore((state) => state.reportText);
  const autoReportModel = useAutoresearchRuntimeStore((state) => state.reportModel);
  const autoReportSaved = useAutoresearchRuntimeStore((state) => state.reportSavedTo);
  const autoReportError = useAutoresearchRuntimeStore((state) => state.reportError);
  const autoReportRunning = useAutoresearchRuntimeStore((state) => state.reportRunning);

  const { models: allModels, loading, catalogEmpty, agentsEmpty } = useReportModels();
  const config = useAutoresearchConfigStore();

  const [selected, setSelected] = useState<string>("");
  const [report, setReport] = useState<string>("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [savedTo, setSavedTo] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [copiedAuto, setCopiedAuto] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => () => abort.current?.abort(), []);

  // The selection falls back during render rather than being repaired by an
  // effect. The model list arrives after the first paint, so there is a window
  // where whatever the user picked is not in the list, and an effect would leave
  // the picker showing a model that does not exist for a frame.
  const selection = allModels.some((model) => model.id === selected) ? selected : "";
  const chosen = allModels.find((model) => model.id === selection) ?? null;

  const handleGenerate = useCallback(async () => {
    if (!chosen) return;
    setGenerating(true);
    setReport("");
    setError(null);
    setWarning(null);
    setSavedTo(null);
    setCopied(false);

    const controller = new AbortController();
    abort.current = controller;
    try {
      const request = reportRequestFor(chosen);
      await streamAutoresearchReport({
        modelId: request.modelId,
        source: request.source,
        agentKey: request.agentKey,
        // The report inherits the loop's agent arguments when the loop has any, so
        // "write it with the same local model" needs no second field.
        agentArgs: parseAgentArgs(config.agentArgs),
        maxTokens: 2000,
        save: true,
        signal: controller.signal,
        onText: (cumulative) => setReport(cumulative),
        onWarning: (message) => setWarning(message),
        onDone: (payload) => setSavedTo(payload.saved_to),
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setGenerating(false);
      abort.current = null;
    }
  }, [chosen, config.agentArgs]);

  const handleStop = useCallback(() => abort.current?.abort(), []);

  const handleOpen = useCallback(async () => {
    setError(null);
    try {
      const saved = await loadAutoresearchReport();
      setReport(saved.text);
      setSavedTo(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, []);

  const handleCopy = useCallback(async () => {
    const copiedOk = await copyToClipboard(report);
    setCopied(copiedOk);
    if (copiedOk) {
      setTimeout(() => setCopied(false), 2000);
    }
  }, [report]);

  const handleCopyAuto = useCallback(async () => {
    const copiedOk = await copyToClipboard(autoReport);
    setCopiedAuto(copiedOk);
    if (copiedOk) {
      setTimeout(() => setCopiedAuto(false), 2000);
    }
  }, [autoReport]);

  if (!resultsLoaded) {
    return (
      <div className="flex justify-center py-12">
        <Spinner />
      </div>
    );
  }

  if (totals.total === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-16 text-center">
        <p className="text-sm text-muted-foreground">Nothing to report on yet.</p>
        <p className="max-w-md text-xs text-muted-foreground/80">
          A report is written from the experiment ledger, so it needs at least one
          recorded experiment.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 pb-8">
      <SectionCard
        icon={<HugeiconsIcon icon={SparklesIcon} className="size-5" />}
        title="Write a report"
        description={`${totals.total} experiments, best val bpb ${formatBpb(totals.bestValBpb)}. The model is given the whole ledger and asked what the search found.`}
      >
        <div className="flex flex-col gap-3">
          {allModels.length === 0 ? (
            loading ? (
              <p className="text-ui-11 text-muted-foreground">Looking for models…</p>
            ) : (
              <p className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-xs text-amber-700 dark:text-amber-400">
                {agentsEmpty ? "No agent is runnable. " : ""}
                {catalogEmpty
                  ? "No catalog model is available either, so the only choices are this search's own checkpoints — the experiment describing itself rather than assessing it."
                  : "Load a model in Chat, or run the loop so there is a checkpoint of its own."}
              </p>
            )
          ) : (
            <>
              <div className="flex flex-col gap-1.5">
                <label htmlFor="autoresearch-report-model" className="text-xs font-medium">
                  Write it with
                </label>
                <select
                  id="autoresearch-report-model"
                  className="h-9 rounded-md border border-input bg-transparent px-3 text-sm"
                  value={selection}
                  disabled={generating}
                  onChange={(event) => setSelected(event.target.value)}
                >
                  {allModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.label}
                      {reportSourceSuffix(model)}
                    </option>
                  ))}
                </select>
                {chosen ? (
                  <p className="text-[11px] text-muted-foreground">{chosen.note}</p>
                ) : catalogEmpty && !agentsEmpty ? (
                  <p className="text-[11px] text-muted-foreground">
                    No catalog model is available, so the choices are the agents
                    and this search&apos;s own checkpoints. An agent usually has a
                    local model behind it; a checkpoint is the experiment
                    describing itself rather than assessing it.
                  </p>
                ) : null}
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button
                  disabled={!chosen || generating}
                  onClick={() => void handleGenerate()}
                  className="gap-1.5"
                >
                  {generating ? <Spinner className="size-4" /> : null}
                  {generating ? "Writingâ€¦" : "Write the report"}
                </Button>
                {generating ? (
                  <Button variant="outline" onClick={handleStop}>
                    Stop
                  </Button>
                ) : null}
                <Button variant="secondary" onClick={() => void handleOpen()}>
                  Open the last one
                </Button>
                {report && !generating ? (
                  <>
                    <Button variant="outline" onClick={() => void handleCopy()}>
                      <HugeiconsIcon icon={Copy01Icon} className="mr-1.5 size-4" />
                      {copied ? "Copied" : "Copy"}
                    </Button>
                    {savedTo ? (
                      <span className="text-[11px] text-muted-foreground">
                        saved to {savedTo}
                      </span>
                    ) : null}
                  </>
                ) : null}
              </div>
            </>
          )}

          {error ? (
            <p className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
              {error}
            </p>
          ) : null}
          {warning ? (
            <p className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-xs text-amber-700 dark:text-amber-400">
              {warning}
            </p>
          ) : null}
        </div>
      </SectionCard>

      {report || generating ? (
        <SectionCard
          icon={<HugeiconsIcon icon={SparklesIcon} className="size-5" />}
          title="The report"
        description={
          generating
            ? "Arriving as it is written."
            : chosen
              ? `Written by ${chosen.label}.`
              : "A report written from the experiment ledger."
        }
      >
        {generating && !report ? (
          <div className="flex justify-center py-8">
            <Spinner />
          </div>
        ) : (
          <div className="min-h-48">
            <MarkdownPreview markdown={report} />
          </div>
        )}
      </SectionCard>
      ) : null}

      {/* The report the loop wrote as its own last phase, shown above anything
          generated by hand. It is the end of the sequence a one-GPU machine
          forces -- experiments, then the GPU, then the writing -- so it is what
          the tab should open on rather than an empty state and a button. */}
      {autoReportRunning ? (
        <SectionCard
          icon={<Spinner className="size-5" />}
          accent="blue"
          title="Writing the report"
          description={`The loop handed the GPU to ${autoReportModel || "the model you chose"}. This is its last step.`}
        >
          <div className="flex justify-center py-8">
            <Spinner />
          </div>
        </SectionCard>
      ) : null}

      {autoReportError && !autoReport ? (
        <p className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-xs text-amber-700 dark:text-amber-400">
          The experiments finished, but the report could not be written:{" "}
          {autoReportError}. The ledger is intact, so you can write one by hand below.
        </p>
      ) : null}

      {autoReport ? (
        <SectionCard
          icon={<HugeiconsIcon icon={SparklesIcon} className="size-5" />}
          title="The report"
          description={
            autoReportModel
              ? `Written by ${autoReportModel} as the loop's final step.`
              : "Written as the loop's final step."
          }
          badge="from the loop"
        >
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
            <Button variant="outline" size="sm" onClick={() => void handleCopyAuto()}>
              <HugeiconsIcon icon={Copy01Icon} className="mr-1.5 size-4" />
              {copiedAuto ? "Copied" : "Copy"}
            </Button>
            {autoReportSaved ? <span>saved to {autoReportSaved}</span> : null}
          </div>
          <div className="min-h-48">
            <MarkdownPreview markdown={autoReport} />
          </div>
        </SectionCard>
      ) : null}
    </div>
  );
}
