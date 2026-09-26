// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Configure tab: choose a depth, a corpus, tasks, benchmarks and parameters.
 *
 * Validation is a debounced call rather than a per-keystroke one, and the reply
 * is discarded if a newer request has come back — otherwise a slow response for
 * depth 12 can land after a fast one for depth 24 and overwrite it, which would
 * show a fit verdict for a model the user is no longer looking at.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { Alert02Icon, DownloadIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import {
  getNanochatEnvironment,
  getNanochatFit,
  getNanochatCatalogue,
  getNanochatPresets,
  installNanochat,
  startNanochatRun,
  validateNanochatConfig,
} from "../api/nanochat-api";
import { useNanochatRuntimeStore } from "../stores/nanochat-runtime-store";
import { useNanochatConfigStore } from "../stores/nanochat-config-store";
import type { NanochatCatalogue, NanochatDepthPreset } from "../types";
import { DepthSelector } from "./depth-selector";
import { BenchmarkSection, DatasetSection, SftTaskSection } from "./pickers";
import { ParamsSection } from "./params-section";
import { RunPreviewCard } from "./run-preview-card";

const VALIDATE_DEBOUNCE_MS = 400;

export function NanochatConfigureView({ onStarted }: { onStarted: () => void }) {
  // useNanochatConfigStore() with no selector subscribes the component to the
  // whole store, which is what gives it every field. The updater is pulled out
  // separately so the component does not re-render on an unrelated field.
  const config = useNanochatConfigStore();
  const setConfig = useNanochatConfigStore((state) => state.apply);

  const environment = useNanochatRuntimeStore((state) => state.environment);
  const setEnvironment = useNanochatRuntimeStore((state) => state.setEnvironment);
  const setFit = useNanochatRuntimeStore((state) => state.setFit);
  const setValidation = useNanochatRuntimeStore((state) => state.setValidation);
  const setValidating = useNanochatRuntimeStore((state) => state.setValidating);
  const markStarted = useNanochatRuntimeStore((state) => state.markStarted);
  // Subscribed, not read through getState(): a fit verdict that arrives after the
  // depth changes has to cause a re-render, or the panel keeps showing the
  // previous depth's numbers.
  const fit = useNanochatRuntimeStore((state) => state.fit);
  const validationErrors = useNanochatRuntimeStore((state) => state.validationErrors);
  const validationWarnings = useNanochatRuntimeStore((state) => state.validationWarnings);
  const isValidating = useNanochatRuntimeStore((state) => state.isValidating);

  const [presets, setPresets] = useState<NanochatDepthPreset[] | null>(null);
  const [catalogue, setCatalogue] = useState<NanochatCatalogue | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const validationAbort = useRef<AbortController | null>(null);
  const fitAbort = useRef<AbortController | null>(null);

  // --- one-time loads ---
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [env, presetResponse, cat] = await Promise.all([
          getNanochatEnvironment(),
          getNanochatPresets(),
          getNanochatCatalogue(),
        ]);
        if (cancelled) return;
        setEnvironment(env);
        setPresets(presetResponse.presets);
        setCatalogue(cat);
      } catch {
        // The tab still renders without these; each section explains what is
        // missing rather than the whole page failing.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setEnvironment]);

  // --- fit for the current depth and batch ---
  useEffect(() => {
    fitAbort.current?.abort();
    const controller = new AbortController();
    fitAbort.current = controller;
    void getNanochatFit({
      depth: config.depth,
      numIterations: config.numIterations,
      totalBatchSize: config.totalBatchSize,
      maxSeqLen: config.maxSeqLen,
      signal: controller.signal,
    })
      .then(setFit)
      .catch(() => {
        /* aborted or unavailable; the last verdict stands */
      });
    return () => controller.abort();
  }, [config.depth, config.numIterations, config.totalBatchSize, config.maxSeqLen, setFit]);

  // --- validation, debounced ---
  useEffect(() => {
    validationAbort.current?.abort();
    const controller = new AbortController();
    validationAbort.current = controller;
    const timer = setTimeout(() => {
      setValidating(true);
      void validateNanochatConfig(config, controller.signal)
        .then((result) => {
          if (controller.signal.aborted) return;
          setValidation([...result.errors, ...result.config_errors], result.warnings);
        })
        .catch(() => {
          /* aborted */
        })
        .finally(() => {
          if (!controller.signal.aborted) setValidating(false);
        });
    }, VALIDATE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [config, setValidation, setValidating]);

  const handleStart = useCallback(async () => {
    setIsStarting(true);
    setStartError(null);
    try {
      const response = await startNanochatRun(config);
      markStarted(response.run_id);
      onStarted();
    } catch (error) {
      setStartError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsStarting(false);
    }
  }, [config, markStarted, onStarted]);

  const installing = environment?.install_state === "installing";
  const notReady = environment !== null && !environment.ready && !installing;

  return (
    <div className="@container/nanochat-configure">
      {notReady ? (
        // Rendered inline rather than returned early: an early return would put
        // the hooks below it behind a conditional, and React requires the same
        // hook order on every render.
        <EnvironmentGate
          environment={environment}
          onInstall={async () => {
            try {
              setEnvironment(await installNanochat());
            } catch {
              /* surfaced by the poll inside the gate */
            }
          }}
        />
      ) : (
        <div className="grid grid-cols-1 gap-8 @4xl/nanochat-configure:grid-cols-[minmax(0,1fr)_320px] @5xl/nanochat-configure:gap-10">
        <div className="flex min-w-0 flex-col gap-8">
          <section className="flex flex-col gap-3">
            <header className="flex flex-col gap-0.5">
              <h2 className="font-heading text-ui-15 font-semibold">Model size</h2>
              <p className="text-xs text-muted-foreground">
                Depth is nanochat&apos;s only architectural dial. It sets how wide the
                model is and how many layers it has.
              </p>
            </header>
            <DepthSelector
              depth={config.depth}
              onDepthChange={(depth) => setConfig({ depth })}
              presets={presets ?? undefined}
              fit={fit}
            />
          </section>

          <DatasetSection
            catalogue={catalogue}
            dataset={config.dataset}
            onDatasetChange={(dataset) => setConfig({ dataset })}
            numShards={config.numShards}
            onNumShardsChange={(numShards) => setConfig({ numShards })}
            vocabSize={config.vocabSize}
            onVocabSizeChange={(vocabSize) => setConfig({ vocabSize })}
            trainTokenizer={config.trainTokenizer}
            onTrainTokenizerChange={(trainTokenizer) => setConfig({ trainTokenizer })}
          />

          <ParamsSection
            config={config}
            set={setConfig}
            paramMode={config.paramMode}
            onParamModeChange={(paramMode) => setConfig({ paramMode })}
          />

          <SftTaskSection
            catalogue={catalogue}
            selected={config.sftTasks}
            onChange={(sftTasks) => setConfig({ sftTasks })}
            mmluEpochs={config.mmluEpochs}
            onMmluEpochsChange={(mmluEpochs) => setConfig({ mmluEpochs })}
            gsm8kEpochs={config.gsm8kEpochs}
            onGsM8kEpochsChange={(gsm8kEpochs) => setConfig({ gsm8kEpochs })}
          />

          <BenchmarkSection
            catalogue={catalogue}
            chatBenchmarks={config.chatBenchmarks}
            onChatBenchmarksChange={(chatBenchmarks) => setConfig({ chatBenchmarks })}
            baseBenchmarks={config.baseBenchmarks}
            onBaseBenchmarksChange={(baseBenchmarks) => setConfig({ baseBenchmarks })}
          />
        </div>

        <div className="@4xl/nanochat-configure:sticky @4xl/nanochat-configure:top-6 @4xl/nanochat-configure:self-start">
          <RunPreviewCard
            config={config}
            fit={fit}
            errors={validationErrors}
            warnings={validationWarnings}
            isValidating={isValidating}
            isStarting={isStarting}
            canStart={!notReady}
            onStart={handleStart}
          />
          {startError ? (
            <p className="mt-2 text-xs text-destructive">{startError}</p>
          ) : null}
        </div>
      </div>
      )}
    </div>
  );
}

function EnvironmentGate({
  environment,
  onInstall,
}: {
  environment: ReturnType<typeof useNanochatRuntimeStore.getState>["environment"];
  onInstall: () => void;
}) {
  const installing = environment?.install_state === "installing";
  const failed = environment?.install_state === "failed";
  const progress = environment?.install_progress ?? 0;
  const message = environment?.install_message ?? "";

  // Poll while installing: the install is a multi-gigabyte download running in a
  // background thread, and nothing else would notice when it finishes.
  useEffect(() => {
    if (!installing) return;
    const timer = setInterval(() => {
      void getNanochatEnvironment()
        .then((status) => useNanochatRuntimeStore.getState().setEnvironment(status))
        .catch(() => undefined);
    }, 2_000);
    return () => clearInterval(timer);
  }, [installing]);

  return (
    <div className="mx-auto flex max-w-lg flex-col items-center gap-4 py-12 text-center">
      <div className="flex flex-col items-center gap-2">
        <h2 className="font-heading text-ui-20 font-semibold">Set up nanochat</h2>
        <p className="text-sm text-muted-foreground">
          nanochat is a separate project with its own pinned dependencies, so it gets
          its own copy and its own environment rather than sharing the one the rest
          of the app runs in.
        </p>
      </div>

      {environment?.blocking_reason ? (
        <p className="flex items-start gap-1.5 text-xs text-muted-foreground">
          <HugeiconsIcon icon={Alert02Icon} className="mt-px size-3.5 shrink-0 text-amber-500" />
          <span>{environment.blocking_reason}</span>
        </p>
      ) : null}

      {installing ? (
        <div className="flex w-full flex-col gap-2">
          <Progress value={Math.round(progress * 100)} />
          <p className="text-xs text-muted-foreground">
            {message || "Working…"} ({Math.round(progress * 100)}%)
          </p>
          <p className="text-[11px] text-muted-foreground/80">
            This downloads PyTorch, which is a few gigabytes. It happens once.
          </p>
        </div>
      ) : (
        <Button onClick={onInstall} className="gap-1.5">
          <HugeiconsIcon icon={DownloadIcon} className="size-4" />
          Download and set up nanochat
        </Button>
      )}

      {failed ? (
        <p className="text-xs text-destructive">{message || "Setup failed."}</p>
      ) : null}

      {installing ? <Spinner /> : null}
    </div>
  );
}
