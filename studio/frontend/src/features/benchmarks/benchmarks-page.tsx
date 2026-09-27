// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactElement,
} from "react";

import {
  buildBenchmarkStartPayload,
  getBenchmarkCatalogue,
  getBenchmarkModels,
  getBenchmarkStatus,
  installNanochat,
  startBenchmarkRun,
  stopBenchmarkRun,
} from "./api/benchmarks-api";
import { BenchmarkSubNav } from "./benchmarks-navigation";
import { useBenchmarkRuntimeLifecycle } from "./hooks/use-benchmark-runtime-lifecycle";
import {
  BenchmarkConfigureForm,
  BenchmarkStartCta,
} from "./sections/configure-view";
import { BenchmarkLeaderboardView } from "./sections/leaderboard-view";
import { BenchmarkLiveRunView } from "./sections/live-run-view";
import { useBenchmarkConfigStore } from "./stores/benchmark-config-store";
import { validateBenchmarkForm } from "./stores/benchmark-config-policy";
import {
  isBenchmarkRunActive,
  isBenchmarkRunFinished,
  hasBenchmarkRun,
  useBenchmarkRuntimeStore,
} from "./stores/benchmark-runtime-store";
import type { BenchmarkStatus } from "./types";

export type BenchmarkTab = "configure" | "current-run" | "leaderboard";

/**
 * The benchmark tab host.
 *
 * The frame is the Train frame, class for class: same measure, same header scale,
 * same sub-nav. What differs is the third tab, because a benchmark leaves a
 * durable ranked result rather than a list of runs.
 */
export function BenchmarkPage(): ReactElement {
  useBenchmarkRuntimeLifecycle();

  const runActive = useBenchmarkRuntimeStore(isBenchmarkRunActive);
  const hasRun = useBenchmarkRuntimeStore(isBenchmarkRunFinished);
  // The sub-nav's trigger follows "a run exists", not "a run produced scores": a
  // run in its first second has no scores yet, and keying the trigger off results
  // would grey out the tab the user was just moved to.
  const canViewRun = useBenchmarkRuntimeStore(hasBenchmarkRun);

  // Selected one field at a time, then assembled below.
  //
  // Not one selector returning an object literal: a fresh object is a new
  // reference on every call, so useSyncExternalStore sees the snapshot change on
  // every render and React re-renders forever ("getSnapshot should be cached",
  // then "maximum update depth exceeded"). Each of these returns a primitive or
  // a reference the store only replaces when it actually changes.
  const runId = useBenchmarkRuntimeStore((state) => state.runId);
  const runStatus = useBenchmarkRuntimeStore((state) => state.status);
  const runPhase = useBenchmarkRuntimeStore((state) => state.phase);
  const runMessage = useBenchmarkRuntimeStore((state) => state.message);
  const runError = useBenchmarkRuntimeStore((state) => state.error);
  const modelId = useBenchmarkRuntimeStore((state) => state.modelId);
  const modelLabel = useBenchmarkRuntimeStore((state) => state.modelLabel);
  const modelFormat = useBenchmarkRuntimeStore((state) => state.modelFormat);
  const loraPath = useBenchmarkRuntimeStore((state) => state.loraPath);
  const loadIn4bit = useBenchmarkRuntimeStore((state) => state.loadIn4bit);
  const judgeModelId = useBenchmarkRuntimeStore((state) => state.judgeModelId);
  const judgeModelLabel = useBenchmarkRuntimeStore(
    (state) => state.judgeModelLabel,
  );
  const requested = useBenchmarkRuntimeStore((state) => state.requested);
  const current = useBenchmarkRuntimeStore((state) => state.current);
  const scores = useBenchmarkRuntimeStore((state) => state.scores);
  const composite = useBenchmarkRuntimeStore((state) => state.composite);
  const scoringMode = useBenchmarkRuntimeStore((state) => state.scoringMode);
  const progressPercent = useBenchmarkRuntimeStore(
    (state) => state.progressPercent,
  );
  const elapsedSeconds = useBenchmarkRuntimeStore(
    (state) => state.elapsedSeconds,
  );
  const startedAt = useBenchmarkRuntimeStore((state) => state.startedAt);
  const endedAt = useBenchmarkRuntimeStore((state) => state.endedAt);
  const logLines = useBenchmarkRuntimeStore((state) => state.logLines);

  // Rebuilt only when one of the parts changes, so the live view sees a stable
  // prop and its own memoisation is not defeated.
  const status = useMemo<BenchmarkStatus | null>(
    () =>
      runId === null
        ? null
        : {
            run_id: runId,
            status: runStatus,
            phase: runPhase,
            message: runMessage,
            error: runError,
            model_id: modelId,
            model_label: modelLabel,
            format: modelFormat,
            lora_path: loraPath,
            load_in_4bit: loadIn4bit,
            judge_model_id: judgeModelId,
            judge_model_label: judgeModelLabel,
            requested,
            current,
            scores,
            composite,
            scoring_mode: scoringMode,
            progress_percent: progressPercent,
            elapsed_seconds: elapsedSeconds,
            started_at: startedAt,
            ended_at: endedAt,
            logs: logLines,
          },
    [
      runId,
      runStatus,
      runPhase,
      runMessage,
      runError,
      modelId,
      modelLabel,
      modelFormat,
      loraPath,
      loadIn4bit,
      judgeModelId,
      judgeModelLabel,
      requested,
      current,
      scores,
      composite,
      scoringMode,
      progressPercent,
      elapsedSeconds,
      startedAt,
      endedAt,
      logLines,
    ],
  );
  const catalogue = useBenchmarkRuntimeStore((state) => state.catalogue);
  const models = useBenchmarkRuntimeStore((state) => state.models);
  const modelsLoading = useBenchmarkRuntimeStore(
    (state) => state.modelsLoading,
  );
  const modelsEmptyReason = useBenchmarkRuntimeStore(
    (state) => state.modelsEmptyReason,
  );
  const actionError = useBenchmarkRuntimeStore((state) => state.actionError);
  const isStarting = useBenchmarkRuntimeStore((state) => state.isStarting);

  const form = useBenchmarkConfigStore((state) => state.form);
  const setBenchmarks = useBenchmarkConfigStore((state) => state.setBenchmarks);
  const toggleBenchmark = useBenchmarkConfigStore(
    (state) => state.toggleBenchmark,
  );

  // A run that is going, or one whose scores are worth looking at, opens on
  // Current Run. Otherwise the tab is a form, and that is what should be showing.
  const [tab, setTab] = useState<BenchmarkTab>(
    runActive || hasRun ? "current-run" : "configure",
  );
  const [installing, setInstalling] = useState(false);

  // The catalogue is the authority on which benchmarks exist, so it is fetched
  // once on mount and the form's stored selection is filtered through it. A
  // benchmark that has disappeared upstream must not be resent, or the start
  // request 400s on a key the user cannot see or remove.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const store = useBenchmarkRuntimeStore.getState();
      store.setCatalogueLoading(true);
      try {
        const [next, modelList] = await Promise.all([
          getBenchmarkCatalogue(),
          getBenchmarkModels().catch(() => null),
        ]);
        if (cancelled) return;
        store.setCatalogue(next);
        if (modelList) store.setModels(modelList);
        const order = next.benchmarks.map((spec) => spec.key);
        // Seed the selection from the backend's own defaults the first time,
        // rather than starting on an empty picker the user has to fill in.
        const current = useBenchmarkConfigStore.getState().form.benchmarkKeys;
        if (current.length === 0 && next.default_benchmarks.length > 0) {
          setBenchmarks(next.default_benchmarks, order);
        } else {
          setBenchmarks(current, order);
        }
      } catch (cause) {
        if (cancelled) return;
        store.setCatalogueLoading(false);
        store.setActionError(
          cause instanceof Error ? cause.message : String(cause),
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setBenchmarks]);

  const specs = useMemo(() => catalogue?.benchmarks ?? [], [catalogue]);
  const order = useMemo(() => specs.map((spec) => spec.key), [specs]);
  const supportedKeys = useMemo(
    () => specs.filter((spec) => spec.supported).map((spec) => spec.key),
    [specs],
  );

  const problems = useMemo(
    () => validateBenchmarkForm(form, supportedKeys),
    [form, supportedKeys],
  );
  const problemsByField = useMemo(() => {
    const map: Record<string, string> = {};
    for (const problem of problems) map[problem.field] = problem.message;
    return map;
  }, [problems]);

  const start = useCallback(async () => {
    const store = useBenchmarkRuntimeStore.getState();
    const model = store.models.find((entry) => entry.path === form.modelPath);
    if (!model) {
      store.setActionError("Choose a model to benchmark.");
      return;
    }
    store.setActionError(null);
    store.setStarting(true);
    try {
      // A judge is only sent when a generative benchmark would actually use it.
      // A judge left selected from an earlier suite is not an error, and sending
      // it anyway would put a model in the config that no pass ever loads.
      const judgeModel = form.judgeModelPath
        ? (store.models.find((entry) => entry.path === form.judgeModelPath) ??
          null)
        : null;
      const started = await startBenchmarkRun(
        buildBenchmarkStartPayload({
          model,
          benchmarks: form.benchmarkKeys,
          maxProblems: form.maxProblems,
          batchSize: form.batchSize,
          maxNewTokens: form.maxNewTokens,
          maxSeqLength: form.maxSeqLength,
          loadIn4bit: form.loadIn4bit,
          judgeModel,
        }),
      );
      // Adopt the run into the store before switching tabs.
      //
      // The store is otherwise written in exactly two places: the lifecycle
      // hook's mount-time hydration, and the progress stream. The hydration ran
      // before this run existed (and got the 404 that means "nothing has ever
      // run"), and the stream only opens once the store already has a running
      // run -- so without this the tab would switch to a run it knows nothing
      // about and sit on "No run yet" for the whole run.
      //
      // The orchestrator registers the run before the start request returns, so
      // this cannot race it. It is scoped to the new run's id rather than
      // unfiltered, so a superseded run cannot overwrite the current one.
      try {
        useBenchmarkRuntimeStore
          .getState()
          .hydrate(await getBenchmarkStatus(started.run_id));
      } catch {
        // The run is going regardless; the next mount hydrates it, and failing
        // here must not report a start that did happen as a failed one.
      }
      setTab("current-run");
    } catch (cause) {
      store.setActionError(
        cause instanceof Error ? cause.message : String(cause),
      );
    } finally {
      store.setStarting(false);
    }
  }, [form]);

  const stop = useCallback(() => {
    useBenchmarkRuntimeStore.getState().markStopRequested();
    void stopBenchmarkRun().catch(() => undefined);
  }, []);

  const handleInstall = useCallback(async () => {
    setInstalling(true);
    try {
      await installNanochat();
      const next = await getBenchmarkCatalogue();
      useBenchmarkRuntimeStore.getState().setCatalogue(next);
    } catch (cause) {
      useBenchmarkRuntimeStore
        .getState()
        .setActionError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setInstalling(false);
    }
  }, []);

  const subtitle =
    tab === "current-run"
      ? "Watch a suite as it scores, then read the numbers"
      : tab === "leaderboard"
        ? "Every model you have benchmarked here, ranked"
        : "Score a model against common benchmarks and rank it against your others";

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <Tabs
        value={tab}
        onValueChange={(value) => setTab(value as BenchmarkTab)}
        className="contents"
      >
        <div className="relative mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] 3xl:max-w-[calc(1440px*var(--ui-space-scale,1))] 4xl:max-w-[calc(1760px*var(--ui-space-scale,1))] flex-col gap-7 px-5 pb-20 pt-8 max-sm:px-4 sm:px-9 sm:pt-10">
          <header className="font-heading flex flex-col gap-5">
            <div className="flex flex-col gap-0.5">
              <h1 className="page-title-halo text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
                Benchmarks
              </h1>
              <p className="page-title-halo text-sm text-muted-foreground">
                {subtitle}
              </p>
            </div>
            <div className="flex min-w-0 flex-wrap items-center gap-3 border-b border-border/60">
              <BenchmarkSubNav
                value={tab}
                runActive={runActive}
                hasRun={canViewRun}
              />
              {status?.model_label && tab === "current-run" ? (
                <div className="ml-auto flex items-center gap-2 pb-2">
                  <span className="rounded-full border border-border/60 px-2.5 py-1 font-mono text-ui-10 text-muted-foreground">
                    run {status.run_id}
                  </span>
                </div>
              ) : null}
            </div>
          </header>

          <div className="flex w-full flex-col gap-6">
            <TabsContent value="configure" className="mt-0">
              {catalogue && !catalogue.available ? (
                // The datasets live in nanochat's environment, so a host without it
                // cannot benchmark anything. Said here rather than as a failure
                // after the user has filled the form in.
                <div className="flex flex-col items-center gap-3 rounded-3xl border border-dashed border-border/70 px-6 py-12 text-center">
                  <p className="text-sm font-medium text-foreground">
                    Benchmarks need the nanochat environment
                  </p>
                  <p className="max-w-md text-sm text-muted-foreground">
                    {catalogue.reason ??
                      "It provides the benchmark questions and the grading rules, and is downloaded once."}
                  </p>
                  <button
                    type="button"
                    disabled={installing}
                    onClick={() => void handleInstall()}
                    className="mt-1 flex h-9 items-center rounded-full bg-foreground px-4 text-ui-13 font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-40"
                  >
                    {installing ? "Installing…" : "Install nanochat"}
                  </button>
                </div>
              ) : modelsLoading && models.length === 0 ? (
                <div className="flex flex-col gap-3">
                  <Progress value={0} className="h-1" />
                  <p className="text-sm text-muted-foreground">
                    Looking for models on this machine…
                  </p>
                </div>
              ) : (
                <div className="flex flex-col gap-6">
                  <BenchmarkConfigureForm
                    form={form}
                    specs={specs}
                    models={models}
                    modelsEmptyReason={modelsEmptyReason}
                    problems={problemsByField}
                    onChange={(patch) => {
                      const store = useBenchmarkConfigStore.getState();
                      if (patch.modelPath !== undefined)
                        store.setModelPath(patch.modelPath);
                      if (patch.judgeModelPath !== undefined)
                        store.setJudgeModelPath(patch.judgeModelPath);
                      if (patch.maxProblems !== undefined)
                        store.setMaxProblems(patch.maxProblems);
                      if (patch.batchSize !== undefined)
                        store.setBatchSize(patch.batchSize);
                      if (patch.maxNewTokens !== undefined)
                        store.setMaxNewTokens(patch.maxNewTokens);
                      if (patch.maxSeqLength !== undefined)
                        store.setMaxSeqLength(patch.maxSeqLength);
                      if (patch.loadIn4bit !== undefined)
                        store.setLoadIn4bit(patch.loadIn4bit);
                    }}
                    onToggleBenchmark={(key) => toggleBenchmark(key, order)}
                  />
                  <div className="@4xl/bench-configure:sticky @4xl/bench-configure:bottom-4 flex justify-end">
                    <div className="w-full @4xl/bench-configure:w-72">
                      <BenchmarkStartCta
                        canStart={problems.length === 0}
                        isStarting={isStarting}
                        runActive={runActive}
                        onStart={() => void start()}
                        error={actionError}
                      />
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="current-run" className="mt-0">
              {status ? (
                <BenchmarkLiveRunView
                  status={status}
                  onStop={stop}
                  onConfigure={() => setTab("configure")}
                />
              ) : (
                <p className="text-sm text-muted-foreground">No run yet.</p>
              )}
            </TabsContent>

            <TabsContent value="leaderboard" className="mt-0">
              <BenchmarkLeaderboardView
                specs={specs}
                onRunAgain={() => setTab("configure")}
              />
            </TabsContent>
          </div>
        </div>
      </Tabs>
    </div>
  );
}
