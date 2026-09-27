// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The two ways the live view can end up showing nothing about a run that is
// actually going. Both are store-level, because both are about what the store
// knows rather than about what any component renders.

import assert from "node:assert/strict";
import test from "node:test";

import {
  hasBenchmarkRun,
  isBenchmarkRunActive,
  isBenchmarkRunFinished,
  useBenchmarkRuntimeStore,
  type BenchmarkRuntimeState,
} from "../src/features/benchmarks/stores/benchmark-runtime-store.ts";
import type { BenchmarkStatus } from "../src/features/benchmarks/types/index.ts";
import { makeScore, makeStatus } from "./helpers/benchmark-fixtures.ts";

// A local alias so the existing call sites read the same, with the defaults
// coming from the shared factory.
function status(overrides: Partial<BenchmarkStatus> = {}): BenchmarkStatus {
  return makeStatus({ progress_percent: 30, elapsed_seconds: 12, started_at: 1000, ...overrides });
}

function freshStore(): BenchmarkRuntimeState {
  return useBenchmarkRuntimeStore.getState();
}

// --- the sub-nav trigger must not be disabled for a run in progress ---

test("a run in its first second still counts as a run to look at", () => {
  // The trigger the user was just moved to. Keyed off results it would be greyed
  // out for the whole run, which reads as the app having refused to show it.
  useBenchmarkRuntimeStore.getState().reset();
  useBenchmarkRuntimeStore.getState().hydrate(status({ scores: [] }));
  const state = freshStore();
  assert.equal(hasBenchmarkRun(state), true);
  assert.equal(isBenchmarkRunActive(state), true);
  // Nothing has finished, which is a different question and also true.
  assert.equal(isBenchmarkRunFinished(state), false);
});

test("nothing counts as a run before one has started", () => {
  useBenchmarkRuntimeStore.getState().reset();
  assert.equal(hasBenchmarkRun(freshStore()), false);
  assert.equal(isBenchmarkRunActive(freshStore()), false);
  assert.equal(isBenchmarkRunFinished(freshStore()), false);
});

test("a finished run counts as both a run and a result", () => {
  useBenchmarkRuntimeStore.getState().reset();
  useBenchmarkRuntimeStore
    .getState()
    .hydrate(status({ status: "completed", phase: "done" }));
  const state = freshStore();
  assert.equal(hasBenchmarkRun(state), true);
  assert.equal(isBenchmarkRunFinished(state), true);
  assert.equal(isBenchmarkRunActive(state), false);
});

test("a run that errored is a run but not a result worth keeping open", () => {
  useBenchmarkRuntimeStore.getState().reset();
  useBenchmarkRuntimeStore
    .getState()
    .hydrate(status({ status: "error", phase: "idle", error: "no items" }));
  const state = freshStore();
  assert.equal(hasBenchmarkRun(state), true);
  // The tab is reachable so the error is visible, but the initial tab is not
  // Current Run, because there is nothing there to read.
  assert.equal(isBenchmarkRunFinished(state), false);
});

test("a run that produced a score is a result even while still running", () => {
  useBenchmarkRuntimeStore.getState().reset();
  useBenchmarkRuntimeStore.getState().hydrate(
    status({
      scores: [
        makeScore({
          correct: 200,
          total: 400,
          centered: 0.3333,
          elapsed_seconds: 30,
        }),
      ],
    }),
  );
  assert.equal(isBenchmarkRunFinished(freshStore()), true);
  assert.equal(isBenchmarkRunActive(freshStore()), true);
});

// --- a started run must reach the store ---

test("hydrating a started run gives the live view something to render", () => {
  // The regression: starting a run switched the tab without ever putting the run
  // in the store, so the view read "No run yet" for the whole run. The store is
  // written by mount-time hydration and by the stream, and the stream only opens
  // once the store already has a running run -- so the start path has to do it.
  useBenchmarkRuntimeStore.getState().reset();
  assert.equal(freshStore().runId, null);

  useBenchmarkRuntimeStore.getState().hydrate(status());
  const state = freshStore();
  assert.equal(state.runId, "r1");
  assert.equal(state.modelLabel, "my model");
  assert.deepEqual(state.requested, ["mmlu"]);
  assert.equal(state.current, "mmlu");
  assert.equal(state.progressPercent, 30);
});

test("a status frame keeps the catalogue and the model inventory", () => {
  // The configure view owns those, and a run's frames say nothing about them.
  // Clobbering them would empty the model picker mid-run.
  const store = useBenchmarkRuntimeStore.getState();
  store.reset();
  store.setModels({
    models: [
      {
        id: "m",
        label: "my model",
        path: "C:/m",
        source: "models_dir",
        format: "safetensors",
        lora: false,
        base_model: null,
      },
    ],
    empty_reason: null,
  });
  store.setCatalogue({
    benchmarks: [],
    default_benchmarks: [],
    available: true,
    reason: null,
    posix_sandbox: true,
  });

  useBenchmarkRuntimeStore.getState().applyStatus(status());
  const state = freshStore();
  assert.equal(state.models.length, 1);
  assert.notEqual(state.catalogue, null);
  assert.equal(state.runId, "r1");
});

test("a stop requested locally is not undone by a running frame", () => {
  // The user asked to stop. A frame that says "running" is the server's view of
  // the run, not a retraction of the request.
  useBenchmarkRuntimeStore.getState().reset();
  useBenchmarkRuntimeStore.getState().markStopRequested();
  useBenchmarkRuntimeStore.getState().applyStatus(status());
  assert.equal(freshStore().stopRequested, true);
});

test("console lines accumulate and the cursor follows the last one", () => {
  useBenchmarkRuntimeStore.getState().reset();
  const line = (seq: number) => ({
    seq,
    line: `line ${seq}`,
    stream: "stdout" as const,
    phase: "score",
    ts: seq,
  });
  useBenchmarkRuntimeStore.getState().appendLogs([line(1), line(2)]);
  useBenchmarkRuntimeStore.getState().appendLogs([line(7)]);
  const state = freshStore();
  assert.deepEqual(
    state.logLines.map((entry) => entry.seq),
    [1, 2, 7],
  );
  // Taken from the line rather than counted, so a gap in the sequence is not
  // papered over by an increment.
  assert.equal(state.logSeq, 7);
});

test("appending nothing changes nothing", () => {
  useBenchmarkRuntimeStore.getState().reset();
  useBenchmarkRuntimeStore
    .getState()
    .appendLogs([{ seq: 1, line: "a", stream: "stdout", phase: null, ts: 0 }]);
  useBenchmarkRuntimeStore.getState().appendLogs([]);
  assert.equal(freshStore().logLines.length, 1);
  assert.equal(freshStore().logSeq, 1);
});
