// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  hasAutoresearchHistory,
  isAutoresearchRunFinished,
  useAutoresearchRuntimeStore,
} from "../src/features/autoresearch/stores/autoresearch-runtime-store.ts";
import type { AutoresearchLogLine, ProgressEvent } from "../src/features/autoresearch/types/index.ts";

// The store holds no timer and touches no network, so each test drives it
// directly and resets it afterwards. The two rules pinned here are the ones that
// keep a long unattended run honest on screen.

function reset(): void {
  useAutoresearchRuntimeStore.getState().reset();
}

function event(patch: Partial<ProgressEvent> = {}): ProgressEvent {
  return {
    run_id: "run_1",
    status: "running",
    phase: "experiment",
    message: "",
    error: null,
    started_at: null,
    ended_at: null,
    duration_seconds: 0,
    experiment: 0,
    total_experiments: 0,
    progress_percent: 0,
    step: 0,
    loss: null,
    tok_per_sec: null,
    mfu: null,
    peak_memory_gb: null,
    elapsed_seconds: 0,
    eta_seconds: null,
    best_val_bpb: null,
    baseline_val_bpb: null,
    kept: 0,
    discarded: 0,
    crashed: 0,
    experiments: [],
    warnings: [],
    resolved_config: {},
    git_branch: "",
    git_head: "",
    stop_requested: false,
    estimated_seconds: 0,
    report_text: "",
    report_model: "",
    report_saved_to: null,
    report_error: null,
    report_running: false,
    ...patch,
  };
}

function logLine(seq: number, line: string): AutoresearchLogLine {
  return { seq, stream: "stdout", line, ts: 0 };
}

test.afterEach(reset);

// --- the run-id guard ---

test("a frame from a superseded run is dropped", () => {
  // A poll that arrives after the user started a new loop must not report the
  // new loop's state to the old tab. The run id is the only thing that tells
  // them apart, which is why the stream carries it.
  useAutoresearchRuntimeStore.getState().markStarted("run_1", 4);
  useAutoresearchRuntimeStore.getState().applyProgress(
    event({ run_id: "run_1", experiment: 2, message: "the real one" }),
  );

  useAutoresearchRuntimeStore.getState().applyProgress(
    event({ run_id: "run_0", experiment: 9, message: "a stale frame" }),
  );

  const state = useAutoresearchRuntimeStore.getState();
  assert.equal(state.runId, "run_1");
  assert.equal(state.experiment, 2);
  assert.equal(state.message, "the real one");
});

test("a frame with no run id is still applied", () => {
  // The backend always sends one, but a heartbeat-shaped frame or a future
  // endpoint might not, and dropping a legitimate update over a missing
  // identifier is worse than applying it.
  useAutoresearchRuntimeStore.getState().markStarted("run_1", 4);
  useAutoresearchRuntimeStore.getState().applyProgress(
    event({ run_id: "", experiment: 3 }),
  );
  assert.equal(useAutoresearchRuntimeStore.getState().experiment, 3);
});

// --- the log cursor ---

test("the same log line arriving twice is only drawn once", () => {
  // The SSE stream and the HTTP poll backstop both deliver lines. Running both
  // is the point: a backgrounded tab or a dropped proxy leaves the stream
  // silently dead while the run carries on. The monotonic seq is what makes that
  // safe, and a console that re-prints its tail on every reconnect is unusable.
  const store = useAutoresearchRuntimeStore.getState();
  store.appendLogs([logLine(1, "first"), logLine(2, "second")]);
  store.appendLogs([logLine(1, "first"), logLine(2, "second")]);

  const state = useAutoresearchRuntimeStore.getState();
  assert.equal(state.logLines.length, 2);
  assert.equal(state.logSeq, 2);
});

test("a log line at or below the cursor is dropped", () => {
  const store = useAutoresearchRuntimeStore.getState();
  store.appendLogs([logLine(1, "a"), logLine(2, "b"), logLine(3, "c")]);
  store.appendLogs([logLine(2, "b"), logLine(4, "d")]);

  const state = useAutoresearchRuntimeStore.getState();
  assert.deepEqual(
    state.logLines.map((line) => line.line),
    ["a", "b", "c", "d"],
  );
  assert.equal(state.logSeq, 4);
});

test("a frame's logs advance the cursor", () => {
  useAutoresearchRuntimeStore.getState().markStarted("run_1", 4);
  useAutoresearchRuntimeStore.getState().applyProgress(
    event({ run_id: "run_1", logs: [logLine(1, "x"), logLine(2, "y")] }),
  );
  const state = useAutoresearchRuntimeStore.getState();
  assert.equal(state.logSeq, 2);
  assert.equal(state.logLines.length, 2);
});

test("an empty append does not touch the cursor", () => {
  const store = useAutoresearchRuntimeStore.getState();
  store.appendLogs([logLine(7, "seven")]);
  store.appendLogs([]);
  assert.equal(useAutoresearchRuntimeStore.getState().logSeq, 7);
});

// --- starting a run ---

test("starting a run clears the previous run's experiments and console", () => {
  // Otherwise the old run's frontier is drawn under the new run's name, and its
  // last log lines appear above the new run's first ones.
  const store = useAutoresearchRuntimeStore.getState();
  store.markStarted("run_1", 3);
  store.appendLogs([logLine(1, "old output")]);
  store.applyProgress(
    event({
      run_id: "run_1",
      experiments: [
        { index: 1, status: "kept", message: "", error: null, val_bpb: 1.2, memory_gb: 3, commit: "a", description: "d", summary: "", started_at: null, ended_at: null, duration_seconds: null, checkpoint: null },
        { index: 2, status: "kept", message: "", error: null, val_bpb: 1.1, memory_gb: 3, commit: "b", description: "d", summary: "", started_at: null, ended_at: null, duration_seconds: null, checkpoint: null },
        { index: 3, status: "pending", message: "", error: null, val_bpb: null, memory_gb: null, commit: "", description: "", summary: "", started_at: null, ended_at: null, duration_seconds: null, checkpoint: null },
      ],
    }),
  );

  store.markStarted("run_2", 5);

  const state = useAutoresearchRuntimeStore.getState();
  assert.equal(state.runId, "run_2");
  assert.equal(state.totalExperiments, 5);
  assert.equal(state.experiments.length, 5);
  assert.ok(state.experiments.every((entry) => entry.status === "pending"));
  assert.deepEqual(state.logLines, []);
  assert.equal(state.logSeq, 0);
  assert.equal(state.bestValBpb, null);
  assert.equal(state.kept, 0);
});

// --- warnings ---

test("warnings accumulate without duplicating", () => {
  // The same warning can arrive on many frames, and a list of twenty copies of
  // one sentence is a list nobody reads.
  useAutoresearchRuntimeStore.getState().markStarted("run_1", 2);
  useAutoresearchRuntimeStore.getState().applyProgress(
    event({ run_id: "run_1", warnings: ["the repository is dirty"] }),
  );
  useAutoresearchRuntimeStore.getState().applyProgress(
    event({ run_id: "run_1", warnings: ["the repository is dirty", "another thing"] }),
  );
  assert.deepEqual(useAutoresearchRuntimeStore.getState().warnings, [
    "the repository is dirty",
    "another thing",
  ]);
});

// --- selectors ---

test("a run counts as finished only once it has stopped doing work", () => {
  useAutoresearchRuntimeStore.getState().markStarted("run_1", 2);
  assert.equal(isAutoresearchRunFinished(useAutoresearchRuntimeStore.getState()), false);

  useAutoresearchRuntimeStore.getState().applyProgress(
    event({ run_id: "run_1", status: "completed", phase: "finished" }),
  );
  assert.equal(isAutoresearchRunFinished(useAutoresearchRuntimeStore.getState()), true);
});

test("an untouched store is not history", () => {
  // This is what decides whether the tab opens on a form or on a run, so an
  // empty ledger and an empty run have to read the same.
  assert.equal(hasAutoresearchHistory(useAutoresearchRuntimeStore.getState()), false);
});

test("recorded results are history even with no run in flight", () => {
  // The ledger outlives any one loop, and the tab is what someone opens the next
  // morning. A search is not defined by the run that happens to be loaded.
  useAutoresearchRuntimeStore.getState().applyResults({
    rows: [
      { index: 1, commit: "a", val_bpb: 1.2, memory_gb: 3, status: "keep", description: "d", error: null, is_best: true, checkpoint: null },
    ],
    total: 1,
    kept: 1,
    discarded: 0,
    crashed: 0,
    best_val_bpb: 1.2,
    best_index: 1,
    baseline_val_bpb: 1.2,
  });
  assert.equal(hasAutoresearchHistory(useAutoresearchRuntimeStore.getState()), true);
});
