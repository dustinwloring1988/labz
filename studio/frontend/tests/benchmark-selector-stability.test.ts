// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A zustand selector that builds a new object is a selector that never stops
// re-rendering. The failure only shows up as a React error in a running app, so
// it is pinned here against the real store rather than left to be rediscovered.

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { useBenchmarkRuntimeStore } from "../src/features/benchmarks/stores/benchmark-runtime-store.ts";
import { makeStatus } from "./helpers/benchmark-fixtures.ts";

const FEATURE = join(
  import.meta.dirname,
  "..",
  "src",
  "features",
  "benchmarks",
);

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...sourceFiles(path));
    else if (/\.tsx?$/.test(entry.name)) out.push(path);
  }
  return out;
}

// --- the store's own selectors are stable ---

test("every exported selector returns a stable reference for unchanged state", () => {
  const state = useBenchmarkRuntimeStore.getState();
  const first = useBenchmarkRuntimeStore.getState();
  assert.equal(state, first, "getState must return the same object, not a copy");
  // The three selectors the page subscribes with. Each is a boolean here, and a
  // boolean can never be a fresh reference, which is the property that matters.
  for (const selector of [
    (s: typeof state) => s.runId !== null,
    (s: typeof state) => s.status === "running",
    (s: typeof state) => s.scores.some((score) => score.status === "complete"),
  ]) {
    assert.equal(selector(state), selector(first));
  }
});

test("run-scoped fields keep their identity across an unrelated update", () => {
  // useSyncExternalStore compares snapshots by reference. A field that is
  // replaced on every store write would re-render the live view on every log
  // line, so the arrays the view reads have to be carried across, not rebuilt.
  useBenchmarkRuntimeStore.getState().reset();
  useBenchmarkRuntimeStore.getState().hydrate(
    makeStatus({ progress_percent: 10, elapsed_seconds: 1, started_at: 1 }),
  );
  const before = useBenchmarkRuntimeStore.getState();
  const scores = before.scores;
  const requested = before.requested;

  // An appendLogs is the highest-frequency write in the run.
  useBenchmarkRuntimeStore
    .getState()
    .appendLogs([{ seq: 1, line: "a", stream: "stdout", phase: null, ts: 0 }]);

  const after = useBenchmarkRuntimeStore.getState();
  assert.equal(after.scores, scores, "scores must keep their reference");
  assert.equal(after.requested, requested, "requested must keep its reference");
  assert.notEqual(after.logLines, before.logLines, "logs must change to be useful");
});

// --- the page's selectors, by source ---

test("no benchmark selector builds a fresh object or array inline", () => {
  // The regression: a selector shaped
  //   useStore((s) => s.runId ? { ...many fields } : null)
  // returns a new object on every call, so the snapshot changes on every render
  // and React re-renders forever. It type-checks, lints clean, and only shows up
  // as "getSnapshot should be cached" and then "maximum update depth exceeded"
  // in a running app.
  const offenders: string[] = [];
  for (const path of sourceFiles(FEATURE)) {
    const text = readFileSync(path, "utf-8");
    // A selector whose body contains a returned object or array literal, i.e. a
    // `{` between the arrow and the closing of the callback.
    const selectors = text.matchAll(
      /useBenchmark\w*Store\(\s*\(state\w*\)\s*=>\s*(?:state\.[\w.]+\s*\?\s*)?[{[]/g,
    );
    for (const match of selectors) {
      offenders.push(`${path}: ${match[0].replace(/\s+/g, " ")}`);
    }
  }
  assert.deepEqual(offenders, []);
});

test("the page subscribes field by field and assembles with useMemo", () => {
  // The shape that is correct: primitives or store-owned references selected
  // individually, combined outside the selector.
  const text = readFileSync(join(FEATURE, "benchmarks-page.tsx"), "utf-8");
  assert.match(text, /useBenchmarkRuntimeStore\(\(state\) => state\.runId\)/);
  assert.match(text, /useMemo<BenchmarkStatus \| null>/);
});
