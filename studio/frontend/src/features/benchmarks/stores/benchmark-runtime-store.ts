// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Live benchmark run state.
 *
 * Not persisted: a run that was in flight when the app closed did not resume, so
 * restoring its progress bar would show progress that is not happening. The
 * leaderboard is persisted separately, in the database, because a result is
 * durable in a way a progress bar is not.
 *
 * The stores stay dumb on purpose. The ordering rules — never letting a late
 * frame from a previous run land on the current one, never mixing two runs' log
 * lines — belong in the consumer, where the frame's run id is known, not in a
 * setter that cannot check it.
 */
import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";

import type {
  BenchmarkCatalogue,
  BenchmarkLogLine,
  BenchmarkModelCandidate,
  BenchmarkModelList,
  BenchmarkRunStatus,
  BenchmarkScore,
  BenchmarkScoringMode,
  BenchmarkStatus,
} from "../types";

// Bounded like the server's buffer, and smaller: the console only ever shows the
// tail, and re-rendering thousands of lines on every frame is the difference
// between a live terminal and a stuttering one.
const MAX_LOG_LINES = 2000;

export type BenchmarkRuntimeState = {
  // identity
  runId: string | null;

  // lifecycle
  status: BenchmarkRunStatus;
  /** Which pass is running: materialize (datasets), score (the model), grade. */
  phase: string;
  message: string;
  error: string | null;
  stopRequested: boolean;

  // what is being measured
  modelId: string;
  modelLabel: string;
  modelFormat: "safetensors" | "gguf";
  loraPath: string | null;
  loadIn4bit: boolean;
  requested: string[];
  current: string | null;
  scores: BenchmarkScore[];
  composite: number | null;
  scoringMode: BenchmarkScoringMode | null;
  // Which model gave the second opinion, if this run has one. Kept beside the
  // model being measured because a judged score is meaningless without knowing
  // what read it.
  judgeModelId: string | null;
  judgeModelLabel: string | null;

  // progress
  progressPercent: number;
  elapsedSeconds: number;
  startedAt: number | null;
  endedAt: number | null;

  // the in-app console
  logLines: BenchmarkLogLine[];
  logSeq: number;

  // catalogue and inventory, owned by the configure view
  catalogue: BenchmarkCatalogue | null;
  catalogueLoading: boolean;
  models: BenchmarkModelCandidate[];
  modelsLoading: boolean;
  modelsEmptyReason: string | null;
  /** Set when a start or fetch failed, so the page can show it once and clear it. */
  actionError: string | null;
  isStarting: boolean;
};

export type BenchmarkRuntimeActions = {
  hydrate: (status: BenchmarkStatus) => void;
  applyStatus: (status: BenchmarkStatus) => void;
  appendLogs: (lines: BenchmarkLogLine[]) => void;
  setCatalogue: (catalogue: BenchmarkCatalogue) => void;
  setCatalogueLoading: (loading: boolean) => void;
  setModels: (list: BenchmarkModelList) => void;
  setModelsLoading: (loading: boolean) => void;
  setStarting: (starting: boolean) => void;
  setActionError: (message: string | null) => void;
  markStopRequested: () => void;
  reset: () => void;
};

const EMPTY_STATE: BenchmarkRuntimeState = {
  runId: null,
  status: "idle",
  phase: "idle",
  message: "",
  error: null,
  stopRequested: false,
  modelId: "",
  modelLabel: "",
  modelFormat: "safetensors",
  loraPath: null,
  loadIn4bit: false,
  requested: [],
  current: null,
  scores: [],
  composite: null,
  scoringMode: null,
  judgeModelId: null,
  judgeModelLabel: null,
  progressPercent: 0,
  elapsedSeconds: 0,
  startedAt: null,
  endedAt: null,
  logLines: [],
  logSeq: 0,
  catalogue: null,
  catalogueLoading: false,
  models: [],
  modelsLoading: false,
  modelsEmptyReason: null,
  actionError: null,
  isStarting: false,
};

function toState(status: BenchmarkStatus): BenchmarkRuntimeState {
  return {
    ...EMPTY_STATE,
    runId: status.run_id,
    status: status.status,
    phase: status.phase,
    message: status.message,
    error: status.error,
    modelId: status.model_id,
    modelLabel: status.model_label,
    modelFormat: status.format,
    loraPath: status.lora_path,
    loadIn4bit: status.load_in_4bit,
    requested: status.requested,
    current: status.current,
    scores: status.scores,
    composite: status.composite,
    scoringMode: status.scoring_mode,
    judgeModelId: status.judge_model_id,
    judgeModelLabel: status.judge_model_label,
    progressPercent: status.progress_percent,
    elapsedSeconds: status.elapsed_seconds,
    startedAt: status.started_at,
    endedAt: status.ended_at,
  };
}

export const useBenchmarkRuntimeStore = create<
  BenchmarkRuntimeState & BenchmarkRuntimeActions
>((set) => ({
  ...EMPTY_STATE,

  // A full replace, because the server's status is the whole truth about a run.
  // Merging would let a field the server stopped sending keep a stale value.
  hydrate: (status) => set(toState(status)),
  applyStatus: (status) =>
    set((state) => {
      const next = toState(status);
      // The catalogue and inventory are owned by the configure view, and a status
      // frame has nothing to say about them. Carrying them across is what lets a
      // live run and the picker coexist without one clobbering the other.
      return {
        ...next,
        catalogue: state.catalogue,
        catalogueLoading: state.catalogueLoading,
        models: state.models,
        modelsLoading: state.modelsLoading,
        modelsEmptyReason: state.modelsEmptyReason,
        // stopRequested is a local intent. A frame saying "running" does not
        // un-request a stop the user already asked for.
        stopRequested: state.stopRequested,
      };
    }),

  appendLogs: (lines) =>
    set((state) => {
      if (lines.length === 0) return state;
      const merged = [...state.logLines, ...lines];
      return {
        logLines:
          merged.length > MAX_LOG_LINES
            ? merged.slice(merged.length - MAX_LOG_LINES)
            : merged,
        // The cursor is what the next fetch resumes from, and it is taken from
        // the last line rather than incremented: a gap in the sequence would
        // otherwise be papered over by a count.
        logSeq: lines[lines.length - 1].seq,
      };
    }),

  setCatalogue: (catalogue) => set({ catalogue, catalogueLoading: false }),
  setCatalogueLoading: (catalogueLoading) => set({ catalogueLoading }),
  setModels: (list) =>
    set({
      models: list.models,
      modelsEmptyReason: list.empty_reason,
      modelsLoading: false,
    }),
  setModelsLoading: (modelsLoading) => set({ modelsLoading }),
  setStarting: (isStarting) => set({ isStarting }),
  setActionError: (actionError) => set({ actionError }),
  markStopRequested: () => set({ stopRequested: true }),

  reset: () => set({ ...EMPTY_STATE }),
}));

/** A run that is going right now. Read through a selector so the sidebar spinner
 *  and the live view agree without either re-rendering on unrelated changes. */
export function isBenchmarkRunActive(state: BenchmarkRuntimeState): boolean {
  return state.status === "running";
}

/** A run that produced something worth looking at, finished or not. */
export function isBenchmarkRunFinished(state: BenchmarkRuntimeState): boolean {
  return (
    state.status === "completed" ||
    state.status === "stopped" ||
    state.scores.some((score) => score.status === "complete")
  );
}

/** Whether there is a run to look at, at any stage.
 *
 * Distinct from `isBenchmarkRunFinished`, which is about results. The sub-nav needs
 * this one: a run that has only just started has nothing to show *yet*, but the
 * tab the user was just moved to must not be disabled underneath them, which is
 * what keying the trigger off the finished check would do for the whole run.
 */
export function hasBenchmarkRun(state: BenchmarkRuntimeState): boolean {
  return state.runId !== null;
}

/** The benchmark keys this run actually produced a score for. */
export function selectScoredKeys(state: BenchmarkRuntimeState): string[] {
  return state.scores
    .filter((score) => score.status === "complete")
    .map((score) => score.key);
}

export function useBenchmarkRunActive(): boolean {
  return useBenchmarkRuntimeStore(isBenchmarkRunActive);
}

export function useBenchmarkRunFinished(): boolean {
  return useBenchmarkRuntimeStore(isBenchmarkRunFinished);
}

export function useBenchmarkScoredKeys(): string[] {
  return useBenchmarkRuntimeStore(useShallow(selectScoredKeys));
}
