// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Live autoresearch run state.
 *
 * Not persisted: a loop that was in flight when the app closed did not resume.
 * The agent it spawned died with the session, and unlike a training run there is
 * no way to reattach to it. Restoring its progress bar would show progress that
 * is not happening.
 *
 * Everything here is written by one place — the SSE consumer and the hydration
 * poll — and read by the live view, the results tab and the sidebar spinner. The
 * stores stay dumb on purpose: the ordering rules (for example, never letting a
 * late event from a previous run land on the current one) belong in the consumer
 * where the event's run id is known, not in a setter that cannot check it.
 */
import { create } from "zustand";

import type {
  AutoresearchAgents,
  AutoresearchCheckpointInfo,
  AutoresearchEnvironmentStatus,
  AutoresearchExperimentInfo,
  AutoresearchMetricPoint,
  AutoresearchLogLine,
  AutoresearchResultRow,
  ProgressEvent,
} from "../types/index.ts";

export type { ProgressEvent } from "../types/index.ts";

/**
 * Bounded like the server's buffer, and a little smaller: the console only ever
 * shows the tail, and re-rendering thousands of lines on every frame is the
 * difference between a live terminal and a stuttering one.
 */
const MAX_LOG_LINES = 2000;
/** A run can be a hundred experiments, and the stepper shows all of them. */
const MAX_EXPERIMENTS = 200;

export type AutoresearchRuntimeState = {
  // identity
  runId: string | null;

  // lifecycle
  status: "idle" | "running" | "completed" | "error" | "stopped";
  phase: "idle" | "preparing" | "experiment" | "finished";
  message: string;
  error: string | null;
  stopRequested: boolean;
  hasHydrated: boolean;

  // progress
  experiment: number;
  totalExperiments: number;
  progressPercent: number;

  // live numbers, read out of run.log
  step: number;
  loss: number | null;
  tokPerSec: number | null;
  mfu: number | null;
  peakMemoryGb: number | null;
  elapsedSeconds: number;
  etaSeconds: number | null;

  // the search's own accounting
  bestValBpb: number | null;
  baselineValBpb: number | null;
  kept: number;
  discarded: number;
  crashed: number;

  experiments: AutoresearchExperimentInfo[];
  /** The ledger, which outlives any one run. */
  results: AutoresearchResultRow[];
  resultsLoaded: boolean;
  resultsTotals: {
    total: number;
    kept: number;
    discarded: number;
    crashed: number;
    bestValBpb: number | null;
    bestIndex: number | null;
    baselineValBpb: number | null;
  };
  /** One point per experiment, for the frontier chart. */
  series: AutoresearchMetricPoint[];

  warnings: string[];

  /** The report the loop wrote as its final phase, shown on the Report tab. */
  reportText: string;
  reportModel: string;
  reportSavedTo: string | null;
  reportError: string | null;
  reportRunning: boolean;

  // The in-app console. `logSeq` is the cursor the next fetch resumes from, so it
  // is the only thing that has to be correct for the console to be complete.
  logLines: AutoresearchLogLine[];
  logSeq: number;

  gitBranch: string;
  gitHead: string;

  // environment and validation, owned by the configure view
  environment: AutoresearchEnvironmentStatus | null;
  agents: AutoresearchAgents | null;
  checkpoints: AutoresearchCheckpointInfo[];
  residentModel: string | null;
};

export type AutoresearchRuntimeActions = {
  hydrate: (status: ProgressEvent) => void;
  applyProgress: (event: ProgressEvent) => void;
  applyResults: (payload: {
    rows: AutoresearchResultRow[];
    total: number;
    kept: number;
    discarded: number;
    crashed: number;
    best_val_bpb: number | null;
    best_index: number | null;
    baseline_val_bpb: number | null;
  }) => void;
  applySeries: (points: AutoresearchMetricPoint[]) => void;
  appendLogs: (lines: AutoresearchLogLine[]) => void;
  setEnvironment: (status: AutoresearchEnvironmentStatus) => void;
  setAgents: (agents: AutoresearchAgents) => void;
  setCheckpoints: (
    checkpoints: AutoresearchCheckpointInfo[],
    resident: string | null,
  ) => void;
  setStopRequested: (value: boolean) => void;
  markStarted: (runId: string, totalExperiments: number) => void;
  reset: () => void;
};

const EMPTY: AutoresearchRuntimeState = {
  runId: null,
  status: "idle",
  phase: "idle",
  message: "",
  error: null,
  stopRequested: false,
  hasHydrated: false,

  experiment: 0,
  totalExperiments: 0,
  progressPercent: 0,

  step: 0,
  loss: null,
  tokPerSec: null,
  mfu: null,
  peakMemoryGb: null,
  elapsedSeconds: 0,
  etaSeconds: null,

  bestValBpb: null,
  baselineValBpb: null,
  kept: 0,
  discarded: 0,
  crashed: 0,

  experiments: [],
  results: [],
  resultsLoaded: false,
  resultsTotals: {
    total: 0,
    kept: 0,
    discarded: 0,
    crashed: 0,
    bestValBpb: null,
    bestIndex: null,
    baselineValBpb: null,
  },
  series: [],

  warnings: [],

  reportText: "",
  reportModel: "",
  reportSavedTo: null,
  reportError: null,
  reportRunning: false,

  logLines: [],
  logSeq: 0,

  gitBranch: "",
  gitHead: "",

  environment: null,
  agents: null,
  checkpoints: [],
  residentModel: null,
};

export const useAutoresearchRuntimeStore = create<
  AutoresearchRuntimeState & AutoresearchRuntimeActions
>((set, get) => ({
  ...EMPTY,

  hydrate: (status) =>
    set((state) => ({
      runId: status.run_id || state.runId,
      status: status.status,
      phase: status.phase,
      message: status.message,
      error: status.error,
      experiment: status.experiment,
      totalExperiments: status.total_experiments || state.totalExperiments,
      progressPercent: status.progress_percent,
      step: status.step,
      loss: status.loss,
      tokPerSec: status.tok_per_sec,
      mfu: status.mfu,
      peakMemoryGb: status.peak_memory_gb,
      elapsedSeconds: status.elapsed_seconds,
      etaSeconds: status.eta_seconds,
      bestValBpb: status.best_val_bpb,
      baselineValBpb: status.baseline_val_bpb,
      kept: status.kept,
      discarded: status.discarded,
      crashed: status.crashed,
      reportText: status.report_text || state.reportText,
      reportModel: status.report_model || state.reportModel,
      reportSavedTo: status.report_saved_to ?? state.reportSavedTo,
      reportError: status.report_error,
      reportRunning: status.report_running,
      experiments: status.experiments.slice(0, MAX_EXPERIMENTS),
      warnings: status.warnings,
      gitBranch: status.git_branch || state.gitBranch,
      gitHead: status.git_head || state.gitHead,
      hasHydrated: true,
    })),

  applyProgress: (event) => {
    // A late event from a run that has been replaced must not overwrite the
    // current one. The id is the only thing that distinguishes them, which is
    // exactly why the stream carries it.
    const current = get().runId;
    if (current && event.run_id && current !== event.run_id) return;

    set((state) => {
      const next: Partial<AutoresearchRuntimeState> = {
        runId: event.run_id || state.runId,
        status: event.status,
        phase: event.phase,
        message: event.message,
        error: event.error,
        experiment: event.experiment,
        totalExperiments: event.total_experiments || state.totalExperiments,
        progressPercent: event.progress_percent,
        step: event.step,
        loss: event.loss,
        tokPerSec: event.tok_per_sec,
        mfu: event.mfu,
        peakMemoryGb: event.peak_memory_gb,
        elapsedSeconds: event.elapsed_seconds,
        etaSeconds: event.eta_seconds,
        bestValBpb: event.best_val_bpb,
        baselineValBpb: event.baseline_val_bpb,
        kept: event.kept,
        discarded: event.discarded,
        crashed: event.crashed,
        reportText: event.report_text || state.reportText,
        reportModel: event.report_model || state.reportModel,
        reportSavedTo: event.report_saved_to ?? state.reportSavedTo,
        reportError: event.report_error,
        reportRunning: event.report_running,
        stopRequested: event.stop_requested,
        gitBranch: event.git_branch || state.gitBranch,
        gitHead: event.git_head || state.gitHead,
      };
      if (event.warnings?.length) {
        next.warnings = [...new Set([...state.warnings, ...event.warnings])].slice(-20);
      }
      if (event.experiments?.length) {
        next.experiments = event.experiments.slice(0, MAX_EXPERIMENTS);
      }
      if (event.logs?.length) {
        const merged = [...state.logLines, ...event.logs];
        next.logLines =
          merged.length > MAX_LOG_LINES ? merged.slice(-MAX_LOG_LINES) : merged;
        next.logSeq = event.logs[event.logs.length - 1].seq;
      }
      return next;
    });
  },

  applyResults: (payload) =>
    set({
      results: payload.rows,
      resultsLoaded: true,
      resultsTotals: {
        total: payload.total,
        kept: payload.kept,
        discarded: payload.discarded,
        crashed: payload.crashed,
        bestValBpb: payload.best_val_bpb,
        bestIndex: payload.best_index,
        baselineValBpb: payload.baseline_val_bpb,
      },
    }),

  applySeries: (points) => set({ series: points }),

  appendLogs: (lines) => {
    if (lines.length === 0) return;
    set((state) => {
      // The stream and the poll backstop both deliver lines, so the same one can
      // arrive twice. `seq` is monotonic, so dropping anything at or below the
      // cursor is what makes the two channels safe to run together.
      const cursor = state.logSeq;
      const fresh = lines.filter((line) => line.seq > cursor);
      if (fresh.length === 0) return state;
      const merged = [...state.logLines, ...fresh];
      return {
        logLines:
          merged.length > MAX_LOG_LINES ? merged.slice(-MAX_LOG_LINES) : merged,
        logSeq: fresh[fresh.length - 1].seq,
      };
    });
  },

  setEnvironment: (environment) => set({ environment }),
  setAgents: (agents) => set({ agents }),
  setCheckpoints: (checkpoints, residentModel) =>
    set({ checkpoints, residentModel }),
  setStopRequested: (stopRequested) => set({ stopRequested }),

  markStarted: (runId, totalExperiments) =>
    set({
      runId,
      status: "running",
      phase: "preparing",
      message: "Preparing the experiment loop",
      error: null,
      // A new run starts from zero. Keeping the previous run's experiments would
      // show its frontier under the new run's name.
      experiment: 0,
      totalExperiments,
      progressPercent: 0,
      step: 0,
      loss: null,
      tokPerSec: null,
      mfu: null,
      peakMemoryGb: null,
      elapsedSeconds: 0,
      etaSeconds: null,
      bestValBpb: null,
      baselineValBpb: null,
      kept: 0,
      discarded: 0,
      crashed: 0,
      experiments: Array.from({ length: totalExperiments }, (_unused, index) => ({
        index: index + 1,
        status: "pending" as const,
        message: "",
        error: null,
        val_bpb: null,
        memory_gb: null,
        commit: "",
        description: "",
        summary: "",
        started_at: null,
        ended_at: null,
        duration_seconds: null,
        checkpoint: null,
      })),
      warnings: [],
      // A new run has not written its report yet, and keeping the previous run's
      // would put a finished search's conclusion above an empty ledger.
      reportText: "",
      reportModel: "",
      reportSavedTo: null,
      reportError: null,
      reportRunning: false,
      // A new run gets a new console. Keeping the previous run's output would
      // show its last lines above the new run's first ones.
      logLines: [],
      logSeq: 0,
      stopRequested: false,
    }),

  reset: () => set({ ...EMPTY, hasHydrated: true }),
}));

// --- selectors, so components subscribe to the narrowest slice ---

export const isAutoresearchRunActive = (state: AutoresearchRuntimeState) =>
  state.status === "running";

export const isAutoresearchRunFinished = (state: AutoresearchRuntimeState) =>
  state.status === "completed" || state.status === "error" || state.status === "stopped";

/** The experiment currently executing, for the stepper. */
export const activeAutoresearchIndex = (state: AutoresearchRuntimeState) =>
  state.experiment;

/** True once there is a search worth showing, rather than an empty form. */
export const hasAutoresearchHistory = (state: AutoresearchRuntimeState) =>
  state.results.length > 0 || state.experiments.some((e) => e.status !== "pending");
