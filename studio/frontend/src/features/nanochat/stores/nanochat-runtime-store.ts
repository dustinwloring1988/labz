// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Live nanochat run state.
 *
 * Not persisted: a run that was in flight when the app closed did not resume, so
 * restoring its progress bar would show progress that is not happening.
 *
 * Everything here is written by one place — the SSE consumer and the hydration
 * poll — and read by the live view, the run preview and the sidebar spinner. The
 * stores stay dumb on purpose: the ordering rules (for example, never letting a
 * late event from a previous run land on the current one) belong in the consumer
 * where the event's run id is known, not in a setter that cannot check it.
 */
import { create } from "zustand";

import type {
  NanochatBenchmark,
  NanochatCheckpointInfo,
  NanochatEnvironmentStatus,
  NanochatFit,
  NanochatLogLine,
  NanochatSample,
  NanochatStageInfo,
  NanochatStatus,
} from "../types/index.ts";

// Bounded like the server's buffer, and a little smaller: the console only ever
// shows the tail, and re-rendering thousands of lines on every frame is the
// difference between a live terminal and a stuttering one.
const MAX_LOG_LINES = 2000;

export type NanochatRuntimeState = {
  // identity
  runId: string | null;

  // lifecycle
  status: "idle" | "running" | "completed" | "error" | "stopped";
  phase: string;
  currentStage: string | null;
  message: string;
  error: string | null;
  stopRequested: boolean;
  hasHydrated: boolean;

  // progress
  step: number;
  totalSteps: number;
  progressPercent: number;

  // live metrics
  loss: number | null;
  valBpb: number | null;
  chatcore: number | null;
  reward: number | null;
  tokPerSec: number | null;
  mfu: number | null;
  peakMemoryBytes: number | null;
  etaSeconds: number | null;
  elapsedSeconds: number;

  // series, keyed by what they measure. Loss and reward are never mixed: they
  // share an event shape but not a scale, and one shared axis would read as a
  // divergence rather than two different measurements.
  series: Record<string, { step: number; value: number | null; totalSteps: number | null }[]>;

  // per-stage and per-artefact
  stages: NanochatStageInfo[];
  samples: NanochatSample[];
  benchmarks: NanochatBenchmark[];
  checkpoints: NanochatCheckpointInfo[];
  warnings: string[];

  // The in-app console. `logSeq` is the cursor the next fetch resumes from, so it
  // is the only thing that has to be correct for the console to be complete.
  logLines: NanochatLogLine[];
  logSeq: number;

  // what nanochat told us about the run it is actually doing
  numParams: number | null;
  resolvedConfig: Record<string, unknown> | null;

  // environment and validation, owned by the configure view
  environment: NanochatEnvironmentStatus | null;
  fit: NanochatFit | null;
  validationErrors: string[];
  validationWarnings: string[];
  isValidating: boolean;
};

export type NanochatRuntimeActions = {
  hydrate: (status: NanochatStatus) => void;
  applyProgress: (event: ProgressEvent) => void;
  applyMetrics: (series: Record<string, NanochatRuntimeState["series"][string]>) => void;
  applySamples: (samples: NanochatSample[]) => void;
  applyBenchmarks: (benchmarks: NanochatBenchmark[]) => void;
  appendLogs: (lines: NanochatLogLine[]) => void;
  setEnvironment: (status: NanochatEnvironmentStatus) => void;
  setFit: (fit: NanochatFit | null) => void;
  setValidation: (errors: string[], warnings: string[]) => void;
  setValidating: (value: boolean) => void;
  setStopRequested: (value: boolean) => void;
  markStarted: (runId: string) => void;
  reset: () => void;
};

export type ProgressEvent = {
  run_id: string;
  status: NanochatRuntimeState["status"];
  phase: string;
  current_stage: string | null;
  message: string;
  step: number;
  total_steps: number;
  progress_percent: number;
  loss: number | null;
  val_bpb: number | null;
  chatcore: number | null;
  reward: number | null;
  tok_per_sec: number | null;
  mfu: number | null;
  eta_seconds: number | null;
  elapsed_seconds: number;
  peak_memory_bytes: number | null;
  num_params: number | null;
  warnings: string[];
  stages: { key: string; status: string; error: string | null }[];
  /** Console lines carried on this frame, if any. */
  logs?: NanochatLogLine[];
};

const EMPTY: NanochatRuntimeState = {
  runId: null,
  status: "idle",
  phase: "idle",
  currentStage: null,
  message: "",
  error: null,
  stopRequested: false,
  hasHydrated: false,

  step: 0,
  totalSteps: 0,
  progressPercent: 0,

  loss: null,
  valBpb: null,
  chatcore: null,
  reward: null,
  tokPerSec: null,
  mfu: null,
  peakMemoryBytes: null,
  etaSeconds: null,
  elapsedSeconds: 0,

  series: {},

  stages: [],
  samples: [],
  benchmarks: [],
  checkpoints: [],
  warnings: [],

  logLines: [],
  logSeq: 0,

  numParams: null,
  resolvedConfig: null,

  environment: null,
  fit: null,
  validationErrors: [],
  validationWarnings: [],
  isValidating: false,
};

export const useNanochatRuntimeStore = create<NanochatRuntimeState & NanochatRuntimeActions>(
  (set, get) => ({
    ...EMPTY,

    hydrate: (status) =>
      set((state) => ({
        runId: status.run_id || state.runId,
        status: status.status,
        phase: status.phase,
        currentStage: status.current_stage,
        message: status.message,
        error: status.error,
        step: status.step,
        totalSteps: status.total_steps,
        progressPercent: status.progress_percent,
        loss: status.loss,
        valBpb: status.val_bpb,
        chatcore: status.chatcore,
        reward: status.reward,
        tokPerSec: status.tok_per_sec,
        mfu: status.mfu,
        peakMemoryBytes: status.peak_memory_bytes ?? state.peakMemoryBytes,
        etaSeconds: status.eta_seconds,
        elapsedSeconds: status.elapsed_seconds,
        numParams: status.num_params ?? state.numParams,
        resolvedConfig: status.resolved_config ?? state.resolvedConfig,
        stages: status.stages,
        // The wire shape already matches NanochatCheckpointInfo (snake_case on
        // both), so this is a plain hand-off rather than a remap.
        checkpoints: status.checkpoints,
        warnings: status.warnings,
        hasHydrated: true,
      })),

    applyProgress: (event) => {
      // A late event from a run that has been replaced must not overwrite the
      // current one. The id is the only thing that distinguishes them, which is
      // exactly why the stream carries it.
      const current = get().runId;
      if (current && event.run_id && current !== event.run_id) return;

      set((state) => {
        const next: Partial<NanochatRuntimeState> = {
          runId: event.run_id || state.runId,
          status: event.status,
          phase: event.phase,
          currentStage: event.current_stage,
          message: event.message,
          step: event.step,
          totalSteps: event.total_steps,
          progressPercent: event.progress_percent,
          loss: event.loss,
          valBpb: event.val_bpb,
          chatcore: event.chatcore,
          reward: event.reward,
          tokPerSec: event.tok_per_sec,
          mfu: event.mfu,
          etaSeconds: event.eta_seconds,
          elapsedSeconds: event.elapsed_seconds,
          peakMemoryBytes: event.peak_memory_bytes ?? state.peakMemoryBytes,
          numParams: event.num_params ?? state.numParams,
        };
        if (event.warnings?.length) {
          next.warnings = [...new Set([...state.warnings, ...event.warnings])].slice(-20);
        }
        if (event.stages?.length) {
          next.stages = event.stages as NanochatStageInfo[];
        }
        if (event.logs?.length) {
          const merged = [...state.logLines, ...event.logs];
          next.logLines = merged.length > MAX_LOG_LINES ? merged.slice(-MAX_LOG_LINES) : merged;
          next.logSeq = event.logs[event.logs.length - 1].seq;
        }
        return next;
      });
    },

    applyMetrics: (series) =>
      set((state) => {
        // Only move a series forward. A resumed run replays earlier steps, and
        // letting them back in would make the chart run backwards.
        const next: NanochatRuntimeState["series"] = {};
        for (const [key, points] of Object.entries(series)) {
          const existing = state.series[key] ?? [];
          const merged = [...existing];
          for (const point of points) {
            const last = merged[merged.length - 1];
            if (!last || point.step > last.step) merged.push(point);
          }
          next[key] = merged;
        }
        // Keep series that only this poll did not mention, so a partial response
        // does not blank a chart.
        return { series: { ...state.series, ...next } };
      }),

    applySamples: (samples) => set({ samples }),
    applyBenchmarks: (benchmarks) => set({ benchmarks }),

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
          logLines: merged.length > MAX_LOG_LINES ? merged.slice(-MAX_LOG_LINES) : merged,
          logSeq: fresh[fresh.length - 1].seq,
        };
      });
    },
    setEnvironment: (environment) => set({ environment }),
    setFit: (fit) => set({ fit }),
    setValidation: (validationErrors, validationWarnings) =>
      set({ validationErrors, validationWarnings }),
    setValidating: (isValidating) => set({ isValidating }),
    setStopRequested: (stopRequested) => set({ stopRequested }),

    markStarted: (runId) =>
      set({
        runId,
        status: "running",
        phase: "starting",
        message: "Starting nanochat",
        error: null,
        // A new run starts from zero. Keeping the previous run's samples and
        // series would show the old run's curve under the new run's name.
        step: 0,
        totalSteps: 0,
        progressPercent: 0,
        loss: null,
        valBpb: null,
        chatcore: null,
        reward: null,
        series: {},
        samples: [],
        benchmarks: [],
        checkpoints: [],
        warnings: [],
        // A new run gets a new console. Keeping the previous run's output would
        // show its last lines above the new run's first ones.
        logLines: [],
        logSeq: 0,
        stopRequested: false,
      }),

    reset: () => set({ ...EMPTY, hasHydrated: true }),
  }),
);

// --- selectors, so components subscribe to the narrowest slice ---

export const isNanochatRunActive = (state: NanochatRuntimeState) => state.status === "running";

export const isNanochatRunFinished = (state: NanochatRuntimeState) =>
  state.status === "completed" || state.status === "error" || state.status === "stopped";

/** The stage currently executing, for the stepper. */
export const activeStageKey = (state: NanochatRuntimeState) => state.currentStage;

export const selectSamplesFor = (stage: string) => (state: NanochatRuntimeState) =>
  state.samples.filter((sample) => sample.stage === stage);
