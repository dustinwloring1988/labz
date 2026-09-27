// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  createJSONStorage,
  persist,
  type StateStorage,
} from "zustand/middleware";

import {
  type BenchmarkFormState,
  DEFAULT_BENCHMARK_FORM,
  clampMaxProblems,
  toggleBenchmarkKey,
} from "./benchmark-config-policy";

// Best-effort persistence: localStorage can be blocked (private browsing) and
// zustand's persist write path is unguarded, so a throw here would break a store
// action. Swallow storage errors and keep the in-memory state.
const guardedLocalStorage: StateStorage = {
  getItem: (name) => {
    try {
      return window.localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name, value) => {
    try {
      window.localStorage.setItem(name, value);
    } catch {
      // ignore: the choice stays in memory for this session
    }
  },
  removeItem: (name) => {
    try {
      window.localStorage.removeItem(name);
    } catch {
      // ignore
    }
  },
};

export type BenchmarkConfigStore = {
  form: BenchmarkFormState;
  setModelPath: (path: string | null) => void;
  setJudgeModelPath: (path: string | null) => void;
  setMaxProblems: (value: number) => void;
  setBatchSize: (value: number) => void;
  setMaxNewTokens: (value: number) => void;
  setMaxSeqLength: (value: number) => void;
  setLoadIn4bit: (value: boolean) => void;
  toggleBenchmark: (key: string, catalogueOrder: readonly string[]) => void;
  setBenchmarks: (keys: string[], catalogueOrder: readonly string[]) => void;
  reset: () => void;
};

/**
 * The benchmark form, persisted so a reload does not lose a suite someone spent a
 * minute assembling.
 *
 * Only the form is persisted, never the run: a run belongs to the backend, and
 * rehydrating a stale one from local storage would show progress for a process
 * this app no longer has.
 */
export const useBenchmarkConfigStore = create<BenchmarkConfigStore>()(
  persist(
    (set) => ({
      form: DEFAULT_BENCHMARK_FORM,
  setModelPath: (path) =>
    set((state) => ({ form: { ...state.form, modelPath: path } })),
  setJudgeModelPath: (path) =>
    set((state) => ({ form: { ...state.form, judgeModelPath: path } })),
      setMaxProblems: (value) =>
        set((state) => ({
          form: { ...state.form, maxProblems: clampMaxProblems(value) },
        })),
      setBatchSize: (value) =>
        set((state) => ({
          form: {
            ...state.form,
            // Clamped rather than rejected: a slider that overshoots should land
            // on the nearest legal value, not refuse to move.
            batchSize: Math.min(256, Math.max(1, Math.round(value) || 1)),
          },
        })),
      setMaxNewTokens: (value) =>
        set((state) => ({
          form: {
            ...state.form,
            maxNewTokens: Math.min(
              8192,
              Math.max(16, Math.round(value) || 512),
            ),
          },
        })),
      setMaxSeqLength: (value) =>
        set((state) => ({
          form: {
            ...state.form,
            maxSeqLength: Math.min(
              131072,
              Math.max(256, Math.round(value) || 2048),
            ),
          },
        })),
      setLoadIn4bit: (value) =>
        set((state) => ({ form: { ...state.form, loadIn4bit: value } })),
      toggleBenchmark: (key, catalogueOrder) =>
        set((state) => ({
          form: {
            ...state.form,
            benchmarkKeys: toggleBenchmarkKey(
              state.form.benchmarkKeys,
              key,
              catalogueOrder,
            ),
          },
        })),
      setBenchmarks: (keys, catalogueOrder) =>
        set((state) => ({
          form: {
            ...state.form,
            benchmarkKeys: catalogueOrder.filter((key) => keys.includes(key)),
          },
        })),
      reset: () => set({ form: DEFAULT_BENCHMARK_FORM }),
    }),
    {
      name: "labz.benchmark-config",
      storage: createJSONStorage(() => guardedLocalStorage),
      // Only the durable half of the form. Everything else here is either derived
      // from the catalogue or re-fetched on mount.
      partialize: (state) => ({
        form: {
          ...state.form,
          // A stored selection is only a hint: the catalogue is the authority, and
          // a benchmark that has since disappeared upstream must not be resent.
          benchmarkKeys: state.form.benchmarkKeys,
        },
      }),
    },
  ),
);
