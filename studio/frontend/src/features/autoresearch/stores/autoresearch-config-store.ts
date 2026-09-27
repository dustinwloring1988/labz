// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import {
  AUTORESEARCH_CONFIG_PERSISTENCE_VERSION,
  migrateAutoresearchConfig,
} from "./autoresearch-config-persistence.ts";
import {
  DEFAULT_AUTORESEARCH_CONFIG,
  type AutoresearchConfigState,
} from "./autoresearch-config-policy.ts";

export const AUTORESEARCH_CONFIG_PERSISTENCE_NAME = "labz_autoresearch_config_v1";

export type AutoresearchConfigStore = AutoresearchConfigState & {
  apply: (patch: Partial<AutoresearchConfigState>) => void;
  reset: () => void;
};

// createJSONStorage takes a getter that must return a StateStorage, not one that
// may return undefined: a store with no storage is a store that cannot persist,
// and silently accepting that here would make "my settings did not save" a
// mystery. The probe below is the guard instead, and the in-memory fallback
// means the store still works for the session on a browser that refuses writes.
//
// Typed against the persisted shape rather than the store: `partialize` below
// strips the actions, so that is the shape that actually goes to storage.
const storage = createJSONStorage<AutoresearchConfigState>(() => {
  try {
    if (typeof localStorage === "undefined") {
      throw new Error("no localStorage");
    }
    // Touch it: Safari's private mode has the object and throws on write, and
    // the throw has to happen here rather than deep inside a persist rehydrate.
    const probe = "__unsloth_autoresearch_probe__";
    localStorage.setItem(probe, "1");
    localStorage.removeItem(probe);
    return localStorage;
  } catch {
    // An in-memory object is still a valid StateStorage, and it means the store
    // works for this session instead of throwing on the first edit.
    const memory = new Map<string, string>();
    return {
      getItem: (name) => memory.get(name) ?? null,
      setItem: (name, value) => void memory.set(name, value),
      removeItem: (name) => void memory.delete(name),
    };
  }
});

export const useAutoresearchConfigStore = create<AutoresearchConfigStore>()(
  persist(
    (set) => ({
      ...DEFAULT_AUTORESEARCH_CONFIG,
      apply: (patch) => set(patch),
      reset: () => set({ ...DEFAULT_AUTORESEARCH_CONFIG }),
    }),
    {
      name: AUTORESEARCH_CONFIG_PERSISTENCE_NAME,
      storage,
      version: AUTORESEARCH_CONFIG_PERSISTENCE_VERSION,
      // The actions are not persisted: a store that came back from localStorage
      // with an `apply` of undefined is a store that throws on the first edit.
      partialize: (state) => {
        const { apply: _apply, reset: _reset, ...config } = state;
        void _apply;
        void _reset;
        return config;
      },
      merge: (persisted, current) => ({
        ...current,
        ...migrateAutoresearchConfig(
          persisted as Partial<AutoresearchConfigState> | undefined,
          current,
        ),
      }),
    },
  ),
);
