// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The editable nanochat run configuration.
 *
 * Split from the runtime store on purpose: this is what the user is editing, and
 * it is persisted. The live run state is not persisted, because a run that was
 * going when the app closed did not resume, and restoring a progress bar for
 * something that is no longer running would be a lie.
 *
 * Defaults come from the backend (`/api/nanochat/presets`) rather than being
 * restated here, so a fresh configuration matches nanochat's published pipeline
 * instead of a copy of it that drifts.
 */
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import { DEFAULT_NANOCHAT_CONFIG, type NanochatConfigState } from "./nanochat-config-policy.ts";
import {
  migrateNanochatConfig,
  NANOCHAT_CONFIG_PERSISTENCE_VERSION,
} from "./nanochat-config-persistence.ts";

export const NANOCHAT_CONFIG_PERSISTENCE_NAME = "labz_nanochat_config_v1";

export type NanochatConfigStore = NanochatConfigState & {
  apply: (patch: Partial<NanochatConfigState>) => void;
  reset: () => void;
};

export const useNanochatConfigStore = create<NanochatConfigStore>()(
  persist(
    (set) => ({
      ...DEFAULT_NANOCHAT_CONFIG,
      apply: (patch: Partial<NanochatConfigState>) => set(patch),
      reset: () => set({ ...DEFAULT_NANOCHAT_CONFIG }),
    }),
    {
      name: NANOCHAT_CONFIG_PERSISTENCE_NAME,
      storage: createJSONStorage(() => localStorage),
      version: NANOCHAT_CONFIG_PERSISTENCE_VERSION,
      // Only the configuration is persisted. Everything below is derived or
      // transient, and persisting it would restore a stale spinner or a
      // validation error from a previous session.
      partialize: (state) => {
        const { apply: _apply, reset: _reset, ...config } = state;
        void _apply;
        void _reset;
        return config as NanochatConfigState;
      },
      // The migrator works on the configuration alone, so the actions have to be
      // carried over from `current` afterwards. Returning a bare config here would
      // leave the store without `apply`/`reset` and every write would be a no-op.
      merge: (persisted, current) => ({
        ...current,
        ...migrateNanochatConfig(persisted as Partial<NanochatConfigState> | undefined, current),
      }),
    },
  ),
);

/** Read a slice without subscribing to the whole store. */
export const selectDepth = (state: NanochatConfigStore) => state.depth;
export const selectDataset = (state: NanochatConfigStore) => state.dataset;
export const selectNumIterations = (state: NanochatConfigStore) => state.numIterations;
export const selectParamMode = (state: NanochatConfigStore) => state.paramMode;

/**
 * Whether a run is currently active.
 *
 * The sidebar watches this to show a spinner on the row, the way it does for
 * Train. Read through the runtime store so the sidebar does not have to poll.
 */
export function isNanochatRunActive(state: { status: string }): boolean {
  return state.status === "running";
}
