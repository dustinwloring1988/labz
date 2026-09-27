// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Owns the browser-storage namespace, and carries every pre-rebrand key forward on first run.
//
// The rename from `unsloth_*` / `unsloth.` to `labz_*` / `labz.` is not a cosmetic change: these
// keys hold the auth token, the chat settings, the HF token, pinned models, the interface scale
// and the appearance. Renaming them without a migration logs everyone out and resets their
// settings, so the old keys are copied rather than abandoned.
//
// Deliberately has no imports. It is imported first in main.tsx precisely so that it evaluates
// before any module that creates a persisted store -- zustand's persist hydrates synchronously at
// store construction, which happens at module evaluation, so a migration that ran any later
// would find the stores already holding defaults.

// The namespace new keys are written under. Two shapes, matching the two the old keys used:
// an underscore for flat flags, a dot for versioned or namespaced records.
export const KEY_PREFIX = "labz_";
export const KEY_PREFIX_DOTTED = "labz.";

// Prefixes a key may have carried before the rebrand. An event name (`unsloth:...`) is not in
// here: those are in-process only, never persisted, so they were renamed in place with no
// migration behind them.
const LEGACY_PREFIXES: readonly string[] = ["unsloth_", "unsloth."];

// The root each namespace hangs off. The migration uses the two precise prefixes above; the
// sign-out sweep uses these, so it keeps the breadth the old `startsWith("unsloth")` had.
const NAMESPACE_ROOT = "labz";
const LEGACY_NAMESPACE_ROOT = "unsloth";

// Set once the sweep has run, so a reload does not re-walk storage. Named under the new prefix
// so it is not itself a candidate for migration on a later run.
const MIGRATION_SENTINEL = "labz_storage_namespace_migrated_v1";

function currentPrefixOf(key: string): string | null {
  for (const legacy of LEGACY_PREFIXES) {
    if (key.startsWith(legacy)) return legacy;
  }
  return null;
}

function stores(): Storage[] {
  // A blocked or absent localStorage (private browsing, a sandboxed webview) must not take the
  // app down; it just means there is nothing to migrate.
  const found: Storage[] = [];
  try {
    if (typeof localStorage !== "undefined") found.push(localStorage);
  } catch {
    /* access itself can throw */
  }
  try {
    if (typeof sessionStorage !== "undefined") found.push(sessionStorage);
  } catch {
    /* access itself can throw */
  }
  return found;
}

/**
 * Copy every legacy-prefixed key onto its new name, then drop the old one.
 *
 * The new value wins where both exist: this runs on every boot, so a key the user has since
 * written under the new name is the live one and the leftover is stale by definition.
 */
export function migrateLegacyStorageKeys(): void {
  for (const store of stores()) {
    try {
      if (store.getItem(MIGRATION_SENTINEL)) continue;

      // Snapshot first: removing while enumerating a live Storage skips entries.
      const legacyKeys: string[] = [];
      for (let i = 0; i < store.length; i += 1) {
        const key = store.key(i);
        if (key && currentPrefixOf(key)) legacyKeys.push(key);
      }

      for (const oldKey of legacyKeys) {
        const legacyPrefix = currentPrefixOf(oldKey)!;
        const newPrefix = legacyPrefix.startsWith("unsloth_")
          ? KEY_PREFIX
          : KEY_PREFIX_DOTTED;
        const newKey = newPrefix + oldKey.slice(legacyPrefix.length);
        if (store.getItem(newKey) === null) {
          const value = store.getItem(oldKey);
          if (value !== null) store.setItem(newKey, value);
        }
        store.removeItem(oldKey);
      }

      store.setItem(MIGRATION_SENTINEL, new Date().toISOString());
    } catch {
      /* one unusable store must not stop the others */
    }
  }
}

/**
 * Whether `key` is ours, under either spelling.
 *
 * Deliberately a broad root match rather than the two precise shapes above, because the sign-out
 * sweep has to be exhaustive: anything of ours that survives an account switch is the previous
 * account's content handed to whoever signs in next. The old sweep matched `startsWith("unsloth")`
 * and caught keys like `unsloth-profile` that the precise prefixes miss, so `labz` is matched the
 * same way. The migration above stays precise on purpose -- it renames only shapes we actually
 * wrote, and copying a key we do not own would be worse than leaving it.
 */
export function isAppStorageKey(key: string): boolean {
  return key.startsWith(NAMESPACE_ROOT) || key.startsWith(LEGACY_NAMESPACE_ROOT);
}

migrateLegacyStorageKeys();
