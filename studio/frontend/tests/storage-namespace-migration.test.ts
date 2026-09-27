// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isAppStorageKey,
  KEY_PREFIX,
  KEY_PREFIX_DOTTED,
  migrateLegacyStorageKeys,
} from "../src/lib/storage-namespace.ts";

/** A Storage stand-in with the subset the migration touches, plus a key-order snapshot. */
function fakeStorage(seed: Record<string, string> = {}): Storage & { seed: Record<string, string> } {
  const map = new Map(Object.entries(seed));
  return {
    seed,
    get length() {
      return map.size;
    },
    key: (i: number) => [...map.keys()][i] ?? null,
    getItem: (k: string) => (map.has(k) ? map.get(k)! : null),
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
  } as Storage & { seed: Record<string, string> };
}

/** Run the migration against a given localStorage, as a fresh module would on boot. */
function withStorage<T>(store: Storage, body: () => T): T {
  const g = globalThis as unknown as { localStorage?: Storage };
  const prior = g.localStorage;
  Object.defineProperty(g, "localStorage", {
    value: store,
    configurable: true,
    writable: true,
  });
  try {
    return body();
  } finally {
    Object.defineProperty(g, "localStorage", {
      value: prior,
      configurable: true,
      writable: true,
    });
  }
}

test("the auth token and every other prefixed key survive the rename", () => {
  const store = fakeStorage({
    // The one that matters: losing it signs the user out.
    unsloth_auth_token: "a.b.c",
    unsloth_auth_refresh_token: "refresh",
    unsloth_hf_token: "hf_xxx",
    unsloth_chat_preferences: '{"theme":"dark"}',
    unsloth_user_profile: '{"displayName":"Robert"}',
    unsloth_interface_scale: "1.25",
    "unsloth.browser-account.v1": '{"username":"unsloth"}',
    unsloth_pinned_models: '["unsloth/gemma-4-E2B-it"]',
    // Not ours: must not be touched.
    theme: "dark",
    palette: "classic",
    sidebar_width: "260",
  });

  withStorage(store, migrateLegacyStorageKeys);

  assert.equal(store.getItem("labz_auth_token"), "a.b.c");
  assert.equal(store.getItem("labz_auth_refresh_token"), "refresh");
  assert.equal(store.getItem("labz_hf_token"), "hf_xxx");
  assert.equal(store.getItem("labz_chat_preferences"), '{"theme":"dark"}');
  assert.equal(store.getItem("labz_user_profile"), '{"displayName":"Robert"}');
  assert.equal(store.getItem("labz_interface_scale"), "1.25");
  assert.equal(store.getItem("labz.browser-account.v1"), '{"username":"unsloth"}');
  assert.equal(store.getItem("labz_pinned_models"), '["unsloth/gemma-4-E2B-it"]');

  // Unprefixed keys belong to nobody's namespace and must be left exactly where they are.
  assert.equal(store.getItem("theme"), "dark");
  assert.equal(store.getItem("palette"), "classic");
  assert.equal(store.getItem("sidebar_width"), "260");

  // The old names are gone, so a stale build cannot resurrect them and diverge.
  for (const old of [
    "unsloth_auth_token",
    "unsloth_chat_preferences",
    "unsloth.browser-account.v1",
  ]) {
    assert.equal(store.getItem(old), null, `${old} was left behind`);
  }
});

test("a value already written under the new name wins over the legacy one", () => {
  // This runs on every boot, so a key written since the migration is the live one and the
  // leftover is stale by definition. Getting this backwards would silently revert a setting.
  const store = fakeStorage({
    unsloth_chat_preferences: '{"old":true}',
    labz_chat_preferences: '{"new":true}',
  });
  withStorage(store, migrateLegacyStorageKeys);
  assert.equal(store.getItem("labz_chat_preferences"), '{"new":true}');
  assert.equal(store.getItem("unsloth_chat_preferences"), null);
});

test("migrating twice is a no-op", () => {
  const store = fakeStorage({ unsloth_auth_token: "a.b.c" });
  withStorage(store, () => {
    migrateLegacyStorageKeys();
    const once = store.getItem("labz_auth_token");
    migrateLegacyStorageKeys();
    assert.equal(store.getItem("labz_auth_token"), once);
    assert.equal(once, "a.b.c");
  });
});

test("a blocked localStorage does not throw", () => {
  const g = globalThis as unknown as { localStorage?: Storage };
  Object.defineProperty(g, "localStorage", {
    get() {
      throw new Error("blocked");
    },
    configurable: true,
  });
  try {
    assert.doesNotThrow(migrateLegacyStorageKeys);
  } finally {
    Object.defineProperty(g, "localStorage", {
      value: undefined,
      configurable: true,
      writable: true,
    });
  }
});

test("the sign-out sweep recognises both spellings", () => {
  // An account's keys have to be cleared whichever name they were written under; a key the
  // migration has not reached yet would otherwise leak into whoever signs in next.
  for (const key of [
    "unsloth_auth_token",
    "unsloth_chat_preferences",
    "unsloth.browser-account.v1",
    "unsloth-profile",
    "labz_auth_token",
    "labz_chat_preferences",
    "labz.browser-account.v1",
    "labz-profile",
  ]) {
    assert.equal(isAppStorageKey(key), true, `${key} not recognised`);
  }
  for (const key of ["theme", "palette", "chat-draft:1", "sidebar_width"]) {
    assert.equal(isAppStorageKey(key), false, `${key} wrongly claimed`);
  }
  // A broad root match on purpose. It also claims a hypothetical `unsloth/<model>`, which
  // costs nothing: no such key is ever written to localStorage, and the old sweep matched
  // the same way. Narrowing it to the two storage shapes would let an unlisted `unsloth-`
  // key survive a sign-out, which is the one failure this function exists to prevent.
  assert.equal(isAppStorageKey("unsloth/gemma-4-E2B-it"), true);
  assert.equal(KEY_PREFIX, "labz_");
  assert.equal(KEY_PREFIX_DOTTED, "labz.");
});
