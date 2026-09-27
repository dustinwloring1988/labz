// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which sidebar an install ends up with, and when it is allowed to be told.
//
// The rule has two halves and both are load-bearing. An install that never chose
// anything should pick up a new row when one ships; an install that arranged its
// own sidebar should be left alone. Getting the second wrong overwrites a real
// choice on every startup, and getting the first wrong is how a row ships to
// nobody -- which is what happened to autoresearch.
//
// The layouts are consolidated to one baseline, so these tests no longer walk a
// history. They pin the two rules, the version pairing, and the reset that
// consolidation depends on.

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_CUSTOMIZATION,
  SIDEBAR_NAV_ITEM_IDS,
  type SidebarNavItemPref,
  migrateShippedSidebarNavDefault,
  sanitizeCustomization,
} from "../src/features/settings/stores/appearance-custom-store.ts";

import { readSrcAsync } from "./helpers/kit.ts";

/** The version the store persists, which is what gates a local rehydrate.
 *
 * Read from the source so a bump cannot leave these tests asserting against a
 * version nothing ships. Distinct from PERSONALIZATION_VERSION, which gates the
 * remote profile path.
 */
async function sidebarNavStoreVersion(): Promise<number> {
  const source = await readSrcAsync(
    "features/settings/stores/appearance-custom-store.ts",
  );
  const match = /version: (\d+),\r?\n\s+storage: createJSONStorage/.exec(source);
  assert.ok(match, "no persisted version in the appearance store");
  return Number(match[1]);
}

async function personalizationVersion(): Promise<number> {
  const source = await readSrcAsync(
    "features/profile/hooks/use-personalization-sync.ts",
  );
  const match = /PERSONALIZATION_VERSION = (\d+)/.exec(source);
  assert.ok(match, "no PERSONALIZATION_VERSION in the sync module");
  return Number(match[1]);
}

/** The one shipped layout, read from the source rather than copied.
 *
 * A hand-copied fixture is how a test ends up asserting against a layout the
 * product stopped shipping, which passes for the wrong reason.
 */
async function shippedLayout(): Promise<SidebarNavItemPref[]> {
  const source = await readSrcAsync(
    "features/settings/stores/appearance-custom-store.ts",
  );
  const block = /const SHIPPED_SIDEBAR_NAV_DEFAULTS[\s\S]*?= \[([\s\S]*?)\n\];/.exec(
    source,
  );
  assert.ok(block, "no SHIPPED_SIDEBAR_NAV_DEFAULTS in the store");
  return sanitizeCustomization({
    sidebarNav: [...block[1].matchAll(/id: "([a-z-]+)", pinned: (true|false)/g)]
      .map((match) => ({
        id: match[1],
        pinned: match[2] === "true",
      })) as SidebarNavItemPref[],
  }).sidebarNav;
}

test("the shipped layout is the current default", async () => {
  // The whole point of the reset: one layout, and it is the one that ships.
  assert.deepEqual(await shippedLayout(), DEFAULT_CUSTOMIZATION.sidebarNav);
});

test("autoresearch is in the shipped layout, pinned", async () => {
  // Named rather than implied, because this is the row that was missing: a
  // leaderboard-style measurement tab that never reached the installs already on
  // the version before it.
  const layout = await shippedLayout();
  const row = layout.find((item) => item.id === "autoresearch");
  assert.ok(row, "autoresearch must be in the shipped layout");
  assert.equal(row.pinned, true, "and pinned, or it hides under More");
  // Ordered as a sibling of nanochat rather than a new kind of thing.
  const ids = layout.filter((item) => item.pinned).map((item) => item.id);
  assert.deepEqual(ids, [
    "hub",
    "projects",
    "images",
    "video",
    "train",
    "nanochat",
    "autoresearch",
    "benchmarks",
  ]);
});

test("an install on an older version adopts the shipped layout", async () => {
  // The regression the consolidation exists to fix. The counter was reset to 2, so
  // installs sitting on 9 or 10 are *newer* than the store -- and a `>=` guard reads
  // that as "already migrated" and skips, stranding them on the layout they had.
  // That is precisely how autoresearch went missing: the row shipped under a
  // version the affected installs had already passed.
  const version = await sidebarNavStoreVersion();
  assert.ok(
    version < 10,
    `the store is at ${version}; the layouts it replaced ran to 10, so this test ` +
      "is no longer covering the reset",
  );
  const migrated = migrateShippedSidebarNavDefault(
    sanitizeCustomization({ sidebarNav: DEFAULT_CUSTOMIZATION.sidebarNav }),
    10,
    version,
  );
  assert.deepEqual(migrated.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
});

test("an install on the same version is left alone", async () => {
  // The other half, and the reason the guard exists at all: once this version has
  // been persisted, a layout that looks shipped may be a choice made since, and
  // re-adopting the default would overwrite it on every startup.
  const version = await sidebarNavStoreVersion();
  const shipped = await shippedLayout();
  const customization = sanitizeCustomization({ sidebarNav: shipped });
  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, version, version),
    customization,
  );
});

test("a user-arranged sidebar survives the migration", () => {
  // Not a shipped arrangement, so the migration must not touch it. This is the
  // guarantee that makes adopting-by-default safe.
  const customization = sanitizeCustomization({
    sidebarNav: DEFAULT_CUSTOMIZATION.sidebarNav.map((item) =>
      item.id === "recipes" ? { ...item, pinned: true } : item,
    ),
  });
  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 1, 2),
    customization,
  );
});

test("a customized sidebar keeps its order and only gains the new rows", () => {
  // An arranged layout is not rebuilt, so a row that arrives after it was written
  // has nowhere to go but the end -- in order of arrival, with its own default pin.
  // Audio lands unpinned under "More" while autoresearch and Benchmarks land
  // pinned, and asserting the last row is unpinned would pass by accident under
  // the old ordering, which is the point.
  const arranged: SidebarNavItemPref[] = [
    { id: "projects", pinned: true },
    { id: "hub", pinned: true },
    { id: "train", pinned: true },
    { id: "images", pinned: true },
    { id: "video", pinned: false },
    { id: "recipes", pinned: true },
    { id: "export", pinned: false },
  ];
  const customization = sanitizeCustomization({ sidebarNav: arranged });

  assert.strictEqual(
    migrateShippedSidebarNavDefault(customization, 1, 2),
    customization,
  );
  // The arranged rows keep their order and their pins, and every id the app knows
  // is present exactly once. Which ids are appended, and in what order, is a
  // function of SIDEBAR_NAV_ITEM_IDS rather than anything this test should pin --
  // asserting the exact list here would break every time a row is added, and would
  // be asserting the wrong thing when it did.
  assert.deepEqual(
    customization.sidebarNav.slice(0, arranged.length),
    arranged,
    "the rows the user arranged come first, in order, unchanged",
  );
  const ids = customization.sidebarNav.map((item) => item.id);
  assert.equal(new Set(ids).size, ids.length, "no id appears twice");
  for (const known of SIDEBAR_NAV_ITEM_IDS) {
    assert.ok(ids.includes(known), `${known} must be present`);
  }
  // Each appended row takes its own default pin, so autoresearch lands visible
  // while audio and api land under "More".
  const appended = new Map(
    customization.sidebarNav
      .slice(arranged.length)
      .map((item) => [item.id, item.pinned]),
  );
  assert.equal(appended.get("autoresearch"), true, "a new row ships visible");
  assert.equal(appended.get("audio"), false, "an unpinned row stays under More");
});

test("a layout missing a trailing row is recognised as untouched once sanitized", () => {
  // The mechanism the reset leans on, and its limit. Sanitize appends an id the
  // payload predates at the END, so a layout missing a trailing row matches a
  // shipped one once that trailing run is stripped.
  const withoutTrailing = DEFAULT_CUSTOMIZATION.sidebarNav.slice(0, -1);
  const migrated = migrateShippedSidebarNavDefault(
    sanitizeCustomization({ sidebarNav: withoutTrailing }),
    1,
    2,
  );
  assert.deepEqual(migrated.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
});

test("a row missing from the middle is not treated as untouched", () => {
  // The limit, stated so it is a decision rather than a surprise. A layout missing
  // a row from the middle is indistinguishable from one whose owner moved or
  // unpinned it, and re-adopting the default there would overwrite a real choice.
  // So the row is appended at the end by sanitize and the arrangement is kept --
  // visible, but where the install put it, not where the default would.
  const withoutMiddle = DEFAULT_CUSTOMIZATION.sidebarNav.filter(
    (item) => item.id !== "autoresearch",
  );
  const migrated = migrateShippedSidebarNavDefault(
    sanitizeCustomization({ sidebarNav: withoutMiddle }),
    1,
    2,
  );
  assert.deepEqual(
    migrated.sidebarNav.map((item) => item.id),
    // Appended last, and autoresearch stays pinned so it is not hidden under More.
    [...withoutMiddle.map((item) => item.id), "autoresearch"],
  );
  assert.equal(
    migrated.sidebarNav.at(-1)?.pinned,
    true,
    "an appended row takes its own default pin",
  );
});

test("a synced profile picks the layout change up too", async () => {
  // Remote hydration replaces the local store wholesale, so a nav default that
  // only migrates locally is overwritten by the stored layout on every login.
  // PERSONALIZATION_VERSION has to move with the layout for that migration to run
  // against the remote record.
  const version = await personalizationVersion();
  const stored = sanitizeCustomization({ sidebarNav: await shippedLayout() });
  const migrated = migrateShippedSidebarNavDefault(
    stored,
    version - 1,
    version,
  );
  assert.deepEqual(migrated.sidebarNav, DEFAULT_CUSTOMIZATION.sidebarNav);
});
