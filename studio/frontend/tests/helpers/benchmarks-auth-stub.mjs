// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for `@/features/auth` so a pure payload/parser test does not pull in
// token storage. Any call is a test bug rather than a thing to assert on, so it
// throws instead of quietly returning something.
export async function authFetch() {
  throw new Error("authFetch was called in a test that does not exercise the network");
}
