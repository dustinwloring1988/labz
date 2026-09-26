// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Shared bits for the nanochat charts.
 *
 * The sync id is separate from Train's on purpose: the two id spaces are
 * separate brush domains, and sharing one would let a nanochat chart silently
 * adopt Train's zoom (or the reverse) if both were ever mounted at once.
 */
export const NANOCHAT_CHART_SYNC_ID = "nanochat-metrics-sync";
