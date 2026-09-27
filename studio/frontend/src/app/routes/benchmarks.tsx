// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

/**
 * Benchmarks measure a model and leave a number behind; they do not change one.
 * That makes it a route of its own rather than a mode inside Train: the work it
 * does (loading a model, running inference, persisting a result) has nothing in
 * common with a training step, and the two are mutually exclusive because both
 * want the whole GPU.
 *
 * Reachable on a chat-only host, where the page explains that the datasets are
 * not set up rather than redirecting: a benchmark needs a model, not a trainer.
 */
const BenchmarksPage = lazyRouteComponent(
  () => import("@/features/benchmarks"),
  "BenchmarkPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/benchmarks",
  staticData: { title: "Benchmarks" },
  beforeLoad: () => requireAuth(),
  component: BenchmarksPage,
});
