// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

/**
 * autoresearch runs nanochat's from-scratch pipeline under an agent loop: it
 * edits the training script, trains for a fixed budget, keeps the change if the
 * validation metric improved, and repeats for a set number of experiments.
 *
 * It is a separate path from both Train (which fine-tunes an existing model) and
 * nanochat (which runs one pipeline once), so it gets its own route rather than a
 * mode inside either of them.
 */
const AutoresearchPage = lazyRouteComponent(
  () => import("@/features/autoresearch"),
  "AutoresearchPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/autoresearch",
  staticData: { title: "autoresearch" },
  beforeLoad: () => requireAuth(),
  component: AutoresearchPage,
});
