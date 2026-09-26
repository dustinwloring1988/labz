// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

/**
 * nanochat trains a model from scratch, in its own virtualenv, through its own
 * pipeline (download data, tokenizer, pretrain, evaluate, SFT, evaluate, RL). It
 * is a separate path from Train, which fine-tunes an existing model, so it gets
 * its own route rather than a mode inside Train.
 */
const NanochatPage = lazyRouteComponent(
  () => import("@/features/nanochat"),
  "NanochatPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/nanochat",
  staticData: { title: "nanochat" },
  beforeLoad: () => requireAuth(),
  component: NanochatPage,
});
