// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The models a report can be written by, from all three sources.
 *
 * A model can live in three places, and the reason the agent comes first is the
 * whole point of this list:
 *
 *  - **An agent CLI.** Usually a *local* model, because that is where a user
 *    configures one. The same opencode or Codex that runs the experiments can also
 *    write the report, which means the report is written by whatever weights the
 *    search ran on rather than by a second, different model the user has to
 *    remember to pick.
 *  - **Catalog models.** Anything the studio can load and serve, from the same
 *    list the Chat tab's own model picker uses.
 *  - **This search's checkpoints.** The honest fallback when nothing else is
 *    available, and informative in its own right.
 *
 * They are kept distinct and labelled, because a report is a summarisation task
 * over a table of numbers. A 50M-parameter model that spent five minutes on
 * TinyStories writes fluent confident nonsense about it, and the user has to be
 * able to see which kind of model they are trusting with that.
 */
import { useEffect, useMemo, useState } from "react";

import { listModels } from "@/features/chat";

import {
  getAutoresearchAgents,
  getAutoresearchCheckpoints,
} from "../api/autoresearch-api";
import { agentLabel } from "../stores/autoresearch-config-policy";
import { useAutoresearchRuntimeStore } from "../stores/autoresearch-runtime-store";
import type { AutoresearchReportModel } from "../types/index.ts";

/** The number of catalog models listed before the list stops being a picker. */
const MAX_LISTED_MODELS = 40;

export function useReportModels(): {
  models: AutoresearchReportModel[];
  loading: boolean;
  /** True when no catalog model is available, so the picker can say so. */
  catalogEmpty: boolean;
  /** True when no agent is runnable, so the picker can say so. */
  agentsEmpty: boolean;
} {
  const [catalog, setCatalog] = useState<AutoresearchReportModel[]>([]);
  const [agents, setAgents] = useState<AutoresearchReportModel[]>([]);
  const [settled, setSettled] = useState(false);
  const checkpoints = useAutoresearchRuntimeStore((state) => state.checkpoints);
  const setCheckpoints = useAutoresearchRuntimeStore((state) => state.setCheckpoints);

  // Read failure-tolerantly on purpose. A search is still worth reporting on
  // when one of these lists cannot be fetched, and the others still work.
  useEffect(() => {
    let cancelled = false;
    void listModels()
      .then((payload) => {
        if (cancelled) return;
        setCatalog(
          (payload.models ?? [])
            .filter((entry) => typeof entry.id === "string" && entry.id.length > 0)
            // A LoRA is an adapter, not something the completions path will serve
            // on its own, so it is not offered here.
            .filter((entry) => !entry.is_lora)
            .slice(0, MAX_LISTED_MODELS)
            .map((entry) => ({
              id: entry.id,
              label: entry.name || entry.id,
              source: "catalog" as const,
              note: "A model the studio can load and serve.",
            })),
        );
      })
      .catch(() => {
        // Left empty, and the picker explains what to do about it.
        if (!cancelled) setCatalog([]);
      });

    void getAutoresearchAgents()
      .then((response) => {
        if (cancelled) return;
        setAgents(
          response.agents
            .filter((agent) => agent.available)
            .map((agent) => ({
              id: `agent:${agent.key}`,
              label: `${agent.label}${agent.version ? ` ${agent.version}` : ""}`,
              source: "agent" as const,
              agentKey: agent.key,
              note:
                "Asks that agent for prose, using whatever model it is configured " +
                "for. Usually a local one, and usually the same one that ran the " +
                "experiments.",
            })),
        );
      })
      .catch(() => {
        if (!cancelled) setAgents([]);
      })
      .finally(() => {
        if (!cancelled) setSettled(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    void getAutoresearchCheckpoints()
      .then((rows) => {
        if (!cancelled) setCheckpoints(rows, null);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [setCheckpoints]);

  const models = useMemo<AutoresearchReportModel[]>(() => {
    const fromCheckpoints = checkpoints
      .filter((checkpoint) => checkpoint.loadable)
      .map((checkpoint) => ({
        id: checkpoint.model_id,
        label: checkpoint.name,
        source: "autoresearch" as const,
        note:
          "One of this search's own models. A report from it is the experiment " +
          "describing itself, not an assessment of it.",
      }));
    // Agents first, then the catalog, then the checkpoints. A run should land on
    // the model the user actually has, and a checkpoint is the last resort rather
    // than the first thing a picker offers.
    return [...agents, ...catalog, ...fromCheckpoints];
  }, [agents, catalog, checkpoints]);

  return {
    models,
    loading: !settled,
    catalogEmpty: settled && catalog.length === 0,
    agentsEmpty: settled && agents.length === 0,
  };
}

/** The suffix a picker entry gets, so the three sources are distinguishable. */
export function reportSourceSuffix(model: AutoresearchReportModel): string {
  if (model.source === "agent") return " (agent)";
  if (model.source === "autoresearch") return " (this search's own)";
  return "";
}

/** How a chosen entry becomes a request, since an agent carries its key apart. */
export function reportRequestFor(model: AutoresearchReportModel): {
  modelId: string;
  source: "catalog" | "autoresearch" | "agent";
  agentKey: string;
} {
  return {
    modelId: model.id,
    source: model.source,
    agentKey: model.agentKey ?? agentLabel(model.id),
  };
}
