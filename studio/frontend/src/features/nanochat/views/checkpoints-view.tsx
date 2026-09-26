// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Checkpoints on disk, and the bridge to the normal Chat tab.
 *
 * A nanochat checkpoint cannot be loaded while a run is going: the run has the
 * GPU, and the checkpoint would need a second copy of the weights. So this view
 * is honest about it — the button explains and offers to stop the run rather than
 * failing on click.
 */
import { useCallback, useEffect, useState } from "react";

import { Alert02Icon, ChatIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { formatBytes } from "../lib/depth-presets";
import {
  loadNanochatChatModel,
  type NanochatChatModel,
} from "../api/nanochat-api";
import { getNanochatChatModels } from "../api/nanochat-api";
import {
  isNanochatRunActive,
  useNanochatRuntimeStore,
} from "../stores/nanochat-runtime-store";

const SOURCE_LABELS: Record<string, string> = {
  base: "Base",
  sft: "Fine-tuned",
  rl: "Reinforcement learning",
};

const SOURCE_HINTS: Record<string, string> = {
  base: "Completes text. It has not been taught to answer questions yet.",
  sft: "Fine-tuned for conversation. This is the one to chat with.",
  rl: "Post-trained with reinforcement learning on GSM8K.",
};

export function NanochatCheckpointsView() {
  const [models, setModels] = useState<NanochatChatModel[] | null>(null);
  const [resident, setResident] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const runActive = useNanochatRuntimeStore(isNanochatRunActive);
  const navigate = useNavigate();

  // The effect calls the API rather than setting state itself: the write happens
  // in the promise continuation, so the first render is a spinner and the list
  // arrives without a cascading render.
  useEffect(() => {
    let cancelled = false;
    void getNanochatChatModels()
      .then((response) => {
        if (cancelled) return;
        setModels(response.models);
        setResident(response.resident?.model ?? null);
      })
      .catch((cause) => {
        if (cancelled) return;
        setModels([]);
        setError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleChat = useCallback(
    async (model: NanochatChatModel) => {
      setLoadingId(model.model_id);
      setError(null);
      try {
        await loadNanochatChatModel(model.model_id);
        // Hand off to the normal chat tab, which is where the conversation
        // happens. The checkpoint stays loaded there until something unloads it.
        await navigate({ to: "/chat" });
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setLoadingId(null);
      }
    },
    [navigate],
  );

  if (models === null) {
    return (
      <div className="flex justify-center py-12">
        <Spinner />
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-16 text-center">
        <p className="text-sm text-muted-foreground">No checkpoints yet.</p>
        <p className="max-w-md text-xs text-muted-foreground/80">
          Checkpoints appear here once a run reaches its first save. A pretraining
          run only saves at the end unless you set a checkpoint interval.
        </p>
      </div>
    );
  }

  // Newest first, so a just-finished run's checkpoint is at the top.
  const sorted = [...models].sort((a, b) => b.modified - a.modified);

  return (
    <div className="flex flex-col gap-3">
      {error ? (
        <p className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
          {error}
        </p>
      ) : null}

      {runActive ? (
        <p className="flex items-start gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-xs text-amber-700 dark:text-amber-400">
          <HugeiconsIcon icon={Alert02Icon} className="mt-px size-3.5 shrink-0" />
          <span>
            A run is in progress and holds the GPU, so a checkpoint cannot be loaded
            for chat until it stops.
          </span>
        </p>
      ) : null}

      <ul className="flex flex-col gap-2">
        {sorted.map((model) => {
          const isResident = resident === model.model_id;
          return (
            <li
              key={model.model_id}
              className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border p-3"
            >
              <div className="flex min-w-0 flex-col gap-1">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-sm font-medium">{model.name}</span>
                  <Badge variant="outline" className="text-[10px]">
                    {SOURCE_LABELS[model.source] ?? model.source}
                  </Badge>
                  {isResident ? (
                    <Badge className="bg-emerald-500/15 text-[10px] text-emerald-600 dark:text-emerald-400">
                      loaded
                    </Badge>
                  ) : null}
                </div>
                <p className="text-[11px] text-muted-foreground">
                  {SOURCE_HINTS[model.source]}
                </p>
                <p className="font-mono text-[10px] text-muted-foreground/70">
                  {formatBytes(model.bytes)}
                  {model.num_params ? ` · ${model.num_params.toLocaleString()} params` : ""}
                  {model.val_bpb !== null && model.val_bpb !== undefined
                    ? ` · val bpb ${model.val_bpb.toFixed(4)}`
                    : ""}
                </p>
                {model.reason ? (
                  <p className="text-[11px] text-amber-600 dark:text-amber-400">
                    {model.reason}
                  </p>
                ) : null}
              </div>

              <Button
                variant={isResident ? "secondary" : "outline"}
                disabled={!model.loadable || runActive || loadingId === model.model_id}
                onClick={() => void handleChat(model)}
                className="gap-1.5"
              >
                <HugeiconsIcon icon={ChatIcon} className="size-4" />
                {loadingId === model.model_id
                  ? "Loading…"
                  : isResident
                    ? "Open chat"
                    : "Chat with this"}
              </Button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
