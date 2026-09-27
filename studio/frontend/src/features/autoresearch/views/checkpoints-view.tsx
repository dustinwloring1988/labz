// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Every model the search produced, and a place to talk to one of them.
 *
 * Chatting happens here rather than by handing off to the Chat tab, and the
 * reason is that these checkpoints cannot be handed off. A checkpoint is a raw
 * `state_dict` plus a tokenizer that only autoresearch's own code in its own
 * virtualenv can read, so there is nothing the general chat stack can load. nanochat
 * gets away with a hand-off because its SFT and RL stages produce a model with a
 * chat template; autoresearch has no chat stage at all.
 *
 * So the conversation is a small local one, and it is labelled honestly: these
 * models continue text rather than reply, because that is what a five-minute
 * pretraining run on TinyStories learned to do.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import {
  Alert02Icon,
  ChatIcon,
  CpuIcon,
  StopCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { SectionCard } from "@/components/section-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";

import {
  getAutoresearchCheckpoints,
  loadAutoresearchChatModel,
  streamAutoresearchChat,
  unloadAutoresearchChatModel,
} from "../api/autoresearch-api";
import {
  isAutoresearchRunActive,
  useAutoresearchRuntimeStore,
} from "../stores/autoresearch-runtime-store";
import { formatBytes, formatBpb, formatParamsM } from "../lib/format";
import type { AutoresearchCheckpointInfo } from "../types/index.ts";

export function AutoresearchCheckpointsView() {
  const checkpoints = useAutoresearchRuntimeStore((state) => state.checkpoints);
  const setCheckpoints = useAutoresearchRuntimeStore((state) => state.setCheckpoints);
  const runActive = useAutoresearchRuntimeStore(isAutoresearchRunActive);

  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<AutoresearchCheckpointInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const rows = await getAutoresearchCheckpoints();
        if (cancelled) return;
        setCheckpoints(rows, null);
        setError(null);
      } catch (caught) {
        if (cancelled) return;
        setError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        if (!cancelled) setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setCheckpoints]);

  const handleOpen = useCallback(
    async (model: AutoresearchCheckpointInfo) => {
      setLoadingId(model.model_id);
      setError(null);
      try {
        await loadAutoresearchChatModel(model.model_id);
        setActiveModel(model);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        setLoadingId(null);
      }
    },
    [],
  );

  const handleClose = useCallback(async () => {
    try {
      await unloadAutoresearchChatModel();
    } catch {
      // Already gone, or the worker never came up. Either way the conversation
      // below is about to be torn down, so a failure here is not worth showing.
    }
    setActiveModel(null);
  }, []);

  // A running loop releases whatever was loaded, because the checkpoint and the
  // experiments cannot both have the GPU. Derived rather than cleared in an
  // effect: the conversation is closed for exactly as long as a loop is going,
  // and an effect would have to run to take it away again.
  const openModel = runActive ? null : activeModel;

  if (!loaded) {
    return (
      <div className="flex justify-center py-12">
        <Spinner />
      </div>
    );
  }

  if (openModel) {
    return (
      <CheckpointChat
        model={openModel}
        onClose={() => void handleClose()}
      />
    );
  }

  if (checkpoints.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-16 text-center">
        <p className="text-sm text-muted-foreground">No models yet.</p>
        <p className="max-w-md text-xs text-muted-foreground/80">
          autoresearch saves one file per run and overwrites it, so the studio copies
          each experiment&apos;s weights aside. One appears here after the first
          experiment that finishes.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 pb-8">
      {error ? (
        <p className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
          {error}
        </p>
      ) : null}

      {runActive ? (
        <p className="flex items-start gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-xs text-amber-700 dark:text-amber-400">
          <HugeiconsIcon icon={Alert02Icon} className="mt-px size-3.5 shrink-0" />
          <span>
            A loop is running and holds the GPU, so a model cannot be loaded until it
            stops. The weights are safe; the chat is what waits.
          </span>
        </p>
      ) : null}

      <ul className="flex flex-col gap-2">
        {checkpoints.map((model) => (
          <li
            key={model.model_id}
            className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border p-3"
          >
            <div className="flex min-w-0 flex-col gap-1">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-sm font-medium">{model.name}</span>
                {model.num_params_m !== null ? (
                  <Badge variant="outline" className="text-[10px]">
                    {formatParamsM(model.num_params_m)}
                  </Badge>
                ) : null}
                {model.depth !== null ? (
                  <Badge variant="outline" className="text-[10px]">
                    depth {model.depth}
                  </Badge>
                ) : null}
              </div>
              <p className="text-[11px] text-muted-foreground">{model.hint}</p>
              <p className="font-mono text-[10px] text-muted-foreground/70">
                {formatBytes(model.bytes)}
                {model.val_bpb !== null ? ` · val bpb ${formatBpb(model.val_bpb, 4)}` : ""}
                {model.commit ? ` · ${model.commit}` : ""}
              </p>
              {model.reason ? (
                <p className="text-[11px] text-amber-600 dark:text-amber-400">
                  {model.reason}
                </p>
              ) : null}
            </div>

            <Button
              variant="outline"
              disabled={!model.loadable || runActive || loadingId === model.model_id}
              onClick={() => void handleOpen(model)}
              className="gap-1.5"
            >
              <HugeiconsIcon icon={ChatIcon} className="size-4" />
              {loadingId === model.model_id ? "Loading…" : "Talk to this one"}
            </Button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * One conversation with one checkpoint.
 *
 * The transcript is held in component state rather than a store, on purpose. It
 * belongs to this view and nothing else reads it: there is no thread to persist,
 * no project to file it under, and no second surface that would need to agree
 * about it. A model loaded for a look is a model loaded for a look.
 */
function CheckpointChat({
  model,
  onClose,
}: {
  model: AutoresearchCheckpointInfo;
  onClose: () => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [turn, setTurn] = useState<{ input: string; output: string } | null>(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const outputRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => () => abort.current?.abort(), []);

  useEffect(() => {
    const node = outputRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [turn?.output]);

  const handleSend = useCallback(async () => {
    const text = prompt.trim();
    if (!text || generating) return;
    setPrompt("");
    setTurn({ input: text, output: "" });
    setGenerating(true);
    setError(null);

    const controller = new AbortController();
    abort.current = controller;
    try {
      await streamAutoresearchChat({
        modelId: model.model_id,
        prompt: text,
        maxNewTokens: 400,
        temperature: 0.8,
        signal: controller.signal,
        onText: (cumulative) => setTurn({ input: text, output: cumulative }),
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setGenerating(false);
      abort.current = null;
    }
  }, [generating, model.model_id, prompt]);

  const handleStop = useCallback(() => {
    abort.current?.abort();
  }, []);

  return (
    <div className="flex flex-col gap-3 pb-8">
      <SectionCard
        icon={<HugeiconsIcon icon={CpuIcon} className="size-5" />}
        title={model.name}
        description={model.hint}
        headerAction={
          <Button variant="secondary" size="sm" onClick={onClose} className="gap-1.5">
            <HugeiconsIcon icon={StopCircleIcon} className="size-4" />
            Unload
          </Button>
        }
      >
        <div className="flex flex-col gap-2 text-[11px] text-muted-foreground">
          <p>
            This model was trained from scratch on TinyStories for about five minutes.
            It has no chat stage, so it does not answer questions — it continues text
            the way it was trained to. A prompt that reads like the start of a story
            gets a lot further than a question.
          </p>
          <p>
            {model.num_params_m !== null ? `${formatParamsM(model.num_params_m)}. ` : ""}
            {model.val_bpb !== null
              ? `Scored val bpb ${formatBpb(model.val_bpb)}. `
              : ""}
            {formatBytes(model.bytes)} on disk.
          </p>
        </div>
      </SectionCard>

      <div
        ref={outputRef}
        className="flex min-h-64 flex-col gap-3 overflow-auto rounded-2xl border border-border bg-muted/20 p-4"
      >
        {turn === null ? (
          <p className="text-ui-12 text-muted-foreground">
            Try continuing a sentence. &quot;Once upon a time, a small robot woke up
            and&quot;
          </p>
        ) : (
          <>
            <div className="flex flex-col gap-1">
              <span className="text-ui-10 uppercase tracking-wide text-muted-foreground">
                Prompt
              </span>
              <p className="whitespace-pre-wrap text-ui-13">{turn.input}</p>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-ui-10 uppercase tracking-wide text-muted-foreground">
                Continuation
              </span>
              <p className="whitespace-pre-wrap text-ui-13">
                {turn.output || (
                  <span className="text-muted-foreground">
                    {generating ? "thinking…" : "(nothing)"}
                  </span>
                )}
                {generating ? (
                  <span className="ml-0.5 inline-block animate-pulse">█</span>
                ) : null}
              </p>
            </div>
          </>
        )}
      </div>

      {error ? (
        <p className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
          {error}
        </p>
      ) : null}

      <div className="flex items-end gap-2">
        <Textarea
          rows={2}
          value={prompt}
          disabled={generating}
          placeholder="Once upon a time,"
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            // Enter sends, Shift+Enter is a newline: the convention every
            // composer in this app uses.
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void handleSend();
            }
          }}
        />
        {generating ? (
          <Button variant="outline" onClick={handleStop}>
            Stop
          </Button>
        ) : (
          <Button disabled={!prompt.trim()} onClick={() => void handleSend()}>
            Continue
          </Button>
        )}
      </div>
    </div>
  );
}
