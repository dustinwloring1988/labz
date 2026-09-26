// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { openStreamResponse } from "@/lib/open-stream-response";
import { takeSseFrame } from "@/lib/sse-framing";

import type {
  NanochatBenchmark,
  NanochatCatalogue,
  NanochatEnvironmentStatus,
  NanochatFit,
  NanochatLogLine,
  NanochatPresetResponse,
  NanochatSample,
  NanochatStatus,
  NanochatValidateResult,
} from "../types";
import type { NanochatConfigState } from "../stores/nanochat-config-policy";

/**
 * The nanochat request payload, built from the edit form.
 *
 * Mapping lives here, in one place, because the form is camelCase and the API is
 * snake_case. Scattering that conversion through the components is how a field
 * silently stops being sent.
 *
 * `paramMode` is deliberately not sent: it is a display choice, and sending it
 * would make the backend care about how the form is drawn.
 */
export function buildNanochatStartPayload(config: NanochatConfigState) {
  return {
    depth: config.depth,
    aspect_ratio: config.aspectRatio,
    head_dim: config.headDim,
    max_seq_len: config.maxSeqLen,
    window_pattern: config.windowPattern,
    model_tag: config.modelTag,
    dataset: config.dataset,
    num_shards: config.numShards,
    vocab_size: config.vocabSize,
    train_tokenizer: config.trainTokenizer,
    num_iterations: config.numIterations,
    param_data_ratio: config.paramDataRatio,
    total_batch_size: config.totalBatchSize,
    device_batch_size: config.deviceBatchSize,
    fp8: config.fp8,
    embedding_lr: config.embeddingLr,
    unembedding_lr: config.unembeddingLr,
    matrix_lr: config.matrixLr,
    scalar_lr: config.scalarLr,
    weight_decay: config.weightDecay,
    warmup_steps: config.warmupSteps,
    warmdown_ratio: config.warmdownRatio,
    final_lr_frac: config.finalLrFrac,
    eval_every: config.evalEvery,
    eval_tokens: config.evalTokens,
    core_metric_every: config.coreMetricEvery,
    sample_every: config.sampleEvery,
    save_every: config.saveEvery,
    core_metric_max_per_task: config.coreMetricMaxPerTask,
    sft_tasks: config.sftTasks,
    mmlu_epochs: config.mmluEpochs,
    gsm8k_epochs: config.gsm8kEpochs,
    sft_iterations: config.sftIterations,
    sft_sample_every: config.sftSampleEvery,
    run_rl: config.runRl,
    rl_epochs: config.rlEpochs,
    rl_examples_per_step: config.rlExamplesPerStep,
    rl_num_samples: config.rlNumSamples,
    rl_max_new_tokens: config.rlMaxNewTokens,
    rl_eval_every: config.rlEvalEvery,
    rl_save_every: config.rlSaveEvery,
    base_benchmarks: config.baseBenchmarks,
    chat_benchmarks: config.chatBenchmarks,
    stages: config.stages,
  };
}

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  throw new Error(await readFastApiError(response));
}

// --- environment ---

export async function getNanochatEnvironment(): Promise<NanochatEnvironmentStatus> {
  const response = await authFetch("/api/nanochat/environment");
  return jsonOrThrow<NanochatEnvironmentStatus>(response);
}

export async function installNanochat(): Promise<NanochatEnvironmentStatus> {
  const response = await authFetch("/api/nanochat/environment/install", { method: "POST" });
  return jsonOrThrow<NanochatEnvironmentStatus>(response);
}

// --- catalogue and presets ---

export async function getNanochatCatalogue(force = false): Promise<NanochatCatalogue> {
  const response = await authFetch(
    `/api/nanochat/catalogue${force ? "?force=true" : ""}`,
  );
  return jsonOrThrow<NanochatCatalogue>(response);
}

export async function getNanochatPresets(): Promise<NanochatPresetResponse> {
  const response = await authFetch("/api/nanochat/presets");
  return jsonOrThrow<NanochatPresetResponse>(response);
}

/**
 * What a depth costs on this machine.
 *
 * Called as the user changes the depth, so it is a cheap GET and the response is
 * discarded if a newer request has already come back. The caller handles the
 * ordering; this only avoids a throw on an abort.
 */
export async function getNanochatFit(params: {
  depth: number;
  numIterations: number;
  totalBatchSize: number;
  maxSeqLen: number;
  signal?: AbortSignal;
}): Promise<NanochatFit> {
  const query = new URLSearchParams({
    depth: String(params.depth),
    num_iterations: String(params.numIterations),
    total_batch_size: String(params.totalBatchSize),
    max_seq_len: String(params.maxSeqLen),
  });
  const response = await authFetch(`/api/nanochat/presets/${params.depth}/fit?${query}`, {
    signal: params.signal,
  });
  return jsonOrThrow<NanochatFit>(response);
}

export async function validateNanochatConfig(
  config: NanochatConfigState,
  signal?: AbortSignal,
): Promise<NanochatValidateResult> {
  const response = await authFetch("/api/nanochat/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config: buildNanochatStartPayload(config) }),
    signal,
  });
  return jsonOrThrow<NanochatValidateResult>(response);
}

// --- run lifecycle ---

export async function startNanochatRun(
  config: NanochatConfigState,
): Promise<{ run_id: string; status: string; message: string; fit: NanochatFit | null }> {
  const response = await authFetch("/api/nanochat/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildNanochatStartPayload(config)),
  });
  return jsonOrThrow(response);
}

export async function stopNanochatRun(save: boolean): Promise<{ status: string; message: string }> {
  const response = await authFetch("/api/nanochat/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ save }),
  });
  return jsonOrThrow(response);
}

export async function getNanochatStatus(
  runId?: string | null,
): Promise<NanochatStatus> {
  const query = runId ? `?expected_run_id=${encodeURIComponent(runId)}` : "";
  const response = await authFetch(`/api/nanochat/status${query}`);
  return jsonOrThrow<NanochatStatus>(response);
}

export async function getNanochatMetrics(
  runId?: string | null,
): Promise<Record<string, { step: number; value: number | null; total_steps: number | null }[]>> {
  const query = runId ? `?expected_run_id=${encodeURIComponent(runId)}` : "";
  const response = await authFetch(`/api/nanochat/metrics${query}`);
  const body = await jsonOrThrow<{ series: Record<string, { step: number; value: number | null; total_steps: number | null }[]> }>(
    response,
  );
  return body.series;
}

export async function getNanochatSamples(
  limit = 100,
  runId?: string | null,
): Promise<NanochatSample[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (runId) params.set("expected_run_id", runId);
  const response = await authFetch(`/api/nanochat/samples?${params}`);
  const body = await jsonOrThrow<{ samples: NanochatSample[] }>(response);
  return body.samples;
}

export async function getNanochatBenchmarks(runId?: string | null): Promise<NanochatBenchmark[]> {
  const query = runId ? `?expected_run_id=${encodeURIComponent(runId)}` : "";
  const response = await authFetch(`/api/nanochat/benchmarks${query}`);
  const body = await jsonOrThrow<{ benchmarks: NanochatBenchmark[] }>(response);
  return body.benchmarks;
}

/**
 * Console output for a run, as an incremental feed.
 *
 * `since` is the last sequence number already drawn. Pass 0 to open a console
 * mid-run, which returns the tail — the right thing for a view that just mounted
 * — and the last cursor afterwards to pick up where it left off.
 */
export async function fetchNanochatLogs(
  options: { runId?: string | null; since?: number; limit?: number } = {},
): Promise<{ lines: NanochatLogLine[]; seq: number }> {
  const { runId, since = 0, limit = 500 } = options;
  const params = new URLSearchParams({ limit: String(limit), since: String(since) });
  if (runId) params.set("expected_run_id", runId);
  const response = await authFetch(`/api/nanochat/log?${params}`);
  const body = await jsonOrThrow<{ lines: NanochatLogLine[]; seq: number }>(response);
  return { lines: body.lines ?? [], seq: body.seq ?? since };
}

// --- progress stream ---

export type NanochatStreamEventName = "progress" | "heartbeat" | "complete" | "error";

export type NanochatStreamEvent = {
  event: NanochatStreamEventName;
  id: number | null;
  payload: Record<string, unknown>;
};

const STREAM_EVENT_NAMES: ReadonlySet<string> = new Set([
  "progress",
  "heartbeat",
  "complete",
  "error",
]);

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Coerce a field to a number, or null. `Infinity` is not JSON so it cannot appear,
 *  but a field the backend omits is common, and `undefined` in a chart is worse
 *  than an explicit gap. */
function nullableNumber(value: unknown): number | null {
  return isFiniteNumber(value) ? value : null;
}

function parseStreamPayload(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const payload = value as Record<string, unknown>;
  // run_id and step are the minimum needed to place a frame: without them a frame
  // cannot be matched to a run or ordered against the others.
  if (typeof payload.run_id !== "string" || !isFiniteNumber(payload.step)) return null;
  return {
    ...payload,
    step: payload.step,
    total_steps: nullableNumber(payload.total_steps) ?? 0,
    progress_percent: nullableNumber(payload.progress_percent) ?? 0,
    loss: nullableNumber(payload.loss),
    val_bpb: nullableNumber(payload.val_bpb),
    chatcore: nullableNumber(payload.chatcore),
    reward: nullableNumber(payload.reward),
    tok_per_sec: nullableNumber(payload.tok_per_sec),
    mfu: nullableNumber(payload.mfu),
    eta_seconds: nullableNumber(payload.eta_seconds),
    elapsed_seconds: nullableNumber(payload.elapsed_seconds) ?? 0,
    peak_memory_bytes: nullableNumber(payload.peak_memory_bytes),
    num_params: nullableNumber(payload.num_params),
    logs: parseLogLines(payload.logs),
  };
}

/**
 * Coerce the frame's console lines.
 *
 * A malformed entry is dropped rather than passed through: `seq` is the cursor
 * the whole feed depends on, and one line with a non-numeric `seq` would either
 * stall the console on a number that never arrives or let a duplicate through.
 */
function parseLogLines(value: unknown): NanochatLogLine[] {
  if (!Array.isArray(value)) return [];
  const lines: NanochatLogLine[] = [];
  for (const entry of value) {
    if (typeof entry !== "object" || entry === null) continue;
    const candidate = entry as Record<string, unknown>;
    if (!isFiniteNumber(candidate.seq) || typeof candidate.line !== "string") continue;
    lines.push({
      seq: candidate.seq,
      stream: candidate.stream === "stderr" ? "stderr" : "stdout",
      stage: typeof candidate.stage === "string" ? candidate.stage : null,
      line: candidate.line,
      ts: isFiniteNumber(candidate.ts) ? candidate.ts : 0,
    });
  }
  return lines;
}

type StreamFields = { eventName: NanochatStreamEventName; id: number | null; dataLines: string[] };

function applyStreamLine(fields: StreamFields, line: string): void {
  if (!line) return;
  if (line.startsWith("event:")) {
    const value = line.slice(6).trim();
    if (STREAM_EVENT_NAMES.has(value)) fields.eventName = value as NanochatStreamEventName;
    return;
  }
  if (line.startsWith("id:")) {
    const value = Number(line.slice(3).trim());
    fields.id = Number.isFinite(value) ? value : null;
    return;
  }
  if (line.startsWith("data:")) {
    // trimStart, not trim: JSON leading whitespace is insignificant, but a
    // trailing space inside a string value would survive a full trim.
    fields.dataLines.push(line.slice(5).trimStart());
  }
}

function parseStreamEvent(rawEvent: string): NanochatStreamEvent | null {
  const fields: StreamFields = { eventName: "progress", id: null, dataLines: [] };
  for (const line of rawEvent.split(/\r?\n/)) applyStreamLine(fields, line);
  if (fields.dataLines.length === 0) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(fields.dataLines.join("\n"));
  } catch {
    return null;
  }
  const payload = parseStreamPayload(parsed);
  if (!payload) return null;
  return { event: fields.eventName, payload, id: fields.id };
}

/**
 * Open the progress stream and call `onEvent` for each frame.
 *
 * Mirrors the Train tab's stream: POST (a quick tunnel holds a streamed GET open
 * until it closes), `Last-Event-ID` resume, and the shared frame splitter. A frame
 * that fails to parse is dropped rather than thrown, because one bad frame must
 * not end a stream that is otherwise delivering progress.
 */
export async function streamNanochatProgress(options: {
  runId: string | null;
  signal: AbortSignal;
  onEvent: (event: NanochatStreamEvent) => void;
  onOpen?: () => void;
}): Promise<void> {
  const params = new URLSearchParams();
  if (options.runId) params.set("expected_run_id", options.runId);

  const response = await openStreamResponse(
    authFetch,
    `/api/nanochat/progress?${params}`,
    { signal: options.signal },
  );
  if (!response.ok || !response.body) {
    throw new Error(await readFastApiError(response));
  }

  options.onOpen?.();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (!options.signal.aborted) {
      const { value, done } = await reader.read();
      if (done || options.signal.aborted) break;
      buffer += decoder.decode(value, { stream: true });
      let frame = takeSseFrame(buffer);
      while (frame && !options.signal.aborted) {
        const rawEvent = frame.event;
        buffer = frame.remainder;
        // The server sends a bare "retry:" preamble to set the reconnect delay.
        if (!rawEvent.startsWith("retry:")) {
          const event = parseStreamEvent(rawEvent);
          if (event) options.onEvent(event);
        }
        frame = takeSseFrame(buffer);
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

// --- chat ---

export type NanochatChatModel = {
  model_id: string;
  name: string;
  backend: string;
  source: string;
  model_tag: string;
  step: number;
  bytes: number;
  modified: number;
  num_params: number | null;
  val_bpb: number | null;
  max_seq_len: number | null;
  loadable: boolean;
  reason: string | null;
};

export async function getNanochatChatModels(): Promise<{
  models: NanochatChatModel[];
  resident: { backend: string; model: string } | null;
}> {
  const response = await authFetch("/api/nanochat/chat/models");
  return jsonOrThrow(response);
}

export async function loadNanochatChatModel(modelId: string): Promise<void> {
  const response = await authFetch(
    `/api/nanochat/chat/load?model_id=${encodeURIComponent(modelId)}`,
    { method: "POST" },
  );
  await jsonOrThrow(response);
}

export async function unloadNanochatChatModel(): Promise<void> {
  const response = await authFetch("/api/nanochat/chat/unload", { method: "POST" });
  await jsonOrThrow(response);
}
