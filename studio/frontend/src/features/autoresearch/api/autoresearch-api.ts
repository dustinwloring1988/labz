// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Every call the autoresearch tab makes, and the SSE reader behind it.
 *
 * All HTTP goes through `authFetch`, and the whole feature converts camelCase to
 * snake_case in exactly one place: `buildAutoresearchStartPayload`. The store
 * then converts back on the way in, so the network layer stays a faithful mirror
 * of the backend and a field renamed on the wire is a one-line change here.
 */
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { openStreamResponse } from "@/lib/open-stream-response";
import { takeSseFrame } from "@/lib/sse-framing";

import type {
  AutoresearchAgents,
  AutoresearchChatModels,
  AutoresearchCheckpointInfo,
  AutoresearchEnvironmentStatus,
  AutoresearchExperimentInfo,
  AutoresearchLogLine,
  AutoresearchMetricPoint,
  AutoresearchResultRow,
  AutoresearchStartResponse,
  AutoresearchStatus,
  ProgressEvent,
} from "../types/index.ts";
import type { AutoresearchConfigState } from "../stores/autoresearch-config-policy.ts";
import { parseAgentArgs } from "../stores/autoresearch-config-policy.ts";

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  throw new Error(await readFastApiError(response));
}

/** A deliberate cancellation, as opposed to a stream that broke. */
export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

// --- environment ---

export function getAutoresearchEnvironment(
  signal?: AbortSignal,
): Promise<AutoresearchEnvironmentStatus> {
  return authFetch("/api/autoresearch/environment", { signal }).then((r) =>
    jsonOrThrow<AutoresearchEnvironmentStatus>(r),
  );
}

export function installAutoresearchEnvironment(
  options?: { prepareData?: boolean },
): Promise<AutoresearchEnvironmentStatus> {
  const params = new URLSearchParams();
  params.set("prepare_data", String(options?.prepareData ?? true));
  return authFetch(`/api/autoresearch/environment/install?${params}`, {
    method: "POST",
  }).then((r) => jsonOrThrow<AutoresearchEnvironmentStatus>(r));
}

export function prepareAutoresearchData(): Promise<AutoresearchEnvironmentStatus> {
  return authFetch("/api/autoresearch/environment/prepare-data", {
    method: "POST",
  }).then((r) => jsonOrThrow<AutoresearchEnvironmentStatus>(r));
}

export function getAutoresearchAgents(
  signal?: AbortSignal,
): Promise<AutoresearchAgents> {
  return authFetch("/api/autoresearch/agents", { signal }).then((r) =>
    jsonOrThrow<AutoresearchAgents>(r),
  );
}

// --- run lifecycle ---

/** The whole form in one place, so a field rename is a one-line change. */
export function buildAutoresearchStartPayload(config: AutoresearchConfigState) {
  return {
    num_experiments: config.numExperiments,
    agent: config.agent,
    agent_args: parseAgentArgs(config.agentArgs),
    skip_permissions: config.skipPermissions,
    agent_timeout_seconds: Math.round(config.agentTimeoutMinutes * 60),
    max_consecutive_failures: config.maxConsecutiveFailures,
    prepare_data: config.prepareData,
    focus: config.focus,
    report_after_finish: config.reportAfterFinish,
    report_model_id: config.reportModelId,
    report_source: config.reportSource,
    report_agent_key: config.reportAgentKey,
  };
}

export function startAutoresearchRun(
  config: AutoresearchConfigState,
): Promise<AutoresearchStartResponse> {
  return authFetch("/api/autoresearch/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildAutoresearchStartPayload(config)),
  }).then((r) => jsonOrThrow<AutoresearchStartResponse>(r));
}

export function stopAutoresearchRun(): Promise<{ status: string }> {
  return authFetch("/api/autoresearch/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ save: true }),
  }).then((r) => jsonOrThrow<{ status: string }>(r));
}

export function getAutoresearchStatus(
  expectedRunId?: string | null,
  signal?: AbortSignal,
): Promise<AutoresearchStatus> {
  const params = new URLSearchParams();
  if (expectedRunId) params.set("expected_run_id", expectedRunId);
  const query = params.toString();
  return authFetch(
    `/api/autoresearch/status${query ? `?${query}` : ""}`,
    { signal },
  ).then((r) => jsonOrThrow<AutoresearchStatus>(r));
}

export function fetchAutoresearchLogs(options: {
  runId?: string | null;
  since?: number;
  limit?: number;
  signal?: AbortSignal;
}): Promise<{ run_id: string; logs: AutoresearchLogLine[]; next_seq: number }> {
  const params = new URLSearchParams();
  if (options.runId) params.set("expected_run_id", options.runId);
  params.set("since", String(options.since ?? 0));
  params.set("limit", String(options.limit ?? 400));
  return authFetch(`/api/autoresearch/log?${params}`, { signal: options.signal }).then(
    (r) =>
      jsonOrThrow<{
        run_id: string;
        logs: AutoresearchLogLine[];
        next_seq: number;
      }>(r),
  );
}

// --- results and metrics ---

export type AutoresearchResultsPayload = {
  rows: AutoresearchResultRow[];
  exists: boolean;
  total: number;
  kept: number;
  discarded: number;
  crashed: number;
  best_val_bpb: number | null;
  best_index: number | null;
  best_commit: string | null;
  baseline_val_bpb: number | null;
};

export function getAutoresearchResults(
  signal?: AbortSignal,
): Promise<AutoresearchResultsPayload> {
  return authFetch("/api/autoresearch/results", { signal }).then((r) =>
    jsonOrThrow<AutoresearchResultsPayload>(r),
  );
}

export function getAutoresearchMetrics(signal?: AbortSignal): Promise<{
  series: AutoresearchMetricPoint[];
  baseline_val_bpb: number | null;
  best_val_bpb: number | null;
}> {
  return authFetch("/api/autoresearch/metrics", { signal }).then((r) =>
    jsonOrThrow<{
      series: AutoresearchMetricPoint[];
      baseline_val_bpb: number | null;
      best_val_bpb: number | null;
    }>(r),
  );
}

// --- checkpoints and chat ---

export function getAutoresearchCheckpoints(
  signal?: AbortSignal,
): Promise<AutoresearchCheckpointInfo[]> {
  return authFetch("/api/autoresearch/checkpoints", { signal }).then((r) =>
    jsonOrThrow<{ checkpoints: AutoresearchCheckpointInfo[] }>(r).then(
      (payload) => payload.checkpoints,
    ),
  );
}

export function getAutoresearchChatModels(signal?: AbortSignal): Promise<AutoresearchChatModels> {
  return authFetch("/api/autoresearch/chat/models", { signal }).then((r) =>
    jsonOrThrow<AutoresearchChatModels>(r),
  );
}

export function loadAutoresearchChatModel(modelId: string): Promise<void> {
  return authFetch(
    `/api/autoresearch/chat/load?model_id=${encodeURIComponent(modelId)}`,
    { method: "POST" },
  ).then((r) => jsonOrThrow<{ status: string }>(r).then(() => undefined));
}

export function unloadAutoresearchChatModel(): Promise<void> {
  return authFetch("/api/autoresearch/chat/unload", { method: "POST" }).then((r) =>
    jsonOrThrow<{ status: string }>(r).then(() => undefined),
  );
}

// --- report ---

export function getAutoresearchReports(signal?: AbortSignal): Promise<{
  reports: { run_id: string; path: string; bytes: number; modified: number }[];
}> {
  return authFetch("/api/autoresearch/reports", { signal }).then((r) =>
    jsonOrThrow<{
      reports: { run_id: string; path: string; bytes: number; modified: number }[];
    }>(r),
  );
}

export function loadAutoresearchReport(
  runId = "latest",
  signal?: AbortSignal,
): Promise<{ run_id: string; text: string }> {
  return authFetch(
    `/api/autoresearch/report?run_id=${encodeURIComponent(runId)}`,
    { signal },
  ).then((r) => jsonOrThrow<{ run_id: string; text: string }>(r));
}

// -----------------------------------------------------------------------------
// Progress stream
// -----------------------------------------------------------------------------

export type AutoresearchStreamEventName = "progress" | "heartbeat" | "complete" | "error" | "warning";

const STREAM_EVENT_NAMES: ReadonlySet<string> = new Set([
  "progress",
  "heartbeat",
  "complete",
  "error",
  "warning",
]);

/**
 * One arm per event name rather than one arm per payload shape.
 *
 * The consumer narrows on `event`, and a union whose arms each cover several
 * names cannot be narrowed that way: after ruling out "error" and "heartbeat"
 * the compiler still has to allow the "warning" arm, so a caller that has done
 * the right thing is still told its payload is the wrong type.
 */
export type AutoresearchStreamEvent =
  | { event: "progress"; payload: ProgressEvent }
  | { event: "complete"; payload: ProgressEvent }
  | { event: "heartbeat"; payload: { message?: string } }
  | { event: "warning"; payload: { message?: string } }
  | { event: "error"; payload: { error?: string; reason?: string } };

type StreamFields = { event: string; data: string };

/**
 * A frame without a run id and a numeric experiment count cannot be placed in
 * time, so it is dropped rather than thrown. A dropped frame costs a render; a
 * thrown one kills the stream and leaves the tab frozen on a stale progress bar.
 */
function parseStreamPayload(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object") return null;
  const payload = value as Record<string, unknown>;
  const runId = payload.run_id;
  if (typeof runId !== "string" || runId.length === 0) return null;
  const experiment = payload.experiment;
  if (experiment !== undefined && typeof experiment !== "number") return null;
  return payload;
}

function isNumberOrNull(value: unknown): value is number | null {
  return value === null || typeof value === "number";
}

function progressFromPayload(payload: Record<string, unknown>): ProgressEvent {
  const warnings = Array.isArray(payload.warnings)
    ? (payload.warnings as unknown[]).filter((w): w is string => typeof w === "string")
    : [];
  const experiments = Array.isArray(payload.experiments)
    ? (payload.experiments as AutoresearchExperimentInfo[])
    : [];
  const logs = Array.isArray(payload.logs)
    ? (payload.logs as AutoresearchLogLine[])
    : undefined;

  return {
    run_id: String(payload.run_id ?? ""),
    status: (payload.status as ProgressEvent["status"]) ?? "running",
    phase: (payload.phase as ProgressEvent["phase"]) ?? "experiment",
    message: typeof payload.message === "string" ? payload.message : "",
    error: typeof payload.error === "string" ? payload.error : null,
    started_at: isNumberOrNull(payload.started_at) ? payload.started_at : null,
    ended_at: isNumberOrNull(payload.ended_at) ? payload.ended_at : null,
    duration_seconds:
      typeof payload.duration_seconds === "number" ? payload.duration_seconds : 0,
    experiment: typeof payload.experiment === "number" ? payload.experiment : 0,
    total_experiments:
      typeof payload.total_experiments === "number" ? payload.total_experiments : 0,
    progress_percent:
      typeof payload.progress_percent === "number" ? payload.progress_percent : 0,
    step: typeof payload.step === "number" ? payload.step : 0,
    loss: isNumberOrNull(payload.loss) ? payload.loss : null,
    tok_per_sec: isNumberOrNull(payload.tok_per_sec) ? payload.tok_per_sec : null,
    mfu: isNumberOrNull(payload.mfu) ? payload.mfu : null,
    peak_memory_gb: isNumberOrNull(payload.peak_memory_gb) ? payload.peak_memory_gb : null,
    elapsed_seconds:
      typeof payload.elapsed_seconds === "number" ? payload.elapsed_seconds : 0,
    eta_seconds: isNumberOrNull(payload.eta_seconds) ? payload.eta_seconds : null,
    best_val_bpb: isNumberOrNull(payload.best_val_bpb) ? payload.best_val_bpb : null,
    baseline_val_bpb: isNumberOrNull(payload.baseline_val_bpb)
      ? payload.baseline_val_bpb
      : null,
    kept: typeof payload.kept === "number" ? payload.kept : 0,
    discarded: typeof payload.discarded === "number" ? payload.discarded : 0,
    crashed: typeof payload.crashed === "number" ? payload.crashed : 0,
    experiments,
    warnings,
    resolved_config:
      payload.resolved_config && typeof payload.resolved_config === "object"
        ? (payload.resolved_config as Record<string, unknown>)
        : {},
    git_branch: typeof payload.git_branch === "string" ? payload.git_branch : "",
    git_head: typeof payload.git_head === "string" ? payload.git_head : "",
    stop_requested: payload.stop_requested === true,
    estimated_seconds:
      typeof payload.estimated_seconds === "number" ? payload.estimated_seconds : 0,
    report_text: typeof payload.report_text === "string" ? payload.report_text : "",
    report_model: typeof payload.report_model === "string" ? payload.report_model : "",
    report_saved_to:
      typeof payload.report_saved_to === "string" ? payload.report_saved_to : null,
    report_error: typeof payload.report_error === "string" ? payload.report_error : null,
    report_running: payload.report_running === true,
    ...(logs ? { logs } : {}),
  };
}

function parseStreamEvent(raw: string): AutoresearchStreamEvent | null {
  const newline = raw.indexOf("\n");
  const name = (newline < 0 ? raw : raw.slice(0, newline)).replace(/^event:\s*/, "").trim();
  const body = newline < 0 ? "" : raw.slice(newline + 1);
  if (!STREAM_EVENT_NAMES.has(name)) return null;

  if (name === "heartbeat" || name === "warning") {
    return {
      event: name,
      payload: { message: body.replace(/^data:\s*/, "").trim() || undefined },
    };
  }
  if (name === "error") {
    try {
      const parsed = JSON.parse(body.replace(/^data:\s*/, ""));
      return {
        event: "error",
        payload: {
          error: typeof parsed.error === "string" ? parsed.error : undefined,
          reason: typeof parsed.reason === "string" ? parsed.reason : undefined,
        },
      };
    } catch {
      return { event: "error", payload: { error: body.replace(/^data:\s*/, "").trim() } };
    }
  }

  const payload = parseStreamPayload(safeJson(body));
  if (!payload) return null;
  return { event: name as "progress" | "complete", payload: progressFromPayload(payload) };
}
function safeJson(body: string): unknown {
  const trimmed = body.replace(/^data:\s*/, "").trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function applyStreamLine(fields: StreamFields, line: string): void {
  if (line.startsWith("event:")) {
    fields.event = line.slice(6).trim();
  } else if (line.startsWith("data:")) {
    fields.data = fields.data ? `${fields.data}\n${line.slice(5).trim()}` : line.slice(5).trim();
  }
}

export async function streamAutoresearchProgress(options: {
  runId: string | null;
  signal: AbortSignal;
  onEvent: (event: AutoresearchStreamEvent) => void;
  onOpen?: () => void;
}): Promise<void> {
  const params = new URLSearchParams();
  if (options.runId) params.set("expected_run_id", options.runId);
  const query = params.toString();
  const response = await openStreamResponse(
    authFetch,
    `/api/autoresearch/progress${query ? `?${query}` : ""}`,
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
        const raw = frame.event;
        buffer = frame.remainder;
        // The bare "retry:" preamble the server opens with carries no event.
        if (!raw.startsWith("retry:") && raw.trim().length > 0) {
          const fields: StreamFields = { event: "progress", data: "" };
          for (const line of raw.split("\n")) applyStreamLine(fields, line);
          const event = parseStreamEvent(`${fields.event}\ndata: ${fields.data}`);
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

// -----------------------------------------------------------------------------
// Chat stream
// -----------------------------------------------------------------------------

export type AutoresearchChatToken = { text: string };

/**
 * Stream one continuation from a checkpoint.
 *
 * The backend yields deltas, so this accumulates and reports cumulative text,
 * which is the convention every other chat surface in the app uses and what a
 * message store expects to append to.
 */
export async function streamAutoresearchChat(options: {
  modelId: string;
  prompt: string;
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  signal: AbortSignal;
  onText: (cumulative: string) => void;
}): Promise<void> {
  const params = new URLSearchParams();
  params.set("model_id", options.modelId);
  params.set("prompt", options.prompt);
  params.set("max_new_tokens", String(options.maxNewTokens ?? 256));
  params.set("temperature", String(options.temperature ?? 0.8));
  params.set("top_k", String(options.topK ?? 50));

  const response = await openStreamResponse(
    authFetch,
    `/api/autoresearch/chat/completions?${params.toString()}`,
    { signal: options.signal },
  );
  if (!response.ok || !response.body) {
    throw new Error(await readFastApiError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";
  try {
    while (!options.signal.aborted) {
      const { value, done } = await reader.read();
      if (done || options.signal.aborted) break;
      buffer += decoder.decode(value, { stream: true });
      let frame = takeSseFrame(buffer);
      while (frame && !options.signal.aborted) {
        const raw = frame.event;
        buffer = frame.remainder;
        if (raw.trim().length > 0) {
          const lines = raw.split("\n");
          const name = (lines[0] ?? "").replace(/^event:\s*/, "").trim();
          const body = lines
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim())
            .join("");
          if (name === "token" && body) {
            const parsed = safeJson(body) as AutoresearchChatToken | null;
            if (parsed && typeof parsed.text === "string") {
              accumulated += parsed.text;
              options.onText(accumulated);
            }
          } else if (name === "error" && body) {
            const parsed = safeJson(body) as { error?: string } | null;
            throw new Error(parsed?.error || "generation failed");
          }
        }
        frame = takeSseFrame(buffer);
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

// -----------------------------------------------------------------------------
// Report stream
// -----------------------------------------------------------------------------

export type AutoresearchReportEvent =
  | { event: "token"; payload: { text: string } }
  | { event: "warning"; payload: { warning: string } }
  | { event: "done"; payload: { saved_to: string | null; run_id: string } }
  | { event: "error"; payload: { error: string } };

/** Stream a report as markdown, accumulating into one growing document. */
export async function streamAutoresearchReport(options: {
  modelId: string;
  source: "catalog" | "autoresearch" | "agent";
  agentKey?: string;
  agentArgs?: string[];
  maxTokens?: number;
  temperature?: number;
  save?: boolean;
  runId?: string;
  signal: AbortSignal;
  onText: (cumulative: string) => void;
  onWarning?: (message: string) => void;
  onDone?: (payload: { saved_to: string | null; run_id: string }) => void;
}): Promise<void> {
  const response = await openStreamResponse(authFetch, "/api/autoresearch/report", {
    method: "POST",
    signal: options.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_id: options.modelId,
      source: options.source,
      agent_key: options.agentKey ?? "",
      agent_args: options.agentArgs ?? [],
      max_tokens: options.maxTokens ?? 2000,
      temperature: options.temperature ?? 0.7,
      save: options.save ?? true,
      run_id: options.runId ?? "latest",
    }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  if (!response.body) {
    throw new Error("The report stream did not open.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";
  try {
    while (!options.signal.aborted) {
      const { value, done } = await reader.read();
      if (done || options.signal.aborted) break;
      buffer += decoder.decode(value, { stream: true });
      let frame = takeSseFrame(buffer);
      while (frame && !options.signal.aborted) {
        const raw = frame.event;
        buffer = frame.remainder;
        if (raw.trim().length > 0 && !raw.startsWith(":")) {
          const lines = raw.split("\n");
          const name = (lines[0] ?? "").replace(/^event:\s*/, "").trim();
          const body = lines
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim())
            .join("");
          const parsed = body ? safeJson(body) : null;
          if (name === "token" && parsed && typeof (parsed as { text?: string }).text === "string") {
            accumulated += (parsed as { text: string }).text;
            options.onText(accumulated);
          } else if (name === "warning" && parsed) {
            options.onWarning?.(String((parsed as { warning?: string }).warning ?? ""));
          } else if (name === "done" && parsed) {
            options.onDone?.(
              parsed as unknown as { saved_to: string | null; run_id: string },
            );
          } else if (name === "error" && parsed) {
            throw new Error(String((parsed as { error?: string }).error ?? "report failed"));
          }
        }
        frame = takeSseFrame(buffer);
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
