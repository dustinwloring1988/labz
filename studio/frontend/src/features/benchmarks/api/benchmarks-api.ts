// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { openStreamResponse } from "@/lib/open-stream-response";
import { takeSseFrame } from "@/lib/sse-framing";

import type {
  BenchmarkCatalogue,
  BenchmarkHistory,
  BenchmarkLeaderboard,
  BenchmarkLogLine,
  BenchmarkModelList,
  BenchmarkModelCandidate,
  BenchmarkRunStatus,
  BenchmarkScore,
  BenchmarkScoringMode,
  BenchmarkStartResponse,
  BenchmarkStatus,
  LeaderboardRow,
} from "../types";

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  throw new Error(await readFastApiError(response));
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** A number, or null. `undefined` in a chart is worse than an explicit gap. */
function nullableNumber(value: unknown): number | null {
  return isFiniteNumber(value) ? value : null;
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

// --- catalogue and models ---

export async function getBenchmarkCatalogue(): Promise<BenchmarkCatalogue> {
  const response = await authFetch("/api/benchmarks/benchmarks");
  return jsonOrThrow<BenchmarkCatalogue>(response);
}

export async function getBenchmarkModels(): Promise<BenchmarkModelList> {
  const response = await authFetch("/api/benchmarks/models");
  return jsonOrThrow<BenchmarkModelList>(response);
}

/**
 * Set up the nanochat environment that owns the benchmark datasets.
 *
 * Not a benchmarks route: the environment is nanochat's, it is what
 * `core/benchmarks/catalogue.py` reads to decide whether anything can run, and
 * its installer already exists with its own progress reporting. Duplicating it
 * here would mean a second installer racing the first for the same directory.
 */
export async function installNanochat(): Promise<void> {
  const response = await authFetch("/api/nanochat/environment/install", {
    method: "POST",
  });
  await jsonOrThrow(response);
}

// --- run lifecycle ---

/** The start payload, built from the form state. The one camelCase-to-snake_case
 *  conversion in this feature, kept here so a field cannot silently stop being
 *  sent. */
export function buildBenchmarkStartPayload(input: {
  model: BenchmarkModelCandidate;
  benchmarks: string[];
  maxProblems: number;
  batchSize: number;
  maxNewTokens: number;
  maxSeqLength: number;
  loadIn4bit: boolean;
  /** The second opinion, when one is configured. Null runs without a judge. */
  judgeModel?: BenchmarkModelCandidate | null;
}) {
  return {
    model_id: input.model.id,
    model_label: input.model.label,
    model_path: input.model.path,
    format: input.model.format,
    // A LoRA candidate's own path is both what is loaded and what is attached,
    // so it is sent once and the scorer attaches it to the base it finds in the
    // adapter metadata.
    lora_path: input.model.lora ? input.model.path : null,
    load_in_4bit: input.loadIn4bit,
    benchmarks: input.benchmarks,
    max_problems: input.maxProblems,
    batch_size: input.batchSize,
    max_new_tokens: input.maxNewTokens,
    max_seq_length: input.maxSeqLength,
    // Sent as a block of nulls rather than omitted, so the request is explicit
    // about having no judge. The backend treats all three as one setting, and a
    // partially filled judge would be a path without a model.
    judge_model_id: input.judgeModel?.id ?? null,
    judge_model_label: input.judgeModel?.label ?? null,
    judge_model_path: input.judgeModel?.path ?? null,
  };
}

export async function startBenchmarkRun(
  payload: ReturnType<typeof buildBenchmarkStartPayload>,
): Promise<BenchmarkStartResponse> {
  const response = await authFetch("/api/benchmarks/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow<BenchmarkStartResponse>(response);
}

export async function stopBenchmarkRun(): Promise<{
  status: string;
  message: string;
}> {
  const response = await authFetch("/api/benchmarks/stop", { method: "POST" });
  return jsonOrThrow(response);
}

export async function getBenchmarkStatus(
  runId?: string | null,
): Promise<BenchmarkStatus> {
  const query = runId ? `?expected_run_id=${encodeURIComponent(runId)}` : "";
  const response = await authFetch(`/api/benchmarks/status${query}`);
  return jsonOrThrow<BenchmarkStatus>(response);
}

// --- history and leaderboard ---

export async function getBenchmarkHistory(
  limit = 50,
): Promise<BenchmarkHistory> {
  const response = await authFetch(`/api/benchmarks/history?limit=${limit}`);
  return jsonOrThrow<BenchmarkHistory>(response);
}

export async function deleteBenchmarkRun(runId: string): Promise<void> {
  const response = await authFetch(
    `/api/benchmarks/runs/${encodeURIComponent(runId)}`,
    {
      method: "DELETE",
    },
  );
  await jsonOrThrow(response);
}

export async function getBenchmarkLeaderboard(
  options: {
    benchmarks?: string[];
    includeReferences?: boolean;
  } = {},
): Promise<BenchmarkLeaderboard> {
  const params = new URLSearchParams();
  if (options.benchmarks && options.benchmarks.length > 0) {
    params.set("benchmarks", options.benchmarks.join(","));
  }
  if (options.includeReferences === false)
    params.set("include_references", "false");
  const query = params.toString();
  const response = await authFetch(
    `/api/benchmarks/leaderboard${query ? `?${query}` : ""}`,
  );
  return jsonOrThrow<BenchmarkLeaderboard>(response);
}

// --- progress stream ---

export type BenchmarkStreamEventName =
  | "progress"
  | "heartbeat"
  | "complete"
  | "error";

export type BenchmarkStreamEvent = {
  event: BenchmarkStreamEventName;
  id: number | null;
  payload: BenchmarkStatus | { reason: string; phase?: string };
};

const STREAM_EVENT_NAMES: ReadonlySet<string> = new Set([
  "progress",
  "heartbeat",
  "complete",
  "error",
]);

function parseScore(value: unknown): BenchmarkScore | null {
  if (typeof value !== "object" || value === null) return null;
  const row = value as Record<string, unknown>;
  if (typeof row.key !== "string") return null;
  return {
    key: row.key,
    label: typeof row.label === "string" ? row.label : row.key,
    kind: row.kind === "generative" ? "generative" : "categorical",
    status:
      row.status === "complete" ||
      row.status === "running" ||
      row.status === "skipped" ||
      row.status === "failed"
        ? row.status
        : "pending",
    accuracy: nullableNumber(row.accuracy),
    baseline: nullableNumber(row.baseline) ?? 0,
    centered: nullableNumber(row.centered),
    correct: isFiniteNumber(row.correct) ? row.correct : 0,
    total: isFiniteNumber(row.total) ? row.total : 0,
    elapsed_seconds: nullableNumber(row.elapsed_seconds) ?? 0,
    truncated: row.truncated === true,
    scoring_mode: parseScoringMode(row.scoring_mode),
    error: optionalString(row.error),
    exact_accuracy: nullableNumber(row.exact_accuracy),
    judge_accuracy: nullableNumber(row.judge_accuracy),
    judge_judged: isFiniteNumber(row.judge_judged) ? row.judge_judged : 0,
    judge_unreadable: isFiniteNumber(row.judge_unreadable) ? row.judge_unreadable : 0,
    disagreements: isFiniteNumber(row.disagreements) ? row.disagreements : 0,
    judge_raised: isFiniteNumber(row.judge_raised) ? row.judge_raised : 0,
    judge_model: optionalString(row.judge_model),
  };
}

function parseScoringMode(value: unknown): BenchmarkScoringMode | null {
  return value === "logits" ||
    value === "generated" ||
    value === "judged" ||
    value === "mixed"
    ? value
    : null;
}

function parseRunStatus(value: unknown): BenchmarkRunStatus {
  return value === "running" ||
    value === "completed" ||
    value === "error" ||
    value === "stopped"
    ? value
    : "idle";
}

/**
 * Coerce the frame's console lines.
 *
 * A malformed entry is dropped rather than passed through: `seq` is the cursor the
 * whole feed depends on, and one line with a non-numeric `seq` would either stall
 * the console on a number that never arrives or let a duplicate through.
 */
function parseLogLines(value: unknown): BenchmarkLogLine[] {
  if (!Array.isArray(value)) return [];
  const lines: BenchmarkLogLine[] = [];
  for (const entry of value) {
    if (typeof entry !== "object" || entry === null) continue;
    const candidate = entry as Record<string, unknown>;
    if (!isFiniteNumber(candidate.seq) || typeof candidate.line !== "string")
      continue;
    lines.push({
      seq: candidate.seq,
      stream: candidate.stream === "stderr" ? "stderr" : "stdout",
      phase: optionalString(candidate.phase),
      line: candidate.line,
      ts: isFiniteNumber(candidate.ts) ? candidate.ts : 0,
    });
  }
  return lines;
}

function parseStatusPayload(value: unknown): BenchmarkStatus | null {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    return null;
  const row = value as Record<string, unknown>;
  // run_id is the minimum needed to place a frame. A heartbeat carries only a
  // reason, and is handled separately rather than coerced into a status.
  if (typeof row.run_id !== "string" || row.run_id.length === 0) return null;
  return {
    run_id: row.run_id,
    status: parseRunStatus(row.status),
    phase: typeof row.phase === "string" ? row.phase : "idle",
    message: typeof row.message === "string" ? row.message : "",
    error: optionalString(row.error),
    model_id: typeof row.model_id === "string" ? row.model_id : "",
    model_label: typeof row.model_label === "string" ? row.model_label : "",
    format: row.format === "gguf" ? "gguf" : "safetensors",
    lora_path: optionalString(row.lora_path),
    load_in_4bit: row.load_in_4bit === true,
    judge_model_id: optionalString(row.judge_model_id),
    judge_model_label: optionalString(row.judge_model_label),
    requested: Array.isArray(row.requested)
      ? row.requested.filter(
          (entry): entry is string => typeof entry === "string",
        )
      : [],
    current: optionalString(row.current),
    scores: Array.isArray(row.scores)
      ? row.scores
          .map(parseScore)
          .filter((entry): entry is BenchmarkScore => entry !== null)
      : [],
    composite: nullableNumber(row.composite),
    scoring_mode: parseScoringMode(row.scoring_mode),
    progress_percent: nullableNumber(row.progress_percent) ?? 0,
    elapsed_seconds: nullableNumber(row.elapsed_seconds) ?? 0,
    started_at: nullableNumber(row.started_at),
    ended_at: nullableNumber(row.ended_at),
    logs: parseLogLines(row.logs),
  };
}

type StreamFields = {
  eventName: BenchmarkStreamEventName;
  id: number | null;
  dataLines: string[];
};

function applyStreamLine(fields: StreamFields, line: string): void {
  if (!line) return;
  if (line.startsWith("event:")) {
    const value = line.slice(6).trim();
    if (STREAM_EVENT_NAMES.has(value))
      fields.eventName = value as BenchmarkStreamEventName;
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

function parseStreamEvent(rawEvent: string): BenchmarkStreamEvent | null {
  const fields: StreamFields = {
    eventName: "progress",
    id: null,
    dataLines: [],
  };
  for (const line of rawEvent.split(/\r?\n/)) applyStreamLine(fields, line);
  if (fields.dataLines.length === 0) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(fields.dataLines.join("\n"));
  } catch {
    return null;
  }
  const status = parseStatusPayload(parsed);
  if (status)
    return { event: fields.eventName, payload: status, id: fields.id };
  if (typeof parsed === "object" && parsed !== null && "reason" in parsed) {
    const row = parsed as Record<string, unknown>;
    return {
      event: fields.eventName,
      payload: {
        reason: String(row.reason),
        phase: optionalString(row.phase) ?? undefined,
      },
      id: fields.id,
    };
  }
  return null;
}

/**
 * Open the progress stream and call `onEvent` for each frame.
 *
 * Mirrors the Train and nanochat streams: POST (a quick tunnel holds a streamed GET
 * open until it closes), and the shared frame splitter. A frame that fails to parse
 * is dropped rather than thrown, because one bad frame must not end a stream that is
 * otherwise delivering progress.
 */
export async function streamBenchmarkProgress(options: {
  runId: string | null;
  signal: AbortSignal;
  onEvent: (event: BenchmarkStreamEvent) => void;
  onOpen?: () => void;
}): Promise<void> {
  const params = new URLSearchParams();
  if (options.runId) params.set("expected_run_id", options.runId);

  const response = await openStreamResponse(
    authFetch,
    `/api/benchmarks/progress?${params}`,
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

export type { BenchmarkLeaderboard, LeaderboardRow };
