// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Owns the live run: hydration, the progress stream, and the fallbacks for when
 * the stream cannot deliver.
 *
 * The stream is the primary channel, but not the only one. A browser tab that was
 * backgrounded, a proxy that dropped the connection, or a laptop that slept will
 * all leave the stream silently dead while the run keeps going. So a poll backs it
 * up, and a stalled stream is a reason to reconnect rather than a reason to show a
 * frozen number.
 */
import { useCallback, useEffect, useRef } from "react";

import {
  fetchNanochatLogs,
  getNanochatBenchmarks,
  getNanochatMetrics,
  getNanochatSamples,
  getNanochatStatus,
  isAbortError,
  streamNanochatProgress,
} from "../api/nanochat-api";
import {
  isNanochatRunActive,
  useNanochatRuntimeStore,
  type ProgressEvent,
} from "../stores/nanochat-runtime-store";

// A run emits a metric event per step, which on a fast small model can be several
// a second, so metrics and samples are polled slower than progress.
const METRICS_POLL_MS = 5_000;
const SAMPLES_POLL_MS = 4_000;
const BENCHMARKS_POLL_MS = 10_000;
// The console normally rides the progress stream, which ticks every second. This
// poll only matters when that stream is dead, so it is slower than the stream is
// fast rather than competing with it.
const LOGS_POLL_MS = 4_000;
// A stream that produces nothing at all for this long is assumed dead. The server
// heartbeats well inside this, so silence means the connection is gone rather than
// the run being quiet.
const STREAM_STALL_MS = 45_000;
const STREAM_RECONNECT_DELAY_MS = 2_500;

function progressFromPayload(payload: Record<string, unknown>): ProgressEvent {
  return {
    run_id: String(payload.run_id ?? ""),
    status: (payload.status as ProgressEvent["status"]) ?? "running",
    phase: String(payload.phase ?? "idle"),
    current_stage: (payload.current_stage as string | null) ?? null,
    message: String(payload.message ?? ""),
    step: Number(payload.step ?? 0),
    total_steps: Number(payload.total_steps ?? 0),
    progress_percent: Number(payload.progress_percent ?? 0),
    loss: (payload.loss as number | null) ?? null,
    val_bpb: (payload.val_bpb as number | null) ?? null,
    chatcore: (payload.chatcore as number | null) ?? null,
    reward: (payload.reward as number | null) ?? null,
    tok_per_sec: (payload.tok_per_sec as number | null) ?? null,
    mfu: (payload.mfu as number | null) ?? null,
    eta_seconds: (payload.eta_seconds as number | null) ?? null,
    elapsed_seconds: Number(payload.elapsed_seconds ?? 0),
    peak_memory_bytes: (payload.peak_memory_bytes as number | null) ?? null,
    num_params: (payload.num_params as number | null) ?? null,
    warnings: Array.isArray(payload.warnings) ? (payload.warnings as string[]) : [],
    stages: Array.isArray(payload.stages)
      ? (payload.stages as ProgressEvent["stages"])
      : [],
    logs: Array.isArray(payload.logs) ? (payload.logs as ProgressEvent["logs"]) : [],
  };
}

export function useNanochatRuntimeLifecycle(): void {
  const runId = useNanochatRuntimeStore((state) => state.runId);
  const active = useNanochatRuntimeStore(isNanochatRunActive);

  // Read through the store inside callbacks rather than closing over the values
  // above, so a long-lived stream does not hold a stale run id.
  const lastFrameAt = useRef<number>(0);
  const streamAbort = useRef<AbortController | null>(null);

  const refreshMetrics = useCallback(async (id: string | null) => {
    if (!id) return;
    try {
      const series = await getNanochatMetrics(id);
      useNanochatRuntimeStore.getState().applyMetrics(
        Object.fromEntries(
          Object.entries(series).map(([key, points]) => [
            key,
            points.map((point) => ({
              step: point.step,
              value: point.value,
              totalSteps: point.total_steps,
            })),
          ]),
        ),
      );
    } catch {
      // A failed poll is not worth surfacing; the next one usually succeeds.
    }
  }, []);

  const refreshSamples = useCallback(async (id: string | null) => {
    if (!id) return;
    try {
      useNanochatRuntimeStore.getState().applySamples(await getNanochatSamples(150, id));
    } catch {
      /* see refreshMetrics */
    }
  }, []);

  const refreshBenchmarks = useCallback(async (id: string | null) => {
    if (!id) return;
    try {
      useNanochatRuntimeStore.getState().applyBenchmarks(await getNanochatBenchmarks(id));
    } catch {
      /* see refreshMetrics */
    }
  }, []);

  /** Pull console output the stream has not delivered yet. */
  const refreshLogs = useCallback(async (id: string | null) => {
    if (!id) return;
    try {
      const since = useNanochatRuntimeStore.getState().logSeq;
      const { lines } = await fetchNanochatLogs({ runId: id, since });
      useNanochatRuntimeStore.getState().appendLogs(lines);
    } catch {
      /* see refreshMetrics */
    }
  }, []);

  // --- hydration: is a run already going when this tab opens? ---
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await getNanochatStatus();
        if (cancelled) return;
        // A run id is absent when nothing has ever run, which is not an error.
        if (status.run_id) {
          useNanochatRuntimeStore.getState().hydrate(status);
          // The console is worth loading whether or not the run is still going:
          // opening a finished run is exactly when its log is most wanted, and
          // there is no stream to carry it.
          void refreshLogs(status.run_id);
          if (status.status === "running") {
            void refreshMetrics(status.run_id);
            void refreshSamples(status.run_id);
            void refreshBenchmarks(status.run_id);
          }
        } else {
          useNanochatRuntimeStore.getState().reset();
        }
      } catch {
        if (!cancelled) useNanochatRuntimeStore.getState().reset();
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshBenchmarks, refreshLogs, refreshMetrics, refreshSamples]);

  // --- the progress stream, with reconnect ---
  useEffect(() => {
    if (!active || !runId) return;

    let disposed = false;
    const controller = new AbortController();
    streamAbort.current = controller;

    const run = async () => {
      while (!disposed && !controller.signal.aborted) {
        try {
          await streamNanochatProgress({
            runId,
            signal: controller.signal,
            onOpen: () => {
              lastFrameAt.current = Date.now();
            },
            onEvent: (event) => {
              lastFrameAt.current = Date.now();
              const store = useNanochatRuntimeStore.getState();
              if (event.event === "error") {
                // A stream-level error ends the stream server-side; reconnecting
                // is the right response, and the run itself is unaffected.
                return;
              }
              if (event.event === "complete") {
                const payload = progressFromPayload(event.payload);
                store.applyProgress({ ...payload, status: payload.status ?? "completed" });
                // Pull the final artefacts: the last metrics, samples and console
                // lines may have been emitted after the last progress frame, and
                // nothing will come along to carry them now the stream is closing.
                void refreshMetrics(runId);
                void refreshSamples(runId);
                void refreshBenchmarks(runId);
                void refreshLogs(runId);
                return;
              }
              if (event.event === "heartbeat") return;
              store.applyProgress(progressFromPayload(event.payload));
            },
          });
        } catch (error) {
          if (disposed || controller.signal.aborted || isAbortError(error)) return;
        }
        if (disposed || controller.signal.aborted) return;
        // The stream ended without a terminal frame. Reconnect, unless the stall
        // watchdog is about to handle it.
        await new Promise((resolve) => setTimeout(resolve, STREAM_RECONNECT_DELAY_MS));
      }
    };

    void run();

    // Stall watchdog. The server heartbeats every few seconds, so a gap this long
    // means the connection died rather than the run being quiet.
    const watchdog = setInterval(() => {
      if (lastFrameAt.current && Date.now() - lastFrameAt.current > STREAM_STALL_MS) {
        // The poll below is the real safety net; forcing a reconnect here would
        // race it. Just note the gap so the UI can show "reconnecting".
        useNanochatRuntimeStore.setState({ message: "Reconnecting to the run…" });
      }
    }, 10_000);

    return () => {
      disposed = true;
      clearInterval(watchdog);
      controller.abort();
      streamAbort.current = null;
    };
  }, [active, runId, refreshBenchmarks, refreshLogs, refreshMetrics, refreshSamples]);

  // --- polling backstop ---
  useEffect(() => {
    if (!active || !runId) return;
    const metrics = setInterval(() => void refreshMetrics(runId), METRICS_POLL_MS);
    const samples = setInterval(() => void refreshSamples(runId), SAMPLES_POLL_MS);
    const benchmarks = setInterval(() => void refreshBenchmarks(runId), BENCHMARKS_POLL_MS);
    // The stream already carries console lines; this is what keeps the console
    // alive when it does not, which is the same reason every other poll is here.
    const logs = setInterval(() => void refreshLogs(runId), LOGS_POLL_MS);
    return () => {
      clearInterval(metrics);
      clearInterval(samples);
      clearInterval(benchmarks);
      clearInterval(logs);
    };
  }, [active, runId, refreshBenchmarks, refreshLogs, refreshMetrics, refreshSamples]);
}
