// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Owns the live run: hydration, the progress stream, and the fallback for when
 * the stream cannot deliver.
 *
 * The stream is the primary channel but not the only one. A backgrounded tab, a
 * proxy that dropped the connection, or a laptop that slept will all leave the
 * stream silently dead while the run keeps going, so a poll backs it up and a
 * stalled stream is a reason to reconnect rather than a reason to show a frozen
 * number.
 */
import { useCallback, useEffect, useRef } from "react";

import {
  getBenchmarkStatus,
  isAbortError,
  streamBenchmarkProgress,
} from "../api/benchmarks-api";
import {
  isBenchmarkRunActive,
  useBenchmarkRuntimeStore,
} from "../stores/benchmark-runtime-store";
import type { BenchmarkStatus } from "../types";

// A stream that produces nothing at all for this long is assumed dead. The server
// heartbeats every few seconds, so silence means the connection is gone rather
// than the run being quiet — and a benchmark run has long quiet stretches, since
// loading a model or downloading a dataset says nothing for minutes.
const STREAM_STALL_MS = 45_000;
const STREAM_RECONNECT_DELAY_MS = 2_500;
// The backstop poll. Only matters when the stream is dead, so it is slower than
// the stream is fast rather than competing with it.
const STATUS_POLL_MS = 6_000;

function isStatusPayload(value: unknown): value is BenchmarkStatus {
  return typeof value === "object" && value !== null && "run_id" in value;
}

export function useBenchmarkRuntimeLifecycle(): void {
  const runId = useBenchmarkRuntimeStore((state) => state.runId);
  const active = useBenchmarkRuntimeStore(isBenchmarkRunActive);

  // Read through the store inside callbacks rather than closing over the values
  // above, so a long-lived stream does not hold a stale run id.
  const streamAbort = useRef<AbortController | null>(null);

  const refreshStatus = useCallback(async (id: string | null) => {
    if (!id) return;
    try {
      const status = await getBenchmarkStatus(id);
      useBenchmarkRuntimeStore.getState().applyStatus(status);
    } catch {
      // A failed poll is not worth surfacing; the next one usually succeeds, and
      // the stream is still the primary channel.
    }
  }, []);

  // --- hydration: is a run already going when this tab opens? ---
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await getBenchmarkStatus();
        if (cancelled) return;
        // An empty run id means nothing has ever run, which is not an error.
        if (status.run_id) {
          useBenchmarkRuntimeStore.getState().hydrate(status);
        } else {
          useBenchmarkRuntimeStore.getState().reset();
        }
      } catch {
        if (!cancelled) useBenchmarkRuntimeStore.getState().reset();
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // --- the progress stream, with reconnect ---
  useEffect(() => {
    if (!active || !runId) return;

    let disposed = false;
    const controller = new AbortController();
    streamAbort.current = controller;

    const run = async () => {
      while (!disposed && !controller.signal.aborted) {
        try {
          await streamBenchmarkProgress({
            runId,
            signal: controller.signal,
            onEvent: (event) => {
              const store = useBenchmarkRuntimeStore.getState();
              if (event.event === "error") {
                // A stream-level error ends the stream server-side; reconnecting is
                // the right response and the run itself is unaffected.
                return;
              }
              if (event.event === "heartbeat") return;
              if (!isStatusPayload(event.payload)) return;
              store.applyStatus(event.payload);
              // The final frame can be emitted before the last console lines are
              // drained, and nothing will come along to carry them now the stream
              // is closing, so a terminal event triggers one last poll.
              if (event.event === "complete") {
                void refreshStatus(runId);
                return;
              }
            },
          });
        } catch (error) {
          if (disposed || controller.signal.aborted || isAbortError(error))
            return;
        }
        if (disposed || controller.signal.aborted) return;
        // The stream ended without a terminal frame, which is what a dropped
        // connection looks like. Reconnect rather than showing a frozen number.
        await new Promise((resolve) =>
          setTimeout(resolve, STREAM_RECONNECT_DELAY_MS),
        );
      }
    };

    void run();

    return () => {
      disposed = true;
      controller.abort();
      streamAbort.current = null;
    };
  }, [active, refreshStatus, runId]);

  // --- polling backstop, and the stall signal the view can show ---
  useEffect(() => {
    if (!active || !runId) return;
    const poll = setInterval(() => void refreshStatus(runId), STATUS_POLL_MS);
    return () => clearInterval(poll);
  }, [active, refreshStatus, runId]);
}

/** Exported so the live view can show a stall without owning the stream itself. */
export const BENCHMARK_STREAM_STALL_MS = STREAM_STALL_MS;
