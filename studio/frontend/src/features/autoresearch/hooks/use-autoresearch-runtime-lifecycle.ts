// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Keeping the store in step with the backend: hydrate on open, stream while a
 * loop is going, and poll as a backstop.
 *
 * Three channels, on purpose. The stream is the fast path, but it is not the
 * only one: a backgrounded tab, a dropped proxy, or a sleeping laptop all leave
 * a stream silently dead while the run itself carries on. The poll costs a
 * request every few seconds and is what makes the tab honest after a reconnect,
 * and the results poll is the only way the frontier chart learns about an
 * experiment the agent committed after the stream had already gone quiet.
 */
import { useCallback, useEffect, useRef } from "react";

import {
  fetchAutoresearchLogs,
  getAutoresearchMetrics,
  getAutoresearchResults,
  getAutoresearchStatus,
  isAbortError,
  streamAutoresearchProgress,
  type AutoresearchStreamEvent,
} from "../api/autoresearch-api";
import { useAutoresearchRuntimeStore } from "../stores/autoresearch-runtime-store";

// A run can sit in a thinking phase for minutes with nothing to report, so the
// slower of these is not an oversight.
const RESULTS_POLL_MS = 10_000;
const METRICS_POLL_MS = 10_000;
const LOGS_POLL_MS = 4_000;

/** A stream with no frame for this long is treated as dead and reconnected. */
const STREAM_STALL_MS = 45_000;
const STREAM_RECONNECT_DELAY_MS = 2_500;

export function useAutoresearchRuntimeLifecycle(): void {
  const runId = useAutoresearchRuntimeStore((state) => state.runId);
  const active = useAutoresearchRuntimeStore((state) => state.status === "running");
  const streamAbort = useRef<AbortController | null>(null);
  const lastFrameAt = useRef(0);

  const refreshResults = useCallback(async () => {
    try {
      const payload = await getAutoresearchResults();
      useAutoresearchRuntimeStore.getState().applyResults(payload);
    } catch {
      // A failed poll is a hiccup, not a state change: the next tick retries and
      // the stream is still the primary channel.
    }
  }, []);

  const refreshMetrics = useCallback(async () => {
    try {
      const payload = await getAutoresearchMetrics();
      useAutoresearchRuntimeStore.getState().applySeries(payload.series);
    } catch {
      // Same as above.
    }
  }, []);

  const refreshLogs = useCallback(async (expectedRunId: string) => {
    const cursor = useAutoresearchRuntimeStore.getState().logSeq;
    try {
      const payload = await fetchAutoresearchLogs({
        runId: expectedRunId,
        since: cursor,
      });
      useAutoresearchRuntimeStore.getState().appendLogs(payload.logs);
    } catch {
      // Same as above.
    }
  }, []);

  // --- hydration --------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await getAutoresearchStatus();
        if (cancelled) return;
        if (status.run_id) {
          useAutoresearchRuntimeStore.getState().hydrate(
            // The status endpoint and the stream share a shape on the wire, and
            // the store's hydrate is the one place that converts it.
            status as unknown as Parameters<
              ReturnType<typeof useAutoresearchRuntimeStore.getState>["hydrate"]
            >[0],
          );
          // A finished loop's log is the most wanted thing on the tab: it is how
          // the user finds out what the agent was doing all night.
          void refreshLogs(status.run_id);
          void refreshResults();
          void refreshMetrics();
        } else {
          useAutoresearchRuntimeStore.getState().reset();
        }
      } catch {
        if (!cancelled) useAutoresearchRuntimeStore.getState().reset();
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshLogs, refreshMetrics, refreshResults]);

  // --- the stream -------------------------------------------------------
  useEffect(() => {
    if (!active || !runId) return;
    let disposed = false;
    const controller = new AbortController();
    streamAbort.current = controller;

    const run = async () => {
      while (!disposed && !controller.signal.aborted) {
        try {
          await streamAutoresearchProgress({
            runId,
            signal: controller.signal,
            onOpen: () => {
              lastFrameAt.current = Date.now();
            },
            onEvent: (event: AutoresearchStreamEvent) => {
              lastFrameAt.current = Date.now();
              const store = useAutoresearchRuntimeStore.getState();
              if (event.event === "error") {
                // Reconnect rather than fail: the run itself is unaffected by a
                // broken stream, and a dropped connection is not a stopped loop.
                return;
              }
              if (event.event === "heartbeat") return;
              if (event.event === "warning") {
                // The backend emits this when a report could not be filed. The
                // report stream owns that message, so the run stream has nothing
                // to do with it beyond proving the connection is alive.
                return;
              }
              if (event.event === "complete") {
                store.applyProgress({
                  ...event.payload,
                  status: event.payload.status ?? "completed",
                });
                // The last artefacts, which the stream may have ended before.
                void refreshMetrics();
                void refreshResults();
                void refreshLogs(runId);
                return;
              }
              store.applyProgress(event.payload);
            },
          });
        } catch (error) {
          if (disposed || controller.signal.aborted || isAbortError(error)) return;
        }
        if (disposed || controller.signal.aborted) return;
        await new Promise((resolve) => setTimeout(resolve, STREAM_RECONNECT_DELAY_MS));
      }
    };
    void run();

    const watchdog = setInterval(() => {
      if (lastFrameAt.current && Date.now() - lastFrameAt.current > STREAM_STALL_MS) {
        useAutoresearchRuntimeStore.setState({
          message: "Reconnecting to the run…",
        });
      }
    }, 10_000);

    return () => {
      disposed = true;
      clearInterval(watchdog);
      controller.abort();
      streamAbort.current = null;
    };
  }, [active, runId, refreshLogs, refreshMetrics, refreshResults]);

  // --- polling backstop -------------------------------------------------
  useEffect(() => {
    // Results are polled whether or not a run is going: the ledger outlives any
    // one loop, and the tab should show the whole search rather than the last run.
    const resultsTimer = setInterval(() => void refreshResults(), RESULTS_POLL_MS);
    const metricsTimer = setInterval(() => void refreshMetrics(), METRICS_POLL_MS);
    return () => {
      clearInterval(resultsTimer);
      clearInterval(metricsTimer);
    };
  }, [refreshMetrics, refreshResults]);

  useEffect(() => {
    if (!active || !runId) return;
    const logsTimer = setInterval(() => void refreshLogs(runId), LOGS_POLL_MS);
    return () => clearInterval(logsTimer);
  }, [active, runId, refreshLogs]);
}
