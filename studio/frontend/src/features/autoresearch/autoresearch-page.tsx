// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * autoresearch: nanochat's from-scratch pipeline, run repeatedly by an agent and
 * judged on one number.
 *
 * Five sub-tabs rather than nanochat's three, and the extra two are not
 * decoration. "Results" is the ledger, which outlives any single loop, so it is
 * the thing someone comes back to the next morning rather than a view of
 * whatever is running. "Models" is where the weights each experiment left behind
 * are, and where they can be talked to. "Report" is a model writing up what the
 * search found, which is a language task and so belongs to a language model
 * rather than to this tab.
 */
import { useCallback, useState } from "react";

import { Tabs, TabsContent } from "@/components/ui/tabs";

import { AutoresearchSubNav, type AutoresearchTab } from "./autoresearch-navigation";
import { AutoresearchStartOverlay } from "./views/autoresearch-console";
import { AutoresearchConfigureView } from "./views/configure-view";
import { AutoresearchLiveRunView } from "./views/live-run-view";
import { AutoresearchResultsView } from "./views/results-view";
import { AutoresearchCheckpointsView } from "./views/checkpoints-view";
import { AutoresearchReportView } from "./views/report-view";
import { useAutoresearchRuntimeLifecycle } from "./hooks/use-autoresearch-runtime-lifecycle";
import {
  hasAutoresearchHistory,
  isAutoresearchRunActive,
  useAutoresearchRuntimeStore,
} from "./stores/autoresearch-runtime-store";

export function AutoresearchPage() {
  useAutoresearchRuntimeLifecycle();

  const runActive = useAutoresearchRuntimeStore(isAutoresearchRunActive);
  const hasHistory = useAutoresearchRuntimeStore(hasAutoresearchHistory);
  const runId = useAutoresearchRuntimeStore((state) => state.runId);
  const resultsLoaded = useAutoresearchRuntimeStore((state) => state.resultsLoaded);
  const resultCount = useAutoresearchRuntimeStore((state) => state.results.length);
  const checkpointCount = useAutoresearchRuntimeStore((state) => state.checkpoints.length);
  // The report tab opens for a report, not for a ledger. A run that was asked to
  // write one has a report even if every experiment crashed and the ledger is
  // empty of scores, and hiding the tab then would hide the explanation of why.
  const reportText = useAutoresearchRuntimeStore((state) => state.reportText);
  const hasReport = resultsLoaded && (resultCount > 0 || reportText.length > 0);

  // A run that is going, or one that has a result worth looking at, opens on
  // Current Run. Otherwise the tab is a form, and that is what should be showing.
  const initialTab: AutoresearchTab = runActive || hasHistory ? "current-run" : "configure";
  const [tab, setTab] = useState<AutoresearchTab>(initialTab);

  const select = useCallback(
    (value: string) => {
      const next = value as AutoresearchTab;
      setTab(next);
      if (next === "configure") {
        // A finished run should not block a new one.
        const store = useAutoresearchRuntimeStore.getState();
        if (!runActive && store.status !== "idle") store.reset();
      }
    },
    [runActive],
  );

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <Tabs value={tab} onValueChange={select} className="contents">
        <div className="relative mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] flex-col">
          <header className="font-heading flex flex-col gap-5">
            <h1 className="page-title-halo text-ui-30 font-semibold tracking-tight">
              autoresearch
            </h1>
            <p className="page-title-halo text-sm text-muted-foreground">
              An agent edits the training script, trains for five minutes, keeps the
              change if the score improved, and repeats.
            </p>
            <div className="flex min-w-0 flex-wrap items-center gap-3 border-b border-border/60">
              <AutoresearchSubNav
                value={tab}
                runActive={runActive}
                showRunView={runActive || hasHistory}
                hasResults={resultCount > 0}
                hasCheckpoints={checkpointCount > 0}
                hasReport={hasReport}
              />
              {runId ? (
                <span className="font-mono text-[10px] text-muted-foreground">
                  {runId.replace(/^run_/, "")}
                </span>
              ) : null}
            </div>
          </header>

          <TabsContent value="configure">
            <AutoresearchConfigureView onStarted={() => setTab("current-run")} />
          </TabsContent>
          <TabsContent value="current-run">
            <AutoresearchLiveRunView onConfigure={() => setTab("configure")} />
          </TabsContent>
          <TabsContent value="results">
            <AutoresearchResultsView />
          </TabsContent>
          <TabsContent value="checkpoints">
            <AutoresearchCheckpointsView />
          </TabsContent>
          <TabsContent value="report">
            <AutoresearchReportView />
          </TabsContent>

          <AutoresearchStartOverlay />
        </div>
      </Tabs>
    </div>
  );
}
