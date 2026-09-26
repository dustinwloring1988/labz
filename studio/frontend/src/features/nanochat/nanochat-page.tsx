// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";

import { Tabs, TabsContent } from "@/components/ui/tabs";

import { useNanochatRuntimeLifecycle } from "./hooks/use-nanochat-runtime-lifecycle";
import { NanochatSubNav } from "./nanochat-navigation";
import {
  isNanochatRunActive,
  useNanochatRuntimeStore,
} from "./stores/nanochat-runtime-store";
import { NanochatCheckpointsView } from "./views/checkpoints-view";
import { NanochatConfigureView } from "./views/configure-view";
import { NanochatStartOverlay } from "./views/nanochat-console";
import { NanochatLiveRunView } from "./views/live-run-view";

export type NanochatTab = "configure" | "current-run" | "checkpoints";

/**
 * The nanochat tab host.
 *
 * The frame is the Train frame, class for class: same measure, same header
 * scale, same sub-nav. What differs is only the third tab, because a nanochat
 * run leaves checkpoints rather than a resumable job list.
 */
export function NanochatPage() {
  useNanochatRuntimeLifecycle();

  const runActive = useNanochatRuntimeStore(isNanochatRunActive);
  const hasRun = useNanochatRuntimeStore((state) => state.stages.length > 0);
  const runId = useNanochatRuntimeStore((state) => state.runId);
  // A run that is going, or one that has a result worth looking at, opens on
  // Current Run. Otherwise the tab is a form, and that is what should be showing.
  const initialTab: NanochatTab = runActive || hasRun ? "current-run" : "configure";
  const [tab, setTab] = useState<NanochatTab>(initialTab);

  const select = useCallback(
    (value: string) => {
      const next = value as NanochatTab;
      setTab(next);
      // A finished run should not be in the way of starting another one.
      if (next === "configure") {
        const store = useNanochatRuntimeStore.getState();
        if (!runActive && store.status !== "idle") store.reset();
      }
    },
    [runActive],
  );

  const subtitle =
    tab === "current-run"
      ? "Watch a nanochat run as it trains"
      : tab === "checkpoints"
        ? "Checkpoints this machine has produced"
        : "Train a language model from scratch, then fine-tune it into a chat model";

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <Tabs value={tab} onValueChange={select} className="contents">
        <div className="relative mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] 3xl:max-w-[calc(1440px*var(--ui-space-scale,1))] 4xl:max-w-[calc(1760px*var(--ui-space-scale,1))] flex-col gap-7 px-5 pb-20 pt-8 max-sm:px-4 sm:px-9 sm:pt-10">
          <header className="font-heading flex flex-col gap-5">
            <div className="flex flex-col gap-0.5">
              <h1 className="page-title-halo text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
                nanochat
              </h1>
              <p className="page-title-halo text-sm text-muted-foreground">{subtitle}</p>
            </div>
            <div className="flex min-w-0 flex-wrap items-center gap-3 border-b border-border/60">
              <NanochatSubNav
                value={tab}
                runActive={runActive}
                showRunView={hasRun}
              />
              {runId ? (
                <div className="ml-auto flex items-center gap-2 pb-2">
                  <span className="rounded-full border border-border/60 px-2.5 py-1 font-mono text-ui-10 text-muted-foreground">
                    run {runId}
                  </span>
                </div>
              ) : null}
            </div>
          </header>

          <div className="flex w-full flex-col gap-6">
            <TabsContent value="configure" className="mt-0">
              <NanochatConfigureView onStarted={() => setTab("current-run")} />
            </TabsContent>
            <TabsContent value="current-run" className="mt-0">
              <NanochatLiveRunView onConfigure={() => setTab("configure")} />
            </TabsContent>
            <TabsContent value="checkpoints" className="mt-0">
              <NanochatCheckpointsView />
            </TabsContent>
          </div>

          <NanochatStartOverlay />
        </div>
      </Tabs>
    </div>
  );
}
