// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The in-app console for an autoresearch loop.
 *
 * This exists because the alternative was a real console window. Each experiment
 * is spawned as a child of a console-less backend, so Windows handed it a fresh
 * black console that nothing ever drew on: the user got a window that said
 * nothing while the actual output went nowhere. The output belongs here instead,
 * where it is readable, scrollable and stays put.
 *
 * Shaped like nanochat's console on purpose — same monospace, same `>` prompt
 * convention, same opening lines — so "what state is this run in" reads the same
 * on both tabs.
 */
import { useLayoutEffect, useRef, useState } from "react";

import { MascotImg } from "@/components/mascot-img";
import { TypingAnimation } from "@/components/ui/terminal";
import { cn } from "@/lib/utils";

import { useAutoresearchRuntimeStore } from "../stores/autoresearch-runtime-store";

/**
 * Lines drawn at once.
 *
 * A loop can print tens of thousands across a hundred experiments. Rendering
 * them all would make every new line a full re-layout, and nothing below the
 * fold is readable anyway, so this is a tail view the way a real terminal is.
 */
const VISIBLE_LINES = 400;

/**
 * How a line is coloured.
 *
 * stderr is red because a line written to stderr is a warning or a traceback,
 * and burying it in the same colour as the progress output is how a failed
 * experiment ends up scrolled past unnoticed.
 */
function lineClassName(stream: "stdout" | "stderr", isMarker: boolean): string {
  if (stream === "stderr") return "text-red-400";
  if (isMarker) return "text-emerald-300";
  return "text-emerald-200/90";
}

/** The `>` lines the orchestrator writes around each experiment. */
function isMarker(line: string): boolean {
  return line.startsWith(">");
}

export function AutoresearchConsole({
  className,
  running = false,
  heightClassName = "h-72",
}: {
  className?: string;
  /** Draws the "starting" opener and keeps the cursor blinking while true. */
  running?: boolean;
  heightClassName?: string;
}) {
  const logLines = useAutoresearchRuntimeStore((state) => state.logLines);
  const message = useAutoresearchRuntimeStore((state) => state.message);
  const phase = useAutoresearchRuntimeStore((state) => state.phase);
  const experiment = useAutoresearchRuntimeStore((state) => state.experiment);
  const totalExperiments = useAutoresearchRuntimeStore((state) => state.totalExperiments);
  const runId = useAutoresearchRuntimeStore((state) => state.runId);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Follow the tail until the user scrolls up to read something. Yanking them
  // back down mid-read is the single most irritating thing a log view can do.
  //
  // The run id is carried in the state so a new run can reset `following` during
  // render rather than in an effect: the buffer is emptied by the store, and a
  // console left pinned to the old run's scroll offset would open on nothing.
  const [follow, setFollow] = useState({ runId: runId ?? null, following: true });
  if (follow.runId !== (runId ?? null)) {
    setFollow({ runId: runId ?? null, following: true });
  }
  const following = follow.following;
  const lastSeq = logLines[logLines.length - 1]?.seq ?? 0;

  useLayoutEffect(() => {
    if (!following) return;
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [lastSeq, following]);

  const visible = logLines.slice(-VISIBLE_LINES);
  const hiddenCount = logLines.length - visible.length;

  return (
    <div
      className={cn(
        "flex flex-col overflow-hidden rounded-2xl border border-border bg-black/90 font-mono text-ui-11 shadow-2xl",
        className,
      )}
    >
      <div
        ref={scrollRef}
        onScroll={(event) => {
          const node = event.currentTarget;
          // Within a line of the bottom counts as "at the bottom", so ordinary
          // rubber-banding does not silently stop the tail from following.
          const atBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 24;
          setFollow((prev) => (prev.following === atBottom ? prev : { ...prev, following: atBottom }));
        }}
        className={cn("w-full overflow-auto px-5 py-4 leading-[1.5]", heightClassName)}
      >
        <div className="whitespace-pre-wrap break-words">
          <TypingAnimation
            duration={28}
            className="bg-gradient-to-r from-emerald-300 via-lime-300 to-teal-300 bg-clip-text font-semibold text-transparent"
          >
            &gt; autoresearch run starts...
          </TypingAnimation>
          <div className="text-emerald-200/90">
            &gt; {message || "Preparing your run..."}
          </div>

          {hiddenCount > 0 ? (
            <div className="text-muted-foreground/60">
              ... {hiddenCount.toLocaleString()} earlier lines
            </div>
          ) : null}

          {visible.map((entry) => (
            <div key={entry.seq} className={lineClassName(entry.stream, isMarker(entry.line))}>
              {entry.line}
            </div>
          ))}

          {running ? (
            <div className="text-emerald-300/80">
              {phase === "preparing"
                ? "> preparing the loop"
                : `> experiment ${experiment || "?"} of ${totalExperiments || "?"}`}
              <span className="ml-1 inline-block animate-pulse">█</span>
            </div>
          ) : null}
        </div>
      </div>

      {!following ? (
        <button
          type="button"
          className="self-center px-3 py-1 text-ui-11 text-muted-foreground hover:text-foreground"
          onClick={() => {
            setFollow((prev) => ({ ...prev, following: true }));
            const node = scrollRef.current;
            if (node) node.scrollTop = node.scrollHeight;
          }}
        >
          Jump to latest
        </button>
      ) : null}
    </div>
  );
}

/**
 * The overlay that greets a run, the way nanochat's and Train's do.
 *
 * It stands down as soon as the run has printed something worth reading, so the
 * console is a moment of "here is what is happening" rather than a modal the user
 * has to dismiss to get on with the run. The console stays in the Current Run tab
 * underneath either way, so nothing that scrolled past is lost.
 */
export function AutoresearchStartOverlay() {
  const running = useAutoresearchRuntimeStore((state) => state.status === "running");
  const phase = useAutoresearchRuntimeStore((state) => state.phase);
  const hasOutput = useAutoresearchRuntimeStore((state) => state.logLines.length > 0);
  // Held until there is something to read, not merely until the loop is
  // launched. A terminal that appears and vanishes before printing a line is
  // worse than no terminal, and the first experiment is exactly when the user
  // wants to watch.
  const show = running && (phase === "preparing" || !hasOutput);

  if (!show) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center rounded-2xl bg-background/45 backdrop-blur-[1px]">
      <MascotImg src="labz-gem.png" className="size-24" />
      <p className="mt-4 font-heading text-ui-20 font-semibold">
        {phase === "preparing" ? "Preparing the experiment loop" : "Starting the first experiment"}
      </p>
      <p className="mt-1 max-w-sm text-center text-ui-12 text-muted-foreground">
        An agent is reading program.md and deciding what to change in train.py. This
        takes a minute or two, and then it trains.
      </p>
    </div>
  );
}
