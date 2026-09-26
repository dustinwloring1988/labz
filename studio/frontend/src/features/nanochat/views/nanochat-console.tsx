// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The in-app console for a nanochat run.
 *
 * This exists because the alternative was a real console window. Every stage used
 * to be spawned as a child of a console-less backend, so Windows handed it a fresh
 * black console that nothing ever drew on: the user got a window that said nothing
 * and the actual output went to /dev/null. The output belongs here instead, where
 * it is readable, scrollable and stays put.
 *
 * Shaped like Train's start terminal on purpose — same monospace, same `>` prompt
 * convention, same opening lines — so "what state is this run in" reads the same
 * on both tabs.
 */
import { useLayoutEffect, useRef, useState } from "react";

import { MascotImg } from "@/components/mascot-img";
import { TypingAnimation } from "@/components/ui/terminal";
import { cn } from "@/lib/utils";

import { useNanochatRuntimeStore } from "../stores/nanochat-runtime-store";

/**
 * Lines drawn at once.
 *
 * A run can print tens of thousands. Rendering them all would make every new line
 * a full re-layout, and nothing below the fold is readable anyway, so this is a
 * tail view the way a real terminal is.
 */
const VISIBLE_LINES = 400;

/**
 * How a line is coloured.
 *
 * stderr is red because a line the stage wrote to stderr is a warning or a
 * traceback, and burying it in the same colour as the progress output is how a
 * real run's failure ends up scrolled past unnoticed.
 */
function lineClassName(stream: "stdout" | "stderr", isMarker: boolean): string {
  if (stream === "stderr") return "text-red-400";
  if (isMarker) return "text-emerald-300";
  return "text-emerald-200/90";
}

/** The `>` lines the orchestrator writes around each stage. */
function isMarker(line: string): boolean {
  return line.startsWith(">");
}

export function NanochatConsole({
  className,
  running = false,
  heightClassName = "h-72",
}: {
  className?: string;
  /** Draws the "starting" opener and keeps the cursor blinking while true. */
  running?: boolean;
  heightClassName?: string;
}) {
  const logLines = useNanochatRuntimeStore((state) => state.logLines);
  const message = useNanochatRuntimeStore((state) => state.message);
  const currentStage = useNanochatRuntimeStore((state) => state.currentStage);
  const runId = useNanochatRuntimeStore((state) => state.runId);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Follow the tail until the user scrolls up to read something. Yanking them back
  // down mid-read is the single most irritating thing a log view can do.
  //
  // The run id is carried in the state so a new run can reset `following` during
  // render rather than in an effect: the buffer is emptied by the store, and a
  // console that stayed pinned to the old run's scroll offset would open on
  // nothing.
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
            &gt; nanochat run starts...
          </TypingAnimation>
          <div className="text-emerald-200/90">&gt; {message || "Preparing your run..."}</div>

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
              {currentStage ? `> ${currentStage} running` : "> waiting for output"}
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
 * The overlay that greets a run, the way Train's does.
 *
 * It stands down as soon as the run has printed something worth reading, so the
 * console is a moment of "here is what is happening" rather than a modal the user
 * has to dismiss to get on with the run. The console stays in the Current Run tab
 * underneath either way, so nothing that scrolled past is lost.
 */
export function NanochatStartOverlay() {
  const running = useNanochatRuntimeStore((state) => state.status === "running");
  const phase = useNanochatRuntimeStore((state) => state.phase);
  const hasOutput = useNanochatRuntimeStore((state) => state.logLines.length > 0);
  // Hold the overlay until there is something to read, not merely until the first
  // stage is launched. Train's does the same thing with its preparation phases:
  // a terminal that appears and vanishes before printing a line is worse than no
  // terminal, and a run's first minutes are exactly when the user wants to watch.
  const show = running && (phase === "starting" || !hasOutput);

  if (!show) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center rounded-2xl bg-background/45 backdrop-blur-[1px]">
      <div className="pointer-events-auto my-auto flex w-[calc(860px*var(--ui-space-scale,1))] max-w-[calc(100%-2rem)] flex-col items-center gap-3">
        <MascotImg src="unsloth-gem.png" className="size-24 object-contain max-sm:size-16" />
        <NanochatConsole className="w-full" running heightClassName="h-80" />
      </div>
    </div>
  );
}
