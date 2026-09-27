// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Setting up a loop: how many experiments, which agent, and what the tab needs
 * installed before any of it can run.
 *
 * The form is short on purpose. autoresearch's actual parameters live in
 * `train.py` as constants and belong to the agent — the whole premise is that it
 * decides what to try. A form full of learning rates would be a second,
 * competing source of truth for the one thing the search is exploring. So what
 * is set here is the shape of the loop, not its content.
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import { Alert02Icon, CheckmarkCircleIcon, Settings02Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { SectionCard } from "@/components/section-card";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";

import {
  reportSourceSuffix,
  useReportModels,
} from "../hooks/use-report-models";
import {
  getAutoresearchAgents,
  getAutoresearchEnvironment,
  installAutoresearchEnvironment,
  prepareAutoresearchData,
  startAutoresearchRun,
} from "../api/autoresearch-api";
import {
  MAX_EXPERIMENTS,
  MAX_TIMEOUT_MINUTES,
  MIN_EXPERIMENTS,
  MIN_TIMEOUT_MINUTES,
  agentLabel,
  describeAgentArgs,
  estimateRunSeconds,
  validateAutoresearchConfig,
} from "../stores/autoresearch-config-policy";
import { useAutoresearchConfigStore } from "../stores/autoresearch-config-store";
import {
  isAutoresearchRunActive,
  useAutoresearchRuntimeStore,
} from "../stores/autoresearch-runtime-store";
import { formatDuration } from "../lib/format";

/** How often the gate re-reads the environment while something is installing. */
const INSTALL_POLL_MS = 2000;

export function AutoresearchConfigureView({
  onStarted,
}: {
  onStarted: () => void;
}) {
  const config = useAutoresearchConfigStore();
  const apply = useAutoresearchConfigStore((state) => state.apply);
  const runActive = useAutoresearchRuntimeStore(isAutoresearchRunActive);
  const markStarted = useAutoresearchRuntimeStore((state) => state.markStarted);

  const environment = useAutoresearchRuntimeStore((state) => state.environment);
  const agents = useAutoresearchRuntimeStore((state) => state.agents);
  const setEnvironment = useAutoresearchRuntimeStore((state) => state.setEnvironment);
  const setAgents = useAutoresearchRuntimeStore((state) => state.setAgents);

  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const reportModels = useReportModels();

  // Derived, not a flag. The install runs on a worker thread and reports through
  // this same status object, so a separate boolean could disagree with it -- and
  // a button that says "installing" while the bar has stopped is the kind of
  // small lie that makes a user press it twice.
  const installing = environment?.install_state === "installing";

  const validation = useMemo(() => validateAutoresearchConfig(config), [config]);
  const estimate = useMemo(() => estimateRunSeconds(config), [config]);

  // The environment and the agent list are read once together: they are both
  // slow-ish on first call (one of them spawns a subprocess to ask a venv what
  // it has) and neither changes while the tab is open except during an install.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const [environmentResult, agentsResult] = await Promise.allSettled([
        getAutoresearchEnvironment(),
        getAutoresearchAgents(),
      ]);
      if (cancelled) return;
      if (environmentResult.status === "fulfilled") setEnvironment(environmentResult.value);
      if (agentsResult.status === "fulfilled") setAgents(agentsResult.value);
    })();
    return () => {
      cancelled = true;
    };
  }, [setAgents, setEnvironment]);

  // While an install is going, keep asking. It runs on a worker thread, so this
  // poll is the only way the bar moves.
  useEffect(() => {
    if (!installing) return;
    const timer = setInterval(() => {
      void getAutoresearchEnvironment()
        .then(setEnvironment)
        .catch(() => undefined);
    }, INSTALL_POLL_MS);
    return () => clearInterval(timer);
  }, [installing, setEnvironment]);

  const handleInstall = useCallback(async () => {
    setStartError(null);
    try {
      setEnvironment(await installAutoresearchEnvironment({ prepareData: config.prepareData }));
    } catch (error) {
      setStartError(error instanceof Error ? error.message : String(error));
    }
  }, [config.prepareData, setEnvironment]);

  const handlePrepare = useCallback(async () => {
    setStartError(null);
    try {
      setEnvironment(await prepareAutoresearchData());
    } catch (error) {
      setStartError(error instanceof Error ? error.message : String(error));
    }
  }, [setEnvironment]);

  const handleStart = useCallback(async () => {
    setIsStarting(true);
    setStartError(null);
    try {
      const response = await startAutoresearchRun(config);
      markStarted(response.run_id, config.numExperiments);
      onStarted();
    } catch (error) {
      setStartError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsStarting(false);
    }
  }, [config, markStarted, onStarted]);

  const chosenAgent = agents?.agents.find((agent) => agent.key === config.agent);
  const agentUnavailable = agents ? !chosenAgent?.available : false;
  const notReady = environment !== null && !environment.ready;
  const canStart =
    !runActive &&
    !isStarting &&
    !notReady &&
    !agentUnavailable &&
    validation.errors.length === 0;

  return (
    <div className="flex flex-col gap-4 pb-8">
      <EnvironmentGate
        onInstall={handleInstall}
        onPrepare={handlePrepare}
        busy={installing}
      />

      <SectionCard
        icon={<HugeiconsIcon icon={Settings02Icon} className="size-5" />}
        title="The loop"
        description="How many experiments to run, and which agent runs each one."
      >
        <div className="flex flex-col gap-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="autoresearch-count">Experiments</Label>
              <Input
                id="autoresearch-count"
                type="number"
                min={MIN_EXPERIMENTS}
                max={MAX_EXPERIMENTS}
                value={config.numExperiments}
                disabled={runActive}
                onChange={(event) =>
                  apply({ numExperiments: Number(event.target.value) })
                }
              />
              <p className="text-ui-11 text-muted-foreground">
                Roughly {formatDuration(estimate)} at about six minutes each, and it
                runs unattended.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="autoresearch-timeout">Minutes per experiment</Label>
              <Input
                id="autoresearch-timeout"
                type="number"
                min={MIN_TIMEOUT_MINUTES}
                max={MAX_TIMEOUT_MINUTES}
                value={config.agentTimeoutMinutes}
                disabled={runActive}
                onChange={(event) =>
                  apply({ agentTimeoutMinutes: Number(event.target.value) })
                }
              />
              <p className="text-ui-11 text-muted-foreground">
                Covers the agent thinking, its edit, the training run and its commit.
                autoresearch suggests killing a training pass past ten minutes.
              </p>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <Label>Agent</Label>
            <p className="text-ui-11 text-muted-foreground">
              One coding agent per experiment. It reads the project's own program.md,
              changes train.py, trains, and commits the change if it helped.
            </p>
            <div className="flex flex-col gap-2">
              {(agents?.agents ?? []).map((agent) => {
                const selected = agent.key === config.agent;
                return (
                  <label
                    key={agent.key}
                    className={[
                      "flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors",
                      selected ? "border-primary/50 bg-primary/5" : "border-border",
                      (!agent.available || runActive) && "cursor-not-allowed opacity-60",
                    ].join(" ")}
                  >
                    <input
                      type="radio"
                      name="autoresearch-agent"
                      className="mt-1"
                      checked={selected}
                      disabled={!agent.available || runActive}
                      onChange={() => apply({ agent: agent.key })}
                    />
                    <span className="flex min-w-0 flex-1 flex-col gap-1">
                      <span className="flex flex-wrap items-center gap-2">
                        <span className="text-ui-13 font-medium">{agent.label}</span>
                        {agent.version ? (
                          <Badge variant="outline" className="font-mono text-[10px]">
                            {agent.version}
                          </Badge>
                        ) : null}
                        {!agent.available ? (
                          <Badge className="bg-amber-500/15 text-[10px] text-amber-600 dark:text-amber-400">
                            unavailable
                          </Badge>
                        ) : null}
                      </span>
                      <span className="text-ui-11 text-muted-foreground">
                        {agent.available ? agent.notes : agent.reason}
                      </span>
                    </span>
                  </label>
                );
              })}
              {agents && agents.agents.length === 0 ? (
                <p className="text-ui-12 text-muted-foreground">
                  No agent was found. Install one of Codex CLI, opencode or Claude
                  Code and reopen the tab.
                </p>
              ) : null}
            </div>
          </div>

          {agents?.git_error ? (
            <p className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-ui-11 text-amber-700 dark:text-amber-400">
              The experiment repository is not ready: {agents.git_error}
            </p>
          ) : agents?.git_dirty ? (
            <p className="flex items-start gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-ui-11 text-amber-700 dark:text-amber-400">
              <HugeiconsIcon icon={Alert02Icon} className="mt-px size-3.5 shrink-0" />
              <span>
                The experiment repository has uncommitted changes, so this run's
                baseline is not the last recorded experiment. A previous run was
                probably interrupted between editing train.py and committing.
              </span>
            </p>
          ) : null}

          <div className="flex flex-col gap-2">
            <Label htmlFor="autoresearch-agent-args">Agent arguments</Label>
            <Input
              id="autoresearch-agent-args"
              value={config.agentArgs}
              disabled={runActive}
              placeholder="--model qwen3-coder-30b"
              onChange={(event) => apply({ agentArgs: event.target.value })}
            />
            <p className="text-ui-11 text-muted-foreground">
              {agentLabel(config.agent)} ships with its own model, which is
              whatever that CLI is configured for. Put the flags here to run the
              loop on a local model instead — whatever {agentLabel(config.agent)}{" "}
              calls its model flag, plus any provider or base URL it needs.
            </p>
            {config.agentArgs.trim() ? (
              <p className="font-mono text-[10px] text-muted-foreground/80">
                will run: {describeAgentArgs(config.agentArgs)}
              </p>
            ) : null}
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="autoresearch-focus">Focus (optional)</Label>
            <Textarea
              id="autoresearch-focus"
              rows={3}
              value={config.focus}
              disabled={runActive}
              placeholder="e.g. the optimiser is the obvious place; the model is small enough that depth changes are cheap"
              onChange={(event) => apply({ focus: event.target.value })}
            />
            <p className="text-ui-11 text-muted-foreground">
              One or two sentences, passed to the agent alongside the project's own
              instructions. It is a steer, not a specification.
            </p>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        icon={<HugeiconsIcon icon={Alert02Icon} className="size-5" />}
        accent="orange"
        title="Safety"
        description="What the agent is allowed to do without being asked."
      >
        <div className="flex flex-col gap-4">
          <label className="flex cursor-pointer items-start gap-3">
            <input
              type="checkbox"
              className="mt-1"
              checked={config.skipPermissions}
              disabled={runActive}
              onChange={(event) => apply({ skipPermissions: event.target.checked })}
            />
            <span className="flex flex-col gap-1">
              <span className="text-ui-13 font-medium">
                Let the agent act without asking
              </span>
              <span className="text-ui-11 text-muted-foreground">
                Required for an unattended loop: each experiment has to edit a file,
                run a five-minute training pass and commit, and none of that can be
                answered by a prompt. The agent works only inside a copy of the
                project in this app's own workspace, never in your own folder.
              </span>
            </span>
          </label>

          <label className="flex cursor-pointer items-start gap-3">
            <input
              type="checkbox"
              className="mt-1"
              checked={config.prepareData}
              disabled={runActive}
              onChange={(event) => apply({ prepareData: event.target.checked })}
            />
            <span className="flex flex-col gap-1">
              <span className="text-ui-13 font-medium">
                Prepare the dataset before the first experiment
              </span>
              <span className="text-ui-11 text-muted-foreground">
                Downloads TinyStories and trains a tokenizer. Only needed once, and
                skipped if it is already on disk.
              </span>
            </span>
          </label>

          <label className="flex cursor-pointer items-start gap-3">
            <input
              type="checkbox"
              className="mt-1"
              checked={config.maxConsecutiveFailures === 0}
              disabled={runActive}
              onChange={(event) =>
                apply({ maxConsecutiveFailures: event.target.checked ? 0 : 3 })
              }
            />
            <span className="flex flex-col gap-1">
              <span className="text-ui-13 font-medium">Never give up</span>
              <span className="text-ui-11 text-muted-foreground">
                On by default, and it is the honest setting: a crash is information,
                and a loop that gives up after two cannot tell a bad idea from a bad
                machine. Turn it off to stop after three failures in a row.
              </span>
            </span>
          </label>
        </div>
      </SectionCard>

      <SectionCard
        icon={<HugeiconsIcon icon={SparklesIcon} className="size-5" />}
        accent="blue"
        title="Then the report"
        description="Have the loop hand the GPU to a model when it finishes."
      >
        <div className="flex flex-col gap-4">
          <label className="flex cursor-pointer items-start gap-3">
            <input
              type="checkbox"
              className="mt-1"
              checked={config.reportAfterFinish}
              disabled={runActive}
              onChange={(event) => apply({ reportAfterFinish: event.target.checked })}
            />
            <span className="flex flex-col gap-1">
              <span className="text-ui-13 font-medium">
                Write the report when the loop finishes
              </span>
              <span className="text-ui-11 text-muted-foreground">
                There is one GPU, and the experiments hold it for hours. The report
                can only be written once they let go, which is a moment you cannot
                plan to be watching for — so tick this and the loop does it as its
                last step. You can still write one by hand at any time.
              </span>
            </span>
          </label>

          {config.reportAfterFinish ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="autoresearch-report-model">Write it with</Label>
              {reportModels.loading ? (
                <p className="text-ui-11 text-muted-foreground">Looking for models…</p>
              ) : reportModels.models.length === 0 ? (
                <p className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-ui-11 text-amber-700 dark:text-amber-400">
                  {reportModels.loading
                    ? "Looking for models…"
                    : "Nothing to write with yet. Install an agent (opencode, Codex or Claude Code), or load a model in Chat."}
                </p>
              ) : (
                <select
                  id="autoresearch-report-model"
                  className="h-9 rounded-md border border-input bg-transparent px-3 text-sm"
                  value={config.reportModelId}
                  disabled={runActive}
                  onChange={(event) => {
                    const next = reportModels.models.find(
                      (model) => model.id === event.target.value,
                    );
                    apply({
                      reportModelId: event.target.value,
                      reportSource: next?.source ?? "catalog",
                      // Carried so the report is written by the same agent, with
                      // the same arguments, that ran the experiments. A second
                      // decision about which model wrote the search is not one the
                      // user should have to make twice.
                      reportAgentKey: next?.agentKey ?? "",
                    });
                  }}
                >
                  <option value="">Choose a model…</option>
                  {reportModels.models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.label}
                      {reportSourceSuffix(model)}
                    </option>
                  ))}
                </select>
              )}
              {reportModels.catalogEmpty && reportModels.agentsEmpty && reportModels.models.length > 0 ? (
                <p className="text-ui-11 text-muted-foreground">
                  No agent is runnable and no catalog model is available, so the
                  only choice is this search&apos;s own checkpoints. A report from
                  one of those is the experiment describing itself, not an
                  assessment of it.
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      </SectionCard>

      {validation.errors.length > 0 ? (
        <ul className="flex flex-col gap-1 rounded-md border border-destructive/30 bg-destructive/5 p-3 text-ui-12 text-destructive">
          {validation.errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      ) : null}

      {validation.warnings.length > 0 ? (
        <ul className="flex flex-col gap-1 rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-ui-12 text-amber-700 dark:text-amber-400">
          {validation.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}

      {startError ? (
        <p className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-ui-12 text-destructive">
          {startError}
        </p>
      ) : null}

      <div className="flex items-center gap-3">
        <Button disabled={!canStart} onClick={() => void handleStart()} className="gap-1.5">
          {isStarting ? <Spinner className="size-4" /> : null}
          Run {config.numExperiments} experiment{config.numExperiments === 1 ? "" : "s"}
        </Button>
        <span className="text-ui-11 text-muted-foreground">
          About {formatDuration(estimate)}. You can close the tab; the loop keeps
          going, and reopening the tab picks it back up.
        </span>
      </div>
    </div>
  );
}

/**
 * What has to be installed before a run can start, and the one button that does
 * it.
 *
 * Shown rather than hidden, because every item on it is something that is
 * genuinely missing from most machines, and a Start button that fails with a
 * backend message the user cannot act on is a worse first experience than a
 * short list with a fix next to each entry.
 */
function EnvironmentGate({
  onInstall,
  onPrepare,
  busy,
}: {
  onInstall: () => Promise<void>;
  onPrepare: () => Promise<void>;
  busy: boolean;
}) {
  const environment = useAutoresearchRuntimeStore((state) => state.environment);
  const [expanded, setExpanded] = useState(false);

  if (environment === null) {
    return (
      <div className="flex justify-center rounded-2xl border border-border py-10">
        <Spinner />
      </div>
    );
  }

  if (environment.ready && environment.data_prepared) {
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-2xl border border-border p-3 text-ui-12">
        <HugeiconsIcon
          icon={CheckmarkCircleIcon}
          className="size-4 shrink-0 text-emerald-500"
        />
        <span>
          autoresearch is ready
          {environment.device_name ? ` on ${environment.device_name}` : ""}
          {environment.torch_version ? `, torch ${environment.torch_version}` : ""}.
        </span>
        {environment.install_message ? (
          <span className="text-muted-foreground">{environment.install_message}</span>
        ) : null}
        <button
          type="button"
          className="ml-auto text-ui-11 text-muted-foreground underline underline-offset-2 hover:text-foreground"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Hide details" : "Details"}
        </button>
      </div>
    );
  }

  const installing = environment.install_state === "installing";
  const failed = environment.install_state === "failed";

  return (
    <div
      className={[
        "flex flex-col gap-3 rounded-2xl border p-4",
        failed ? "border-destructive/40 bg-destructive/5" : "border-border",
      ].join(" ")}
    >
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <p className="text-ui-14 font-medium">
            {installing
              ? "Setting autoresearch up"
              : failed
                ? "The setup did not finish"
                : "autoresearch is not set up yet"}
          </p>
          <p className="text-ui-12 text-muted-foreground">
            {installing
              ? environment.install_message || "Working…"
              : (environment.blocking_reason ??
                "One-time setup: copies the project in, builds its environment and downloads the dataset.")}
          </p>
        </div>
        {!installing ? (
          <Button onClick={() => void onInstall()} disabled={busy}>
            Set up
          </Button>
        ) : null}
      </div>

      {installing ? (
        <div className="flex flex-col gap-1.5">
          <Progress value={Math.round(environment.install_progress * 100)} />
          <p className="text-ui-11 text-muted-foreground">
            {Math.round(environment.install_progress * 100)}% · this downloads torch
            and the dataset, so it takes a while
          </p>
        </div>
      ) : null}

      <ul className="flex flex-col gap-1.5 text-ui-11">
        <Requirement
          ok={environment.source_present}
          label="Found the autoresearch-win-rtx folder"
          detail={environment.source_path || "Searched for a folder holding prepare.py and train.py"}
        />
        <Requirement
          ok={environment.checkout_present}
          label="Copied into this app's workspace"
          detail={environment.checkout_path}
        />
        <Requirement
          ok={environment.venv_present && environment.torch_installed}
          label="Built its own environment with torch"
          detail={environment.torch_version ? `torch ${environment.torch_version}` : "uv, then torch"}
        />
        <Requirement
          ok={environment.git_ready}
          label="Initialised the experiment repository"
          detail="The keep-or-discard protocol is expressed in git, so it needs one commit to return to"
        />
        <Requirement
          ok={environment.data_prepared}
          label="Prepared the dataset and tokenizer"
          detail="autoresearch's train.py reads the tokenizer from here and dies without it"
        />
      </ul>

      {!installing && !environment.data_prepared && environment.venv_present ? (
        <Button variant="outline" onClick={() => void onPrepare()} disabled={busy}>
          Prepare data only
        </Button>
      ) : null}

      {failed && environment.install_message ? (
        <p className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-ui-11 text-destructive">
          {environment.install_message}
        </p>
      ) : null}

      <button
        type="button"
        className="self-start text-ui-11 text-muted-foreground underline underline-offset-2 hover:text-foreground"
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? "Hide paths" : "Where this all lives"}
      </button>
      {expanded ? (
        <dl className="grid gap-1 font-mono text-[10px] text-muted-foreground">
          <Row label="source" value={environment.source_path} />
          <Row label="checkout" value={environment.checkout_path} />
          <Row label="python" value={environment.python_path} />
        </dl>
      ) : null}

      {!environment.cuda_available ? (
        <p className="text-ui-11 text-amber-700 dark:text-amber-400">
          No CUDA device is visible. autoresearch will still run, but its whole
          design assumes an NVIDIA GPU: on CPU an experiment takes far longer than
          its five-minute budget and the results are not comparable.
        </p>
      ) : null}
    </div>
  );
}

function Requirement({
  ok,
  label,
  detail,
}: {
  ok: boolean;
  label: string;
  detail: string;
}) {
  return (
    <li className="flex items-start gap-2">
      <HugeiconsIcon
        icon={ok ? CheckmarkCircleIcon : Alert02Icon}
        className={[
          "mt-px size-3.5 shrink-0",
          ok ? "text-emerald-500" : "text-amber-500",
        ].join(" ")}
      />
      <span className="flex min-w-0 flex-col">
        <span className={ok ? "" : "text-foreground"}>{label}</span>
        {detail ? (
          <span className="truncate text-muted-foreground/70">{detail}</span>
        ) : null}
      </span>
    </li>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  if (!value) return null;
  return (
    <div className="flex gap-2">
      <dt className="shrink-0 text-muted-foreground/60">{label}</dt>
      <dd className="min-w-0 break-all">{value}</dd>
    </div>
  );
}
