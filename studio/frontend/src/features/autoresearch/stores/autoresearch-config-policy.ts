// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The editable configuration, its defaults, and the rules it has to keep.
 *
 * Small on purpose, unlike nanochat's. autoresearch's own parameters live in
 * `train.py` as constants and belong to the agent, not to a form: the whole
 * premise is that the agent decides what to try, and a UI that let a person set
 * the learning rate would be a second, competing source of truth for the one
 * thing the search is exploring. So what is configurable here is the shape of
 * the loop, not its content.
 */

export type AutoresearchConfigState = {
  /** The headline setting: how many experiments to run. */
  numExperiments: number;
  /** Which coding agent runs each experiment. */
  agent: string;
  /**
   * Extra arguments for the agent CLI, verbatim.
   *
   * The lever for running the loop on a local model rather than the agent CLI's
   * own default. Each CLI spells this differently, so this takes whatever that
   * CLI accepts — `--model <name>` and a provider or base-url flag, typically.
   *
   * The agent is a separate program with its own model setting that the studio
   * cannot know, and a loop that quietly trains against a cloud model while the
   * user believes it is running on theirs is worse than one that does not run.
   */
  agentArgs: string;
  /**
   * Whether the agent may edit files and run commands without being asked.
   *
   * It has to be on for a loop to run unattended at all, so it defaults on. It
   * is exposed rather than assumed because this is the setting that decides
   * whether an LLM can run shell commands on this machine.
   */
  skipPermissions: boolean;
  /**
   * Per-experiment wall clock, covering the agent's thinking, its edit, the
   * training run and its commit.
   */
  agentTimeoutMinutes: number;
  /**
   * Give up after this many failures in a row. Zero means never.
   *
   * Never is the honest default: a crash is information, and a loop that gives
   * up after two cannot tell a bad idea from a bad machine.
   */
  maxConsecutiveFailures: number;
  /** Download the dataset and train the tokenizer before the first experiment. */
  prepareData: boolean;
  /** A one-line steer, prepended to the project's own instructions. */
  focus: string;
  /**
   * Write the report as the run's last phase, with this model.
   *
   * On one GPU the report cannot be written while the experiments hold the card,
   * and the moment it becomes possible is when the queue drains — which is not a
   * moment a user can be asked to be watching for. Ticking this makes the loop
   * hand the GPU to the model when it finishes.
   */
  reportAfterFinish: boolean;
  reportModelId: string;
  reportSource: "catalog" | "autoresearch" | "agent";
  /** Which CLI, when the report source is an agent. Blank means the loop's own. */
  reportAgentKey: string;
};

export const DEFAULT_AUTORESEARCH_CONFIG: AutoresearchConfigState = {
  numExperiments: 8,
  // opencode rather than codex: it is the one of the three most often pointed at
  // a local model, because its provider configuration is where a user sets one
  // up. An experiment loop that runs on the user's own weights rather than a
  // cloud account is the right default for a project whose whole point is
  // measuring one specific machine.
  agent: "opencode",
  agentArgs: "",
  skipPermissions: true,
  agentTimeoutMinutes: 45,
  maxConsecutiveFailures: 0,
  prepareData: true,
  focus: "",
  reportAfterFinish: false,
  reportModelId: "",
  reportSource: "catalog",
  reportAgentKey: "",
};

/**
 * A day of unattended work. Each experiment is a real agent invocation that
 * trains a model, and the point of a bound is that a typo in a number field
 * cannot start a weekend of GPU time with nobody watching.
 */
export const MIN_EXPERIMENTS = 1;
export const MAX_EXPERIMENTS = 144;

/**
 * Per-experiment timeouts, in minutes.
 *
 * The floor is above autoresearch's own guidance to kill a training pass past
 * ten minutes: cutting an agent off while it is mid-commit loses the
 * experiment's work, which is the one thing the loop cannot recover.
 */
export const MIN_TIMEOUT_MINUTES = 15;
export const MAX_TIMEOUT_MINUTES = 360;

/** autoresearch's per-experiment budget is a fixed 300s in prepare.py, plus a
 *  21M-token evaluation. Used only to estimate a run's wall clock. */
export const SECONDS_PER_EXPERIMENT_ESTIMATE = 6 * 60;

export const AGENT_LABELS: Record<string, string> = {
  codex: "Codex CLI",
  opencode: "opencode",
  claude: "Claude Code",
};

/**
 * Flags the loop sets itself, which a passthrough must not be able to override.
 * Mirrors the backend's list, and is asserted against it by a test: a field that
 * can change the working directory or the brief can change what the run does.
 */
export const AGENT_OWNED_FLAGS = new Set([
  "-C",
  "--cd",
  "--dir",
  "--directory",
  "-w",
  "--cwd",
]);

export function agentLabel(key: string): string {
  return AGENT_LABELS[key] ?? key;
}

/** What a run of this size costs, in seconds, roughly. */
export function estimateRunSeconds(config: AutoresearchConfigState): number {
  return config.numExperiments * SECONDS_PER_EXPERIMENT_ESTIMATE;
}

/**
 * Split the agent-args field into argv entries.
 *
 * One whitespace-separated word per argument, and no quoting.
 *
 * Quoting was considered and dropped: `"` is a character a shell treats as
 * syntax, and a field whose contents reach a command line is not the place to
 * support quoting halfway. It is also unnecessary, because the flags this field
 * exists for -- `--model <name>`, `--config <key=value>`, a provider or base-url
 * flag -- are all single tokens. A value that genuinely needs a space is better
 * expressed with the CLI's own environment or config file than by smuggling a
 * quoted string through a field that rejects quotes.
 */
export function parseAgentArgs(value: string): string[] {
  return value.split(/\s+/).filter((entry) => entry.length > 0);
}

/** What the user typed, as the argv it becomes. For showing them what will run. */
export function describeAgentArgs(value: string): string {
  const parsed = parseAgentArgs(value);
  return parsed.length === 0 ? "the agent's own defaults" : parsed.join(" ");
}

/**
 * Whether the configuration can be started as it stands.
 *
 * Reports rather than blocks, and deliberately does not check whether the agent
 * is installed: the EnvironmentGate beside it already says so, and a form that
 * disables Start for a reason printed three fields away is a worse experience
 * than a click that explains itself.
 */
export function validateAutoresearchConfig(config: AutoresearchConfigState): {
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];

  // A character here reaches a command line, and on Windows the agent's .cmd
  // shim hands it to cmd.exe first. Caught in the form so the reason is beside
  // the field rather than in a 422 after a click.
  const shellSyntax = /[&|<>^%!`"';$()\n\r\t]/;

  if (!Number.isInteger(config.numExperiments)) {
    errors.push("The experiment count must be a whole number.");
  } else if (config.numExperiments < MIN_EXPERIMENTS) {
    errors.push(`Run at least ${MIN_EXPERIMENTS} experiment.`);
  } else if (config.numExperiments > MAX_EXPERIMENTS) {
    errors.push(`${MAX_EXPERIMENTS} experiments is the ceiling.`);
  }

  if (
    !Number.isFinite(config.agentTimeoutMinutes) ||
    config.agentTimeoutMinutes < MIN_TIMEOUT_MINUTES
  ) {
    errors.push(
      `Give each experiment at least ${MIN_TIMEOUT_MINUTES} minutes, which is above the point where autoresearch itself would have killed a training pass.`,
    );
  } else if (config.agentTimeoutMinutes > MAX_TIMEOUT_MINUTES) {
    errors.push(`${MAX_TIMEOUT_MINUTES} minutes is the ceiling per experiment.`);
  }

  if (config.maxConsecutiveFailures < 0) {
    errors.push("The failure limit cannot be negative.");
  }

  if (!config.agent) {
    errors.push("Choose an agent to run the experiments.");
  }

  if (config.reportAfterFinish && !config.reportModelId) {
    // The one error that is about this form rather than about the machine: the
    // backend refuses it too, but a disabled Start button next to the reason
    // beats a 409 after a click.
    errors.push("Choose a model to write the report, or untick writing it when the loop finishes.");
  }

  if (shellSyntax.test(config.agentArgs)) {
    errors.push(
      "The agent arguments contain a character a shell treats as syntax. A model name or base URL does not need one.",
    );
  }
  for (const entry of parseAgentArgs(config.agentArgs)) {
    const head = entry.split("=", 1)[0];
    if (AGENT_OWNED_FLAGS.has(head)) {
      errors.push(
        `Remove ${head}: the loop sets the agent's working directory and brief itself.`,
      );
    }
  }

  // Not errors. These change how long a run takes or how it is judged, and the
  // user is entitled to choose, but neither is a mistake.
  if (estimateRunSeconds(config) > 12 * 3600) {
    warnings.push(
      "That is more than half a day of training. Consider running it while you are away rather than while you are watching.",
    );
  }
  if (config.numExperiments <= 2) {
    warnings.push(
      "One or two experiments rarely find anything: the first sets the baseline and the second has almost nothing to work from.",
    );
  }
  if (config.focus.trim().length > 400) {
    warnings.push(
      "A long focus is unlikely to be followed. One or two sentences works better than a specification.",
    );
  }

  return { errors, warnings };
}
