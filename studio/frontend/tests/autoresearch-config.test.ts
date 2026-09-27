// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_AUTORESEARCH_CONFIG,
  AGENT_OWNED_FLAGS,
  MAX_EXPERIMENTS,
  MAX_TIMEOUT_MINUTES,
  MIN_EXPERIMENTS,
  MIN_TIMEOUT_MINUTES,
  describeAgentArgs,
  estimateRunSeconds,
  parseAgentArgs,
  validateAutoresearchConfig,
  type AutoresearchConfigState,
} from "../src/features/autoresearch/stores/autoresearch-config-policy.ts";
import { migrateAutoresearchConfig } from "../src/features/autoresearch/stores/autoresearch-config-persistence.ts";
import { formatBpb, formatBytes, formatDelta, formatDuration } from "../src/features/autoresearch/lib/format.ts";
import { readText } from "./helpers/kit.ts";

function config(patch: Partial<AutoresearchConfigState> = {}): AutoresearchConfigState {
  return { ...DEFAULT_AUTORESEARCH_CONFIG, ...patch };
}

// --- validation ---
//
// Each experiment is a real agent invocation that trains a model. A bad number in
// this form does not fail the run, it commits the user to hours of GPU time, so
// the bounds below are about the request rather than about the software.

test("the defaults validate", () => {
  const { errors } = validateAutoresearchConfig(config());
  assert.deepEqual(errors, []);
});

test("an experiment count outside the bounds is refused", () => {
  assert.ok(
    validateAutoresearchConfig(config({ numExperiments: MIN_EXPERIMENTS - 1 })).errors
      .length > 0,
  );
  assert.ok(
    validateAutoresearchConfig(config({ numExperiments: MAX_EXPERIMENTS + 1 })).errors
      .length > 0,
  );
  assert.deepEqual(
    validateAutoresearchConfig(config({ numExperiments: MAX_EXPERIMENTS })).errors,
    [],
  );
});

test("a fractional experiment count is refused", () => {
  // 4.5 experiments is not a thing, and rounding it silently would start a run
  // the user did not ask for.
  const { errors } = validateAutoresearchConfig(
    config({ numExperiments: 4.5 }),
  );
  assert.ok(errors.some((error) => error.includes("whole number")));
});

test("a timeout below what a training pass needs is refused", () => {
  // autoresearch's own guidance is to kill a training pass past ten minutes, and
  // a per-experiment budget below that would cut off a legitimate run.
  assert.ok(
    validateAutoresearchConfig(
      config({ agentTimeoutMinutes: MIN_TIMEOUT_MINUTES - 1 }),
    ).errors.length > 0,
  );
  assert.deepEqual(
    validateAutoresearchConfig(
      config({ agentTimeoutMinutes: MIN_TIMEOUT_MINUTES }),
    ).errors,
    [],
  );
  assert.ok(
    validateAutoresearchConfig(
      config({ agentTimeoutMinutes: MAX_TIMEOUT_MINUTES + 1 }),
    ).errors.length > 0,
  );
});

test("an unselected agent is refused", () => {
  assert.ok(validateAutoresearchConfig(config({ agent: "" })).errors.length > 0);
});

test("a negative failure limit is refused", () => {
  assert.ok(
    validateAutoresearchConfig(config({ maxConsecutiveFailures: -1 })).errors
      .length > 0,
  );
  // Zero is the default and is meaningful: never give up.
  assert.deepEqual(
    validateAutoresearchConfig(config({ maxConsecutiveFailures: 0 })).errors,
    [],
  );
});

test("a very long run is a warning rather than an error", () => {
  // Half a day of training is a choice, not a mistake, and the user is entitled
  // to make it -- but should be told what they are choosing.
  const { errors, warnings } = validateAutoresearchConfig(
    config({ numExperiments: MAX_EXPERIMENTS }),
  );
  assert.deepEqual(errors, []);
  assert.ok(warnings.some((warning) => warning.includes("half a day")));
});

test("a one or two experiment run is a warning", () => {
  const { warnings } = validateAutoresearchConfig(config({ numExperiments: 2 }));
  assert.ok(warnings.some((warning) => warning.includes("baseline")));
});

test("a runaway focus is a warning, not a rejection", () => {
  const { errors, warnings } = validateAutoresearchConfig(
    config({ focus: "x".repeat(500) }),
  );
  assert.deepEqual(errors, []);
  assert.ok(warnings.some((warning) => warning.includes("focus")));
});

// --- which agent, and why that one ---

test("opencode is the default agent", () => {
  // It is the one of the three most often pointed at a local model, because that
  // is where a user configures one. A loop measuring one machine should default
  // to that machine's own weights.
  assert.equal(DEFAULT_AUTORESEARCH_CONFIG.agent, "opencode");
  assert.deepEqual(validateAutoresearchConfig(config()).errors, []);
});

test("the default agent is one the backend knows how to drive", () => {
  // A default that is only right in the UI would fail the first run of a fresh
  // install on a name the backend does not recognise.
  const backend = readText("../../backend/core/autoresearch/agent.py");
  const match = /PREFERRED_AGENT_ORDER = \(([^)]*)\)/.exec(backend);
  assert.ok(match, "no PREFERRED_AGENT_ORDER in the backend agent module");
  const order = [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  assert.equal(order[0], DEFAULT_AUTORESEARCH_CONFIG.agent);
});

test("the backend's own default matches the form's", () => {
  // A form that opens on one agent and a request that defaults to another is a
  // loop that quietly runs on someone else's account.
  const models = readText("../../backend/models/autoresearch.py");
  const match = /^\s{4}agent: str = "([a-z]+)"/m.exec(models);
  assert.ok(match, "no agent default in models/autoresearch.py");
  assert.equal(match[1], DEFAULT_AUTORESEARCH_CONFIG.agent);
});

test("the report source can be an agent", () => {
  const stored = config({
    reportAfterFinish: true,
    reportModelId: "agent:opencode",
    reportSource: "agent",
    reportAgentKey: "opencode",
  });
  assert.deepEqual(migrateAutoresearchConfig(stored, DEFAULT_AUTORESEARCH_CONFIG), stored);
});

test("an invented report source falls back to the default", () => {
  const merged = migrateAutoresearchConfig(
    { reportSource: "telepathy" } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.reportSource, DEFAULT_AUTORESEARCH_CONFIG.reportSource);
});

test("a report agent key of the wrong type falls back to the default", () => {
  const merged = migrateAutoresearchConfig(
    { reportAgentKey: 7 } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.reportAgentKey, DEFAULT_AUTORESEARCH_CONFIG.reportAgentKey);
});

// --- the agent's own model ---
//
// The agent is a separate program with its own model setting. The studio cannot
// know how someone has configured theirs, so the field takes whatever that CLI
// accepts — and it reaches a command line, which is what makes the character
// rules necessary rather than fussy.

test("agent arguments split on whitespace", () => {
  assert.deepEqual(parseAgentArgs("--model qwen3-coder-30b"), [
    "--model",
    "qwen3-coder-30b",
  ]);
});

test("empty agent arguments produce no argv", () => {
  assert.deepEqual(parseAgentArgs(""), []);
  assert.deepEqual(parseAgentArgs("   "), []);
  assert.equal(describeAgentArgs(""), "the agent's own defaults");
});

test("a run of spaces collapses rather than becoming empty arguments", () => {
  // The field is a textarea, so a wrapped line arrives with runs of spaces and a
  // newline in it.
  assert.deepEqual(parseAgentArgs("  --model   local \n --config  x=1 "), [
    "--model",
    "local",
    "--config",
    "x=1",
  ]);
});

test("a quote is refused rather than half-supported", () => {
  // Quoting is deliberately not offered: a field whose contents reach a command
  // line should not also be a quoting mini-language. A value that needs a space
  // belongs in the CLI's own config, not here.
  const { errors } = validateAutoresearchConfig(config({ agentArgs: '--model "a b"' }));
  assert.ok(errors.length > 0);
});

test("a local model flag is accepted", () => {
  // The refusal rules exist to enable this, not to block it.
  const { errors } = validateAutoresearchConfig(
    config({ agentArgs: "--model qwen3-coder-30b --config model_provider=local" }),
  );
  assert.deepEqual(errors, []);
});

test("a shell metacharacter in agent arguments is refused", () => {
  // These reach the agent's .cmd shim, so cmd.exe sees them.
  for (const args of [
    "--model x;calc",
    "--model a&calc",
    "--model a|calc",
    "--model a>b",
    "--model a%PATH%",
    "--model a!b",
    "--model a^b",
  ]) {
    const { errors } = validateAutoresearchConfig(config({ agentArgs: args }));
    assert.ok(errors.length > 0, `accepted shell syntax: ${args}`);
  }
});

test("the loop's own flags cannot be overridden from the field", () => {
  for (const args of ["-C /somewhere", "--cd /x", "--dir /x", "--cwd /x"]) {
    const { errors } = validateAutoresearchConfig(config({ agentArgs: args }));
    assert.ok(errors.length > 0, `accepted a loop-owned flag: ${args}`);
  }
});

test("the loop-owned flag list is the same on both sides", () => {
  // A field that can change the working directory or the brief can change what
  // the run does, so the two lists are asserted equal rather than kept in step
  // by hand.
  const backend = readText("../../backend/core/autoresearch/agent.py");
  const match = /FORBIDDEN_EXTRA_ARGS = \(([^)]*)\)/.exec(backend);
  assert.ok(match, "no FORBIDDEN_EXTRA_ARGS in the backend agent module");
  const backendFlags = [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...AGENT_OWNED_FLAGS].sort(), backendFlags.sort());
});

test("agent arguments round-trip through persistence", () => {
  const stored = config({ agentArgs: "--model local" });
  assert.deepEqual(migrateAutoresearchConfig(stored, DEFAULT_AUTORESEARCH_CONFIG), stored);
});

test("a multi-line agent arguments field is collapsed", () => {
  const merged = migrateAutoresearchConfig(
    { agentArgs: "--model\nlocal" } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.agentArgs, "--model local");
});

// --- the report phase ---
//
// On one GPU the report can only be written once the experiments let go of the
// card, so the loop offers to do it itself rather than asking the user to be
// watching for the moment a queue drains.

test("the report is off until it is asked for", () => {
  assert.equal(DEFAULT_AUTORESEARCH_CONFIG.reportAfterFinish, false);
  assert.deepEqual(validateAutoresearchConfig(config()).errors, []);
});

test("asking for a report with no model is refused", () => {
  const { errors } = validateAutoresearchConfig(
    config({ reportAfterFinish: true, reportModelId: "" }),
  );
  assert.ok(errors.some((error) => error.includes("Choose a model")));
});

test("asking for a report with a model is accepted", () => {
  const { errors } = validateAutoresearchConfig(
    config({ reportAfterFinish: true, reportModelId: "qwen/qwen3-8b" }),
  );
  assert.deepEqual(errors, []);
});

test("a blank report model is harmless while the box is unticked", () => {
  // A form nobody filled in is not a mistake, and the backend agrees.
  assert.deepEqual(
    validateAutoresearchConfig(config({ reportAfterFinish: false, reportModelId: "" }))
      .errors,
    [],
  );
});

test("the report choice round-trips through persistence", () => {
  const stored = config({
    reportAfterFinish: true,
    reportModelId: "autoresearch/003",
    reportSource: "autoresearch",
  });
  assert.deepEqual(migrateAutoresearchConfig(stored, DEFAULT_AUTORESEARCH_CONFIG), stored);
});

test("a report source from a newer build falls back to the default", () => {
  const merged = migrateAutoresearchConfig(
    { reportSource: "something-else" } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.reportSource, DEFAULT_AUTORESEARCH_CONFIG.reportSource);
});

test("a report flag of the wrong type falls back to the default", () => {
  const merged = migrateAutoresearchConfig(
    { reportAfterFinish: "yes" } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(
    merged.reportAfterFinish,
    DEFAULT_AUTORESEARCH_CONFIG.reportAfterFinish,
  );
});

// --- the estimate ---

test("the estimate grows linearly with the experiment count", () => {
  const one = estimateRunSeconds(config({ numExperiments: 1 }));
  const twelve = estimateRunSeconds(config({ numExperiments: 12 }));
  assert.equal(twelve, one * 12);
});

// --- persistence ---
//
// The stored configuration is what a returning user gets. Anything that changes
// silently here changes what a run will do without the user having decided it.

test("a stored configuration round-trips", () => {
  const stored = config({ numExperiments: 24, agent: "opencode", focus: "optimiser" });
  assert.deepEqual(migrateAutoresearchConfig(stored, DEFAULT_AUTORESEARCH_CONFIG), stored);
});

test("an absent stored configuration takes the defaults", () => {
  assert.deepEqual(migrateAutoresearchConfig(undefined, DEFAULT_AUTORESEARCH_CONFIG), DEFAULT_AUTORESEARCH_CONFIG);
});

test("an unknown key from an older or newer build is dropped", () => {
  const merged = migrateAutoresearchConfig(
    { numExperiments: 5, somethingRemoved: true } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.numExperiments, 5);
  assert.ok(!("somethingRemoved" in merged));
});

test("an out-of-bounds count is clamped rather than kept", () => {
  // A stored 500 is from a build with a wider ceiling. Honouring it would start
  // a run of a hundred real training passes on a load.
  const merged = migrateAutoresearchConfig(
    { numExperiments: 5000 } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.numExperiments, MAX_EXPERIMENTS);
});

test("an out-of-bounds timeout is clamped both ways", () => {
  const tooShort = migrateAutoresearchConfig(
    { agentTimeoutMinutes: 1 } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  const tooLong = migrateAutoresearchConfig(
    { agentTimeoutMinutes: 100000 } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(tooShort.agentTimeoutMinutes, MIN_TIMEOUT_MINUTES);
  assert.equal(tooLong.agentTimeoutMinutes, MAX_TIMEOUT_MINUTES);
});

test("a fractional stored number is rounded to a whole experiment", () => {
  const merged = migrateAutoresearchConfig(
    { numExperiments: 7.6 } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.numExperiments, 8);
  assert.ok(Number.isInteger(merged.numExperiments));
});

test("a non-numeric stored number falls back to the default", () => {
  const merged = migrateAutoresearchConfig(
    { numExperiments: "eight" } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.numExperiments, DEFAULT_AUTORESEARCH_CONFIG.numExperiments);
});

test("a stored flag of the wrong type falls back to the default", () => {
  const merged = migrateAutoresearchConfig(
    { skipPermissions: "yes", prepareData: 0 } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.skipPermissions, DEFAULT_AUTORESEARCH_CONFIG.skipPermissions);
  assert.equal(merged.prepareData, DEFAULT_AUTORESEARCH_CONFIG.prepareData);
});

test("a multi-line focus is flattened, because it is spliced into a prompt", () => {
  // The backend flattens it too, for the same reason: a newline in the middle
  // of the focus reads as the end of the instruction block.
  const merged = migrateAutoresearchConfig(
    { focus: "first line\nsecond line\n\nthird" } as never,
    DEFAULT_AUTORESEARCH_CONFIG,
  );
  assert.equal(merged.focus, "first line second line third");
  assert.ok(!merged.focus.includes("\n"));
});

// --- formatting ---
//
// val_bpb is six-decimal fixed point and the difference between two real attempts
// is often in the fourth place. Three decimals would show several neighbouring
// experiments as identical, which is the one thing this number is for.

test("a bpb score keeps the precision that distinguishes two attempts", () => {
  assert.equal(formatBpb(1.234567), "1.234567");
  assert.notEqual(formatBpb(1.23451), formatBpb(1.23457));
});

test("a missing bpb reads as absent rather than zero", () => {
  // Zero is a real score in this domain -- it is the project's own placeholder
  // for a crash -- so it must not be what "no score" renders as.
  assert.equal(formatBpb(null), "—");
  assert.equal(formatBpb(undefined), "—");
  assert.notEqual(formatBpb(null), formatBpb(0));
});

test("a delta carries its sign", () => {
  assert.equal(formatDelta(0.3), "+0.300000");
  assert.equal(formatDelta(-0.3), "-0.300000");
  assert.equal(formatDelta(0), "0");
});

test("a duration reads in the largest unit that says something", () => {
  assert.equal(formatDuration(45), "45s");
  assert.equal(formatDuration(60), "1m");
  assert.equal(formatDuration(90), "1m 30s");
  assert.equal(formatDuration(3600), "1h");
  assert.equal(formatDuration(5400), "1h 30m");
  // Nobody is waiting for a stopwatch: what matters is whether a run is five
  // minutes or all afternoon.
  assert.equal(formatDuration(3600 * 30), "1d 6h");
});

test("a missing duration reads as absent", () => {
  assert.equal(formatDuration(null), "—");
});

test("bytes are reported in binary units, which is what a VRAM figure uses", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1024), "1.0 KiB");
  assert.equal(formatBytes(7421 * 1024 * 1024), "7.2 GiB");
});
