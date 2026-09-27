// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Resolves the `@/` specifiers that benchmarks-api.ts imports. Auth is stubbed
// because it is not what these tests are about and it is not safely importable
// outside an app; the other two are real, because the payload builder's neighbours
// (the SSE framing and the FastAPI error reader) are what a parser bug would show
// up in.
import { existsSync } from "node:fs";

const AUTH_STUB = new URL("./benchmarks-auth-stub.mjs", import.meta.url).href;

// `../..` because this file lives in tests/helpers/, so the frontend's src/ is two
// levels up rather than one.
const ROOT = new URL("../../src/", import.meta.url);

// The source imports the alias without an extension, which Vite resolves and node
// does not. Rather than change the imports to suit the test runner, the runner is
// taught the same three candidates Vite tries, in Vite's order.
const CANDIDATES = ["", ".ts", ".tsx", "/index.ts", "/index.tsx"];

function firstExisting(base) {
  for (const suffix of CANDIDATES) {
    const candidate = new URL(base.href + suffix);
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

export function resolve(specifier, context, next) {
  if (specifier === "@/features/auth") {
    return next(AUTH_STUB, context);
  }
  if (specifier.startsWith("@/")) {
    const resolved = firstExisting(new URL(specifier.slice(2), ROOT));
    return next((resolved ?? new URL(specifier.slice(2), ROOT)).href, context);
  }
  return next(specifier, context);
}
