# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""nanochat integration: environment management, presets, and run orchestration.

nanochat is a separate project with its own pinned torch, so it never runs inside
this process. Every stage is a child process in its own virtualenv that reports
progress as JSONL (see ``environment``), and this package turns that into the
run state the API and UI consume (see ``orchestrator``).
"""
