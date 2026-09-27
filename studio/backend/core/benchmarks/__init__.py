# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark benchmarking integration: catalogue, orchestration, and the run passes.

Why the passes are split across two interpreters
------------------------------------------------
nanochat's benchmarks are defined in its own checkout (``tasks/*.py``) and are
executed in its own virtualenv, which carries a different pinned torch. This
app's models are ordinary HF checkpoints, and the environment that can load one
(unsloth, transformers, peft) is this process's, not nanochat's. The nanochat
virtualenv has no ``transformers`` and no ``peft`` at all, so a third-party model
simply cannot be loaded there.

So the work is cut along the dependency line rather than around it:

    materialize   nanochat venv   tasks -> items on disk      needs pyarrow only
    score         backend venv    items -> predictions        needs torch + unsloth
    grade         nanochat venv   predictions -> accuracy     needs pyarrow only

nanochat owns the first and the last: the datasets, the prompt rendering, and
the grading itself. This package owns the middle, and the plumbing around all
three. ``task.evaluate()`` is called in nanochat's own interpreter against
nanochat's own ``Task`` objects, so the answer-normalisation rules (GSM8K's
``#### N``, HumanEval's sandboxed execution, the multiple-choice letter
assertions) are never restated here and cannot drift.
"""
