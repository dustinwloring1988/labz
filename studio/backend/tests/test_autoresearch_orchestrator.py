# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The live numbers read out of a run log, and the estimate shown before a run.

The progress bar is the thing a user watches for hours, so the parsing behind it
is worth pinning. ``train.py`` prints its progress as one ``\\r``-overwritten
line, so every number below has to come out of that line rather than out of a
parsed stream, and a field name that is a prefix of another ("loss" inside
"lrm"-adjacent text, "step" inside "steps") is the obvious way to get it wrong.
"""

from __future__ import annotations

import pytest

from core.autoresearch import agent
from core.autoresearch.orchestrator import (
    AutoresearchConfig,
    _count_repeat,
    _explain_failure,
    _field,
    _float_from,
    _int_from,
    estimate_duration_seconds,
)

A_PROGRESS_LINE = (
    "step 00042 (14%) | loss: 4.123456 | lrm: 0.041200 | dt: 118.32ms | "
    "tok/sec: 4429.1 | mfu: 41.22 | epoch: 1 | remaining: 246s"
)


def test_every_number_comes_out_of_the_progress_line() -> None:
    assert _int_from(A_PROGRESS_LINE, "step") == 42
    assert _float_from(A_PROGRESS_LINE, "loss") == pytest.approx(4.123456)
    assert _float_from(A_PROGRESS_LINE, "lrm") == pytest.approx(0.0412)
    assert _float_from(A_PROGRESS_LINE, "dt") == pytest.approx(118.32)
    assert _int_from(A_PROGRESS_LINE, "tok/sec") == 4429
    assert _float_from(A_PROGRESS_LINE, "mfu") == pytest.approx(41.22)
    assert _int_from(A_PROGRESS_LINE, "epoch") == 1
    assert _int_from(A_PROGRESS_LINE, "remaining") == 246


def test_a_field_does_not_swallow_the_next_one() -> None:
    """The failure this guards against is silent: "lrm: 0.04" read as a loss."""
    line = "step 1 (0%) | loss: 9.9 | lrm: 0.5"
    assert _float_from(line, "loss") == pytest.approx(9.9)
    assert _float_from(line, "lrm") == pytest.approx(0.5)


def test_a_missing_field_is_absent_rather_than_zero() -> None:
    """Zero is a real value for step and loss, so absence has to be distinguishable."""
    assert _int_from("step 5 | loss: 1.0", "tok/sec") is None
    assert _float_from("step 5 | loss: 1.0", "mfu") is None


def test_a_field_with_no_value_is_absent() -> None:
    assert _field("step 5 | loss: | mfu: 1.0", "loss") is None
    assert _float_from("step 5 | loss: | mfu: 1.0", "loss") is None


def test_a_truncated_progress_line_still_parses() -> None:
    """The tail of run.log is whatever the last \\r left behind, mid-write."""
    assert _int_from("step 00007 (2%) | loss: 8.1", "step") == 7
    assert _float_from("step 00007 (2%) | loss: 8.1", "loss") == pytest.approx(8.1)


def test_the_estimate_is_linear_in_the_experiment_count() -> None:
    """autoresearch's budget is a fixed wall clock, so the estimate is honest."""
    one = estimate_duration_seconds(AutoresearchConfig(num_experiments = 1))
    twelve = estimate_duration_seconds(AutoresearchConfig(num_experiments = 12))
    assert twelve == one * 12
    # A touch over five minutes each, which is training plus evaluation.
    assert 5 * 60 <= one <= 10 * 60


def test_the_agent_timeout_reaches_the_command(monkeypatch) -> None:
    """What the user set in minutes is the budget the agent actually gets."""
    monkeypatch.setattr(agent, "resolve_executable", lambda _spec: ("/usr/bin/fake", "1.0"))
    command = agent.build_command(
        agent_key = "codex",
        prompt = "x",
        cwd = "/w",
        skip_permissions = True,
        timeout_seconds = 1800,
    )
    assert command.timeout_seconds == 1800


# --- explaining a failure ---
#
# When an experiment produces no score, the experiment row is the one place the
# user is guaranteed to look. "the run did not print a val_bpb" is true and
# useless when the agent never got as far as the training run, which is the most
# common way an unattended loop fails.


def test_an_agents_own_error_is_quoted_rather_than_the_absent_score() -> None:
    tail = [
        "OpenAI Codex v0.150.1",
        "workdir: C:\\checkout",
        'ERROR: {"type":"error","status":400,"error":{"type":"invalid_request_error",'
        '"message":"The \\"gpt-6-luna\\" model is not supported when using Codex with a '
        'ChatGPT account."}}',
    ]
    assert "not supported when using Codex" in _explain_failure(tail, 1)


def test_a_bare_error_line_is_quoted_as_written() -> None:
    tail = ["", "fatal: not a git repository", "shutting down"]
    assert _explain_failure(tail, 128) == "fatal: not a git repository"


def test_the_last_error_wins_over_earlier_ones() -> None:
    # A preamble can mention "error" in passing; the cause is the final one.
    tail = ["error: using cached metadata", "ERROR: disk full"]
    assert _explain_failure(tail, 1) == "ERROR: disk full"


def test_a_plain_banner_is_not_mistaken_for_a_cause() -> None:
    # Nothing here names a failure, so the honest answer is the exit code rather
    # than quoting "workdir: ..." as though it were a diagnosis.
    tail = ["OpenAI Codex v0.150.1", "workdir: C:\\checkout", "model: gpt-6-luna"]
    reason = _explain_failure(tail, 1)
    assert "exited with code 1" in reason
    assert "console" in reason


def test_a_silent_failure_falls_back_to_the_absent_score() -> None:
    # Nothing on stderr and a clean exit: the run genuinely started and produced
    # no number, which is exactly what program.md's rule describes.
    assert "val_bpb" in _explain_failure([], 0)


def test_a_very_long_error_line_is_trimmed() -> None:
    tail = ["error: " + "x" * 500]
    reason = _explain_failure(tail, 1)
    assert len(reason) <= 300
    assert reason.endswith("…")


# --- repeated crashes are a setup problem, not a result ---


def test_one_crash_is_not_enough_to_stop() -> None:
    # A single failure can be a genuine one-off. Stopping here would throw away a
    # loop that was about to learn something.
    repeats, stop = _count_repeat(None, "CUDA out of memory", 0)
    assert (repeats, stop) == (1, False)


def test_the_same_crash_twice_stops_the_run() -> None:
    # This is the case that cost thirteen experiments: one refused model, repeated
    # verbatim, at about eleven minutes apiece. The second one settles it -- the
    # machine, the brief and the agent have all been ruled out by elimination.
    repeats, stop = _count_repeat("model is not supported", "model is not supported", 1)
    assert (repeats, stop) == (2, True)


def test_two_different_crashes_are_both_results_and_neither_stops_the_run() -> None:
    # The whole reason this is separate from max_consecutive_failures: different
    # failures are information, and a loop that gave up on two of them could not
    # tell a bad idea from a bad machine.
    repeats, stop = _count_repeat("CUDA out of memory", "the agent exited with code 1", 1)
    assert (repeats, stop) == (1, False)


def test_a_success_between_two_crashes_resets_the_count() -> None:
    # Two identical crashes do stop the run.
    repeats, stop = _count_repeat(None, "same reason", 0)
    repeats, stop = _count_repeat("same reason", "same reason", repeats)
    assert (repeats, stop) == (2, True)
    # And a success in between starts the count over, so a crash seen earlier in
    # the run is not held against a later one. The loop clears both on a success,
    # which arrives here as a fresh previous and a zeroed count.
    fresh, stop = _count_repeat(None, "same reason", 0)
    assert (fresh, stop) == (1, False)


def test_an_unattributable_crash_is_never_treated_as_a_repeat() -> None:
    # Two failures with no stated reason might be two different problems wearing
    # the same silence, and stopping on that would hide the second one.
    repeats, stop = _count_repeat(None, None, 0)
    assert (repeats, stop) == (1, False)
    repeats, stop = _count_repeat(None, None, repeats)
    assert (repeats, stop) == (1, False)
    assert stop is False


def test_the_limit_is_two_not_three() -> None:
    # Pinned because it is a judgement call about how much a user pays to be told.
    from core.autoresearch.orchestrator import REPEATED_CRASH_LIMIT
    assert REPEATED_CRASH_LIMIT == 2
