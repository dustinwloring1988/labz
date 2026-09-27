# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The report prompt, and what the saved file is allowed to look like.

The prompt is worth testing because it is the one place where a wrong assumption
becomes a plausible-sounding document. A model handed a table of numbers will
happily write a confident report about it, so the tests here check the three
things that keep that from being misleading:

  - the headline numbers are stated as facts, computed from the ledger, rather
    than left for the model to work out (and get wrong)
  - a crash is described as a crash, not as a score of zero
  - the model reporting on its own training run is told that is what it is
"""
from __future__ import annotations

from core.autoresearch import results
from core.autoresearch.report import (
    build_report_prompt,
    save_report,
    subject_note_for,
)


def _row(commit: str, bpb: str, memory: str, status: str, description: str) -> str:
    """One ledger line.

    Built from an explicit join rather than written inline with escapes. A
    literal tab escape in a string that then continues with the letter t is read
    as the escape plus the rest of the word, which silently eats a letter in the
    fixture -- and a fixture saying "rained for longer" instead of "trained"
    fails for a reason that has nothing to do with the code under test.
    """
    return "\t".join([commit, bpb, memory, status, description]) + "\n"


def _header() -> str:
    return "\t".join(results.RESULTS_COLUMNS) + "\n"


def _summary(tmp_path, body: str) -> results.ResultsSummary:
    path = tmp_path / results.RESULTS_FILENAME
    path.write_text(body, encoding="utf-8", newline="\n")
    return results.read_results(tmp_path)


THREE_EXPERIMENTS = _header() + (
    _row("aaa", "1.500000", "3.5000", "keep", "lowered the matrix learning rate")
    + _row("bbb", "1.200000", "3.6000", "keep", "squared-ReLU instead of a gated MLP")
    + _row("ccc", "1.400000", "3.5500", "discard", "extended the warmup")
)


def test_the_prompt_states_the_headline_numbers_rather_than_leaving_them_to_the_model(
    tmp_path,
) -> None:
    summary = _summary(tmp_path, THREE_EXPERIMENTS)
    prompt = build_report_prompt(summary)

    assert "Experiments run: 3" in prompt
    assert "Baseline val_bpb: 1.500000" in prompt
    assert "Best val_bpb: 1.200000" in prompt
    # Given as an absolute change and a percentage: "it got better" is not
    # actionable, and a model asked to divide a six-digit number by another one
    # will sometimes do it wrong.
    assert "0.300000 val_bpb (20.00%)" in prompt
    assert "Winning change: squared-ReLU instead of a gated MLP" in prompt


def test_a_crash_is_named_as_a_crash_and_never_as_a_zero(tmp_path) -> None:
    summary = _summary(
        tmp_path,
        _header()
        + _row("aaa", "1.500000", "3.5", "keep", "the baseline")
        + _row("bbb", "0.000000", "0.0", "crash", "CUDA OOM on step 12"),
    )
    prompt = build_report_prompt(summary)

    assert "crashed: 1" in prompt
    assert "Best val_bpb: 1.500000" in prompt
    # The crash row carries no score cell at all, so the table cannot be misread
    # as a run that scored zero. Zero sorts as a perfect result.
    assert "| 2 | - |" in prompt
    assert "CUDA OOM on step 12" in prompt


def test_the_ledger_is_a_table_and_keeps_every_row(tmp_path) -> None:
    summary = _summary(tmp_path, THREE_EXPERIMENTS)
    prompt = build_report_prompt(summary)
    assert "| # | val_bpb | memory_gb | status | description |" in prompt
    for description in (
        "lowered the matrix learning rate",
        "squared-ReLU instead of a gated MLP",
        "extended the warmup",
    ):
        assert description in prompt


def test_a_pipe_in_an_agent_description_cannot_break_the_table(tmp_path) -> None:
    summary = _summary(
        tmp_path,
        _header() + _row("aaa", "1.500000", "3.5", "keep", "set lr | depth"),
    )
    prompt = build_report_prompt(summary)
    # One unescaped pipe here would shift every later column in the model's
    # reading, and a shifted table produces a confidently wrong report.
    assert r"set lr \| depth" in prompt


def test_a_long_search_is_truncated_and_says_so(tmp_path) -> None:
    body = _header() + "".join(
        _row(f"{index:04x}", f"{2.0 - index * 0.001:.6f}", "3.5", "keep", f"attempt {index}")
        for index in range(1, 121)
    )
    summary = _summary(tmp_path, body)
    prompt = build_report_prompt(summary)

    assert "experiments in the middle of the search are omitted" in prompt
    # The frontier itself must survive the truncation, or the report is about a
    # search whose best result is not in it.
    assert "Best val_bpb:" in prompt
    assert "1.999000" in prompt


def test_a_checkpoint_writing_its_own_report_is_told_what_it_is() -> None:
    """A 50M-parameter model writing about its own training run, unlabelled,
    reads as an assessment of that run when it is nothing of the kind."""
    note = subject_note_for(None)
    assert "you are one of the models this search produced" in note
    assert "no ability to check it" in note


def test_a_catalog_model_is_not_given_that_note(tmp_path) -> None:
    summary = _summary(tmp_path, THREE_EXPERIMENTS)
    prompt = build_report_prompt(summary, subject_note="")
    assert "you are one of the models this search produced" not in prompt


def test_the_prompt_asks_for_the_failure_pattern_not_just_the_score(tmp_path) -> None:
    summary = _summary(tmp_path, THREE_EXPERIMENTS)
    prompt = build_report_prompt(summary)
    # The most useful section and the easiest to skip: a pattern in the discards
    # is worth more than the frontier itself.
    assert "What did not work" in prompt
    assert "Crashes" in prompt
    assert "What to try next" in prompt


def test_a_whole_document_code_fence_is_stripped(tmp_path) -> None:
    """A model asked for Markdown will sometimes fence the whole answer, and a
    report that opens with a fence is not a report."""
    fenced = "```markdown\n# The search\n\nIt found a thing.\n```"
    saved = save_report(tmp_path, "run_1", fenced)
    text = (tmp_path / "reports" / "run_1.md").read_text(encoding="utf-8")
    assert text.startswith("# The search")
    assert "```" not in text
    assert saved.endswith("run_1.md")


def test_an_unfenced_report_is_left_exactly_as_it_arrived(tmp_path) -> None:
    body = (
        "# The search\n\nA fenced code block *inside* the report is content.\n\n"
        "```python\nx = 1\n```\n"
    )
    save_report(tmp_path, "run_1", body)
    text = (tmp_path / "reports" / "run_1.md").read_text(encoding="utf-8")
    assert text == body.rstrip() + "\n"
