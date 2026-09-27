# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A judged run: the judge's verdict is reported, the exact one is kept, and the
difference is counted.

The three things worth protecting are that the exact verdict is never thrown
away, that a judge failure leaves the run's real scores alone, and that a run
which declined to judge the categorical benchmarks says so rather than implying
it did.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE", "1")

import pytest

from core.benchmarks.orchestrator import BenchmarkConfig, BenchmarkRun, RunState, ScoreState


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))


def _run(tmp_path, **config_overrides) -> BenchmarkRun:
    defaults = {
        "model_id": "me/m",
        "model_label": "my model",
        "model_path": "C:/m",
        "benchmarks": ["gsm8k", "mmlu"],
        "judge_model_id": "me/judge",
        "judge_model_label": "my judge",
        "judge_model_path": "C:/j",
    }
    defaults.update(config_overrides)
    run = BenchmarkRun(run_id = "r1", config = BenchmarkConfig(**defaults))
    run.run_dir.mkdir(parents = True, exist_ok = True)
    # nanochat's registry is what says gsm8k is generative, and these tests run
    # against an isolated home that has no nanochat in it. The kinds are therefore
    # set from the registry's own values rather than read through it, so the test
    # exercises the run's wiring rather than the catalogue's availability. That
    # the registry really does classify them this way is covered by the catalogue
    # tests and by the end-to-end run.
    run._score_state("gsm8k").kind = "generative"
    run._score_state("mmlu").kind = "categorical"
    return run


def _write_judge_results(run: BenchmarkRun, payload: dict) -> None:
    (run.run_dir / "judge.json").write_text(json.dumps(payload), encoding = "utf-8")


def _write_judged_items(run: BenchmarkRun, key: str, rows: list[dict]) -> None:
    run._judged_dir.mkdir(parents = True, exist_ok = True)
    (run._judged_dir / f"{key}.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding = "utf-8"
    )


def _graded(run: BenchmarkRun, key: str, accuracy: float, correct: int, total: int) -> None:
    score = run._score_state(key)
    score.status = "complete"
    score.accuracy = accuracy
    score.correct = correct
    score.total = total
    score.baseline = 0.25 if key != "gsm8k" else 0.0
    score.scoring_mode = "generated"


# --- which benchmarks get judged ---


def test_a_judge_runs_only_for_the_generative_benchmarks(tmp_path):
    run = _run(tmp_path)
    # gsm8k is generative in nanochat's table, mmlu categorical.
    assert run._judge_wanted("gsm8k") is True
    assert run._judge_wanted("mmlu") is False


def test_no_judge_configured_means_the_pass_does_nothing(tmp_path):
    run = _run(tmp_path, judge_model_path = None)
    # Not an error and not a crash: the run simply has no second opinion.
    run._pass_judge()
    assert run.state.phase != "judge"


# --- the merge ---


def test_the_judged_verdict_becomes_the_reported_score(tmp_path):
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.0, correct = 0, total = 10)
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_correct": 7,
                "judge_judged": 10,
                "judge_accuracy": 0.7,
                "judge_model": "my judge",
            }
        },
    )
    run._absorb_judge_results()

    score = run._score_state("gsm8k")
    assert score.accuracy == pytest.approx(0.7)
    assert score.correct == 7
    assert score.total == 10
    assert score.scoring_mode == "judged"


def test_the_exact_verdict_is_kept_beside_the_judged_one(tmp_path):
    # The whole reason to run a judge is that the two can differ, so the exact
    # number is kept on the same row. Collapsing to one column here would leave
    # only the opinion and throw away the definition.
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.1, correct = 1, total = 10)
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_correct": 7,
                "judge_judged": 10,
                "judge_accuracy": 0.7,
                "judge_model": "my judge",
            }
        },
    )
    run._absorb_judge_results()

    score = run._score_state("gsm8k")
    assert score.judge_accuracy == pytest.approx(0.7)
    assert score.exact_accuracy == pytest.approx(0.1)
    assert score.judge_model == "my judge"


def test_an_unjudged_score_keeps_no_judge_fields(tmp_path):
    # A categorical score from a run that had a judge configured must not grow
    # judge fields, or the UI would offer to show a second opinion that does not
    # exist for it.
    run = _run(tmp_path)
    _graded(run, "mmlu", accuracy = 0.5, correct = 5, total = 10)
    _write_judge_results(
        run, {"gsm8k": {"judge_correct": 1, "judge_judged": 1, "judge_accuracy": 1.0}}
    )
    run._absorb_judge_results()

    score = run._score_state("mmlu")
    assert score.judge_accuracy is None
    assert score.exact_accuracy is None
    assert score.judge_model is None
    assert score.scoring_mode == "generated"


def test_a_judge_that_read_nothing_leaves_the_exact_score_alone(tmp_path):
    # Every verdict unreadable. The run keeps the number it can defend rather
    # than dropping to zero because the judge formatted badly.
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.4, correct = 4, total = 10)
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_correct": 0,
                "judge_judged": 0,
                "judge_accuracy": None,
                "judge_unreadable": 10,
            }
        },
    )
    run._absorb_judge_results()

    score = run._score_state("gsm8k")
    assert score.accuracy == pytest.approx(0.4)
    assert score.judge_unreadable == 10
    assert score.judge_judged == 0
    # And the run says why, rather than presenting the exact score as judged.
    assert score.error is not None
    assert score.scoring_mode == "generated"


def test_no_judge_results_file_changes_nothing(tmp_path):
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.4, correct = 4, total = 10)
    run._absorb_judge_results()
    assert run._score_state("gsm8k").accuracy == pytest.approx(0.4)
    assert run._score_state("gsm8k").judge_accuracy is None


def test_unparseable_judge_output_changes_nothing(tmp_path):
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.4, correct = 4, total = 10)
    (run.run_dir / "judge.json").write_text("{not json", encoding = "utf-8")
    run._absorb_judge_results()
    assert run._score_state("gsm8k").accuracy == pytest.approx(0.4)


# --- disagreements ---


def test_disagreements_are_counted_and_the_way_is_recorded(tmp_path):
    # "The judge and exact matching disagree" is a statement about the benchmark.
    # Losing the direction would leave only the chosen number, which is the one
    # number that is not evidence of anything.
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.0, correct = 0, total = 4)
    run._exact_verdict = {
        ("gsm8k", 0): False,
        ("gsm8k", 1): False,
        ("gsm8k", 2): True,
        ("gsm8k", 3): True,
    }
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_correct": 3,
                "judge_judged": 4,
                "judge_accuracy": 0.75,
                "judge_model": "J",
            }
        },
    )
    _write_judged_items(
        run,
        "gsm8k",
        [
            {"index": 0, "verdict": {"correct": True, "parsed": "json"}},
            {"index": 1, "verdict": {"correct": True, "parsed": "json"}},
            {"index": 2, "verdict": {"correct": False, "parsed": "json"}},
            {"index": 3, "verdict": {"correct": True, "parsed": "json"}},
        ],
    )
    run._absorb_judge_results()

    score = run._score_state("gsm8k")
    assert score.disagreements == 3
    # Two the judge accepted that exact marking rejected, one it rejected.
    assert score.judge_raised == 2


def test_an_answer_the_judge_skipped_is_not_a_disagreement(tmp_path):
    # The scoring pass never produced an answer for one item, so there is no
    # exact verdict for the judge to differ from. Counting it would inflate the
    # disagreement rate with items where the judge had nothing to rule on.
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.0, correct = 0, total = 2)
    run._exact_verdict = {("gsm8k", 0): False}
    _write_judge_results(
        run, {"gsm8k": {"judge_correct": 1, "judge_judged": 1, "judge_accuracy": 1.0}}
    )
    _write_judged_items(
        run,
        "gsm8k",
        [
            {"index": 0, "verdict": {"correct": True, "parsed": "json"}},
            {"index": 1, "verdict": None, "skipped": "no answer"},
        ],
    )
    run._absorb_judge_results()

    score = run._score_state("gsm8k")
    # One real verdict, and it disagreed with the exact one. The skipped item is
    # not a second disagreement.
    assert score.disagreements == 1
    assert score.judge_raised == 1
    # And the skipped item is not counted as answered either.
    assert score.judge_judged == 1


def test_agreement_is_not_counted_as_a_disagreement(tmp_path):
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.5, correct = 2, total = 2)
    run._exact_verdict = {("gsm8k", 0): True, ("gsm8k", 1): False}
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_correct": 1,
                "judge_judged": 2,
                "judge_accuracy": 0.5,
                "judge_model": "J",
            }
        },
    )
    _write_judged_items(
        run,
        "gsm8k",
        [
            {"index": 0, "verdict": {"correct": True, "parsed": "json"}},
            {"index": 1, "verdict": {"correct": False, "parsed": "json"}},
        ],
    )
    run._absorb_judge_results()
    assert run._score_state("gsm8k").disagreements == 0


def test_disagreements_are_zero_when_no_per_item_verdicts_were_kept(tmp_path):
    # The grader may not have written them; the totals still stand and the
    # breakdown is simply unavailable rather than invented.
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.1, correct = 1, total = 10)
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_correct": 7,
                "judge_judged": 10,
                "judge_accuracy": 0.7,
                "judge_model": "J",
            }
        },
    )
    run._absorb_judge_results()
    score = run._score_state("gsm8k")
    assert score.accuracy == pytest.approx(0.7)
    assert score.disagreements == 0


# --- the run reports the judge ---


def test_the_run_names_its_judge(tmp_path):
    run = _run(tmp_path)
    payload = run.state.to_dict()
    assert payload["judge_model_id"] == "me/judge"
    assert payload["judge_model_label"] == "my judge"


def test_a_run_with_no_judge_names_none(tmp_path):
    run = _run(tmp_path, judge_model_path = None, judge_model_id = None, judge_model_label = None)
    payload = run.state.to_dict()
    assert payload["judge_model_id"] is None
    assert payload["judge_model_label"] is None


def test_the_judge_reaches_the_client_on_every_score(tmp_path):
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.0, correct = 0, total = 4)
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_correct": 3,
                "judge_judged": 4,
                "judge_accuracy": 0.75,
                "judge_model": "J",
                "judge_unreadable": 0,
            }
        },
    )
    run._absorb_judge_results()
    row = run.state.to_dict()["scores"][0]
    for field in (
        "judge_model",
        "exact_accuracy",
        "judge_accuracy",
        "judge_judged",
        "judge_unreadable",
        "disagreements",
        "judge_raised",
    ):
        assert field in row, field
    assert row["judge_accuracy"] == pytest.approx(0.75)
    assert row["exact_accuracy"] == pytest.approx(0.0)


# --- ordering ---


def test_judged_and_exact_rows_are_never_ranked_as_one_group():
    # Two different measurements of the same benchmark. A judged row scoring higher
    # does not mean the model is better, so the board must not put them in one
    # column where the higher number wins.
    from models.benchmarks import LeaderboardRow
    from routes.benchmarks import _rank_key

    exact = LeaderboardRow(
        run_id = "a", model_label = "exact", model_id = "a", scoring_mode = "logits", composite = 0.2
    )
    judged = LeaderboardRow(
        run_id = "b",
        model_label = "judged",
        model_id = "b",
        scoring_mode = "judged",
        composite = 0.9,
    )
    generated = LeaderboardRow(
        run_id = "c", model_label = "gen", model_id = "c", scoring_mode = "generated", composite = 0.5
    )
    # Sorted by the board's own key, the exact row leads despite scoring lowest.
    ordered = sorted([judged, generated, exact], key = _rank_key)
    assert [row.run_id for row in ordered] == ["a", "c", "b"]


# --- the item cache and the problem cap ---


def test_the_cache_is_keyed_on_the_cap_as_well_as_the_dataset():
    """The cached file is the *truncated* item list, so the cap is part of its identity.

    Keyed on the dataset alone, a run capped at 8 problems found and reused a file
    written for 400 and scored all 400 -- reporting a total the user never asked
    for, and marking the run "sampled" for the wrong reason. Nothing in the stamp
    had changed, so nothing looked wrong.
    """
    from core.benchmarks.item_cache import item_cache_stamp

    # Benchmarks the materializer knows, so only the cap differs between calls.
    at_8 = item_cache_stamp("mmlu", 8)
    at_400 = item_cache_stamp("mmlu", 400)
    uncapped = item_cache_stamp("mmlu", None)

    assert at_8 != at_400, "two caps must not share a cache entry"
    assert at_8 != uncapped, "a cap and no cap must not share a cache entry"
    # And the same cap is the same entry, or nothing would ever be reused.
    assert at_8 == item_cache_stamp("mmlu", 8)
    # Different benchmarks never collide.
    assert at_8 != item_cache_stamp("gsm8k", 8)
    # An unknown benchmark still discriminates on the cap rather than collapsing
    # to one shared entry.
    assert item_cache_stamp("not-a-benchmark", 8) != item_cache_stamp("not-a-benchmark", 400)


# --- when a judge's verdicts are too incomplete to report ---


def test_a_judge_that_read_little_is_still_reported_with_its_coverage(tmp_path):
    """One readable verdict out of eight is reported, and cannot outrank exact.

    The score is worth seeing -- a judge that read one answer has still measured
    something -- but a leaderboard cannot tell that row from a model that answered
    all eight unless the coverage travels with it. So the judged figure is
    reported, the counts are kept, the unreadable answers are not quietly counted
    as failures, and the row ranks below every exact measurement.
    """
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.0, correct = 0, total = 8)
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_accuracy": 1.0,
                "judge_correct": 1,
                "judge_judged": 1,
                "judge_unreadable": 7,
                "judge_model": "judge/1b",
            }
        },
    )
    run._absorb_judge_results()

    score = run._score_state("gsm8k")
    assert score.accuracy == 1.0, "a judge that read an answer reports what it found"
    assert score.scoring_mode == "judged"
    # The denominator is what the judge actually read, so accuracy stays equal to
    # correct / total and the seven unread answers are not scored as wrong.
    assert (score.correct, score.total) == (1, 1)
    assert score.judge_judged == 1 and score.judge_unreadable == 7, (
        "coverage is recorded, so the row cannot be read as a clean sweep"
    )
    assert score.exact_accuracy == 0.0, "the exact grade is still beside it"
    # And the ranking is what stops a partial judge from looking like the best
    # model on the board: every judged row sorts below every exact one, whatever
    # its coverage.
    from models.benchmarks import LeaderboardRow
    from routes.benchmarks import _rank_key

    def row(mode, composite):
        return LeaderboardRow(
            run_id = mode,
            model_label = mode,
            model_id = mode,
            scoring_mode = mode,
            composite = composite,
        )

    # The partial judge's perfect score, against a real exact measurement.
    ordered = sorted([row("judged", 1.0), row("logits", 0.9)], key = _rank_key)
    assert [r.scoring_mode for r in ordered] == ["logits", "judged"]


def test_the_reported_score_is_an_average_over_the_verdicts_the_judge_read(tmp_path):
    """`accuracy` stays equal to `correct / total`, and unread answers are not failures.

    Six readable verdicts out of eight is a real measurement of part of the suite,
    so it is reported -- but the two answers the judge could not parse are not
    counted as wrong answers, because the judge never reached a verdict on them.
    Averaging them in as failures would report a score for a question that was
    never graded, and would make `accuracy` disagree with `correct / total`.
    """
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.0, correct = 0, total = 8)
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_accuracy": 0.5,
                "judge_correct": 3,
                "judge_judged": 6,
                "judge_unreadable": 2,
                "judge_model": "judge/1b",
            }
        },
    )
    run._absorb_judge_results()

    score = run._score_state("gsm8k")
    assert score.accuracy == 0.5
    assert score.scoring_mode == "judged"
    assert (score.correct, score.total) == (3, 6), "the average is over what was read"
    assert score.accuracy == pytest.approx(score.correct / score.total)
    assert score.judge_judged == 6 and score.judge_unreadable == 2, (
        "the gap between the two is what the reader needs, and it is kept"
    )


def test_a_judged_run_reports_itself_as_judged(tmp_path):
    """The run's mode has to follow its scores, or the board files it wrongly.

    The grade pass sets the run's mode to `generated` and the judge pass then
    changes the score to `judged` without touching it. The leaderboard groups and
    ranks on the *run's* mode, so a stale one files a judged run among the exact
    ones and the separate judged group never appears at all.
    """
    run = _run(tmp_path)
    _graded(run, "gsm8k", accuracy = 0.0, correct = 0, total = 4)
    # What the grade pass leaves behind: every generative answer read, so the run
    # is `generated` before any judge has looked at it.
    run.state.scoring_mode = "generated"
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_accuracy": 0.25,
                "judge_correct": 1,
                "judge_judged": 4,
                "judge_unreadable": 0,
                "judge_model": "judge/1b",
            }
        },
    )
    run._absorb_judge_results()
    assert run.state.scoring_mode == "judged"


def test_a_run_mixing_a_judged_and_an_exact_benchmark_is_mixed(tmp_path):
    # Not one method and not the other, so it is not filed under either. This is
    # also where the board already ranks it: below every single-method row.
    run = _run(tmp_path)
    _graded(run, "mmlu", accuracy = 0.5, correct = 5, total = 10)
    _graded(run, "gsm8k", accuracy = 0.0, correct = 0, total = 4)
    _write_judge_results(
        run,
        {
            "gsm8k": {
                "judge_accuracy": 0.5,
                "judge_correct": 2,
                "judge_judged": 4,
                "judge_unreadable": 0,
                "judge_model": "judge/1b",
            }
        },
    )
    run._absorb_judge_results()
    assert run.state.scoring_mode == "mixed"


def test_a_row_with_no_mode_sorts_after_the_known_ones():
    # `scoring_mode` is nullable, and a run that never recorded one is reachable
    # (an older stored row). It must not sort as though it were the exact
    # measurement just because it has no mode to disqualify it.
    from models.benchmarks import LeaderboardRow
    from routes.benchmarks import _rank_key

    unknown = LeaderboardRow(
        run_id = "b", model_label = "b", model_id = "b", scoring_mode = None, composite = 1.0
    )
    known = LeaderboardRow(
        run_id = "a", model_label = "a", model_id = "a", scoring_mode = "judged", composite = 0.0
    )
    ordered = sorted([unknown, known], key = _rank_key)
    assert [row.run_id for row in ordered] == ["a", "b"]


def test_a_mode_the_board_has_never_heard_of_sorts_last():
    # Unreachable through validation -- ScoringMode is a Literal, so a mode this
    # board does not know cannot be constructed. The board still defends against
    # one, because a mode is a value read back out of storage that a future
    # backend may have written, and a new measurement should land at the end of the
    # board rather than interleaved with the ones a reader already trusts.
    from models.benchmarks import LeaderboardRow
    from routes.benchmarks import _MODE_ORDER, _rank_key

    assert _MODE_ORDER.get("something-new", 3) == 3
    future = LeaderboardRow.model_construct(
        run_id = "b",
        model_label = "b",
        model_id = "b",
        scoring_mode = "something-new",
        composite = 1.0,
    )
    known = LeaderboardRow(
        run_id = "a", model_label = "a", model_id = "a", scoring_mode = "judged", composite = 0.0
    )
    ordered = sorted([future, known], key = _rank_key)
    assert [row.run_id for row in ordered] == ["a", "b"]
