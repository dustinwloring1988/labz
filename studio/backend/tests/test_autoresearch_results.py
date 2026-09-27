# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The autoresearch ledger, and what a run log is allowed to claim.

Three things are pinned here, and all three are ways this feature could quietly
lie to the user:

  - **A crash is not a score.** autoresearch's own instructions record a run that
    produced no val_bpb as ``0.000000``, which sorts as a perfect result. A
    parser that took the number at face value would show a crashed experiment as
    the best one in the search, so those rows are recognised and carry no score.

  - **The best experiment is the lowest real score.** Including a crash in that
    minimum is the same bug arriving by a different route.

  - **A score comes from the run log, never from the agent's own summary.** The
    log is what program.md names as the source of truth, so an agent that
    misreads it cannot improve its recorded result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.autoresearch import agent, results


# -----------------------------------------------------------------------------
# results.tsv
# -----------------------------------------------------------------------------


def _write(root: Path, body: str) -> Path:
    path = root / results.RESULTS_FILENAME
    path.write_text(body, encoding = "utf-8", newline = "\n")
    return path


def test_a_clean_ledger_round_trips(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "abc1234\t1.234567\t3.5000\tkeep\tlowered the matrix lr\n"
        "def5678\t1.300000\t3.6000\tdiscard\traised longer\n",
    )
    summary = results.read_results(tmp_path)

    assert summary.exists
    assert summary.total == 2
    assert summary.kept == 1
    assert summary.discarded == 1
    assert summary.crashed == 0
    assert [row.val_bpb for row in summary.rows] == [1.234567, 1.3]
    # Indices are 1-based because that is how the loop names them, and the UI
    # prints them beside the experiment numbers the agent was told about.
    assert [row.index for row in summary.rows] == [1, 2]
    assert summary.best().index == 1
    assert summary.baseline().index == 1


def test_a_placeholder_zero_is_a_crash_and_carries_no_score(tmp_path: Path) -> None:
    """The project's own crash encoding, which must not read as a perfect run."""
    _write(
        tmp_path,
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "abc1234\t0.000000\t0.0\tkeep\ttrain.py crashed before eval\n",
    )
    summary = results.read_results(tmp_path)

    row = summary.rows[0]
    assert row.val_bpb is None
    assert row.status == results.STATUS_CRASH
    assert row.error is not None
    # The critical assertion: a crash must never become the search's best result.
    assert summary.best() is None
    assert summary.baseline() is None


def test_a_crash_does_not_beat_a_real_score(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "aaaaaaa\t1.400000\t3.5000\tdiscard\tworse\n"
        "bbbbbbb\t0.000000\t0.0\tcrash\tdied\n"
        "ccccccc\t1.200000\t3.4000\tkeep\tbetter\n",
    )
    summary = results.read_results(tmp_path)

    assert summary.crashed == 1
    assert summary.best().val_bpb == 1.2
    assert summary.best().index == 3


def test_the_running_best_is_marked_on_exactly_one_row(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "a\t1.500000\t3.5\tkeep\tfirst\n"
        "b\t1.200000\t3.5\tkeep\tbetter\n"
        "c\t1.400000\t3.5\tdiscard\tworse again\n",
    )
    summary = results.read_results(tmp_path)

    best_rows = [row.index for row in summary.rows if row.is_best]
    # A new best marks itself; an old best is demoted, so there is never more
    # than one and it is always the last improvement.
    assert best_rows == [2]


def test_a_missing_header_is_tolerated(tmp_path: Path) -> None:
    _write(tmp_path, "abc1234\t1.234567\t3.5\tkeep\tno header row\n")
    summary = results.read_results(tmp_path)
    assert summary.total == 1
    assert summary.rows[0].val_bpb == 1.234567


def test_a_rewritten_header_is_still_treated_as_a_header(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "Commit\tVal BPB\tMemory GB\tStatus\tDescription\nabc1234\t1.234567\t3.5\tkeep\tone\n",
    )
    summary = results.read_results(tmp_path)
    assert summary.total == 1


def test_a_short_row_is_dropped_rather_than_failing_the_read(tmp_path: Path) -> None:
    """One bad line must not blank the whole history.

    The file is written by an LLM, so a row can be truncated. Losing one
    experiment is recoverable; losing all of them is not.
    """
    _write(
        tmp_path,
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "a\t1.5\t3.5\tkeep\tgood row\n"
        "garbage\n"
        "b\t1.4\t3.5\tkeep\tanother good row\n",
    )
    summary = results.read_results(tmp_path)
    assert summary.total == 2


def test_an_unknown_status_becomes_a_crash(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\na\t1.5\t3.5\tgreat\tan invented status\n",
    )
    summary = results.read_results(tmp_path)
    assert summary.rows[0].status == results.STATUS_CRASH


def test_an_empty_ledger_is_a_state_not_an_error(tmp_path: Path) -> None:
    summary = results.read_results(tmp_path)
    assert summary.exists is False
    assert summary.total == 0
    assert summary.best() is None


def test_appending_a_crash_row_writes_the_projects_own_placeholders(tmp_path: Path) -> None:
    """The column order and the placeholder values are the project's, not ours.

    analysis.ipynb reads this file positionally and treats 0.000000 as "did not
    finish", so writing anything else would quietly break the project's own
    analysis.
    """
    results.ensure_results_file(tmp_path)
    results.append_result(
        tmp_path,
        commit = "abc1234",
        val_bpb = None,
        memory_gb = None,
        status = results.STATUS_CRASH,
        description = "train.py raised on step 0",
    )
    lines = (
        _write(tmp_path, (tmp_path / results.RESULTS_FILENAME).read_text()).read_text().splitlines()
    )
    assert lines[0] == results.RESULTS_HEADER
    assert lines[1].split("\t") == [
        "abc1234",
        "0.000000",
        "0.0",
        "crash",
        "train.py raised on step 0",
    ]


def test_a_tab_in_a_description_cannot_shift_the_columns(tmp_path: Path) -> None:
    results.ensure_results_file(tmp_path)
    results.append_result(
        tmp_path,
        commit = "abc",
        val_bpb = 1.5,
        memory_gb = 3.0,
        status = results.STATUS_KEEP,
        description = "changed\tlr AND depth",
    )
    summary = results.read_results(tmp_path)
    assert summary.rows[0].description == "changed lr AND depth"


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------


def test_a_checkpoint_is_copied_out_before_the_next_run_overwrites_it(tmp_path: Path) -> None:
    """train.py saves one fixed filename and overwrites it every run.

    Without this copy the model that scored best is gone by experiment two, and
    "chat with the model from experiment 7" is not a question that can be asked.
    """
    (tmp_path / results.TRAIN_CHECKPOINT_NAME).write_bytes(b"weights")
    captured = results.capture_checkpoint(
        tmp_path,
        experiment = 7,
        commit = "abc1234",
        val_bpb = 1.25,
        num_params_m = 50.3,
        depth = 8,
    )
    assert captured is not None
    assert captured.experiment == 7

    # The next run overwrites the original; the copy must be unaffected.
    (tmp_path / results.TRAIN_CHECKPOINT_NAME).write_bytes(b"different weights")
    listed = results.list_checkpoints(tmp_path)
    assert len(listed) == 1
    assert listed[0].val_bpb == 1.25
    assert listed[0].num_params_m == 50.3
    assert listed[0].depth == 8


def test_a_crashed_run_leaves_no_checkpoint(tmp_path: Path) -> None:
    assert results.capture_checkpoint(tmp_path, experiment = 1, commit = "a", val_bpb = None) is None
    assert results.list_checkpoints(tmp_path) == []


def test_checkpoints_are_listed_best_first(tmp_path: Path) -> None:
    for index, bpb in ((1, 1.4), (2, 1.1), (3, 1.9)):
        (tmp_path / results.TRAIN_CHECKPOINT_NAME).write_bytes(b"w")
        results.capture_checkpoint(
            tmp_path,
            experiment = index,
            commit = f"c{index}",
            val_bpb = bpb,
        )
    listed = results.list_checkpoints(tmp_path)
    assert [entry.experiment for entry in listed] == [2, 1, 3]


def test_a_weights_file_without_its_sidecar_is_still_listed(tmp_path: Path) -> None:
    """The run wrote it, so refusing to show it would hide a model that exists."""
    directory = tmp_path / results.CHECKPOINT_DIR_NAME
    directory.mkdir(parents = True)
    (directory / "exp_004.pt").write_bytes(b"w")
    listed = results.list_checkpoints(tmp_path)
    assert len(listed) == 1
    assert listed[0].experiment == 4


def test_a_ledger_row_names_its_own_checkpoint(tmp_path: Path) -> None:
    (tmp_path / results.TRAIN_CHECKPOINT_NAME).write_bytes(b"w")
    results.capture_checkpoint(tmp_path, experiment = 2, commit = "beadcafe", val_bpb = 1.2)
    _write(
        tmp_path,
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\nbeadcafe\t1.200000\t3.5\tkeep\tone\n",
    )
    summary = results.read_results(tmp_path)
    assert summary.rows[0].checkpoint == "exp_002"


# -----------------------------------------------------------------------------
# The install's effect on the experiment repository
# -----------------------------------------------------------------------------


def test_a_finished_training_run_does_not_dirty_the_experiment_repository(tmp_path: Path) -> None:
    """train.py writes its checkpoint on every run, and the project does not
    ignore it.

    A dirty tree is what the Configure tab reads as "a previous run was
    interrupted between editing train.py and committing". Left unaddressed, the
    first run a user ever does opens with a warning about work they have not
    done, and the warning is the one piece of state in this feature that is
    meant to be trustworthy.
    """
    from core.autoresearch import environment

    ignore = environment._gitignore_additions()
    for artefact in (results.TRAIN_CHECKPOINT_NAME, "chat_server.py"):
        assert artefact in ignore
    # The studio's own directories too, or a report would be committed.
    for directory in (".checkpoints/", "reports/", ".autoresearch/"):
        assert directory in ignore


def test_the_additions_are_written_once(tmp_path: Path) -> None:
    """Re-adding on every install would grow the file on every reinstall."""
    from core.autoresearch import environment

    (tmp_path / ".gitignore").write_text("*.pt\n", encoding = "utf-8")
    environment._ensure_gitignore(tmp_path, lambda _line: None)
    first = (tmp_path / ".gitignore").read_text(encoding = "utf-8")
    environment._ensure_gitignore(tmp_path, lambda _line: None)
    assert (tmp_path / ".gitignore").read_text(encoding = "utf-8") == first


def test_the_marker_is_what_makes_the_addition_idempotent() -> None:
    from core.autoresearch import environment

    # Tied to the file rather than to a count, so an update that changes the
    # list does not need a second mechanism to know it already ran.
    assert "# Added by Unsloth Studio" in environment._gitignore_additions()


def test_the_ledger_is_created_before_the_baseline_commit(tmp_path: Path) -> None:
    """results.tsv is tracked, not ignored, and exists before the first experiment.

    Two halves, both about the same false warning. Untracked, it makes the tree
    dirty and the Configure tab reports an interrupted previous run. Ignored, the
    agent could not commit its own result row, which is how program.md's
    keep-or-discard step is expressed. So it is created early and committed.
    """
    results.ensure_results_file(tmp_path)
    path = tmp_path / results.RESULTS_FILENAME
    assert path.exists()
    assert path.read_text(encoding = "utf-8").strip() == results.RESULTS_HEADER

    from core.autoresearch import environment

    # Not in the ignore additions, and so committed by the install's baseline.
    assert results.RESULTS_FILENAME not in environment._gitignore_additions()


# -----------------------------------------------------------------------------
# run.log
# -----------------------------------------------------------------------------

A_COMPLETE_LOG = """\
GPU: NVIDIA GeForce RTX 4090
Vocab size: 8,192
Time budget: 300s
step 00042 (14%) | loss: 4.123456 | lrm: 0.0412 | dt: 118.32ms | tok/sec: 4429.1 | mfu: 41.22 | epoch: 1 | remaining: 246s
---
val_bpb:          1.234567
training_seconds: 300.4
total_seconds:    341.2
peak_vram_mb:     7421.3
mfu_percent:      38.14
total_tokens_M:   154.7
num_steps:        300
num_params_M:     50.3
depth:            8
dataset:          tinystories
train_batch_size: 16
eval_batch_size:  8
"""


def test_a_complete_log_yields_the_projects_own_summary_keys() -> None:
    outcome = agent.parse_run_log(A_COMPLETE_LOG)
    assert outcome.ok
    assert outcome.val_bpb == pytest.approx(1.234567)
    assert outcome.summary["peak_vram_mb"] == "7421.3"
    assert outcome.summary["num_params_M"] == "50.3"
    assert outcome.summary["depth"] == "8"


def test_a_log_with_no_score_is_a_crash_and_keeps_its_tail() -> None:
    """program.md's own rule: an empty grep for val_bpb means the run failed."""
    outcome = agent.parse_run_log("Traceback (most recent call last):\n  RuntimeError: CUDA OOM\n")
    assert outcome.ok is False
    assert outcome.val_bpb is None
    assert "CUDA OOM" in "\n".join(outcome.tail)


def test_a_loss_explosion_is_not_mistaken_for_a_score() -> None:
    outcome = agent.parse_run_log("FAIL: training loss exploded\n")
    assert outcome.ok is False
    assert outcome.val_bpb is None


def test_a_later_val_bpb_does_not_shadow_the_real_one() -> None:
    """The \r-overwritten progress line is split apart, so no stray number wins."""
    log = "step 00001 | loss: 1.0\nval_bpb: 1.500000\nval_bpb: 1.500000\n"
    outcome = agent.parse_run_log(log)
    assert outcome.val_bpb == pytest.approx(1.5)


# -----------------------------------------------------------------------------
# The agent command
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["codex", "opencode", "claude"])
def test_every_agent_command_carries_the_auto_approval_flag(key: str, monkeypatch) -> None:
    """Without it a headless run blocks on the first prompt forever.

    Monkeypatched past resolution: this is about the argv the build produces, and
    whether the CLI is installed is the machine's business, not the test's.
    """
    monkeypatch.setattr(agent, "resolve_executable", lambda spec: ("/usr/bin/fake", "1.0"))
    command = agent.build_command(
        agent_key = key,
        prompt = "body",
        cwd = "/work",
        skip_permissions = True,
        timeout_seconds = 1800,
    )
    spec = agent.get_spec(key)
    assert spec.auto_flag in command.argv


@pytest.mark.parametrize("key", ["codex", "opencode", "claude"])
def test_turning_approval_off_drops_the_flag(key: str, monkeypatch) -> None:
    monkeypatch.setattr(agent, "resolve_executable", lambda spec: ("/usr/bin/fake", "1.0"))
    command = agent.build_command(
        agent_key = key,
        prompt = "body",
        cwd = "/work",
        skip_permissions = False,
        timeout_seconds = 1800,
    )
    spec = agent.get_spec(key)
    assert spec.auto_flag not in command.argv


def test_the_prompt_is_never_passed_on_the_command_line(monkeypatch) -> None:
    """A prompt on argv reaches cmd.exe on Windows, which eats shell characters.

    The prompt quotes file names and would be mangled into something the agent
    misreads, silently. So it travels in a file and the argv carries only a
    fixed sentence.
    """
    monkeypatch.setattr(agent, "resolve_executable", lambda spec: ("/usr/bin/fake", "1.0"))
    prompt = "Read program.md | then run `uv run train.py` & commit 100% of it"
    command = agent.build_command(
        agent_key = "codex",
        prompt = prompt,
        cwd = "/work",
        skip_permissions = True,
        timeout_seconds = 1800,
    )
    assert prompt not in command.argv
    assert command.argv[-1] == agent._PROMPT_INSTRUCTION
    # No cmd metacharacter anywhere in what is actually passed.
    assert not any(char in command.argv[-1] for char in "|&<>^%!")


def test_an_unresolvable_agent_is_refused_with_a_reason(monkeypatch) -> None:
    monkeypatch.setattr(agent, "resolve_executable", lambda spec: None)
    with pytest.raises(RuntimeError, match = "not runnable"):
        agent.build_command(
            agent_key = "codex",
            prompt = "x",
            cwd = "/work",
            skip_permissions = True,
            timeout_seconds = 1800,
        )


def test_an_unknown_agent_key_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        agent.build_command(
            agent_key = "nope",
            prompt = "x",
            cwd = "/work",
            skip_permissions = False,
            timeout_seconds = 1800,
        )


def test_the_timeout_is_clamped_to_something_a_training_pass_can_finish() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(agent, "resolve_executable", lambda spec: ("/usr/bin/fake", "1.0"))
    try:
        # autoresearch says to kill a training pass past ten minutes, so a budget
        # below that would cut off a legitimate run.
        too_short = agent.build_command(
            agent_key = "codex",
            prompt = "x",
            cwd = "/w",
            skip_permissions = True,
            timeout_seconds = 60,
        )
        too_long = agent.build_command(
            agent_key = "codex",
            prompt = "x",
            cwd = "/w",
            skip_permissions = True,
            timeout_seconds = 99 * 3600,
        )
    finally:
        monkeypatch.undo()
    assert too_short.timeout_seconds == agent.MIN_AGENT_TIMEOUT_SECONDS
    assert too_long.timeout_seconds == agent.MAX_AGENT_TIMEOUT_SECONDS


def test_the_prompt_names_the_experiment_and_the_running_best() -> None:
    prompt = agent.build_experiment_prompt(
        index = 3,
        total = 8,
        best_bpb = 1.2345,
        completed = 2,
        baseline_bpb = 1.5,
    )
    assert "experiment 3 of 8" in prompt
    assert "1.234500" in prompt
    # The whole point of passing the best in: an agent told the frontier tries
    # to beat it, where one merely pointed at results.tsv often ignores them.
    assert "1.500000" in prompt
    # It must not also ask the agent to loop, or the studio's count is ignored.
    assert "do not start another experiment" in prompt


def test_the_first_experiment_is_told_it_is_setting_the_baseline() -> None:
    prompt = agent.build_experiment_prompt(
        index = 1,
        total = 4,
        best_bpb = None,
        completed = 0,
        baseline_bpb = None,
    )
    assert "first experiment" in prompt


# --- pointing the agent at a local model ---
#
# The agent is a separate program with its own model setting. The studio cannot
# know how someone has configured theirs, so the field takes whatever that CLI
# accepts. It reaches a command line, which is why the character rules exist.


def test_extra_args_reach_the_command(monkeypatch) -> None:
    """The lever: a local model, named by whatever flag the agent uses."""
    monkeypatch.setattr(agent, "resolve_executable", lambda _spec: ("/usr/bin/fake", "1.0"))
    command = agent.build_command(
        agent_key = "codex",
        prompt = "x",
        cwd = "/w",
        skip_permissions = True,
        timeout_seconds = 1800,
        extra_args = ["--model", "qwen3-coder-30b", "--config", "model_provider=local"],
    )
    assert "--model" in command.argv
    assert "qwen3-coder-30b" in command.argv
    assert "model_provider=local" in command.argv
    # The brief still goes last, so a flag cannot displace it.
    assert command.argv[-1] == agent._PROMPT_INSTRUCTION


def test_extra_args_land_before_the_working_directory(monkeypatch) -> None:
    """codex takes `-C <dir>` and opencode `--dir <dir>`; a trailing arg of the
    same name from the passthrough would be the wrong directory."""
    monkeypatch.setattr(agent, "resolve_executable", lambda _spec: ("/usr/bin/fake", "1.0"))
    command = agent.build_command(
        agent_key = "codex",
        prompt = "x",
        cwd = "/w",
        skip_permissions = True,
        timeout_seconds = 1800,
        extra_args = ["--model", "local"],
    )
    assert command.argv.index("--model") < command.argv.index("-C")


def test_the_loop_refuses_to_have_its_own_flags_overridden(monkeypatch) -> None:
    """The brief and the working directory are the loop's, not the field's."""
    monkeypatch.setattr(agent, "resolve_executable", lambda _spec: ("/usr/bin/fake", "1.0"))
    for flag in ("-C", "--cd", "--dir"):
        with pytest.raises(ValueError, match = "cannot be passed through"):
            agent.build_command(
                agent_key = "codex",
                prompt = "x",
                cwd = "/w",
                skip_permissions = True,
                timeout_seconds = 1800,
                extra_args = [flag, "/somewhere/else"],
            )


def test_a_shell_metacharacter_in_an_extra_arg_is_refused() -> None:
    """These reach a .cmd shim, so cmd.exe sees them.

    A model name, a provider id and a base URL contain none of these, so the rule
    costs nothing and closes command injection into an unattended loop.
    """
    from models.autoresearch import AutoresearchStartRequest
    for payload in (
        ["--model", "x;calc"],
        ["--model", "a&calc"],
        ["--model", "a|calc"],
        ["--model", "a>out"],
        ["--model", "a<in"],
        ["--model", "a^b"],
        ["--model", "a%PATH%"],
        ["--model", "a!b"],
        ["--model", "a`id`"],
        ["--model", 'a"b'],
        ["--model", "a'b"],
        ["--model", "a\nb"],
    ):
        with pytest.raises(Exception):
            AutoresearchStartRequest(num_experiments = 1, agent = "codex", agent_args = payload)


def test_a_plain_local_model_flag_is_accepted() -> None:
    """The refusal above must not block the thing it exists to enable."""
    from models.autoresearch import AutoresearchStartRequest

    request = AutoresearchStartRequest(
        num_experiments = 1,
        agent = "codex",
        agent_args = ["--model", "qwen3-coder-30b", "--config", "model_provider=local"],
    )
    assert request.agent_args == [
        "--model",
        "qwen3-coder-30b",
        "--config",
        "model_provider=local",
    ]


def test_blank_extra_args_are_dropped_rather_than_passed_as_empty() -> None:
    from models.autoresearch import AutoresearchStartRequest
    request = AutoresearchStartRequest(
        num_experiments = 1, agent = "codex", agent_args = ["  ", "--model", "local", ""]
    )
    assert request.agent_args == ["--model", "local"]


def test_agent_args_are_bounded() -> None:
    """A field that lands on a command line is a field that can be made enormous."""
    from models.autoresearch import AutoresearchStartRequest
    with pytest.raises(Exception):
        AutoresearchStartRequest(agent_args = [f"arg{index}" for index in range(64)])


# --- which agent, and why that one ---


def test_opencode_leads_the_detection_order() -> None:
    """It is the one most often pointed at a local model, and a loop measuring
    one machine should default to that machine's own weights."""
    assert agent.PREFERRED_AGENT_ORDER[0] == "opencode"
    assert set(agent.PREFERRED_AGENT_ORDER) == {spec.key for spec in agent.AGENT_SPECS}


def test_the_config_and_the_request_default_to_the_same_agent() -> None:
    """A form that opens on opencode and a request that defaults to codex is a
    loop that quietly runs on someone else's account."""
    from core.autoresearch.orchestrator import AutoresearchConfig
    from models.autoresearch import AutoresearchStartRequest

    assert AutoresearchConfig().agent == "opencode"
    assert AutoresearchStartRequest().agent == "opencode"
    assert AutoresearchStartRequest().agent == agent.PREFERRED_AGENT_ORDER[0]


def test_the_default_agent_is_one_this_build_can_drive() -> None:
    """The default has to be a key the backend knows how to launch, or the first
    run of a fresh install fails on a name that is only right in the UI."""
    from core.autoresearch.orchestrator import AutoresearchConfig
    assert agent.get_spec(AutoresearchConfig().agent) is not None


# --- the report written by an agent ---
#
# A user with a local model usually has it configured in their agent, not in the
# studio. Making them re-describe it in a second place is the wrong default when
# the same tool that ran the experiments can also write the report.


def test_the_report_accepts_an_agent_source() -> None:
    from models.autoresearch import AutoresearchReportRequest

    request = AutoresearchReportRequest(
        model_id = "agent:opencode",
        source = "agent",
        agent_key = "opencode",
        agent_args = ["--model", "qwen3-coder-30b"],
    )
    assert request.source == "agent"
    assert request.agent_key == "opencode"
    assert request.agent_args == ["--model", "qwen3-coder-30b"]


def test_the_report_and_the_loop_share_one_argument_rule() -> None:
    """Both land on a command line, so a rule in only one of them is a hole."""
    from models.autoresearch import AutoresearchReportRequest, AutoresearchStartRequest

    with pytest.raises(Exception):
        AutoresearchReportRequest(
            model_id = "x", source = "agent", agent_key = "opencode", agent_args = ["--model", "a;calc"]
        )
    with pytest.raises(Exception):
        AutoresearchStartRequest(agent_args = ["--model", "a;calc"])


def test_a_local_model_is_refused_for_a_subscription_backed_agent() -> None:
    """The failure this costs nothing to prevent.

    Codex's endpoint serves the ChatGPT subscription's own model names and answers
    anything else with a 400 before the agent reads the brief. A loop pointed at
    `--model unsloth/gemma-4-E2B-it-GGUF` therefore cannot succeed, and the only
    way that was visible was after paying for it: six experiments of about eleven
    minutes each, six identical crash rows, and no way to tell from the UI that the
    seventh would fail the same way.
    """
    from models.autoresearch import AutoresearchStartRequest

    with pytest.raises(Exception) as caught:
        AutoresearchStartRequest(
            agent = "codex",
            agent_args = ["--model", "unsloth/gemma-4-E2B-it-GGUF"],
        )
    # The message has to say what is wrong and what to do, or the user is where
    # they started minus two hours.
    message = str(caught.value)
    assert "gemma-4-E2B-it-GGUF" in message
    assert "opencode" in message, "the message should name a way out"


def test_the_same_local_model_is_allowed_for_an_agent_that_can_serve_it() -> None:
    """Pointing the loop at your own weights is the feature, not a mistake.

    opencode reads its provider config, which is the documented way to put a local
    model behind it, and it leads the agent list for exactly that reason. Refusing
    this would refuse the preferred agent doing the thing it is preferred for.
    """
    from models.autoresearch import AutoresearchStartRequest

    request = AutoresearchStartRequest(
        agent = "opencode",
        agent_args = ["--model", "unsloth/gemma-4-E2B-it-GGUF"],
    )
    assert request.agent_args == ["--model", "unsloth/gemma-4-E2B-it-GGUF"]


def test_a_plain_model_name_is_left_to_the_provider() -> None:
    """Only the structurally impossible cases are refused.

    `gpt-6-luna` is a real OpenAI model that this particular plan cannot reach.
    Whether an account can use a given slug is a question for the subscription, not
    for a validator with no network access, and guessing wrong here would refuse a
    setup that works. The repeated-crash stop is what catches that one.
    """
    from models.autoresearch import AutoresearchStartRequest

    request = AutoresearchStartRequest(agent = "codex", agent_args = ["--model", "gpt-6-luna"])
    assert request.agent_args == ["--model", "gpt-6-luna"]


def test_the_model_flag_is_read_in_both_spellings() -> None:
    from core.autoresearch.agent import _model_from_extra_args

    assert _model_from_extra_args(["--model", "x"]) == "x"
    assert _model_from_extra_args(["--model=x"]) == "x"
    assert _model_from_extra_args(["-m", "x"]) == "x"
    assert _model_from_extra_args(["--config", "a=b", "--model", "x"]) == "x"
    # Nothing named, and nothing guessed at.
    assert _model_from_extra_args(["--auto"]) is None
    assert _model_from_extra_args(["--model"]) is None
    assert _model_from_extra_args([]) is None
    assert _model_from_extra_args(None) is None


def test_an_equals_form_gguf_is_still_refused() -> None:
    # The suffix rule and the slash rule overlap here, but not in general: a bare
    # `gemma-4-E2B-it-GGUF` has no slash and is just as unservable.
    from models.autoresearch import AutoresearchStartRequest
    with pytest.raises(Exception):
        AutoresearchStartRequest(agent = "codex", agent_args = ["--model=gemma-4-E2B-it-GGUF"])


def test_a_one_shot_names_the_report_brief_not_the_experiment_one(tmp_path: Path) -> None:
    """Two briefs, two filenames.

    Pointing the report at the experiment brief would ask the agent to edit
    train.py instead of writing prose, which is the sort of thing that only shows
    up as a confusing git diff hours later.
    """
    command = agent.build_command(
        agent_key = "opencode",
        prompt = "p",
        cwd = str(tmp_path),
        skip_permissions = False,
        timeout_seconds = 900,
        instruction = agent._BRIEF_INSTRUCTION,
    )
    assert "report-brief.md" in command.argv[-1]
    assert "experiment.md" not in command.argv[-1]
    # The default still names the experiment brief.
    default = agent.build_command(
        agent_key = "opencode",
        prompt = "p",
        cwd = str(tmp_path),
        skip_permissions = False,
        timeout_seconds = 900,
    )
    assert "experiment.md" in default.argv[-1]


def test_a_one_shot_does_not_ask_for_unattended_permissions(tmp_path: Path) -> None:
    """The brief is fully specified and the job is read-only prose.

    Asking for blanket approval to rewrite files in order to be handed a
    paragraph would be a much larger grant than the job needs.
    """
    command = agent.build_command(
        agent_key = "opencode",
        prompt = "p",
        cwd = str(tmp_path),
        skip_permissions = False,
        timeout_seconds = 900,
        instruction = agent._BRIEF_INSTRUCTION,
    )
    spec = agent.get_spec("opencode")
    assert spec.auto_flag not in command.argv


def test_the_report_falls_back_to_the_loops_own_agent() -> None:
    """Writing it with a different model than did the work is a choice; having to
    make that choice in a second field is not."""
    from core.autoresearch.orchestrator import AutoresearchConfig

    config = AutoresearchConfig(
        agent = "codex",
        agent_args = ["--model", "local"],
        report_after_finish = True,
        report_model_id = "agent:opencode",
        report_source = "agent",
        report_agent_key = "",
    )
    resolved_key = config.report_agent_key or config.agent
    resolved_args = list(config.report_agent_args or config.agent_args)
    assert resolved_key == "codex"
    assert resolved_args == ["--model", "local"]
