"""The orchestrator must turn nanochat's event stream into correct run state.

The event contract is the seam between the two projects: nanochat writes these
records and the orchestrator folds them. The records below are copied from real
runs, so a change to either side that breaks the seam fails here.
"""
import json
import subprocess
from pathlib import Path

import pytest

from core.nanochat.orchestrator import (
    NanochatConfig,
    NanochatRun,
    _EventTail,
)


def make_run(**config_kwargs) -> NanochatRun:
    return NanochatRun(run_id="test", config=NanochatConfig(**config_kwargs))


def feed(run: NanochatRun, records) -> NanochatRun:
    for record in records:
        run._apply_event(record.get("stage", "pretrain"), record)
    return run


# --- records copied from a real d6 pretrain run on an RTX 4060 Ti ---

PRETRAIN_CONFIG = {
    "kind": "config", "stage": "pretrain", "num_iterations": 40, "total_steps": 40,
    "total_batch_size": 1024, "device_batch_size": 2, "max_seq_len": 512,
    "grad_accum_steps": 1, "ddp_world_size": 1, "total_tokens": 40960,
    "num_params": 18481406, "param_breakdown": {"total": 18481406},
    "model_config": {"n_layer": 6, "n_embd": 384}, "peak_memory_bytes": 344635904,
    "gpu": "NVIDIA GeForce RTX 4060 Ti", "compute_dtype": "torch.bfloat16",
}

PRETRAIN_METRIC = {
    "kind": "metric", "stage": "pretrain", "step": 39, "total_steps": 40,
    "loss": 8.3346, "lr_multiplier": 1.0, "step_ms": 15.0, "tok_per_sec": 68277,
    "mfu": 26.85, "epoch": 1, "progress_percent": 97.5, "elapsed_seconds": 0.6,
    "eta_seconds": 0.01, "num_tokens": 39936,
}

RL_METRIC = {"kind": "metric", "stage": "rl", "step": 5, "total_steps": 60,
             "reward": 0.5, "mean_sequence_length": 88.3}

RL_SAMPLE = {"kind": "sample", "stage": "rl", "step": 5, "mode": "rollout",
             "prompt": "Mimi picked up 2 dozen seashells.", "completion": "48",
             "reward": 1.0, "advantage": 0.625, "sample_index": 0, "num_samples": 16}

BASE_EVAL_BPB = {"kind": "eval", "stage": "base_eval", "step": 40, "metric": "bpb",
                 "value": 3.904, "split": "val", "tokens": 8192}
BASE_EVAL_BPB_TRAIN = dict(BASE_EVAL_BPB, value=4.007, split="train")
CHATCORE_PARTIAL = {"kind": "eval", "stage": "chat_eval", "metric": "chatcore_subset",
                    "value": 0.31, "per_task": {"GSM8K": 0.0}, "is_partial": True,
                    "num_tasks": 1, "label": "Centered mean (1/5 tasks)"}


# --- the tail reader ---

def test_event_tail_only_reads_complete_lines(tmp_path):
    """A half-written record must not be parsed or consumed.

    nanochat appends to this file while we read it, so the writer is routinely
    mid-record. Consuming a partial line would either raise or, worse, advance
    the offset past bytes that are then never seen.
    """
    path = tmp_path / "events.jsonl"
    # Split at a point where each half is still a valid prefix of the record.
    path.write_text('{"kind":"a","n":1}\n{"kind":"b","n":2}\n{"kind":"c"', encoding="utf-8")

    tail = _EventTail(path)
    first = tail.read_new()
    assert [r["n"] for r in first] == [1, 2]
    assert tail.read_new() == [], "the partial line must not be returned yet"

    # The writer finishes the record.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(',"n":3}\n')
    assert [r["n"] for r in tail.read_new()] == [3]


def test_event_tail_keeps_a_record_split_across_reads(tmp_path):
    """A record arriving in pieces must be reassembled, not dropped.

    The reader polls on a timer while the writer flushes per record, so any
    record can straddle a poll boundary. The `config` event is the largest and
    carries the resolved step count; losing it leaves the UI unable to draw a
    progress bar at all.
    """
    path = tmp_path / "events.jsonl"
    tail = _EventTail(path)
    record = {"kind": "config", "total_steps": 4321, "num_params": 1384122122}
    line = json.dumps(record) + "\n"
    seen = []
    # One byte at a time is the worst case a poll boundary can produce.
    for index, character in enumerate(line):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(character)
        seen.extend(tail.read_new())
    assert seen == [record], f"lost or mangled a record: {seen}"


def test_event_tail_handles_a_large_record_split_in_two(tmp_path):
    """The realistic case: a big event cut in half by one read."""
    path = tmp_path / "events.jsonl"
    tail = _EventTail(path)
    big = {
        "kind": "config",
        "total_steps": 1000,
        "param_breakdown": {"total": 286261730, "wte": 25165824},
        "model_config": {"n_layer": 12, "n_embd": 768, "window_pattern": "SSSL"},
    }
    line = json.dumps(big) + "\n"
    cut = len(line) // 2
    path.write_text(line[:cut], encoding="utf-8")
    assert tail.read_new() == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line[cut:])
    assert tail.read_new() == [big]


def test_event_tail_does_not_replay_records_it_already_returned(tmp_path):
    """The carry buffer must not cause a re-read of the previous record."""
    path = tmp_path / "events.jsonl"
    tail = _EventTail(path)
    path.write_text('{"kind":"a","n":1}\n{"kind":"b","n":2}\n', encoding="utf-8")
    assert [r["n"] for r in tail.read_new()] == [1, 2]
    assert tail.read_new() == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"c","n":3}\n')
    assert [r["n"] for r in tail.read_new()] == [3]


def test_event_tail_resets_when_the_file_shrinks(tmp_path):
    """A reused path truncates; the reader must start over rather than seek past the end."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"kind":"a","n":1}\n' * 50, encoding="utf-8")
    tail = _EventTail(path)
    assert len(tail.read_new()) == 50
    path.write_text('{"kind":"b","n":9}\n', encoding="utf-8")
    assert [r["n"] for r in tail.read_new()] == [9]


def test_event_tail_skips_a_malformed_line_without_raising(tmp_path):
    """One bad line must not kill a run that is otherwise progressing."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"kind":"a","n":1}\nnot json at all\n{"kind":"c","n":3}\n', encoding="utf-8")
    records = _EventTail(path).read_new()
    assert [r["n"] for r in records] == [1, 3]


# --- config folding ---

def test_config_event_populates_totals_and_param_count():
    run = feed(make_run(), [PRETRAIN_CONFIG])
    assert run.state.total_steps == 40
    assert run.state.num_params == 18481406
    assert run.state.peak_memory_bytes == 344635904
    assert run.state.resolved_config["gpu"] == "NVIDIA GeForce RTX 4060 Ti"


def test_metric_event_updates_live_numbers():
    run = feed(make_run(), [PRETRAIN_CONFIG, PRETRAIN_METRIC])
    s = run.state
    assert s.step == 39
    assert s.progress_percent == 97.5
    assert s.tok_per_sec == 68277
    assert s.mfu == 26.85
    assert s.loss == 8.3346


def test_metric_series_is_keyed_by_what_it_measures():
    """Loss and reward share a metric event shape but not a scale.

    Plotting them on one axis would make an RL reward curve look like a
    divergence, so they go to separate series.
    """
    run = feed(make_run(), [
        dict(PRETRAIN_METRIC, stage="pretrain", loss=8.0),
        RL_METRIC,
    ])
    metrics = run.metrics()
    assert "loss" in metrics and "reward" in metrics
    assert metrics["loss"][0]["value"] == 8.0
    assert metrics["reward"][0]["value"] == 0.5
    assert run.state.reward == 0.5


def test_series_never_moves_backwards():
    """A resumed run replays earlier steps; letting them in would make the chart rewind."""
    run = feed(make_run(), [
        {"kind": "metric", "stage": "pretrain", "step": 10, "total_steps": 40, "loss": 5.0},
        {"kind": "metric", "stage": "pretrain", "step": 20, "total_steps": 40, "loss": 4.0},
        {"kind": "metric", "stage": "pretrain", "step": 15, "total_steps": 40, "loss": 4.5},
    ])
    steps = [p["step"] for p in run.metrics()["loss"]]
    assert steps == [10, 20]


def test_non_finite_loss_is_dropped_not_plotted():
    """A NaN must leave a gap in the chart, not a spike or a stale value."""
    run = feed(make_run(), [
        {"kind": "metric", "stage": "pretrain", "step": 1, "total_steps": 10, "loss": 5.0},
        {"kind": "metric", "stage": "pretrain", "step": 2, "total_steps": 10, "loss": None},
    ])
    assert [p["value"] for p in run.metrics()["loss"]] == [5.0]


def test_metric_series_is_bounded():
    from core.nanochat.orchestrator import _MAX_METRIC_POINTS

    records = [
        {"kind": "metric", "stage": "pretrain", "step": i, "total_steps": 99999, "loss": 1.0}
        for i in range(_MAX_METRIC_POINTS + 500)
    ]
    run = feed(make_run(), records)
    points = run.metrics()["loss"]
    assert len(points) == _MAX_METRIC_POINTS
    # The oldest are dropped, so the newest step survives.
    assert points[-1]["step"] == _MAX_METRIC_POINTS + 499


# --- samples ---

def test_every_sample_keeps_its_prompt():
    """The chat feed renders prompt and completion as a turn; an unattributed
    sample would render as a bare blob of text."""
    run = feed(make_run(), [
        {"kind": "sample", "stage": "pretrain", "step": 20, "mode": "completion",
         "prompt": "The capital of France is", "completion": " Paris"},
        {"kind": "sample", "stage": "sft", "step": 5, "mode": "chat",
         "prompt": "What is 2+2?", "completion": "4"},
    ])
    samples = run.samples()
    assert len(samples) == 2
    assert all(s["prompt"] for s in samples)
    assert {s["mode"] for s in samples} == {"completion", "chat"}


def test_rl_rollouts_carry_reward_and_advantage():
    run = feed(make_run(), [RL_SAMPLE])
    sample = run.samples()[0]
    assert sample["reward"] == 1.0
    assert sample["advantage"] == 0.625
    assert sample["mode"] == "rollout"


def test_sample_feed_is_bounded():
    from core.nanochat.orchestrator import _MAX_SAMPLES

    records = [
        {"kind": "sample", "stage": "pretrain", "step": i, "mode": "completion",
         "prompt": "p", "completion": "c"}
        for i in range(_MAX_SAMPLES + 100)
    ]
    run = feed(make_run(), records)
    # samples() defaults to a smaller page than the storage cap, so ask for the
    # full retained history to check the bound rather than the default.
    assert len(run.samples(limit=_MAX_SAMPLES)) == _MAX_SAMPLES
    # The default page is what the UI actually asks for.
    assert len(run.samples()) == 100


# --- benchmarks ---

def test_base_eval_reports_train_and_val_bpb_separately():
    """They are not the same number and merging them would hide overfitting."""
    run = feed(make_run(), [BASE_EVAL_BPB, BASE_EVAL_BPB_TRAIN])
    names = {b["name"] for b in run.benchmarks()}
    assert names == {"bpb:val", "bpb:train"}


def test_partial_chatcore_is_flagged():
    """A centered mean over one task is not ChatCORE and must not read as one."""
    run = feed(make_run(), [CHATCORE_PARTIAL])
    row = run.benchmarks()[0]
    assert row["is_partial"] is True
    assert row["name"] == "chatcore_subset"


def test_benchmark_rows_carry_baseline_and_centered_score():
    run = feed(make_run(), [{
        "kind": "benchmark", "stage": "chat_eval", "name": "MMLU",
        "accuracy": 0.3, "baseline": 0.25, "centered": 0.0667,
    }])
    row = run.benchmarks()[0]
    assert row["name"] == "MMLU"
    assert row["accuracy"] == 0.3
    assert row["baseline"] == 0.25
    assert row["centered"] == 0.0667


# --- checkpoints and warnings ---

def test_checkpoint_events_are_collected_per_source():
    run = feed(make_run(), [
        {"kind": "checkpoint", "stage": "pretrain", "source": "base", "step": 40,
         "path": "/x/base_checkpoints/d6", "model_tag": "d6", "num_params": 18},
        {"kind": "checkpoint", "stage": "sft", "source": "sft", "step": 9,
         "path": "/x/chatsft_checkpoints/d6", "model_tag": "d6"},
    ])
    assert [c["source"] for c in run.state.checkpoints] == ["base", "sft"]


def test_warnings_are_kept_bounded():
    from core.nanochat.orchestrator import _MAX_METRIC_POINTS

    records = [{"kind": "warning", "stage": "pretrain", "message": f"w{i}"} for i in range(500)]
    run = feed(make_run(), records)
    assert 0 < len(run.state.warnings) <= 40
    assert run.state.warnings[-1] == "w499"


def test_stage_end_carries_peak_memory_and_val_bpb():
    run = feed(make_run(), [{
        "kind": "stage_end", "stage": "pretrain", "status": "ok", "step": 40,
        "val_bpb": 3.9, "peak_memory_bytes": 344635904, "checkpoint_dir": "/x/d6",
    }])
    assert run.state.val_bpb == 3.9
    assert run.state.peak_memory_bytes == 344635904


# --- log tail ---

def test_log_lines_are_mirrored_and_bounded():
    from core.nanochat.orchestrator import _MAX_LOG_LINES

    records = [{"kind": "log", "line": f"line {i}"} for i in range(_MAX_LOG_LINES + 200)]
    run = feed(make_run(), records)
    tail = run.log_tail(limit=10)
    assert len(tail) == 10
    assert tail[-1]["line"] == f"line {_MAX_LOG_LINES + 199}"


def test_logs_since_resumes_without_replaying():
    """The console is a cursor feed, so a reconnect must not duplicate lines.

    Re-sending the tail would make the terminal repeat itself every time the SSE
    connection drops, which is the common case rather than the rare one.
    """
    run = feed(make_run(), [{"kind": "log", "line": f"line {i}"} for i in range(5)])

    fresh, cursor = run.logs_since(0)
    assert [entry["line"] for entry in fresh] == [f"line {i}" for i in range(5)]

    # Nothing new since: an empty batch, and the cursor is handed back unchanged
    # so the client keeps asking from the same place.
    again, same_cursor = run.logs_since(cursor)
    assert again == []
    assert same_cursor == cursor

    run._apply_event("pretrain", {"kind": "log", "line": "line 5"})
    tail, advanced = run.logs_since(cursor)
    assert [entry["line"] for entry in tail] == ["line 5"]
    assert advanced == cursor + 1


def test_progress_bar_frames_collapse_to_the_last_pass():
    """A redraw writes ``\\r``, not ``\\n``, so a text pipe yields all the frames
    of a progress bar as one line.

    Only the last frame is what a terminal would have shown. Keeping the whole
    string would render every frame concatenated onto one line.
    """
    run = make_run()
    run._append_log("step 1/100\rstep 2/100\rstep 3/100\n")
    assert [entry["line"] for entry in run.log_tail()] == ["step 3/100"]


def test_blank_lines_are_not_rendered():
    run = make_run()
    run._append_log("   ")
    run._append_log("")
    assert run.log_tail() == []


def test_stage_output_streams_to_the_console_with_its_stream_tagged():
    run = make_run()
    run._append_log("loss 2.31", stream="stdout", stage="pretrain")
    run._append_log("CUDA out of memory", stream="stderr", stage="pretrain")
    tail = run.log_tail()
    assert [entry["stream"] for entry in tail] == ["stdout", "stderr"]
    assert [entry["stage"] for entry in tail] == ["pretrain", "pretrain"]


def test_hms_is_readable_at_every_scale():
    from core.nanochat.orchestrator import _hms

    assert _hms(9) == "9s"
    assert _hms(249) == "4m 09s"
    assert _hms(3909) == "1h 05m 09s"
    # A negative duration is a clock artifact, not a message to render as such.
    assert _hms(-5) == "0s"


# --- argv ---

def test_pretrain_argv_passes_every_flag_explicitly():
    """A run must be reproducible from the stored config alone.

    nanochat's defaults are the published pipeline's, which is not what a UI user
    is running, so anything the run depends on is passed rather than defaulted.
    """
    run = make_run(depth=12, num_iterations=777, total_batch_size=16384, fp8=True)
    module, argv = run._stage_argv("pretrain")
    assert module == "scripts.base_train"
    joined = " ".join(argv)
    assert "--depth 12" in joined
    assert "--num-iterations 777" in joined
    assert "--total-batch-size 16384" in joined
    assert "--fp8" in argv
    assert "--model-tag d12" in joined


def test_fp8_is_absent_unless_requested():
    run = make_run(fp8=False)
    _module, argv = run._stage_argv("pretrain")
    assert "--fp8" not in argv


def test_sft_argv_carries_the_selected_tasks_and_benchmarks():
    run = make_run(sft_tasks=["gsm8k"], chat_benchmarks=["mmlu", "gsm8k"])
    _module, argv = run._stage_argv("sft")
    # argv is a flat list, so each flag is checked by its following value.
    assert argv[argv.index("--sft-tasks") + 1] == "gsm8k"
    assert argv[argv.index("--benchmarks") + 1] == "mmlu,gsm8k"


def test_chat_eval_argv_joins_benchmarks_with_a_pipe():
    """chat_eval splits multiple task names on '|', not a comma."""
    run = make_run(chat_benchmarks=["arc-easy", "arc-challenge", "mmlu"])
    _module, argv = run._stage_argv("chat_eval")
    assert "-a" in argv
    assert "arc-easy|arc-challenge|mmlu" in argv


def test_chat_eval_targets_the_sft_checkpoint():
    run = make_run(depth=20)
    _module, argv = run._stage_argv("chat_eval")
    assert argv[argv.index("-i") + 1] == "sft"
    assert argv[argv.index("--model-tag") + 1] == "d20"


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        make_run()._stage_argv("nonsense")


def test_model_tag_override_is_used_everywhere():
    run = make_run(depth=12, model_tag="my-run")
    for key in ("pretrain", "base_eval", "sft", "chat_eval", "rl"):
        _module, argv = run._stage_argv(key)
        assert "my-run" in argv, f"{key} did not use the model tag override"


def test_checkpoints_dir_maps_source_to_nanochat_folder():
    run = make_run(depth=12)
    assert run.checkpoints_dir("base").name == "d12"
    assert "base_checkpoints" in str(run.checkpoints_dir("base"))
    assert "chatsft_checkpoints" in str(run.checkpoints_dir("sft"))
    assert "chatrl_checkpoints" in str(run.checkpoints_dir("rl"))


def test_nanochat_base_dir_is_inside_the_account_workspace():
    """Never the user's real ~/.cache/nanochat, so accounts stay isolated."""
    run = make_run()
    assert ".cache" not in str(run.nanochat_base_dir)
    assert "nanochat" in str(run.nanochat_base_dir)


# --- the stage child: spawned windowless, with its output kept ---

class _FakeStream:
    """A text-mode pipe stand-in.

    Iterating yields one chunk per newline, which is what a real text stream over
    a pipe does. ``lines`` is therefore given as the already-split lines, and a
    single chunk may carry several lines when the child wrote them in one flush.
    """

    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        return None


class _FakeProc:
    def __init__(self, stdout_lines, stderr_lines, returncode = 0):
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream(stderr_lines)
        self.returncode = returncode
        self.pid = 4242

    def wait(self, timeout = None):
        return self.returncode

    def poll(self):
        return self.returncode


@pytest.fixture
def stage_run(monkeypatch):
    """A run whose child is faked, so the spawn path runs with no GPU and no venv."""
    from core.nanochat import environment, orchestrator

    monkeypatch.setattr(orchestrator, "is_process_shutting_down", lambda: False)
    monkeypatch.setattr(orchestrator, "adopt_pid", lambda _pid: None)
    monkeypatch.setattr(orchestrator, "_free_vram_for_stage", lambda self, _key: [], raising = False)
    monkeypatch.setattr(environment, "checkout_root", lambda: Path("."))
    monkeypatch.setattr(
        environment,
        "stage_environment",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        environment,
        "stage_command",
        lambda module, argv: [module, *argv],
    )

    spawned = {}

    def _popen(command, **kwargs):
        spawned["command"] = command
        spawned["kwargs"] = kwargs
        # One chunk carrying two lines, because a child is free to write both in a
        # single flush and the drain has to cope rather than assume one per read.
        return _FakeProc(["step 1/10\nloss 3.10\n"], ["a warning\n"])

    monkeypatch.setattr(orchestrator.subprocess, "Popen", _popen)
    return make_run(), spawned


def test_stage_child_gets_no_console_window(stage_run):
    """The windowless flag is the whole reason there is no black console.

    The backend itself is launched with CREATE_NO_WINDOW, so it owns no console.
    A child created without the flag is handed a brand-new one, which is the empty
    black window users were seeing.
    """
    run, spawned = stage_run
    run._run_stage("pretrain")
    kwargs = spawned["kwargs"]
    assert kwargs.get("creationflags") == getattr(subprocess, "CREATE_NO_WINDOW", 0)


def test_stage_child_output_reaches_the_console(stage_run):
    """stdout is piped into the log, not sent to DEVNULL.

    DEVNULL is why the console window stayed black: the output was being thrown
    away, and there was nothing for the in-app terminal to show either.
    """
    run, spawned = stage_run
    run._run_stage("pretrain")
    assert spawned["kwargs"]["stdout"] == subprocess.PIPE
    assert spawned["kwargs"]["stderr"] == subprocess.PIPE

    lines = run.log_tail()
    text = [entry["line"] for entry in lines]
    assert "step 1/10" in text
    assert "loss 3.10" in text
    # A stage's stderr is a warning or a traceback, and is kept apart so the UI can
    # colour it as the problem it usually is.
    warn = [entry for entry in lines if entry["line"] == "a warning"]
    assert warn and warn[0]["stream"] == "stderr"


def test_a_failing_stage_surfaces_its_stderr(stage_run, monkeypatch):
    from core.nanochat import orchestrator

    def _failing_popen(command, **kwargs):
        return _FakeProc([], ["Traceback (most recent call last):\nKeyError: boom\n"], 1)

    monkeypatch.setattr(orchestrator.subprocess, "Popen", _failing_popen)
    run, _ = stage_run
    assert run._run_stage("pretrain") == "failed"
    assert "KeyError: boom" in (run.state.stages["pretrain"].error or "")
    # The failure is announced in the console too, so a user watching the terminal
    # learns what happened without having to go looking at the error card.
    assert any("failed" in entry["line"] for entry in run.log_tail())
