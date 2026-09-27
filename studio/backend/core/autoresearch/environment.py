# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Locating, installing and running autoresearch as a separate, isolated environment.

autoresearch pins ``torch==2.9.1`` from the cu128 index and needs ``rustbpe``, a
Rust extension. The studio backend carries its own transformers / peft /
bitsandbytes stack in the same interpreter, so importing autoresearch into that
process is not viable for the same reasons nanochat's is not: two torch versions
cannot coexist in one environment.

So autoresearch gets its own checkout and its own virtualenv, and every
experiment runs as a child process in that venv. This module owns:

  - finding the user's autoresearch-win-rtx folder
  - copying it into the account workspace
  - making it a git repository, because the agent's whole keep/discard protocol
    is expressed in git terms
  - creating the venv with uv
  - running the one-time data preparation
  - reporting install progress to the UI
  - building the environment variables a child process needs

The machine-readable channel from a running experiment back to the UI is the
JSONL file named by ``AUTORESEARCH_EVENT_FILE``; see ``orchestrator.py`` for the
reader. Nothing here parses stdout for numbers.

Why a copy rather than running in place: autoresearch writes ``results.tsv``,
``run.log`` and ``checkpoint_pre_eval.pt`` into its working directory and the
agent edits ``train.py`` in it. Running in the user's own checkout would fill it
with experiment debris and leave a half-finished edit as the state of their
source. The copy means the user's folder is an input, not a scratch space, and
two accounts never share one experiment history.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from utils.paths.storage_roots import account_path, cache_root
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# The folder name autoresearch-win-rtx ships under. Searched for by walking up
# from the studio checkout, so a developer who keeps the two projects as
# siblings gets it found without configuring anything.
_PROJECT_DIR_NAME = "autoresearch-win-rtx"

# Overridable so a developer can point at a different tree, and so tests never
# touch the real one.
SOURCE_OVERRIDE_ENV = "UNSLOTH_AUTORESEARCH_SOURCE"

_CHECKOUT_SUBPATH = "autoresearch/checkout"
# The data cache is redirected out of %LOCALAPPDATA% for the same reason
# NANOCHAT_BASE_DIR is: several accounts must not share one corpus, and a reset
# must not reach outside the app.
_CACHE_SUBPATH = "autoresearch/cache"

# Files whose presence means "this is an autoresearch checkout". Deliberately
# the two the project treats as fixed: prepare.py holds the constants and the
# dataloader, train.py is the file the agent edits.
_REQUIRED_FILES = ("prepare.py", "train.py")

# uv is the package manager autoresearch is built around (its pyproject is
# uv-first, and the agent shells out to `uv run`). Missing uv is the single most
# common reason a first experiment cannot start, so the error says how to fix it.
_UV_DOWNLOAD_URL = "https://astral.sh/uv/install.sh"
_UV_WINDOWS_URL = "https://astral.sh/uv/install.ps1"

# Copied trees carry none of this: a checkout's own venv, caches and history are
# the user's, not part of what the experiments import. .git is dropped on
# purpose and re-initialised below, so the agent starts from one known commit
# rather than inheriting whatever branch the user happened to be on.
_COPY_EXCLUDES = shutil.ignore_patterns(
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "*.pyc",
    "results.tsv", "run.log", "checkpoint_pre_eval.pt", "progress.png",
)


# -----------------------------------------------------------------------------
# Locations
# -----------------------------------------------------------------------------

def checkout_root() -> Path:
    return account_path(_CHECKOUT_SUBPATH)


def cache_root_path() -> Path:
    return account_path(_CACHE_SUBPATH)


def venv_root() -> Path:
    """Inside the checkout, not beside it.

    autoresearch's own workflow is `uv run train.py` from the project root, and
    the agent is told to run exactly that. uv installs into the project's own
    .venv unless UV_PROJECT_ENVIRONMENT says otherwise, so putting the venv
    anywhere else means the agent's command resolves a different interpreter
    than the one that was installed.
    """
    return checkout_root() / ".venv"


def venv_python() -> Path:
    """Interpreter inside the autoresearch venv.

    Resolved per platform because a Windows venv puts it in Scripts/ and POSIX
    puts it at the root.
    """
    root = venv_root()
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def is_checkout_present() -> bool:
    root = checkout_root()
    return all((root / name).exists() for name in _REQUIRED_FILES)


def is_data_prepared() -> bool:
    """Whether ``prepare.py`` has produced a tokenizer and a dataset.

    Checked against the redirected cache rather than the machine default,
    because the experiments are given AUTORESEARCH_CACHE_DIR and would otherwise
    go looking in %LOCALAPPDATA% for something that is not there.
    """
    tokenizer_dir = cache_root_path() / "datasets"
    if not tokenizer_dir.is_dir():
        return False
    return any(tokenizer_dir.glob("*/tokenizer/tokenizer.pkl"))


def _uv_executable() -> str | None:
    """Path to uv, or None.

    Prefers a vendored copy so a studio install does not depend on the user
    having uv on PATH, then falls back to PATH.
    """
    bundled = cache_root() / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    if bundled.exists():
        return str(bundled)
    return shutil.which("uv")


def _studio_root() -> Path:
    # .../studio/backend/core/autoresearch/environment.py -> .../studio
    return Path(__file__).resolve().parents[3]


def _discover_source() -> Path | None:
    """The user's autoresearch-win-rtx folder, or None.

    Three ways to find it, in order: an explicit override, a walk up from the
    studio checkout looking for the project folder name, then a search of the
    studio checkout's own parent. The walk matters because the usual layout is
    two sibling repos, and hardcoding either one's absolute path would make the
    feature depend on one machine's directory structure.
    """
    override = os.environ.get(SOURCE_OVERRIDE_ENV)
    if override:
        candidate = Path(override).expanduser()
        if _looks_like_checkout(candidate):
            return candidate.resolve()

    start = _studio_root()
    for parent in [start, *start.parents]:
        candidate = parent / _PROJECT_DIR_NAME
        if _looks_like_checkout(candidate):
            return candidate.resolve()
    return None


def _looks_like_checkout(path: Path) -> bool:
    try:
        return path.is_dir() and all((path / name).exists() for name in _REQUIRED_FILES)
    except OSError:
        return False


def source_path() -> Path | None:
    """The configured source folder, whether or not it has been copied in yet."""
    return _discover_source()


# -----------------------------------------------------------------------------
# State, shared with the UI
# -----------------------------------------------------------------------------

@dataclass
class EnvironmentStatus:
    """What the UI needs to decide whether the autoresearch tab is usable yet."""

    source_present: bool = False
    source_path: str = ""
    checkout_present: bool = False
    checkout_path: str = ""
    venv_present: bool = False
    python_path: str = ""
    uv_available: bool = False
    torch_installed: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    device_name: str | None = None
    data_prepared: bool = False
    git_ready: bool = False
    blocking_reason: str | None = None
    install_state: str = "absent"  # absent | installing | ready | failed
    install_progress: float = 0.0
    install_message: str = ""

    def ready(self) -> bool:
        return self.venv_present and self.torch_installed and not self.blocking_reason


_install_lock = threading.Lock()
_install_state = {
    "state": "absent",
    "progress": 0.0,
    "message": "",
    "error": None,
}


def _set_install_state(state: str, progress: float, message: str, error: str | None = None) -> None:
    with _install_lock:
        _install_state.update(state=state, progress=progress, message=message, error=error)


def _get_install_state() -> dict:
    with _install_lock:
        return dict(_install_state)


# -----------------------------------------------------------------------------
# Status probing
# -----------------------------------------------------------------------------

# Importing torch in the autoresearch venv costs seconds, and
# environment_status() is on the path of every autoresearch request, so an
# uncached probe turns each poll into a multi-second call.
_PROBE_TTL_SECONDS = 30.0
_probe_lock = threading.Lock()
_probe_cache: dict[str, tuple[float, tuple[bool, str | None, bool, str | None]]] = {}


def _invalidate_probe_cache() -> None:
    with _probe_lock:
        _probe_cache.clear()


def _probe_torch() -> tuple[bool, str | None, bool, str | None]:
    """Ask the venv what it actually has installed.

    Runs in the autoresearch venv, not this one, so the answer reflects the
    environment experiments will really use. Never raises: an unprobeable
    environment is reported as "not installed" with no version.
    """
    python = venv_python()
    if not python.exists():
        return False, None, False, None
    key = str(python)
    now = time.monotonic()
    with _probe_lock:
        cached = _probe_cache.get(key)
    if cached is not None and now - cached[0] < _PROBE_TTL_SECONDS:
        return cached[1]
    result = _probe_torch_uncached(python)
    with _probe_lock:
        _probe_cache[key] = (time.monotonic(), result)
    return result


def _probe_torch_uncached(python: Path) -> tuple[bool, str | None, bool, str | None]:
    code = (
        "import json,torch;"
        "print(json.dumps({"
        "'version': torch.__version__,"
        "'cuda': bool(torch.cuda.is_available()),"
        "'device': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
        "}))"
    )
    try:
        proc = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            encoding="utf-8",
            errors="replace",
            **windows_hidden_subprocess_kwargs(),
        )
    except Exception:
        return False, None, False, None
    if proc.returncode != 0:
        return False, None, False, None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return False, None, False, None
    try:
        payload = json.loads(line[-1])
    except Exception:
        return False, None, False, None
    return True, payload.get("version"), bool(payload.get("cuda")), payload.get("device")


def _git_ready() -> bool:
    """Whether the copy is a git repository with at least one commit.

    The agent's protocol is entirely git: branch, commit, keep or `git reset`.
    Without a repository every experiment would be a crash, so this is a hard
    requirement rather than a nicety.
    """
    root = checkout_root()
    if not (root / ".git").exists():
        return False
    if not shutil.which("git"):
        return False
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
            **windows_hidden_subprocess_kwargs(),
        )
    except Exception:
        return False
    return proc.returncode == 0


def environment_status() -> EnvironmentStatus:
    source = _discover_source()
    torch_installed, torch_version, cuda, device_name = _probe_torch()
    state = _get_install_state()

    status = EnvironmentStatus(
        source_present=source is not None,
        source_path=str(source) if source else "",
        checkout_present=is_checkout_present(),
        checkout_path=str(checkout_root()),
        venv_present=venv_python().exists(),
        python_path=str(venv_python()),
        uv_available=_uv_executable() is not None,
        torch_installed=torch_installed,
        torch_version=torch_version,
        cuda_available=cuda,
        device_name=device_name,
        data_prepared=is_data_prepared(),
        git_ready=_git_ready(),
        install_state=state["state"],
        install_progress=state["progress"],
        install_message=state["message"],
    )

    if not status.source_present:
        status.blocking_reason = (
            f"No autoresearch-win-rtx folder was found. Set {SOURCE_OVERRIDE_ENV} to "
            f"the folder holding prepare.py and train.py."
        )
    elif not status.checkout_present:
        status.blocking_reason = "autoresearch has not been copied into the workspace yet"
    elif not shutil.which("git"):
        status.blocking_reason = "git is required: the experiment loop keeps or discards with git"
    elif not status.venv_present:
        status.blocking_reason = "the autoresearch environment has not been created yet"
    elif not torch_installed:
        status.blocking_reason = "torch is not installed in the autoresearch environment"
    elif not cuda:
        # Not fatal: the autotuner in train.py falls back to a CPU path, so a run
        # is slow but works. A warning rather than a hard stop.
        status.install_message = (
            "No CUDA device visible. autoresearch's fastest path is an NVIDIA GPU; "
            "a CPU run works but is far slower."
        )
    return status


# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------

_LINE_RE = re.compile(r"(?P<bytes>\d[\d.,]*)\s*(?P<unit>[KMGT]?i?B)", re.IGNORECASE)
_UV_VERBOSE_RE = re.compile(
    r"(?P<resolved>\d[\d.,]*)\s*(?P<unit>[KMGT]?i?B)\s*(?P<what>downloaded|built|installed|resolved)",
    re.IGNORECASE,
)


def _parse_download_progress(line: str) -> float | None:
    """Best-effort 0..1 from uv's progress output.

    uv does not emit machine-readable progress, so this is only used to keep the
    bar moving. A None means "no idea", and callers should leave the previous
    value alone rather than jump backwards.
    """
    match = _UV_VERBOSE_RE.search(line)
    if not match:
        return None
    try:
        size = float(match.group("resolved").replace(",", ""))
    except ValueError:
        return None
    unit = (match.group("unit") or "B").upper()
    scale = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "KIB": 1024,
             "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4}.get(unit, 1)
    resolved_bytes = size * scale
    # Torch wheels are the bulk; cap the denominator at a size that still reads
    # sensibly on a fast connection.
    total = 3.5e9
    return max(0.0, min(0.95, resolved_bytes / total))


def _run_streaming(cmd: Iterable[str], *, cwd: Path | None,
                   on_line: Callable[[str], None], timeout: float | None = None,
                   env: dict[str, str] | None = None) -> int:
    """Run a command, forwarding each output line to ``on_line``.

    Line-buffered with stdin closed and stderr merged, mirroring the pattern used
    elsewhere in the backend for installers: a single reader loop with no
    communicate(), so a long download never deadlocks on a full pipe. ``env``
    replaces the environment for the child; without it the parent's is inherited.
    """
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **windows_hidden_subprocess_kwargs(),
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            on_line(line.rstrip())
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.returncode or -1


def _git(args: list[str], root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **windows_hidden_subprocess_kwargs(),
    )


def copy_source(emit: Callable[[str], None]) -> None:
    """Copy the user's autoresearch folder into the account workspace.

    Replaces whatever was there. The copy is what the agent edits, so keeping the
    previous one would let an abandoned experiment's half-finished change become
    the starting point of the next loop.
    """
    source = _discover_source()
    if source is None:
        raise RuntimeError(
            f"No autoresearch-win-rtx folder was found. Set {SOURCE_OVERRIDE_ENV} to "
            f"the folder holding prepare.py and train.py."
        )
    root = checkout_root()
    if root.exists():
        # The venv is multi-gigabyte and does not need to survive: it is rebuilt
        # from uv.lock below, which is also the only way a changed pyproject
        # takes effect.
        shutil.rmtree(root, ignore_errors=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    emit(f"Copying autoresearch from {source}")
    shutil.copytree(source, root, ignore=_COPY_EXCLUDES, symlinks=False)
    missing = [name for name in _REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"{source} is missing {', '.join(missing)}")
    emit("Copied autoresearch")


def init_git(emit: Callable[[str], None]) -> None:
    """One repository, one commit, so the agent's protocol has a baseline.

    autoresearch's program.md tells the agent to branch, commit, and reset back
    when a change does not help. All three need a commit to move away from, and
    without an initial one `git reset` has nothing to return to and every
    experiment looks like the first one.
    """
    root = checkout_root()
    if not shutil.which("git"):
        raise RuntimeError("git is required for the autoresearch experiment loop but was not found on PATH")
    if not (root / ".git").exists():
        emit("Initializing the experiment repository")
        _git(["init", "-b", "autoresearch/main"], root)
    # A fresh copy of a .git-less folder has no identity, and a commit fails
    # without one. Set it locally on the copy so the user's global config is
    # neither required nor modified.
    _git(["config", "user.name", "LABZ autoresearch"], root)
    _git(["config", "user.email", "autoresearch@unsloth.local"], root)
    _git(["config", "commit.gpgsign", "false"], root)

    head = _git(["rev-parse", "--verify", "HEAD"], root)
    if head.returncode != 0:
        emit("Recording the baseline commit")
        _git(["add", "-A"], root)
        commit = _git(
            ["commit", "-m", "autoresearch: baseline before the first experiment"], root
        )
        if commit.returncode != 0:
            raise RuntimeError(
                f"could not create the baseline commit: {(commit.stderr or '').strip()[-400:]}"
            )
    emit("Experiment repository ready")


def _gitignore_additions() -> str:
    """What the copy's .gitignore is missing.

    autoresearch ships a .gitignore, but the artefacts the studio loop produces on
    top of the project's own are not in it. Without this the agent's `git reset`
    would happily restore a 200MB checkpoint, and `git add -A` in a commit would
    stage one.

    ``checkpoint_pre_eval.pt`` is the one that matters most: train.py writes it
    on every single run and the project does not ignore it, so a run that ends
    normally leaves the tree dirty. The Configure tab reads a dirty tree as "a
    previous run was interrupted mid-edit" and says so, which on the first run
    ever would be a false alarm about work the user has not done.
    """
    return (
        "\n# Added by LABZ Studio\n"
        ".checkpoints/\n"
        "reports/\n"
        ".autoresearch/\n"
        "chat_server.py\n"
        "checkpoint_pre_eval.pt\n"
    )


def commit_install_changes(emit: Callable[[str], None]) -> bool:
    """Commit anything the install itself changed, so the tree starts clean.

    ``uv sync`` can rewrite ``uv.lock``, and that is the install's doing rather
    than the user's. Left uncommitted it makes the tree dirty before a single
    experiment has run, and the Configure tab reports a dirty tree as "a previous
    run was interrupted between editing train.py and committing" — a warning
    about work that has not happened.

    Returns whether anything was committed.
    """
    root = checkout_root()
    if not (root / ".git").exists() or not shutil.which("git"):
        return False
    try:
        status = _git(["status", "--porcelain"], root)
    except Exception:
        return False
    if status.returncode != 0 or not (status.stdout or "").strip():
        return False
    changed = (status.stdout or "").strip().splitlines()
    emit(f"Recording the post-install state ({len(changed)} file(s))")
    _git(["add", "-A"], root)
    commit = _git(
        ["commit", "-m", "autoresearch: state after environment setup"], root
    )
    if commit.returncode != 0:
        # Nothing to commit is the common case and not a problem; anything else
        # leaves the tree as it is, which the next run's warning will explain.
        return False
    return True


def install(on_line: Callable[[str], None] | None = None, *,
            prepare_data: bool = True) -> EnvironmentStatus:
    """Create the checkout and the venv, then prepare the data. Idempotent.

    Blocking, and can take a long time (torch is a large download, and
    prepare.py pulls a dataset and trains a tokenizer), so the UI calls it on a
    worker and polls ``environment_status()`` for progress.
    """
    emit = on_line or (lambda _line: None)
    uv = _uv_executable()
    if uv is None:
        _set_install_state("failed", 0.0, "", "uv was not found on PATH")
        raise RuntimeError(
            "uv is required to set up autoresearch. Install it with "
            f"`irm {_UV_WINDOWS_URL} | iex` (PowerShell) or "
            f"`curl -LsSf {_UV_DOWNLOAD_URL} | sh`."
        )

    # --- 1. checkout -------------------------------------------------------
    root = checkout_root()
    if not is_checkout_present():
        _set_install_state("installing", 0.02, "Copying autoresearch")
        copy_source(emit)

    _ensure_gitignore(root, emit)
    init_git(emit)

    # --- 2. venv -----------------------------------------------------------
    if not venv_python().exists():
        _set_install_state("installing", 0.06, "Creating the autoresearch environment")
        emit("Creating the autoresearch environment")
        code = _run_streaming([uv, "venv", str(venv_root())], cwd=root, on_line=emit)
        if code != 0:
            _set_install_state("failed", 0.0, "", "could not create the autoresearch environment")
            raise RuntimeError(f"uv venv failed (exit {code})")

    # --- 3. dependencies ---------------------------------------------------
    # uv sync against the checkout's own pyproject/uv.lock, so the versions are
    # the ones autoresearch was tested with rather than whatever resolves today.
    # autoresearch pins torch to the cu128 index and names that index explicitly
    # in its pyproject, so no --extra is needed here: the extra exists to pick a
    # wheel flavour, and this project has already picked one.
    _set_install_state("installing", 0.08, "Installing autoresearch dependencies (torch cu128)")
    emit("Installing autoresearch dependencies")

    def _sync_line(line: str) -> None:
        progress = _parse_download_progress(line)
        if progress is not None:
            _set_install_state(
                "installing", 0.08 + progress * 0.62,
                f"Installing dependencies: {line[:120]}",
            )
        emit(line)

    env = {
        **os.environ,
        "VIRTUAL_ENV": str(venv_root()),
        "UV_PROJECT_ENVIRONMENT": str(venv_root()),
    }
    # --frozen keeps uv.lock authoritative. Resolving fresh would silently pick
    # up a different torch than the one the project was built against.
    code = _run_streaming([uv, "sync", "--frozen"], cwd=root, on_line=_sync_line, env=env)
    if code != 0:
        # --frozen fails if the lock is out of step with the pyproject, which a
        # user's edit can cause. Fall back to a resolving sync rather than
        # leaving the user stuck.
        emit("Frozen sync failed; retrying without --frozen")
        code = _run_streaming([uv, "sync"], cwd=root, on_line=_sync_line, env=env)
    if code != 0:
        _set_install_state("failed", 0.0, "", "could not install autoresearch dependencies")
        raise RuntimeError(f"uv sync failed (exit {code})")

    _invalidate_probe_cache()
    torch_installed, _, _, _ = _probe_torch()
    if not torch_installed:
        _set_install_state("failed", 0.0, "", "torch is missing after a successful sync")
        raise RuntimeError("torch could not be imported in the autoresearch environment")

    # The sync may have rewritten uv.lock, and the ledger is created here rather
    # than by the first agent. Fold both into the baseline: a tree that is dirty
    # before the first experiment is a tree the Configure tab reports as "a
    # previous run was interrupted", which is a warning about work nobody did.
    try:
        from core.autoresearch import results as _results

        _results.ensure_results_file(root)
    except Exception:
        # Not fatal. The loop creates it on demand; this only saves the first run
        # a spurious dirty-tree warning.
        logger.debug("could not pre-create the autoresearch ledger", exc_info=True)
    commit_install_changes(emit)

    # --- 4. data -----------------------------------------------------------
    # prepare.py downloads the dataset and trains the tokenizer. It is not
    # optional: train.py reads the tokenizer from the cache and dies without it.
    if prepare_data and not is_data_prepared():
        _set_install_state("installing", 0.72, "Downloading the dataset and training the tokenizer")
        emit("Preparing data (this downloads the dataset and trains a tokenizer)")
        code = _run_streaming(
            [uv, "run", "prepare.py"],
            cwd=root,
            on_line=_prepare_line,
            env=experiment_environment(),
        )
        if code != 0:
            _set_install_state("failed", 0.0, "", "data preparation failed")
            raise RuntimeError(f"prepare.py failed (exit {code})")

    _set_install_state("ready", 1.0, "autoresearch is ready")
    return environment_status()


def _ensure_gitignore(root: Path, emit: Callable[[str], None]) -> None:
    """Add the studio's artefacts to the copy's .gitignore, once."""
    marker = "# Added by LABZ Studio"
    path = root / ".gitignore"
    try:
        existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    except OSError:
        existing = ""
    if marker in existing:
        return
    emit("Updating .gitignore for the studio's artefacts")
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_gitignore_additions())
    except OSError as exc:
        # Not fatal: a missing ignore only costs disk, and the agent's own
        # `git add train.py program.md` style commands are unaffected.
        emit(f"Could not update .gitignore ({exc}); experiment artefacts may be committed")


def _prepare_line(line: str) -> None:
    """Map prepare.py's own progress printouts onto the install bar.

    prepare.py reports in words rather than bytes, so this counts the phases it
    announces. Faking a percentage from a phase name would be worse than a
    message that moves, so the bar creeps forward on each one.
    """
    lowered = line.lower()
    if "downloading" in lowered:
        return
    _bump_install_progress(0.72, line[:140])


_prepare_lock = threading.Lock()


def _bump_install_progress(floor_value: float, message: str) -> None:
    with _prepare_lock:
        current = _get_install_state()
        if current["state"] != "installing":
            return
        # Never move backwards: a repeated line would otherwise rewind the bar.
        progress = max(float(current["progress"]), min(0.99, floor_value))
        _set_install_state("installing", progress, message or current["message"])


def install_async(*, prepare_data: bool = True) -> None:
    """Start ``install`` on a daemon thread; poll ``environment_status()``."""
    if _get_install_state()["state"] == "installing":
        return
    _set_install_state("installing", 0.01, "Starting")

    def _run() -> None:
        try:
            install(lambda _line: None, prepare_data=prepare_data)
        except Exception as exc:  # surfaced through environment_status()
            _set_install_state("failed", 0.0, "", str(exc))

    threading.Thread(target=_run, name="autoresearch-install", daemon=True).start()


def prepare_data_async() -> None:
    """Run just ``prepare.py``, for a workspace that already has a venv."""
    if _get_install_state()["state"] == "installing":
        return
    _set_install_state("installing", 0.72, "Preparing data")

    def _run() -> None:
        uv = _uv_executable()
        if uv is None:
            _set_install_state("failed", 0.0, "", "uv was not found on PATH")
            return
        try:
            code = _run_streaming(
                [uv, "run", "prepare.py"],
                cwd=checkout_root(),
                on_line=_prepare_line,
                env=experiment_environment(),
            )
        except Exception as exc:
            _set_install_state("failed", 0.0, "", str(exc))
            return
        if code != 0:
            _set_install_state("failed", 0.0, "", f"prepare.py failed (exit {code})")
        else:
            _set_install_state("ready", 1.0, "Data prepared")

    threading.Thread(target=_run, name="autoresearch-prepare", daemon=True).start()


# -----------------------------------------------------------------------------
# Running experiments
# -----------------------------------------------------------------------------

class AutoresearchNotInstalled(RuntimeError):
    """Raised when an experiment is requested before the environment exists."""


def experiment_environment(*, event_file: Path | None = None,
                           stop_file: Path | None = None,
                           extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for one autoresearch child process.

    Three things are overridden. The cache dir keeps each account's dataset to
    itself, matching what prepare.py was given at install time — pointing the
    experiments somewhere else would make train.py fail to find the tokenizer it
    was prepared with. The event file is the machine-readable channel to the UI.
    And unbuffered output means a log line reaches the console as it is produced
    rather than when a 4KB block happens to fill.
    """
    env = dict(os.environ)
    env["AUTORESEARCH_CACHE_DIR"] = str(cache_root_path())
    env["PYTHONUNBUFFERED"] = "1"
    # torch's own threading is left alone: train.py's MuonAdamW steps are the
    # bottleneck and capping the CPU pool there is what makes a small card
    # slower for no benefit. This only guards against a runaway BLAS.
    env.setdefault("OMP_NUM_THREADS", "4")
    if event_file is not None:
        env["AUTORESEARCH_EVENT_FILE"] = str(event_file)
    if stop_file is not None:
        env["AUTORESEARCH_STOP_FILE"] = str(stop_file)
    if extra:
        env.update(extra)
    return env


def training_command(*, smoke_test: bool = False,
                     dataset: str | None = None) -> list[str]:
    """argv for one training pass.

    ``uv run`` rather than the venv interpreter directly, because that is the
    command program.md tells the agent to use and the two must agree: if the
    agent ran the interpreter and the studio ran uv, a pyproject edit would take
    effect for one and not the other.
    """
    argv = ["uv", "run", "train.py"]
    if smoke_test:
        argv.append("--smoke-test")
    if dataset:
        argv.extend(["--dataset", dataset])
    return argv


def import_autoresearch_versions() -> dict:
    """Version summary for the About/diagnostics panel."""
    torch_installed, torch_version, cuda, device_name = _probe_torch()
    return {
        "autoresearch_source": str(source_path() or ""),
        "autoresearch_checkout": str(checkout_root()),
        "autoresearch_cache": str(cache_root_path()),
        "python": str(venv_python()),
        "torch_installed": torch_installed,
        "torch_version": torch_version,
        "cuda_available": cuda,
        "device_name": device_name,
        "host_python": sys.executable,
    }
