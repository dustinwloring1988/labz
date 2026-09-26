"""Locating, installing and running nanochat as a separate, isolated environment.

nanochat pins ``torch==2.9.1`` and needs ``rustbpe`` (a Rust extension) plus the
``kernels`` package. The studio backend has its own transformers / peft /
bitsandbytes stack in the same interpreter. Importing nanochat into that process
is not viable: two torch versions cannot coexist, and nanochat's import-time
device detection and DDP setup would fight the backend's own torch usage.

So nanochat gets its own checkout and its own virtualenv, and every stage runs
as a child process in that venv. This module owns:

  - where the checkout and the venv live
  - cloning / updating the checkout
  - creating the venv with uv
  - reporting install progress to the UI
  - building the environment variables a stage needs

The machine-readable channel from a running stage back to the UI is the JSONL
file named by ``NANOCHAT_EVENT_FILE``; see ``nanochat/events.py`` on the nanochat
side. Nothing here parses stdout for numbers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from utils.paths.storage_roots import account_path, cache_root
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

# Where nanochat lives and what it is pinned to. Overridable so a developer can
# point at a local checkout (and so tests never touch the real one).
NANOCHAT_REPO = os.environ.get("UNSLOTH_NANOCHAT_REPO", "https://github.com/karpathy/nanochat.git")
# master, not main: that is the branch the upstream default HEAD points at, and
# cloning --branch main fails outright against it.
NANOCHAT_REF = os.environ.get("UNSLOTH_NANOCHAT_REF", "master")

_CHECKOUT_SUBPATH = "nanochat/nanochat-checkout"
_VENV_SUBPATH = "nanochat/venv"

# uv is the package manager nanochat is built around (its pyproject is uv-first,
# and the speedrun scripts shell out to it). Missing uv is the single most common
# reason a first run cannot start, so the error says how to fix it.
_UV_DOWNLOAD_URL = "https://astral.sh/uv/install.sh"
_UV_WINDOWS_URL = "https://astral.sh/uv/install.ps1"

# Stage scripts, in the order a full run uses them. Kept as data so the UI can
# render the pipeline and the orchestrator can sequence it without duplicating.
PIPELINE_STAGES: tuple[tuple[str, str, str], ...] = (
    # (stage key, display group, module)
    ("dataset", "prepare", "nanochat.dataset"),
    ("tokenizer", "prepare", "scripts.tok_train"),
    ("pretrain", "train", "scripts.base_train"),
    ("base_eval", "evaluate", "scripts.base_eval"),
    ("sft", "train", "scripts.chat_sft"),
    ("chat_eval", "evaluate", "scripts.chat_eval"),
    ("rl", "train", "scripts.chat_rl"),
)


def checkout_root() -> Path:
    return account_path(_CHECKOUT_SUBPATH)


def venv_root() -> Path:
    return account_path(_VENV_SUBPATH)


def venv_python() -> Path:
    """Interpreter inside the nanochat venv.

    Resolved per platform because a Windows venv puts it in Scripts/ and POSIX
    puts it at the root.
    """
    root = venv_root()
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def venv_site_packages() -> Path:
    if os.name == "nt":
        return venv_root() / "Lib" / "site-packages"
    return venv_root() / "lib" / "python3" / "site-packages"


def _uv_executable() -> str | None:
    """Path to uv, or None.

    Prefers a vendored copy so a studio install does not depend on the user
    having uv on PATH, then falls back to PATH.
    """
    bundled = cache_root() / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    if bundled.exists():
        return str(bundled)
    return shutil.which("uv")


def _torch_extra() -> str:
    """Which torch wheel to install.

    nanochat's pyproject offers mutually exclusive ``cpu`` and ``gpu`` extras.
    Picking the wrong one means either a 2.5GB CUDA download on a laptop or a
    CPU-only install that cannot use the user's GPU, so the choice follows the
    hardware actually present.

    MLX counts as cpu: nanochat has an MPS code path but it is exercised through
    the cpu extra, and installing the CUDA build on a Mac is simply wrong.
    """
    from utils.hardware import get_device

    try:
        device = get_device()
    except Exception:
        device = None
    # get_device() can be None before detection finishes. Default to gpu, because
    # a wrong 'cpu' choice silently costs the user all of their GPU, whereas a
    # wrong 'gpu' choice is merely a large download.
    if device is None:
        return "gpu"
    return "cpu" if getattr(device, "value", str(device)) in ("cpu", "mlx") else "gpu"


# -----------------------------------------------------------------------------
# State, shared with the UI
# -----------------------------------------------------------------------------

@dataclass
class EnvironmentStatus:
    """What the UI needs to decide whether the nanochat tab is usable yet."""

    checkout_present: bool = False
    checkout_path: str = ""
    checkout_ref: str = ""
    venv_present: bool = False
    python_path: str = ""
    uv_available: bool = False
    torch_installed: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    device_name: str | None = None
    # Human-readable next step when something is missing.
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

# Importing torch in the nanochat venv costs seconds, and environment_status() is
# on the path of every nanochat request, so an uncached probe turns each poll into
# a multi-second call. Short TTL: long enough that a UI poll loop hits the cache,
# short enough that a finished install shows up without a restart.
_PROBE_TTL_SECONDS = 30.0
_probe_lock = threading.Lock()
_probe_cache: dict[str, tuple[float, tuple[bool, str | None, bool, str | None]]] = {}


def _invalidate_probe_cache() -> None:
    with _probe_lock:
        _probe_cache.clear()


def _probe_torch() -> tuple[bool, str | None, bool, str | None]:
    """Ask the venv what it actually has installed.

    Runs in the nanochat venv, not this one, so the answer reflects the
    environment stages will really use. Never raises: an unprobeable environment
    is reported as "not installed" with no version.
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
            timeout=120,
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


def _current_ref() -> str:
    root = checkout_root()
    head = root / ".git" / "HEAD"
    try:
        if head.exists():
            return head.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        pass
    return ""


def environment_status() -> EnvironmentStatus:
    root = checkout_root()
    checkout_present = (root / "nanochat" / "__init__.py").exists()
    torch_installed, torch_version, cuda, device_name = _probe_torch()
    state = _get_install_state()

    status = EnvironmentStatus(
        checkout_present=checkout_present,
        checkout_path=str(root),
        checkout_ref=_current_ref(),
        venv_present=venv_python().exists(),
        python_path=str(venv_python()),
        uv_available=_uv_executable() is not None,
        torch_installed=torch_installed,
        torch_version=torch_version,
        cuda_available=cuda,
        device_name=device_name,
        install_state=state["state"],
        install_progress=state["progress"],
        install_message=state["message"],
    )

    if not checkout_present:
        status.blocking_reason = "nanochat has not been downloaded yet"
    elif not status.venv_present:
        status.blocking_reason = "the nanochat environment has not been created yet"
    elif not torch_installed:
        status.blocking_reason = "torch is not installed in the nanochat environment"
    elif not cuda and _torch_extra() == "gpu":
        # Not fatal: a CPU run is slow but works, so this is a warning rather than
        # a hard stop, and the status still reports ready.
        status.install_message = "No CUDA device visible; runs will be CPU-only and slow."
    return status


# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------

_LINE_RE = re.compile(r"(?P<bytes>\d[\d.,]*)\s*(?P<unit>[KMGT]?i?B)", re.IGNORECASE)
_UV_VERBOSE_RE = re.compile(
    r"(?P<resolved>\d[\d.,]*)\s*(?P<unit>[KMGT]?i?B)\s*(?P<what>downloaded|built|installed|resolved)",
    re.IGNORECASE,
)
_UV_TOTAL_RE = re.compile(r"Downloading\s+(?P<name>\S+)[^\n]*?\(?([\d.,]+\s*[KMGT]?i?B)?")


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


def _run_streaming(cmd: Iterable[str], *, cwd: Path | None, on_line: Callable[[str], None],
                   timeout: float | None = None,
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


def _local_source() -> Path | None:
    """A developer checkout to install from, or None to clone ``NANOCHAT_REPO``.

    Set ``UNSLOTH_NANOCHAT_REPO`` to a directory to use it. The stage runner and
    the event stream live in this project's patches to nanochat, so a run against
    pristine upstream is a run with no progress reporting at all.
    """
    candidate = Path(NANOCHAT_REPO)
    try:
        if candidate.is_dir() and (candidate / "nanochat" / "__init__.py").exists():
            return candidate.resolve()
    except OSError:
        pass
    return None


# Copied trees carry none of this: a checkout's own venv, caches and history are
# the developer's, not part of what the stages import.
_COPY_EXCLUDES = shutil.ignore_patterns(
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "*.pyc",
)


def _copy_local_checkout(source: Path, root: Path, emit: Callable[[str], None]) -> None:
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    emit(f"Copying nanochat from {source}")
    shutil.copytree(source, root, ignore=_COPY_EXCLUDES, symlinks=False, dirs_exist_ok=True)
    if not (root / "nanochat" / "__init__.py").exists():
        _set_install_state("failed", 0.0, "", "the local checkout has no nanochat package")
        raise RuntimeError(f"{source} does not look like a nanochat checkout")
    emit("Copied nanochat")


def install(on_line: Callable[[str], None] | None = None,
            *, update_checkout: bool = True) -> EnvironmentStatus:
    """Create the checkout and the venv. Idempotent; safe to call again.

    Blocking and can take several minutes (torch is a large download), so the UI
    should call it on a worker and poll ``environment_status()`` for progress.
    """
    emit = on_line or (lambda _line: None)
    root = checkout_root()
    root.mkdir(parents=True, exist_ok=True)

    # --- 1. checkout -------------------------------------------------------
    if not (root / "nanochat" / "__init__.py").exists():
        _set_install_state("installing", 0.02, "Downloading nanochat")
        emit("Downloading nanochat")
        source = _local_source()
        if source is not None:
            # A developer checkout (UNSLOTH_NANOCHAT_REPO pointing at a directory)
            # is copied rather than cloned: it usually has no git history, and the
            # point of pointing at one is to run the edits in it, not its commits.
            _copy_local_checkout(source, root, emit)
        else:
            if not shutil.which("git") and not (shutil.which("git.exe")):
                _set_install_state("failed", 0.0, "", "git is required to download nanochat")
                raise RuntimeError("git is required to download nanochat but was not found on PATH")
            if root.exists() and any(root.iterdir()):
                # A partial checkout from an interrupted run. git clone refuses a
                # non-empty target, so clear it rather than leaving a broken tree.
                shutil.rmtree(root, ignore_errors=True)
                root.mkdir(parents=True, exist_ok=True)
            code = _run_streaming(
                ["git", "clone", "--depth", "1", "--branch", NANOCHAT_REF, NANOCHAT_REPO, str(root)],
                cwd=None, on_line=emit,
            )
            if code != 0 or not (root / "nanochat" / "__init__.py").exists():
                _set_install_state("failed", 0.0, "", "could not download nanochat")
                raise RuntimeError(f"git clone of {NANOCHAT_REPO} failed (exit {code})")
    elif update_checkout and _local_source() is None:
        # The checkout is a git clone, so a local edit (this work adds a dataset
        # registry and an event stream) can be pulled. Best effort: a detached or
        # dirty tree must not block using what is already there.
        _set_install_state("installing", 0.05, "Updating nanochat")
        def _update_line(line: str) -> None:
            emit(line)
        _run_streaming(["git", "pull", "--ff-only"], cwd=root, on_line=_update_line, timeout=300)

    # --- 2. venv -----------------------------------------------------------
    uv = _uv_executable()
    if uv is None:
        url = _UV_WINDOWS_URL if os.name == "nt" else _UV_DOWNLOAD_URL
        _set_install_state("failed", 0.0, "", "uv is required to set up the nanochat environment")
        raise RuntimeError(
            f"uv was not found. Install it from {url} and reopen Unsloth Studio."
        )

    extra = _torch_extra()
    if not venv_python().exists():
        _set_install_state("installing", 0.06, "Creating the nanochat environment")
        emit("Creating the nanochat environment")
        code = _run_streaming([uv, "venv", str(venv_root())], cwd=root, on_line=emit)
        if code != 0:
            _set_install_state("failed", 0.0, "", "could not create the nanochat environment")
            raise RuntimeError(f"uv venv failed (exit {code})")

    # --- 3. dependencies ---------------------------------------------------
    # uv sync against the checkout's own pyproject/uv.lock, so the versions are
    # the ones nanochat was tested with rather than whatever resolves today.
    _set_install_state("installing", 0.08, f"Installing nanochat dependencies (torch {extra})")
    emit(f"Installing nanochat dependencies (torch {extra} extra)")

    def _sync_line(line: str) -> None:
        progress = _parse_download_progress(line)
        if progress is not None:
            _set_install_state("installing", 0.08 + progress * 0.9, f"Installing dependencies: {line[:120]}")
        emit(line)

    # uv sync installs into the PROJECT's own .venv unless it is told otherwise,
    # and VIRTUAL_ENV alone is not that instruction (that needs --active). Left
    # unset, every package lands in <checkout>/.venv while the stages run
    # venv_root()'s interpreter, which then cannot even import torch.
    env = {
        **os.environ,
        "VIRTUAL_ENV": str(venv_root()),
        "UV_PROJECT_ENVIRONMENT": str(venv_root()),
    }
    # --frozen keeps uv.lock authoritative. Resolving fresh would silently pick
    # up a different torch than the one nanochat's numbers were produced with.
    code = _run_streaming(
        [uv, "sync", f"--extra", extra, "--frozen"],
        cwd=root, on_line=_sync_line, env=env,
    )
    if code != 0:
        # --frozen fails if the lock is out of step with the pyproject, which a
        # local edit can cause. Fall back to a resolving sync rather than leaving
        # the user stuck.
        emit("Frozen sync failed; retrying without --frozen")
        code = _run_streaming([uv, "sync", f"--extra", extra], cwd=root, on_line=_sync_line, env=env)
    if code != 0:
        _set_install_state("failed", 0.0, "", "could not install nanochat dependencies")
        raise RuntimeError(f"uv sync failed (exit {code})")

    _invalidate_probe_cache()
    _set_install_state("ready", 1.0, "nanochat is ready")
    return environment_status()


def install_async() -> None:
    """Start ``install`` on a daemon thread; poll ``environment_status()``."""
    if _get_install_state()["state"] == "installing":
        return
    _set_install_state("installing", 0.01, "Starting")

    def _run() -> None:
        try:
            install(lambda _line: None)
        except Exception as exc:  # surfaced through environment_status()
            _set_install_state("failed", 0.0, "", str(exc))

    threading.Thread(target=_run, name="nanochat-install", daemon=True).start()


# -----------------------------------------------------------------------------
# Running stages
# -----------------------------------------------------------------------------

class NanochatNotInstalled(RuntimeError):
    """Raised when a stage is requested before the environment exists."""


def stage_environment(*, event_file: Path, stop_file: Path | None = None,
                      base_dir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for one nanochat child process.

    ``base_dir`` is where nanochat keeps checkpoints, tokenizer and data. It is
    deliberately inside the account workspace rather than the user's real
    ``~/.cache/nanochat``, so several accounts do not share one corpus and a
    reset does not reach outside the app.
    """
    env = dict(os.environ)
    env["NANOCHAT_EVENT_FILE"] = str(event_file)
    env["NANOCHAT_BASE_DIR"] = str(base_dir)
    # Unbuffered so a log line reaches the event file as it is produced rather
    # than when a 4KB block happens to fill.
    env["PYTHONUNBUFFERED"] = "1"
    # torchrun is not used (single device), but nanochat's CPU thread guidance
    # still applies and leaving this unset oversubscribes the machine.
    env.setdefault("OMP_NUM_THREADS", "1")
    if stop_file is not None:
        env["NANOCHAT_STOP_FILE"] = str(stop_file)
    if extra:
        env.update(extra)
    return env


def stage_command(module: str, argv: list[str]) -> list[str]:
    """The full argv for a stage: the venv interpreter running ``-m <module>``."""
    return [str(venv_python()), "-m", module, *argv]


def python_executable() -> Path:
    return venv_python()


def import_nanochat_versions() -> dict:
    """Version summary for the About/diagnostics panel."""
    torch_installed, torch_version, cuda, device_name = _probe_torch()
    return {
        "nanochat_ref": _current_ref(),
        "nanochat_checkout": str(checkout_root()),
        "python": str(venv_python()),
        "torch_installed": torch_installed,
        "torch_version": torch_version,
        "cuda_available": cuda,
        "device_name": device_name,
        "host_python": sys.executable,
    }
