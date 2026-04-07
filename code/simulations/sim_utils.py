"""
sim_utils.py — Shared path helpers and parameter-logging utilities.

Directory layout enforced by these helpers:
    datasets/synthetic/<run_id>/timeseries/   — raw simulation time series (.npz)
    code/simulations/output/<run_id>/          — analysis results (.npz, .json)
    code/simulations/figures/<run_id>/         — figures (.pdf, .png)

Each run also gets a params.json written to both the timeseries dir and the
analysis dir so that either directory alone is self-documenting.
"""

import datetime
import json
import os
import subprocess
import sys

# Repo root: two levels up from this file (code/simulations/ → code/ → repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def timeseries_dir(run_id: str) -> str:
    """datasets/synthetic/<run_id>/timeseries/"""
    return os.path.join(REPO_ROOT, "datasets", "synthetic", run_id, "timeseries")


def analysis_dir(run_id: str) -> str:
    """code/simulations/output/<run_id>/"""
    return os.path.join(REPO_ROOT, "code", "simulations", "output", run_id)


def figures_dir(run_id: str) -> str:
    """code/simulations/figures/<run_id>/"""
    return os.path.join(REPO_ROOT, "code", "simulations", "figures", run_id)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(requested: str | None = None) -> str:
    """Detect the best available compute device, print a summary, and return
    a concrete device string suitable for passing directly to run_sim().

    Resolving the device once at script startup means every run_sim() call
    receives an explicit string (e.g. "cuda") and skips re-detection.

    Priority (when requested=None): CUDA → MPS → CPU.
    """
    try:
        import torch
    except ImportError:
        print("[device] torch not found — falling back to CPU", file=sys.stderr)
        return "cpu"

    if requested is not None:
        dev = torch.device(requested)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    # Report
    if dev.type == "cuda":
        n = torch.cuda.device_count()
        lines = [f"[device] CUDA ({n} GPU{'s' if n > 1 else ''} available)"]
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024 ** 3
            lines.append(f"         cuda:{i}  {props.name}  {mem_gb:.1f} GB")
        # If no specific index was requested, use cuda:0
        if requested is None or requested == "cuda":
            lines.append(f"         → using cuda:0")
        print("\n".join(lines))
    elif dev.type == "mps":
        print("[device] MPS (Apple Silicon GPU)")
    else:
        import os as _os
        print(f"[device] CPU ({_os.cpu_count()} logical cores) — no GPU available")

    return str(dev)


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def make_run_id(tag: str) -> str:
    """Return a timestamped run ID: YYYYMMDD_HHMMSS_<tag>."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tag}"


# ---------------------------------------------------------------------------
# Parameter logging
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _write_params_json(directory: str, payload: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "params.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {path}")


def save_params(run_id: str, script: str, args_dict: dict) -> None:
    """Write params.json to both the timeseries dir and the analysis dir."""
    payload = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": _git_hash(),
        "script": script,
        "args": args_dict,
    }
    for d in [timeseries_dir(run_id), analysis_dir(run_id)]:
        _write_params_json(d, payload)


def save_ts_params(run_id: str, script: str, args_dict: dict) -> None:
    """Write params.json only to the timeseries dir (for timeseries-only runs)."""
    payload = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": _git_hash(),
        "script": script,
        "args": args_dict,
    }
    _write_params_json(timeseries_dir(run_id), payload)
