"""
msc_paths.py — Canonical path resolution for all MSC scripts.

All paths are derived from BASE_DIR so the scripts work unchanged
across machines (local Mac, lab workstation, UTD cluster, BioHPC).
"""

import os
import sys

_CANDIDATES = [
    '/Users/aj/dmello_lab/fmri_connectivity_trees',
    '/Users/ajjain/Downloads/Code/fmri_connectivity_trees',
    '/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees',
    '/project/greencenter/Lin_lab/s229618/fmri_connectivity_trees',
]


def get_base_dir() -> str:
    for d in _CANDIDATES:
        if os.path.isdir(d):
            return d
    raise RuntimeError(
        "Cannot locate repo root. Add your machine's path to msc_paths._CANDIDATES."
    )


BASE_DIR = get_base_dir()
MSC_DIR  = os.path.join(BASE_DIR, 'code', 'functional_connectivity', 'midnight_scan_club')
ATLAS_DIR = os.path.join(BASE_DIR, 'atlases')


# ---------------------------------------------------------------------------
# Raw fMRI inputs (time series extracted from NIfTI files)
# ---------------------------------------------------------------------------

def ts_root() -> str:
    """datasets/midnight_scan_club/roi_time_series/"""
    return os.path.join(BASE_DIR, 'datasets', 'midnight_scan_club', 'roi_time_series')


def ts_dir(subject: str, session: str, atlas: str, task_dir: str) -> str:
    """Full path to a subject/session/atlas/task time-series directory."""
    return os.path.join(ts_root(), subject, session, atlas, task_dir)


# ---------------------------------------------------------------------------
# Intermediate processing outputs (MI, covariance, joint probs, entropies)
# ---------------------------------------------------------------------------

def output_dir(measure: str, subject: str, atlas_subdir: str) -> str:
    return os.path.join(MSC_DIR, 'output', measure, subject, atlas_subdir)

# internal alias used by msc_mutual_info.py and msc_covariance.py
_output = output_dir


def mi_dir(subject: str, atlas_subdir: str) -> str:
    return _output('mutual_information', subject, atlas_subdir)


def cov_dir(subject: str, atlas_subdir: str) -> str:
    return _output('covariance', subject, atlas_subdir)


def jp_dir(subject: str, atlas_subdir: str) -> str:
    return _output('joint_probs', subject, atlas_subdir)


def pm_dir(subject: str, atlas_subdir: str) -> str:
    return _output('product_of_marginals', subject, atlas_subdir)


def ent_dir(subject: str, atlas_subdir: str) -> str:
    return _output('entropies', subject, atlas_subdir)


# ---------------------------------------------------------------------------
# Analysis and figure outputs
# ---------------------------------------------------------------------------

def analysis_dir(subdir: str = '') -> str:
    """code/functional_connectivity/midnight_scan_club/analysis/[subdir]"""
    base = os.path.join(MSC_DIR, 'analysis')
    return os.path.join(base, subdir) if subdir else base


def figures_dir(subdir: str = '') -> str:
    """code/functional_connectivity/midnight_scan_club/figures/[subdir]"""
    base = os.path.join(MSC_DIR, 'figures')
    return os.path.join(base, subdir) if subdir else base


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(requested=None):
    """Detect the best available torch device, print a summary, and return it.

    Priority (when requested=None): CUDA → MPS → CPU.
    Returns a torch.device, which can be passed directly to torch.tensor(..., device=...).
    """
    try:
        import torch
    except ImportError:
        print("[device] torch not found — using CPU placeholder", file=sys.stderr)
        class _FakeDev:
            type = "cpu"
            def __str__(self): return "cpu"
        return _FakeDev()

    if requested is not None:
        dev = torch.device(requested)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    if dev.type == "cuda":
        n = torch.cuda.device_count()
        lines = [f"[device] CUDA ({n} GPU{'s' if n > 1 else ''} available)"]
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024 ** 3
            lines.append(f"         cuda:{i}  {props.name}  {mem_gb:.1f} GB")
        print("\n".join(lines))
    elif dev.type == "mps":
        print("[device] MPS (Apple Silicon GPU)")
    else:
        print(f"[device] CPU ({os.cpu_count()} logical cores) — no GPU available")

    return dev


# ---------------------------------------------------------------------------
# Atlas helpers
# ---------------------------------------------------------------------------

def glasser360_labels_path() -> str:
    return os.path.join(ATLAS_DIR, 'glasser360', 'glasser360NodeNames.txt')


def suit_lut_path() -> str:
    return os.path.join(ATLAS_DIR, 'SUIT', 'atl-Anatom.tsv')
