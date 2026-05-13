"""Shared ROI definitions and helpers for canonical-circuits analysis."""

import os
import sys
import json
import numpy as np

_CANONICAL_DIR = os.path.dirname(os.path.abspath(__file__))
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)
sys.path.insert(0, _MSC_DIR)

from msc_paths import (BASE_DIR, ATLAS_DIR, ts_root,
                       analysis_dir, figures_dir, glasser360_labels_path)

# ---------------------------------------------------------------------------
# Glasser360 motor / premotor ROI indices (0-based in combined atlas array)
# NIfTI label value = index + 1
# Right hemisphere: indices 0–179  |  Left hemisphere: indices 180–359
# ---------------------------------------------------------------------------

PRIMARY_MOTOR_IDX  = [7, 187]              # Area 4  (R, L)

PRIMARY_SENS_IDX   = [8, 188,              # Area 3b (R, L) — tactile
                      52, 232,             # Area 3a (R, L) — proprioceptive
                      50, 230,             # Area 1  (R, L)
                      51, 231]             # Area 2  (R, L)

SMA_IDX            = [54, 234]             # 6mp — SMA proper
PRE_SMA_IDX        = [43, 223]             # 6ma — pre-SMA
SCEF_IDX           = [42, 222]             # SCEF
PARACENTRAL_IDX    = [35, 215, 38, 218]    # 5m (R,L), 5L (R,L)

PMD_IDX            = [53, 233,             # 6d  — PMd proper
                      77, 257,             # 6r
                      95, 275]             # 6a

PMV_IDX            = [55, 235,             # 6v  — PMv
                      11, 191]             # 55b — speech-motor

MOTOR_CORTEX_ALL = sorted(set(
    PRIMARY_MOTOR_IDX + PRIMARY_SENS_IDX + SMA_IDX + PRE_SMA_IDX +
    SCEF_IDX + PARACENTRAL_IDX + PMD_IDX + PMV_IDX
))

# ---------------------------------------------------------------------------
# SUIT cerebellar indices (offset 360 in combined atlas)
# ---------------------------------------------------------------------------

CEREB_OFFSET       = 360
CEREB_MOTOR_LOCAL  = [0, 1, 5, 6, 20, 21]   # lobules IV/V, VI, VIII
CEREB_ALL_LOCAL    = list(range(34))
CEREB_MOTOR_IDX    = [CEREB_OFFSET + i for i in CEREB_MOTOR_LOCAL]
CEREB_ALL_IDX      = [CEREB_OFFSET + i for i in CEREB_ALL_LOCAL]

CC_MOTOR_IDX = sorted(MOTOR_CORTEX_ALL + CEREB_MOTOR_IDX)
CC_FULL_IDX  = sorted(MOTOR_CORTEX_ALL + CEREB_ALL_IDX)

# ---------------------------------------------------------------------------
# Effector ROI sets (cortical indices; cerebellar added by scripts)
# ---------------------------------------------------------------------------

EFFECTOR_ROIS = {
    'foot': {
        'cortex': [7, 187,    # 4  L/R
                   52, 232,   # 3a L/R
                   35, 215,   # 5m L/R
                   54, 234,   # 6mp (SMA) L/R
                   43, 223],  # 6ma (pre-SMA) L/R
    },
    'hand': {
        'cortex': [7, 187,    # 4  L/R
                   8, 188,    # 3b L/R
                   50, 230,   # 1  L/R
                   51, 231,   # 2  L/R
                   53, 233,   # 6d L/R
                   77, 257,   # 6r L/R
                   55, 235,   # 6v L/R
                   95, 275],  # 6a L/R
    },
    'tongue': {
        'cortex': [7, 187,    # 4  L/R
                   8, 188,    # 3b L/R
                   11, 191,   # 55b L/R
                   55, 235],  # 6v L/R
    },
}

# ---------------------------------------------------------------------------
# Localizer sub-parcellation config
#
# SUB_PARCELLATE: parcels large enough for effector-specific voxel selection
#   (area 4 and 3b, both hemispheres).  Key = label string, value = 0-based idx.
# WHOLE_PARCEL_IDX: all other motor parcels — used as full-parcel ROIs.
# ---------------------------------------------------------------------------

SUB_PARCELLATE = {
    'Right_4':  7,
    'Left_4':   187,
    'Right_3b': 8,
    'Left_3b':  188,
}

# All motor parcels minus the four that are sub-parcellated
_sub_set = set(SUB_PARCELLATE.values())
WHOLE_PARCEL_IDX = [i for i in MOTOR_CORTEX_ALL if i not in _sub_set]

# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def cc_analysis_dir(submodule: str) -> str:
    d = os.path.join(analysis_dir(), 'canonical_circuits', submodule)
    os.makedirs(d, exist_ok=True)
    return d


def cc_figures_dir(submodule: str) -> str:
    d = os.path.join(figures_dir(), 'canonical_circuits', submodule)
    os.makedirs(d, exist_ok=True)
    return d


def mask_output_dir(subject: str) -> str:
    d = os.path.join(cc_analysis_dir('motor_cortex'), 'localizer_masks', subject)
    os.makedirs(d, exist_ok=True)
    return d


def timeseries_output_dir(subject: str, session: str) -> str:
    d = os.path.join(ts_root(), subject, session, 'localizer_functional_rois', 'rest')
    os.makedirs(d, exist_ok=True)
    return d

# ---------------------------------------------------------------------------
# Atlas label helpers
# ---------------------------------------------------------------------------

def load_glasser360_labels() -> list:
    with open(glasser360_labels_path()) as f:
        return [line.strip() for line in f if line.strip()]


def load_combined_labels(atlas_subdir='glasser360_SUIT_Thalamus') -> list:
    """Return full node label list for the combined atlas (Glasser + SUIT)."""
    from msc_chow_liu import load_node_labels
    return load_node_labels(atlas_subdir, n_rois=360 + 34)

# ---------------------------------------------------------------------------
# MI matrix helpers (for combined-atlas pipeline)
# ---------------------------------------------------------------------------

def subset_mi(mi_matrix: np.ndarray, roi_indices: list) -> np.ndarray:
    """Extract a square sub-matrix for the given ROI indices."""
    idx = np.array(roi_indices)
    return mi_matrix[np.ix_(idx, idx)]


def load_mi_matrix(subject: str, atlas_subdir: str = 'glasser360_SUIT_Thalamus',
                   task: str = 'motor', num_bins: int = 100,
                   suffix: str = 'func-01-10') -> np.ndarray:
    """Load precomputed MI matrix; falls back to rest if motor file absent."""
    from msc_paths import mi_dir
    fname = f'{task}_{num_bins}bins_{suffix}.npy'
    fpath = os.path.join(mi_dir(subject, atlas_subdir), fname)
    if not os.path.exists(fpath):
        fname = f'rest_{num_bins}bins_{suffix}.npy'
        fpath = os.path.join(mi_dir(subject, atlas_subdir), fname)
    return np.load(fpath)

# ---------------------------------------------------------------------------
# Localizer MI adapter
# Loads saved localizer timeseries, concatenates across sessions, computes MI.
# ---------------------------------------------------------------------------

def load_localizer_timeseries(subject: str, session_list: list):
    """
    Load and concatenate localizer-ROI timeseries across sessions.
    Returns (ts_concat, roi_keys) where ts_concat is (n_TRs, n_ROIs).
    Returns (None, None) if no session files are found.
    """
    all_ts   = []
    roi_keys = None

    for session in session_list:
        d        = timeseries_output_dir(subject, session)
        csv_path = os.path.join(d, 'rest.csv')
        meta_path = os.path.join(d, 'roi_metadata.json')
        if not os.path.exists(csv_path):
            continue
        ts = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        if roi_keys is None and os.path.exists(meta_path):
            with open(meta_path) as f:
                roi_keys = json.load(f)['roi_keys']
        all_ts.append(ts)

    if not all_ts:
        return None, None

    return np.concatenate(all_ts, axis=0), roi_keys


def compute_localizer_mi(subject: str, session_list: list,
                         num_bins: int = 100, device=None):
    """
    Compute pairwise MI matrix from localizer-ROI timeseries.
    Returns (mi_matrix np.ndarray, roi_keys list).
    """
    import torch
    from msc_mutual_info import pairwise_roi_mutual_information
    from msc_paths import detect_device

    if device is None:
        device = detect_device()

    ts_concat, roi_keys = load_localizer_timeseries(subject, session_list)
    if ts_concat is None:
        raise FileNotFoundError(
            f'No localizer timeseries found for {subject}. '
            'Run msc_localizer_roi_timeseries.py first.'
        )

    ts_tensor = torch.tensor(
        ts_concat.T.astype(np.float32), device=device
    )  # (n_ROIs, n_TRs)

    mi, _, _, _ = pairwise_roi_mutual_information(ts_tensor, num_bins=num_bins)
    return mi.cpu().numpy(), roi_keys
