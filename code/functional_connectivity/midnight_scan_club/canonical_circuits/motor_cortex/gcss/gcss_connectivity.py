"""
gcss_connectivity.py — FC, MI, and Chow-Liu tree analysis for GCSS motor fROIs.

Loads per-session resting-state timeseries produced by gcss_timeseries.py and
computes three connectivity measures:
  - FC (Pearson correlation matrix)
  - MI (pairwise mutual information, same discretization as main pipeline)
  - CL (Chow-Liu maximum spanning tree of MI)

Outputs are saved both per-session and as a concatenated summary across all
available sessions for each subject.

Output directories (relative to BASE_DIR):
  FC : datasets/midnight_scan_club/connectivity/fc/{sub}/{ses}/gcss_motor/
  MI : datasets/midnight_scan_club/connectivity/mi/{sub}/{ses}/gcss_motor/
  CL : datasets/midnight_scan_club/connectivity/cl/{sub}/{ses}/gcss_motor/

  Concatenated (all sessions):
  FC : datasets/midnight_scan_club/connectivity/fc/{sub}/all_sessions/gcss_motor/
  MI : datasets/midnight_scan_club/connectivity/mi/{sub}/all_sessions/gcss_motor/
  CL : datasets/midnight_scan_club/connectivity/cl/{sub}/all_sessions/gcss_motor/

Usage
-----
  python gcss_connectivity.py
  python gcss_connectivity.py --subjects MSC01 --num-bins 100
  python gcss_connectivity.py --no-concat   # skip concatenated outputs
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # motor_cortex/gcss/
_MOTOR_DIR     = os.path.dirname(_SCRIPT_DIR)                  # motor_cortex/
_CANONICAL_DIR = os.path.dirname(_MOTOR_DIR)                   # canonical_circuits/
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)               # midnight_scan_club/
sys.path.insert(0, _MSC_DIR)
sys.path.insert(0, _CANONICAL_DIR)

from msc_paths import BASE_DIR, ts_root, detect_device
from msc_mutual_info import pairwise_roi_mutual_information
from msc_chow_liu import chow_liu_tree, draw_hierarchical_tree

# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _connectivity_dir(measure: str, subject: str, session: str) -> str:
    """
    Return (and create) the output directory for one connectivity measure.

    measure  : 'fc', 'mi', or 'cl'
    session  : e.g. 'func01' or 'all_sessions'
    """
    d = os.path.join(BASE_DIR, 'datasets', 'midnight_scan_club',
                     'connectivity', measure, subject, session, 'gcss_motor')
    os.makedirs(d, exist_ok=True)
    return d


def _figures_dir(subject: str) -> str:
    d = os.path.join(_MSC_DIR, 'figures', 'canonical_circuits',
                     'motor_cortex', 'gcss', subject)
    os.makedirs(d, exist_ok=True)
    return d


def _timeseries_dir(subject: str, session: str) -> str:
    return os.path.join(ts_root(), subject, session, 'gcss_motor')


# ---------------------------------------------------------------------------
# Timeseries loading
# ---------------------------------------------------------------------------

def load_session_timeseries(
    subject: str, session: str
) -> tuple[np.ndarray, list] | tuple[None, None]:
    """
    Load one session's rest.csv and roi_metadata.json.

    Returns (ts np.ndarray (n_TRs, n_ROIs), roi_keys list) or (None, None).
    """
    d         = _timeseries_dir(subject, session)
    csv_path  = os.path.join(d, 'rest.csv')
    meta_path = os.path.join(d, 'roi_metadata.json')

    if not os.path.exists(csv_path):
        return None, None

    ts = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    roi_keys = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            roi_keys = json.load(fh)['roi_keys']
    return ts, roi_keys


def load_concat_timeseries(
    subject: str, sessions: list
) -> tuple[np.ndarray, list] | tuple[None, None]:
    """
    Load and concatenate timeseries across all available sessions.

    Returns (ts_concat (n_TRs_total, n_ROIs), roi_keys) or (None, None).
    """
    all_ts   = []
    roi_keys = None

    for ses in sessions:
        ts, keys = load_session_timeseries(subject, ses)
        if ts is None:
            continue
        if roi_keys is None:
            roi_keys = keys
        all_ts.append(ts)

    if not all_ts:
        return None, None
    return np.concatenate(all_ts, axis=0), roi_keys


# ---------------------------------------------------------------------------
# Connectivity computation
# ---------------------------------------------------------------------------

def compute_fc(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix (diagonal = 0). Input shape: (TRs, ROIs)."""
    corr = np.corrcoef(ts.T)
    np.fill_diagonal(corr, 0.0)
    return corr


def compute_mi(ts: np.ndarray, num_bins: int, device) -> np.ndarray:
    """
    Pairwise MI matrix via the same discretization pipeline as the main analysis.
    Input: (TRs, ROIs). Output: (ROIs, ROIs) numpy array.
    """
    ts_t = torch.tensor(ts.T.astype(np.float32), device=device)  # (ROIs, TRs)
    mi, _, _, _ = pairwise_roi_mutual_information(ts_t, num_bins=num_bins)
    return mi.cpu().numpy()


def compute_cl(mi: np.ndarray) -> tuple[nx.Graph, np.ndarray]:
    """
    Build Chow-Liu tree (maximum spanning tree of MI graph).

    Returns
    -------
    tree   : NetworkX Graph (N nodes, N-1 edges)
    cl_adj : (N, N) numpy adjacency with MI weights on tree edges
    """
    _, tree = chow_liu_tree(mi)
    cl_adj  = nx.to_numpy_array(tree, weight='weight')
    return tree, cl_adj


# ---------------------------------------------------------------------------
# Save connectivity outputs
# ---------------------------------------------------------------------------

def save_connectivity(
    subject: str,
    session: str,
    fc: np.ndarray,
    mi: np.ndarray,
    cl_adj: np.ndarray,
    roi_keys: list,
) -> None:
    """Save FC, MI, CL arrays and roi_keys to their respective directories."""
    for measure, mat in [('fc', fc), ('mi', mi), ('cl', cl_adj)]:
        d = _connectivity_dir(measure, subject, session)
        np.save(os.path.join(d, f'{subject}_{session}_gcss_{measure}.npy'), mat)

    # roi_keys are the same for all three measures; save once in fc dir
    fc_dir = _connectivity_dir('fc', subject, session)
    with open(os.path.join(fc_dir, f'{subject}_{session}_gcss_roi_keys.json'), 'w') as fh:
        json.dump(roi_keys, fh, indent=2)

    print(f'    saved FC/MI/CL → connectivity/{{fc,mi,cl}}/{subject}/{session}/gcss_motor/')


# ---------------------------------------------------------------------------
# ROI color helpers (for figures)
# ---------------------------------------------------------------------------

_SEMANTIC_COLORS = {
    'R_M1_foot':   '#e31a1c',   # dark red
    'L_M1_foot':   '#fb9a99',   # light red
    'R_M1_hand':   '#1f78b4',   # dark blue
    'L_M1_hand':   '#a6cee3',   # light blue
    'L_M1_tongue': '#33a02c',   # dark green
    'R_M1_tongue': '#b2df8a',   # light green
    'SMA':         '#ff7f00',   # orange
    'L_PMd':       '#e377c2',   # pink
    'R_PMv':       '#9467bd',   # purple
    'L_S1S2':      '#17becf',   # teal
    'R_S1S2':      '#8c564b',   # brown
}

_ROI_GROUP = {
    'R_M1_foot': 'foot', 'L_M1_foot': 'foot',
    'R_M1_hand': 'hand', 'L_M1_hand': 'hand',
    'L_M1_tongue': 'tongue', 'R_M1_tongue': 'tongue',
    'SMA': 'premotor', 'L_PMd': 'premotor', 'R_PMv': 'premotor',
    'L_S1S2': 'premotor', 'R_S1S2': 'premotor',
}


def _roi_color(roi_key: str) -> str:
    return _SEMANTIC_COLORS.get(roi_key, '#dddddd')


def _short_label(roi_key: str) -> str:
    return roi_key


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _draw_contrast_dividers(ax, roi_keys: list) -> None:
    """Thin divider lines at somatotopic-group transitions in a matrix heatmap."""
    n = len(roi_keys)
    groups = [_ROI_GROUP.get(k, k) for k in roi_keys]
    for i in range(1, n):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color='white', lw=0.8)
            ax.axvline(i - 0.5, color='white', lw=0.8)


def make_figures(
    subject: str,
    session_label: str,
    fc: np.ndarray,
    mi: np.ndarray,
    cl_adj: np.ndarray,
    tree: nx.Graph,
    roi_keys: list,
    out_dir: str,
) -> None:
    """Generate FC matrix, MI matrix, CL tree, and FC-vs-MI scatter figures."""
    n     = len(roi_keys)
    short = [_short_label(k) for k in roi_keys]
    tag   = f'{subject}_{session_label}'

    # -- FC heatmap --
    fig, ax = plt.subplots(figsize=(max(8, n * 0.32), max(7, n * 0.28)))
    im = ax.imshow(fc, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r', fraction=0.03)
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=5)
    _draw_contrast_dividers(ax, roi_keys)
    ax.set_title(f'{tag} — GCSS Motor FC', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{tag}_gcss_fc_matrix.pdf'), bbox_inches='tight')
    plt.close(fig)

    # -- MI heatmap --
    fig, ax = plt.subplots(figsize=(max(8, n * 0.32), max(7, n * 0.28)))
    im = ax.imshow(mi, cmap='viridis', vmin=0, aspect='auto')
    plt.colorbar(im, ax=ax, label='MI (nats)', fraction=0.03)
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=5)
    _draw_contrast_dividers(ax, roi_keys)
    ax.set_title(f'{tag} — GCSS Motor MI', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{tag}_gcss_mi_matrix.pdf'), bbox_inches='tight')
    plt.close(fig)

    # -- CL tree --
    fig, ax = plt.subplots(figsize=(max(18, n * 0.8), 10))
    draw_hierarchical_tree(
        tree, roi_keys,
        title=f'{tag} — GCSS Motor CL Tree',
        ax=ax,
        color_fn=_roi_color,
        label_fn=_short_label,
        label_fontsize=6,
    )
    patches = [mpatches.Patch(color=col, label=name)
               for name, col in _SEMANTIC_COLORS.items()
               if name in roi_keys]
    ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{tag}_gcss_cl_tree.pdf'), bbox_inches='tight')
    plt.close(fig)

    # -- FC vs MI scatter --
    rows, cols_idx = np.triu_indices(n, k=1)
    fc_vals  = fc[rows, cols_idx]
    mi_vals  = mi[rows, cols_idx]
    cl_mask  = cl_adj[rows, cols_idx] > 0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mi_vals[~cl_mask], fc_vals[~cl_mask],
               s=8, alpha=0.4, color='#aaaaaa', label='Non-CL')
    ax.scatter(mi_vals[cl_mask], fc_vals[cl_mask],
               s=30, alpha=0.9, color='#d62728', zorder=5,
               label=f'CL edges (n={cl_mask.sum()})')
    ax.set_xlabel('MI (nats)'); ax.set_ylabel('Pearson r')
    ax.set_title(f'{tag} — GCSS Motor FC vs MI', fontsize=9)
    ax.legend(fontsize=8)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{tag}_gcss_fc_vs_mi.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f'    4 figures → {out_dir}')


# ---------------------------------------------------------------------------
# Per-subject pipeline
# ---------------------------------------------------------------------------

def run_subject(
    subject: str,
    sessions: list,
    num_bins: int,
    device,
    make_figs: bool,
    compute_concat: bool,
    force: bool,
) -> None:
    print(f'\n{"="*55}\nSubject: {subject}')
    figs_dir = _figures_dir(subject)

    # ---- Per-session ----
    for session in sessions:
        fc_path = os.path.join(
            _connectivity_dir('fc', subject, session),
            f'{subject}_{session}_gcss_fc.npy',
        )
        if not force and os.path.exists(fc_path):
            print(f'  {session}: already computed, skipping (--force to overwrite)')
            continue

        ts, roi_keys = load_session_timeseries(subject, session)
        if ts is None:
            print(f'  {session}: timeseries not found — skipping')
            continue

        n_trs, n_rois = ts.shape
        print(f'  {session}: {n_trs} TRs × {n_rois} ROIs', end=' ... ', flush=True)

        fc      = compute_fc(ts)
        mi      = compute_mi(ts, num_bins=num_bins, device=device)
        tree, cl_adj = compute_cl(mi)

        save_connectivity(subject, session, fc, mi, cl_adj, roi_keys)

        if make_figs:
            make_figures(subject, session, fc, mi, cl_adj, tree, roi_keys, figs_dir)

        print('done')

    # ---- Concatenated (all sessions) ----
    if not compute_concat:
        return

    concat_fc_path = os.path.join(
        _connectivity_dir('fc', subject, 'all_sessions'),
        f'{subject}_all_sessions_gcss_fc.npy',
    )
    if not force and os.path.exists(concat_fc_path):
        print(f'  all_sessions: already computed, skipping')
        return

    ts_concat, roi_keys = load_concat_timeseries(subject, sessions)
    if ts_concat is None:
        print(f'  all_sessions: no timeseries found')
        return

    n_trs, n_rois = ts_concat.shape
    print(f'  all_sessions: {n_trs} TRs × {n_rois} ROIs (concatenated) ...', end=' ', flush=True)

    fc       = compute_fc(ts_concat)
    mi       = compute_mi(ts_concat, num_bins=num_bins, device=device)
    tree, cl_adj = compute_cl(mi)

    save_connectivity(subject, 'all_sessions', fc, mi, cl_adj, roi_keys)

    if make_figs:
        make_figures(subject, 'all_sessions', fc, mi, cl_adj, tree, roi_keys, figs_dir)

    print('done')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='FC, MI, and CL tree analysis for GCSS motor fROI timeseries.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--subjects', nargs='+',
                   default=[f'MSC{i:02d}' for i in range(1, 11)])
    p.add_argument('--sessions', nargs='+',
                   default=[f'func{i:02d}' for i in range(1, 11)])
    p.add_argument('--num-bins', type=int, default=100,
                   help='Bins for MI discretization (100 recommended for concatenated; '
                        'consider 50 for per-session if ROI count is large).')
    p.add_argument('--no-figures', action='store_true',
                   help='Skip figure generation.')
    p.add_argument('--no-concat', action='store_true',
                   help='Skip computing connectivity on concatenated timeseries.')
    p.add_argument('--force', action='store_true',
                   help='Recompute and overwrite existing outputs.')
    return p.parse_args()


def main():
    args   = parse_args()
    device = detect_device()

    print('=== GCSS Motor Connectivity: FC / MI / CL ===')
    print(f'Subjects : {args.subjects}')
    print(f'Sessions : {args.sessions}')
    print(f'MI bins  : {args.num_bins}')
    print(f'Concat   : {not args.no_concat}')

    for subject in args.subjects:
        run_subject(
            subject        = subject,
            sessions       = args.sessions,
            num_bins       = args.num_bins,
            device         = device,
            make_figs      = not args.no_figures,
            compute_concat = not args.no_concat,
            force          = args.force,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
