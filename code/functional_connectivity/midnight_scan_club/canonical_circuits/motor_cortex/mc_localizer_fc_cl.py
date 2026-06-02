"""
FC vs. Chow-Liu tree analysis for localizer-defined motor ROIs.

Reproduces and extends the notebook analysis in msc_fc_cl_motor.ipynb /
msc_fc_cl_motor_fig.ipynb using the subject-specific localizer ROI timeseries
produced by msc_localizer_roi_timeseries.py.

For each subject:
  1. Load and concatenate localizer timeseries across all sessions.
  2. Compute FC (Pearson correlation) matrix.
  3. Compute pairwise MI matrix (same discretization as the main pipeline).
  4. Build the Chow-Liu tree (max spanning tree of MI).
  5. Save matrices and produce 5 figures.

Figures
-------
  Fig 1 — FC matrix heatmap (38 × 38)
  Fig 2 — MI matrix heatmap (38 × 38)
  Fig 3 — Hierarchical CL tree (nodes colored by effector type)
  Fig 4 — FC vs MI scatter (all pairs; CL edges highlighted)
  Fig 5 — FC adjacency vs CL adjacency side-by-side heatmaps

Usage
-----
    python mc_localizer_fc_cl.py [--subjects MSC01 ...] [--sessions func01 ...]
                                  [--num-bins 100] [--no-figures]
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

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # motor_cortex/
_CANONICAL_DIR = os.path.dirname(_SCRIPT_DIR)                  # canonical_circuits/
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)               # midnight_scan_club/
sys.path.insert(0, _MSC_DIR)        # msc_paths, msc_mutual_info, msc_chow_liu
sys.path.insert(0, _CANONICAL_DIR)  # canonical_utils

from msc_paths import detect_device
from msc_mutual_info import pairwise_roi_mutual_information
from msc_chow_liu import chow_liu_tree, draw_hierarchical_tree
from canonical_utils import (
    cc_analysis_dir, cc_figures_dir,
    load_localizer_timeseries,
)

# ---------------------------------------------------------------------------
# ROI color scheme
# ---------------------------------------------------------------------------

_EFFECTOR_COLORS = {
    'foot':   '#d62728',   # red
    'hand':   '#1f77b4',   # blue
    'tongue': '#2ca02c',   # green
    'whole':  '#aaaaaa',   # gray
}


def _roi_color(label: str) -> str:
    """Return a hex color for a localizer ROI key (e.g. 'foot__Right_4')."""
    prefix = label.split('__')[0]
    return _EFFECTOR_COLORS.get(prefix, '#dddddd')


def _short_label(label: str) -> str:
    """Strip effector prefix for display: 'foot__Right_4' → 'Right_4'."""
    parts = label.split('__', 1)
    return parts[1] if len(parts) == 2 else label


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def _analysis_out(submodule: str = 'motor_cortex') -> str:
    return cc_analysis_dir(submodule)


def _figures_out(submodule: str = 'motor_cortex') -> str:
    return cc_figures_dir(submodule)


# ---------------------------------------------------------------------------
# Step 2: Functional connectivity (Pearson correlation)
# ---------------------------------------------------------------------------

def compute_fc(ts_concat: np.ndarray) -> np.ndarray:
    """Return Pearson correlation matrix; diagonal set to 0.

    Parameters
    ----------
    ts_concat : (n_TRs, n_ROIs)
    """
    corr = np.corrcoef(ts_concat.T)   # (n_ROIs, n_ROIs)
    np.fill_diagonal(corr, 0.0)
    return corr


# ---------------------------------------------------------------------------
# Step 3: Mutual information matrix
# ---------------------------------------------------------------------------

def compute_mi(ts_concat: np.ndarray, num_bins: int, device) -> np.ndarray:
    """Pairwise MI matrix via the same pipeline as the main analysis.

    Parameters
    ----------
    ts_concat : (n_TRs, n_ROIs)
    """
    ts_t = torch.tensor(
        ts_concat.T.astype(np.float32), device=device
    )  # (n_ROIs, n_TRs)
    mi, _, _, _ = pairwise_roi_mutual_information(ts_t, num_bins=num_bins)
    return mi.cpu().numpy()


# ---------------------------------------------------------------------------
# Save / load matrices
# ---------------------------------------------------------------------------

def save_matrices(
    subject: str,
    fc: np.ndarray,
    mi: np.ndarray,
    cl_adj: np.ndarray,
    roi_keys: list,
    out_dir: str,
) -> None:
    np.save(os.path.join(out_dir, f'{subject}_localizer_fc.npy'), fc)
    np.save(os.path.join(out_dir, f'{subject}_localizer_mi.npy'), mi)
    np.save(os.path.join(out_dir, f'{subject}_localizer_cl_adj.npy'), cl_adj)
    with open(os.path.join(out_dir, f'{subject}_localizer_roi_keys.json'), 'w') as f:
        json.dump(roi_keys, f, indent=2)
    print(f'  Saved matrices → {out_dir}')


# ---------------------------------------------------------------------------
# Figure 1 — FC matrix heatmap
# ---------------------------------------------------------------------------

def fig_fc_matrix(fc: np.ndarray, roi_keys: list, subject: str, out_dir: str) -> None:
    short = [_short_label(k) for k in roi_keys]
    n = len(roi_keys)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.3), max(7, n * 0.28)))
    im = ax.imshow(fc, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r', fraction=0.03)
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=5)
    _draw_effector_dividers(ax, roi_keys)
    ax.set_title(f'{subject} — Localizer ROI Functional Connectivity (FC)', fontsize=9)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f'{subject}_localizer_fc_matrix.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 1 → {fpath}')


# ---------------------------------------------------------------------------
# Figure 2 — MI matrix heatmap
# ---------------------------------------------------------------------------

def fig_mi_matrix(mi: np.ndarray, roi_keys: list, subject: str, out_dir: str) -> None:
    short = [_short_label(k) for k in roi_keys]
    n = len(roi_keys)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.3), max(7, n * 0.28)))
    im = ax.imshow(mi, cmap='viridis', vmin=0, aspect='auto')
    plt.colorbar(im, ax=ax, label='Mutual Information (nats)', fraction=0.03)
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=5)
    _draw_effector_dividers(ax, roi_keys)
    ax.set_title(f'{subject} — Localizer ROI Mutual Information', fontsize=9)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f'{subject}_localizer_mi_matrix.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 2 → {fpath}')


# ---------------------------------------------------------------------------
# Figure 3 — Hierarchical CL tree (effector-colored)
# ---------------------------------------------------------------------------

def fig_cl_tree(
    tree: nx.Graph,
    roi_keys: list,
    subject: str,
    out_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))
    draw_hierarchical_tree(
        tree, roi_keys,
        title=f'{subject} — Localizer Motor ROI Chow-Liu Tree',
        ax=ax,
        color_fn=_roi_color,
        label_fn=_short_label,
        label_fontsize=6,
    )
    # Manual legend for effector colors
    patches = [
        mpatches.Patch(color=col, label=eff.capitalize())
        for eff, col in _EFFECTOR_COLORS.items()
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f'{subject}_localizer_cl_tree.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 3 → {fpath}')


# ---------------------------------------------------------------------------
# Figure 4 — FC vs MI scatter (CL edges highlighted)
# ---------------------------------------------------------------------------

def fig_fc_vs_mi(
    fc: np.ndarray,
    mi: np.ndarray,
    cl_adj: np.ndarray,
    subject: str,
    out_dir: str,
) -> None:
    n = fc.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    fc_vals  = fc[rows, cols]
    mi_vals  = mi[rows, cols]
    cl_mask  = cl_adj[rows, cols] > 0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mi_vals[~cl_mask], fc_vals[~cl_mask],
               s=8, alpha=0.4, color='#aaaaaa', label='Non-CL pairs')
    ax.scatter(mi_vals[cl_mask], fc_vals[cl_mask],
               s=30, alpha=0.9, color='#d62728', zorder=5,
               label=f'CL edges (n={cl_mask.sum()})')
    ax.set_xlabel('Mutual Information (nats)', fontsize=9)
    ax.set_ylabel('Pearson r (FC)', fontsize=9)
    ax.set_title(f'{subject} — FC vs MI for Localizer Motor ROIs', fontsize=9)
    ax.legend(fontsize=8)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f'{subject}_localizer_fc_vs_mi.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 4 → {fpath}')


# ---------------------------------------------------------------------------
# Figure 5 — FC adjacency vs CL adjacency side-by-side
# ---------------------------------------------------------------------------

def fig_fc_vs_cl_heatmaps(
    fc: np.ndarray,
    cl_adj: np.ndarray,
    roi_keys: list,
    subject: str,
    out_dir: str,
    fc_threshold: float = 0.3,
) -> None:
    short = [_short_label(k) for k in roi_keys]
    n = len(roi_keys)

    # Threshold FC to retain only strong positive connections
    fc_thresh = np.where(fc >= fc_threshold, fc, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n * 0.55), max(6, n * 0.25)))
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], fc_thresh, f'FC (r ≥ {fc_threshold})', 'Reds', 0, 1),
        (axes[1], cl_adj,    'CL Tree (MI weight)',       'Purples', 0, None),
    ]:
        vmax_ = vmax if vmax is not None else mat.max() if mat.max() > 0 else 1
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax_, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=5)
        ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=5)
        ax.set_title(title, fontsize=9)
        _draw_effector_dividers(ax, roi_keys)

    fig.suptitle(f'{subject} — Localizer Motor ROIs: FC vs CL Adjacency', fontsize=10)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f'{subject}_localizer_fc_vs_cl_heatmaps.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 5 → {fpath}')


# ---------------------------------------------------------------------------
# Helper: draw divider lines separating effector groups in a matrix
# ---------------------------------------------------------------------------

def _draw_effector_dividers(ax, roi_keys: list) -> None:
    """Draw thin lines in a matrix heatmap at transitions between effector groups."""
    n = len(roi_keys)
    prefixes = [k.split('__')[0] for k in roi_keys]
    for i in range(1, n):
        if prefixes[i] != prefixes[i - 1]:
            ax.axhline(i - 0.5, color='white', lw=1.0)
            ax.axvline(i - 0.5, color='white', lw=1.0)


# ---------------------------------------------------------------------------
# Main per-subject pipeline
# ---------------------------------------------------------------------------

def run_subject(
    subject: str,
    sessions: list,
    num_bins: int,
    device,
    make_figures: bool,
    analysis_dir: str,
    figures_dir: str,
) -> None:
    print(f'\n{"="*55}')
    print(f'Subject: {subject}')

    ts_concat, roi_keys = load_localizer_timeseries(subject, sessions)
    if ts_concat is None:
        print(f'  No timeseries found — skipping.')
        return

    n_trs, n_rois = ts_concat.shape
    print(f'  Timeseries: {n_trs} TRs × {n_rois} ROIs')

    # --- FC ---
    print('  Computing FC (Pearson)...', end=' ', flush=True)
    fc = compute_fc(ts_concat)
    print('done')

    # --- MI ---
    print(f'  Computing MI ({num_bins} bins)...', end=' ', flush=True)
    mi = compute_mi(ts_concat, num_bins=num_bins, device=device)
    print('done')

    # --- CL tree ---
    _, tree = chow_liu_tree(mi)
    cl_adj  = nx.to_numpy_array(tree, weight='weight')
    n_edges = tree.number_of_edges()
    print(f'  CL tree: {n_rois} nodes, {n_edges} edges')

    # --- Save matrices ---
    save_matrices(subject, fc, mi, cl_adj, roi_keys, analysis_dir)

    # --- Figures ---
    if make_figures:
        fig_fc_matrix(fc, roi_keys, subject, figures_dir)
        fig_mi_matrix(mi, roi_keys, subject, figures_dir)
        fig_cl_tree(tree, roi_keys, subject, figures_dir)
        fig_fc_vs_mi(fc, mi, cl_adj, subject, figures_dir)
        fig_fc_vs_cl_heatmaps(fc, cl_adj, roi_keys, subject, figures_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='FC and CL tree analysis for localizer-defined motor ROIs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--subjects', nargs='+',
                   default=[f'MSC{i:02d}' for i in range(1, 11)])
    p.add_argument('--sessions', nargs='+',
                   default=[f'func{i:02d}' for i in range(1, 11)])
    p.add_argument('--num-bins', type=int, default=100,
                   help='Bins for MI discretization.')
    p.add_argument('--no-figures', action='store_true',
                   help='Skip figure generation.')
    return p.parse_args()


def main():
    args = parse_args()
    device = detect_device()

    adir = _analysis_out('motor_cortex')
    fdir = _figures_out('motor_cortex')

    print('=== Localizer ROI FC vs CL Analysis ===')
    print(f'Subjects : {args.subjects}')
    print(f'Sessions : {args.sessions}')
    print(f'MI bins  : {args.num_bins}')

    for subject in args.subjects:
        run_subject(
            subject    = subject,
            sessions   = args.sessions,
            num_bins   = args.num_bins,
            device     = device,
            make_figures = not args.no_figures,
            analysis_dir = adir,
            figures_dir  = fdir,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
