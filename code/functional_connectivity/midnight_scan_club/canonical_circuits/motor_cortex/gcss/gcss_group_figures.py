"""
gcss_group_figures.py — Publication-quality figures and tables for the GCSS motor network.

Figures
-------
  Fig 1   Pipeline schematic (methods panel)
  Fig 4   Group-mean FC and MI matrices
  Fig 5   Consensus CL tree (frequency-weighted group graph)
  Fig 6   Per-subject CL trees (10-panel mosaic)
  Fig 7   Hub node degree distribution
  Fig 8   FC vs MI scatter with CL annotations
  Fig 9   CL tree drawn on the brain (connectome style)
           — group canonical + all subjects if --all-subjects
  Fig 10  Aesthetic CL tree (poster / graphical-abstract version)
           — group consensus + all subjects if --all-subjects
  Fig 11  Per-subject CL networks on the brain (brain mosaic, 2×5)

Tables
------
  Table 1  Group-mean pairwise connectivity (FC, MI, CL frequency)
  Table 2  Per-subject summary statistics

Usage
-----
  python gcss_group_figures.py                        # all figures + tables
  python gcss_group_figures.py --figs 5 9 10          # specific figures
  python gcss_group_figures.py --figs 9 10 --all-subjects  # include per-subject variants
  python gcss_group_figures.py --tables-only
"""

import argparse
import io
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # gcss/
_MOTOR_DIR     = os.path.dirname(_SCRIPT_DIR)                  # motor_cortex/
_CANONICAL_DIR = os.path.dirname(_MOTOR_DIR)                   # canonical_circuits/
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)               # midnight_scan_club/
sys.path.insert(0, _MSC_DIR)
sys.path.insert(0, _CANONICAL_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from msc_paths import BASE_DIR

try:
    from gcss_timeseries import load_contrast_zmap, ODD_RUNS, semantic_roi_name
    _TIMESERIES_OK = True
except Exception:
    _TIMESERIES_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS = [f'MSC{i:02d}' for i in range(1, 11)]

CONN_DIR   = os.path.join(BASE_DIR, 'datasets', 'midnight_scan_club', 'connectivity')
_GCSS_ANA  = os.path.join(_MSC_DIR, 'analysis', 'canonical_circuits',
                           'motor_cortex', 'gcss')
_FROI_DIR  = os.path.join(_GCSS_ANA, 'froi_masks')
_GMAP_DIR  = os.path.join(_GCSS_ANA, 'group_maps')
_CMAP_DIR  = os.path.join(_GCSS_ANA, 'cluster_maps')

OUT_BASE     = os.path.join(_MSC_DIR, 'figures', 'canonical_circuits',
                            'motor_cortex', 'gcss', 'analysis')
PUB_DIR      = os.path.join(OUT_BASE, 'pub_figures')
TAB_DIR      = os.path.join(OUT_BASE, 'tables')
SUBJ_FIG_BASE = os.path.join(_MSC_DIR, 'figures', 'canonical_circuits',
                              'motor_cortex', 'gcss')

# Alphabetical key order as stored in roi_metadata.json
ROI_KEYS = ['L_M1_foot', 'L_M1_hand', 'L_M1_tongue', 'L_PMd', 'L_S1S2',
            'R_M1_foot', 'R_M1_hand', 'R_M1_tongue', 'R_PMv', 'R_S1S2', 'SMA']

# Somatotopic order for matrices
MATRIX_ORDER = ['R_M1_foot', 'L_M1_foot',
                'R_M1_hand', 'L_M1_hand',
                'R_M1_tongue', 'L_M1_tongue',
                'R_PMv', 'R_S1S2', 'L_S1S2', 'L_PMd', 'SMA']

_GROUP_BOUNDARIES = [2, 4, 6]

SEMANTIC_COLORS = {
    'R_M1_foot':   '#e31a1c',
    'L_M1_foot':   '#fb9a99',
    'R_M1_hand':   '#1f78b4',
    'L_M1_hand':   '#a6cee3',
    'L_M1_tongue': '#33a02c',
    'R_M1_tongue': '#b2df8a',
    'SMA':         '#ff7f00',
    'L_PMd':       '#e377c2',
    'R_PMv':       '#9467bd',
    'L_S1S2':      '#17becf',
    'R_S1S2':      '#8c564b',
}

# Two-line short labels for node interiors
_SHORT = {
    'R_M1_foot':   'R M1\nfoot',
    'L_M1_foot':   'L M1\nfoot',
    'R_M1_hand':   'R M1\nhand',
    'L_M1_hand':   'L M1\nhand',
    'R_M1_tongue': 'R M1\ntongue',
    'L_M1_tongue': 'L M1\ntongue',
    'SMA':         'SMA',
    'L_PMd':       'L PMd',
    'R_PMv':       'R PMv',
    'L_S1S2':      'L S1/S2',
    'R_S1S2':      'R S1/S2',
}

# Fixed anatomical layout (unit square)
_TREE_POS = {
    'SMA':         (0.50, 0.90),
    'L_PMd':       (0.16, 0.72),
    'R_PMv':       (0.76, 0.72),
    'L_M1_hand':   (0.10, 0.52),
    'R_M1_tongue': (0.62, 0.58),
    'L_M1_tongue': (0.88, 0.58),
    'R_M1_hand':   (0.10, 0.32),
    'R_S1S2':      (0.62, 0.38),
    'L_S1S2':      (0.88, 0.38),
    'R_M1_foot':   (0.22, 0.12),
    'L_M1_foot':   (0.43, 0.12),
}

# MNI centroids for connectome brain figure
NODE_COORDS_MNI = {
    'R_M1_foot':   (  5.8, -25.0,  69.7),
    'L_M1_foot':   ( -5.8, -22.8,  68.9),
    'R_M1_hand':   ( 39.5, -20.9,  57.7),
    'L_M1_hand':   (-39.4, -23.3,  57.4),
    'R_M1_tongue': ( 57.6,  -5.1,  28.6),
    'L_M1_tongue': (-57.5,  -8.7,  30.0),
    'SMA':         ( -1.8,  -4.6,  61.7),
    'L_PMd':       (-45.4,  -8.8,  58.8),
    'R_PMv':       ( 57.1,   1.5,  35.1),
    'L_S1S2':      (-58.8, -11.5,  32.4),
    'R_S1S2':      ( 58.0, -13.9,  27.1),
}

# Edge category colours (used in Figs 6, 10 per-subject)
_EDGE_CAT_COLORS = {
    'canonical':     '#222222',   # ≥ 7/10
    'intermediate':  '#777777',   # 4–6/10
    'idiosyncratic': '#bbbbbb',   # 1–3/10
}

_SUBJECTS_MI_ORDER = ['MSC05', 'MSC09', 'MSC03', 'MSC01', 'MSC02',
                       'MSC04', 'MSC07', 'MSC10', 'MSC06', 'MSC08']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_cat(freq: int) -> str:
    if freq >= 7: return 'canonical'
    if freq >= 4: return 'intermediate'
    return 'idiosyncratic'


def _make_adj(mat: np.ndarray) -> np.ndarray:
    a = mat.copy()
    np.fill_diagonal(a, 0)
    return a


def _nilearn_to_img(disp) -> np.ndarray:
    """
    Rasterise a nilearn display or matplotlib Figure to an RGBA array.

    plot_stat_map / plot_roi return nilearn display objects (with .frame_axes);
    plot_surf_roi returns a plain matplotlib Figure.  Both are handled here.
    """
    if hasattr(disp, 'frame_axes'):
        fig = disp.frame_axes.figure
    else:
        fig = disp   # already a matplotlib Figure
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _apply_style():
    matplotlib.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':          8,
        'axes.titlesize':     9,
        'axes.labelsize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'axes.linewidth':     0.8,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    })


def _save(fig, name: str, out_dir: str = None) -> None:
    target = out_dir if out_dir is not None else PUB_DIR
    os.makedirs(target, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(target, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    rel = os.path.relpath(target)
    print(f'  saved → {rel}/{name}.{{pdf,png}}')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_group_data():
    """
    Load all-sessions FC, MI, CL matrices for all 10 subjects.

    Returns
    -------
    fc_arr  : (10, 11, 11) float32
    mi_arr  : (10, 11, 11) float32
    cl_arr  : (10, 11, 11) float32  — MI weight on CL edges, 0 elsewhere
    keys    : list[str]  length 11 (alphabetical ROI order)
    """
    fc_all, mi_all, cl_all = [], [], []
    for sub in SUBJECTS:
        d_fc = os.path.join(CONN_DIR, 'fc', sub, 'all_sessions', 'gcss_motor')
        d_mi = os.path.join(CONN_DIR, 'mi', sub, 'all_sessions', 'gcss_motor')
        d_cl = os.path.join(CONN_DIR, 'cl', sub, 'all_sessions', 'gcss_motor')
        fc_all.append(np.load(os.path.join(d_fc, f'{sub}_all_sessions_gcss_fc.npy')))
        mi_all.append(np.load(os.path.join(d_mi, f'{sub}_all_sessions_gcss_mi.npy')))
        cl_all.append(np.load(os.path.join(d_cl, f'{sub}_all_sessions_gcss_cl.npy')))

    keys_path = os.path.join(CONN_DIR, 'fc', 'MSC01', 'all_sessions', 'gcss_motor',
                              'MSC01_all_sessions_gcss_roi_keys.json')
    with open(keys_path) as fh:
        keys = json.load(fh)

    return (np.stack(fc_all).astype(np.float32),
            np.stack(mi_all).astype(np.float32),
            np.stack(cl_all).astype(np.float32),
            keys)


def _derived(fc_arr, mi_arr, cl_arr, keys):
    """Pre-compute commonly needed group-level quantities."""
    n  = len(keys)
    fc_mean = fc_arr.mean(axis=0)
    mi_mean = mi_arr.mean(axis=0)
    fc_std  = fc_arr.std(axis=0)
    mi_std  = mi_arr.std(axis=0)
    cl_bin  = (cl_arr > 0).astype(int)
    ef      = cl_bin.sum(axis=0)              # (11, 11) edge frequency
    mean_deg = cl_bin.sum(axis=2).mean(axis=0)
    mat_idx  = [keys.index(r) for r in MATRIX_ORDER]
    return dict(
        fc_mean=fc_mean, mi_mean=mi_mean, fc_std=fc_std, mi_std=mi_std,
        cl_bin=cl_bin, ef=ef, mean_deg=mean_deg, mat_idx=mat_idx, n=n,
    )


# ---------------------------------------------------------------------------
# Helper: draw matrix dividers
# ---------------------------------------------------------------------------

def _matrix_dividers(ax, n_matrix=11):
    for b in _GROUP_BOUNDARIES:
        ax.axhline(b - 0.5, color='white', lw=1.2)
        ax.axvline(b - 0.5, color='white', lw=1.2)


# ---------------------------------------------------------------------------
# Figure 1 — Pipeline Schematic
# ---------------------------------------------------------------------------

def fig_pipeline_schematic(subject: str = 'MSC01',
                            contrast: str = 'combined_motor') -> None:
    """
    Three-row schematic: group probability map → cluster segmentation →
    subject z-map with fROI overlay.

    Each nilearn panel is rendered independently to a BytesIO buffer then
    stitched into a plain matplotlib figure — avoids the SubFigure
    incompatibility with nilearn's internal figure management.
    """
    from nilearn import plotting, datasets as nlds, image as nlimage
    import nibabel as nib

    print('Fig 1 — Pipeline schematic...')

    prob_path    = os.path.join(_GMAP_DIR, f'{contrast}_prob.nii.gz')
    cluster_path = os.path.join(_CMAP_DIR, f'{contrast}_clusters.nii.gz')
    info_path    = os.path.join(_CMAP_DIR, f'{contrast}_cluster_info.json')

    if not all(os.path.exists(p) for p in (prob_path, cluster_path, info_path)):
        print(f'  WARNING: group maps not found for {contrast}, skipping Fig 1')
        return

    prob_img    = nib.load(prob_path)
    cluster_img = nib.load(cluster_path)

    with open(info_path) as fh:
        raw = json.load(fh)
    cluster_info = {int(k): v for k, v in raw.items()}

    template = nlds.load_mni152_template(resolution=2)
    Z_CUTS = [28, 44, 58, 66]

    # Build semantic ROI volume
    cluster_data = np.asarray(cluster_img.dataobj, dtype=np.int16)
    roi_vol = np.zeros_like(cluster_data, dtype=np.float32)
    sem_names_ordered = []
    for cid, info in sorted(cluster_info.items()):
        sem = semantic_roi_name(contrast, info) if _TIMESERIES_OK else info.get('label', f'cluster_{cid}')
        if sem not in sem_names_ordered:
            sem_names_ordered.append(sem)
        roi_vol[cluster_data == cid] = sem_names_ordered.index(sem) + 1
    roi_img = nib.Nifti1Image(roi_vol, cluster_img.affine)

    n_colors = max(len(sem_names_ordered), 2)
    roi_cmap = ListedColormap([SEMANTIC_COLORS.get(s, '#aaaaaa') for s in sem_names_ordered])

    # Subject z-map
    zmap_img = None
    if _TIMESERIES_OK:
        try:
            zmap_img = load_contrast_zmap(subject, contrast, ODD_RUNS)
            if zmap_img is not None:
                zmap_img = nlimage.resample_to_img(zmap_img, template, interpolation='continuous')
        except Exception:
            zmap_img = None

    # Subject fROI masks
    froi_dir = os.path.join(_FROI_DIR, subject)
    froi_combined = np.zeros(np.asarray(template.dataobj).shape[:3], dtype=np.float32)
    if os.path.isdir(froi_dir):
        for ri, roi in enumerate(sem_names_ordered, start=1):
            mpath = os.path.join(froi_dir, f'{subject}_gcss_{roi}_mask.nii.gz')
            if os.path.exists(mpath):
                m = nlimage.resample_to_img(nib.load(mpath), template, interpolation='nearest')
                froi_combined[np.asarray(m.dataobj, dtype=bool)] = float(ri)
    froi_img = nib.Nifti1Image(froi_combined, template.affine)

    # ---- Render each panel independently to a buffer ----
    disp_a = plotting.plot_stat_map(
        prob_img, display_mode='z', cut_coords=Z_CUTS,
        threshold=0.6, cmap='YlOrRd', vmax=1.0, colorbar=True,
        title=f'A   Group probability map  ({contrast}, threshold ≥ 0.60)',
    )
    disp_a.add_contours(roi_img, levels=np.arange(0.5, n_colors + 0.5),
                        colors=['white'], linewidths=[0.8])
    img_a = _nilearn_to_img(disp_a)

    disp_b = plotting.plot_roi(
        roi_img, bg_img=template, display_mode='z', cut_coords=Z_CUTS,
        cmap=roi_cmap, vmin=0.5, vmax=n_colors + 0.5,
        colorbar=False, alpha=0.85,
        title=f'B   Group cluster map  ({len(sem_names_ordered)} semantic ROIs)',
    )
    patches_b = [mpatches.Patch(color=SEMANTIC_COLORS.get(s, '#aaaaaa'), label=s)
                 for s in sem_names_ordered]
    disp_b.frame_axes.figure.legend(handles=patches_b, loc='lower center', ncol=6,
                                     fontsize=6, framealpha=0.8)
    img_b = _nilearn_to_img(disp_b)

    imgs = [img_a, img_b]
    tags = ['A', 'B']

    if zmap_img is not None:
        disp_c = plotting.plot_stat_map(
            zmap_img, display_mode='z', cut_coords=Z_CUTS,
            threshold=2.3, cmap='hot', vmax=8.0, colorbar=True,
            title=f'C   {subject} z-map  (z > 2.3) with fROI masks (outlines)',
        )
        if froi_combined.any():
            disp_c.add_contours(froi_img, levels=np.arange(0.5, n_colors + 0.5),
                                colors=['cyan'], linewidths=[1.2])
        imgs.append(_nilearn_to_img(disp_c))
        tags.append('C')

    # ---- Stitch panels into a single figure ----
    n_panels = len(imgs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(13.6, 3.2 * n_panels))
    if n_panels == 1:
        axes = [axes]
    for ax, img, tag in zip(axes, imgs, tags):
        ax.imshow(img)
        ax.axis('off')
        ax.text(0.005, 0.97, tag, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', color='white',
                bbox=dict(facecolor='black', edgecolor='none', pad=1.5))
    fig.tight_layout(pad=0.3)
    _save(fig, 'fig1_pipeline_schematic')


# ---------------------------------------------------------------------------
# Figure 4 — Group FC and MI Matrices
# ---------------------------------------------------------------------------

def fig_group_matrices(d: dict, keys: list) -> None:
    """Group-mean FC (left) and MI (right) matrices in somatotopic order."""
    print('Fig 4 — Group FC and MI matrices...')

    idx    = d['mat_idx']
    fc_m   = d['fc_mean'][np.ix_(idx, idx)]
    mi_m   = d['mi_mean'][np.ix_(idx, idx)]
    ef     = d['ef'][np.ix_(idx, idx)]
    labels = MATRIX_ORDER

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))

    for ax, mat, cmap, vmin, vmax, cbar_label, title in [
        (axes[0], fc_m, 'RdBu_r', -1,  1,   'Pearson r',   'Group-mean FC'),
        (axes[1], mi_m, 'viridis', 0,  None, 'MI (nats)',   'Group-mean MI'),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
        ax.set_xticks(range(11))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6.5)
        ax.set_yticks(range(11))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(title, fontsize=9, pad=4)
        _matrix_dividers(ax)
        for i in range(11):
            for j in range(11):
                if i != j and ef[i, j] >= 7:
                    ax.plot(j, i, 'w*', ms=5, zorder=5)

    star = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='w',
                      markersize=6, linestyle='None', label='CL edge ≥ 7/10 subjects')
    axes[1].legend(handles=[star], loc='lower right', fontsize=7, framealpha=0.8)

    fig.suptitle('GCSS Motor Network — Group Connectivity (MSC01–MSC10)',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig4_group_fc_mi_matrices')


# ---------------------------------------------------------------------------
# Figure 5 — Consensus CL Tree
# ---------------------------------------------------------------------------

def fig_consensus_tree(d: dict, keys: list) -> None:
    """
    Frequency-weighted group consensus CL tree on a fixed anatomical layout.

    Nodes are drawn as matplotlib Circle patches (fixed axes-coordinate radius)
    so that the two-line _SHORT labels fit cleanly inside the node circle.
    """
    print('Fig 5 — Consensus CL tree...')

    ef       = d['ef']
    mi_mean  = d['mi_mean']
    mean_deg = d['mean_deg']
    n        = d['n']
    rows, cols = np.triu_indices(n, k=1)

    NODE_R_BASE  = 0.050
    NODE_R_RANGE = 0.025
    min_d = mean_deg.min()
    max_d = mean_deg.max()

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

    # Draw edges (width and alpha ∝ frequency)
    for i, j in zip(rows, cols):
        freq = int(ef[i, j])
        if freq < 1:
            continue
        width = 0.8 + (freq / 10) * 6.0
        alpha = 0.25 + (freq / 10) * 0.75
        color = plt.cm.Greys(0.3 + (freq / 10) * 0.65)
        xu, yu = _TREE_POS[keys[i]]
        xv, yv = _TREE_POS[keys[j]]
        ax.plot([xu, xv], [yu, yv], color=color, linewidth=width,
                alpha=alpha, zorder=1, solid_capstyle='round')
        mx, my = (xu + xv) / 2, (yu + yv) / 2
        ax.text(mx, my, f'{freq}/10', ha='center', va='center',
                fontsize=6.5, color='#333333', zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.5, alpha=0.85))

    # Draw nodes as Circle patches so labels fit inside
    for ki, roi in enumerate(keys):
        x, y = _TREE_POS[roi]
        r = NODE_R_BASE + NODE_R_RANGE * (mean_deg[ki] - min_d) / max(max_d - min_d, 1e-3)
        circle = Circle((x, y), radius=r, color=SEMANTIC_COLORS[roi],
                        zorder=5, ec='white', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, _SHORT[roi], ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white',
                zorder=6, linespacing=1.2)

    # Legend: edge thickness scale
    for freq, lbl in [(3, '3/10'), (6, '6/10'), (10, '10/10')]:
        ax.plot([], [], color='grey',
                linewidth=0.8 + (freq / 10) * 6.0,
                alpha=0.25 + (freq / 10) * 0.75,
                label=lbl)
    ax.legend(title='CL freq.', loc='lower left', fontsize=7,
              title_fontsize=7, framealpha=0.8)

    ax.set_title('GCSS Motor Network — Group Consensus CL Tree\n'
                 'Edge width ∝ frequency (N=10 subjects)',
                 fontsize=9)
    fig.tight_layout()
    _save(fig, 'fig5_consensus_cl_tree')


# ---------------------------------------------------------------------------
# Figure 6 — Per-Subject CL Trees
# ---------------------------------------------------------------------------

def fig_subject_trees(fc_arr, mi_arr, cl_arr, keys, d: dict) -> None:
    """
    2 × 5 mosaic of per-subject CL trees on the shared anatomical layout.

    Node labels are omitted (panels are too small for readable text) and
    replaced by a shared ROI colour legend below the mosaic.
    """
    print('Fig 6 — Per-subject CL trees...')

    ef     = d['ef']
    cl_bin = d['cl_bin']

    def _edge_style(i, j):
        freq = ef[i, j]
        if freq >= 7:  return '#222222', '-',  2.0
        if freq >= 4:  return '#777777', '-',  1.2
        return '#bbbbbb', '--', 0.7

    fig, axes = plt.subplots(2, 5, figsize=(13.6, 5.6))
    axes = axes.ravel()
    n = len(keys)

    for si, (sub, ax) in enumerate(zip(SUBJECTS, axes)):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis('off')

        cl_s   = cl_bin[si]
        rows_s, cols_s = np.where(np.triu(cl_s, k=1))

        for i, j in zip(rows_s, cols_s):
            col, ls, lw = _edge_style(i, j)
            xu, yu = _TREE_POS[keys[i]]
            xv, yv = _TREE_POS[keys[j]]
            ax.plot([xu, xv], [yu, yv], color=col, linewidth=lw,
                    linestyle=ls, zorder=1, solid_capstyle='round')

        deg_s = cl_s.sum(axis=1)
        for ki, roi in enumerate(keys):
            x, y = _TREE_POS[roi]
            sz   = 120 + 100 * min(deg_s[ki], 4) / 4
            ec   = 'black' if deg_s[ki] >= 3 else 'white'
            lw_n = 2.0   if deg_s[ki] >= 3 else 0.6
            ax.scatter(x, y, s=sz, c=SEMANTIC_COLORS[roi], zorder=3,
                       edgecolors=ec, linewidths=lw_n)

        mi_mean_s = mi_arr[si][np.triu_indices(n, k=1)].mean()
        ax.set_title(f'{sub}\n(MĪ={mi_mean_s:.2f})', fontsize=7.5, pad=2)

    # Edge-style legend
    legend_e = [
        plt.Line2D([0], [0], color='#222222', lw=2.0, label='Canonical (≥7/10)'),
        plt.Line2D([0], [0], color='#777777', lw=1.2, label='Intermediate (4–6/10)'),
        plt.Line2D([0], [0], color='#bbbbbb', lw=0.7, ls='--', label='Idiosyncratic (1–3/10)'),
        plt.Line2D([0], [0], marker='o', ls='None', markerfacecolor='grey',
                   markeredgecolor='black', markersize=7, markeredgewidth=1.5,
                   label='Hub node (degree ≥ 3)'),
    ]
    fig.legend(handles=legend_e, loc='lower left', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.01, -0.10))

    # ROI colour legend
    patches_roi = [mpatches.Patch(color=SEMANTIC_COLORS[r],
                                   label=_SHORT[r].replace('\n', ' '))
                   for r in keys]
    fig.legend(handles=patches_roi, loc='lower right', ncol=6,
               fontsize=6.5, framealpha=0.9, bbox_to_anchor=(0.99, -0.10),
               title='ROI colour', title_fontsize=7)

    fig.suptitle('GCSS Motor Network — Per-Subject CL Trees (all-sessions)',
                 fontsize=9, y=1.01)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    _save(fig, 'fig6_subject_cl_trees')


# ---------------------------------------------------------------------------
# Figure 7 — Hub Node Degree Distribution
# ---------------------------------------------------------------------------

def fig_hub_degrees(cl_arr, keys, d: dict) -> None:
    """Bar chart of mean degree ± SD + heatmap of per-subject degrees."""
    print('Fig 7 — Hub degree distribution...')

    cl_bin   = d['cl_bin']
    mean_deg = d['mean_deg']
    std_deg  = cl_bin.sum(axis=2).std(axis=0)
    all_deg  = cl_bin.sum(axis=2)

    sort_idx     = np.argsort(-mean_deg)
    sorted_keys  = [keys[i] for i in sort_idx]
    sorted_mean  = mean_deg[sort_idx]
    sorted_std   = std_deg[sort_idx]
    sorted_deg   = all_deg[:, sort_idx]

    subj_hm_idx  = [SUBJECTS.index(s) for s in _SUBJECTS_MI_ORDER]
    sorted_deg_hm = sorted_deg[subj_hm_idx, :]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5),
                             gridspec_kw=dict(width_ratios=[1, 1.2], wspace=0.35))

    ax = axes[0]
    bar_colors = [SEMANTIC_COLORS[k] for k in sorted_keys]
    ax.barh(range(11), sorted_mean, xerr=sorted_std,
            color=bar_colors, edgecolor='white', linewidth=0.5,
            error_kw=dict(ecolor='#555555', capsize=3, lw=1),
            zorder=2)
    ax.axvline(2.0, color='#333333', lw=1.0, ls='--', alpha=0.7, zorder=1)
    ax.text(2.05, -0.6, 'hub threshold', fontsize=6.5, color='#333333', va='top')

    rng = np.random.default_rng(42)
    for ki, row in enumerate(sorted_deg.T):
        jitter = rng.uniform(-0.22, 0.22, size=len(row))
        ax.scatter(row, np.full(len(row), ki) + jitter,
                   s=12, color='#333333', alpha=0.55, zorder=3)

    ax.set_yticks(range(11))
    ax.set_yticklabels(sorted_keys, fontsize=7)
    ax.set_xlabel('Node degree (CL tree)', fontsize=8)
    ax.set_title('Mean degree ± SD across subjects', fontsize=8)
    ax.set_xlim(0, 5.5)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)

    ax2 = axes[1]
    im = ax2.imshow(sorted_deg_hm.T, cmap='Blues', vmin=0, vmax=5,
                    aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax2, label='Degree', fraction=0.046, pad=0.04)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(_SUBJECTS_MI_ORDER, rotation=45, ha='right', fontsize=6.5)
    ax2.set_yticks(range(11))
    ax2.set_yticklabels(sorted_keys, fontsize=7)
    ax2.set_title('Per-subject degree  (columns: low → high mean MI)', fontsize=8)
    for yi in range(11):
        for xi in range(10):
            val = int(sorted_deg_hm[xi, yi])
            if val > 0:
                ax2.text(xi, yi, str(val), ha='center', va='center',
                         fontsize=6, color='white' if val >= 3 else '#333333')

    fig.suptitle('GCSS Motor Network — Hub Node Degree Distribution',
                 fontsize=9, y=1.02)
    _save(fig, 'fig7_hub_degrees')


# ---------------------------------------------------------------------------
# Figure 8 — FC vs MI Scatter
# ---------------------------------------------------------------------------

def fig_group_fc_vs_mi(fc_arr, mi_arr, keys, d: dict) -> None:
    """FC vs MI scatter: group means highlighted by CL frequency category."""
    print('Fig 8 — FC vs MI scatter...')

    ef      = d['ef']
    fc_mean = d['fc_mean']
    mi_mean = d['mi_mean']
    n       = d['n']
    rows, cols = np.triu_indices(n, k=1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for si in range(len(SUBJECTS)):
        fc_s = fc_arr[si][rows, cols]
        mi_s = mi_arr[si][rows, cols]
        ax.scatter(mi_s, fc_s, s=6, alpha=0.12, color='#999999', zorder=1)

    fc_g   = fc_mean[rows, cols]
    mi_g   = mi_mean[rows, cols]
    freq_g = ef[rows, cols]

    _cat_style = {
        'canonical':     ('#b30000', '*', 100, '≥ 7/10'),
        'intermediate':  ('#e07800', 'o',  60, '4–6/10'),
        'idiosyncratic': ('#aaaaaa', 'o',  30, '1–3/10'),
        'absent':        ('#dddddd', 'o',  18, '0/10'),
    }

    def _cat(f):
        if f >= 7: return 'canonical'
        if f >= 4: return 'intermediate'
        if f >= 1: return 'idiosyncratic'
        return 'absent'

    for cat, (col, mk, sz, lbl) in _cat_style.items():
        mask = np.array([_cat(f) == cat for f in freq_g])
        ax.scatter(mi_g[mask], fc_g[mask], s=sz, color=col, marker=mk,
                   label=f'{lbl} (n={mask.sum()})', zorder=3, edgecolors='none')

    canonical_pairs = [(i, j) for i, j in zip(rows, cols) if ef[i, j] >= 7]
    for i, j in canonical_pairs:
        lbl = f'{keys[i]}\n{keys[j]}'
        ax.annotate(lbl, (mi_mean[i, j], fc_mean[i, j]),
                    xytext=(6, 4), textcoords='offset points',
                    fontsize=5.5, color='#b30000',
                    arrowprops=dict(arrowstyle='-', color='#b30000', lw=0.5))

    all_mi = mi_arr[:, rows, cols].ravel()
    all_fc = fc_arr[:, rows, cols].ravel()
    m, b   = np.polyfit(all_mi, all_fc, 1)
    xlim   = ax.get_xlim()
    xfit   = np.array([max(0, xlim[0]), xlim[1]])
    ax.plot(xfit, m * xfit + b, color='#333333', lw=1.2, ls='--', alpha=0.7, zorder=2)

    from scipy.stats import pearsonr
    r, p = pearsonr(all_mi, all_fc)
    ax.text(0.97, 0.05, f'r = {r:.3f}, p < 0.001', transform=ax.transAxes,
            ha='right', fontsize=7, color='#333333')

    ax.set_xlabel('MI (nats)', fontsize=8)
    ax.set_ylabel('Pearson r', fontsize=8)
    ax.set_title('FC vs MI — Group (mean) and individual subjects', fontsize=9)
    ax.legend(title='CL frequency', fontsize=7, title_fontsize=7,
              loc='upper left', framealpha=0.85)
    ax.axhline(0, color='#aaaaaa', lw=0.6, ls=':')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    _save(fig, 'fig8_fc_vs_mi')


# ---------------------------------------------------------------------------
# Figure 9 — CL Tree on the Brain
# ---------------------------------------------------------------------------

def fig_cl_tree_on_brain(fc_arr, mi_arr, cl_arr, keys, d: dict,
                          subject: str = 'MSC01',
                          all_subjects: bool = False) -> None:
    """
    Connectome-style figure: CL tree edges drawn as arcs between MNI centroids.

    Always generates:
      - group_canonical  (≥ 7/10, mean MI)
      - the single `subject` variant

    With all_subjects=True, also generates one file per subject MSC01–MSC10.

    ROI labels are shown via a coloured legend below the brain (avoids white-on-
    white text that was invisible on nilearn's glass-brain background).
    """
    from nilearn import plotting

    print(f'Fig 9 — CL tree on brain (subject={subject}, all_subjects={all_subjects})...')

    n        = d['n']
    ef       = d['ef']
    mi_mean  = d['mi_mean']
    mean_deg = d['mean_deg']

    node_coords = np.array([NODE_COORDS_MNI[k] for k in keys])
    node_color  = [SEMANTIC_COLORS[k] for k in keys]
    node_size_g = 20 + 80 * (mean_deg - mean_deg.min()) / max(
                      mean_deg.max() - mean_deg.min(), 1e-3)

    nonzero = mi_arr[cl_arr > 0]
    mi_vmax = float(nonzero.max()) * 0.9 if len(nonzero) else 1.0

    def _node_size_sub(si):
        deg_s = (cl_arr[si] > 0).sum(axis=1).astype(float)
        return (20 + 80 * (deg_s - deg_s.min()) / max(deg_s.max() - deg_s.min(), 1e-3)).tolist()

    def _render_one(tag, label, adj_mi, ns):
        fig = plt.figure(figsize=(10.0, 5.0))
        disp = plotting.plot_connectome(
            adj_mi, node_coords,
            node_color=node_color,
            node_size=ns,
            edge_cmap='plasma',
            edge_vmin=0.1, edge_vmax=mi_vmax,
            display_mode='lzr',
            colorbar=True,
            title=label,
            figure=fig,
        )
        # ROI legend below brain (coloured patches — avoids white-on-white annotation)
        patches = [mpatches.Patch(color=SEMANTIC_COLORS[r],
                                   label=_SHORT[r].replace('\n', ' '))
                   for r in keys]
        fig.legend(handles=patches, loc='lower center', ncol=6,
                   fontsize=7, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.01),
                   title='ROI', title_fontsize=7)
        fig.subplots_adjust(bottom=0.14)
        _save(fig, f'fig9_cl_tree_on_brain_{tag}')

    # Group canonical
    _render_one(
        'group_canonical',
        'Group canonical CL edges (≥ 7/10 subjects, mean MI)',
        _make_adj(np.where(ef >= 7, mi_mean, 0.0)),
        node_size_g.tolist(),
    )

    # Subjects to plot
    subs = SUBJECTS if all_subjects else [subject]
    for sub in subs:
        si = SUBJECTS.index(sub)
        _render_one(
            sub,
            f'{sub} — GCSS Motor CL Tree (all sessions)',
            _make_adj(cl_arr[si] * mi_arr[si]),
            _node_size_sub(si),
        )


# ---------------------------------------------------------------------------
# Figure 10 — Aesthetic CL Tree
# ---------------------------------------------------------------------------

def fig_aesthetic_tree(d: dict, keys: list,
                        cl_arr=None, mi_arr=None,
                        subject: str = None,
                        out_dir: str = None) -> None:
    """
    Large-node, high-contrast CL tree for posters and graphical abstracts.

    Group version (subject=None):
      - edges ≥ 4/10, width/colour ∝ CL frequency, labelled "N/10"

    Per-subject version (subject='MSC01' etc.):
      - all edges in that subject's CL tree
      - colour by group-frequency category (canonical/intermediate/idiosyncratic)
      - width ∝ subject MI value
      - labelled with edge MI weight
    """
    if subject is not None:
        print(f'Fig 10 — Aesthetic CL tree ({subject})...')
    else:
        print('Fig 10 — Aesthetic CL tree (group)...')

    ef       = d['ef']
    mi_mean  = d['mi_mean']
    mean_deg = d['mean_deg']
    n        = d['n']
    rows, cols = np.triu_indices(n, k=1)

    NODE_R = 0.065
    FIG_BG = '#f4f4f4'

    # Per-subject setup
    if subject is not None:
        si    = SUBJECTS.index(subject)
        cl_s  = (cl_arr[si] > 0)
        mi_s  = mi_arr[si]
        mi_max = float(mi_s.max()) if mi_s.max() > 0 else 1.0
        deg_s  = cl_s.sum(axis=1).astype(float)
        mean_deg_s = deg_s
    else:
        freq_cmap = matplotlib.colormaps['Greys']
        freq_norm = Normalize(vmin=3, vmax=10)

    fig, ax = plt.subplots(figsize=(11.0, 9.0), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.08, 1.05)
    ax.axis('off')

    # Draw edges
    for i, j in zip(rows, cols):
        xu, yu = _TREE_POS[keys[i]]
        xv, yv = _TREE_POS[keys[j]]

        if subject is not None:
            if not cl_s[i, j]:
                continue
            freq  = int(ef[i, j])
            color = _EDGE_CAT_COLORS[_edge_cat(freq)]
            lw    = 1.5 + (mi_s[i, j] / mi_max) * 8.5
            edge_label = f'MI={mi_s[i,j]:.3f}'
        else:
            freq = int(ef[i, j])
            if freq < 4:
                continue
            lw    = 1.5 + (freq - 4) / 6 * 8.5
            color = freq_cmap(freq_norm(freq))
            edge_label = f'{freq}/10'

        arrow = FancyArrowPatch(
            (xu, yu), (xv, yv),
            connectionstyle='arc3,rad=0.08',
            arrowstyle='-',
            linewidth=lw,
            color=color,
            zorder=1,
            transform=ax.transData,
        )
        ax.add_patch(arrow)

        mx = (xu + xv) / 2
        my = (yu + yv) / 2 + 0.015
        ax.text(mx, my, edge_label,
                ha='center', va='center', fontsize=9, color='#333333',
                fontweight='bold', zorder=4,
                bbox=dict(facecolor=FIG_BG, edgecolor='none', pad=1.5, alpha=0.9))

    # Draw nodes
    deg_use = (mean_deg_s if subject is not None else mean_deg)
    min_d = deg_use.min()
    max_d = deg_use.max()
    for ki, roi in enumerate(keys):
        x, y = _TREE_POS[roi]
        circle = Circle((x, y), radius=NODE_R,
                         color=SEMANTIC_COLORS[roi],
                         zorder=5, ec='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, _SHORT[roi],
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='white',
                zorder=6, linespacing=1.2)

    # Legend
    if subject is not None:
        legend_handles = [
            plt.Line2D([0], [0], color=_EDGE_CAT_COLORS['canonical'],
                       linewidth=5, label='Canonical (≥7/10)'),
            plt.Line2D([0], [0], color=_EDGE_CAT_COLORS['intermediate'],
                       linewidth=3, label='Intermediate (4–6/10)'),
            plt.Line2D([0], [0], color=_EDGE_CAT_COLORS['idiosyncratic'],
                       linewidth=1.5, label='Idiosyncratic (1–3/10)'),
        ]
        ax.legend(handles=legend_handles, title='Group CL category',
                  loc='lower left', fontsize=10, title_fontsize=10,
                  framealpha=0.9, bbox_to_anchor=(0.0, 0.0))
        title_str = f'{subject} — GCSS Motor Chow-Liu Tree\nEdge width ∝ MI; colour = group frequency category'
    else:
        legend_handles = []
        freq_cmap2 = matplotlib.colormaps['Greys']
        freq_norm2 = Normalize(vmin=3, vmax=10)
        for freq, lbl in [(4, '4/10'), (7, '7/10'), (10, '10/10')]:
            lw  = 1.5 + (freq - 4) / 6 * 8.5
            col = freq_cmap2(freq_norm2(freq))
            legend_handles.append(plt.Line2D([0], [0], color=col, linewidth=lw, label=lbl))
        ax.legend(handles=legend_handles, title='CL frequency', loc='lower left',
                  fontsize=10, title_fontsize=10, framealpha=0.9,
                  bbox_to_anchor=(0.0, 0.0))
        title_str = ('GCSS Motor Network — Chow-Liu Tree\n'
                     'Midnight Scan Club (N = 10), 11 motor fROIs')

    ax.set_title(title_str, fontsize=13, fontweight='bold', pad=12, color='#222222')

    caption = ('Edge thickness = MI weight; colour = fraction of subjects with this CL edge.'
               if subject is not None else
               'Edge thickness = fraction of subjects (N=10) in which the edge appears. '
               'Only edges present in ≥ 4/10 subjects shown.')
    ax.text(0.5, -0.06, caption,
            ha='center', va='top', fontsize=8, color='#555555',
            transform=ax.transAxes)

    fig.tight_layout()
    tag = subject if subject is not None else 'group'
    _save(fig, f'fig10_aesthetic_cl_tree_{tag}', out_dir=out_dir)


# ---------------------------------------------------------------------------
# Figure 11 — Per-Subject CL Networks on the Brain (Brain Mosaic)
# ---------------------------------------------------------------------------

def fig_subject_brain_mosaic(fc_arr, mi_arr, cl_arr, keys, d: dict) -> None:
    """
    2 × 5 mosaic of per-subject CL networks on the glass brain.

    Each panel is rendered by nilearn.plotting.plot_connectome into a BytesIO
    buffer (display_mode='z', axial cuts at z=28/55/68 capturing tongue/hand/
    foot), then stitched into a plain matplotlib grid — the same approach used
    for Fig 1 to work around nilearn's internal figure management.

    A single shared plasma colourbar for MI (nats) sits at the right margin.
    """
    from nilearn import plotting

    print('Fig 11 — Per-subject brain mosaic...')

    node_coords = np.array([NODE_COORDS_MNI[k] for k in keys])
    node_color  = [SEMANTIC_COLORS[k] for k in keys]

    nonzero = mi_arr[cl_arr > 0]
    mi_vmax = float(np.percentile(nonzero, 95)) if len(nonzero) else 1.0
    n = len(keys)

    def _render_panel(si):
        deg_s = (cl_arr[si] > 0).sum(axis=1).astype(float)
        ns    = (20 + 80 * (deg_s - deg_s.min()) /
                 max(deg_s.max() - deg_s.min(), 1e-3)).tolist()
        adj   = _make_adj(cl_arr[si] * mi_arr[si])
        disp  = plotting.plot_connectome(
            adj, node_coords,
            node_color=node_color,
            node_size=ns,
            edge_cmap='plasma',
            edge_vmin=0.1, edge_vmax=mi_vmax,
            display_mode='z',
            colorbar=False,
        )
        return _nilearn_to_img(disp)

    # Render all 10 panels (serial — each closes its own figure)
    imgs = [_render_panel(si) for si in range(len(SUBJECTS))]

    # ---- Assemble mosaic ----
    # Leave a 0.07-wide strip on the right for the colourbar
    fig = plt.figure(figsize=(14.4, 5.8))
    gs  = fig.add_gridspec(2, 5, left=0.01, right=0.89, top=0.90, bottom=0.13,
                           hspace=0.12, wspace=0.04)

    for si, sub in enumerate(SUBJECTS):
        ax = fig.add_subplot(gs[si // 5, si % 5])
        ax.imshow(imgs[si])
        ax.axis('off')
        mi_mean_s = mi_arr[si][np.triu_indices(n, k=1)].mean()
        ax.set_title(f'{sub}\n(MĪ={mi_mean_s:.2f})', fontsize=7.5, pad=2)

    # Shared colourbar in its own dedicated axes (entirely outside the mosaic)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.60])   # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(
        cmap='plasma',
        norm=mcolors.Normalize(vmin=0.1, vmax=mi_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='MI (nats)')
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('MI (nats)', fontsize=7.5)

    # ROI colour legend
    patches_roi = [mpatches.Patch(color=SEMANTIC_COLORS[r],
                                   label=_SHORT[r].replace('\n', ' '))
                   for r in keys]
    fig.legend(handles=patches_roi, loc='lower center', ncol=6,
               fontsize=6.5, framealpha=0.9,
               bbox_to_anchor=(0.45, 0.00),
               title='ROI colour', title_fontsize=7)

    fig.suptitle('GCSS Motor Network — Per-Subject CL Trees on the Brain (all-sessions)',
                 fontsize=9, y=0.97)
    _save(fig, 'fig11_subject_brain_mosaic')


# ---------------------------------------------------------------------------
# Table 1 — Pairwise connectivity
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure 12 — 11 ROIs on Cortical Flat Map
# ---------------------------------------------------------------------------

def fig_roi_flatmap(keys: list) -> None:
    """
    Flat cortical surface map (fsaverage5) showing all 11 GCSS motor fROIs.

    Workflow
    --------
    1. Build a labeled MNI volume by merging all contrast cluster maps:
       each voxel gets an integer 1–11 corresponding to its semantic ROI.
    2. Project to fsaverage5 pial surface with nearest-neighbour interpolation
       (preserves integer labels; no interpolation artefacts at ROI borders).
    3. Display on the flat surface mesh (flat_left / flat_right from fsaverage5)
       with sulcal depth as a grey-scale background.
    4. Stitch left and right hemispheres side-by-side with a shared ROI legend.
    """
    from nilearn import plotting, datasets as nlds, surface
    import nibabel as nib

    print('Fig 12 — ROI flat map...')

    contrasts  = ['LFoot', 'RFoot', 'LHand', 'RHand', 'tongue', 'combined_motor']
    roi_order  = list(SEMANTIC_COLORS.keys())   # fixed 11-element order

    # ---- Build labeled MNI volume ----
    ref_path = None
    for c in contrasts:
        p = os.path.join(_CMAP_DIR, f'{c}_clusters.nii.gz')
        if os.path.exists(p):
            ref_path = p
            break
    if ref_path is None:
        print('  WARNING: no cluster maps found, skipping Fig 12')
        return

    ref_img  = nib.load(ref_path)
    roi_vol  = np.zeros(ref_img.shape[:3], dtype=np.float32)

    for contrast in contrasts:
        cpath = os.path.join(_CMAP_DIR, f'{contrast}_clusters.nii.gz')
        ipath = os.path.join(_CMAP_DIR, f'{contrast}_cluster_info.json')
        if not (os.path.exists(cpath) and os.path.exists(ipath)):
            continue
        cdata = np.asarray(nib.load(cpath).dataobj, dtype=np.int16)
        with open(ipath) as fh:
            info_dict = {int(k): v for k, v in json.load(fh).items()}
        for cid, info in info_dict.items():
            if not _TIMESERIES_OK:
                continue
            sem = semantic_roi_name(contrast, info)
            if sem in roi_order:
                roi_vol[cdata == cid] = roi_order.index(sem) + 1   # 1-indexed

    roi_img = nib.Nifti1Image(roi_vol, ref_img.affine)

    # ---- Surface setup ----
    fsaverage = nlds.fetch_surf_fsaverage('fsaverage5')

    # Discrete colormap: index 0 = background (light grey), 1–11 = ROI colours
    cmap_colors = [(0.82, 0.82, 0.82, 1.0)] + \
                  [mcolors.to_rgba(SEMANTIC_COLORS[r]) for r in roi_order]
    roi_cmap    = ListedColormap(cmap_colors, name='gcss_rois')

    # ---- Render each hemisphere ----
    # view=(90, 0): elevation=90° looks straight down at the xy-plane of the flat
    # surface (all flat-surface z values ≈ 0), giving a true 2-D flat map.
    # If the surface renderer rejects this, fall back to 4 panels (lat + med).
    panels      = []
    panel_titles = []
    use_flatmap  = True

    for hemi in ('left', 'right'):
        pial_mesh = getattr(fsaverage, f'pial_{hemi}')
        flat_mesh = getattr(fsaverage, f'flat_{hemi}')
        sulc_map  = getattr(fsaverage, f'sulc_{hemi}')

        # Project volume → pial vertices (nearest neighbour preserves integer labels)
        texture  = surface.vol_to_surf(roi_img, pial_mesh, interpolation='nearest')
        tex_plot = texture.astype(float)
        tex_plot[tex_plot == 0] = np.nan   # background → transparent

        if use_flatmap:
            try:
                disp = plotting.plot_surf_roi(
                    flat_mesh, tex_plot,
                    hemi=hemi,
                    view=(90, 0),   # straight down = flat projection of flat surface
                    bg_map=sulc_map,
                    bg_on_data=True,
                    cmap=roi_cmap,
                    vmin=0.5, vmax=len(roi_order) + 0.5,
                    colorbar=False,
                    darkness=0.5,
                    title=f'{"Left" if hemi == "left" else "Right"} hemisphere (flat)',
                )
                panels.append(_nilearn_to_img(disp))
                panel_titles.append(f'{"Left" if hemi=="left" else "Right"} hemisphere')
            except Exception as exc:
                print(f'  flat view failed ({exc}); switching to lateral + medial')
                use_flatmap = False
                panels.clear()
                panel_titles.clear()

        if not use_flatmap:
            for view_name in ('lateral', 'medial'):
                disp = plotting.plot_surf_roi(
                    pial_mesh, tex_plot,
                    hemi=hemi,
                    view=view_name,
                    bg_map=sulc_map,
                    bg_on_data=True,
                    cmap=roi_cmap,
                    vmin=0.5, vmax=len(roi_order) + 0.5,
                    colorbar=False,
                    darkness=0.5,
                    title=f'{"Left" if hemi == "left" else "Right"} ({view_name})',
                )
                panels.append(_nilearn_to_img(disp))
                panel_titles.append(f'{"Left" if hemi=="left" else "Right"} ({view_name})')

    # ---- Stitch into a single figure ----
    n_panels = len(panels)
    if n_panels <= 2:
        nrows, ncols = 1, 2
    else:
        nrows, ncols = 2, 2   # 4-panel fallback: 2 rows × 2 cols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 5.5 * nrows))
    axes_flat = np.array(axes).ravel()

    for ax, img in zip(axes_flat, panels):
        ax.imshow(img)
        ax.axis('off')
    for ax in axes_flat[n_panels:]:
        ax.axis('off')

    # ROI legend
    patches_roi = [mpatches.Patch(color=SEMANTIC_COLORS[r],
                                   label=_SHORT[r].replace('\n', ' '))
                   for r in roi_order]
    fig.legend(handles=patches_roi, loc='lower center', ncol=6,
               fontsize=7, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02),
               title='ROI', title_fontsize=8)

    fig.suptitle('GCSS Motor Network — 11 Motor fROIs on Cortical Flat Map\n'
                 '(Group cluster maps projected to fsaverage5)',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, 'fig12_roi_flatmap')


# ---------------------------------------------------------------------------
def table_pairwise_connectivity(fc_arr, mi_arr, keys, d: dict) -> None:
    """CSV + LaTeX: all 55 ROI pairs with FC mean/SD, MI mean/SD, CL freq."""
    print('Table 1 — Pairwise connectivity table...')
    os.makedirs(TAB_DIR, exist_ok=True)

    ef       = d['ef']
    fc_mean  = d['fc_mean']
    fc_std   = d['fc_std']
    mi_mean  = d['mi_mean']
    mi_std   = d['mi_std']
    n        = d['n']
    rows, cols = np.triu_indices(n, k=1)

    records = []
    for i, j in zip(rows, cols):
        records.append({
            'ROI_A':   keys[i],
            'ROI_B':   keys[j],
            'FC_mean': round(float(fc_mean[i, j]), 3),
            'FC_SD':   round(float(fc_std[i, j]),  3),
            'MI_mean': round(float(mi_mean[i, j]), 3),
            'MI_SD':   round(float(mi_std[i, j]),  3),
            'CL_freq': f'{int(ef[i, j])}/10',
        })

    df = pd.DataFrame(records).sort_values('MI_mean', ascending=False)
    csv_path = os.path.join(TAB_DIR, 'table1_pairwise_connectivity.csv')
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(TAB_DIR, 'table1_pairwise_connectivity.tex')
    with open(tex_path, 'w') as fh:
        fh.write('\\begin{longtable}{llrrrrl}\n')
        fh.write('\\hline\n')
        fh.write('ROI A & ROI B & FC mean & FC SD & MI mean (nats) & MI SD & CL freq. \\\\\n')
        fh.write('\\hline\\endhead\n')
        for _, row in df.iterrows():
            fh.write(f'{row.ROI_A} & {row.ROI_B} & '
                     f'{row.FC_mean:.3f} & {row.FC_SD:.3f} & '
                     f'{row.MI_mean:.3f} & {row.MI_SD:.3f} & '
                     f'{row.CL_freq} \\\\\n')
        fh.write('\\hline\n\\end{longtable}\n')

    print(f'  saved → tables/table1_pairwise_connectivity.{{csv,tex}}')


# ---------------------------------------------------------------------------
# Table 2 — Per-subject summary
# ---------------------------------------------------------------------------

def table_subject_summary(fc_arr, mi_arr, cl_arr, keys, d: dict) -> None:
    """CSV: per-subject MI stats, strongest pair, hub nodes, canonical edge count."""
    print('Table 2 — Per-subject summary...')
    os.makedirs(TAB_DIR, exist_ok=True)

    ef     = d['ef']
    cl_bin = d['cl_bin']
    n      = d['n']
    rows, cols = np.triu_indices(n, k=1)
    canonical_edges = [(keys[i], keys[j]) for i, j in zip(rows, cols) if ef[i, j] >= 7]

    records = []
    for si, sub in enumerate(SUBJECTS):
        mi_s  = mi_arr[si][rows, cols]
        fc_s  = fc_arr[si][rows, cols]
        best  = mi_s.argmax()
        deg_s = cl_bin[si].sum(axis=1)
        hubs  = [keys[i] for i in range(n) if deg_s[i] >= 3]
        n_canon = sum(1 for (a, b) in canonical_edges
                      if cl_bin[si][keys.index(a), keys.index(b)] > 0)
        records.append({
            'Subject':         sub,
            'TRs':             8180 if sub != 'MSC06' else 8179,
            'MI_mean':         round(float(mi_s.mean()), 3),
            'MI_max':          round(float(mi_s.max()),  3),
            'FC_mean':         round(float(fc_s.mean()), 3),
            'FC_max':          round(float(fc_s.max()),  3),
            'Strongest_pair':  f'{keys[rows[best]]} ↔ {keys[cols[best]]}',
            'Hub_nodes':       ', '.join(hubs) if hubs else '—',
            'Canonical_edges': f'{n_canon}/{len(canonical_edges)}',
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(TAB_DIR, 'table2_subject_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'  saved → tables/table2_subject_summary.csv')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate GCSS motor network publication figures.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--figs', nargs='+', type=int,
                   choices=[1, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                   help='Figure numbers to generate (default: all)')
    p.add_argument('--tables-only', action='store_true',
                   help='Generate only the CSV/LaTeX tables')
    p.add_argument('--subject', default='MSC01',
                   help='Representative subject for Fig 1 and single-subject Fig 9/10')
    p.add_argument('--all-subjects', action='store_true',
                   help='Also generate per-subject variants of Fig 9 and Fig 10 '
                        '(MSC01–MSC10)')
    return p.parse_args()


def main():
    _apply_style()
    args = _parse_args()

    print('Loading group connectivity data...')
    fc_arr, mi_arr, cl_arr, keys = _load_group_data()
    d = _derived(fc_arr, mi_arr, cl_arr, keys)
    print(f'  Loaded {len(SUBJECTS)} subjects × {d["n"]} ROIs\n')

    want = set(args.figs or [1, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    if args.tables_only:
        want = set()

    if  1 in want: fig_pipeline_schematic(subject=args.subject)
    if  4 in want: fig_group_matrices(d, keys)
    if  5 in want: fig_consensus_tree(d, keys)
    if  6 in want: fig_subject_trees(fc_arr, mi_arr, cl_arr, keys, d)
    if  7 in want: fig_hub_degrees(cl_arr, keys, d)
    if  8 in want: fig_group_fc_vs_mi(fc_arr, mi_arr, keys, d)
    if  9 in want:
        fig_cl_tree_on_brain(fc_arr, mi_arr, cl_arr, keys, d,
                              subject=args.subject,
                              all_subjects=args.all_subjects)
    if 10 in want:
        fig_aesthetic_tree(d, keys)   # group version → pub_figures/
        for sub in SUBJECTS:
            sub_dir = os.path.join(SUBJ_FIG_BASE, sub)
            fig_aesthetic_tree(d, keys, cl_arr=cl_arr, mi_arr=mi_arr,
                               subject=sub, out_dir=sub_dir)
    if 11 in want: fig_subject_brain_mosaic(fc_arr, mi_arr, cl_arr, keys, d)
    if 12 in want: fig_roi_flatmap(keys)

    table_pairwise_connectivity(fc_arr, mi_arr, keys, d)
    table_subject_summary(fc_arr, mi_arr, cl_arr, keys, d)

    print('\nDone.')


if __name__ == '__main__':
    main()
