"""
Brain-space visualization of localizer-defined motor ROIs, FC, and CL trees.

Viz 1  — Composite ROI mask overlay on MNI template slices (sagittal + coronal)
Viz 2  — Glass-brain connectome: FC top-N edges and CL tree, side by side
Viz 3  — CL tree edges on ortho anatomical slices
Viz 4  — Surface projection of composite ROI mask onto fsaverage5
Viz 5  — Somatotopic separation score: MNI-z centroids per effector across subjects
Viz 6  — Flat map: ROI overlay on flattened cortical surface (L and R hemispheres)

Usage
-----
    python mc_localizer_brain_viz.py [--subjects MSC01 ...]
                                      [--viz all|roi|connectome|ortho|surface|flatmap|soma]
                                      [--fc-top-n 74]
                                      [--no-surface]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # motor_cortex/
_CANONICAL_DIR = os.path.dirname(_SCRIPT_DIR)                  # canonical_circuits/
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)               # midnight_scan_club/
sys.path.insert(0, _MSC_DIR)        # msc_paths (via canonical_utils chain)
sys.path.insert(0, _CANONICAL_DIR)  # canonical_utils

from nilearn import datasets, image, plotting, surface

from canonical_utils import (
    cc_analysis_dir, cc_figures_dir, mask_output_dir,
    compute_roi_centroids,
)

# ---------------------------------------------------------------------------
# Effector colors (shared with mc_localizer_fc_cl.py)
# ---------------------------------------------------------------------------

_EFFECTOR_COLORS = {
    'foot':   '#d62728',
    'hand':   '#1f77b4',
    'tongue': '#2ca02c',
    'whole':  '#aaaaaa',
}


def _roi_color(label: str) -> str:
    return _EFFECTOR_COLORS.get(label.split('__')[0], '#dddddd')


def _node_color_list(roi_keys: list) -> list:
    return [_roi_color(k) for k in roi_keys]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_matrices(subject: str, analysis_dir: str):
    """Return (fc, cl_adj, roi_keys) for a subject."""
    fc     = np.load(os.path.join(analysis_dir, f'{subject}_localizer_fc.npy'))
    cl_adj = np.load(os.path.join(analysis_dir, f'{subject}_localizer_cl_adj.npy'))
    with open(os.path.join(analysis_dir, f'{subject}_localizer_roi_keys.json')) as f:
        roi_keys = json.load(f)
    return fc, cl_adj, roi_keys


def _threshold_top_n_edges(sym_mat: np.ndarray, n: int) -> np.ndarray:
    """Return a copy of sym_mat with all but the top-n edges (by |value|) zeroed."""
    rows, cols = np.triu_indices(sym_mat.shape[0], k=1)
    vals = np.abs(sym_mat[rows, cols])
    if n >= len(vals):
        result = sym_mat.copy()
        np.fill_diagonal(result, 0)
        return result
    threshold = np.sort(vals)[-n]
    result = np.where(np.abs(sym_mat) >= threshold, sym_mat, 0.0)
    np.fill_diagonal(result, 0)
    return result


def _centroids_array(centroids: dict, roi_keys: list) -> np.ndarray:
    """Return (n_ROIs, 3) MNI coordinate array in roi_keys order."""
    return np.array([centroids[k] for k in roi_keys])


# ---------------------------------------------------------------------------
# Viz 1 — ROI mask overlay on MNI template
# ---------------------------------------------------------------------------

def _build_composite_mask(subject: str, roi_keys: list) -> nib.Nifti1Image:
    """Integer-labeled NIfTI: foot=1, hand=2, tongue=3.  Whole parcels omitted."""
    m_dir   = mask_output_dir(subject)
    ref_img = None
    for key in roi_keys:
        prefix, parcel = key.split('__', 1)
        if prefix in ('foot', 'hand', 'tongue'):
            fpath = os.path.join(
                m_dir, f'{subject}_effector-{prefix}_parcel-{parcel}_mask.nii.gz'
            )
            if os.path.exists(fpath):
                ref_img = nib.load(fpath)
                break
    if ref_img is None:
        raise FileNotFoundError(f'No localizer mask NIfTIs found for {subject}')

    label_map = {'foot': 1, 'hand': 2, 'tongue': 3}
    composite = np.zeros(ref_img.shape[:3], dtype=np.int16)
    for key in roi_keys:
        prefix, parcel = key.split('__', 1)
        if prefix not in label_map:
            continue
        fpath = os.path.join(
            m_dir, f'{subject}_effector-{prefix}_parcel-{parcel}_mask.nii.gz'
        )
        if os.path.exists(fpath):
            data = np.asarray(nib.load(fpath).dataobj, dtype=bool)
            composite[data] = label_map[prefix]
    return image.new_img_like(ref_img, composite)


def fig_roi_overlay(
    subject: str,
    roi_keys: list,
    figures_dir: str,
) -> None:
    composite   = _build_composite_mask(subject, roi_keys)
    mni_bg      = datasets.load_mni152_template(resolution=2)
    cmap        = mcolors.ListedColormap(['#d62728', '#1f77b4', '#2ca02c'])

    fig = plt.figure(figsize=(18, 10))
    ax_sag = fig.add_axes([0.00, 0.10, 1.00, 0.42])
    ax_cor = fig.add_axes([0.00, 0.55, 1.00, 0.42])

    # Sagittal cuts: show L and R area 4 (x ≈ ±36 mm)
    plotting.plot_roi(
        composite, bg_img=mni_bg,
        cut_coords=[-52, -36, 36, 52],
        display_mode='x',
        cmap=cmap, vmin=1, vmax=3,
        axes=ax_sag, colorbar=False,
    )
    ax_sag.set_title('Sagittal (x = –52, –36, +36, +52 mm)', fontsize=9, pad=2)

    # Coronal cuts: show foot–tongue somatotopy (y ≈ –24 to 0 mm)
    plotting.plot_roi(
        composite, bg_img=mni_bg,
        cut_coords=[-24, -12, 0, 12],
        display_mode='y',
        cmap=cmap, vmin=1, vmax=3,
        axes=ax_cor, colorbar=False,
    )
    ax_cor.set_title('Coronal (y = –24, –12, 0, +12 mm)', fontsize=9, pad=2)

    patches = [
        mpatches.Patch(color='#d62728', label='Foot'),
        mpatches.Patch(color='#1f77b4', label='Hand'),
        mpatches.Patch(color='#2ca02c', label='Tongue'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f'{subject} — Localizer Sub-Parcel ROI Masks (areas 4 & 3b)',
                 fontsize=11, y=0.99)

    fpath = os.path.join(figures_dir, f'{subject}_localizer_roi_overlay.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Viz 1 → {fpath}')


# ---------------------------------------------------------------------------
# Viz 2 — Glass-brain connectome (FC top-N and CL tree)
# ---------------------------------------------------------------------------

def fig_glass_brain(
    subject: str,
    fc: np.ndarray,
    cl_adj: np.ndarray,
    roi_keys: list,
    centroids: dict,
    fc_top_n: int,
    figures_dir: str,
) -> None:
    node_coords = _centroids_array(centroids, roi_keys)
    node_colors = _node_color_list(roi_keys)
    fc_thresh   = _threshold_top_n_edges(fc, fc_top_n)
    cl_sym      = np.maximum(cl_adj, cl_adj.T)   # ensure exact symmetry
    n_cl_edges  = int((cl_sym > 0).sum()) // 2

    fig = plt.figure(figsize=(24, 9))
    ax_fc = fig.add_axes([0.01, 0.05, 0.48, 0.88])
    ax_cl = fig.add_axes([0.51, 0.05, 0.48, 0.88])

    plotting.plot_connectome(
        fc_thresh, node_coords,
        node_color=node_colors, node_size=20,
        display_mode='lyrz',
        edge_cmap='RdBu_r', edge_vmin=-1, edge_vmax=1,
        axes=ax_fc, colorbar=False,
    )
    ax_fc.set_title(f'FC top-{fc_top_n} edges (|r|)', fontsize=9, pad=3)

    plotting.plot_connectome(
        cl_sym, node_coords,
        node_color=node_colors, node_size=20,
        display_mode='lyrz',
        edge_cmap='Purples',
        axes=ax_cl, colorbar=False,
    )
    ax_cl.set_title(f'CL tree ({n_cl_edges} edges, MI weight)', fontsize=9, pad=3)

    patches = [
        mpatches.Patch(color=col, label=eff.capitalize())
        for eff, col in _EFFECTOR_COLORS.items()
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'{subject} — Motor ROI Connectivity: Glass Brain', fontsize=11, y=1.0)

    fpath = os.path.join(figures_dir, f'{subject}_localizer_glass_brain.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Viz 2 → {fpath}')


# ---------------------------------------------------------------------------
# Viz 3 — CL tree edges on ortho anatomical slices
# ---------------------------------------------------------------------------

def fig_ortho_edges(
    subject: str,
    cl_adj: np.ndarray,
    roi_keys: list,
    centroids: dict,
    figures_dir: str,
) -> None:
    node_coords = _centroids_array(centroids, roi_keys)
    node_colors = _node_color_list(roi_keys)
    cl_sym      = np.maximum(cl_adj, cl_adj.T)

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_axes([0.0, 0.05, 1.0, 0.90])

    plotting.plot_connectome(
        cl_sym, node_coords,
        node_color=node_colors, node_size=25,
        display_mode='ortho',
        edge_cmap='Purples',
        axes=ax, colorbar=False,
    )

    patches = [
        mpatches.Patch(color=col, label=eff.capitalize())
        for eff, col in _EFFECTOR_COLORS.items()
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'{subject} — CL Tree Edges on Anatomical Slices', fontsize=11, y=1.0)

    fpath = os.path.join(figures_dir, f'{subject}_localizer_cl_ortho.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Viz 3 → {fpath}')


# ---------------------------------------------------------------------------
# Viz 4 — Surface projection of composite ROI mask onto fsaverage5
# ---------------------------------------------------------------------------

def fig_surface_projection(
    subject: str,
    roi_keys: list,
    figures_dir: str,
) -> None:
    composite = _build_composite_mask(subject, roi_keys)
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    cmap      = mcolors.ListedColormap(['#d62728', '#1f77b4', '#2ca02c'])

    panels = [
        ('left',  'lateral',  fsaverage['pial_left'],  fsaverage['sulc_left']),
        ('left',  'medial',   fsaverage['pial_left'],  fsaverage['sulc_left']),
        ('right', 'lateral',  fsaverage['pial_right'], fsaverage['sulc_right']),
        ('right', 'medial',   fsaverage['pial_right'], fsaverage['sulc_right']),
    ]
    titles = ['Left lateral', 'Left medial', 'Right lateral', 'Right medial']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                             subplot_kw={'projection': '3d'})
    axes_flat = axes.flatten()

    for i, ((hemi, view, pial, sulc), title) in enumerate(zip(panels, titles)):
        texture = surface.vol_to_surf(composite, pial, interpolation='nearest')
        texture[texture == 0] = np.nan

        plotting.plot_surf_roi(
            pial, roi_map=texture,
            hemi=hemi, view=view,
            bg_map=sulc, bg_on_data=True,
            cmap=cmap, vmin=1, vmax=3,
            axes=axes_flat[i],
        )
        axes_flat[i].set_title(title, fontsize=10)

    patches = [
        mpatches.Patch(color='#d62728', label='Foot'),
        mpatches.Patch(color='#1f77b4', label='Hand'),
        mpatches.Patch(color='#2ca02c', label='Tongue'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f'{subject} — Localizer ROI Masks on Inflated Surface (fsaverage5)',
                 fontsize=11)

    fpath = os.path.join(figures_dir, f'{subject}_localizer_surface_projection.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Viz 4 → {fpath}')


# ---------------------------------------------------------------------------
# Viz 5 — Somatotopic separation score (group level)
# ---------------------------------------------------------------------------

def fig_somatotopic_score(
    subjects: list,
    analysis_dir: str,
    figures_dir: str,
) -> None:
    """
    For each subject, plot the MNI z-coordinate (superior–inferior) of the
    foot, hand, and tongue sub-parcel centroids within area 4 (L+R averaged).
    Expected ordering: foot most superior (highest z), tongue most inferior.
    """
    effectors = ['foot', 'hand', 'tongue']
    colors    = [_EFFECTOR_COLORS[e] for e in effectors]
    data      = {e: [] for e in effectors}
    valid_subs = []

    for sub in subjects:
        json_path = os.path.join(analysis_dir, f'{sub}_localizer_roi_keys.json')
        if not os.path.exists(json_path):
            print(f'  Viz 5: no roi_keys for {sub} — skipping')
            continue
        with open(json_path) as f:
            roi_keys = json.load(f)

        centroids = compute_roi_centroids(sub, roi_keys)
        ok = True
        z_vals: dict = {}
        for eff in effectors:
            keys_4 = [k for k in roi_keys if k.startswith(f'{eff}__') and '_4' in k]
            if not keys_4:
                ok = False
                break
            z_vals[eff] = np.mean([centroids[k][2] for k in keys_4])

        if ok:
            valid_subs.append(sub)
            for eff in effectors:
                data[eff].append(z_vals[eff])

    if not valid_subs:
        print('  Viz 5: no subjects with complete data — skipping')
        return

    n    = len(valid_subs)
    x    = np.arange(n)
    w    = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 5))
    for i, (eff, col) in enumerate(zip(effectors, colors)):
        ax.bar(x + (i - 1) * w, data[eff], width=w, color=col,
               label=eff.capitalize(), alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(valid_subs, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('MNI z-coordinate (mm)', fontsize=10)
    ax.set_title('Somatotopic Separation — Area 4 Centroids\n'
                 '(foot superior → hand → tongue inferior)', fontsize=10)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.8)
    fig.tight_layout()

    fpath = os.path.join(figures_dir, 'somatotopic_separation_score.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  Viz 5 → {fpath}')


# ---------------------------------------------------------------------------
# Viz 6 — Flat map of ROI overlay (L and R hemispheres, 2D)
# ---------------------------------------------------------------------------

def fig_flat_map(subject: str, roi_keys: list, figures_dir: str) -> None:
    """
    Flat map using matplotlib's Triangulation to render mesh with ROI coloring.
    """
    try:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    except Exception as e:
        print(f'  Viz 6: fsaverage5 not available ({e}) — skipping')
        return

    composite = _build_composite_mask(subject, roi_keys)

    fig = plt.figure(figsize=(18, 8))

    for hemi_idx, hemi in enumerate(['left', 'right']):
        pial_mesh = fsaverage[f'pial_{hemi}']

        # Load mesh
        mesh_img = nib.load(pial_mesh)
        coords_3d = mesh_img.darrays[0].data.astype(np.float32)  # (n_vertices, 3)
        faces = mesh_img.darrays[1].data.astype(np.int32)        # (n_triangles, 3)

        # Project ROI texture to surface
        texture_roi = surface.vol_to_surf(composite, pial_mesh, interpolation='nearest')
        texture_roi = np.clip(texture_roi, 0, 3).astype(np.int32)

        # 2D projection using PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(coords_3d)

        # Normalize coords to [0, 1]
        coords_min = coords_2d.min(axis=0)
        coords_max = coords_2d.max(axis=0)
        coords_2d_norm = (coords_2d - coords_min) / (coords_max - coords_min + 1e-8)

        ax = fig.add_subplot(1, 2, hemi_idx + 1)

        # Create triangulation
        triang = mtri.Triangulation(coords_2d_norm[:, 0], coords_2d_norm[:, 1], faces)

        # Compute average ROI value per triangle (for face coloring)
        face_values = np.mean(texture_roi[faces], axis=1)

        # Plot triangulation with face colors
        cmap = mcolors.ListedColormap(['white', '#d62728', '#1f77b4', '#2ca02c'])
        norm = mcolors.Normalize(vmin=0, vmax=3)

        # Render filled triangles with ROI coloring
        tripcolor = ax.tripcolor(triang, face_values, cmap=cmap, norm=norm,
                               edgecolors='gray', linewidths=0.1, rasterized=True)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{hemi.capitalize()} Hemisphere', fontsize=11, pad=10)

    # Legend
    patches = [
        mpatches.Patch(color='white', label='None', edgecolor='gray'),
        mpatches.Patch(color='#d62728', label='Foot'),
        mpatches.Patch(color='#1f77b4', label='Hand'),
        mpatches.Patch(color='#2ca02c', label='Tongue'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{subject} — Flat Map: Localizer ROI Overlay (PCA-projected mesh)',
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    fpath = os.path.join(figures_dir, f'{subject}_localizer_flat_map.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f'  Viz 6 → {fpath}')


# ---------------------------------------------------------------------------
# Per-subject runner
# ---------------------------------------------------------------------------

def run_subject(
    subject: str,
    viz_set: set,
    fc_top_n: int,
    no_surface: bool,
    analysis_dir: str,
    figures_dir: str,
) -> None:
    print(f'\n{"="*55}')
    print(f'Subject: {subject}')

    fc, cl_adj, roi_keys = _load_matrices(subject, analysis_dir)
    centroids = compute_roi_centroids(subject, roi_keys)

    if 'roi' in viz_set:
        fig_roi_overlay(subject, roi_keys, figures_dir)

    if 'connectome' in viz_set:
        fig_glass_brain(subject, fc, cl_adj, roi_keys, centroids,
                        fc_top_n, figures_dir)

    if 'ortho' in viz_set:
        fig_ortho_edges(subject, cl_adj, roi_keys, centroids, figures_dir)

    if 'surface' in viz_set and not no_surface:
        fig_surface_projection(subject, roi_keys, figures_dir)

    if 'flatmap' in viz_set and not no_surface:
        fig_flat_map(subject, roi_keys, figures_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Brain-space visualization of localizer motor ROIs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--subjects', nargs='+',
                   default=[f'MSC{i:02d}' for i in range(1, 11)])
    p.add_argument('--viz', nargs='+',
                   choices=['all', 'roi', 'connectome', 'ortho', 'surface', 'flatmap', 'soma'],
                   default=['all'],
                   help='Which visualizations to produce.')
    p.add_argument('--fc-top-n', type=int, default=74,
                   help='Number of FC edges to show in glass-brain plot (default: 2×37).')
    p.add_argument('--no-surface', action='store_true',
                   help='Skip fsaverage5 surface projection (Viz 4).')
    return p.parse_args()


def main():
    args    = parse_args()
    adir    = cc_analysis_dir('motor_cortex')
    fdir    = cc_figures_dir('motor_cortex')
    viz_set = set(args.viz)
    if 'all' in viz_set:
        viz_set = {'roi', 'connectome', 'ortho', 'surface', 'flatmap', 'soma'}

    print('=== Localizer ROI Brain Visualization ===')
    print(f'Subjects : {args.subjects}')
    print(f'Viz set  : {sorted(viz_set)}')
    print(f'FC top-N : {args.fc_top_n}')

    for subject in args.subjects:
        try:
            run_subject(
                subject      = subject,
                viz_set      = viz_set,
                fc_top_n     = args.fc_top_n,
                no_surface   = args.no_surface,
                analysis_dir = adir,
                figures_dir  = fdir,
            )
        except Exception as exc:
            print(f'  ERROR for {subject}: {exc}')

    if 'soma' in viz_set:
        print('\n--- Somatotopic separation score (all subjects) ---')
        fig_somatotopic_score(args.subjects, adir, fdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
