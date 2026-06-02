"""
gcss_viz_group_maps.py — Visualize GCSS group probability maps and cluster maps.

Produces figures for quality-checking the Phase 1+2 outputs of gcss_timeseries.py.
Only Tier 1 contrasts are included: LFoot, RFoot, LHand, RHand, tongue, combined_motor.

  Fig 1  — Probability threshold curves (voxel count vs threshold)
  Fig 2  — Probability maps: somatotopic effectors (LFoot/RFoot/LHand/RHand/tongue)
  Fig 3  — Probability map: combined_motor (premotor network)
  Fig 4  — Cluster maps: one PDF per contrast, retained ROIs overlaid on MNI
  Fig 5  — Somatotopy check: bilateral foot / hand / tongue probability overlay
  Fig 6  — Cluster centroid scatter (MNI z-gradient and x-vs-z anatomy)

Output: code/.../canonical_circuits/motor_cortex/gcss/roi_figures/

Usage
-----
  python gcss_viz_group_maps.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from nilearn import plotting, image as nlimage

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_CANONICAL_DIR = os.path.dirname(_SCRIPT_DIR)
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)
sys.path.insert(0, _MSC_DIR)
sys.path.insert(0, _CANONICAL_DIR)

from msc_paths import BASE_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GCSS_BASE   = os.path.join(_MSC_DIR, 'analysis', 'canonical_circuits',
                             'motor_cortex', 'gcss')
GROUP_MAPS   = os.path.join(_GCSS_BASE, 'group_maps')
CLUSTER_MAPS = os.path.join(_GCSS_BASE, 'cluster_maps')
OUT_DIR      = os.path.join(_SCRIPT_DIR, 'gcss', 'roi_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Contrast lists and display settings
# ---------------------------------------------------------------------------

SOMATOTOPIC   = ['LFoot', 'RFoot', 'LHand', 'RHand', 'tongue']
PREMOTOR      = ['combined_motor']
ALL_CONTRASTS = SOMATOTOPIC + PREMOTOR

CONTRAST_CMAPS = {
    'LFoot':          'Oranges',
    'RFoot':          'Purples',
    'LHand':          'YlOrBr',
    'RHand':          'RdPu',
    'tongue':         'Greens',
    'combined_motor': 'YlOrRd',
}

CONTRAST_COLORS = {
    'LFoot':          '#ff7f0e',
    'RFoot':          '#9467bd',
    'LHand':          '#8c564b',
    'RHand':          '#e377c2',
    'tongue':         '#2ca02c',
    'combined_motor': '#17becf',
}

# Cut coordinates focused on motor cortex
MOTOR_CUTS = dict(
    x=[-40, -4, 40],    # left hand/foot | SMA/foot | right hand/foot
    y=[-25, -10,  0],   # posterior (foot M1) → anterior (premotor)
    z=[ 28,  58, 70],   # tongue M1 | hand M1 | foot M1
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prob_path(contrast):
    return os.path.join(GROUP_MAPS, f'{contrast}_prob.nii.gz')

def _cluster_path(contrast):
    return os.path.join(CLUSTER_MAPS, f'{contrast}_clusters.nii.gz')

def _info_path(contrast):
    return os.path.join(CLUSTER_MAPS, f'{contrast}_cluster_info.json')

def _load_info(contrast):
    with open(_info_path(contrast)) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def _voxel_counts_at_thresholds(contrast, thresholds):
    prob = nib.load(_prob_path(contrast)).get_fdata()
    return [int((prob >= t).sum()) for t in thresholds]

def _mni_template():
    from nilearn.datasets import load_mni152_template
    return load_mni152_template(resolution=2)


# ---------------------------------------------------------------------------
# Fig 1 — Probability threshold curves
# ---------------------------------------------------------------------------

def fig_threshold_curves():
    print('Fig 1 — Threshold curves...')
    thresholds = np.arange(0.1, 1.01, 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, group, title in [
        (axes[0], SOMATOTOPIC, 'Somatotopic effectors'),
        (axes[1], PREMOTOR,    'Premotor (combined_motor)'),
    ]:
        for contrast in group:
            if not os.path.exists(_prob_path(contrast)):
                continue
            counts = _voxel_counts_at_thresholds(contrast, thresholds)
            ax.plot(thresholds, counts, label=contrast,
                    color=CONTRAST_COLORS[contrast], lw=2)

        ax.axvline(0.6, color='k', ls='--', lw=1, label='60% threshold (used)')
        ax.axvline(0.5, color='gray', ls=':', lw=1, label='50% threshold')
        ax.axhline(15,  color='gray', ls=':', lw=0.8, label='min 15 vox')
        ax.set_xlabel('Group overlap threshold')
        ax.set_ylabel('Voxels above threshold')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0.1, 1.0)
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)

    fig.suptitle('GCSS Group Probability Maps — Voxel Count vs. Threshold\n'
                 'Dashed = 60% group-overlap cutoff used for clustering',
                 fontsize=10)
    fig.tight_layout()
    fpath = os.path.join(OUT_DIR, 'fig1_threshold_curves.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Shared probability map row renderer
# ---------------------------------------------------------------------------

def _plot_prob_row(contrast, axes_row, template, threshold=0.5):
    """Three-panel axial/coronal/sagittal probability map for one contrast."""
    prob_path = _prob_path(contrast)
    if not os.path.exists(prob_path):
        for ax in axes_row:
            ax.set_visible(False)
        return

    prob_img = nib.load(prob_path)
    n_clusters = 0
    if os.path.exists(_info_path(contrast)):
        n_clusters = len(_load_info(contrast))

    configs = [
        ('z', MOTOR_CUTS['z']),
        ('y', MOTOR_CUTS['y']),
        ('x', MOTOR_CUTS['x']),
    ]
    for ax, (direction, cuts) in zip(axes_row, configs):
        plotting.plot_stat_map(
            prob_img, bg_img=template,
            display_mode=direction, cut_coords=cuts,
            threshold=threshold, vmax=1.0,
            colorbar=False, axes=ax,
            annotate=True, draw_cross=False,
            cmap=CONTRAST_CMAPS.get(contrast, 'hot_r'),
        )

    axes_row[0].set_title(
        f'{contrast}  ({n_clusters} cluster{"s" if n_clusters != 1 else ""})',
        fontsize=9, loc='left', pad=2,
        color=CONTRAST_COLORS.get(contrast, 'black'),
    )


# ---------------------------------------------------------------------------
# Fig 2 — Somatotopic effector probability maps
# ---------------------------------------------------------------------------

def fig_somatotopic_prob_maps():
    print('Fig 2 — Somatotopic probability maps...')
    available = [c for c in SOMATOTOPIC if os.path.exists(_prob_path(c))]
    n = len(available)
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.8 * n))
    if n == 1:
        axes = [axes]

    template = _mni_template()
    for row_axes, contrast in zip(axes, available):
        _plot_prob_row(contrast, row_axes, template, threshold=0.5)

    for ax, label in zip(axes[0],
                         [f'Axial  z = {MOTOR_CUTS["z"]}',
                          f'Coronal  y = {MOTOR_CUTS["y"]}',
                          f'Sagittal  x = {MOTOR_CUTS["x"]}']):
        ax.set_title(label, fontsize=8, color='gray')

    fig.suptitle(
        'GCSS group probability maps — somatotopic effectors\n'
        '(group probability ≥ 50% shown; motor cortex mask applied to LHand/RHand)',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fpath = os.path.join(OUT_DIR, 'fig2_somatotopic_prob_maps.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Fig 3 — Combined motor probability map
# ---------------------------------------------------------------------------

def fig_premotor_prob_map():
    print('Fig 3 — Premotor (combined_motor) probability map...')
    available = [c for c in PREMOTOR if os.path.exists(_prob_path(c))]
    if not available:
        print('  [combined_motor] no probability map found')
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    template = _mni_template()
    _plot_prob_row('combined_motor', axes, template, threshold=0.5)

    for ax, label in zip(axes,
                         [f'Axial  z = {MOTOR_CUTS["z"]}',
                          f'Coronal  y = {MOTOR_CUTS["y"]}',
                          f'Sagittal  x = {MOTOR_CUTS["x"]}']):
        ax.set_title(label, fontsize=8, color='gray')

    if os.path.exists(_info_path('combined_motor')):
        info = _load_info('combined_motor')
        legend_lines = '\n'.join(
            f'  c{cid}: {v["label"]}  MNI {v["centroid_mni"]}  ({v["n_voxels"]} vox)'
            for cid, v in sorted(info.items())
        )
        fig.text(0.01, -0.02, legend_lines, fontsize=6.5,
                 va='top', ha='left', family='monospace')

    fig.suptitle(
        'GCSS group probability map — combined motor > rest  (z > 2.3, ≥ 60% overlap)\n'
        'Expected clusters: SMA/pre-SMA, PMd (6d), PMv (6v), bilateral S2 (PFop)',
        fontsize=9,
    )
    fig.tight_layout()
    fpath = os.path.join(OUT_DIR, 'fig3_premotor_prob_map.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Fig 4 — Cluster maps (one PDF per contrast)
# ---------------------------------------------------------------------------

def fig_cluster_maps():
    print('Fig 4 — Cluster maps...')
    template = _mni_template()

    for contrast in ALL_CONTRASTS:
        cluster_path = _cluster_path(contrast)
        if not os.path.exists(cluster_path):
            print(f'  [{contrast}] no cluster map found')
            continue

        cluster_img = nib.load(cluster_path)
        data        = np.asarray(cluster_img.dataobj)
        n_clusters  = int(data.max())
        info        = _load_info(contrast) if os.path.exists(_info_path(contrast)) else {}

        fig, axes = plt.subplots(1, 3, figsize=(13, 3))

        configs = [
            ('z', MOTOR_CUTS['z']),
            ('y', MOTOR_CUTS['y']),
            ('x', MOTOR_CUTS['x']),
        ]
        for ax, (direction, cuts) in zip(axes, configs):
            if n_clusters > 0:
                plotting.plot_roi(
                    cluster_img, bg_img=template,
                    display_mode=direction, cut_coords=cuts,
                    axes=ax, annotate=True, draw_cross=False,
                    colorbar=False, alpha=0.75,
                )
            else:
                plotting.plot_anat(
                    template, display_mode=direction, cut_coords=cuts,
                    axes=ax, annotate=True, draw_cross=False,
                )

        if info:
            legend_txt = '\n'.join(
                f'  c{cid}: {v["label"]:28s}  MNI {v["centroid_mni"]}  ({v["n_voxels"]} vox)'
                for cid, v in sorted(info.items())
            )
        else:
            legend_txt = '  (no clusters retained)'

        axes[0].set_title(
            f'{contrast} — {n_clusters} cluster(s)',
            fontsize=9, loc='left',
            color=CONTRAST_COLORS.get(contrast, 'black'),
        )
        fig.text(0.01, -0.02, legend_txt, fontsize=6.5,
                 va='top', ha='left', family='monospace')

        fig.suptitle(f'GCSS cluster map: {contrast}', fontsize=9)
        fig.tight_layout()
        fpath = os.path.join(OUT_DIR, f'fig4_clusters_{contrast}.pdf')
        fig.savefig(fpath, bbox_inches='tight')
        plt.close(fig)

    print(f'  cluster map PDFs → {OUT_DIR}/')


# ---------------------------------------------------------------------------
# Fig 5 — Somatotopy check: bilateral foot / hand / tongue overlay
# ---------------------------------------------------------------------------

def fig_somatotopy_check():
    """
    Overlay bilaterally-averaged foot (LFoot+RFoot), hand (LHand+RHand), and
    tongue probability maps on coronal slices to verify the superior→inferior
    somatotopic gradient.
    """
    print('Fig 5 — Somatotopy check...')
    template = _mni_template()

    def _bilateral_avg(c1, c2):
        """Average two probability map images voxel-wise."""
        p1 = _prob_path(c1)
        p2 = _prob_path(c2)
        if not os.path.exists(p1) or not os.path.exists(p2):
            return None
        img1 = nib.load(p1)
        img2 = nib.load(p2)
        avg  = (np.asarray(img1.dataobj, dtype=np.float32) +
                np.asarray(img2.dataobj, dtype=np.float32)) / 2.0
        return nib.Nifti1Image(avg, img1.affine, img1.header)

    foot_img   = _bilateral_avg('LFoot', 'RFoot')
    hand_img   = _bilateral_avg('LHand', 'RHand')
    tongue_img = nib.load(_prob_path('tongue')) if os.path.exists(_prob_path('tongue')) else None

    coronal_cuts = [-30, -22, -15, -8, -2, 4]
    fig, axes = plt.subplots(1, len(coronal_cuts), figsize=(3.5 * len(coronal_cuts), 4.5))

    overlay_params = [
        (foot_img,   0.45, 'Reds',   '#d62728', 'foot (LFoot+RFoot avg, thr≥0.45)'),
        (hand_img,   0.45, 'Blues',  '#1f77b4', 'hand (LHand+RHand avg, thr≥0.45)'),
        (tongue_img, 0.55, 'Greens', '#2ca02c', 'tongue (thr≥0.55)'),
    ]

    for ax, y_cut in zip(axes, coronal_cuts):
        disp = plotting.plot_anat(
            template, display_mode='y', cut_coords=[y_cut],
            axes=ax, annotate=False, draw_cross=False,
        )
        for img, thr, cmap, _, _ in overlay_params:
            if img is None:
                continue
            disp.add_overlay(img, threshold=thr, cmap=cmap, vmax=1.0, alpha=0.65)
        ax.set_title(f'y = {y_cut}', fontsize=8)

    legend_patches = [
        mpatches.Patch(color=col, label=lbl)
        for _, _, _, col, lbl in overlay_params
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        'Somatotopy check — bilateral average probability maps\n'
        'foot (red) superior/medial  ·  hand (blue) intermediate  ·  tongue (green) inferior/lateral',
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fpath = os.path.join(OUT_DIR, 'fig5_somatotopy_check.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Fig 7 — All 11 ROIs on a single labeled brain figure
# ---------------------------------------------------------------------------

# ROI definitions: (display_label, source_contrast, cluster_id, hex_color)
# Colors: paired dark/light for foot (red), hand (blue), tongue (green);
#         single colors for SMA, PMd, PMv, L/R S1S2.
ALL_ROI_DEFS = [
    ('R_M1_foot',   'LFoot',          1, '#e31a1c'),  # dark red
    ('L_M1_foot',   'RFoot',          1, '#fb9a99'),  # light red
    ('R_M1_hand',   'LHand',          1, '#1f78b4'),  # dark blue
    ('L_M1_hand',   'RHand',          1, '#a6cee3'),  # light blue
    ('L_M1_tongue', 'tongue',         1, '#33a02c'),  # dark green
    ('R_M1_tongue', 'tongue',         6, '#b2df8a'),  # light green
    ('SMA',         'combined_motor', 5, '#ff7f00'),  # orange
    ('L_PMd',       'combined_motor', 4, '#e377c2'),  # pink
    ('R_PMv',       'combined_motor', 7, '#9467bd'),  # purple
    ('L_S1S2',      'combined_motor', 1, '#17becf'),  # teal
    ('R_S1S2',      'combined_motor', 8, '#8c564b'),  # brown
]

# Anatomical subtitles for legend entries
_ROI_SUBTITLES = {
    'R_M1_foot':   'right M1 paracentral',
    'L_M1_foot':   'left M1 paracentral',
    'R_M1_hand':   'right M1 hand knob',
    'L_M1_hand':   'left M1/S1 hand',
    'L_M1_tongue': 'left M1 lateral face',
    'R_M1_tongue': 'right M1 lateral face',
    'SMA':         'SMA / pre-SMA (medial)',
    'L_PMd':       'left dorsal premotor (6d)',
    'R_PMv':       'right ventral premotor (6v)',
    'L_S1S2':      'left parietal operculum S2',
    'R_S1S2':      'right parietal operculum S2',
}


def fig_all_rois_labeled():
    """
    Single figure showing all 11 GCSS ROIs on MNI brain slices.
    Rows: glass brain overview · axial · coronal · sagittal · legend.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    print('Fig 7 — All 11 ROIs labeled...')
    template = _mni_template()

    # ------------------------------------------------------------------
    # Build combined labeled volume in MNI template space (int labels 1-11)
    # ------------------------------------------------------------------
    combined_vol = np.zeros(template.shape[:3], dtype=np.float32)
    centroids    = {}

    for roi_idx, (label, contrast, cid, _) in enumerate(ALL_ROI_DEFS, start=1):
        cpath = _cluster_path(contrast)
        if not os.path.exists(cpath):
            print(f'    WARNING: {contrast} cluster map missing')
            continue
        cluster_img = nib.load(cpath)
        if cluster_img.shape[:3] != template.shape[:3]:
            cluster_img = nlimage.resample_to_img(
                cluster_img, template, interpolation='nearest')
        cluster_data = np.asarray(cluster_img.dataobj, dtype=np.int16)
        mask = cluster_data == cid
        combined_vol[mask] = float(roi_idx)
        if mask.any():
            vox = np.array(np.where(mask), dtype=float).T.mean(axis=0)
            centroids[label] = nib.affines.apply_affine(template.affine, vox)

    combined_img = nib.Nifti1Image(combined_vol, template.affine)

    n = len(ALL_ROI_DEFS)
    roi_colors = [c for _, _, _, c in ALL_ROI_DEFS]
    cmap       = ListedColormap(roi_colors)
    # BoundaryNorm: n+1 boundaries so each integer 1..n maps to its own color
    bounds = np.arange(0.5, n + 1.5, 1.0)
    norm   = BoundaryNorm(bounds, ncolors=n)

    common_kw = dict(
        bg_img=template,
        cmap=cmap,
        vmin=0.5,
        vmax=n + 0.5,
        threshold=0.5,
        colorbar=False,
        annotate=True,
        draw_cross=False,
        alpha=0.88,
    )

    # ------------------------------------------------------------------
    # Layout: 5 rows
    #   0 — glass brain (3 panels)
    #   1 — axial slices
    #   2 — coronal slices
    #   3 — sagittal slices
    #   4 — legend
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 16))
    gs  = fig.add_gridspec(
        5, 1,
        height_ratios=[0.9, 1.0, 1.0, 1.0, 0.35],
        hspace=0.40,
    )
    ax_glass   = fig.add_subplot(gs[0])
    ax_axial   = fig.add_subplot(gs[1])
    ax_coronal = fig.add_subplot(gs[2])
    ax_sagit   = fig.add_subplot(gs[3])
    ax_legend  = fig.add_subplot(gs[4])

    # Glass brain (overview)
    plotting.plot_glass_brain(
        combined_img,
        display_mode='lyrz',
        cmap=cmap,
        vmin=0.5,
        vmax=n + 0.5,
        threshold=0.5,
        colorbar=False,
        annotate=False,
        draw_cross=False,
        alpha=0.75,
        axes=ax_glass,
    )
    ax_glass.set_title(
        'Glass brain — all 11 ROIs (sagittal L · axial · coronal · sagittal R)',
        fontsize=8, pad=3,
    )

    # Axial slices: tongue M1 (28) · PMv/S1S2 (38) · hand M1 / PMd (55) · SMA (62) · foot M1 (70)
    plotting.plot_stat_map(
        combined_img, display_mode='z',
        cut_coords=[28, 38, 55, 62, 70],
        axes=ax_axial, **common_kw,
    )
    ax_axial.set_title(
        'Axial  z = 28 · 38 · 55 · 62 · 70 mm'
        '   [tongue M1 · PMv/S1S2 · hand M1/PMd · SMA · foot M1]',
        fontsize=8, pad=3,
    )

    # Coronal slices: foot (−28) · hand (−18) · hand/SMA (−8) · premotor (+2)
    plotting.plot_stat_map(
        combined_img, display_mode='y',
        cut_coords=[-28, -18, -8, 2],
        axes=ax_coronal, **common_kw,
    )
    ax_coronal.set_title(
        'Coronal  y = −28 · −18 · −8 · +2 mm'
        '   [foot M1 · hand M1 · hand/SMA · premotor]',
        fontsize=8, pad=3,
    )

    # Sagittal: L tongue/S1S2 (−58) · L PMd (−42) · medial SMA/foot (−5) · R hand (42) · R tongue/PMv (58)
    plotting.plot_stat_map(
        combined_img, display_mode='x',
        cut_coords=[-58, -42, -5, 42, 58],
        axes=ax_sagit, **common_kw,
    )
    ax_sagit.set_title(
        'Sagittal  x = −58 · −42 · −5 · +42 · +58 mm'
        '   [L tongue/S1S2 · L PMd · medial SMA/foot · R hand · R tongue/PMv]',
        fontsize=8, pad=3,
    )

    # ------------------------------------------------------------------
    # Legend with color patch + ROI label + anatomical subtitle
    # ------------------------------------------------------------------
    ax_legend.axis('off')
    patches = [
        mpatches.Patch(
            color=color,
            label=f'{label}  ({_ROI_SUBTITLES.get(label, "")})',
        )
        for label, _, _, color in ALL_ROI_DEFS
    ]
    ax_legend.legend(
        handles=patches,
        loc='center',
        ncol=4,
        fontsize=8.5,
        framealpha=0.9,
        handlelength=1.8,
        columnspacing=1.5,
        labelspacing=0.5,
    )

    fig.suptitle(
        'GCSS Motor Network — All 11 Group-Constrained ROIs\n'
        'Paired colors: dark/light = right/left hemisphere  ·  '
        'Foot (red)  ·  Hand (blue)  ·  Tongue (green)  ·  '
        'SMA (orange)  ·  PMd (pink)  ·  PMv (purple)  ·  S1S2 (teal/brown)',
        fontsize=10,
        y=1.005,
    )

    fpath = os.path.join(OUT_DIR, 'fig7_all_rois_labeled.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Fig 6 — Cluster centroid scatter
# ---------------------------------------------------------------------------

def fig_centroid_scatter():
    """
    Left panel: MNI z per contrast (somatotopic gradient check).
    Right panel: MNI x vs z for all somatotopic clusters (2-D anatomy view).
    """
    print('Fig 6 — Centroid scatter...')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: z-coordinate by contrast ---
    ax = axes[0]
    contrast_order = ALL_CONTRASTS
    for i, contrast in enumerate(contrast_order):
        if not os.path.exists(_info_path(contrast)):
            continue
        info   = _load_info(contrast)
        zvals  = [v['centroid_mni'][2] for v in info.values()]
        labels = [v['label']           for v in info.values()]
        ax.scatter([i] * len(zvals), zvals,
                   color=CONTRAST_COLORS.get(contrast, 'gray'),
                   s=70, zorder=3)
        for z, lab in zip(zvals, labels):
            ax.annotate(lab, (i, z), xytext=(5, 0),
                        textcoords='offset points', fontsize=5.5, va='center')

    ax.set_xticks(range(len(contrast_order)))
    ax.set_xticklabels(contrast_order, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Centroid MNI z (mm)  [superior ↑]', fontsize=8)
    ax.set_title(
        'Cluster centroid z-coordinate\n'
        'Expected: foot z≈69  >  hand z≈58  >  tongue z≈29',
        fontsize=8,
    )
    # Shade expected z-ranges
    ax.axhspan(62, 76, alpha=0.07, color='#d62728', label='Foot M1 zone (z 62–76)')
    ax.axhspan(50, 62, alpha=0.07, color='#1f77b4', label='Hand M1 zone (z 50–62)')
    ax.axhspan(20, 38, alpha=0.07, color='#2ca02c', label='Tongue M1 zone (z 20–38)')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.legend(fontsize=7, loc='lower right')

    # --- Right: MNI x vs z (anatomical 2-D view) ---
    ax2 = axes[1]
    for contrast in SOMATOTOPIC:
        if not os.path.exists(_info_path(contrast)):
            continue
        info = _load_info(contrast)
        for cid, v in info.items():
            x_mni = v['centroid_mni'][0]
            z_mni = v['centroid_mni'][2]
            ax2.scatter(x_mni, z_mni,
                        color=CONTRAST_COLORS.get(contrast, 'gray'),
                        s=90, zorder=3)
            ax2.annotate(
                f'{contrast}\n{v["label"]}',
                (x_mni, z_mni), xytext=(4, 2),
                textcoords='offset points', fontsize=5.5,
            )

    ax2.set_xlabel('Centroid MNI x (mm)  [L ← | → R]', fontsize=8)
    ax2.set_ylabel('Centroid MNI z (mm)  [superior ↑]', fontsize=8)
    ax2.set_title(
        'Cluster centroids: x vs z  (somatotopic contrasts)\n'
        'Foot near x ≈ ±5 z ≈ 69  ·  Hand near x ≈ ±40 z ≈ 58  ·  Tongue x ≈ ±57 z ≈ 29',
        fontsize=8,
    )
    ax2.axvline(0, color='gray', lw=0.5, ls='--')

    # Shade expected homunculus zones
    ax2.axhspan(62, 76, alpha=0.07, color='#d62728')
    ax2.axhspan(50, 62, alpha=0.07, color='#1f77b4')
    ax2.axhspan(20, 38, alpha=0.07, color='#2ca02c')

    legend_patches = [
        mpatches.Patch(color=CONTRAST_COLORS[c], label=c)
        for c in SOMATOTOPIC if os.path.exists(_info_path(c))
    ]
    ax2.legend(handles=legend_patches, fontsize=7, loc='lower left')

    fig.suptitle('GCSS Cluster Centroid Positions — Somatotopic Gradient Check', fontsize=10)
    fig.tight_layout()
    fpath = os.path.join(OUT_DIR, 'fig6_centroid_scatter.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Subject-specific fROI figure
# ---------------------------------------------------------------------------

def fig_subject_rois(subject: str) -> None:
    """
    Multi-slice figure showing all 11 subject-specific GCSS fROI masks.
    Reads saved NIfTI masks from froi_masks/{subject}/ and builds a
    combined labeled volume in the same style as fig_all_rois_labeled().
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    print(f'Subject ROI figure — {subject}...')
    template = _mni_template()
    froi_dir = os.path.join(_GCSS_BASE, 'froi_masks', subject)

    combined_vol = np.zeros(template.shape[:3], dtype=np.float32)
    n_found = 0

    for roi_idx, (label, *_) in enumerate(ALL_ROI_DEFS, start=1):
        fname = f'{subject}_gcss_{label}_mask.nii.gz'
        fpath = os.path.join(froi_dir, fname)
        if not os.path.exists(fpath):
            print(f'    WARNING: missing {fname}')
            continue
        mask_img = nib.load(fpath)
        if mask_img.shape[:3] != template.shape[:3]:
            mask_img = nlimage.resample_to_img(
                mask_img, template, interpolation='nearest')
        mask_data = np.asarray(mask_img.dataobj, dtype=bool)
        combined_vol[mask_data] = float(roi_idx)
        n_found += 1

    print(f'    {n_found}/{len(ALL_ROI_DEFS)} fROI masks loaded')
    combined_img = nib.Nifti1Image(combined_vol, template.affine)

    n = len(ALL_ROI_DEFS)
    roi_colors = [c for _, _, _, c in ALL_ROI_DEFS]
    cmap   = ListedColormap(roi_colors)
    bounds = np.arange(0.5, n + 1.5, 1.0)
    norm   = BoundaryNorm(bounds, ncolors=n)  # noqa: F841

    common_kw = dict(
        bg_img=template,
        cmap=cmap,
        vmin=0.5,
        vmax=n + 0.5,
        threshold=0.5,
        colorbar=False,
        annotate=True,
        draw_cross=False,
        alpha=0.88,
    )

    fig = plt.figure(figsize=(20, 16))
    gs  = fig.add_gridspec(
        5, 1,
        height_ratios=[0.9, 1.0, 1.0, 1.0, 0.35],
        hspace=0.40,
    )
    ax_glass   = fig.add_subplot(gs[0])
    ax_axial   = fig.add_subplot(gs[1])
    ax_coronal = fig.add_subplot(gs[2])
    ax_sagit   = fig.add_subplot(gs[3])
    ax_legend  = fig.add_subplot(gs[4])

    plotting.plot_glass_brain(
        combined_img, display_mode='lyrz',
        cmap=cmap, vmin=0.5, vmax=n + 0.5, threshold=0.5,
        colorbar=False, annotate=False, draw_cross=False, alpha=0.75,
        axes=ax_glass,
    )
    ax_glass.set_title(
        f'Glass brain — {subject} subject-specific fROIs'
        '  (sagittal L · axial · coronal · sagittal R)',
        fontsize=8, pad=3,
    )

    plotting.plot_stat_map(
        combined_img, display_mode='z',
        cut_coords=[28, 38, 55, 62, 70],
        axes=ax_axial, **common_kw,
    )
    ax_axial.set_title(
        'Axial  z = 28 · 38 · 55 · 62 · 70 mm'
        '   [tongue M1 · PMv/S1S2 · hand M1/PMd · SMA · foot M1]',
        fontsize=8, pad=3,
    )

    plotting.plot_stat_map(
        combined_img, display_mode='y',
        cut_coords=[-28, -18, -8, 2],
        axes=ax_coronal, **common_kw,
    )
    ax_coronal.set_title(
        'Coronal  y = −28 · −18 · −8 · +2 mm'
        '   [foot M1 · hand M1 · hand/SMA · premotor]',
        fontsize=8, pad=3,
    )

    plotting.plot_stat_map(
        combined_img, display_mode='x',
        cut_coords=[-58, -42, -5, 42, 58],
        axes=ax_sagit, **common_kw,
    )
    ax_sagit.set_title(
        'Sagittal  x = −58 · −42 · −5 · +42 · +58 mm'
        '   [L tongue/S1S2 · L PMd · medial SMA/foot · R hand · R tongue/PMv]',
        fontsize=8, pad=3,
    )

    ax_legend.axis('off')
    patches = [
        mpatches.Patch(
            color=color,
            label=f'{label}  ({_ROI_SUBTITLES.get(label, "")})',
        )
        for label, _, _, color in ALL_ROI_DEFS
    ]
    ax_legend.legend(
        handles=patches, loc='center', ncol=4, fontsize=8.5,
        framealpha=0.9, handlelength=1.8, columnspacing=1.5, labelspacing=0.5,
    )

    fig.suptitle(
        f'GCSS Motor Network — {subject} Subject-Specific fROIs\n'
        'Group-constrained subject-specific parcels  ·  '
        'Foot (red)  ·  Hand (blue)  ·  Tongue (green)  ·  '
        'SMA (orange)  ·  PMd (pink)  ·  PMv (purple)  ·  S1S2 (teal/brown)',
        fontsize=10,
        y=1.005,
    )

    out_subj = os.path.join(OUT_DIR, subject)
    os.makedirs(out_subj, exist_ok=True)
    fpath = os.path.join(out_subj, f'{subject}_all_rois.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description='Visualise GCSS group and subject-level maps.')
    p.add_argument('--subject', default=None,
                   help='If given, generate subject-specific fROI figure for this subject '
                        '(e.g. MSC01) in addition to group figures.')
    p.add_argument('--subject-only', action='store_true',
                   help='Skip group figures and only generate subject fROI figure '
                        '(requires --subject).')
    args = p.parse_args()

    print(f'Output directory: {OUT_DIR}\n')

    if not args.subject_only:
        fig_threshold_curves()
        fig_somatotopic_prob_maps()
        fig_premotor_prob_map()
        fig_cluster_maps()
        fig_somatotopy_check()
        fig_centroid_scatter()
        fig_all_rois_labeled()

    if args.subject:
        fig_subject_rois(args.subject)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
