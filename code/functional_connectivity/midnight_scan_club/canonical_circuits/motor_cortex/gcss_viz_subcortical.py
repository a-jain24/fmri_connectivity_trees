"""
gcss_viz_subcortical.py — Cerebellar and subcortical localizer figures.

Visualizes motor task group probability maps on cerebellar and subcortical
slices with SUIT cerebellar atlas and Morel thalamus motor-nucleus overlays.

  Fig 1  — Cerebellar probability maps: all 6 contrasts (axial + coronal)
  Fig 2  — Cerebellar somatotopy: foot / hand / tongue winner-takes-all
  Fig 3  — SUIT motor ROIs: atlas lobule labels used in connectivity analysis
  Fig 4  — Subcortical activation: thalamus + BG (combined_motor + effectors)

Output: code/.../canonical_circuits/motor_cortex/gcss/roi_figures/group/
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import nibabel as nib
import numpy as np
from nilearn import plotting, image as nlimage

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_CANONICAL_DIR = os.path.dirname(_SCRIPT_DIR)
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)
sys.path.insert(0, _MSC_DIR)

from msc_paths import BASE_DIR, ATLAS_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GCSS_BASE   = os.path.join(_MSC_DIR, 'analysis', 'canonical_circuits',
                             'motor_cortex', 'gcss')
GROUP_MAPS   = os.path.join(_GCSS_BASE, 'group_maps')
OUT_DIR      = os.path.join(_SCRIPT_DIR, 'gcss', 'roi_figures', 'group')
os.makedirs(OUT_DIR, exist_ok=True)

# SUIT cerebellar atlas (1 mm MNI space, labels 1–34)
SUIT_ATLAS = '/mfs/io/groups/dmello/projects/face/encoding_baby/code/atl-Anatom_space-MNI_dseg.nii'
SUIT_TSV   = '/mfs/io/groups/dmello/projects/egcerebellum/code/figures/resources/atl-Anatom.tsv'

# Morel thalamus atlas (1 mm, per-nucleus binary volumes)
_MOREL_BASE = os.path.join(ATLAS_DIR, 'MorelAtlasMNI152')
MOREL_MOTOR_NUCLEI = ['VLa', 'VLpd', 'VLpv', 'VAmc', 'VApc']   # cerebellar/BG relay

# ---------------------------------------------------------------------------
# SUIT label lookup
# ---------------------------------------------------------------------------

def _load_suit_labels() -> dict:
    """Return {index: name} for all 34 SUIT labels."""
    labels = {}
    with open(SUIT_TSV) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
    return labels

# Motor-relevant SUIT lobules (1-based NIfTI label):
# Lobules I–IV/V bilat (cerebellar-spinal), VI bilat+vermis (hand/arm),
# VIIIa/b bilat+vermis (hindlimb).
SUIT_MOTOR_LABELS = {
    1:  ('Left I–IV',    '#e31a1c'),
    2:  ('Right I–IV',   '#fb9a99'),
    3:  ('Left V',       '#ff7f00'),
    4:  ('Right V',      '#fdbf6f'),
    5:  ('Left VI',      '#1f78b4'),
    6:  ('Vermis VI',    '#a6cee3'),
    7:  ('Right VI',     '#a6cee3'),
    17: ('Left VIIIa',   '#33a02c'),
    18: ('Vermis VIIIa', '#b2df8a'),
    19: ('Right VIIIa',  '#b2df8a'),
    20: ('Left VIIIb',   '#6a3d9a'),
    21: ('Vermis VIIIb', '#cab2d6'),
    22: ('Right VIIIb',  '#cab2d6'),
    29: ('Left Dentate', '#b15928'),
    30: ('Right Dentate','#b15928'),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prob_path(contrast):
    return os.path.join(GROUP_MAPS, f'{contrast}_prob.nii.gz')

def _mni_template():
    from nilearn.datasets import load_mni152_template
    return load_mni152_template(resolution=2)

def _avg_prob(contrasts: list) -> nib.Nifti1Image:
    """Average probability maps across a list of contrasts."""
    imgs = [nib.load(_prob_path(c)) for c in contrasts if os.path.exists(_prob_path(c))]
    if not imgs:
        return None
    avg = np.mean([np.asarray(i.dataobj, dtype=np.float32) for i in imgs], axis=0)
    return nib.Nifti1Image(avg, imgs[0].affine)

# ---------------------------------------------------------------------------
# Shared single-row prob-map renderer (cerebellar cuts)
# ---------------------------------------------------------------------------

CEREB_AX_CUTS  = [-42, -36, -28, -22, -14]   # inferior → superior lobules
CEREB_COR_CUTS = [-60, -52, -44, -36]          # posterior fossa
CEREB_SAG_CUTS = [-22, -12, 0, 12, 22]         # L hemi · vermis · R hemi

SUBCORT_AX_CUTS  = [-4, 4, 12, 20]    # thalamus / BG levels
SUBCORT_COR_CUTS = [-22, -12, -2, 8]
SUBCORT_SAG_CUTS = [-16, -8, 0, 8, 16]

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
    'combined_motor': '#d62728',
}

ALL_CONTRASTS  = ['LFoot', 'RFoot', 'LHand', 'RHand', 'tongue', 'combined_motor']
EFFECTOR_PAIRS = [('LFoot', 'RFoot'), ('LHand', 'RHand')]


def _plot_prob_row(contrast, axes, template, cut_coords, direction, threshold=0.4):
    prob_path = _prob_path(contrast)
    if not os.path.exists(prob_path):
        for ax in axes:
            ax.set_visible(False)
        return
    prob_img = nib.load(prob_path)
    for ax in axes:
        plotting.plot_stat_map(
            prob_img, bg_img=template,
            display_mode=direction, cut_coords=cut_coords,
            threshold=threshold, vmax=1.0, cmap=CONTRAST_CMAPS[contrast],
            colorbar=False, annotate=False, draw_cross=False, alpha=0.85,
            axes=ax,
        )


# ---------------------------------------------------------------------------
# Fig 1 — Cerebellar probability maps (all 6 contrasts)
# ---------------------------------------------------------------------------

def fig_cereb_prob_maps():
    """Axial + coronal cerebellar cuts, one row per contrast."""
    print('Fig 1 — Cerebellar probability maps...')
    template = _mni_template()

    n_contrast = len(ALL_CONTRASTS)
    fig, axes = plt.subplots(
        n_contrast, 2,
        figsize=(14, 2.6 * n_contrast),
    )

    for row, contrast in enumerate(ALL_CONTRASTS):
        prob_path = _prob_path(contrast)
        if not os.path.exists(prob_path):
            continue
        prob_img = nib.load(prob_path)
        kw = dict(
            bg_img=template, threshold=0.4, vmax=1.0,
            cmap=CONTRAST_CMAPS[contrast],
            colorbar=False, annotate=False, draw_cross=False, alpha=0.85,
        )
        plotting.plot_stat_map(
            prob_img, display_mode='z', cut_coords=CEREB_AX_CUTS,
            axes=axes[row, 0], **kw,
        )
        axes[row, 0].set_title(
            f'{contrast} — axial (z = {", ".join(str(c) for c in CEREB_AX_CUTS)})',
            fontsize=7, pad=2,
        )
        plotting.plot_stat_map(
            prob_img, display_mode='y', cut_coords=CEREB_COR_CUTS,
            axes=axes[row, 1], **kw,
        )
        axes[row, 1].set_title(
            f'{contrast} — coronal (y = {", ".join(str(c) for c in CEREB_COR_CUTS)})',
            fontsize=7, pad=2,
        )
        # Left label
        axes[row, 0].set_ylabel(contrast, fontsize=8, rotation=90, labelpad=4)

    fig.suptitle(
        'Cerebellar Motor Localizer — Group Probability Maps (threshold ≥ 0.40)\n'
        'All 6 contrasts on inferior–superior axial and posterior-fossa coronal cuts',
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fpath = os.path.join(OUT_DIR, 'fig1_cereb_prob_maps.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Fig 2 — Cerebellar somatotopy check (winner-takes-all)
# ---------------------------------------------------------------------------

def fig_cereb_somatotopy():
    """
    Foot / hand / tongue probability winner-takes-all on cerebellar slices.
    Reveals anterior-posterior somatotopic gradient across lobules.
    """
    print('Fig 2 — Cerebellar somatotopy check...')
    template = _mni_template()

    foot_img  = _avg_prob(['LFoot', 'RFoot'])
    hand_img  = _avg_prob(['LHand', 'RHand'])
    tongue_img = nib.load(_prob_path('tongue')) if os.path.exists(_prob_path('tongue')) else None

    if foot_img is None or hand_img is None or tongue_img is None:
        print('  Skipping — missing probability maps')
        return

    # Resample all to common 2mm space (prob maps already are, but tongue may differ)
    ref = foot_img
    hand_img   = nlimage.resample_to_img(hand_img,   ref, interpolation='continuous',
                                         force_resample=True, copy_header=True)
    tongue_img = nlimage.resample_to_img(tongue_img, ref, interpolation='continuous',
                                         force_resample=True, copy_header=True)

    foot_d   = np.asarray(foot_img.dataobj,   dtype=np.float32)
    hand_d   = np.asarray(hand_img.dataobj,   dtype=np.float32)
    tongue_d = np.asarray(tongue_img.dataobj, dtype=np.float32)

    # Winner-takes-all at voxels where any effector >= 0.30
    stack  = np.stack([foot_d, hand_d, tongue_d], axis=0)  # (3, x, y, z)
    winner = np.argmax(stack, axis=0) + 1   # 1=foot, 2=hand, 3=tongue
    active = stack.max(axis=0) >= 0.30
    wta    = (winner * active).astype(np.float32)

    wta_img = nib.Nifti1Image(wta, ref.affine)

    colors = ['#e31a1c', '#1f78b4', '#33a02c']   # foot=red, hand=blue, tongue=green
    cmap   = ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm   = BoundaryNorm(bounds, ncolors=3)

    common_kw = dict(
        bg_img=template, cmap=cmap, vmin=0.5, vmax=3.5,
        threshold=0.5, colorbar=False, annotate=False, draw_cross=False, alpha=0.85,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    plotting.plot_stat_map(
        wta_img, display_mode='z', cut_coords=CEREB_AX_CUTS,
        axes=axes[0], **common_kw,
    )
    axes[0].set_title(
        f'Axial z = {CEREB_AX_CUTS}', fontsize=8, pad=3)

    plotting.plot_stat_map(
        wta_img, display_mode='y', cut_coords=CEREB_COR_CUTS,
        axes=axes[1], **common_kw,
    )
    axes[1].set_title(
        f'Coronal y = {CEREB_COR_CUTS}', fontsize=8, pad=3)

    plotting.plot_stat_map(
        wta_img, display_mode='x', cut_coords=CEREB_SAG_CUTS,
        axes=axes[2], **common_kw,
    )
    axes[2].set_title(
        f'Sagittal x = {CEREB_SAG_CUTS}', fontsize=8, pad=3)

    patches = [
        mpatches.Patch(color='#e31a1c', label='Foot  (LFoot + RFoot avg)'),
        mpatches.Patch(color='#1f78b4', label='Hand  (LHand + RHand avg)'),
        mpatches.Patch(color='#33a02c', label='Tongue'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        'Cerebellar Somatotopy Check — Winner-Takes-All Effector Map (group prob ≥ 0.30)\n'
        'Expected: foot → lobule IV/V (inferior/posterior) · hand → VI · tongue → VI anterior / Crus I',
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fpath = os.path.join(OUT_DIR, 'fig2_cereb_somatotopy.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Fig 3 — SUIT motor ROI atlas overlay
# ---------------------------------------------------------------------------

def fig_suit_motor_rois():
    """
    Show SUIT cerebellar atlas motor lobules labeled on MNI template.
    Two panels: all-lobule atlas (for context) + motor-only subsets.
    """
    print('Fig 3 — SUIT motor ROI atlas...')

    if not os.path.exists(SUIT_ATLAS):
        print(f'  SUIT atlas not found: {SUIT_ATLAS}')
        return

    template = _mni_template()
    suit_img  = nib.load(SUIT_ATLAS)

    # Resample SUIT 1mm to MNI 2mm template space
    suit_2mm = nlimage.resample_to_img(
        suit_img, template, interpolation='nearest',
        force_resample=True, copy_header=True,
    )
    suit_data = np.asarray(suit_2mm.dataobj, dtype=np.int16)

    # --- Panel A: all 34 lobules with discrete colors ---
    labels = _load_suit_labels()

    # Assign a color per label using a qualitative colormap
    import matplotlib.cm as mcm
    n_labels = 34
    base_colors = [mcm.tab20(i % 20) for i in range(n_labels)]
    all_colors  = ['#000000'] + [matplotlib.colors.to_hex(c) for c in base_colors]
    all_cmap    = ListedColormap(all_colors)
    all_bounds  = np.arange(-0.5, n_labels + 1.5, 1.0)
    all_norm    = BoundaryNorm(all_bounds, ncolors=n_labels + 1)  # noqa: F841

    suit_float = nib.Nifti1Image(suit_data.astype(np.float32), suit_2mm.affine)

    # --- Panel B: motor lobules only ---
    motor_vol = np.zeros_like(suit_data, dtype=np.float32)
    motor_color_list = []
    motor_label_list = []
    color_idx = 1
    motor_idx_map = {}
    sorted_motor = sorted(SUIT_MOTOR_LABELS.keys())
    for lbl in sorted_motor:
        name, color = SUIT_MOTOR_LABELS[lbl]
        motor_vol[suit_data == lbl] = float(color_idx)
        motor_color_list.append(color)
        motor_label_list.append(name)
        motor_idx_map[lbl] = color_idx
        color_idx += 1

    motor_img   = nib.Nifti1Image(motor_vol, suit_2mm.affine)
    n_motor     = len(sorted_motor)
    motor_cmap  = ListedColormap(motor_color_list)
    motor_bounds = np.arange(0.5, n_motor + 1.5, 1.0)
    motor_norm  = BoundaryNorm(motor_bounds, ncolors=n_motor)  # noqa: F841

    common_kw = dict(
        bg_img=template, threshold=0.5, colorbar=False,
        annotate=False, draw_cross=False, alpha=0.85,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    cut_sets = [
        ('z', CEREB_AX_CUTS,  'Axial'),
        ('y', CEREB_COR_CUTS, 'Coronal'),
        ('x', CEREB_SAG_CUTS, 'Sagittal'),
    ]

    for col, (direction, cuts, label) in enumerate(cut_sets):
        # Row 0: all lobules
        plotting.plot_stat_map(
            suit_float, display_mode=direction, cut_coords=cuts,
            cmap=all_cmap, vmin=0.5, vmax=n_labels + 0.5,
            axes=axes[0, col], **common_kw,
        )
        axes[0, col].set_title(f'All 34 lobules — {label}', fontsize=8, pad=3)

        # Row 1: motor lobules only
        plotting.plot_stat_map(
            motor_img, display_mode=direction, cut_coords=cuts,
            cmap=motor_cmap, vmin=0.5, vmax=n_motor + 0.5,
            axes=axes[1, col], **common_kw,
        )
        axes[1, col].set_title(f'Motor lobules — {label}', fontsize=8, pad=3)

    # Legend for motor lobules
    patches = [
        mpatches.Patch(color=color, label=motor_label_list[i])
        for i, (_, (_, color)) in enumerate(
            sorted(SUIT_MOTOR_LABELS.items(), key=lambda x: sorted_motor.index(x[0]))
        )
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    fig.suptitle(
        'SUIT Cerebellar Atlas — All Lobules (top) and Motor Lobules (bottom)\n'
        'Motor lobules: I–V (foot/hindlimb)  ·  VI (hand/arm)  ·  VIIIa/b (hand) · Dentate (output)',
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fpath = os.path.join(OUT_DIR, 'fig3_suit_motor_rois.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Fig 4 — Subcortical activation (thalamus + basal ganglia)
# ---------------------------------------------------------------------------

def fig_subcortical_prob_maps():
    """
    Combined_motor and per-effector probability maps on thalamic/BG slices,
    with Morel motor-thalamus nucleus contours (VLa, VLpd, VLpv) overlaid.
    """
    print('Fig 4 — Subcortical activation (thalamus + BG)...')
    template = _mni_template()

    # Build Morel combined motor-thalamus mask (both hemispheres)
    morel_motor_imgs = []
    for nucleus in MOREL_MOTOR_NUCLEI:
        for side in ['left', 'right']:
            fp = os.path.join(_MOREL_BASE, f'{side}-vols-1mm', f'{nucleus}.nii.gz')
            if os.path.exists(fp):
                morel_motor_imgs.append(nib.load(fp))

    morel_combined = None
    if morel_motor_imgs:
        ref   = morel_motor_imgs[0]
        vol   = np.zeros(ref.shape[:3], dtype=np.float32)
        for img in morel_motor_imgs:
            d = np.asarray(img.dataobj, dtype=np.float32)
            if d.shape == vol.shape:
                vol = np.maximum(vol, d)
            else:
                img_r = nlimage.resample_to_img(img, ref, interpolation='nearest',
                                                force_resample=True, copy_header=True)
                vol = np.maximum(vol, np.asarray(img_r.dataobj, dtype=np.float32))
        morel_combined = nib.Nifti1Image((vol > 0).astype(np.float32), ref.affine)
        # Resample to 2mm template
        morel_2mm = nlimage.resample_to_img(
            morel_combined, template, interpolation='nearest',
            force_resample=True, copy_header=True,
        )

    contrasts_to_show = ['combined_motor', 'LFoot', 'RFoot', 'LHand', 'RHand', 'tongue']
    n = len(contrasts_to_show)

    fig, axes = plt.subplots(n, 2, figsize=(12, 2.4 * n))

    for row, contrast in enumerate(contrasts_to_show):
        prob_path = _prob_path(contrast)
        if not os.path.exists(prob_path):
            for ax in axes[row]:
                ax.set_visible(False)
            continue
        prob_img = nib.load(prob_path)
        kw = dict(
            bg_img=template, threshold=0.30, vmax=1.0,
            cmap=CONTRAST_CMAPS[contrast],
            colorbar=False, annotate=False, draw_cross=False, alpha=0.85,
        )
        plotting.plot_stat_map(
            prob_img, display_mode='z', cut_coords=SUBCORT_AX_CUTS,
            axes=axes[row, 0], **kw,
        )
        axes[row, 0].set_title(
            f'{contrast} — axial z = {SUBCORT_AX_CUTS}', fontsize=7, pad=2)

        plotting.plot_stat_map(
            prob_img, display_mode='y', cut_coords=SUBCORT_COR_CUTS,
            axes=axes[row, 1], **kw,
        )
        axes[row, 1].set_title(
            f'{contrast} — coronal y = {SUBCORT_COR_CUTS}', fontsize=7, pad=2)

        axes[row, 0].set_ylabel(contrast, fontsize=8, rotation=90, labelpad=4)

        # Morel motor-thalamus contour overlay (first column only)
        if morel_combined is not None:
            try:
                plotting.plot_roi(
                    morel_2mm, bg_img=template,
                    display_mode='z', cut_coords=SUBCORT_AX_CUTS,
                    colors='white', linewidths=0.8,
                    alpha=0.0,   # transparent fill, contour only
                    axes=axes[row, 0],
                    annotate=False, draw_cross=False,
                )
            except Exception:
                pass

    fig.suptitle(
        'Subcortical Motor Localizer — Group Probability Maps (threshold ≥ 0.30)\n'
        'White contours on axial = Morel motor-thalamus (VLa + VLpd + VLpv + VAmc + VApc)\n'
        'Expected: VL thalamus · putamen/caudate · subthalamic nucleus',
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpath = os.path.join(OUT_DIR, 'fig4_subcortical_prob_maps.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Fig 5 — Combined whole-brain overview: cortex + cerebellum + subcortex
# ---------------------------------------------------------------------------

def fig_whole_brain_overview():
    """
    Glass brain overview showing combined_motor group probability map at
    three thresholds — puts cerebellar and cortical activations in context.
    """
    print('Fig 5 — Whole-brain overview...')
    template = _mni_template()

    prob_path = _prob_path('combined_motor')
    if not os.path.exists(prob_path):
        print('  combined_motor prob map not found')
        return

    prob_img = nib.load(prob_path)

    thresholds = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, thresh in zip(axes, thresholds):
        plotting.plot_glass_brain(
            prob_img,
            display_mode='lyrz',
            threshold=thresh, vmax=1.0,
            cmap='hot', colorbar=False,
            annotate=False, draw_cross=False, alpha=0.8,
            axes=ax,
        )
        ax.set_title(f'Group overlap ≥ {thresh:.0%}', fontsize=9, pad=3)

    fig.suptitle(
        'Combined Motor Localizer — Whole-Brain Group Probability Map\n'
        'Cortex: M1 · SMA · PMd · PMv  |  Cerebellum: lobules IV–VI, VIIIa/b  |  Subcortex: VL thalamus · putamen',
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fpath = os.path.join(OUT_DIR, 'fig5_whole_brain_overview.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Subject-level cerebellar and subcortical localizer figures
# ---------------------------------------------------------------------------

# Re-use gcss_timeseries helpers via direct import
_GCSS_SCRIPT = os.path.join(_SCRIPT_DIR, 'gcss', 'gcss_timeseries.py')

def _load_subject_zmap(subject: str, contrast: str):
    """
    Load and average MSC01 odd-run z-maps for a given contrast.
    Returns a NIfTI image or None.
    """
    import importlib.util, types
    spec = importlib.util.spec_from_file_location('gcss_ts', _GCSS_SCRIPT)
    mod  = importlib.util.load_from_spec = None
    # Direct import via sys.path trick
    if _GCSS_SCRIPT not in sys.path:
        sys.path.insert(0, os.path.dirname(_GCSS_SCRIPT))
    import gcss_timeseries as gts
    return gts.load_contrast_zmap(subject, contrast, gts.ODD_RUNS)


def fig_subject_cereb(subject: str) -> None:
    """
    Subject-specific cerebellar localizer: individual z-maps on cerebellar cuts.
    Generates two panels per contrast (axial + coronal) and a somatotopy check.
    """
    print(f'Subject cerebellar localizer — {subject}...')

    # Lazy-import gcss_timeseries from the gcss sub-folder
    gcss_dir = os.path.join(_SCRIPT_DIR, 'gcss')
    if gcss_dir not in sys.path:
        sys.path.insert(0, gcss_dir)
    import gcss_timeseries as gts

    template = _mni_template()
    contrasts = list(gts.CONTRAST_CONFIG.keys())
    out_subj  = os.path.join(_SCRIPT_DIR, 'gcss', 'roi_figures', subject)
    os.makedirs(out_subj, exist_ok=True)

    # ----------------------------------------------------------------
    # Fig A: cerebellar z-map grid (all contrasts, axial + coronal)
    # ----------------------------------------------------------------
    n = len(contrasts)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.6 * n))

    for row, contrast in enumerate(contrasts):
        z_thresh, _, _ = gts.CONTRAST_CONFIG[contrast]
        zmap_img = gts.load_contrast_zmap(subject, contrast, gts.ODD_RUNS)
        if zmap_img is None:
            axes[row, 0].axis('off'); axes[row, 1].axis('off')
            continue
        kw = dict(
            bg_img=template, threshold=z_thresh, vmax=z_thresh * 2.5,
            cmap=CONTRAST_CMAPS.get(contrast, 'hot'),
            colorbar=False, annotate=False, draw_cross=False, alpha=0.85,
        )
        plotting.plot_stat_map(
            zmap_img, display_mode='z', cut_coords=CEREB_AX_CUTS,
            axes=axes[row, 0], **kw,
        )
        axes[row, 0].set_title(
            f'{contrast} — axial z = {CEREB_AX_CUTS}', fontsize=7, pad=2)
        plotting.plot_stat_map(
            zmap_img, display_mode='y', cut_coords=CEREB_COR_CUTS,
            axes=axes[row, 1], **kw,
        )
        axes[row, 1].set_title(
            f'{contrast} — coronal y = {CEREB_COR_CUTS}', fontsize=7, pad=2)
        axes[row, 0].set_ylabel(contrast, fontsize=8, rotation=90, labelpad=4)

    fig.suptitle(
        f'{subject} — Cerebellar Motor Localizer Z-Maps (odd-run average)\n'
        'Threshold = per-contrast z (3.7 for effectors, 2.3 for combined_motor)',
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fpath = os.path.join(out_subj, f'{subject}_cereb_zmaps.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')

    # ----------------------------------------------------------------
    # Fig B: cerebellar somatotopy for this subject (WTA)
    # ----------------------------------------------------------------
    def _avg_zmap(ctrs):
        imgs = [gts.load_contrast_zmap(subject, c, gts.ODD_RUNS) for c in ctrs]
        imgs = [i for i in imgs if i is not None]
        if not imgs: return None
        avg = np.mean([np.asarray(i.dataobj, dtype=np.float32) for i in imgs], axis=0)
        return nib.Nifti1Image(avg, imgs[0].affine)

    foot_img   = _avg_zmap(['LFoot', 'RFoot'])
    hand_img   = _avg_zmap(['LHand', 'RHand'])
    tongue_img = gts.load_contrast_zmap(subject, 'tongue', gts.ODD_RUNS)

    if foot_img and hand_img and tongue_img:
        ref    = foot_img
        hand_r   = nlimage.resample_to_img(hand_img, ref, interpolation='continuous',
                                           force_resample=True, copy_header=True)
        tongue_r = nlimage.resample_to_img(tongue_img, ref, interpolation='continuous',
                                           force_resample=True, copy_header=True)

        foot_d   = np.asarray(ref.dataobj,     dtype=np.float32)
        hand_d   = np.asarray(hand_r.dataobj,  dtype=np.float32)
        tongue_d = np.asarray(tongue_r.dataobj, dtype=np.float32)

        stack  = np.stack([foot_d, hand_d, tongue_d], axis=0)
        winner = np.argmax(stack, axis=0) + 1
        active = stack.max(axis=0) >= 2.3
        wta    = (winner * active).astype(np.float32)
        wta_img = nib.Nifti1Image(wta, ref.affine)

        colors  = ['#e31a1c', '#1f78b4', '#33a02c']
        cmap_wta = ListedColormap(colors)

        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))
        kw_wta = dict(
            bg_img=template, cmap=cmap_wta, vmin=0.5, vmax=3.5,
            threshold=0.5, colorbar=False, annotate=False, draw_cross=False, alpha=0.85,
        )
        plotting.plot_stat_map(wta_img, display_mode='z', cut_coords=CEREB_AX_CUTS,
                               axes=axes2[0], **kw_wta)
        axes2[0].set_title(f'Axial z = {CEREB_AX_CUTS}', fontsize=8, pad=3)
        plotting.plot_stat_map(wta_img, display_mode='y', cut_coords=CEREB_COR_CUTS,
                               axes=axes2[1], **kw_wta)
        axes2[1].set_title(f'Coronal y = {CEREB_COR_CUTS}', fontsize=8, pad=3)
        plotting.plot_stat_map(wta_img, display_mode='x', cut_coords=CEREB_SAG_CUTS,
                               axes=axes2[2], **kw_wta)
        axes2[2].set_title(f'Sagittal x = {CEREB_SAG_CUTS}', fontsize=8, pad=3)

        patches = [
            mpatches.Patch(color='#e31a1c', label='Foot  (LFoot + RFoot avg)'),
            mpatches.Patch(color='#1f78b4', label='Hand  (LHand + RHand avg)'),
            mpatches.Patch(color='#33a02c', label='Tongue'),
        ]
        fig2.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
                    bbox_to_anchor=(0.5, -0.04))
        fig2.suptitle(
            f'{subject} — Cerebellar Somatotopy (winner-takes-all, z ≥ 2.3)',
            fontsize=9,
        )
        fig2.tight_layout(rect=[0, 0.05, 1, 0.95])
        fpath2 = os.path.join(out_subj, f'{subject}_cereb_somatotopy.pdf')
        fig2.savefig(fpath2, bbox_inches='tight', dpi=150)
        plt.close(fig2)
        print(f'  → {fpath2}')


def fig_subject_subcortical(subject: str) -> None:
    """
    Subject-specific subcortical localizer: individual z-maps on thalamus/BG
    cuts, with Morel motor-thalamus nucleus contours overlaid.
    """
    print(f'Subject subcortical localizer — {subject}...')

    gcss_dir = os.path.join(_SCRIPT_DIR, 'gcss')
    if gcss_dir not in sys.path:
        sys.path.insert(0, gcss_dir)
    import gcss_timeseries as gts

    template  = _mni_template()
    contrasts = list(gts.CONTRAST_CONFIG.keys())
    out_subj  = os.path.join(_SCRIPT_DIR, 'gcss', 'roi_figures', subject)
    os.makedirs(out_subj, exist_ok=True)

    # Build Morel motor-thalamus mask (reuse from fig_subcortical_prob_maps)
    morel_imgs = []
    for nucleus in MOREL_MOTOR_NUCLEI:
        for side in ['left', 'right']:
            fp = os.path.join(_MOREL_BASE, f'{side}-vols-1mm', f'{nucleus}.nii.gz')
            if os.path.exists(fp):
                morel_imgs.append(nib.load(fp))
    morel_2mm = None
    if morel_imgs:
        ref_m = morel_imgs[0]
        vol   = np.zeros(ref_m.shape[:3], dtype=np.float32)
        for img in morel_imgs:
            d = np.asarray(img.dataobj, dtype=np.float32)
            if d.shape == vol.shape:
                vol = np.maximum(vol, d)
        morel_combined = nib.Nifti1Image((vol > 0).astype(np.float32), ref_m.affine)
        morel_2mm = nlimage.resample_to_img(
            morel_combined, template, interpolation='nearest',
            force_resample=True, copy_header=True,
        )

    n   = len(contrasts)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.4 * n))

    for row, contrast in enumerate(contrasts):
        z_thresh, _, _ = gts.CONTRAST_CONFIG[contrast]
        zmap_img = gts.load_contrast_zmap(subject, contrast, gts.ODD_RUNS)
        if zmap_img is None:
            axes[row, 0].axis('off'); axes[row, 1].axis('off')
            continue
        kw = dict(
            bg_img=template, threshold=z_thresh, vmax=z_thresh * 2.5,
            cmap=CONTRAST_CMAPS.get(contrast, 'hot'),
            colorbar=False, annotate=False, draw_cross=False, alpha=0.85,
        )
        plotting.plot_stat_map(
            zmap_img, display_mode='z', cut_coords=SUBCORT_AX_CUTS,
            axes=axes[row, 0], **kw,
        )
        axes[row, 0].set_title(
            f'{contrast} — axial z = {SUBCORT_AX_CUTS}', fontsize=7, pad=2)
        plotting.plot_stat_map(
            zmap_img, display_mode='y', cut_coords=SUBCORT_COR_CUTS,
            axes=axes[row, 1], **kw,
        )
        axes[row, 1].set_title(
            f'{contrast} — coronal y = {SUBCORT_COR_CUTS}', fontsize=7, pad=2)
        axes[row, 0].set_ylabel(contrast, fontsize=8, rotation=90, labelpad=4)
        if morel_2mm is not None:
            try:
                plotting.plot_roi(
                    morel_2mm, bg_img=template,
                    display_mode='z', cut_coords=SUBCORT_AX_CUTS,
                    colors='white', linewidths=0.8, alpha=0.0,
                    axes=axes[row, 0], annotate=False, draw_cross=False,
                )
            except Exception:
                pass

    fig.suptitle(
        f'{subject} — Subcortical Motor Localizer Z-Maps (odd-run average)\n'
        'White contours = Morel motor-thalamus (VLa + VLpd + VLpv + VAmc + VApc)',
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpath = os.path.join(out_subj, f'{subject}_subcortical_zmaps.pdf')
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  → {fpath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description='Subcortical and cerebellar localizer figures.')
    p.add_argument('--subject', default=None,
                   help='Subject ID (e.g. MSC01) for subject-specific figures.')
    p.add_argument('--subject-only', action='store_true',
                   help='Skip group figures.')
    args = p.parse_args()

    print(f'Output directory: {OUT_DIR}\n')

    if not args.subject_only:
        fig_cereb_prob_maps()
        fig_cereb_somatotopy()
        fig_suit_motor_rois()
        fig_subcortical_prob_maps()
        fig_whole_brain_overview()

    if args.subject:
        fig_subject_cereb(args.subject)
        fig_subject_subcortical(args.subject)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
