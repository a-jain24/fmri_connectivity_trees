"""
gcss_timeseries.py — GCSS motor parcellation and resting-state timeseries extraction.

Three-phase pipeline:
  Phase 1  Build group probability maps from individual odd-run z-maps.
  Phase 2  Connected-component segmentation of group maps → group parcels.
  Phase 3  Subject-specific fROI extraction (individual activation ∩ group parcel).
  Phase 4  Resting-state timeseries extraction using subject-specific fROI masks.

Contrasts:
  foot, hand, tongue            — bilateral somatotopic effectors (z > 3.7)
  LFoot, RFoot, LHand, RHand    — lateralized effectors        (z > 3.7)
  combined_motor                — average of foot+hand+tongue   (z > 2.3, premotor)
  foot_vs_hand, hand_vs_tongue,
  foot_vs_tongue                — differential contrasts        (z > 2.3)

Output directories (relative to BASE_DIR):
  group maps:   code/.../analysis/canonical_circuits/motor_cortex/gcss/group_maps/
  cluster maps: code/.../analysis/canonical_circuits/motor_cortex/gcss/cluster_maps/
  fROI masks:   code/.../analysis/canonical_circuits/motor_cortex/gcss/froi_masks/{sub}/
  timeseries:   datasets/midnight_scan_club/roi_time_series/{sub}/{ses}/gcss_motor/

Usage
-----
  # All phases for all subjects:
  python gcss_timeseries.py

  # Group maps only (run once before per-subject work):
  python gcss_timeseries.py --phase group_maps

  # Timeseries for specific subjects:
  python gcss_timeseries.py --phase timeseries --subjects MSC01 MSC02
"""

import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import new_img_like, resample_to_img
from nilearn import signal as ni_signal
from scipy.ndimage import label as nd_label, maximum_filter

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # motor_cortex/gcss/
_MOTOR_DIR     = os.path.dirname(_SCRIPT_DIR)                  # motor_cortex/
_CANONICAL_DIR = os.path.dirname(_MOTOR_DIR)                   # canonical_circuits/
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)               # midnight_scan_club/
sys.path.insert(0, _MSC_DIR)
sys.path.insert(0, _CANONICAL_DIR)

from msc_paths import BASE_DIR, ATLAS_DIR, ts_root
from canonical_utils import load_glasser360_labels, MOTOR_CORTEX_ALL

# ---------------------------------------------------------------------------
# External data roots
# ---------------------------------------------------------------------------

FMRIPREP_BASE = (
    '/mfs/io/groups/dmello/projects/cerebellum_reliability'
    '/derivatives/fmriprep/ds000224'
)
GLM_BASE = (
    '/mfs/io/groups/dmello/projects/cerebellum_reliability'
    '/derivatives/firstLevel/multiTaskL1/ds000224'
)
GLASSER_IMG = os.path.join(ATLAS_DIR, 'glasser360', 'glasser360MNI.nii.gz')

# ---------------------------------------------------------------------------
# Contrast configuration
#   key: contrast name used throughout the pipeline
#   value: (z_threshold, group_min_overlap_fraction, min_froi_voxels)
# ---------------------------------------------------------------------------

CONTRAST_CONFIG = {
    # Lateralized somatotopic effectors (contralateral hemisphere dominant).
    # Peak group overlap 0.80–0.90; motor mask removes task-visual VVC artifact.
    'LFoot':          (3.7, 0.60, 15),   # left foot  → right M1 paracentral
    'RFoot':          (3.7, 0.60, 15),   # right foot → left  M1 paracentral
    'LHand':          (3.7, 0.60, 15),   # left hand  → right M1 hand knob
    'RHand':          (3.7, 0.60, 15),   # right hand → left  M1/S1 hand
    # Bilateral tongue: max prob=1.0, bilateral lateral M1 (face/tongue area).
    'tongue':         (3.7, 0.60, 20),
    # Combined motor > rest: premotor network (SMA, PMd, PMv) at relaxed threshold.
    'combined_motor': (2.3, 0.60, 20),
}

# Lateralized hand and foot contrasts require the Glasser motor cortex mask to
# suppress the task-visual VVC activation present in 9/10 MSC subjects.
# tongue and combined_motor are whole-brain clean and do not need masking.
MOTOR_MASKED_CONTRASTS = {'LFoot', 'RFoot', 'LHand', 'RHand'}

# Only odd motor runs for z-map averaging (avoids allruns GLM inversion, Bug 1).
ODD_RUNS  = [f'run-{i:02d}' for i in range(1, 21, 2)]
EVEN_RUNS = [f'run-{i:02d}' for i in range(2, 21, 2)]   # reserved for test-retest

DEFAULT_CONFOUND_COLS = [
    'csf', 'white_matter',
    'trans_x', 'trans_y', 'trans_z',
    'rot_x',   'rot_y',   'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1',   'rot_y_derivative1',   'rot_z_derivative1',
]

# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _gcss_base() -> str:
    d = os.path.join(_MSC_DIR, 'analysis', 'canonical_circuits',
                     'motor_cortex', 'gcss')
    os.makedirs(d, exist_ok=True)
    return d


def group_maps_dir() -> str:
    d = os.path.join(_gcss_base(), 'group_maps')
    os.makedirs(d, exist_ok=True)
    return d


def cluster_maps_dir() -> str:
    d = os.path.join(_gcss_base(), 'cluster_maps')
    os.makedirs(d, exist_ok=True)
    return d


def froi_masks_dir(subject: str) -> str:
    d = os.path.join(_gcss_base(), 'froi_masks', subject)
    os.makedirs(d, exist_ok=True)
    return d


def timeseries_output_dir(subject: str, session: str) -> str:
    d = os.path.join(ts_root(), subject, session, 'gcss_motor')
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# File path helpers for raw data
# ---------------------------------------------------------------------------

def bold_path(subject: str, session: str) -> str:
    return os.path.join(
        FMRIPREP_BASE, f'sub-{subject}', f'ses-{session}', 'func',
        f'sub-{subject}_ses-{session}_task-rest'
        '_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
    )


def confounds_path(subject: str, session: str) -> str:
    return os.path.join(
        FMRIPREP_BASE, f'sub-{subject}', f'ses-{session}', 'func',
        f'sub-{subject}_ses-{session}_task-rest'
        '_desc-confounds_timeseries.tsv',
    )


def zmap_file(subject: str, contrast: str, run: str) -> str:
    """Return the on-disk path for a primary contrast z-map file."""
    return os.path.join(
        GLM_BASE, f'sub-{subject}', 'task-motor', run,
        f'sub-{subject}_task-motor_contrast-{contrast}_zmap.nii.gz',
    )


# ---------------------------------------------------------------------------
# Phase 1 helpers: z-map loading
# ---------------------------------------------------------------------------

_PRIMARY_CONTRASTS = {'foot', 'hand', 'tongue', 'LFoot', 'RFoot', 'LHand', 'RHand'}


def load_mean_zmap(subject: str, contrast: str, runs: list) -> nib.Nifti1Image | None:
    """
    Average individual-run z-maps over the specified runs.
    Returns None if no valid files exist for this subject/contrast.
    """
    arrays, ref_img = [], None
    for run in runs:
        fp = zmap_file(subject, contrast, run)
        if not os.path.exists(fp):
            continue
        img = nib.load(fp)
        if ref_img is None:
            ref_img = img
        arrays.append(np.squeeze(np.asarray(img.dataobj, dtype=np.float32)))
    if not arrays:
        return None
    return nib.Nifti1Image(np.mean(arrays, axis=0), ref_img.affine, ref_img.header)


def load_contrast_zmap(
    subject: str, contrast: str, runs: list
) -> nib.Nifti1Image | None:
    """
    Return a z-map for any contrast type.
    Primary contrasts are loaded directly; derived contrasts are computed.
    """
    if contrast in _PRIMARY_CONTRASTS:
        return load_mean_zmap(subject, contrast, runs)

    if contrast == 'combined_motor':
        parts, ref_img = [], None
        for c in ('foot', 'hand', 'tongue'):
            img = load_mean_zmap(subject, c, runs)
            if img is None:
                continue
            if ref_img is None:
                ref_img = img
            parts.append(np.squeeze(np.asarray(img.dataobj, dtype=np.float32)))
        if not parts:
            return None
        return nib.Nifti1Image(np.mean(parts, axis=0), ref_img.affine, ref_img.header)

    if '_vs_' in contrast:
        c_a, c_b = contrast.split('_vs_', 1)
        img_a = load_mean_zmap(subject, c_a, runs)
        img_b = load_mean_zmap(subject, c_b, runs)
        if img_a is None or img_b is None:
            return None
        diff = (np.squeeze(np.asarray(img_a.dataobj, dtype=np.float32)) -
                np.squeeze(np.asarray(img_b.dataobj, dtype=np.float32)))
        return nib.Nifti1Image(diff, img_a.affine, img_a.header)

    raise ValueError(f'Unknown contrast: {contrast!r}')


# ---------------------------------------------------------------------------
# Phase 1: Group probability map
# ---------------------------------------------------------------------------

def build_group_probability_map(
    contrast: str,
    subjects: list,
    z_threshold: float,
    motor_runs: list,
    ref_shape: tuple,
    ref_affine: np.ndarray,
) -> np.ndarray:
    """
    Compute voxel-wise activation probability across subjects.

    Each subject contributes one binary map (thresholded odd-run average).
    Returns a float32 array in [0, 1] with shape ref_shape.
    """
    overlap = np.zeros(ref_shape, dtype=np.float32)
    n_valid = 0

    for subject in subjects:
        img = load_contrast_zmap(subject, contrast, motor_runs)
        if img is None:
            print(f'    [{contrast}] {subject}: no z-map — skipping')
            continue
        # Resample to reference grid if needed
        if img.shape[:3] != ref_shape:
            ref_nii = nib.Nifti1Image(np.zeros(ref_shape, dtype=np.float32), ref_affine)
            img = resample_to_img(img, ref_nii, interpolation='continuous',
                                   force_resample=True, copy_header=True)
        data = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
        overlap += (data > z_threshold).astype(np.float32)
        n_valid += 1

    if n_valid == 0:
        return overlap
    return overlap / n_valid


# ---------------------------------------------------------------------------
# Phase 2: Cluster segmentation and Glasser annotation
# ---------------------------------------------------------------------------

def segment_group_map(
    group_prob: np.ndarray,
    min_overlap: float,
    min_cluster_voxels: int = 10,
    motor_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """
    Threshold the group probability map and label connected components.

    Parameters
    ----------
    group_prob        : float32 (x, y, z) in [0, 1]
    min_overlap       : retain voxels where group_prob >= this value
    min_cluster_voxels: discard clusters smaller than this
    motor_mask        : optional binary float32 array (same shape as group_prob).
                        When provided, non-motor voxels are zeroed before
                        thresholding, preventing off-target clusters (e.g.
                        visual cortex in hand contrasts).

    Returns
    -------
    cluster_map : int16 (x, y, z), each cluster has a unique label 1..N, 0=bg
    n_clusters  : int
    """
    if motor_mask is not None:
        group_prob = group_prob * motor_mask
    binary = (group_prob >= min_overlap)
    struct = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    raw_labels, n_raw = nd_label(binary, structure=struct)

    # Remove clusters smaller than min_cluster_voxels
    cluster_map = np.zeros_like(raw_labels, dtype=np.int16)
    kept = 0
    for cid in range(1, n_raw + 1):
        mask = raw_labels == cid
        if mask.sum() >= min_cluster_voxels:
            kept += 1
            cluster_map[mask] = kept

    return cluster_map, kept


def _cluster_centroid_mni(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """MNI mm centre-of-mass for a boolean cluster mask."""
    coords = np.array(np.where(mask), dtype=float).T   # (N, 3)
    if len(coords) == 0:
        return np.zeros(3)
    mean_vox = coords.mean(axis=0)
    return nib.affines.apply_affine(affine, mean_vox)


def annotate_clusters(
    cluster_map: np.ndarray,
    cluster_map_affine: np.ndarray,
    glasser_data: np.ndarray,
    glasser_labels: list,
) -> dict:
    """
    For each cluster, record:
      - 'label'        : unique ROI key of the form '{contrast}__{parcel_name}'
      - 'peak_glasser' : Glasser parcel name with most voxel overlap
      - 'overlap_frac' : fraction of cluster voxels inside that parcel
      - 'centroid_mni' : [x, y, z] MNI mm
      - 'n_voxels'     : cluster size

    Returns dict keyed by integer cluster id.
    """
    glasser_resampled = glasser_data

    cluster_ids = np.unique(cluster_map)
    cluster_ids = cluster_ids[cluster_ids > 0]

    info = {}
    used_parcel_counts: dict = {}

    for cid in sorted(cluster_ids.tolist()):
        mask = cluster_map == cid
        centroid = _cluster_centroid_mni(mask, cluster_map_affine)

        # Find most-overlapping Glasser parcel
        gl_vals = glasser_resampled[mask].astype(int)
        valid = gl_vals > 0
        if valid.sum() > 0:
            counts = np.bincount(gl_vals[valid], minlength=len(glasser_labels) + 1)
            best_gl_idx = int(np.argmax(counts[1:])) + 1   # 1-based
            parcel_name = glasser_labels[best_gl_idx - 1]  # 0-based list
            overlap_frac = float(counts[best_gl_idx]) / float(valid.sum())
        else:
            parcel_name  = f'cluster_{cid:02d}'
            overlap_frac = 0.0

        # Make label unique within this run of annotate_clusters
        used_parcel_counts[parcel_name] = used_parcel_counts.get(parcel_name, 0) + 1
        cnt = used_parcel_counts[parcel_name]
        unique_label = parcel_name if cnt == 1 else f'{parcel_name}_v{cnt}'

        info[int(cid)] = {
            'label':        unique_label,
            'peak_glasser': parcel_name,
            'overlap_frac': round(overlap_frac, 3),
            'centroid_mni': [round(float(v), 1) for v in centroid],
            'n_voxels':     int(mask.sum()),
        }

    return info


# ---------------------------------------------------------------------------
# Phase 3: Subject-specific fROI extraction
# ---------------------------------------------------------------------------

def semantic_roi_name(contrast: str, info: dict) -> str:
    """Map a Glasser cluster (contrast + centroid) to an anatomical ROI name."""
    cx, cy, cz = info['centroid_mni']
    if contrast == 'LFoot':         return 'R_M1_foot'
    if contrast == 'RFoot':         return 'L_M1_foot'
    if contrast == 'LHand':         return 'R_M1_hand'
    if contrast == 'RHand':         return 'L_M1_hand'
    if contrast == 'tongue':
        return 'L_M1_tongue' if cx < 0 else 'R_M1_tongue'
    if contrast == 'combined_motor':
        if abs(cx) < 12:                      return 'SMA'
        if cx < -12 and cz > 50:              return 'L_PMd'
        if cx >  12 and cy > -8 and cz > 25:  return 'R_PMv'
        return 'L_S1S2' if cx < 0 else 'R_S1S2'
    side = 'L' if cx < 0 else 'R'
    return f'{side}_{contrast}'


def extract_subject_frois(
    subject: str,
    contrast: str,
    cluster_map: np.ndarray,
    cluster_info: dict,
    z_threshold: float,
    min_froi_voxels: int,
    motor_runs: list,
    glasser_2mm: nib.Nifti1Image,
    save_masks: bool,
) -> dict:
    """
    For each group cluster, intersect with this subject's activation map, then
    merge clusters sharing the same semantic ROI name (union of voxels).

    Returns
    -------
    froi_masks : dict  {sem_name: NIfTI binary mask (2 mm)}
    """
    zmap_img = load_contrast_zmap(subject, contrast, motor_runs)
    if zmap_img is None:
        print(f'    [{contrast}] {subject}: no z-map')
        return {}

    # Resample z-map to 2 mm atlas grid
    zmap_data = np.squeeze(np.asarray(
        resample_to_img(zmap_img, glasser_2mm, interpolation='continuous',
                        force_resample=False, copy_header=True).dataobj,
        dtype=np.float32,
    ))

    subject_active         = (zmap_data > z_threshold).astype(np.uint8)
    subject_active_relaxed = (zmap_data > 2.3).astype(np.uint8)

    mdir = froi_masks_dir(subject)
    sem_accum: dict[str, np.ndarray] = {}

    for cid, info in cluster_info.items():
        group_cluster = (cluster_map == cid).astype(np.uint8)
        froi_data = subject_active * group_cluster

        if int(froi_data.sum()) < min_froi_voxels:
            froi_relaxed = subject_active_relaxed * group_cluster
            if int(froi_relaxed.sum()) >= min_froi_voxels:
                froi_data = froi_relaxed
            else:
                froi_data = group_cluster  # whole-parcel fallback

        sem_name = semantic_roi_name(contrast, info)
        if sem_name not in sem_accum:
            sem_accum[sem_name] = np.zeros_like(froi_data, dtype=np.uint8)
        np.clip(sem_accum[sem_name] + froi_data, 0, 1, out=sem_accum[sem_name])

    froi_masks: dict = {}
    for sem_name, merged_data in sem_accum.items():
        mask_img = new_img_like(glasser_2mm, merged_data, copy_header=True)
        froi_masks[sem_name] = mask_img
        if save_masks:
            fname = f'{subject}_gcss_{sem_name}_mask.nii.gz'
            mask_img.to_filename(os.path.join(mdir, fname))

    return froi_masks


# ---------------------------------------------------------------------------
# Phase 4: Resting-state timeseries extraction
# ---------------------------------------------------------------------------

def extract_session_timeseries(
    subject: str,
    session: str,
    froi_masks: dict,
    confound_cols: list,
    include_motion_outliers: bool = True,
) -> dict | None:
    """
    Load the BOLD volume once, apply all fROI masks, clean signal.

    Returns
    -------
    dict {roi_key: np.ndarray (n_TRs,)} or None if BOLD missing.
    """
    bp = bold_path(subject, session)
    if not os.path.exists(bp):
        print(f'    BOLD not found: {bp}')
        return None

    cp = confounds_path(subject, session)
    conf_df = pd.read_csv(cp, sep='\t', on_bad_lines='skip',
                          encoding='latin-1', engine='python')

    cosine_cols  = [c for c in conf_df.columns if c.startswith('cosine')]
    base_cols    = [c for c in confound_cols if c in conf_df.columns]
    outlier_cols = []
    if include_motion_outliers:
        outlier_cols = [c for c in conf_df.columns
                        if c.startswith('motion_outlier')
                        or c.startswith('non_steady_state')]
    cols      = cosine_cols + base_cols + outlier_cols
    confounds = conf_df[cols].fillna(0.0).values

    bold_img  = nib.load(bp)
    bold_ref  = bold_img.slicer[..., 0]
    bold_data = np.asarray(bold_img.dataobj, dtype=np.float32)

    def _mean_ts(mask_img: nib.Nifti1Image) -> np.ndarray:
        mask_r    = resample_to_img(mask_img, bold_ref, interpolation='nearest',
                                    force_resample=False, copy_header=True)
        mask_bool = np.asarray(mask_r.dataobj, dtype=bool)
        if not mask_bool.any():
            return np.zeros(bold_data.shape[-1], dtype=np.float32)
        return bold_data[mask_bool].mean(axis=0)

    ordered_keys = sorted(froi_masks.keys())
    raw_matrix   = np.stack([_mean_ts(froi_masks[k]) for k in ordered_keys], axis=1)

    clean_matrix = ni_signal.clean(
        raw_matrix.astype(np.float64),
        confounds=confounds,
        detrend=True,
        standardize='zscore_sample',
        standardize_confounds='zscore_sample',
        t_r=2.2,
    )
    return {k: clean_matrix[:, i] for i, k in enumerate(ordered_keys)}


# ---------------------------------------------------------------------------
# Save timeseries
# ---------------------------------------------------------------------------

def save_timeseries(
    timeseries_dict: dict, subject: str, session: str,
    contrast_meta: dict,
) -> None:
    """
    Save rest.csv (TRs × ROIs) and roi_metadata.json for one subject/session.
    """
    out_dir  = timeseries_output_dir(subject, session)
    roi_keys = sorted(timeseries_dict.keys())
    matrix   = np.column_stack([timeseries_dict[k] for k in roi_keys])

    csv_path = os.path.join(out_dir, 'rest.csv')
    np.savetxt(csv_path, matrix, delimiter=',',
               header=','.join(roi_keys), comments='')

    meta = {
        'roi_keys':      roi_keys,
        'n_timepoints':  int(matrix.shape[0]),
        'n_rois':        int(matrix.shape[1]),
        'subject':       subject,
        'session':       session,
        'contrast_info': contrast_meta,
    }
    with open(os.path.join(out_dir, 'roi_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'    saved {matrix.shape[0]} TRs × {matrix.shape[1]} ROIs → {csv_path}')


# ---------------------------------------------------------------------------
# Top-level phase runners
# ---------------------------------------------------------------------------

def run_phase_group_maps(
    contrasts: list,
    subjects: list,
    motor_runs: list,
    ref_shape: tuple,
    ref_affine: np.ndarray,
    glasser_2mm: nib.Nifti1Image,
    glasser_labels: list,
    force: bool = False,
) -> dict:
    """
    Phase 1 + 2 for all contrasts.

    Returns
    -------
    cluster_data : {contrast: (cluster_map np.ndarray, cluster_info dict)}
    """
    glasser_data_2mm = np.asarray(glasser_2mm.dataobj, dtype=np.int32)

    # Binary motor-cortex mask (MOTOR_CORTEX_ALL Glasser parcel voxels only).
    # Applied to contrasts in MOTOR_MASKED_CONTRASTS to suppress off-target
    # activations (visual cortex in hand contrasts; non-motor in differentials).
    _motor_nii_indices = [i + 1 for i in MOTOR_CORTEX_ALL]  # NIfTI labels are 1-based
    motor_mask = np.isin(glasser_data_2mm, _motor_nii_indices).astype(np.float32)
    print(f'  Motor cortex mask: {int(motor_mask.sum())} voxels '
          f'({len(MOTOR_CORTEX_ALL)} Glasser parcels)')

    cluster_data = {}

    for contrast in contrasts:
        z_thresh, min_overlap, _ = CONTRAST_CONFIG[contrast]
        use_motor_mask = contrast in MOTOR_MASKED_CONTRASTS
        prob_path    = os.path.join(group_maps_dir(),   f'{contrast}_prob.nii.gz')
        cluster_path = os.path.join(cluster_maps_dir(), f'{contrast}_clusters.nii.gz')
        info_path    = os.path.join(cluster_maps_dir(), f'{contrast}_cluster_info.json')

        if not force and os.path.exists(cluster_path) and os.path.exists(info_path):
            print(f'  [{contrast}] Loading cached cluster map...')
            cluster_map = np.asarray(nib.load(cluster_path).dataobj, dtype=np.int16)
            with open(info_path) as fh:
                # JSON keys are strings; convert to int
                raw = json.load(fh)
                cluster_info = {int(k): v for k, v in raw.items()}
        else:
            mask_note = ' [motor masked]' if use_motor_mask else ''
            print(f'  [{contrast}] Building group probability map '
                  f'(z > {z_thresh}, overlap ≥ {min_overlap:.0%}){mask_note}...')
            group_prob = build_group_probability_map(
                contrast, subjects, z_thresh, motor_runs, ref_shape, ref_affine)

            # Save probability map (before masking so full-brain map is preserved)
            nib.save(nib.Nifti1Image(group_prob, ref_affine), prob_path)

            cluster_map, n = segment_group_map(
                group_prob, min_overlap,
                motor_mask=motor_mask if use_motor_mask else None,
            )
            print(f'    → {n} clusters retained')

            nib.save(nib.Nifti1Image(cluster_map, ref_affine), cluster_path)

            cluster_info = annotate_clusters(
                cluster_map, ref_affine, glasser_data_2mm, glasser_labels)

            with open(info_path, 'w') as fh:
                json.dump(cluster_info, fh, indent=2)

        if not cluster_info:
            print(f'  [{contrast}] WARNING: no clusters found — skipping')
            continue

        cluster_data[contrast] = (cluster_map, cluster_info)
        for cid, info in cluster_info.items():
            print(f'    cluster {cid:02d}: {info["label"]:30s} '
                  f'{info["n_voxels"]:4d} vox  centroid {info["centroid_mni"]}')

    return cluster_data


def run_phase_timeseries(
    subjects: list,
    sessions: list,
    cluster_data: dict,
    glasser_2mm: nib.Nifti1Image,
    motor_runs: list,
    save_masks: bool,
    force: bool,
) -> None:
    """Phase 3 + 4 for all subjects × sessions."""
    for subject in subjects:
        print(f'\n{"="*55}\nSubject: {subject}')

        # --- Phase 3: Build subject-specific fROI masks ---
        all_froi_masks: dict = {}
        all_contrast_meta: dict = {}

        for contrast, (cluster_map, cluster_info) in cluster_data.items():
            z_thresh, _, min_vox = CONTRAST_CONFIG[contrast]
            masks = extract_subject_frois(
                subject      = subject,
                contrast     = contrast,
                cluster_map  = cluster_map,
                cluster_info = cluster_info,
                z_threshold  = z_thresh,
                min_froi_voxels = min_vox,
                motor_runs   = motor_runs,
                glasser_2mm  = glasser_2mm,
                save_masks   = save_masks,
            )
            all_froi_masks.update(masks)
            for cid, info in cluster_info.items():
                sem_name = semantic_roi_name(contrast, info)
                if sem_name not in all_contrast_meta:
                    all_contrast_meta[sem_name] = {
                        'contrast':        contrast,
                        'semantic_name':   sem_name,
                        'source_clusters': [],
                    }
                all_contrast_meta[sem_name]['source_clusters'].append({
                    'cluster_id':   cid,
                    'peak_glasser': info['peak_glasser'],
                    'centroid_mni': info['centroid_mni'],
                })

        if not all_froi_masks:
            print(f'  WARNING: no fROI masks built for {subject} — skipping timeseries')
            continue

        print(f'  {len(all_froi_masks)} fROI masks ready: '
              + ', '.join(sorted(all_froi_masks)))

        # --- Phase 4: Extract timeseries per session ---
        for session in sessions:
            out_csv = os.path.join(timeseries_output_dir(subject, session), 'rest.csv')
            if not force and os.path.exists(out_csv):
                print(f'  {session}: already exists, skipping (use --force to overwrite)')
                continue

            print(f'  {session} ...', end=' ', flush=True)
            ts = extract_session_timeseries(
                subject  = subject,
                session  = session,
                froi_masks = all_froi_masks,
                confound_cols = DEFAULT_CONFOUND_COLS,
            )
            if ts is not None:
                save_timeseries(ts, subject, session, all_contrast_meta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='GCSS motor parcellation and resting-state timeseries extraction.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--subjects', nargs='+',
                   default=[f'MSC{i:02d}' for i in range(1, 11)])
    p.add_argument('--sessions', nargs='+',
                   default=[f'func{i:02d}' for i in range(1, 11)])
    p.add_argument('--contrasts', nargs='+', default=list(CONTRAST_CONFIG),
                   help='Contrasts to include. Default: all defined contrasts.')
    p.add_argument('--motor-runs', nargs='+', default=ODD_RUNS,
                   help='Motor task runs to average for z-maps (default: odd runs).')
    p.add_argument('--phase', choices=['all', 'group_maps', 'timeseries'], default='all',
                   help='"group_maps" = Phase 1+2 only; '
                        '"timeseries" = Phase 3+4 (requires cached group maps); '
                        '"all" = full pipeline.')
    p.add_argument('--no-save-masks', action='store_true',
                   help='Skip writing per-subject fROI NIfTI masks to disk.')
    p.add_argument('--force', action='store_true',
                   help='Recompute and overwrite existing outputs.')
    return p.parse_args()


def main():
    args = parse_args()

    # Validate requested contrasts
    invalid = [c for c in args.contrasts if c not in CONTRAST_CONFIG]
    if invalid:
        raise ValueError(f'Unknown contrasts: {invalid}. '
                         f'Valid: {list(CONTRAST_CONFIG)}')

    print('=== GCSS Motor Parcellation: Timeseries Extraction ===')
    print(f'Subjects  : {args.subjects}')
    print(f'Sessions  : {args.sessions}')
    print(f'Contrasts : {args.contrasts}')
    print(f'Motor runs: {args.motor_runs}')
    print(f'Phase     : {args.phase}')

    # Resample Glasser atlas to 2 mm using first available BOLD as reference
    ref_bold = None
    for sub in args.subjects:
        for ses in args.sessions:
            bp = bold_path(sub, ses)
            if os.path.exists(bp):
                ref_bold = bp
                break
        if ref_bold:
            break
    if ref_bold is None:
        raise RuntimeError('No BOLD files found for any requested subject/session.')

    print(f'\nResampling Glasser360 to 2 mm (reference: {ref_bold})...')
    ref_img     = nib.load(ref_bold)
    ref_3d      = ref_img.slicer[..., 0]
    glasser_2mm = resample_to_img(
        GLASSER_IMG, ref_3d, interpolation='nearest',
        copy=True, force_resample=True, copy_header=True,
    )
    ref_shape   = glasser_2mm.shape
    ref_affine  = glasser_2mm.affine
    print(f'  Atlas shape: {ref_shape}')

    glasser_labels = load_glasser360_labels()

    # --- Phase 1+2 ---
    cluster_data = {}
    if args.phase in ('all', 'group_maps'):
        print('\n--- Phase 1+2: Group probability maps and cluster segmentation ---')
        cluster_data = run_phase_group_maps(
            contrasts     = args.contrasts,
            subjects      = args.subjects,
            motor_runs    = args.motor_runs,
            ref_shape     = ref_shape,
            ref_affine    = ref_affine,
            glasser_2mm   = glasser_2mm,
            glasser_labels = glasser_labels,
            force         = args.force,
        )

    # For timeseries-only phase, load cached cluster data
    if args.phase == 'timeseries':
        print('\n--- Loading cached cluster maps ---')
        for contrast in args.contrasts:
            cluster_path = os.path.join(cluster_maps_dir(), f'{contrast}_clusters.nii.gz')
            info_path    = os.path.join(cluster_maps_dir(), f'{contrast}_cluster_info.json')
            if not os.path.exists(cluster_path) or not os.path.exists(info_path):
                print(f'  [{contrast}] No cached cluster map found — run '
                      f'--phase group_maps first.')
                continue
            cluster_map = np.asarray(nib.load(cluster_path).dataobj, dtype=np.int16)
            with open(info_path) as fh:
                raw = json.load(fh)
                cluster_info = {int(k): v for k, v in raw.items()}
            cluster_data[contrast] = (cluster_map, cluster_info)

    # --- Phase 3+4 ---
    if args.phase in ('all', 'timeseries') and cluster_data:
        print('\n--- Phase 3+4: Subject-specific fROIs and timeseries extraction ---')
        run_phase_timeseries(
            subjects     = args.subjects,
            sessions     = args.sessions,
            cluster_data = cluster_data,
            glasser_2mm  = glasser_2mm,
            motor_runs   = args.motor_runs,
            save_masks   = not args.no_save_masks,
            force        = args.force,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
