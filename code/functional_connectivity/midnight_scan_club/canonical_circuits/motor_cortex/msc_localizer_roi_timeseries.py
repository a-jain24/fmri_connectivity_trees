"""
Extract resting-state timeseries using subject-specific localizer-defined ROIs.

For each subject:
  1. Load the subject's motor-task GLM contrast z-maps (foot / hand / tongue).
  2. Intersect each z-map with Glasser360 motor parcels to select the top-k
     most-activated voxels per effector × parcel — giving subject-specific
     functional sub-parcels for area 4 and 3b.
  3. Use the remaining motor parcels (SMA, pre-SMA, PMd, PMv, etc.) as
     whole-parcel ROIs (same for all subjects).
  4. Apply NiftiMasker to resting-state BOLD for each session to extract
     the mean timeseries within each ROI.
  5. Save per-session CSVs (TRs × ROIs) and roi_metadata.json.

All data remain in MNI152NLin2009cAsym 2mm space.  "Subject-specific" refers
to the individualised ROI boundaries derived from each subject's own
activation pattern, not to a different coordinate space.

Usage
-----
    python msc_localizer_roi_timeseries.py [options]
    python msc_localizer_roi_timeseries.py --subjects MSC01 --sessions func01
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

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # motor_cortex/
_CANONICAL_DIR = os.path.dirname(_SCRIPT_DIR)                  # canonical_circuits/
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)               # midnight_scan_club/
sys.path.insert(0, _MSC_DIR)        # msc_paths, msc_mutual_info, msc_chow_liu
sys.path.insert(0, _CANONICAL_DIR)  # canonical_utils

from msc_paths import ATLAS_DIR, ts_root
from canonical_utils import (
    SUB_PARCELLATE, WHOLE_PARCEL_IDX,
    mask_output_dir, timeseries_output_dir,
    load_glasser360_labels,
)

# ---------------------------------------------------------------------------
# External data paths (hard-coded; identical across all MSC scripts on this
# server — same convention as msc_extract_timeseries_optimized.py)
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

# Cosine regressors are added dynamically from the TSV (Bug 3 fix: all ~27 columns).
# Motion derivative columns included per Gordon et al. 12-parameter motion model (Issue 5).
DEFAULT_CONFOUND_COLS = [
    'csf', 'white_matter',
    'trans_x', 'trans_y', 'trans_z',
    'rot_x',   'rot_y',   'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1',   'rot_y_derivative1',   'rot_z_derivative1',
]

# Default z-map averaging: odd-numbered motor runs (run-01, 03, …, 19).
# These show correct foot > rest activation; allruns GLM normalizes against
# the mean motor condition and produces inverted z-scores (Bug 1 fix).
_DEFAULT_MOTOR_RUNS = [f'run-{i:02d}' for i in range(1, 21, 2)]


# ---------------------------------------------------------------------------
# File path helpers
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


def zmap_path(subject: str, effector: str, glm_run: str) -> str:
    return os.path.join(
        GLM_BASE, f'sub-{subject}', 'task-motor', glm_run,
        f'sub-{subject}_task-motor_contrast-{effector}_zmap.nii.gz',
    )


def load_mean_zmap(
    subject: str, effector: str, motor_runs: list
) -> nib.Nifti1Image | None:
    """
    Load z-maps from multiple motor runs and return their voxel-wise mean.

    Averaging individual-run z-maps avoids the differential-contrast inversion
    produced by the allruns GLM, which normalizes foot against the mean across
    all motor conditions (Bug 1 fix).
    """
    arrays = []
    ref_img = None
    for run in motor_runs:
        zp = zmap_path(subject, effector, run)
        if not os.path.exists(zp):
            continue
        img = nib.load(zp)
        if ref_img is None:
            ref_img = img
        arrays.append(np.squeeze(np.asarray(img.dataobj, dtype=np.float32)))
    if not arrays:
        return None
    mean_data = np.mean(arrays, axis=0)
    return nib.Nifti1Image(mean_data, ref_img.affine, ref_img.header)


# ---------------------------------------------------------------------------
# Step 1 — Resample Glasser360 atlas to 2 mm
# ---------------------------------------------------------------------------

def load_glasser_2mm(ref_bold_path: str) -> nib.Nifti1Image:
    """Resample the 1mm Glasser360 atlas to the 2mm BOLD grid."""
    ref_img = nib.load(ref_bold_path)
    # Use a single-volume reference (slicer drops the time axis)
    ref_3d  = ref_img.slicer[..., 0]
    return resample_to_img(
        GLASSER_IMG, ref_3d, interpolation='nearest',
        copy=True, force_resample=True, copy_header=True,
    )


# ---------------------------------------------------------------------------
# Step 2 — Build subject-specific functional ROI masks
# ---------------------------------------------------------------------------

def build_roi_masks(
    subject: str,
    effectors: list,
    top_k: int,
    z_floor: float,
    min_voxels: int,
    motor_runs: list,
    save_masks: bool,
    glasser_2mm: nib.Nifti1Image,
) -> dict:
    """
    For each effector × sub-parcellated parcel, select the top-k voxels by
    z-score within the parcel boundary.

    Returns
    -------
    roi_masks : dict  {effector: {parcel_name: NIfTI binary mask (2mm)}}
    """
    atlas_data = np.asarray(glasser_2mm.dataobj)
    ref_shape  = glasser_2mm.shape

    roi_masks = {eff: {} for eff in effectors}

    for effector in effectors:
        zmap_img = load_mean_zmap(subject, effector, motor_runs)
        if zmap_img is None:
            print(f'  WARNING: no z-maps found for {subject}/{effector} '
                  f'in runs: {motor_runs}')
            continue
        n_found = sum(
            1 for r in motor_runs if os.path.exists(zmap_path(subject, effector, r))
        )
        print(f'  Averaged {n_found}/{len(motor_runs)} runs for {subject}/{effector}')

        # Resample averaged z-map to atlas grid; squeeze degenerate 4th dim
        zmap_data = np.squeeze(np.asarray(
            resample_to_img(zmap_img, glasser_2mm, interpolation='continuous',
                            force_resample=False, copy_header=True).dataobj
        ))

        for parcel_name, parcel_idx in SUB_PARCELLATE.items():
            # glasser360MNI.nii.gz stores labels as 1-based integers
            label_val   = parcel_idx + 1
            parcel_mask = (atlas_data == label_val)
            n_parcel    = int(parcel_mask.sum())

            if n_parcel == 0:
                print(f'  WARNING: parcel {parcel_name} has 0 voxels at 2mm — skipping')
                continue

            flat_z   = zmap_data[parcel_mask]     # 1-D array of z-scores in parcel
            k_actual = min(top_k, n_parcel)

            # Select top-k voxels by z-score.
            # When k_actual == n_parcel we want all voxels; argpartition
            # requires kth < n so we take the argsort path instead.
            if k_actual >= n_parcel:
                top_local = np.arange(n_parcel)
            else:
                top_local = np.argpartition(flat_z, -k_actual)[-k_actual:]

            # Z-floor fallback: if too few top voxels actually exceed z_floor,
            # use the whole parcel to avoid an ROI driven purely by noise.
            n_above_floor = int((flat_z[top_local] >= z_floor).sum())
            if n_above_floor < min_voxels:
                print(
                    f'  FALLBACK [{subject}/{effector}/{parcel_name}]: '
                    f'only {n_above_floor} voxels ≥ z={z_floor:.1f} '
                    f'(max z={flat_z.max():.2f}) — using whole parcel'
                )
                roi_data = parcel_mask.astype(np.uint8)
            else:
                roi_data        = np.zeros(ref_shape, dtype=np.uint8)
                parcel_coords   = np.argwhere(parcel_mask)
                selected_coords = parcel_coords[top_local]
                roi_data[
                    selected_coords[:, 0],
                    selected_coords[:, 1],
                    selected_coords[:, 2],
                ] = 1

            mask_img = new_img_like(glasser_2mm, roi_data, copy_header=True)
            roi_masks[effector][parcel_name] = mask_img

            if save_masks:
                safe_name = parcel_name.replace('/', '-')
                fname = (
                    f'{subject}_effector-{effector}'
                    f'_parcel-{safe_name}_mask.nii.gz'
                )
                mask_img.to_filename(os.path.join(mask_output_dir(subject), fname))

    return roi_masks


# ---------------------------------------------------------------------------
# Step 3 — Build whole-parcel masks (same for all subjects)
# ---------------------------------------------------------------------------

def build_whole_parcel_masks(glasser_2mm: nib.Nifti1Image) -> dict:
    """
    Build binary masks for all motor parcels that are NOT sub-parcellated.
    These are identical across subjects.

    Returns
    -------
    whole_masks : dict  {parcel_name: NIfTI binary mask (2mm)}
    """
    atlas_data = np.asarray(glasser_2mm.dataobj)
    labels     = load_glasser360_labels()
    whole_masks = {}

    for idx in WHOLE_PARCEL_IDX:
        label_val   = idx + 1        # 1-based NIfTI label
        parcel_name = labels[idx]
        mask_data   = (atlas_data == label_val).astype(np.uint8)
        if mask_data.sum() == 0:
            print(f'  WARNING: whole parcel {parcel_name} has 0 voxels at 2mm')
            continue
        whole_masks[parcel_name] = new_img_like(
            glasser_2mm, mask_data, copy_header=True
        )

    return whole_masks


# ---------------------------------------------------------------------------
# Step 4 — Extract timeseries for one session
# ---------------------------------------------------------------------------

def extract_session_timeseries(
    subject: str,
    session: str,
    roi_masks: dict,
    whole_masks: dict,
    confound_cols: list,
    include_motion_outliers: bool = True,
) -> dict | None:
    """
    Load the BOLD volume once, then apply all ROI masks in memory.

    Returns
    -------
    timeseries : dict  {roi_key: np.ndarray shape (n_TRs,)} or None if BOLD absent.
      roi_key format:
        '{effector}__{parcel_name}'  — localizer-defined ROI
        'whole__{parcel_name}'       — whole-parcel ROI
    """
    bp = bold_path(subject, session)
    if not os.path.exists(bp):
        print(f'  BOLD not found, skipping: {bp}')
        return None

    cp = confounds_path(subject, session)
    conf_df = pd.read_csv(
        cp, sep='\t', on_bad_lines='skip',
        encoding='latin-1', engine='python',
    )

    # Cosine regressors: all columns present in TSV (full ~0.0075 Hz high-pass,
    # Bug 3 fix — fMRIPrep generates ~27 cosines, not just 4).
    cosine_cols = [c for c in conf_df.columns if c.startswith('cosine')]

    # Base confound columns — drop any absent from this TSV.
    base_cols = [c for c in confound_cols if c in conf_df.columns]

    # Scrubbing: motion spikes + non-steady-state frames (Bug 2 fix).
    outlier_cols = []
    if include_motion_outliers:
        outlier_cols = [c for c in conf_df.columns
                        if c.startswith('motion_outlier')
                        or c.startswith('non_steady_state')]

    cols      = cosine_cols + base_cols + outlier_cols
    # Derivative columns have NaN in row 0 (no preceding TR); fill with 0.
    confounds = conf_df[cols].fillna(0.0).values

    # Load BOLD once — shape (x, y, z, T)
    bold_img  = nib.load(bp)
    bold_ref  = bold_img.slicer[..., 0]   # 3-D reference for resampling masks
    bold_data = np.asarray(bold_img.dataobj, dtype=np.float32)  # (x,y,z,T)
    n_trs     = bold_data.shape[-1]

    def _mean_ts(mask_img: nib.Nifti1Image) -> np.ndarray:
        """Resample mask to BOLD grid, return mean time series (n_TRs,)."""
        mask_r = resample_to_img(
            mask_img, bold_ref,
            interpolation='nearest', force_resample=False, copy_header=True,
        )
        mask_bool = np.asarray(mask_r.dataobj, dtype=bool)
        # bold_data[mask_bool] → (n_voxels, T); mean over voxels → (T,)
        return bold_data[mask_bool].mean(axis=0)

    # Collect all raw (undetrended, uncleaned) mean time series
    ordered_keys = []
    raw_matrix   = []

    for effector, parcel_dict in roi_masks.items():
        for parcel_name, mask_img in parcel_dict.items():
            ordered_keys.append(f'{effector}__{parcel_name}')
            raw_matrix.append(_mean_ts(mask_img))

    for parcel_name, mask_img in whole_masks.items():
        ordered_keys.append(f'whole__{parcel_name}')
        raw_matrix.append(_mean_ts(mask_img))

    # Stack to (T, n_rois), then clean: detrend + confound regression + standardize
    raw_matrix = np.stack(raw_matrix, axis=1).astype(np.float64)  # (T, n_rois)
    clean_matrix = ni_signal.clean(
        raw_matrix,
        confounds=confounds,
        detrend=True,
        standardize='zscore_sample',
        standardize_confounds='zscore_sample',
        t_r=2.2,
    )  # (T, n_rois)

    return {k: clean_matrix[:, i] for i, k in enumerate(ordered_keys)}


# ---------------------------------------------------------------------------
# Step 5 — Save timeseries and metadata
# ---------------------------------------------------------------------------

def save_timeseries(timeseries_dict: dict, subject: str, session: str) -> None:
    """
    Save (n_TRs × n_ROIs) CSV and roi_metadata.json for one subject/session.
    Column header row lists roi_keys in sorted order.
    """
    out_dir  = timeseries_output_dir(subject, session)
    roi_keys = sorted(timeseries_dict.keys())
    matrix   = np.column_stack([timeseries_dict[k] for k in roi_keys])

    csv_path = os.path.join(out_dir, 'rest.csv')
    np.savetxt(csv_path, matrix, delimiter=',',
               header=','.join(roi_keys), comments='')

    meta = {
        'roi_keys':     roi_keys,
        'n_timepoints': int(matrix.shape[0]),
        'n_rois':       int(matrix.shape[1]),
        'subject':      subject,
        'session':      session,
    }
    with open(os.path.join(out_dir, 'roi_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'    saved {matrix.shape[0]} TRs × {matrix.shape[1]} ROIs → {csv_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Extract resting-state timeseries with subject-specific '
                    'localizer-defined motor ROIs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--subjects', nargs='+',
                   default=[f'MSC{i:02d}' for i in range(1, 11)])
    p.add_argument('--sessions', nargs='+',
                   default=[f'func{i:02d}' for i in range(1, 11)])
    p.add_argument('--effectors', nargs='+',
                   default=['foot', 'hand', 'tongue'])
    p.add_argument('--top-k', type=int, default=50,
                   help='Voxels selected per effector × sub-parcellated parcel.')
    p.add_argument('--z-floor', type=float, default=1.5,
                   help='Fallback to whole parcel if fewer than --min-voxels '
                        'exceed this z-score threshold.')
    p.add_argument('--min-voxels', type=int, default=10,
                   help='Minimum voxels that must exceed z-floor; '
                        'triggers whole-parcel fallback if not met.')
    p.add_argument('--motor-runs', nargs='+', default=_DEFAULT_MOTOR_RUNS,
                   help='Motor task runs to average for z-maps '
                        '(default: odd runs 01,03,...,19; avoids allruns GLM inversion).')
    p.add_argument('--no-motion-outliers', action='store_true',
                   help='Do not append motion_outlier* / non_steady_state_outlier* regressors.')
    p.add_argument('--no-save-masks', action='store_true',
                   help='Skip writing NIfTI mask files to disk.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print('=== MSC Localizer ROI Timeseries Extraction ===')
    print(f'Subjects : {args.subjects}')
    print(f'Sessions : {args.sessions}')
    print(f'Effectors: {args.effectors}')
    print(f'Top-k    : {args.top_k}  |  z-floor: {args.z_floor}  |  '
          f'min-voxels: {args.min_voxels}')
    print(f'Motor runs: {args.motor_runs}')

    # Resample atlas once using the first available BOLD as the 2mm reference.
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

    print(f'\nResampling Glasser360 atlas to 2mm using {ref_bold} as reference...')
    glasser_2mm = load_glasser_2mm(ref_bold)
    print(f'  Atlas resampled: {glasser_2mm.shape}')

    print('\nBuilding whole-parcel masks (shared across subjects)...')
    whole_masks = build_whole_parcel_masks(glasser_2mm)
    print(f'  {len(whole_masks)} whole-parcel ROIs')

    for subject in args.subjects:
        print(f'\n{"="*55}')
        print(f'Subject: {subject}')
        print('="*55')

        print(f'  Building subject-specific ROI masks (top-k={args.top_k})...')
        roi_masks = build_roi_masks(
            subject     = subject,
            effectors   = args.effectors,
            top_k       = args.top_k,
            z_floor     = args.z_floor,
            min_voxels  = args.min_voxels,
            motor_runs  = args.motor_runs,
            save_masks  = not args.no_save_masks,
            glasser_2mm = glasser_2mm,
        )

        n_loc = sum(len(v) for v in roi_masks.values())
        print(f'  {n_loc} localizer ROI masks built')

        for session in args.sessions:
            print(f'  {session} ...', end=' ', flush=True)
            ts = extract_session_timeseries(
                subject                 = subject,
                session                 = session,
                roi_masks               = roi_masks,
                whole_masks             = whole_masks,
                confound_cols           = DEFAULT_CONFOUND_COLS,
                include_motion_outliers = not args.no_motion_outliers,
            )
            if ts is not None:
                save_timeseries(ts, subject, session)

    print('\nDone.')


if __name__ == '__main__':
    main()
