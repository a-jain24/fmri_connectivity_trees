# Subject-Space Functional ROI Timeseries — Implementation Plan

## Overview

This plan describes `msc_localizer_roi_timeseries.py`, a new script under
`canonical_circuits/` that defines **individual-specific functional ROIs** for
each effector (foot / hand / tongue) by intersecting each subject's motor-task
activation maps with Glasser360 motor parcels, then extracts resting-state
timeseries from those subject-specific masks.

The output replaces the uniform MNI atlas ROIs used by `msc_extract_timeseries.py`
with ROIs whose *boundaries are determined by each subject's own activation
pattern*. Every subject gets a different set of voxel masks even within the same
nominal parcel (e.g., area 4). This is what "subject space" means here: the
functional sub-parcellation varies per individual, not the coordinate space
(which remains MNI152NLin2009cAsym 2mm throughout because no T1w-space BOLD is
available in the fmriprep derivatives).

---

## Data Inventory

### fmriprep BOLD (MNI space, 2 mm)

```
/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/
  ds000224/sub-{sub}/ses-{session}/func/
    sub-{sub}_ses-{session}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
    sub-{sub}_ses-{session}_task-rest_desc-confounds_timeseries.tsv
```

Subjects: MSC01–MSC10. Sessions: func01–func10 (10 per subject).
TR = 2 s. File-ID string for rest: `rest` (from `msc_rest_id.txt`).

The `s_` prefix marks spatially-smoothed copies. Use the **unsmoothed** file
(`sub-…`, not `s_sub-…`) for timeseries extraction to avoid blurring across
the functionally-defined ROI boundary.

### Existing Motor-Task GLM Outputs

Pre-computed first-level GLM outputs (nilearn, all-runs fixed-effects pooled
across all 20 motor runs per subject) are at:

```
/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/firstLevel/
  multiTaskL1/ds000224/sub-{sub}/task-motor/allruns/
    sub-{sub}_task-motor_contrast-foot_zmap.nii.gz    # LFoot + RFoot vs baseline
    sub-{sub}_task-motor_contrast-hand_zmap.nii.gz    # LHand + RHand vs baseline
    sub-{sub}_task-motor_contrast-tongue_zmap.nii.gz  # Tongue vs baseline
    sub-{sub}_task-motor_contrast-foot_stat.nii.gz    # same, t-stat
    sub-{sub}_task-motor_contrast-hand_stat.nii.gz
    sub-{sub}_task-motor_contrast-tongue_stat.nii.gz
```

All 10 subjects (MSC01–MSC10) have these outputs. Space: MNI152NLin2009cAsym
2 mm (91×109×91), matching the fmriprep BOLD.

Per-run outputs also exist (`run-01` through `run-20`) for cross-run stability
checks, but `allruns` is the primary source for ROI definition.

### Glasser360 Atlas

```
atlases/glasser360/glasser360MNI.nii.gz       # integer label volume, MNI 1mm
atlases/glasser360/glasser360NodeNames.txt    # 360 labels, Right 0-179 / Left 180-359
```

Labels are already in the same MNI space; will be resampled to 2mm to match
the BOLD grid.

---

## Conceptual Design

### Why Intersection Rather Than Peak-Coordinate

The Glasser360 atlas defines the *boundary* of each motor parcel but treats
the interior as homogeneous. Within area 4, for example, the foot representation
sits in the paracentral lobule (superior, medial) while the hand representation
occupies the middle of the central sulcus. At 2mm resolution these occupy
distinct voxels within the same parcel label.

The intersection approach:
1. Restricts candidate voxels to those inside the target Glasser360 parcel —
   preventing the functional ROI from expanding into an anatomically wrong region.
2. Ranks voxels by each subject's own z-score for the effector contrast.
3. Selects the top-k voxels (or applies a z-threshold with a fallback minimum)
   to form the functional ROI for that subject.

The result is that two subjects may have the same parcel boundary but different
ROI voxels, reflecting their individual somatotopic organization.

### Effector → Contrast Mapping

| Effector label | Contrast file             | Conditions pooled |
|---|---|---|
| `foot`         | `contrast-foot_zmap`      | LFoot + RFoot     |
| `hand`         | `contrast-hand_zmap`      | LHand + RHand     |
| `tongue`       | `contrast-tongue_zmap`    | Tongue            |

The bilateral (`foot`, `hand`) contrasts are preferred over unilateral ones
because motor cortex has bilateral representation and pooling increases SNR.

### Motor Parcels to Sub-Parcellate

Only parcels where somatotopic sub-structure is meaningful are sub-parcellated.
Other motor parcels (SMA, pre-SMA, SCEF, PMd, PMv, S1) are used as whole-parcel
ROIs regardless of effector, because they are small enough that top-k selection
within a 15–40 voxel parcel would be noisy at 2mm resolution.

```python
# Parcels where effector-specific sub-parcellation is applied
SUB_PARCELLATE = {
    'Right_4': 7,    # Primary motor cortex — full somatotopic span
    'Left_4':  187,
    'Right_3b': 8,   # Primary somatosensory — tactile, also somatotopic
    'Left_3b': 188,
}

# Parcels used as whole units regardless of effector
WHOLE_PARCEL = {
    # SMA, pre-SMA, SCEF, S1 areas 3a/1/2, PMd, PMv  (from canonical_utils.py)
    # these feed into the ROI set for resting-state connectivity unchanged
}
```

### ROI Selection Strategy

For each subject × effector × sub-parcellate parcel:

1. Load the parcel binary mask (Glasser360 label == parcel_id, resampled to 2mm).
2. Load the effector z-map.
3. Compute `z_in_parcel = z_map[parcel_mask]`.
4. **Top-k selection** (default `k=50` voxels): take the indices of the
   top-k z-scores within the parcel. If the parcel has fewer than k voxels,
   take all of them.
5. **Z-threshold floor check**: if fewer than `min_voxels=10` voxels have
   z > `z_floor=1.5`, fall back to the whole parcel and emit a warning.
   This handles subjects with low SNR in a particular run set.
6. Write the selected voxels as a binary NIfTI mask at 2mm.

The `k=50` default is empirically motivated: area 4 in the right hemisphere
covers ~350 voxels at 2mm; 50 voxels (~14%) is a compact but stable functional
region. Make `k` a CLI argument.

---

## Script: `msc_localizer_roi_timeseries.py`

Location: `canonical_circuits/msc_localizer_roi_timeseries.py`

### CLI

```python
def parse_args():
    p = argparse.ArgumentParser(
        description='Extract resting-state timeseries using subject-specific '
                    'localizer-defined functional ROIs.')
    p.add_argument('--subjects', nargs='+',
                   default=[f'MSC{i:02d}' for i in range(1, 11)])
    p.add_argument('--sessions', nargs='+',
                   default=[f'func{i:02d}' for i in range(1, 11)])
    p.add_argument('--effectors', nargs='+',
                   default=['foot', 'hand', 'tongue'])
    p.add_argument('--top-k', type=int, default=50,
                   help='Voxels per effector × sub-parcel ROI.')
    p.add_argument('--z-floor', type=float, default=1.5,
                   help='Min z-score threshold; fallback to whole parcel if '
                        'fewer than --min-voxels pass.')
    p.add_argument('--min-voxels', type=int, default=10)
    p.add_argument('--glm-run', default='allruns',
                   help='GLM run-set to use for contrast maps '
                        '(allruns | run-01 | …).')
    p.add_argument('--confound-cols', nargs='+',
                   default=['cosine00', 'cosine01', 'cosine02', 'cosine03',
                            'csf', 'rot_x', 'rot_y', 'rot_z',
                            'trans_x', 'trans_y', 'trans_z', 'white_matter'])
    p.add_argument('--no-save-masks', action='store_true',
                   help='Skip writing NIfTI mask files (faster if masks exist).')
    return p.parse_args()
```

### Path Helpers

```python
FMRIPREP_BASE = (
    '/mfs/io/groups/dmello/projects/cerebellum_reliability/'
    'derivatives/fmriprep/ds000224'
)
GLM_BASE = (
    '/mfs/io/groups/dmello/projects/cerebellum_reliability/'
    'derivatives/firstLevel/multiTaskL1/ds000224'
)
GLASSER_IMG = os.path.join(ATLAS_DIR, 'glasser360', 'glasser360MNI.nii.gz')

def bold_path(subject, session):
    return os.path.join(
        FMRIPREP_BASE, f'sub-{subject}', f'ses-{session}', 'func',
        f'sub-{subject}_ses-{session}_task-rest'
        f'_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    )

def confounds_path(subject, session):
    return os.path.join(
        FMRIPREP_BASE, f'sub-{subject}', f'ses-{session}', 'func',
        f'sub-{subject}_ses-{session}_task-rest'
        f'_desc-confounds_timeseries.tsv'
    )

def zmap_path(subject, effector, glm_run='allruns'):
    return os.path.join(
        GLM_BASE, f'sub-{subject}', 'task-motor', glm_run,
        f'sub-{subject}_task-motor_contrast-{effector}_zmap.nii.gz'
    )

def mask_output_dir(subject):
    d = os.path.join(
        cc_analysis_dir('motor_cortex'),
        'localizer_masks', subject
    )
    os.makedirs(d, exist_ok=True)
    return d

def timeseries_output_dir(subject, session):
    d = os.path.join(
        ts_root(), subject, session,
        'localizer_functional_rois', 'rest'
    )
    os.makedirs(d, exist_ok=True)
    return d
```

### Step 1 — Build Subject-Specific ROI Masks

```python
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img, new_img_like

def build_roi_masks(subject, effectors, top_k, z_floor, min_voxels,
                    glm_run, save_masks, glasser_2mm):
    """
    Returns dict:
      roi_masks[effector][parcel_name] = NIfTI binary mask image (2mm)
    Also writes masks to mask_output_dir(subject)/ if save_masks.
    """
    # Load atlas resampled to 2mm BOLD grid (done once per subject call)
    # glasser_2mm is preloaded: resample_to_img(GLASSER_IMG, bold_ref, interpolation='nearest')

    atlas_data = np.asarray(glasser_2mm.dataobj)
    ref_affine = glasser_2mm.affine
    ref_shape  = glasser_2mm.shape

    roi_masks = {eff: {} for eff in effectors}

    for effector in effectors:
        zmap_img  = nib.load(zmap_path(subject, effector, glm_run))
        zmap_data = np.asarray(zmap_img.dataobj)

        for parcel_name, parcel_idx in SUB_PARCELLATE.items():
            parcel_mask = (atlas_data == parcel_idx)
            n_parcel    = parcel_mask.sum()

            z_in_parcel = zmap_data * parcel_mask  # zero outside parcel
            flat_z      = z_in_parcel[parcel_mask]  # 1-D

            k_actual = min(top_k, n_parcel)
            top_indices_local = np.argpartition(flat_z, -k_actual)[-k_actual:]

            # Z-floor fallback
            if (flat_z[top_indices_local] < z_floor).sum() > (k_actual - min_voxels):
                print(f'  [{subject}/{effector}/{parcel_name}] z-floor fallback '
                      f'(max z={flat_z.max():.2f}) — using whole parcel')
                roi_data = parcel_mask.astype(np.uint8)
            else:
                roi_data = np.zeros(ref_shape, dtype=np.uint8)
                parcel_coords = np.argwhere(parcel_mask)
                selected_coords = parcel_coords[top_indices_local]
                roi_data[
                    selected_coords[:, 0],
                    selected_coords[:, 1],
                    selected_coords[:, 2]
                ] = 1

            mask_img = new_img_like(glasser_2mm, roi_data)
            roi_masks[effector][parcel_name] = mask_img

            if save_masks:
                fname = (f'{subject}_effector-{effector}'
                         f'_parcel-{parcel_name.replace("/", "-")}_mask.nii.gz')
                mask_img.to_filename(
                    os.path.join(mask_output_dir(subject), fname)
                )

    return roi_masks
```

### Step 2 — Build Whole-Parcel Masks (non-sub-parcellated regions)

```python
def build_whole_parcel_masks(glasser_2mm):
    """
    Returns dict: parcel_name → NIfTI binary mask (same for all subjects).
    Covers SMA, pre-SMA, SCEF, S1 sub-areas, PMd, PMv.
    """
    from canonical_utils import (SMA_IDX, PRE_SMA_IDX, SCEF_IDX,
                                  PRIMARY_SENS_IDX, PMD_IDX, PMV_IDX,
                                  PARACENTRAL_IDX)
    from msc_chow_liu import load_glasser360_labels

    labels = load_glasser360_labels()
    atlas_data = np.asarray(glasser_2mm.dataobj)

    # Build label → index map (0-based atlas label = integer value - 1 in volume)
    # glasser360MNI.nii.gz stores labels as 1-based integers
    whole_masks = {}
    all_whole_indices = (SMA_IDX + PRE_SMA_IDX + SCEF_IDX +
                         PRIMARY_SENS_IDX + PMD_IDX + PMV_IDX + PARACENTRAL_IDX)
    for idx in all_whole_indices:
        label_val = idx + 1          # NIfTI stores 1-based
        parcel_name = labels[idx]
        mask_data = (atlas_data == label_val).astype(np.uint8)
        whole_masks[parcel_name] = new_img_like(glasser_2mm, mask_data)

    return whole_masks
```

**Note**: Confirm the exact label encoding in `glasser360MNI.nii.gz` before
running. The file may store labels as 1-based integers (label value = 0-based
index + 1) or match the `regionID` convention from `HCP-MMP1_UniqueRegionList.csv`.
Print `np.unique(atlas_data)` to verify.

### Step 3 — Extract Timeseries

For each session, apply a `NiftiMasker` to each ROI mask and extract the
mean timeseries (one value per TR = spatial average over ROI voxels).

```python
from nilearn.maskers import NiftiMasker

def extract_session_timeseries(subject, session, roi_masks, confound_cols,
                                whole_masks):
    """
    Returns dict: roi_key → np.ndarray of shape (n_TRs,)
    roi_key format: '{effector}__{parcel_name}' for localizer ROIs,
                    'whole__{parcel_name}'       for whole-parcel ROIs.
    """
    bold = bold_path(subject, session)
    if not os.path.exists(bold):
        print(f'  BOLD not found: {bold}')
        return None

    conf_df  = pd.read_csv(confounds_path(subject, session),
                           sep='\t', on_bad_lines='skip',
                           encoding='latin-1', engine='python')
    confounds = conf_df[confound_cols].values

    timeseries = {}

    # Localizer-defined ROIs (effector × parcel)
    for effector, parcel_dict in roi_masks.items():
        for parcel_name, mask_img in parcel_dict.items():
            key = f'{effector}__{parcel_name}'
            masker = NiftiMasker(
                mask_img=mask_img,
                t_r=2.0,
                detrend=True,
                standardize='zscore_sample',
                standardize_confounds='zscore_sample',
                resampling_target=None,   # mask already at 2mm
            )
            ts = masker.fit_transform(bold, confounds=confounds)  # (n_TRs, n_voxels)
            timeseries[key] = ts.mean(axis=1)                    # spatial mean → (n_TRs,)

    # Whole-parcel ROIs
    for parcel_name, mask_img in whole_masks.items():
        key = f'whole__{parcel_name}'
        masker = NiftiMasker(
            mask_img=mask_img,
            t_r=2.0,
            detrend=True,
            standardize='zscore_sample',
            standardize_confounds='zscore_sample',
            resampling_target=None,
        )
        ts = masker.fit_transform(bold, confounds=confounds)
        timeseries[key] = ts.mean(axis=1)

    return timeseries
```

### Step 4 — Save Timeseries

Output format matches the existing `msc_extract_timeseries.py` output so that
`msc_mutual_info.py` can consume it with a new `atlas_subdir` argument.

```python
def save_timeseries(timeseries_dict, subject, session):
    """
    Saves a single CSV with columns = ROI keys, rows = TRs.
    Also saves a metadata JSON listing effector → [parcel_name, ...] mapping.
    """
    import json

    out_dir = timeseries_output_dir(subject, session)

    # Stack into (n_TRs, n_ROIs) matrix
    roi_keys = sorted(timeseries_dict.keys())
    matrix   = np.column_stack([timeseries_dict[k] for k in roi_keys])

    np.savetxt(
        os.path.join(out_dir, 'rest.csv'),
        matrix,
        delimiter=',',
        header=','.join(roi_keys),
        comments=''
    )

    meta = {
        'roi_keys': roi_keys,
        'n_timepoints': matrix.shape[0],
        'n_rois': matrix.shape[1],
        'subject': subject,
        'session': session,
    }
    with open(os.path.join(out_dir, 'roi_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
```

### Main Loop

```python
def main():
    args = parse_args()

    # Preload atlas resampled to 2mm using any subject's BOLD as reference grid
    ref_bold   = bold_path(args.subjects[0], args.sessions[0])
    ref_img    = nib.load(ref_bold)
    glasser_2mm = resample_to_img(
        GLASSER_IMG, ref_img,
        interpolation='nearest', copy=True
    )

    whole_masks = build_whole_parcel_masks(glasser_2mm)

    for subject in args.subjects:
        print(f'\n=== {subject} ===')

        # Build subject-specific ROI masks once per subject
        roi_masks = build_roi_masks(
            subject, args.effectors,
            top_k=args.top_k, z_floor=args.z_floor,
            min_voxels=args.min_voxels, glm_run=args.glm_run,
            save_masks=not args.no_save_masks,
            glasser_2mm=glasser_2mm
        )

        for session in args.sessions:
            print(f'  {session}')
            ts = extract_session_timeseries(
                subject, session, roi_masks, args.confound_cols, whole_masks
            )
            if ts is not None:
                save_timeseries(ts, subject, session)
```

---

## Output Directory Layout

```
code/functional_connectivity/midnight_scan_club/
  output/
    roi_time_series/
      {subject}/{session}/localizer_functional_rois/rest/
        rest.csv                  # (n_TRs × n_ROIs); columns = roi_keys
        roi_metadata.json         # roi_keys, n_timepoints, n_rois

analysis/
  canonical_circuits/
    motor_cortex/
      localizer_masks/
        {subject}/
          {subject}_effector-foot_parcel-Right_4_mask.nii.gz
          {subject}_effector-hand_parcel-Right_4_mask.nii.gz
          {subject}_effector-tongue_parcel-Right_4_mask.nii.gz
          … (one mask per effector × sub-parcellated parcel)
```

---

## Integration with `canonical_circuits`

`mc_effector_trees.py` (from `plan.md`) needs MI matrices as input. The
new timeseries feed into `msc_mutual_info.py` via a new `atlas_subdir`:

```
msc_mutual_info.py --atlas-subdir localizer_functional_rois --task rest
```

`msc_mutual_info.py` will need a small update to:
1. Read a directory containing `rest.csv` files (not per-atlas NIfTI masking).
2. Use the `roi_metadata.json` to align the ROI ordering with the MI matrix columns.

Alternatively, write a thin adapter in `canonical_utils.py`:

```python
def load_localizer_mi(subject, session_list, top_k=50):
    """
    Load localizer-ROI timeseries across sessions, concatenate, compute MI.
    Returns (MI matrix, roi_keys list).
    """
    from msc_mutual_info import get_mutual_information, discretize_time_series
    import json

    all_ts = []
    roi_keys = None
    for session in session_list:
        d = timeseries_output_dir(subject, session)
        csv_path  = os.path.join(d, 'rest.csv')
        meta_path = os.path.join(d, 'roi_metadata.json')
        if not os.path.exists(csv_path):
            continue
        ts = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        with open(meta_path) as f:
            meta = json.load(f)
        if roi_keys is None:
            roi_keys = meta['roi_keys']
        all_ts.append(ts)

    ts_concat = np.concatenate(all_ts, axis=0)   # (total_TRs, n_ROIs)
    disc = discretize_time_series(ts_concat.T)    # (n_ROIs, total_TRs)
    mi   = get_mutual_information(disc)
    return mi, roi_keys
```

---

## Implementation Notes

1. **Label encoding in `glasser360MNI.nii.gz`**: Before running, verify with
   `np.unique(nib.load(GLASSER_IMG).get_fdata())`. The 0-based parcel index used
   in `canonical_utils.py` likely maps to integer label = index + 1 in the NIfTI.
   Confirm this against `glasser360NodeNames.txt`.

2. **Resampling order**: Resample Glasser360 atlas to 2mm once and reuse it.
   Do not resample per-subject; the atlas is the same for all subjects. The BOLD
   reference grid is the same across subjects (all fmriprep outputs are in the
   same 2mm MNI space).

3. **Motion outlier columns**: The fmriprep confounds TSV includes `motion_outlier00…`
   columns (binary spike regressors). These are not included in the default
   `confound_cols` above. For high-motion subjects, consider passing them in:
   ```python
   conf_cols_full = args.confound_cols + [c for c in conf_df.columns
                                          if c.startswith('motion_outlier')]
   ```

4. **Tongue vs. mouth ROIs**: The events file uses the label `Tongue`. The
   contrast map is `contrast-tongue_zmap.nii.gz`. In `canonical_utils.py`,
   the mouth effector includes area 55b and 6v in addition to area 4. The
   tongue contrast should show clear activation in area 4's ventral portion
   and in 55b; apply sub-parcellation to area 4 and 3b, use whole-parcel for 55b.

5. **Cross-run stability check (optional)**: The `multiTaskL1` directory also
   has per-run outputs (`run-01` through `run-20`). A validation step could
   compute the spatial overlap (Jaccard) of the top-k mask across even/odd runs
   per subject to confirm the localizer is reliable. Only flag subjects with
   Jaccard < 0.3 for manual review.

6. **No new dependency on T1w space**: This design is fully self-contained within
   MNI space. A future extension using `fmriprep`'s `from-MNI152NLin2009cAsym_to-T1w`
   transform (available in `anat/`) could project masks back to T1w space for
   surface visualization, but this is not needed for MI/CL connectivity analysis.

---

## Implementation Order

1. Verify Glasser360 label encoding (print unique values, cross-check with node names).
2. Implement `build_roi_masks()` and test on MSC01 foot/hand/tongue for area 4.
   Visualize masks with `nilearn.plotting.plot_roi` overlaid on T1w to confirm
   somatotopic location (foot = superior, hand = middle, tongue = ventral).
3. Implement `build_whole_parcel_masks()`.
4. Implement `extract_session_timeseries()` and `save_timeseries()`.
5. Run full pipeline for MSC01 all sessions; check output CSV shape and metadata.
6. Run all 10 subjects; check for fallback warnings.
7. Update `canonical_utils.py` with `load_localizer_mi()` adapter.
8. Connect to `mc_effector_trees.py` via the adapter.
