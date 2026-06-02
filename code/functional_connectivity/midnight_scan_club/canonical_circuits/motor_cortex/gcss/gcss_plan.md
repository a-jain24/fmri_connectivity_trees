# Group-Constrained Subject-Specific (GCSS) Motor Parcellation Plan

## Background: Why Individual-Specific Parcellation?

Standard group-level atlases (Yeo 2011, Schaefer 2018, Glasser 2016) assign every subject the same parcel boundaries despite well-documented individual variability in the spatial topography of functional networks. Individual-specific parcellation methods attempt to personalize boundaries while remaining constrained by group-level knowledge, yielding:

- Higher **functional homogeneity** within parcels
- Better **behavioral prediction** from functional connectivity
- More accurate **task-fMRI** activation containment
- Improved **test-retest reproducibility** of individual parcels

The four methods below span a design space from task-fMRI watershed localizers (GCSS) to probabilistic generative models (MS-HBM), Bayesian MRF optimization (GPIP), and fully data-driven graph methods (GWC).

---

## Method 1: GCSS — Group-Constrained Subject-Specific

> **Not a resting-state method** — GCSS operates on **task-fMRI localizer contrasts**, defining functional ROIs in individual subjects within group-level spatial constraints.

### Citation
- **Fedorenko E, Hsieh P-J, Nieto-Castañón A, Whitfield-Gabrieli S, Kanwisher N** (2010). New method for fMRI investigations of language: defining ROIs functionally in individual subjects. *J Neurophysiology* 104(2):1177–94. DOI: [10.1152/jn.00032.2010](https://doi.org/10.1152/jn.00032.2010)
- **Julian JB, Fedorenko E, Webster J, Kanwisher N** (2012). An algorithmic method for functionally defining regions of interest in the ventral visual pathway. *NeuroImage* 60(4):2357–64. DOI: [10.1016/j.neuroimage.2012.02.055](https://doi.org/10.1016/j.neuroimage.2012.02.055)

### Core Algorithm
GCSS is a **three-step watershed + intersection procedure** on task activation maps:

1. **Group probabilistic map:** Individual thresholded activation maps (p < 0.0001 uncorrected, or p < 0.05 FDR) are overlaid in MNI space to produce a voxel-wise overlap probability map.
2. **Parcel segmentation:** A watershed algorithm segments the group map into functional parcels following its topographic peaks. Parcels present in ≥ 80% of subjects (Fedorenko 2010) or ≥ 60% (Julian 2012) are retained as canonical group-level spatial boundaries.
3. **Subject-specific fROI selection:** Each individual's thresholded activation map is intersected with the group parcels; only suprathreshold voxels within a given parcel boundary constitute that subject's fROI.

**"Group-constrained"** = boundaries derived from population overlap maps; **"Subject-specific"** = peaks selected per individual within those constraints.

### Initialization
No atlas prior — the group probability map is derived from the localizer contrast itself (e.g., sentences > nonwords for language; faces/objects/places > scrambled for visual regions).

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Individual activation threshold | p < 0.0001 uncorrected (or p < 0.05 FDR) |
| Min group overlap to retain parcel | ≥ 80% (language); ≥ 60% (ventral visual) |
| Watershed algorithm | Standard topographic segmentation |

### Datasets
- **Fedorenko 2010:** N = 25, Siemens 3T Trio; 3 experiments (N = 12, 13, 12)
- **Julian 2012:** N = 40 for parcel derivation; multiple validation cohorts

### Performance Metrics
- **Within-subject correlation (odd vs. even runs):** r = 0.52 for language fROIs
- **Between-subject correlation:** r = 0.10 (confirming individual specificity)
- **Coverage:** 13 language-sensitive ROIs (Fedorenko 2010); 14 ventral visual ROIs (Julian 2012)
- ≥ 89% of voxels fell within the largest contiguous cluster (effectively contiguous in practice)

### Generalization to Task-fMRI
By construction — GCSS is designed for task-fMRI. Individual language fROIs show higher functional selectivity (e.g., N400-like responses) than group-level masks. The approach has been extended to face/body/scene/object visual regions, theory of mind, music cognition, and mental number line networks.

### Code & Data
- **SPM toolbox + pre-derived parcels:** https://web.mit.edu/evlab/funcloc/
- **EvLab localizer paradigms:** http://evlab.mit.edu
- Language and ventral visual parcels available for download at the Kanwisher Lab GSS page

## Overview

**GCSS Approach:** Start from a group-level motor map (pooled across all subjects/runs), use it as a spatial constraint, then refine individual motor parcels within that constraint using each subject's own motor task activation. This captures individual motor somatotopy and broader motor network organization while maintaining cross-subject correspondence.

**Current Limitation (Localizer Approach):**
- ROIs forced into Glasser360 parcel boundaries (areas 4, 3b, etc.)
- Only captures top-50 voxels per effector
- Misses broader motor network (SMA, pre-SMA, PMd, PMv not individually optimized)
- No subject-specific boundary refinement

**GCSS Advantage:**
- Individual motor areas unconstrained by atlas boundaries
- Captures full motor network including premotor and supplementary motor areas
- Each subject's parcels are anatomically matched across subjects
- Flexible ROI sizes based on individual activation strength
- Better reflects individual motor organization while preserving group structure

**Extended GCSS (this plan):** Beyond the three canonical effectors (foot/hand/tongue), GCSS is applied to five additional localizer contrasts — lateralized effectors, combined motor > rest, differential effector contrasts, cerebellar space (SUIT), and subcortical space (putamen/thalamus). This extends the motor network from 3 ROIs to 20–30 individually-optimized fROIs suitable for non-trivial CL tree analysis.

**Key Papers & References:**
- Gordon et al. (2017) "Precision Functional Mapping" — describes individual parcellation via boundary identification
- Laumann et al. (2015) "On the Stability of BOLD fMRI Correlations" — group constraints for individual parcellation
- Braga & Buckner (2017) "Parallel Interacting Networks" — individual-specific cortical organization
- Xu et al. (2020) "Striatal Functional Topography" — GCSS for subcortical structures

---

## Critical Note on Z-Map Source

> **All GCSS phases that load individual z-maps must use averaged odd-run z-maps, not the `allruns` GLM.**

The `allruns` GLM normalizes each effector regressor against the mean across all motor conditions, producing differential rather than activation-vs-baseline z-scores. In M1 foot representation (MNI 0, −24, +66) this produces z = −6.38 in the `allruns` map vs. z = +4.15 in individual odd runs. Using `allruns` would select the *least-deactivated* voxels rather than the most-activated ones.

**Fix (already implemented in `msc_localizer_roi_timeseries.py`):** Use `load_mean_zmap(subject, effector, motor_runs)` with `motor_runs = [run-01, run-03, ..., run-19]` (odd runs). Reserve even runs (run-02, run-04, …, run-20) for split-half test-retest validation of fROI stability.

See `motor_cortex/analysis_plan.md` Bug 1 for full documentation.

---

---

## Implementation Plan for MSC Motor Cortex

### Phase 1: Build Group Probabilistic Activation Map

#### Step 1.1: Threshold Individual Motor Activation Maps

For each subject and motor run, threshold the GLM z-map for each contrast.

**Available contrasts in the MSC motor GLM design matrix:**
- Bilateral effectors: `foot` (LFoot + RFoot > rest), `hand` (LHand + RHand > rest), `tongue` (> rest)
- Lateralized effectors (if individual-limb z-maps exist): `lfoot`, `rfoot`, `lhand`, `rhand`
- Combined motor: `motor` (foot + hand + tongue > rest, computed by averaging z-maps)
- Differential: `foot_vs_hand` (foot − hand z-maps), `hand_vs_tongue`, `foot_vs_tongue`

**Input:**
- 10 subjects × 10 odd motor runs × N contrasts = 100 × N individual z-maps
- Path: `/derivatives/firstLevel/multiTaskL1/ds000224/sub-{subject}/task-motor/run-{run}/sub-{subject}_task-motor_contrast-{effector}_zmap.nii.gz`
- **Use only odd runs** (run-01, run-03, …, run-19) per Bug 1 fix; reserve even runs for test-retest validation.

**Threshold:** p < 0.0001 uncorrected (equivalent to z > 3.7 for two-tailed)
- Following Fedorenko 2010 (language); motor cortex may need adjustment to p < 0.05 FDR if activation is sparse
- Subcortical contrasts (putamen, thalamus) may require relaxed threshold: p < 0.01 (z > 2.3)

```python
import nibabel as nib
import numpy as np

subjects = [f'MSC{i:02d}' for i in range(1, 11)]
ODD_RUNS = [f'run-{i:02d}' for i in range(1, 21, 2)]   # run-01,03,...,19
EVEN_RUNS = [f'run-{i:02d}' for i in range(2, 21, 2)]  # reserved for test-retest


def load_mean_zmap(subject, effector, motor_runs):
    """Average z-maps across specified runs. Avoids allruns GLM inversion (Bug 1)."""
    arrays, ref_img = [], None
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
    return nib.Nifti1Image(np.mean(arrays, axis=0), ref_img.affine, ref_img.header)


# Bilateral effectors — used for somatotopic GCSS (Sections A and B below)
BILATERAL_EFFECTORS = ['foot', 'hand', 'tongue']

# Threshold maps (z > 3.7 ≈ p < 0.0001)
z_threshold = 3.7

individual_thresholded_maps = {}
for effector in BILATERAL_EFFECTORS:
    thresholded_maps = []
    for subject in subjects:
        zmap_img = load_mean_zmap(subject, effector, ODD_RUNS)
        if zmap_img is None:
            continue
        zmap_data = zmap_img.get_fdata()
        thresholded = (zmap_data > z_threshold).astype(np.float32)
        thresholded_maps.append((subject, thresholded))
    individual_thresholded_maps[effector] = thresholded_maps
```

#### Step 1.2: Compute Group Probabilistic Overlap Map

For each contrast, overlay all thresholded maps to compute voxel-wise activation probability.

```python
def compute_group_probability_map(thresholded_maps, affine, shape):
    """
    Overlay all thresholded maps; return voxel-wise probability of activation.
    """
    overlap_count = np.zeros(shape, dtype=np.float32)
    for subject, thresholded_map in thresholded_maps:
        overlap_count += thresholded_map
    return overlap_count / len(thresholded_maps)  # probability in [0, 1]


for effector in BILATERAL_EFFECTORS:
    group_prob = compute_group_probability_map(
        individual_thresholded_maps[effector],
        affine=ref_affine,
        shape=ref_shape
    )
    group_prob_img = nib.Nifti1Image(group_prob, affine=ref_affine)
    nib.save(group_prob_img, f'group_motor_probability_{effector}.nii.gz')
```

**Expected output:** One probability map per contrast, where each voxel = fraction of subjects showing suprathreshold activation.

---

### Phase 2: Watershed Segmentation of Group Maps

#### Step 2.1: Identify Activation Peaks (Local Maxima)

Using the group probability map, identify connected clusters (candidate fROIs).

```python
from scipy.ndimage import label

def find_activation_peaks(group_prob_map, min_overlap_fraction=0.6):
    """
    Identify connected clusters in group probability map.
    Return label map where each cluster is a candidate parcel.
    """
    binary_mask = (group_prob_map > min_overlap_fraction).astype(np.int32)
    labeled_array, num_clusters = label(binary_mask)
    return labeled_array, num_clusters


for effector in BILATERAL_EFFECTORS:
    group_prob = nib.load(f'group_motor_probability_{effector}.nii.gz').get_fdata()
    labeled_clusters, n_clusters = find_activation_peaks(group_prob, min_overlap_fraction=0.6)
    print(f"{effector}: {n_clusters} clusters identified")
    cluster_img = nib.Nifti1Image(labeled_clusters.astype(np.int16), affine=ref_affine)
    nib.save(cluster_img, f'group_motor_clusters_{effector}.nii.gz')
```

**Output:** For each contrast, a labeled map where each unique integer = one candidate ROI.

#### Step 2.2: Watershed Segmentation (Optional Refinement)

For more precise boundary identification, apply watershed segmentation to the inverted probability map.

> **Note:** Use `skimage.segmentation.watershed`, not `scipy.ndimage` — scipy does not expose a public watershed function.

```python
from skimage.segmentation import watershed
from scipy.ndimage import maximum_filter, label

def watershed_segmentation(group_prob_map):
    """
    Apply watershed to identify functional boundaries between activation peaks.
    """
    inverted_prob = 1.0 - group_prob_map

    # Identify local maxima as watershed seeds
    local_max = maximum_filter(group_prob_map, size=5) == group_prob_map
    peak_labels, _ = label(local_max)

    # Watershed: assigns each voxel to the basin of the nearest peak
    segmented = watershed(inverted_prob, markers=peak_labels)
    return segmented


for effector in BILATERAL_EFFECTORS:
    group_prob = nib.load(f'group_motor_probability_{effector}.nii.gz').get_fdata()
    segmented = watershed_segmentation(group_prob)
    seg_img = nib.Nifti1Image(segmented.astype(np.int16), affine=ref_affine)
    nib.save(seg_img, f'group_motor_watershed_{effector}.nii.gz')
```

**Note:** Watershed is optional; simple connected-component labeling (Step 2.1) is often sufficient for motor cortex.

---

### Phase 3: Subject-Specific fROI Selection (Intersection)

#### Step 3.1: For Each Subject, Intersect with Group Parcels

For each subject and each group parcel:
1. Load subject's thresholded activation map (same odd-run average as Step 1.1)
2. Intersect with group parcel boundary
3. Extract only suprathreshold voxels that fall within parcel
4. Save as subject-specific fROI mask

```python
def extract_subject_specific_froi(subject, effector, watershed_path):
    """
    Extract subject-specific fROI by intersecting:
    - Subject's thresholded activation map (p < 0.0001, odd runs averaged)
    - Group parcel boundary (from watershed)
    """
    group_parcels = nib.load(watershed_path).get_fdata()

    zmap_img = load_mean_zmap(subject, effector, ODD_RUNS)
    if zmap_img is None:
        return None, None

    subject_zmap = zmap_img.get_fdata()
    subject_thresholded = (subject_zmap > 3.7).astype(np.uint8)

    # Intersection: subject activation AND within group parcel
    froi_mask = subject_thresholded * (group_parcels > 0)
    return froi_mask, subject_zmap


for subject in subjects:
    for effector in BILATERAL_EFFECTORS:
        froi_mask, zmap = extract_subject_specific_froi(
            subject, effector,
            f'group_motor_watershed_{effector}.nii.gz'
        )
        if froi_mask is None:
            continue
        mask_img = nib.Nifti1Image(froi_mask, affine=ref_affine)
        nib.save(mask_img, f'gcss_parcels/{subject}_{effector}_froi_mask.nii.gz')
        print(f"{subject}/{effector}: {int(np.sum(froi_mask))} voxels")
```

#### Step 3.2: Handle Subjects with Small or Missing fROIs

If a subject's fROI for a contrast is too small (< 20 voxels), options:
1. **Relaxed threshold:** Reduce individual threshold to p < 0.01 (z > 2.3) for that subject/contrast
2. **Whole-parcel fallback:** Use the full group parcel as the fROI
3. **Exclude:** Mark as missing and exclude from resting-state analysis

```python
for subject in subjects:
    for effector in BILATERAL_EFFECTORS:
        froi_mask = np.load(f'gcss_parcels/{subject}_{effector}_froi_mask.npy')
        n_voxels = int(np.sum(froi_mask))
        if n_voxels < 20:
            print(f"WARNING: {subject}/{effector} has only {n_voxels} voxels")
            # Decide: relax threshold or use fallback
```

---

### Phase 3B: Lateralized Effector GCSS

**Goal:** Capture hemisphere-specific individual variability in M1 hand and foot representations. Subjects differ in degree of lateralization, which bilateral contrasts mask.

**Contrasts:** `lhand`, `rhand`, `lfoot`, `rfoot` — the MSC motor GLM design matrix contains LHand, RHand, LFoot, RFoot, and Tongue as separate regressors. Check whether individual z-maps for these contrasts exist:

```python
LATERALIZED_EFFECTORS = ['lfoot', 'rfoot', 'lhand', 'rhand']
for effector in LATERALIZED_EFFECTORS:
    zp = zmap_path('MSC01', effector, 'run-01')
    if os.path.exists(zp):
        print(f"  {effector} z-maps available")
    else:
        print(f"  {effector} z-maps not found — may need to re-run GLM with per-limb contrasts")
```

**Expected anatomy:** RHand activates left M1 (contralateral); LHand activates right M1. Lateralized GCSS would correctly attribute subject-specific variability in each hemisphere rather than averaging them away.

**Implementation:** Identical to Phase 1–3 with `LATERALIZED_EFFECTORS` replacing `BILATERAL_EFFECTORS`. If per-limb z-maps are not in the derivatives, skip this section and note it as a future GLM re-analysis.

---

### Phase 3C: Premotor and Supplementary Motor GCSS

**Goal:** SMA, pre-SMA, PMd, and PMv are currently fixed whole-parcel ROIs identical across all subjects (see `canonical_utils.py: WHOLE_PARCEL_IDX`). Individual variability in premotor activation is well-documented. GCSS applied to these areas using a combined motor > rest contrast provides subject-specific boundaries without requiring new effector conditions.

**Localizer contrast:** Combine the three bilateral effector z-maps into a single motor > rest map by averaging voxel-wise. This activates M1, SMA, pre-SMA, PMd, PMv reliably across all subjects and runs.

```python
def build_combined_motor_zmap(subject, motor_runs):
    """Average foot + hand + tongue z-maps to produce a combined motor > rest map."""
    all_arrays = []
    ref_img = None
    for effector in BILATERAL_EFFECTORS:
        img = load_mean_zmap(subject, effector, motor_runs)
        if img is None:
            continue
        if ref_img is None:
            ref_img = img
        all_arrays.append(np.squeeze(np.asarray(img.dataobj, dtype=np.float32)))
    if not all_arrays:
        return None
    combined = np.mean(all_arrays, axis=0)
    return nib.Nifti1Image(combined, ref_img.affine, ref_img.header)
```

**Group probability map:** Threshold each subject's combined z-map at z > 2.3 (p < 0.01, more lenient than M1 to capture premotor regions that activate less strongly). Build group overlap map. Expect reliable (≥ 60%) clusters in bilateral SMA (6mp), pre-SMA (6ma), PMd (6d, 6r, 6a), and PMv (6v).

**Intersection:** For each discovered premotor cluster, extract each subject's suprathreshold voxels within the group parcel. Subjects with low premotor activation fall back to the whole-parcel.

**Premotor Glasser indices (from `canonical_utils.py`):**
- SMA (6mp): indices 54, 234
- pre-SMA (6ma): indices 43, 223
- SCEF: indices 42, 222
- PMd (6d, 6r, 6a): indices 53, 233, 77, 257, 95, 275
- PMv (6v): indices 55, 235

Use these to restrict watershed to premotor voxels and avoid M1 parcels already covered by Phase 3.

---

### Phase 3D: Differential Contrast GCSS

**Goal:** Within M1 and premotor cortex, some regions prefer one effector over others but are not captured by a simple activation threshold (e.g., the hand knob at the omega-shaped gyrus responds to hand but also to foot). Differential contrasts sharpen subject-specific ROI definitions within anatomically ambiguous regions.

**Contrasts to derive:**
| Contrast | Purpose |
|----------|---------|
| foot − hand | Identifies foot-preferring voxels in paracentral M1, SMA |
| hand − foot | Identifies hand-preferring voxels in M1 hand knob, PMd |
| hand − tongue | Identifies hand-preferring PMv vs. tongue-preferring 55b |
| tongue − hand | Isolates tongue/speech motor representation (area 55b, PMv) |

```python
def compute_differential_zmap(subject, effector_a, effector_b, motor_runs):
    """Subtract effector_b z-map from effector_a to create a differential contrast."""
    img_a = load_mean_zmap(subject, effector_a, motor_runs)
    img_b = load_mean_zmap(subject, effector_b, motor_runs)
    if img_a is None or img_b is None:
        return None
    data_diff = (np.squeeze(np.asarray(img_a.dataobj, dtype=np.float32)) -
                 np.squeeze(np.asarray(img_b.dataobj, dtype=np.float32)))
    return nib.Nifti1Image(data_diff, img_a.affine, img_a.header)
```

**Implementation:** Run Phase 1–3 with differential z-maps in place of raw effector z-maps. Threshold at z > 2.3 (p < 0.01) since differential contrasts have lower SNR. The group probability map identifies regions that are consistently effector-preferring across subjects; intersection gives subject-specific preference maps.

**When to use:** Primarily useful for disambiguating overlapping effector representations in PMv (hand vs. tongue), pre-SMA (foot vs. hand), and SCEF.

---

### Phase 3E: Cerebellar GCSS (SUIT Atlas)

**Goal:** The cerebellum has well-documented somatotopic organization — foot in lobule V, hand in lobule VI (anterior), face/tongue in lobule VI (posterior) / VII. This somatotopy can be individually localized using the same bilateral effector contrasts applied to cerebellar voxels using the SUIT atlas already in the pipeline.

**Implementation:** Identical to Phase 1–3 but restricting all operations to the SUIT cerebellar mask. Motor-relevant SUIT indices (from `canonical_utils.py`):

```python
CEREB_MOTOR_LOCAL = [0, 1, 5, 6, 20, 21]  # lobules IV/V, VI, VIII
CEREB_OFFSET = 360
CEREB_MOTOR_IDX = [CEREB_OFFSET + i for i in CEREB_MOTOR_LOCAL]
```

**Threshold:** Relax to z > 2.3 (p < 0.01) for cerebellar z-maps; cerebellar activation is reliably present but at lower z than cortex in MNI-space GLMs.

**Expected output:** For each bilateral effector (foot/hand/tongue), one GCSS fROI per subject in cerebellar space. Combined with cortical GCSS fROIs, this adds 3–6 cerebellar ROIs per subject to the connectivity analysis.

**Validation:** Known cerebellar somatotopy predicts foot fROIs should be in lobule V (superior) and hand/tongue in lobule VI. Verify centroid MNI z-coordinates follow this gradient.

---

### Phase 3F: Subcortical GCSS (Putamen and Thalamus)

**Goal:** The putamen shows a dorsal-to-ventral somatotopic gradient (foot dorsal, tongue ventral) and the motor thalamus (VLa/VLp nuclei in the Morel atlas) relays motor signals. Both activate during the MSC motor task at lower thresholds.

**Implementation:**
- Use combined motor > rest z-maps (from Phase 3C) at z > 2.3 threshold
- Restrict watershed to putamen and thalamic motor nuclei masks
- Putamen can be masked using the MNI152 Harvard-Oxford subcortical atlas or a binary putamen ROI
- Thalamus uses the Morel atlas already referenced in `canonical_utils.py` (Thalamus atlas offset)

**Caveat:** Spatial resolution (2 mm) limits fROI granularity in subcortical structures. Fallback to whole-structure ROI is more likely here than in cortex.

---

### Phase 4: Resting-State Timeseries Extraction on GCSS fROIs

Extract timeseries using the subject-specific fROI masks from all GCSS phases.

```python
def extract_gcss_timeseries(subject, session, froi_masks_by_contrast):
    """
    Extract resting-state timeseries for all GCSS motor fROIs.

    froi_masks_by_contrast: dict keyed by contrast name (e.g. 'foot', 'hand',
    'tongue', 'combined_motor', 'foot_vs_hand', 'cereb_foot', etc.),
    each value a binary NIfTI mask.
    """
    bold_path = f'/derivatives/fmriprep/.../sub-{subject}_ses-{session}_task-rest_bold.nii.gz'
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()

    confounds_path = f'/derivatives/fmriprep/.../sub-{subject}_ses-{session}_confounds.tsv'
    confounds_df = pd.read_csv(confounds_path, sep='\t')

    cosine_cols = [c for c in confounds_df.columns if c.startswith('cosine')]
    base_cols = ['csf', 'white_matter',
                 'trans_x', 'trans_y', 'trans_z',
                 'rot_x', 'rot_y', 'rot_z',
                 'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                 'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1']
    outlier_cols = [c for c in confounds_df.columns
                    if c.startswith('motion_outlier') or c.startswith('non_steady_state')]
    cols = cosine_cols + base_cols + outlier_cols
    confounds = confounds_df[cols].fillna(0.0).values

    timeseries_dict = {}
    for contrast_name, mask_img in froi_masks_by_contrast.items():
        mask_r = resample_to_img(mask_img, bold_img.slicer[..., 0],
                                 interpolation='nearest', force_resample=False)
        mask_bool = np.asarray(mask_r.dataobj, dtype=bool)
        if mask_bool.sum() == 0:
            continue
        timeseries_dict[contrast_name] = bold_data[mask_bool].mean(axis=0)

    roi_keys = sorted(timeseries_dict.keys())
    ts_matrix = np.column_stack([timeseries_dict[k] for k in roi_keys])
    clean_matrix = nilearn.signal.clean(
        ts_matrix, confounds=confounds, detrend=True,
        standardize='zscore_sample', t_r=2.2
    )

    output_dir = f'datasets/midnight_scan_club/roi_time_series/{subject}/{session}/gcss_frois/'
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(f'{output_dir}/rest.csv', clean_matrix, delimiter=',',
               header=','.join(roi_keys), comments='')
    return clean_matrix, roi_keys
```

**Expected ROI count per subject (all GCSS phases combined):**

| GCSS Phase | Contrasts | Expected fROIs |
|-----------|-----------|---------------|
| 3 — Lateralized somatotopic (cortical) | LFoot, RFoot, LHand, RHand, tongue | 6 (contralateral M1 per effector/hemisphere) |
| 3C — Premotor (combined motor) | combined motor > rest | 4–6 (SMA, PMd, PMv, S2) |
| 3E — Cerebellar (optional) | foot, hand, tongue in SUIT | 3 (if SUIT atlas available) |
| 3F — Subcortical (optional) | combined motor (putamen/thalamus) | 2–4 |
| **Total (core)** | | **10–12 subject-specific fROIs** |

---

### Phase 5: Connectivity Analysis on GCSS fROIs

Compute FC, MI, CL trees using resting-state timeseries from all GCSS fROIs.

```python
def compute_gcss_connectivity(subject, sessions=['func01', ..., 'func10'],
                              gcss_variant='full'):
    """
    Compute FC, MI, CL for GCSS motor fROIs.
    gcss_variant: 'somatotopic' (3 ROIs), 'cortical' (cortical phases only),
                  'full' (all phases, 21-30 ROIs)
    """
    ts_list = []
    roi_keys = None
    for session in sessions:
        ts_path = f'datasets/midnight_scan_club/roi_time_series/{subject}/{session}/gcss_frois/rest.csv'
        if not os.path.exists(ts_path):
            continue
        ts = np.loadtxt(ts_path, delimiter=',', skiprows=1)
        if roi_keys is None:
            with open(ts_path.replace('rest.csv', 'roi_metadata.json')) as f:
                roi_keys = json.load(f)['roi_keys']
        ts_list.append(ts)

    ts_concat = np.concatenate(ts_list, axis=0)  # (n_TRs, n_ROIs)

    fc = np.corrcoef(ts_concat.T)
    np.save(f'analysis/canonical_circuits/motor_cortex/gcss/{subject}_gcss_fc.npy', fc)

    mi = pairwise_roi_mutual_information(ts_concat, num_bins=100)
    np.save(f'analysis/canonical_circuits/motor_cortex/gcss/{subject}_gcss_mi.npy', mi)

    cl_tree = chow_liu_tree(mi)
    cl_adj = nx.to_numpy_array(cl_tree)
    np.save(f'analysis/canonical_circuits/motor_cortex/gcss/{subject}_gcss_cl_adj.npy', cl_adj)

    return fc, mi, cl_adj, roi_keys
```

**Network size and CL tree complexity by variant:**

| Variant | ROIs | CL edges | Notes |
|---------|------|----------|-------|
| Somatotopic only (Phase 3) | 3 | 2 | Tests motor somatotopy preservation |
| Cortical (Phases 3 + 3B + 3C + 3D) | ~16 | 15 | Comparable to localizer 12-ROI network |
| Full (all phases) | ~25 | 24 | Rich enough for meaningful tree analysis |

**Recommended starting point:** Run the somatotopic variant first as a sanity check (known structure), then scale to the cortical variant for the main individual-differences analysis.

---

## GCSS vs. Localizer Comparison

| Aspect | Localizer | GCSS Somatotopic | GCSS Full |
|--------|-----------|-----------------|-----------|
| **ROI Definition** | Top-50 voxels in Glasser areas 4/3b | Intersection of subject activation with group parcel | Same, applied to 6 contrast types |
| **Spatial Constraint** | Glasser360 atlas boundaries | Group probability map (60–80% overlap) | Same, per contrast |
| **Atlas Prior** | Yes (Glasser360) | No (derives from task activation) | No |
| **Individual Variability** | Within Glasser parcel bounds | Unrestricted, but group-constrained | Unrestricted |
| **ROI Count** | 38 (12 localizer + 26 whole) | 3 | 21–30 |
| **Method** | Top-k selection | Watershed + intersection | Watershed + intersection |
| **CL tree edges** | 37 | 2 | 20–29 |
| **Citation** | Gordon et al. 2017 | Fedorenko 2010; Julian 2012 | Same |

---

## Implementation Timeline

1. **Phase 1–2:** Threshold individual maps, build group probability maps, watershed (~2 hours)
2. **Phase 3:** Extract bilateral effector fROIs (~2 hours)
3. **Phase 3B:** Check lateralized z-map availability; extract lateralized fROIs if present (~1 hour)
4. **Phase 3C:** Build combined motor > rest map; extract premotor fROIs (~2 hours)
5. **Phase 3D:** Compute differential contrast maps; extract differential fROIs (~2 hours)
6. **Phase 3E:** Cerebellar GCSS using SUIT mask (~1 hour)
7. **Phase 3F:** Subcortical GCSS (~1 hour)
8. **Phase 4:** Extract resting-state timeseries for all fROIs (~4 hours; parallelize across subjects)
9. **Phase 5:** Compute FC, MI, CL trees (~1 hour)

**Total:** ~2 days (parallelizable across subjects in Phases 3–4)

---

## Expected Outcomes

**Advantages over localizer:**
- **No atlas bias:** ROIs derived purely from task activation, not forced into Glasser boundaries
- **Broader motor network:** Premotor, cerebellar, and subcortical ROIs individually optimized per subject
- **High individual specificity:** Different subjects can have differently-sized or positioned fROIs for the same contrast
- **Cross-subject correspondence maintained:** All subjects' fROIs defined within same group-level boundaries
- **Richer connectivity analysis:** 21–30 ROI network supports non-trivial CL tree topologies

**Potential caveats:**
- **Small ROI sizes in premotor/cerebellar areas:** fROIs may fall back to whole parcel for some subjects at z > 3.7; relax threshold to z > 2.3 for Phase 3C–3F
- **Lateralized z-maps may not exist:** Phase 3B depends on whether individual-limb GLM contrasts were saved in the MSC derivatives
- **Differential contrasts have lower SNR:** Phase 3D fROIs will be smaller; use z > 2.3 threshold and a lower min-voxel cutoff (min 10 voxels)

---

## Validation

1. **Reproduce group map:** Does group probability map overlap with known motor cortex (areas 4, 3a, 3b)?
2. **Split-half test-retest:** Use odd runs (run-01, 03, …) to define fROIs; use even runs (run-02, 04, …) to define an independent fROI set. Correlate overlap — matches Fedorenko 2010's within-subject r = 0.52 benchmark. This is the correct design given the odd-run constraint from Bug 1.
3. **Somatotopic validation:** Do foot/hand/tongue cortical fROIs follow expected superior-to-inferior gradient in MNI z-coordinate?
4. **Cerebellar somatotopy:** Do cerebellar foot fROIs fall in lobule V (superior) and hand/tongue in lobule VI, consistent with known cerebellar somatotopy?
5. **Resting-state test:** Do GCSS fROIs show stronger within-effector FC than cross-effector FC?
6. **Comparison with localizer:** Compare FC and CL tree structure between GCSS fROIs and the existing localizer-defined 38-ROI network. GCSS premotor ROIs should show different (less atlas-constrained) connectivity profiles.

---

**Status:** Group maps computed. See "Group Map Audit" section below for correctness assessment.
**Date:** 2026-05-05 (audit 2026-05-12)
**Key References:**
- Fedorenko et al. 2010 (DOI: 10.1152/jn.00032.2010)
- Julian et al. 2012 (DOI: 10.1016/j.neuroimage.2012.02.055)
- `motor_cortex/analysis_plan.md` — Bug 1 (allruns inversion), Bug 2 (frame censoring), Bug 3 (cosine regressors)

---

## Group Map Audit (2026-05-12)

Group maps were computed from all 10 MSC subjects using odd-run z-maps. The table below summarises group-overlap statistics and cluster correctness against known motor somatotopy (Penfield & Rasmussen 1950; Yeo 2011; Glasser 2016).

### Expected somatotopy (MNI)
| Effector | Region | Approximate MNI |
|----------|--------|----------------|
| Foot/leg | Paracentral M1 (area 4 medial) | ±5, −25, +68 |
| Hand | "Hand knob" M1 (area 4 lateral) | ±38, −22, +58 |
| Face/tongue | Lateral M1 (area 4 face) | ±57, −7, +28 |
| Premotor (all) | SMA (6mp), pre-SMA (6ma), PMd (6d), PMv (6v) | z ≥ 50 |

### Cluster correctness per contrast

| Contrast | Max prob | Vox ≥60% | Clusters | Verdict |
|----------|----------|---------|----------|---------|
| `LFoot` | 0.90 | 184 | Right_4 paracentral (5.8, −25, 69.7) | **Correct** |
| `RFoot` | 0.80 | 188 | Left_4 paracentral (−5.8, −22.8, 68.9) | **Correct** |
| `LHand` | 0.90 | 1712→516* | Right_4 hand knob (39.5, −20.9, 57.7) | **Correct after motor mask** |
| `RHand` | 0.90 | 1410→445* | Left_3b hand (−39.4, −23.3, 57.4) | **Correct after motor mask** |
| `tongue` | 1.00 | 2296 | Bilateral Left_4/Right_4 lateral (±57, −7, +28); bilateral FOP2 | **Correct** |
| `combined_motor` | 0.90 | 1332 | Left_6d (PMd), Right_6v (PMv), bilateral PFop (S2), Left_SCEF† (SMA) | **Correct** |

\* Voxel count after motor cortex mask removes task-visual VVC cluster (9/10 subjects, max prob=0.90).  
† "Left_SCEF" centroid at (−1.8, −4.6, 61.7) is anatomically in SMA/pre-SMA; Glasser SCEF parcel overlaps 6mp at the medial wall.

Bilateral `foot`/`hand` and all differential contrasts (`foot_vs_hand`, `hand_vs_tongue`, `foot_vs_tongue`) were evaluated and dropped: bilateral effectors are superceded by the cleaner lateralized contrasts, and differential contrasts produce fully redundant clusters (2–6 mm from Tier 1 centroids) at lower SNR.

### Key findings

1. **Lateralized effectors are the most reliable somatotopic localizers.** `LFoot`/`RFoot` give perfectly clean contralateral paracentral M1 clusters (all 184/188 voxels in the correct region). `LHand`/`RHand` give strong contralateral M1/S1 hand clusters but require a motor cortex mask to remove a task-visual VVC artifact.

2. **Visual cortex contamination in hand lateralized contrasts.** The MSC motor task used visual cues; the hand-movement z-maps contain a highly consistent (9/10 subjects, max prob=0.90) VVC cluster at MNI ≈ ±17, −50, −20. Applied fix: Glasser `MOTOR_CORTEX_ALL` mask in `segment_group_map()` before connected-component labeling. After masking: LHand→516 vox (clean Right_4), RHand→445 vox (clean Left_3b).

3. **`combined_motor` is correct but Glasser labeling is ambiguous.** The large cluster at (−1.8, −4.6, 61.7) labeled "Left_SCEF" is anatomically in the SMA/pre-SMA region (Glasser SCEF parcel boundary overlaps 6mp at the medial wall).

4. **Bilateral effectors and differential contrasts were evaluated and dropped.** Bilateral `foot`/`hand` are superceded by the cleaner lateralized contrasts. Differential contrasts (`foot_vs_hand`, `hand_vs_tongue`, `foot_vs_tongue`) produce clusters within 2–6 mm of Tier 1 centroids — fully redundant at lower SNR.

### Fix applied (2026-05-12): Motor cortex mask in `segment_group_map()`

`segment_group_map()` now accepts a `motor_mask` parameter. When provided, non-motor voxels are zeroed before thresholding. The mask is built from `MOTOR_CORTEX_ALL` Glasser parcel indices (30 parcels, 9840 voxels at 2 mm) and applied to `LFoot`, `RFoot`, `LHand`, `RHand` (set `MOTOR_MASKED_CONTRASTS` in `gcss_timeseries.py`).

| Contrast | Before masking | After masking |
|----------|---------------|---------------|
| `LHand` | Right_4 (1338v) + Left_VVC (364v, 9/10 subjects) | **Right_4 (516v)** — clean |
| `RHand` | Left_3b (1113v) + Right_VVC (257v) | **Left_3b (445v)** — clean |

**Confirmed somatotopic gradient:**
```
Foot M1 (z ≈ 69)  >  Hand M1 (z ≈ 58)  >  Tongue M1 (z ≈ 29)   [Penfield homunculus ✓]
```

---

### ROI naming scheme

ROIs are named by **semantic anatomical region** rather than by Glasser parcel label. Each ROI name encodes hemisphere (L/R) and structure (`M1_foot`, `M1_hand`, `M1_tongue`, `PMd`, `PMv`, `S1S2`, `SMA`). This makes figures and tree edges self-explanatory without reference to the atlas.

**Merge rule:** All Glasser-labeled clusters from the same contrast that are in the same hemisphere are merged into a single ROI (union of voxels). For example, the `tongue` contrast produces clusters in Left_4, Left_FOP2, Left_PoI2, and an unlabeled peri-insular region — all are merged into `L_M1_tongue`. This yields exactly **11 ROIs** regardless of how many Glasser sub-clusters are present.

**Routing for `combined_motor` clusters** (centroid-based):

| Condition | ROI name |
|-----------|----------|
| `|cx| < 12` | `SMA` |
| `cx < −12` and `cz > 50` | `L_PMd` |
| `cx > 12` and `cy > −8` and `cz > 25` | `R_PMv` |
| `cx < 0` (else) | `L_S1S2` |
| `cx > 0` (else) | `R_S1S2` |

**Mask filename convention:** `{subject}_gcss_{sem_name}_mask.nii.gz`  
(e.g., `MSC01_gcss_L_M1_tongue_mask.nii.gz`, `MSC01_gcss_SMA_mask.nii.gz`)

---

### Recommended ROI scheme for CL tree analysis

The following **11-ROI scheme** uses only the 6 Tier 1 contrasts. It yields 10 CL tree edges with fully interpretable anatomy.

| ROI label | Source contrast | Glasser clusters merged | Centroid MNI (group) | n_vox (group) |
|-----------|----------------|------------------------|----------------------|---------------|
| `R_M1_foot` | `LFoot` | Right_4 | (5.8, −25.0, 69.7) | 184 |
| `L_M1_foot` | `RFoot` | Left_4 | (−5.8, −22.8, 68.9) | 188 |
| `R_M1_hand` | `LHand` | Right_4 | (39.5, −20.9, 57.7) | 516 |
| `L_M1_hand` | `RHand` | Left_3b | (−39.4, −23.3, 57.4) | 445 |
| `L_M1_tongue` | `tongue` | Left_4, Left_FOP2, Left_PoI2, cluster_04 | (−57.5, −8.7, 30.0) | 1148+ |
| `R_M1_tongue` | `tongue` | Right_4, Right_FOP2 | (57.6, −5.1, 28.6) | 995+ |
| `SMA` | `combined_motor` | Left_SCEF/6mp† | (−1.8, −4.6, 61.7) | 428 |
| `L_PMd` | `combined_motor` | Left_6d | (−45.4, −8.8, 58.8) | 30 |
| `R_PMv` | `combined_motor` | Right_6v | (57.1, 1.5, 35.1) | 173 |
| `L_S1S2` | `combined_motor` | Left_PFop, Left_PF | (−58.8, −11.5, 32.4) | 466+ |
| `R_S1S2` | `combined_motor` | Right_PFop | (58.0, −13.9, 27.1) | 99 |

† Centroid at (−1.8, −4.6, 61.7) is anatomically SMA/pre-SMA; Glasser's SCEF parcel overlaps 6mp at the medial wall.

**Why lateralized instead of bilateral for foot/hand?** Bilateral contrasts in the MSC GLM are effector-relative (vs. mean motor) rather than vs. rest (Bug 1), suppressing M1. Lateralized contrasts are individual-limb vs. baseline, giving peak group probabilities 0.80–0.90 and clean contralateral M1 clusters.

**Tongue bilateral** is kept because tongue M1 is inherently bilateral (midline organ, max prob = 1.0).

**Confirmed somatotopic gradient (Penfield homunculus):**
```
L/R foot M1   z ≈ 69   (paracentral lobule)
L/R hand M1   z ≈ 58   (hand knob / omega-shaped gyrus)
L/R tongue M1 z ≈ 29   (lateral face M1)
```

---

### ROI Info

#### R_M1_foot / L_M1_foot — Primary motor cortex, foot/leg representation
**Glasser area:** 4 (Right / Left)  
**Centroid MNI:** R (5.8, −25.0, 69.7) · L (−5.8, −22.8, 68.9)  
**Macro-anatomy:** Posterior bank of the central sulcus, paracentral lobule (medial surface of the hemisphere near the vertex). The foot representation sits at the top of the homunculus, medial to the hand area, extending onto the mesial cortical surface.  
**Cytoarchitecture:** Brodmann area 4 (agranular cortex); characterized by giant Betz cells in layer V that project directly to spinal cord interneurons and motor neurons via the lateral corticospinal tract. The foot representation has particularly large Betz cells.  
**Motor function:** Primary motor control of the contralateral foot, ankle, and lower leg. Source of fast-conducting (~70 m/s) corticospinal fibers for distal leg movements. Bilateral foot tasks activate SMA more strongly than either contralateral M1 in isolation, but single-foot tasks show a clean contralateral preference (confirmed: LFoot → Right_4; RFoot → Left_4 in MSC data).  
**Connectivity:** Strongly coupled to SMA (movement preparation), ipsilateral PMd (postural control), and spinal cord (direct projection). Interhemispheric M1-foot coupling via corpus callosum is relatively weak compared to M1-hand, reflecting independent (rather than mirrored) leg movements. Expect moderate FC with `SMA` and weak FC with `R_M1_hand`/`L_M1_hand` in the CL tree.  
**Individual variability:** Low — foot M1 is the most spatially consistent effector representation in the MSC data (group overlap 0.80–0.90; all voxels in the correct paracentral region). Explained by the foot area's position at the crown of the central sulcus, which is a stable gyral landmark.

---

#### R_M1_hand — Primary motor cortex, right-hemisphere hand representation
**Glasser area:** 4 (Right)  
**Centroid MNI:** (39.5, −20.9, 57.7)  
**Macro-anatomy:** Posterior bank of the central sulcus, lateral surface of the precentral gyrus. The hand representation occupies the "hand knob" — an omega- or epsilon-shaped folding of the precentral gyrus visible on axial MRI, typically at MNI z ≈ 55–65. This landmark is identifiable in individual subjects and is one of the most reliably localizable cortical areas.  
**Cytoarchitecture:** Area 4, with dense Betz cells in layer V. The hand representation is disproportionately large (~35% of total M1 surface area) relative to hand size, reflecting the precision demands of finger movements.  
**Motor function:** Primary motor control of the contralateral (left) hand and individual fingers. Direct corticospinal projections to cervical spinal cord motor neurons. Represents the most complex motor territory in M1, with fine-grained digit-level somatotopy (index finger most lateral, thumb most superior).  
**Connectivity:** Strong FC with left M1-hand (interhemispheric), ipsilateral PMd (movement selection), and PMv (grasping/reach). Weaker FC with foot M1 (somatotopic distance). In the CL tree, expected to cluster with `L_M1_hand` and `L_PMd`.  
**Source note:** Defined by the `LHand` contrast (left hand → right M1). The pre-masking cluster contained 1338 motor voxels + 364 VVC voxels; post-masking retains 516 clean M1 voxels.

---

#### L_M1_hand — Primary somatosensory cortex, left-hemisphere hand representation
**Glasser area:** 3b (Left)  
**Centroid MNI:** (−39.4, −23.3, 57.4)  
**Macro-anatomy:** Anterior bank of the postcentral gyrus (posterior to the central sulcus), at the hand knob level. Area 3b is the thalamo-recipient zone of primary somatosensory cortex receiving cutaneous input from the contralateral hand (VPLc thalamic nucleus).  
**Cytoarchitecture:** Area 3b (granular cortex); dense layer IV with thalamocortical terminations. Distinct from M1 (area 4) which is directly anterior across the central sulcus. Area 3b has no Betz cells and receives rather than sends motor commands, but is routinely coactivated with M1 during voluntary movements via proprioceptive feedback and efference copy.  
**Motor function:** Not a motor output area per se. Area 3b carries fine-touch and texture discrimination signals from the contralateral hand that are essential for sensorimotor control (grip force modulation, object manipulation). Its coactivation during hand movements reflects the dense reciprocal connections between M1 and S1 through which somatosensory feedback shapes ongoing motor commands.  
**Note on labeling asymmetry:** `R_M1_hand` centroid falls in area 4 while `L_M1_hand` centroid falls in area 3b. Both are in the same anatomical zone (hand knob sensorimotor area); the centroid shift reflects that the 445-voxel cluster spans both 4 and 3b, with 3b slightly dominating due to the cluster's precise center of mass in this subject group. The functional interpretation is the same: left-hemisphere sensorimotor hand area (S1+M1 combined).  
**Connectivity:** Strong FC with R_M1_hand (interhemispheric), L_PMd (motor planning), thalamus. In the CL tree, expected to mirror R_M1_hand connectivity.

---

#### L_M1_tongue / R_M1_tongue — Primary motor cortex, face/tongue representation
**Glasser area:** 4 (Left / Right)  
**Centroid MNI:** L (−57.5, −8.7, 30.0) · R (57.6, −5.1, 28.6)  
**Macro-anatomy:** Lateral surface of the precentral gyrus, just superior to the lateral (Sylvian) fissure. The face/tongue area of M1 is the most inferior and lateral part of the homunculus, transitioning into premotor and prefrontal cortex (areas 44/45, Broca's area) immediately below. The tongue area is bounded superiorly by the lip representation and inferiorly by the larynx/pharynx representation.  
**Cytoarchitecture:** Area 4, with moderately-sized Betz cells (smaller than in the foot representation). Transitions to area 6v (PMv) anteriorly and area 3b posteriorly.  
**Motor function:** Primary motor control of tongue, lip, and jaw muscles via the corticobulbar tract (projecting to cranial nerve nuclei VII, X, XII rather than spinal cord). The tongue representation in M1 is disproportionately large given tongue mass — comparable in surface area to the entire lower limb — reflecting the high precision demands of speech and swallowing. Bilateral activation during tongue tasks is expected because (a) the tongue is a midline structure and (b) corticobulbar projections are inherently bilateral, with each hemisphere projecting to both ipsilateral and contralateral facial/hypoglossal nuclei.  
**Connectivity:** Strong bilateral FC (L↔R) is expected and is anatomically grounded — unique among the somatotopic ROIs, which otherwise show contralateral dominance. Strong FC with adjacent premotor areas (PMv, FOP2) for speech motor coordination. In the CL tree, L_M1_tongue and R_M1_tongue are expected to be directly connected or near-connected, forming the bottom of the homunculus arm of the tree.  
**Group consistency:** Highest of all contrasts (max prob = 1.0; all 10 MSC subjects active at z > 3.7). The tongue motor cortex is particularly reliable because the MSC motor task required active tongue movements and the tongue representation in lateral M1 is a stable cortical landmark with low inter-subject variability.

---

#### SMA — Supplementary motor area / pre-SMA
**Glasser area:** SCEF boundary / 6mp (Left, medial wall)  
**Centroid MNI:** (−1.8, −4.6, 61.7)  
**Macro-anatomy:** Medial surface of the superior frontal gyrus, anterior to the paracentral lobule. Spans the SMA–pre-SMA boundary: SMA proper (6mp) lies posterior to the vertical plane through the anterior commissure (VCA line, approximately y = 0), while pre-SMA (6ma) lies anterior to it. The centroid at y = −4.6 places this cluster near the SMA proper / pre-SMA transition.  
**Glasser labeling note:** The cluster is labeled "Left_SCEF" because the Glasser SCEF parcel (supplementary and cingulate eye fields) boundary overlaps with 6mp/6ma on the medial wall at this MNI coordinate. The true SCEF is slightly more anterior (y > 0); this cluster is functionally SMA/pre-SMA.  
**Cytoarchitecture:** Area 6 (agranular premotor cortex), medial wall. SMA proper has direct corticospinal projections and dense callosal connections to contralateral SMA. Pre-SMA lacks direct spinal projections and connects more strongly to prefrontal and cingulate cortex.  
**Motor function:** SMA is the primary integrative hub of the voluntary motor system. It is activated by ALL motor conditions (foot, hand, tongue) — hence its presence in `combined_motor`. Key roles include: (1) internally-driven movement initiation (as opposed to externally-cued movements driven more by PMd/PMv), (2) bimanual and bilateral movement coordination, (3) movement sequence planning and timing, (4) inhibitory control (pre-SMA projects to subthalamic nucleus via hyperdirect pathway). Lesions produce akinesia (SMA) or apraxia (pre-SMA).  
**Connectivity:** The expected hub node of the CL tree. Strong FC with all M1 somatotopic ROIs (the SMA projects to M1 foot, hand, and face areas via U-fibers) and with PMd/PMv. Callosal connections to contralateral SMA. In the CL tree, `SMA` is likely to be the root or a high-degree node, serving as a convergence point between the foot-hand-tongue somatotopic branches.

---

#### L_PMd — Dorsal premotor cortex, left hemisphere
**Glasser area:** 6d (Left)  
**Centroid MNI:** (−45.4, −8.8, 58.8)  
**Macro-anatomy:** Dorsal portion of the left precentral gyrus, anterior to the M1 hand area. Located on the superolateral convexity of the frontal lobe, approximately at the junction of the precentral and superior frontal gyri.  
**Cytoarchitecture:** Area 6d (agranular premotor cortex, dorsal subdivision). Lacks the layer IV granularity of sensory cortex and the Betz cells of M1. Has direct corticospinal projections (though fewer than M1) and dense connections to M1 via short U-fibers.  
**Motor function:** Movement preparation and selection, particularly for visually-guided reaching and complex sequences. PMd is engaged before movement onset when a target is identified and a movement plan is formed. It receives inputs from posterior parietal cortex (area 5, SPL) for spatial motor planning, and from DLPFC (area 46) for conditional rule-based action selection ("if signal X, make movement Y"). PMd activity predicts which movement will be executed before M1 activity begins.  
**Left-hemisphere note:** Only left PMd survives the 60% group-overlap threshold. This likely reflects left-hemisphere dominance for complex sequential motor planning (consistent with the literature showing greater left PMd engagement for learned motor sequences in right-handed subjects). Right PMd may be present but slightly less consistent across MSC subjects in this combined-motor contrast.  
**Connectivity:** Strong ipsilateral FC with L_M1_hand (motor execution) and SMA (sequence timing). Weaker FC with tongue/foot M1 (effector specificity). In CL tree, expected to connect L_M1_hand → L_PMd → SMA.

---

#### R_PMv — Ventral premotor cortex, right hemisphere
**Glasser area:** 6v (Right)  
**Centroid MNI:** (57.1, 1.5, 35.1)  
**Macro-anatomy:** Inferior precentral gyrus, ventral portion, immediately superior to the lateral sulcus on the right hemisphere. Adjacent to the right M1 tongue/face area posteriorly and to the right IFG (pars opercularis, area 44) anteriorly.  
**Cytoarchitecture:** Area 6v (agranular premotor, ventral). In macaques the homologous area (F5) contains canonical neurons (responsive to object properties) and mirror neurons (active during both action execution and observation). Human PMv is functionally analogous but with expanded language-related connectivity compared to non-human primates.  
**Motor function:** Grasping, object manipulation, and mouth-hand coordination. PMv is especially active during reach-to-grasp movements and tool use, where it integrates visual object properties with motor commands for hand shaping. Also plays a role in orofacial motor control (chewing, speech articulation) given proximity to the tongue M1 area and frontal operculum.  
**Right-hemisphere note:** Right PMv is found while the corresponding left cluster is not clearly resolved in `combined_motor`. This may reflect the overlap of left PMv with the large L_S1S2 cluster (which spans the lateral sensorimotor strip), or genuine right-lateralization of visuomotor premotor activity in some MSC subjects.  
**Connectivity:** Strong FC with R_M1_hand/tongue region, R_S1S2, and the frontal operculum (FOP2, which appears in the `tongue` contrast). In the CL tree, R_PMv is expected to connect to R_M1_tongue or R_S1S2 rather than to the contralateral PMd.

---

#### L_S1S2 / R_S1S2 — Secondary somatosensory cortex / parietal operculum
**Glasser area:** PFop (Left / Right)  
**Centroid MNI:** L (−58.8, −11.5, 32.4) · R (58.0, −13.9, 27.1)  
**Macro-anatomy:** Parietal operculum, covering the upper bank of the lateral (Sylvian) fissure, posterior to the central sulcus. Corresponds to the secondary somatosensory cortex (S2/SII, also termed OP1–OP4 in the operculo-parietal cytoarchitectonic scheme). Directly below and posterior to the M1/S1 hand and face representations. The left cluster is substantially larger (466 vs. 99 vox), likely because the combined motor contrast activates a larger extent of the left lateral sensorimotor strip.  
**Cytoarchitecture:** PFop (inferior parietal, opercular part) is granular cortex with dense layer IV, distinct from the agranular premotor areas anterior to the central sulcus. Receives input from VPLc and VPM thalamic nuclei (proprioceptive and tactile channels), from S1 (areas 3b/1/2), and from the posterior insular cortex.  
**Motor function:** Secondary somatosensory processing and sensorimotor integration. S2 integrates somatosensory signals from both body sides (unlike S1, which is predominantly contralateral), making it important for bilateral hand coordination and for learning object properties through touch. During voluntary movement, S2 processes proprioceptive feedback (from muscle spindles and joint receptors) that is used to update ongoing motor commands and detect errors. S2 is also a gateway to the insular cortex, which encodes pain and interoceptive signals relevant to motor effort.  
**Connectivity:** Strong FC with M1 (all body parts — S2 receives proprioceptive feedback from the entire body), with SMA (for error monitoring), and with posterior parietal cortex. The bilateral presence of S1S2 ROIs (L and R) reflects that proprioceptive feedback during unilateral movements activates S2 bilaterally (unlike S1 which is strictly contralateral). In the CL tree, L_S1S2 and R_S1S2 are expected to be peripherally placed (leaves or near-leaves), connected to the M1 somatotopic nodes on their respective sides.
