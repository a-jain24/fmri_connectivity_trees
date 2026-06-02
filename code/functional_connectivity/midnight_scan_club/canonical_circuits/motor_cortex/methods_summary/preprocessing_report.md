# Midnight Scan Club — Full Preprocessing Pipeline

**Dataset:** Midnight Scan Club (MSC); 10 subjects (MSC01–MSC10), 12 sessions each  
**Primary citation:** Gordon EM et al. (2017). Precision Functional Mapping of Individual Human Brains. *Neuron* 95, 791–807.  
**GCSS method citation:** Fedorenko E et al. (2010). New method for fMRI investigations of language: defining ROIs functionally in individual subjects. *J Neurophysiol* 104(2):1177–94. DOI: [10.1152/jn.00032.2010](https://doi.org/10.1152/jn.00032.2010)

---

## 0. Dataset Structure

Each subject underwent 12 sessions:
- **2 structural sessions** (`ses-struct01`, `ses-struct02`): T1w, T2w, angiography, venography
- **10 functional sessions** (`ses-func01`–`ses-func10`): resting-state + task fMRI

**BIDS root:** `/mfs/io/groups/dmello/projects/archived_projects/midnight_scan_club/`

### Structural acquisitions (per session)
| Sequence | Parameters |
|----------|-----------|
| T1w MPRAGE | 3T Siemens Trio; 0.8×0.8×0.8 mm; 224 slices; TR=2400 ms; TE=3.74 ms; TI=1000 ms; FA=8°; 2 runs per session |
| T2w | 3T Siemens Trio; 0.8×0.8×0.8 mm |
| MR Angiography | Dedicated angio sequence |

### Functional acquisitions (per functional session)
| Task | Volumes | TR | Slices | Runs |
|------|---------|-----|--------|------|
| Resting-state | 818 | 2.2 s | 36 | 1 |
| Motor localizer | 104 | 2.2 s | 36 | 2 |
| Glass lexical decision | variable | 2.2 s | 36 | 2 |
| Memory (faces/scenes/words) | variable | 2.2 s | 36 | 1 each |

**Acquisition parameters (all BOLD):** 3T Siemens Trio; EPI; 4×4×4 mm; FA=90°; TE=27 ms; phase encoding = A→P (j−)

### Fieldmaps (per functional session)
Dual-echo GRE fieldmap: TE1=5.19 ms, TE2=7.65 ms; TR=667 ms; FA=60°; 4×4×4 mm, 36 slices. Used by fMRIPrep for susceptibility distortion correction.

---

## 1. Quality Control — MRIQC

**Tool:** MRIQC (run on structural and functional data)  
**Outputs:** `/mfs/io/groups/dmello/projects/archived_projects/midnight_scan_club/derivatives/mriqc/`  
- `T1w.csv`, `T2w.csv`, `bold.csv` — image quality metrics per run
- `reports/` — per-subject HTML visual QC reports

MRIQC is run prior to fMRIPrep to identify outlier scans.

---

## 2. Minimal Preprocessing — fMRIPrep 23.1.4

**Tool:** fMRIPrep 23.1.4, run via Apptainer container  
**Container:** `$software_dir/fmriprep/fmriprep-23.1.4.sif`  
**Script:** `code/fmriprep/fmriprep_singlesubj.sh`  
**Output space:** `MNI152NLin2009cAsym` at 2 mm isotropic  
**Resources:** 16 threads; 95 GB RAM  
**Key flags:** `--fs-no-reconall` (FreeSurfer surface reconstruction skipped); `--skip-bids-validation`

### fMRIPrep processing steps (per BOLD run):
1. **Head motion estimation** — 6 rigid-body parameters (3 translation, 3 rotation) via MCFLIRT
2. **Slice-time correction** — using SliceTiming metadata (interleaved, TR=2.2 s, 36 slices)
3. **Susceptibility distortion correction** — GRE fieldmap (TE1/TE2) used via SDC-SyN
4. **EPI → T1w registration** — boundary-based registration (BBR)
5. **T1w → MNI152NLin2009cAsym** — ANTs nonlinear registration
6. **Resampling** — final BOLD resampled to 2 mm MNI space
7. **Brain masking** — computed per-run in MNI space
8. **Confound estimation** — framewise displacement (FD), DVARS, aCompCor (WM/CSF), tCompCor, edge components, 24-parameter head motion model (6 params + derivatives + quadratics), global signal, cosine basis functions

**Outputs per run** (in `derivatives/fmriprep/sub-MSC0X/ses-funcXX/func/`):
- `*_desc-preproc_bold.nii.gz` — minimally preprocessed BOLD (no denoising applied yet)
- `*_desc-confounds_timeseries.tsv` — ~70-column confound matrix including FD, DVARS, 24 motion params, aCompCor, tCompCor, WM/CSF signal, global signal, cosine regressors, motion outlier spikes
- `*_desc-brain_mask.nii.gz` — per-run brain mask
- `*_boldref.nii.gz` — reference volume

**Anatomical outputs** (in `derivatives/fmriprep/sub-MSC0X/anat/`):
- `*_desc-preproc_T1w.nii.gz` — skull-stripped T1w
- `*_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5` — nonlinear warp
- `*_label-{GM,WM,CSF}_probseg.nii.gz` — tissue probability maps
- `*_dseg.nii.gz` — discrete segmentation

---

## 3. Spatial Smoothing

**Tool:** AFNI 3dmerge, run via Apptainer container  
**Script:** `code/smoothing/ss_smoothing.sh`  
**Kernel:** 6.0 mm FWHM Gaussian  
**Input:** `*_desc-preproc_bold.nii.gz` from fMRIPrep output  
**Output prefix:** `s_` prepended to filename (e.g., `s_sub-MSC01_ses-func01_task-motor_run-01_...nii.gz`)

Smoothing is applied to BOLD volumes already in MNI space. Smoothed files are stored alongside the fMRIPrep outputs in the same `func/` directory.

---

## 4. Motion Outlier Identification

**Script:** `code/outlier/fmriprep_fd_outliers.py`  
**FD cutoff:** 0.5 mm  
**Output directory:** `derivatives/outliers_05/`

For each run, volumes where framewise displacement exceeds the cutoff are identified from the fMRIPrep confounds TSV. A binary spike regressor column is created per outlier volume (one column = one outlier timepoint). These spike regressors are passed to the GLM in the next step.

**Output format:** TSV files of binary spike regressors, one per run, named `*_desc-confounds_timeseries_fd.tsv`.

---

## 5. First-Level GLM (Task fMRI)

**Tool:** nilearn `FirstLevelModel`  
**Script:** `code/firstlevel/firstlevel.py`  
**Task:** Motor localizer  
**TR:** 2.2 s  
**High-pass filter:** 128 s (cosine drift basis)  
**HRF model:** SPM canonical HRF  
**Noise model:** AR(1) autocorrelation  
**Input:** Smoothed fMRIPrep BOLD (`s_*_desc-preproc_bold.nii.gz`)  
**Brain mask:** `code/firstlevel/mask20_no_eyeballs.nii` (MNI, removes eyeballs and non-brain tissue)  
**Confounds included in design matrix:**
- Cosine drift regressors (high-pass)
- FD outlier spike regressors (from step 4)

**Contrasts computed (each × all 10 sessions + combined across sessions):**
- `Tongue` — tongue movement vs. rest
- `LHand` — left hand vs. rest
- `RHand` — right hand vs. rest
- `LHand-RHand` — lateralization contrast
- `LFoot` — left foot vs. rest
- `RFoot` — right foot vs. rest
- `LFoot-RFoot` — lateralization contrast

**Output maps per contrast (in `derivatives/l1_output/sub-MSC0X/task-motor/`):**
- `*_result-tstat_cond-{contrast}_run-{XX}.nii.gz` — t-statistic map
- `*_result-beta_cond-{contrast}_run-{XX}.nii.gz` — effect size (beta) map

**Note on parallel GLM (cerebellum_reliability project):** A separate multi-task first-level GLM was run under `derivatives/firstLevel/multiTaskL1/ds000224/` (stored in `/mfs/io/groups/dmello/projects/cerebellum_reliability/`), using a broader set of contrasts and individual motor runs. This GLM produces **z-maps** (`*_contrast-{contrast}_zmap.nii.gz`) per run that are used by the GCSS parcellation pipeline (step 7 below). The run-level design matrices and masks are stored under `task-motor/run-{XX}/` and `task-motor/allruns/`.

---

## 6. Surface Pipeline (External — Provided with MSC Dataset)

These derivatives were distributed with the MSC dataset and are not re-computed:

**Location:** `derivatives/surface_pipeline/sub-MSC0X/`

| Subdirectory | Contents |
|---|---|
| `task_timecourses/ses-funcXX/` | CIFTI dense time series (`*.dtseries.nii`) and event TSV files in fs_LR_32k space |
| `processed_restingstate_timecourses/` | Denoised resting-state CIFTI time series |
| `surface_parcellation/` | Individual cortical parcellations (Gordon 2016 network parcels) |
| `fs_LR_Talairach/` | Registered cortical surfaces in fs_LR_32k space |
| `myelin_map/` | T1w/T2w myelin maps |
| `cifti_distances/` | Geodesic/Euclidean distance matrices |
| `task_contrasts_cifti/` | Task contrast maps in CIFTI space |

These CIFTI time series and contrast maps represent the original publication's outputs. The event TSVs from this pipeline serve as inputs to the local first-level GLM (step 5).

---

## 7. GCSS Motor Parcellation — Group Probability Maps and Cluster Segmentation

**Script:** `gcss_timeseries.py` (Phase 1+2, `--phase group_maps`)  
**Method:** Group-Constrained Subject-Specific (Fedorenko et al., 2010; Julian et al., 2012)  
**Input z-maps source:** `cerebellum_reliability/derivatives/firstLevel/multiTaskL1/ds000224/sub-MSC0X/task-motor/run-{XX}/`

### Contrasts and thresholds

| Contrast | z-threshold | Min group overlap | Min cluster voxels | Notes |
|---|---|---|---|---|
| `LFoot` | z > 3.7 | 60% | 15 | Contralateral right M1 foot area; motor-mask applied |
| `RFoot` | z > 3.7 | 60% | 15 | Contralateral left M1 foot area; motor-mask applied |
| `LHand` | z > 3.7 | 60% | 15 | Contralateral right M1 hand knob; motor-mask applied |
| `RHand` | z > 3.7 | 60% | 15 | Contralateral left M1/S1 hand; motor-mask applied |
| `tongue` | z > 3.7 | 60% | 20 | Bilateral lateral M1 face/tongue area; no masking |
| `combined_motor` | z > 2.3 | 60% | 20 | Avg(foot+hand+tongue); premotor network; no masking |

**Motor-cortex mask:** Binary mask derived from Glasser360 parcels corresponding to motor cortex (`MOTOR_CORTEX_ALL` list in `canonical_utils.py`). Applied to the LFoot/RFoot/LHand/RHand contrasts to suppress off-target visual cortex (VVC) activations present in 9/10 subjects.

**Runs used:** Odd motor runs only (`run-01, run-03, run-05, ...`) to leave even runs for independent test-retest validation.

### Phase 1: Group probability map
For each contrast, individual subjects' odd-run z-maps are averaged and thresholded (z > threshold). The resulting binary activation maps are summed across all 10 subjects and divided by 10 to yield a voxel-wise probability (0–1) of activation.

**Output:** `analysis/canonical_circuits/motor_cortex/gcss/group_maps/{contrast}_prob.nii.gz`

### Phase 2: Cluster segmentation
1. Threshold the group probability map at `min_overlap` (60%)
2. 26-connected-component labeling (scipy `ndimage.label` with full 3×3×3 structuring element)
3. Discard clusters smaller than `min_cluster_voxels` voxels
4. Annotate each cluster with its peak Glasser360 parcel (most-overlapping parcel) and MNI centroid

**Output:**
- `analysis/.../gcss/cluster_maps/{contrast}_clusters.nii.gz` — integer label map
- `analysis/.../gcss/cluster_maps/{contrast}_cluster_info.json` — per-cluster metadata (Glasser label, centroid, n_voxels)

### Resulting 11 ROIs (semantic naming)

| Semantic name | Source contrast | Laterality logic |
|---|---|---|
| `R_M1_foot` | LFoot | Contralateral hemisphere |
| `L_M1_foot` | RFoot | Contralateral hemisphere |
| `R_M1_hand` | LHand | Contralateral hemisphere |
| `L_M1_hand` | RHand | Contralateral hemisphere |
| `R_M1_tongue` | tongue | cx > 0 |
| `L_M1_tongue` | tongue | cx < 0 |
| `SMA` | combined_motor | \|cx\| < 12 mm |
| `L_PMd` | combined_motor | cx < −12 and cz > 50 |
| `R_PMv` | combined_motor | cx > 12 and cy > −8 and cz > 25 |
| `L_S1S2` | combined_motor | cx < 0 (other) |
| `R_S1S2` | combined_motor | cx > 0 (other) |

---

## 8. GCSS Motor Parcellation — Subject-Specific fROI Extraction

**Script:** `gcss_timeseries.py` (Phase 3, part of `--phase timeseries`)

For each subject and each group cluster:

1. Load subject's odd-run mean z-map for the contrast
2. Resample to the 2 mm atlas grid
3. Intersect subject's thresholded activation (z > 3.7 for foot/hand/tongue; z > 2.3 for premotor) with the group cluster mask
4. If fewer than `min_froi_voxels` voxels survive the primary threshold, fall back to relaxed threshold (z > 2.3); if still insufficient, use the entire group cluster (whole-parcel fallback)
5. Merge clusters sharing the same semantic ROI name (union of voxels)

**Output masks:** `analysis/.../gcss/froi_masks/MSC0X/{MSC0X}_gcss_{roi_name}_mask.nii.gz` — binary NIfTI (2 mm MNI space)

---

## 9. Resting-State Timeseries Extraction

**Script:** `gcss_timeseries.py` (Phase 4, part of `--phase timeseries`)  
**Input BOLD:** `cerebellum_reliability/derivatives/fmriprep/ds000224/sub-MSC0X/ses-funcXX/func/*_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`  
**Sessions:** `func01`–`func10` (10 resting-state sessions per subject)

### Confound removal (applied via nilearn `signal.clean`)

Confound regressors selected from fMRIPrep's confounds TSV:

| Regressor group | Columns used |
|---|---|
| WM/CSF signal | `csf`, `white_matter` |
| Head motion (12-param) | `trans_x/y/z`, `rot_x/y/z`, `trans_x/y/z_derivative1`, `rot_x/y/z_derivative1` |
| Cosine drift basis | All `cosine*` columns (high-pass filtering) |
| Motion outlier spikes | All `motion_outlier*` and `non_steady_state*` columns |

**Signal cleaning parameters:**
- `detrend=True` — linear and constant trend removal
- `standardize='zscore_sample'` — z-score each ROI timeseries
- `standardize_confounds='zscore_sample'`
- `t_r=2.2`
- Global signal regression: **not applied** (not included in confound set)

### ROI timeseries averaging
Each fROI mask is resampled to the BOLD reference grid (`interpolation='nearest'`), and the mean BOLD signal across all voxels within the mask is extracted per TR. This produces one scalar timeseries per ROI per session.

**Output per session** (`datasets/midnight_scan_club/roi_time_series/MSC0X/funcXX/gcss_motor/`):
- `rest.csv` — (n_TRs × 11 ROIs) matrix, header row = ROI names
- `roi_metadata.json` — ROI keys, n_timepoints, n_rois, subject, session, contrast provenance

---

## 10. Connectivity Computation

**Script:** `gcss_connectivity.py`

### Functional Connectivity (FC)
Pearson correlation matrix computed on the cleaned, z-scored ROI timeseries. Diagonal set to 0.  
Shape: (11 × 11)

### Mutual Information (MI)
**Tool:** Custom PyTorch implementation (`msc_mutual_info.py`)  
**Method:** Histogram-based discretization → joint probability → MI in nats

1. Discretize each ROI timeseries into equal-width bins (`num_bins=100` by default, configurable via `--num-bins`)
2. Compute joint probability histogram for each ROI pair
3. MI(X,Y) = Σ p(x,y) log[p(x,y) / (p(x)·p(y))]
4. GPU-accelerated via PyTorch (CUDA if available, else CPU)

Shape: (11 × 11)

### Chow-Liu Tree (CL)
Maximum spanning tree of the MI graph, computed via NetworkX (`maximum_spanning_tree`, Kruskal's algorithm). Yields exactly N−1 = 10 edges for 11 ROIs.

- `cl_matrix`: (11 × 11) binary adjacency × MI weight (MI value on CL edges, 0 elsewhere)

### Session-level vs. concatenated outputs

Both per-session and all-sessions-concatenated versions are computed:

| Version | Timeseries length | Path |
|---|---|---|
| Per session | ~818 TRs (single rest run) | `connectivity/{fc,mi,cl}/MSC0X/funcXX/gcss_motor/` |
| All sessions | ~8180 TRs (10 sessions concatenated) | `connectivity/{fc,mi,cl}/MSC0X/all_sessions/gcss_motor/` |

**Files per output** (e.g. `MSC01_all_sessions_gcss_{fc,mi,cl}.npy`):
- `*.npy` — (11 × 11) float32 matrix

---

## 11. Group-Level Analysis

**Script:** `gcss_group_figures.py`

Per-subject all-sessions FC, MI, and CL matrices are loaded and analyzed:

- **Edge frequency matrix** — for each of the N(N−1)/2=55 ROI pairs, count how many of the 10 subjects include that edge in their CL tree (max=10)
- **Canonical edges:** frequency ≥ 7/10
- **Intermediate edges:** frequency 4–6/10
- **Idiosyncratic edges:** frequency 1–3/10
- **Mean MI matrix** — average MI across subjects
- **Mean FC matrix** — average Pearson correlation across subjects

---

## Summary: Data Flow

```
Raw BIDS (sub-MSC0X/)
    │
    ├─ [MRIQC]  Quality metrics
    │
    ├─ [fMRIPrep 23.1.4]  Motion correction, SDC, T1→MNI warp, confound estimation
    │       └─ derivatives/fmriprep/  *_desc-preproc_bold.nii.gz + confounds TSV
    │
    ├─ [AFNI 3dmerge, 6mm FWHM]  Spatial smoothing
    │       └─ s_*_desc-preproc_bold.nii.gz  (motor task runs)
    │
    ├─ [FD outliers, cutoff=0.5mm]  Motion scrubbing regressors
    │       └─ derivatives/outliers_05/  *_fd.tsv
    │
    ├─ [nilearn FirstLevelModel, SPM HRF, AR(1)]  Task GLM
    │       └─ derivatives/l1_output/  *_result-{tstat,beta}_*.nii.gz
    │
    ├─ [MultiTaskL1 GLM — cerebellum_reliability]  Per-run z-maps
    │       └─ firstLevel/multiTaskL1/  *_contrast-{X}_zmap.nii.gz
    │
    ├─ [GCSS Phase 1+2]  Group probability maps → cluster segmentation
    │       └─ analysis/.../gcss/{group_maps,cluster_maps}/
    │
    ├─ [GCSS Phase 3]  Subject-specific fROI masks (11 ROIs per subject)
    │       └─ analysis/.../gcss/froi_masks/MSC0X/
    │
    ├─ [GCSS Phase 4, nilearn signal.clean]  Resting-state timeseries extraction
    │       └─ datasets/.../roi_time_series/MSC0X/funcXX/gcss_motor/rest.csv
    │
    └─ [gcss_connectivity.py]  FC (Pearson), MI (histogram, 100 bins), CL tree (MST)
            └─ datasets/.../connectivity/{fc,mi,cl}/MSC0X/{per-session,all_sessions}/
```
