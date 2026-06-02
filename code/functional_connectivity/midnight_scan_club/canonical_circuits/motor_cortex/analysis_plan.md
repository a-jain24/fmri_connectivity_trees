# Canonical Circuits — Analysis Plan

New updates:

Cross-referencing the Gordon et al. (2017) *Precision Functional Mapping of
Individual Human Brains* paper (the primary MSC methods source) against the
current implementation revealed the following bugs and issues.

---

### Bug 1 — `allruns` z-maps are inverted in primary motor cortex (CRITICAL)

**Finding:** The script uses `firstLevel/multiTaskL1/allruns` z-maps for
effector-specific voxel selection within areas 4 and 3b.  The `allruns` GLM
combines all 20 motor task runs across 10 sessions.  Inspection of the bilateral
`foot` z-map shows **z = −6.38** at the expected primary motor cortex foot
representation (paracentral lobule, MNI 0, −24, +66).  Individual motor run
z-maps show the expected **positive** activation at the same location (run-01:
z = +5.90).

Across all 20 runs the z-values at MNI 0,−24,+66 alternate systematically:

| Runs | M1-foot z (mean) |
|------|-----------------|
| Odd (01,03,…,19) | +4.15 |
| Even (02,04,…,20) | −0.09 |
| `allruns` | −6.38 |

Despite having identical design matrices (LFoot, LHand, RFoot, RHand, tongue,
scrub/drift), the `allruns` GLM appears to normalize the foot regressor against
the mean across all motor conditions, producing differential rather than
activation-vs-rest z-maps.  As a result, the top-50 voxels selected within
area 4 are the *least-deactivated* voxels in the parcel — not the most
foot-specifically activated voxels.

**Fix:** Replace `allruns` with a mean z-map averaged across odd-numbered runs
(run-01, run-03, …, run-19), which consistently show correct foot > baseline
responses in M1.  Update `zmap_path` in `msc_localizer_roi_timeseries.py` to
accept a `motor_runs` list and average those z-maps at load time.

---

### Bug 2 — Frame censoring confounds not included in resting-state preprocessing (CRITICAL)

**Finding:** Gordon et al. (2017) applied motion censoring (scrubbing) to all
resting-state fMRI data — a key preprocessing step documented as essential for
reliability.  The fMRIPrep confounds TSV provides two sets of binary indicator
vectors for this purpose:

- `motion_outlier00` – `motion_outliern`: one column per flagged frame (64
  columns in MSC01 func01, 7.8 % of frames flagged; range 23–114 per session).
- `non_steady_state_outlier00` – `non_steady_state_outlier02`: flags for the
  first 3 TRs before magnetization reaches steady state.

Neither set appears in `DEFAULT_CONFOUND_COLS`.  Regressing these out is
equivalent to scrubbing: the flagged frames are projected out of the signal
without physical removal, preserving contiguous TR counts for subsequent MI and
FC estimation.  Omitting them leaves motion-contaminated frames in the
timeseries, artificially inflating short-range FC and biasing MI estimates.

**Fix:** Dynamically append all `motion_outlier*` and `non_steady_state_outlier*`
columns from the confounds file to the confound matrix.  These are the column
names used by fMRIPrep for scrubbing; their presence/count varies by session.

```python
extra = [c for c in confounds_df.columns
         if c.startswith('motion_outlier') or c.startswith('non_steady_state')]
confound_cols = DEFAULT_CONFOUND_COLS + extra
```

---

### Bug 3 — Incomplete high-pass filter regressors (SIGNIFICANT)

**Finding:** fMRIPrep generates 27 DCT-basis cosine regressors
(`cosine00`–`cosine26`) per session to implement a ~0.0075 Hz high-pass
filter.  The current `DEFAULT_CONFOUND_COLS` includes only `cosine00`–`cosine03`,
providing an effective cutoff of ~0.001 Hz — far below the standard 0.01 Hz
threshold used in the MSC preprocessing and in most resting-state fMRI
literature.  Frequencies in the 0.001–0.0075 Hz band (slow scanner drift,
physiological noise at very low frequencies) are not removed, contaminating the
MI and FC matrices.

**Fix:** Replace the four hard-coded cosine regressors with a dynamic selection
of all cosine regressors present in the confounds file:

```python
cosine_cols = [c for c in confounds_df.columns if c.startswith('cosine')]
```

---

### Issue 4 — TR = 2.0 s in preprocessing call, actual TR = 2.2 s (MINOR)

**Finding:** The BOLD header confirms `pixdim[4] = 2.2 s` (TR = 2.2 s) for all
MSC resting-state runs.  `extract_session_timeseries` calls
`nilearn.signal.clean(..., t_r=2.0)`.  The `t_r` argument is currently unused
(no frequency-based filtering is applied), so this does not affect existing
outputs.  However, it is incorrect metadata and will cause silent errors if
`high_pass` or `low_pass` filtering is ever added.

**Fix:** Set `t_r=2.2` in all `nilearn.signal.clean` calls.

---

### Issue 5 — Motion parameter derivatives not included (MINOR)

**Finding:** The Gordon et al. (2017) supplementary methods describe a 12-
or 24-parameter motion model (6 parameters + 6 first derivatives, optionally
with quadratics).  The fMRIPrep confounds file includes
`trans_{x,y,z}_derivative1` and `rot_{x,y,z}_derivative1`.  The current
`DEFAULT_CONFOUND_COLS` uses only the 6 basic motion parameters, omitting
derivatives.  This leaves residual motion signal in the timeseries that
co-varies with neural activity, particularly for subjects with abrupt head
movements.

**Fix:** Extend `DEFAULT_CONFOUND_COLS` to include the 6 derivative columns:

```python
DEFAULT_CONFOUND_COLS = [
    'cosine00', ...,                   # replaced dynamically per Bug 3 fix
    'csf', 'white_matter',
    'trans_x', 'trans_y', 'trans_z',
    'rot_x',   'rot_y',   'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1',   'rot_y_derivative1',   'rot_z_derivative1',
]
```

---


*Last updated: 2026-04-29*

---

## Fixes (4/29)

All five issues above were implemented in `msc_localizer_roi_timeseries.py`.
The other two scripts (`mc_localizer_fc_cl.py`, `mc_localizer_brain_viz.py`) received
the path-setup fix only.  All 100 sessions need to be re-extracted with the new script.

### Path setup — `motor_cortex/` reorganization

All three scripts had a two-level `os.path.dirname` chain that previously resolved
to `midnight_scan_club/` (when files lived directly in `canonical_circuits/`).
After the move to `canonical_circuits/motor_cortex/` the chain stopped one level
short, so `from msc_paths import …` and other `msc_*` imports broke silently.

**Fix (all three scripts):**
```python
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # motor_cortex/
_CANONICAL_DIR = os.path.dirname(_SCRIPT_DIR)                  # canonical_circuits/
_MSC_DIR       = os.path.dirname(_CANONICAL_DIR)               # midnight_scan_club/
sys.path.insert(0, _MSC_DIR)        # msc_paths, msc_mutual_info, msc_chow_liu
sys.path.insert(0, _CANONICAL_DIR)  # canonical_utils
```

### Bug 1 — Averaged motor-run z-maps (replaces `allruns`)

Added `load_mean_zmap(subject, effector, motor_runs)` that loads each run's z-map,
averages voxel-wise, and returns a single NIfTI.  `build_roi_masks()` now takes
`motor_runs: list` instead of `glm_run: str`.  CLI default:

```python
_DEFAULT_MOTOR_RUNS = [f'run-{i:02d}' for i in range(1, 21, 2)]  # run-01,03,...,19
```

Old `--glm-run allruns` argument replaced by `--motor-runs` (accepts any list of run IDs).

### Bug 2 — Frame censoring (non-steady-state outliers added)

`motion_outlier*` columns were already appended; `non_steady_state_outlier*`
columns (first 3 TRs per session) were not.  Fixed in the outlier-column selection:

```python
outlier_cols = [c for c in conf_df.columns
                if c.startswith('motion_outlier')
                or c.startswith('non_steady_state')]
```

### Bug 3 — All cosine regressors

`cosine00`–`cosine03` removed from `DEFAULT_CONFOUND_COLS`.  Replaced with dynamic
selection of all cosine columns in `extract_session_timeseries()`:

```python
cosine_cols = [c for c in conf_df.columns if c.startswith('cosine')]
```

### Issue 4 — TR corrected

`t_r=2.0` → `t_r=2.2` in `nilearn.signal.clean`.

### Issue 5 — Motion derivatives added

`DEFAULT_CONFOUND_COLS` extended with the six first-derivative columns:
`trans_{x,y,z}_derivative1`, `rot_{x,y,z}_derivative1`.

---

## Overview

This module performs subject-specific functional connectivity (FC) and
Chow-Liu (CL) tree analysis on motor cortex ROIs defined by each subject's
own motor-task activation. The pipeline builds on the exploratory analysis in
`jupyter_notebooks/midnight_scan_club/msc_fc_cl_motor.ipynb` and produces
production-quality outputs for all 10 MSC subjects.

---

## Directory Layout

```
canonical_circuits/
├── canonical_utils.py              ← shared ROI indices, path helpers, MI adapters
└── motor_cortex/                   ← this module (reorganized 2026-04-29)
    ├── analysis_plan.md            ← this file
    ├── msc_localizer_roi_timeseries.py ← Step 1: extract subject-specific ROI timeseries
    ├── mc_localizer_fc_cl.py           ← Step 2: FC + MI + CL tree; 5 figures
    └── mc_localizer_brain_viz.py       ← Step 3: brain-space visualizations
```

Output roots (relative to `midnight_scan_club/`):
```
analysis/canonical_circuits/motor_cortex/   ← .npy matrices, .json metadata
figures/canonical_circuits/motor_cortex/    ← .pdf figures
datasets/midnight_scan_club/roi_time_series/{sub}/{ses}/localizer_functional_rois/rest/
                                            ← rest.csv (TRs × 38 ROIs), roi_metadata.json
```

---

## Step 1 — Subject-Specific Localizer ROI Timeseries

**Script:** `msc_localizer_roi_timeseries.py`

**What it does:**

For each subject × session:
1. Loads the subject's motor-task GLM z-maps (foot / hand / tongue bilateral
   contrasts) from the fmriprep derivatives at
   `/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/`.
   Averages z-maps across odd-numbered motor runs (run-01, 03, …, 19) to obtain
   correct activation-vs-rest z-scores (the `allruns` GLM produces inverted scores
   due to differential normalization — see Bug 1 in "New updates:").
2. Resamples the Glasser360 atlas (1 mm) to the BOLD grid (2 mm, 97×115×97).
3. For areas 4 and 3b (both hemispheres), selects the top-50 voxels by effector
   z-score within each parcel → 4 effector-specific sub-parcel masks × 3 effectors
   = 12 localizer ROI masks per subject.
4. Uses the remaining 26 motor/premotor parcels (SMA, pre-SMA, SCEF, PMd, PMv,
   S1 areas 1/2/3a, paracentral) as whole-parcel ROIs identical across subjects.
5. Loads the BOLD volume once per session, applies all 38 masks in memory, then
   runs `nilearn.signal.clean` (detrend + confound regression + z-score).
6. Saves `rest.csv` (TRs × 38 ROIs) and `roi_metadata.json` per session.

**ROI naming convention:**
- `'{effector}__{parcel}'` — localizer-defined sub-parcel (e.g. `foot__Right_4`)
- `'whole__{parcel}'` — whole Glasser360 parcel (e.g. `whole__Right_6mp`)

**Parameters:** top-k=50, z-floor=1.5, min-voxels=10

**Status (as of 2026-04-29): All 100 sessions complete (re-extracted with bug fixes).**

| Subject | Sessions complete |
|---------|-------------------|
| MSC01   | 10 / 10           |
| MSC02   | 10 / 10           |
| MSC03   | 10 / 10           |
| MSC04   | 10 / 10           |
| MSC05   | 10 / 10           |
| MSC06   | 10 / 10           |
| MSC07   | 10 / 10           |
| MSC08   | 10 / 10           |
| MSC09   | 10 / 10           |
| MSC10   | 10 / 10           |

**Mask outputs** (saved per subject in
`analysis/canonical_circuits/motor_cortex/localizer_masks/{subject}/`):

```
{subject}_effector-{foot|hand|tongue}_parcel-{Left|Right}_{4|3b}_mask.nii.gz
```
12 NIfTI binary masks per subject, on the 2 mm BOLD grid
(97×115×97, MNI152NLin2009cAsym).

---

## Step 2 — FC + MI + CL Tree Analysis

**Script:** `mc_localizer_fc_cl.py`

**What it does:**

For each subject, loads and concatenates timeseries across all available
sessions, then:

1. **FC matrix** — Pearson correlation across the 38 ROIs (diagonal = 0).
2. **MI matrix** — Pairwise mutual information using the same
   `pairwise_roi_mutual_information` function (100 bins) as the main pipeline.
3. **CL tree** — Maximum spanning tree of the MI matrix via
   `msc_chow_liu.chow_liu_tree`. N-1 = 37 edges for 38 nodes.

**Saved outputs** (per subject in `analysis/canonical_circuits/motor_cortex/`):

| File | Content |
|------|---------|
| `{sub}_localizer_fc.npy` | 38×38 Pearson correlation matrix |
| `{sub}_localizer_mi.npy` | 38×38 mutual information matrix |
| `{sub}_localizer_cl_adj.npy` | 38×38 sparse CL adjacency (MI weights on tree edges) |
| `{sub}_localizer_roi_keys.json` | ordered ROI key list |

**Figures** (in `figures/canonical_circuits/motor_cortex/`):

| Figure | Description |
|--------|-------------|
| `{sub}_localizer_fc_matrix.pdf` | FC heatmap, ROI-labeled, effector group dividers |
| `{sub}_localizer_mi_matrix.pdf` | MI heatmap, same layout |
| `{sub}_localizer_cl_tree.pdf` | Hierarchical CL tree; nodes labeled with region name, colored by effector (red=foot, blue=hand, green=tongue, gray=whole-parcel) |
| `{sub}_localizer_fc_vs_mi.pdf` | Scatter: MI vs Pearson r for all 703 pairs; CL edges highlighted in red |
| `{sub}_localizer_fc_vs_cl_heatmaps.pdf` | Side-by-side: thresholded FC (r≥0.3) vs CL adjacency |

**Status:** Complete for all 10 subjects (8180 TRs × 38 ROIs per subject, all 10 sessions concatenated).

---

## Step 3 — Brain-Space Visualization (Complete)

**Script:** `mc_localizer_brain_viz.py`

**Status:** Complete for all 10 subjects (50 subject-level PDFs + 1 group-level somatotopic score).

**Output directory:** `figures/canonical_circuits/motor_cortex/`

| Visualization | Files | Content |
|---|---|---|
| Viz 1 — ROI Overlay | 10 PDFs | Localizer-defined foot/hand/tongue voxels on MNI template (sagittal + coronal) |
| Viz 2 — Glass Brain | 10 PDFs | FC (top-N edges) and CL tree (all 37 edges), nodes colored by effector |
| Viz 3 — Ortho Slices | 10 PDFs | CL tree edges on anatomical slices (sagittal/coronal/axial) |
| Viz 4 — Surface Projection | 10 PDFs | ROI masks on fsaverage5 inflated surface (L lat, L med, R lat, R med) |
| Viz 5 — Somatotopic Score | 1 PDF | MNI z-coordinate centroids (foot/hand/tongue) across all 10 subjects |

---

### Motivation

The matrix and tree figures describe *which* ROIs are connected but give no
spatial context. Brain-space visualization answers:
- Where exactly are the localizer-defined sub-parcels within areas 4 and 3b?
- How does somatotopic organization (foot superior → hand → tongue inferior)
  appear in the actual voxels selected?
- Which FC / CL edges connect spatially proximal vs. distal regions?

### Data available for visualization

| Data | Source |
|------|--------|
| 12 binary mask NIfTIs per subject (2 mm) | `localizer_masks/{sub}/` |
| 26 whole-parcel masks (shared) | Derivable from `glasser360MNI.nii.gz` |
| ROI centroid coordinates (to be computed) | from mask NIfTIs |
| FC matrix | `{sub}_localizer_fc.npy` |
| CL adjacency | `{sub}_localizer_cl_adj.npy` |
| MNI T1w template | nilearn `load_mni152_template()` |
| Glasser360 atlas (2 mm) | resampled on demand in existing scripts |

### Planned visualizations

---

#### Viz 1 — ROI Mask Overlay (somatotopic slice view)

**What:** A single composite NIfTI is built by assigning each localizer mask a
unique integer label (foot=1, hand=2, tongue=3, whole=4), then plotted as
`nilearn.plotting.plot_roi` on the MNI template.

**Key insight shown:** Where the top-50 activated voxels for each effector fall
within areas 4 and 3b, and whether they separate along the expected
superior–inferior somatotopic axis.

**Implementation sketch:**
```python
from nilearn.image import new_img_like, math_img
import nibabel as nib, numpy as np

def build_composite_mask(subject, mask_dir, ref_img):
    composite = np.zeros(ref_img.shape[:3], dtype=np.int16)
    label_map = {'foot': 1, 'hand': 2, 'tongue': 3}
    for effector, label in label_map.items():
        for parcel in ['Left_4', 'Right_4', 'Left_3b', 'Right_3b']:
            fpath = os.path.join(mask_dir, subject,
                f'{subject}_effector-{effector}_parcel-{parcel}_mask.nii.gz')
            if os.path.exists(fpath):
                data = np.asarray(nib.load(fpath).dataobj).astype(bool)
                composite[data] = label
    return new_img_like(ref_img, composite)
```

**Suggested display:** `plot_roi` with `cut_coords` along the sagittal axis
centered on area 4 (MNI x ≈ ±36), showing 3 coronal slices (y = –20, –10, 0).

---

#### Viz 2 — Glass Brain Connectome: FC and CL

**What:** Two `nilearn.plotting.plot_connectome` panels — FC (top-N edges by
|r|) and CL tree (all 37 edges) — overlaid on the glass brain. ROI nodes
colored by effector type.

**Centroid computation:**
```python
from nilearn.plotting import find_xyz_cut_coords

def roi_centroid_mni(mask_img):
    """Center of mass of a binary mask in MNI mm coordinates."""
    return find_xyz_cut_coords(mask_img)
```

For whole-parcel ROIs (no saved mask), derive on the fly from the resampled
Glasser atlas using the label integer.

**Matrix preparation:**
- FC panel: zero out all but the top-N edges by |r| (suggested N = 2*(n-1) = 74,
  i.e. 2× the CL edge count, for a fair visual comparison).
- CL panel: use `cl_adj` directly; edge width proportional to MI weight.

**Node coloring:** pass a `node_color` array to `plot_connectome`, using the
`_EFFECTOR_COLORS` dict from `mc_localizer_fc_cl.py`.

---

#### Viz 3 — CL Tree Edges on Anatomical Slices

**What:** For each CL edge, draw a line between the two ROI centroids overlaid
on axial/coronal/sagittal MNI slices, edge width ∝ MI weight, edge color by
effector type of the source node. Uses `nilearn.plotting.plot_connectome` with
`display_mode='ortho'`.

**Why separate from Viz 2:** The glass brain view can be hard to read for
densely connected local pairs. The ortho slice view shows depth and exact
anatomical location.

---

#### Viz 4 — Surface Projection of ROI Masks

**What:** Project the composite mask NIfTI onto the `fsaverage5` inflated
surface using `nilearn.plotting.plot_surf_roi`. Shows left and right
hemispheres in lateral and medial views, with effector color coding.

**Implementation note:** requires `nilearn.surface.vol_to_surf` to sample the
volumetric label map onto the surface mesh.

```python
from nilearn import surface, plotting

def project_to_surface(composite_img, hemi='left'):
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    texture = surface.vol_to_surf(
        composite_img,
        fsaverage[f'pial_{hemi}'],
        interpolation='nearest',
    )
    return texture
```

**Output:** 4-panel figure (L lateral, L medial, R lateral, R medial).

---

#### Viz 5 — Per-Subject Somatotopic Separation Score

**What:** A scalar summary of whether foot/hand/tongue sub-parcels are
spatially separated as expected. For each subject, compute the MNI z-coordinate
(superior–inferior axis) centroid for the foot, hand, and tongue sub-regions
of area 4. Plot as a grouped bar chart across subjects.

**Interpretation:** If the localizer correctly captures somatotopy, foot
centroids should be most superior (highest z), tongue most inferior (lowest z),
with hand in between.

---

### Implementation order

1. **Centroid computation utility** — add `compute_roi_centroids(subject, mask_dir, roi_keys)` to `canonical_utils.py`. Returns a dict `{roi_key: (x_mni, y_mni, z_mni)}`.
2. **Viz 1** (ROI overlay) — simplest; no connectivity data needed. Good sanity check that the localizer masks land where expected anatomically.
3. **Viz 2** (glass brain connectome) — uses centroids + existing matrices.
4. **Viz 3** (ortho slice edges) — variant of Viz 2.
5. **Viz 4** (surface projection) — depends on fsaverage being available; verify with `nilearn.datasets.fetch_surf_fsaverage('fsaverage5')`.
6. **Viz 5** (somatotopic separation score) — requires all 10 subjects complete.

### CLI design

```
python mc_localizer_brain_viz.py [--subjects MSC01 ...]
                                  [--viz {all,roi,connectome,surface,soma}]
                                  [--fc-top-n 74]
                                  [--no-surface]  # skip if fsaverage unavailable
```

---

## Relationship to `circuit_plan.md`

`circuit_plan.md` describes the broader canonical circuits module including
cortico-cerebellar trees and effector clustering. The analyses in this file
focus specifically on the **subject-specific localizer ROI** sub-pipeline, which
is a prerequisite for the effector-level tree analysis (`mc_effector_trees.py`)
planned in `circuit_plan.md`. Once visual mapping confirms the localizer masks
are anatomically sensible, the localizer-defined ROI timeseries can replace the
whole-parcel approach for effector trees.
