# fmri_connectivity_trees — Deep Research Report

## 1. Project Purpose

The central research question: **Does the Chow-Liu (CL) tree recover sparse/hierarchical ground-truth brain connectivity more reliably than standard functional connectivity (FC), and does it detect nonlinear coupling that Pearson correlation misses?**

Two parallel tracks address this:

| Track | Driver | Ground Truth | Purpose |
|---|---|---|---|
| **Empirical** | `code/functional_connectivity/` + `code/jupyter_notebooks/` | None — real fMRI | Extract ROI time series → MI → FC → CL trees on MSC/ABIDE/LISTEN |
| **Simulation** | `code/simulations/` | Synthetic connectivity matrices | Quantify CL-vs-FC recovery across network types, noise levels, and coupling modes |

---

## 2. Repository Layout

```
CLAUDE.md                         project documentation
README.md
atlases/                          Glasser360, SUIT, MorelAtlasMNI152, Brainstem, Thalamus, HCP-MMP1
datasets/synthetic/<run_id>/      simulation time-series output (NPZ)
environments/dynamric1.yml        conda env "dynamric", Python 3.11
code/
  functional_connectivity/{midnight_scan_club,abide,listen}/
  jupyter_notebooks/{midnight_scan_club,abide,listen}/
  simulations/                    dynamics engine, analysis, SLURM jobs
  classification/abide/           BrainNetCNN + pretrained backbones
misc/CL_notes.md                  research hypothesis log (Milo-annotated)
misc/noise.md                     noise_sim hypothesis rationale
```

Three strict conventions tie the codebase together:

1. **Run ID**: `YYYYMMDD_HHMMSS_<tag>` created by `sim_utils.make_run_id()`, used as the directory name in timeseries, analysis output, and figures.
2. **`params.json`**: written to both the timeseries dir and analysis dir for every run; includes git hash, script name, and full argparse args — either directory is self-documenting.
3. **Multi-machine path resolution**: MSC scripts use a `_CANDIDATES` list in `msc_paths.py` probed at import time. ABIDE/LISTEN use hard-coded `/mfs/io/groups/dmello/...` paths (inconsistent).

---

## 3. Empirical Track — MSC Pipeline (Primary)

### 3.1 Path & Device Helpers (`msc_paths.py`)

- `BASE_DIR`: resolved by probing `_CANDIDATES`; returns first existing path.
- Directory helpers: `ts_root()`, `ts_dir(subject)`, `mi_dir(subject)`, `cov_dir(subject)`, `jp_dir()`, `pm_dir()`, `ent_dir()`, `analysis_dir()`, `figures_dir()`.
- `detect_device()`: priority CUDA → MPS → CPU.
- `glasser360_labels_path()`: resolves the atlas node-labels file.

### 3.2 Time-Series Extraction (`msc_extract_timeseries.py`)

`fetch_atlas(name)` / `get_masker(atlas, name)` support:
- **HarvardOxford**, **Schaefer** (1000 ROIs × 17 Yeo nets), **MSDL**
- **glasser360** (360 cortical parcels)
- **SUIT** (34 cerebellar)
- **Morel_Left/Right_Global_Thalamus**, **Morel_All** (returns a dict of per-subregion `NiftiLabelsMasker` objects)

`construct_url()` builds the ds000224 fmriprep path:
```
<base>/sub-MSC0X/ses-func0Y/func/sub-MSC0X_ses-func0Y_task-<id>_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
```

`get_confounds()` selects 12 columns from fmriprep TSV:
```python
['cosine00','cosine01','cosine02','cosine03',
 'csf','rot_x','rot_y','rot_z','trans_x','trans_y','trans_z','white_matter']
```
(TSV read with `engine='python'`, `encoding='latin-1'`, `on_bad_lines='skip'`.)

Output: one CSV per task at:
```
output/roi_time_series/<subject>/<session>/<atlas>/all_tasks/pooled/<task>.csv
                                                             /shape/<task>.csv
```

**Note:** `msc_extract_timeseries_optimized.py` (which adds `ProcessPoolExecutor`) has a syntax error on line 99 (missing trailing comma in `NiftiLabelsMasker`) and will not run as-is.

### 3.3 Mutual Information (`msc_mutual_info.py`)

Torch-based, GPU-capable pipeline:

1. `discretize_time_series(ts, num_bins)` — `torch.bucketize` on `linspace(ts.min(), ts.max(), num_bins+1)[1:-1]`.
2. `joint_prob(a, b, disc, K)` — `torch.bincount(K*a + b, minlength=K²).reshape(K, K)`.
3. `get_product_of_marginals()` — `torch.outer(p_a, p_b)`.
4. `get_mutual_information()` — `MI = Σ jp · log(jp / pm)`, clamped at `1e-10`.

Default: `num_bins=100`, sessions `func01–func10`, task `rest`. Combines atlases by concatenating ROI time-series columns (`glasser360 + SUIT + all Thalamus subregions`). Output → `output/mutual_information/<subject>/<atlas>/<task>_100bins_func-01-10.npy`.

### 3.4 Covariance (`msc_covariance.py`)

Mirrors MI script: `torch.cov` with diagonal zeroed, plus a parallel nilearn `ConnectivityMeasure` path. Output → `output/covariance/`.

### 3.5 Chow-Liu Trees (`msc_chow_liu.py`)

Key functions:

| Function | What it does |
|---|---|
| `load_glasser360_labels()` / `load_node_labels()` | Parse atlas labels |
| `infer_network()` | Map ROI name → one of 9 networks (Visual, Somatomotor, DorsAttn, Frontoparietal, DefaultMode, Language, Cerebellum, Thalamus, Other) |
| `mi_to_graph(mi)` | Build weighted `nx.Graph` from MI matrix |
| `chow_liu_tree(G)` | `nx.maximum_spanning_tree(G, algorithm='kruskal', weight='weight')` |
| `_hierarchy_pos()` | BFS from root → top-down tree layout (no graphviz dependency) |
| `draw_hierarchical_tree()` / `plot_cl_trees()` | Per-subject PDFs |
| `plot_consensus_tree()` | Edge vote matrix across subjects, threshold ≥50%, saves `edge_vote_matrix.npy` + consensus PDF |

CLI flags: `--subjects`, `--atlas-subdir` (default `glasser360_SUIT_Thalamus`), `--num-bins` (100), `--suffix` (`func-01-10`).

### 3.6 Clustering Comparison (`msc_clustering.py`)

Compares hierarchical clusterings from three edge-weightings:

- **Correlation distance**: `1 - |r|`
- **MI distance**: `1 / (1 + MI)`
- **CL tree**: graph distance on the max-spanning-tree

`compute_linkages()` runs ward + average linkage. `network_coherence(k)` = fraction of within-Yeo-network ROI pairs in the same cluster at cut level *k*.

Output figures:
- `fig1_colored_dendrograms.pdf` — per-method dendrograms coloured by canonical network
- `fig2_tanglegram_*.pdf` — pairwise tanglegrams (Correlation↔MI, Correlation↔CL, MI↔CL)
- `fig4a/b_*_coherence*.pdf` — network coherence vs cut level *k*
- `fig5_coclustering_agreement.pdf`, `fig5b_ari_nmi.pdf` — ARI/NMI heatmaps

### 3.7 MSC Notebooks

| Notebook | Role |
|---|---|
| `msc_fc_cl.ipynb` | End-to-end playground: volumes → pooled CSV → MI → FC → CL |
| `msc_fc_cl_motor.ipynb` / `msc_fc_cl_motor_fig.ipynb` | Cortico-cerebellar motor hierarchy figures |
| `msc_cl_rs_playground.ipynb` | Resting-state variant |
| `msc_mutual_info.ipynb` | MI walkthrough |
| `chow_liu_tree_visualization.ipynb` | Visual refinements / consensus-tree figures |
| `hierarchical_clustering.ipynb` | Tanglegram/ARI exploration |
| `cereb_flat_map.ipynb` | SUIT-space cerebellum flat-map visualizations |
| `sfn_figures.ipynb` | SfN poster figure compilation (4.4 MB) |
| `nilearn_playground.ipynb` | Atlas/masker sandbox |

---

## 4. ABIDE & LISTEN Tracks

### 4.1 ABIDE (`abide_extract_timeseries.py`)

- `construct_url()` → `<base>/<file_id>_func_preproc.nii.gz` (Preprocessed Connectomes Project naming).
- `get_phenotypes()` reads `Phenotypic_V1_0b_preprocessed1.csv`, maps `FILE_ID → DX_GROUP` (ASD=1, TDC=0).
- Same 12-column confound design as MSC.
- Default: 100 subjects, `glasser360`, output → `output/roi_time_series/<n>_<atlas>/pooled/<subject_id>.csv`.

Notebooks: `abide_extract_timeseries.ipynb`, `abide_funccon_chow_liu_40.ipynb`, `abide_msdl_chowliu_simulation.ipynb`, `abide_msdl_prediction_simulation.ipynb` — classification-oriented.

### 4.2 ABIDE CNN Classifier (`abide_cnn.py`)

PyTorch model zoo for connectivity-matrix classification (ASD vs TDC):

- **BrainNetCNN** — Kawahara-style E2E (`1×N` + `N×1`) conv → E2N → N2G → linear. Input: `[batch, 1, N, N]`.
- **VGG16_Connectivity** / **ResNet_Connectivity** — pretrained ImageNet backbones with `conv1` rewritten to 1-channel input.
- `evaluate(y, y_pred, y_prob)` → accuracy, sensitivity, specificity, ROC-AUC.

### 4.3 LISTEN (`listen_extract_timeseries.py`)

Narrative-listening dataset. Same masker/atlas machinery; per-story `file_ids` lists. Default atlas = Schaefer 100/7 nets. Output → `output/roi_time_series/<subject>/<session>/<atlas>/all_tasks/pooled/<file_id>.csv`.

---

## 5. Simulation Framework

### 5.1 Dynamics Engine (`dynamics_simulation.py`)

Custom **Torch-accelerated Heun stochastic integrator** for a Generic2dOscillator neural-mass model.

**Coupling-type constants:**
```
LINEAR=1, QUADRATURE=2, RECTIFIED=3, SQUARED=4, PAC=5
```
Per-edge coupling transforms in `_coupling_transform_torch(V_hist, W_hist, type_idx)`:
- `LINEAR`: `V`
- `QUADRATURE`: `W`
- `RECTIFIED`: `max(V, 0)`
- `SQUARED`: `V²`
- `PAC`: `(0.5 + 0.4·V)·W`

**Model RHS (`rhs_generic_2d`):**
```
dV = d·tau·(-f·V³ + e·V² + g·V + α·W + γ·coupling)
dW = (d/tau)·(c·V² + b·V - β·W + a)
```

**TVB defaults** with per-node heterogeneous parameters:
```
a=0, b=-10, c=0, d=0.02, e=3, f=1, g=0
alpha=1, beta=1, gamma=1, tau=1
Sigmas: b_σ=0.5, tau_σ=0.15 (others smaller)
```

**Connectivity generators** (`build_connectivity(N, mode, rng, ...)`):
- `"tree"` — `nx.random_labeled_tree` (Prüfer), weights uniform `[0.5, 3.0]`
- `"hierarchical"` — BFS with 1–`branching` children/node
- `"erdos_renyi"` — random edges at `density`, bridged for connectivity
- `"dense"` — asymmetric `rng.integers(0, 4)` weights

**Integration:** Heun stochastic with circular history buffer of size `max(idelays)+1`. Advanced indexing + `torch.where` gathers per-edge delayed V/W. `TemporalAverage` monitor period = 5 ms (200 Hz output). Defaults: `dt=0.5 ms`, `conduction_speed=3.0 mm/ms`, `simlen=600000 ms` (10 min).

**Performance modes:**
- `"fidelity"` (default): float64 on CUDA/CPU, float32 on MPS
- `"fast"`: float32 + TF32 matmul

### 5.2 Analysis (`analyze_sim.py`)

| Function | Description |
|---|---|
| `load_simulation()` | Handles both `weights` (TVB) and `conn` (custom) keys in NPZ |
| `fc_significance_threshold()` | Fisher-z → z-stat `Z·√(T-3)` → FDR via `statsmodels.fdrcorrection` |
| `density_threshold_matrix(M, n)` | `np.argpartition` keeps top-n upper-triangle edges, makes symmetric |
| `compute_fc()` | Returns `(fc_raw, fc_significance, fc_density)` |
| `compute_mi(V, num_bins=100)` | Same torch-based pairwise MI as MSC pipeline |
| `compute_chow_liu(mi)` | `nx.maximum_spanning_tree` via Kruskal |
| `evaluate(C, matrices_dict)` | Auto-detects sparse GT (≤4N edges); computes Pearson, MSE, TP/FP/TN/FN, F1, sensitivity, specificity, precision |

Sweep helpers: `discover_sweep()`, `analyze_trial()`, `merge_and_save_sweep_results()`.

### 5.3 Sweep Orchestrators

| Script | Sweeps | Output |
|---|---|---|
| `sweep_sim.py` | N values × trials; all 7 methods | Per-trial analysis + `sweep_results.npz` + `sweep_summary.json` |
| `grid_sim.py` | (N, G, connectivity) cells; D varies per individual | Only timeseries (no analysis) |
| `noise_sim.py` | Fixed 15-node tree, 5 individuals × 5 noise levels | Noise-robustness hypothesis test |

**`sweep_sim.py` methods:** `fc_significance`, `fc_density`, `fc_matched`, `mi_raw`, `mi_density`, `mi_matched`, `cl_adjacency`.

**`sweep_sim.py` metrics:** `pearson`, `pearson_pvalue`, `f1`, `sensitivity`, `specificity`, `precision`, `mse`.

Defaults: `--N-values 5 10 15 20 25 --trials 10 --simlen 600000 --G 0.3 --D 2e-4 --num-bins 100`.

**`grid_sim.py`** parses connectivity specs like `'erdos_renyi:0.1'`, `'hierarchical:2'`. Output:
```
datasets/synthetic/<run_id>/timeseries/N{n}/G{g:.2f}/{conn_tag}/individual_{k+1}_D{D:.0e}.npz
```

**`noise_sim.py`:** `N_NODES=15`, `N_INDIVIDUALS=5`, `NOISE_LEVELS=[1e-5, 5e-5, 2e-4, 8e-4, 3e-3]`, `G=0.3`, LINEAR coupling, shared tree GT. Metrics: `edge_set()`, `jaccard()`, `pairwise_jaccard_matrix()`. Produces:
- `noise_sim_networks.pdf` — GT + CL + FC panels per noise level
- `noise_sim_consistency.pdf` — Jaccard heatmaps + GT recovery bars

### 5.4 Grid Analysis (`grid_analysis.py`)

Self-contained GPU MI + FC + CL computation on grid_sim output.

`analyze_individual()` returns: `auc_roc`, `auc_pr`, `f1/precision/recall` at matched edge count, `degree_corr` via `scipy.stats.spearmanr`.

Figures:
- `fig_a_recovery.png` — F1/AUC/AUC-PR/Degree-Corr vs noise D with connectivity rows
- `fig_b_consistency.png` — per-condition pairwise Jaccard, CL vs FC matrix correlation
- `fig_b_consistency_summary.png` — aggregated consistency
- `fig_structure_heatmaps.png` — adjacency panels
- `fig_summary_bars.png`

Constants: `CONN_ORDER = ['er_0p10', 'er_0p50', 'tree']`, `METHOD_COLORS = {CL: '#5e3c99', FC: '#e66101'}`.

### 5.5 Visualization Scripts

| Script | Role |
|---|---|
| `visualize_sweep.py` (1019 lines) | Reads `sweep_results.npz`; produces `metrics_vs_N.png` (2×3 metric grid), `matrices_N{N}.png` (GT/FC-sig/FC-matched/MI-raw/MI-matched/CL heatmaps), `graphs_N{N}.png`, `coupling_graphs_N{N}.png` (TP green, FP faint gray, FN dashed gray). Multi-sweep `combine_sweeps()` with `:N` filter-spec parser. |
| `plot_coupling_types.py` | 2-node unidirectional setup, one sim per coupling type; shows source V/W + coupling signal + target response. Uses Hilbert transform + Butterworth bandpass to recover W post-hoc. |
| `mi_vs_correlation_demo.py` | Three modes (`--mode abstract/oscillator/coupling-types`); shows `|r| ≈ 0` but `MI > 0` for Quadrature/Squared/PAC. Core empirical justification for MI over FC. |
| `_regen_figures.py` | Reload noise_sim NPZs and regenerate figures without re-simulating. |

### 5.6 Legacy / TVB Scripts

| Script | Role |
|---|---|
| `generic_2doscillator.py` | Older CPU-only simulator with inner Python loop (reference before torch rewrite) |
| `generic_2doscillator_tvb.py` | TVB API equivalent (`simulator.Simulator` + `coupling.Linear` + `integrators.HeunStochastic` + `monitors.TemporalAverage`) |
| `TVB_simFC.py` | TVB FC pipeline: Pearson → Fisher-z → Z-stat → FDR; iterates over patient connectomes |
| `TVB_simMI.py` | Same framework, torch-based MI |
| `TVB_simCL.py` | TVB CL helpers: `mi_to_graph`, `chow_liu_tree`, `calculate_chow_liu_trees` |
| `TVB_evalsim.py` | Evaluation against `SCthrAn.mat` reference connectome |
| `tvb_utils.py` | TVB visualization helpers (threaded runner, cortex triangulation plots) |

### 5.7 SLURM Jobs (`code/simulations/jobs/`)

**`grid_job.sh`** — SLURM array `0-7` over 8 connectivity specs (tree, hier:2, hier:3, er:0.1/0.3/0.5/0.7/0.9):
- Sweeps `N_VALUES="5 10 15 20 25 50 75 100 200 400"`, `G_VALUES="0.1 0.3 0.5"`, `NOISE_LEVELS="1e-5 5e-5 2e-4 8e-4 3e-3"`
- GPU partition, 64 GB RAM, 3-day walltime, `SEED=2000+task_id`

**`sweep_job.sh`** — SLURM array `0-48` over 7×7 connectivity × coupling grid:
- `CONN_IDX 0-4` = ER densities; `5`=hier:2; `6`=hier:3
- `COUP_IDX 0-4` = LINEAR/QUADRATURE/RECTIFIED/SQUARED/PAC; `5`=random_all; `6`=random_lqs
- 32 GB RAM, 2-day walltime

---

## 6. Data Flow

### Empirical (MSC)
```
NIfTI (fMRIPrep)
  └─→ msc_extract_timeseries.py  [NiftiLabelsMasker + confound removal]
        └─→ pooled/<task>.csv
              ├─→ msc_mutual_info.py  [torch pairwise MI]
              │     └─→ output/mutual_information/<subject>/<atlas>/<task>_100bins_func-01-10.npy
              │           ├─→ msc_chow_liu.py  [nx.maximum_spanning_tree]
              │           │     └─→ per-subject CL tree PDFs + consensus PDF + edge_vote_matrix.npy
              │           │           └─→ msc_clustering.py  [linkage + tanglegram + ARI]
              │           │                 └─→ fig1-fig5 PDFs
              │           └─→ (notebooks for additional figures)
              └─→ msc_covariance.py  [torch.cov / nilearn ConnectivityMeasure]
                    └─→ output/covariance/
```

### Simulation
```
build_connectivity (tree/hier/ER/dense) + heterogeneous params
  └─→ run_sim  [Heun stochastic, torch GPU]
        └─→ V(t)  [NPZ in datasets/synthetic/<run_id>/]
              └─→ compute_fc / compute_mi / compute_chow_liu
                    └─→ evaluate(C, matrices)  [pearson/F1/TP-FP-TN-FN]
                          └─→ sweep_results.npz + sweep_summary.json
                                └─→ visualize_sweep.py  [metrics/matrices/graphs PNGs]
```

---

## 7. Key Algorithms

### 7.1 Chow-Liu Tree
Maximum spanning tree of the pairwise-MI graph:
```python
nx.maximum_spanning_tree(G, algorithm='kruskal', weight='weight')
```
Under the tree distribution assumption, this is the information-maximizing dependency tree (Chow & Liu, 1968).

### 7.2 Vectorized Pairwise MI
1. Discretize `(N, T)` time series to K=100 bins via `torch.bucketize`
2. Encode joint bins as `K·a + b`, compute joint probability via `torch.bincount`
3. Product-of-marginals from row/column sums
4. `MI = Σ jp · log(jp / pm)` with `1e-10` clamp

### 7.3 FC Significance Thresholding
Pearson → Fisher-z `arctanh` → z-statistic `Z·√(T-3)` → two-sided p-values → `statsmodels.fdrcorrection` on upper-triangle → `FC_thr = C * sig`.

### 7.4 Density-Matched Thresholding
`np.argpartition(|weights|, -N)[-N:]` keeps top-N upper-triangle edges, symmetrized. Used to match FC/MI edge count to CL's N-1 edges.

### 7.5 Density Matching Nuance
In `sweep_sim.py::run_single_trial`: uses `n_match = n_gt_edges if n_gt_edges ≤ 4N else N-1`. So for sparse GT (tree/hierarchical) it matches the actual GT edge count; for dense GT it collapses to N-1 (tree-density). This produces `fc_matched` / `mi_matched` alongside `_significance` / `_density` / `_raw` variants.

### 7.6 Consensus Tree
Binary edge-existence accumulated across subjects; threshold at ≥50%; saved as `edge_vote_matrix.npy`.

### 7.7 Network Coherence at Cut k
Fraction of intra-Yeo-network ROI pairs assigned to the same cluster at dendrogram cut k; used to show CL trees respect known network boundaries.

### 7.8 Edge-Set Jaccard
`|E₁ ∩ E₂| / |E₁ ∪ E₂|` used for (a) GT recovery in `noise_sim.py`, (b) inter-individual CL consistency in `grid_analysis.py`.

### 7.9 Heun Stochastic Integration
Circular history buffer of size `max(idelays)+1`; `torch.where`-based gather reads per-edge delayed V/W; additive Gaussian noise with amplitude D; predict-then-correct Heun step.

---

## 8. The Key Experimental Lever: Coupling Types

The 5 coupling transforms are the core experimental design — they probe whether FC's linearity bias causes it to miss nonlinear dependencies that MI/CL can detect:

| Coupling | FC predicts? | MI predicts? | Why |
|---|---|---|---|
| LINEAR | Yes — `|r|` high | Yes | Pearson captures linear covariation |
| QUADRATURE | No — `|r| ≈ 0` | Yes | Phase coupling, not amplitude |
| RECTIFIED | Partial | Yes | Asymmetric, not zero-mean |
| SQUARED | No — even-function | Yes | Non-monotonic |
| PAC | No — amplitude-phase | Yes | Phase-amplitude coupling |

`mi_vs_correlation_demo.py --mode coupling-types` and `plot_coupling_types.py` exist specifically to visualize this effect empirically.

---

NOTE: this is not currently my key experimental lever. I use linear coupling and inject nonlinearity through having heterogenous per-node parameters. In the future I will determine how to include these coupling types in a way my inference models can detect (currently they cannot). 

## 9. Combined Atlas Design

MSC runs use `glasser360 + SUIT + Morel_All`, combining atlases by **concatenating ROI time-series columns**. The resulting MI matrix captures cortico-cerebellar-thalamic dependencies in a single CL tree — directly enabling research question #4 (motor cortex → motor cerebellum → other cerebellum hierarchy).

---

## 10. Notable Implementation Quirks

1. **Syntax bug** — `msc_extract_timeseries_optimized.py:99` missing trailing comma in `NiftiLabelsMasker` call; script will not run.

2. **Two MI implementations** — `msc_mutual_info.py` and `analyze_sim.py::compute_mi` both use bincount-based MI; `TVB_simMI.py` and `mi_vs_correlation_demo.py` have slight clamp differences (`== 0 → 1e-10` vs `.clamp(min=1e-10)`). Numerically identical; a future refactor should unify to a single function.

3. **Multi-machine paths are inconsistent** — MSC uses the `_CANDIDATES` probe; ABIDE/LISTEN use hard-coded `/mfs/io/` paths. The latter only run on the lab server.

4. **TVB reference** — `generic_2doscillator_tvb.py` is the validation reference for the custom torch integrator; they should produce statistically equivalent dynamics given the same parameters.

5. **No graphviz dependency** — `msc_chow_liu._hierarchy_pos()` and `visualize_sweep.py` implement BFS-based tree layout from scratch. Critical for HPC/container environments.

6. **No test suite** — no `test_*.py` files or `tests/` directory. Correctness is validated by reproducing the same figure types across TVB and the custom engine.

7. **Abundance of notebook artifacts** — root-level `msc_chow_liu.ipynb` plus ~20 notebooks and dozens of PDFs in `code/jupyter_notebooks/midnight_scan_club/`. Scripts are the cluster-runnable backbone; notebooks handle publication figures.

8. **Research status (from `misc/CL_notes.md`)** — CL beats FC on sparse/hierarchical GT; fails on dense GT; noise robustness inconclusive; MI vs FC nonlinear-coupling advantage present in simulation, needs more empirical data.

---

## 11. Load-Bearing Files Reference

**Must understand for the full picture:**

| File | Role |
|---|---|
| `code/functional_connectivity/midnight_scan_club/msc_paths.py` | Path/device resolution |
| `code/functional_connectivity/midnight_scan_club/msc_extract_timeseries.py` | fMRI → ROI time series |
| `code/functional_connectivity/midnight_scan_club/msc_mutual_info.py` | Torch pairwise MI |
| `code/functional_connectivity/midnight_scan_club/msc_chow_liu.py` | CL tree construction + visualization |
| `code/functional_connectivity/midnight_scan_club/msc_clustering.py` | Clustering comparison |
| `code/simulations/dynamics_simulation.py` | Neural-mass model + integrator |
| `code/simulations/analyze_sim.py` | FC/MI/CL computation + evaluation metrics |
| `code/simulations/sweep_sim.py` | Main sweep orchestrator |
| `code/simulations/grid_sim.py` | Grid sweep (N × G × connectivity) |
| `code/simulations/noise_sim.py` | Noise robustness hypothesis test |
| `code/simulations/grid_analysis.py` | Grid output analysis |
| `code/simulations/visualize_sweep.py` | Figure production |
| `code/simulations/sim_utils.py` | Shared utilities (run ID, paths, device) |
| `misc/CL_notes.md` | Research hypothesis log |
