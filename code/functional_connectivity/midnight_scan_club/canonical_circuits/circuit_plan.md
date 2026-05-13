# Canonical Circuits — Implementation Plan

## Overview

This plan describes the implementation of `canonical_circuits`, a new analysis module under
`code/functional_connectivity/midnight_scan_club/canonical_circuits/`. It ports the exploratory motor
connectivity analysis from the Jupyter notebooks (`msc_fc_cl_motor.ipynb`, `msc_fc_cl_motor_fig.ipynb`)
into production-quality Python scripts, extends the analysis to the full Glasser360 + SUIT parcellation,
adds effector-level resolution (foot / hand / mouth), and integrates the clustering pipeline already
used by `msc_clustering.py`.

Two sub-modules are created:

| Sub-module | Directory | Scope |
|---|---|---|
| **Cortico-cerebellar motor** | `corticocerebellar_motor/` | All motor + premotor cortex ROIs (Glasser360) × all cerebellar ROIs (SUIT 34-parcel) |
| **Motor cortex effectors** | `motor_cortex/` | Foot / hand / mouth effector ROIs in motor and premotor cortex, Gordon-et-al-defined, including pre-SMA and cingulate motor areas |

---

## Scientific Motivation

### 1. The Cortico-Cerebellar Motor Circuit

The cerebellum is reciprocally connected to motor cortex via pontine nuclei and thalamus, forming a
closed-loop circuit that is critical for skilled movement. Buckner et al. (2011) demonstrated using
1,000-subject resting-state fMRI that cerebellar lobules IV/V/VI and VIII mirror the cerebral motor
homunculus, with a *double somatotopic representation* (an inverted and an upright map). Stoodley &
Schmahmann (2009) confirmed via meta-analysis that lobule V shows preferential foot/leg activation,
lobule VI shows hand/finger activation, and lobule VIII contains a second complete somatotopic map.
Guell et al. (2018) further showed a third non-motor cerebellar zone corresponding to the default
mode and frontoparietal networks.

Current standard FC (Pearson correlation) describes only pairwise linear dependencies. The CL tree
approach recovers the *hierarchical* dependency structure of these circuits — that is, it finds the
spanning tree of maximum total MI, which under the tree-distribution assumption is the most
information-preserving sparse model of the joint distribution (Chow & Liu, 1968). Research question
#5 in `misc/CL_notes.md` asks specifically whether CL reveals a motor cortex → motor cerebellum →
other cerebellum hierarchy that FC cannot detect.

### 2. Effector-Specific Somatotopy in Motor Cortex

Glasser et al. (2016) delineated 180 cortical areas per hemisphere using multimodal MRI including
myelin maps, task activations, connectivity topography, and cortical thickness. Area 4 (Primary
Motor Cortex) is a single parcel in the atlas but spans the motor homunculus from the paracentral
lobule (foot) through the lateral central sulcus (hand) down to the ventral opercular region (mouth).

Gordon et al. (2023) reanalyzed the MSC dataset's motor task (the same dataset used here) and made
a critical discovery: motor cortex is *not* a single continuous homunculus but instead contains
interdigitated effector-specific regions (foot, hand, mouth) alternating with regions of a
Somato-Cognitive Action Network (SCAN) with strong cingulo-opercular connectivity. This challenges
the classical view and implies that effector-specific CL trees may reveal distinct circuit
architectures for each effector.

Picard & Strick (2001) established that pre-SMA and dorsal/ventral premotor cortex (PMd/PMv,
corresponding to Glasser areas 6d/6v/6r/6a) have distinct functional roles in action selection and
preparation, with anatomical projections to separate cerebellar zones.

### 3. Why CL Trees Over FC for These Circuits

Standard FC computes the full N×N correlation matrix. For motor circuits this is both redundant (many
indirect paths inflate the apparent connectivity of central hubs) and insensitive to nonlinear coupling
such as phase-amplitude coupling (PAC) or quadrature coupling, which may be important for
cerebellar timing signals. Mutual information (used as the CL edge weight) is sensitive to all
statistical dependencies, not only linear ones, as demonstrated by `mi_vs_correlation_demo.py`.
Simulation results in `grid_analysis.py` show that CL trees recover sparse tree-structured ground
truths with higher F1 than density-matched FC.

---

## Citations

- Buckner, R.L., Krienen, F.M., Castellanos, A., Diaz, J.C., & Yeo, B.T.T. (2011). The organization
  of the human cerebellum estimated by intrinsic functional connectivity. *Journal of Neurophysiology*,
  106(5), 2322–2345.
- Chow, C.K., & Liu, C.N. (1968). Approximating discrete probability distributions with dependence
  trees. *IEEE Transactions on Information Theory*, 14(3), 462–467.
- Diedrichsen, J., King, M., Hernandez-Castillo, C., Sereno, M., & Ivry, R.B. (2019). Universal
  transform or multiple functionality? Understanding the contribution of the human cerebellum across
  task domains. *Neuron*, 102(5), 918–928.
- Glasser, M.F., Coalson, T.S., Robinson, E.C., ..., & Van Essen, D.C. (2016). A multi-modal
  parcellation of human cerebral cortex. *Nature*, 536, 171–178.
- Gordon, E.M., Laumann, T.O., Adeyemo, B., ..., & Petersen, S.E. (2017). Precision functional
  mapping of individual human brains. *Neuron*, 95(4), 791–807.
- Gordon, E.M., Chauvin, R.J., Van, A.N., ..., & Dosenbach, N.U.F. (2023). A somato-cognitive action
  network alternates with effector regions in motor cortex. *Nature*, 617, 351–359.
- Guell, X., Schmahmann, J.D., Gabrieli, J.D.E., & Ghosh, S.S. (2018). Functional gradients of the
  cerebellum. *eLife*, 7, e36652.
- Picard, N., & Strick, P.L. (2001). Imaging the premotor areas. *Current Opinion in Neurobiology*,
  11(6), 663–672.
- Stoodley, C.J., & Schmahmann, J.D. (2009). Functional topography in the human cerebellum: A
  meta-analysis of neuroimaging studies. *NeuroImage*, 44(2), 489–501.

---

## Directory Layout (after implementation)

```
canonical_circuits/
├── plan.md                          ← this file
├── canonical_utils.py               ← shared ROI definitions, label helpers, I/O
├── corticocerebellar_motor/
│   ├── cc_motor_trees.py            ← CL trees for cortico-cerebellar motor circuit
│   ├── cc_motor_clustering.py       ← clustering dendrograms + clustering maps
│   └── cc_motor_figures.py          ← figure assembly (brain maps, connectome overlays)
└── motor_cortex/
    ├── mc_effector_trees.py         ← per-effector CL trees (foot/hand/mouth)
    ├── mc_effector_clustering.py    ← per-effector clustering
    └── mc_effector_figures.py       ← somatotopic layout figures
```

Output directories (created by the scripts, following `msc_chow_liu.py` conventions):

```
code/functional_connectivity/midnight_scan_club/
├── analysis/
│   ├── canonical_circuits/
│   │   ├── corticocerebellar_motor/   ← CL adjacency NPY, clustering NPZ
│   │   └── motor_cortex/
└── figures/
    ├── canonical_circuits/
    │   ├── corticocerebellar_motor/   ← PDF/PNG figures
    │   └── motor_cortex/
```

---

## ROI Definitions

### Glasser360 Motor and Premotor Parcels

The Glasser360 atlas labels are stored in `atlases/glasser360/glasser360NodeNames.txt` and loaded by
`msc_chow_liu.load_glasser360_labels()`. The atlas orders **Right hemisphere first (indices 0–179)**,
then **Left hemisphere (indices 180–359)**. The CSV `HCP-MMP1_UniqueRegionList.csv` lists parcels
with `regionID` where Right IDs are 201–380 (0-based index = regionID − 201) and Left IDs are 1–180
(0-based index = regionID + 179).

Key parcels by function:

```python
# ─── Primary Motor ─────────────────────────────────────────────────────────────
AREA_4 = {          # Primary Motor Cortex — full homunculus
    'Right_4': 7,   # 0-based glasser360 index
    'Left_4':  187,
}

# ─── Primary Somatosensory ──────────────────────────────────────────────────────
AREA_3b = {'Right_3b': 8,  'Left_3b':  188}   # Tactile input to motor circuit
AREA_3a = {'Right_3a': 52, 'Left_3a':  232}   # Proprioceptive
AREA_1  = {'Right_1':  50, 'Left_1':   230}
AREA_2  = {'Right_2':  51, 'Left_2':   231}

# ─── Supplementary Motor Area / Paracentral ─────────────────────────────────────
SMA = {                      # SMA proper (6mp) — movement initiation, sequencing
    'Right_6mp': 54, 'Left_6mp': 234,
}
PRE_SMA = {                  # Pre-SMA (6ma) — action selection, higher-order planning
    'Right_6ma': 43, 'Left_6ma': 223,
}
SCEF = {                     # Supplementary & Cingulate Eye Field
    'Right_SCEF': 42, 'Left_SCEF': 222,
}
PARACENTRAL = {              # Area 5m, 5L — somatosensory association
    'Right_5m': 35,  'Left_5m':  215,
    'Right_5L': 38,  'Left_5L':  218,
}

# ─── Dorsal Premotor Cortex (PMd) ───────────────────────────────────────────────
PMD = {
    'Right_6d': 53, 'Left_6d': 233,   # PMd proper — visually-guided movements
    'Right_6r': 77, 'Left_6r': 257,   # Rostral 6 — conditional motor learning
    'Right_6a': 95, 'Left_6a': 275,   # Anterior 6 — movement preparation
}

# ─── Ventral Premotor Cortex (PMv) ──────────────────────────────────────────────
PMV = {
    'Right_6v': 55,  'Left_6v':  235,   # PMv — grasp, action observation
    'Right_55b': 11, 'Left_55b': 191,   # Area 55b — mouth/larynx (speech-adjacent)
}

# ─── Eye / Attention (optional; exclude by default) ─────────────────────────────
FEF = {'Right_FEF': 9, 'Left_FEF': 189}
PEF = {'Right_PEF': 10, 'Left_PEF': 190}
```

**Effector-specific subsets** following Gordon et al. (2023). Within area 4, somatotopic regions
are not independently parcellated in Glasser360, but we can use spatial location + task contrast
to define functional subregions. The recommended approach:

1. **Task-contrast approach (preferred)**: Use the MSC motor task fMRI (motor_run-01, motor_run-02)
   and compute task-evoked activation separately for foot-tapping, hand-squeezing, and tongue-movement
   runs to identify which parcels are maximally activated by each effector.
2. **Coordinate-based approach (fallback)**: Use center-of-gravity coordinates from
   `HCP-MMP1_UniqueRegionList.csv` (columns `x-cog, y-cog, z-cog` in the atlas's MNI space) to
   assign parcels to effector classes by their superior–inferior position (foot = most superior/medial,
   hand = mid, mouth = most inferior/lateral).

For the initial implementation, define three ROI sets spanning the full motor circuit:

```python
# Motor circuits grouped by presumed effector affinity (Glasser360 0-based indices)
EFFECTOR_ROIS = {
    'foot': {
        # Paracentral — superior/medial M1, S1; 5m; 6mp/SMA
        'cortex': [7, 187,   # 4 L/R
                   52, 232,  # 3a L/R — proprioceptive input
                   35, 215,  # 5m L/R — somatosensory
                   54, 234,  # 6mp (SMA) L/R
                   43, 223], # 6ma (pre-SMA) L/R
    },
    'hand': {
        # Lateral central sulcus M1/S1; PMd/PMv (primary reach/grasp zone)
        'cortex': [7, 187,   # 4 L/R  (shared with foot; will be refined)
                   8, 188,   # 3b L/R
                   50, 230,  # 1 L/R
                   51, 231,  # 2 L/R
                   53, 233,  # 6d (PMd) L/R
                   77, 257,  # 6r L/R
                   55, 235,  # 6v (PMv) L/R
                   95, 275], # 6a L/R
    },
    'mouth': {
        # Ventral/opercular M1/S1; 55b; 6v
        'cortex': [7, 187,   # 4 L/R
                   8, 188,   # 3b L/R
                   11, 191,  # 55b L/R — speech-motor
                   55, 235], # 6v L/R
    },
}
```

### SUIT Cerebellar Parcels (34 regions)

The SUIT atlas (`atlases/SUIT/atl-Anatom_space-MNISym_dseg.nii`) parcellates the cerebellum into
34 functional regions. When the combined `glasser360_SUIT_Thalamus` atlas is used, SUIT parcels
occupy **indices 360–393** (0-based) in the concatenated ROI array, labeled `Cereb_1` through
`Cereb_34`.

Mapping to lobules (from Diedrichsen 2019 / `atl-Anatom` labelling):

```python
# SUIT parcel index within the 34-parcel set → canonical lobule names
# (indices here are 0-based within the 34 SUIT parcels)
SUIT_MOTOR_LOBULES = {
    # Primary motor representation (Stoodley & Schmahmann 2009; Buckner 2011)
    'Lobule_IV_V':    [0, 1],    # anterior lobe — foot/leg primary motor
    'Lobule_VI':      [5, 6],    # superior posterior — hand/finger primary motor + cognitive
    'Lobule_VIII':    [20, 21],  # inferior posterior — secondary somatomotor map (Wiestler 2011)
    # Association / non-motor
    'Lobule_VII':     [10, 11, 12, 13, 14, 15],  # Crus I/II — default mode + frontoparietal
    'Lobule_IX':      [22, 23],  # limbic, somatomotor secondary
    'Lobule_X':       [24],      # vestibular
    'Vermis':         [25, 26, 27, 28, 29, 30, 31, 32, 33],
}
```

In the combined atlas array, add offset 360 to get the global 0-based index:
```python
cereb_motor_indices = [360 + i for sublist in [
    SUIT_MOTOR_LOBULES['Lobule_IV_V'],
    SUIT_MOTOR_LOBULES['Lobule_VI'],
    SUIT_MOTOR_LOBULES['Lobule_VIII'],
] for i in sublist]
```

---

## Shared Utilities (`canonical_utils.py`)

This module centralizes ROI definitions, label helpers, and I/O. All other scripts in
`canonical_circuits/` import from it.

```python
# canonical_utils.py
"""Shared ROI definitions and helpers for canonical-circuits analysis."""

import os
import sys
import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────────────────────────
_CANONICAL_DIR = os.path.dirname(__file__)
_MSC_DIR = os.path.dirname(_CANONICAL_DIR)
sys.path.insert(0, _MSC_DIR)
from msc_paths import (BASE_DIR, ATLAS_DIR, mi_dir, cov_dir,
                        analysis_dir, figures_dir, glasser360_labels_path)
from msc_chow_liu import (load_glasser360_labels, load_node_labels,
                           infer_network, _NETWORK_COLORS,
                           mi_to_graph, chow_liu_tree)

# ── Glasser360 motor/premotor ROI indices (0-based in combined atlas) ────────────
# Right hemisphere: 0-based index = regionID − 201
# Left hemisphere:  0-based index = regionID + 179

PRIMARY_MOTOR_IDX = [7, 187]          # Area 4  (R, L)
PRIMARY_SENS_IDX  = [8, 188,          # Area 3b (R, L) — tactile
                     52, 232,         # Area 3a (R, L) — proprioceptive
                     50, 230,         # Area 1  (R, L)
                     51, 231]         # Area 2  (R, L)
SMA_IDX           = [54, 234]         # Area 6mp — SMA proper
PRE_SMA_IDX       = [43, 223]         # Area 6ma — pre-SMA
SCEF_IDX          = [42, 222]         # SCEF
PARACENTRAL_IDX   = [35, 215, 38, 218]  # 5m, 5L
PMD_IDX           = [53, 233,         # 6d  — PMd proper
                     77, 257,         # 6r
                     95, 275]         # 6a
PMV_IDX           = [55, 235,         # 6v  — PMv proper
                     11, 191]         # 55b — speech-motor

MOTOR_CORTEX_ALL = sorted(set(
    PRIMARY_MOTOR_IDX + PRIMARY_SENS_IDX + SMA_IDX + PRE_SMA_IDX +
    SCEF_IDX + PARACENTRAL_IDX + PMD_IDX + PMV_IDX
))

# ── SUIT cerebellar indices (offset 360 in combined atlas) ───────────────────────
# Motor lobules: IV/V (0,1), VI (5,6), VIII (20,21) within 34-parcel set
CEREB_OFFSET = 360
CEREB_MOTOR_LOCAL = [0, 1, 5, 6, 20, 21]   # lobules IV/V, VI, VIII
CEREB_ALL_LOCAL   = list(range(34))
CEREB_MOTOR_IDX   = [CEREB_OFFSET + i for i in CEREB_MOTOR_LOCAL]
CEREB_ALL_IDX     = [CEREB_OFFSET + i for i in CEREB_ALL_LOCAL]

CC_MOTOR_IDX = sorted(MOTOR_CORTEX_ALL + CEREB_MOTOR_IDX)   # cortico-cerebellar subset
CC_FULL_IDX  = sorted(MOTOR_CORTEX_ALL + CEREB_ALL_IDX)     # with all cerebellar parcels


def load_combined_labels(atlas_subdir='glasser360_SUIT_Thalamus'):
    """Return full node label list for the combined atlas."""
    glasser_labels = load_glasser360_labels()
    n_total = 360 + 34  # Glasser + SUIT (no Thalamus for motor analysis)
    labels = load_node_labels(atlas_subdir, n_rois=n_total)
    return labels


def subset_mi(mi_matrix, roi_indices):
    """Extract a square sub-matrix for the given ROI indices."""
    idx = np.array(roi_indices)
    return mi_matrix[np.ix_(idx, idx)]


def load_mi_matrix(subject, atlas_subdir='glasser360_SUIT_Thalamus',
                   task='motor', num_bins=100, suffix='func-01-10'):
    """Load precomputed MI matrix for a subject.

    Falls back to rest task if motor MI does not exist.
    """
    fname = f'{task}_{num_bins}bins_{suffix}.npy'
    fpath = os.path.join(mi_dir(subject, atlas_subdir), fname)
    if not os.path.exists(fpath):
        # fallback to rest
        fname = f'rest_{num_bins}bins_{suffix}.npy'
        fpath = os.path.join(mi_dir(subject, atlas_subdir), fname)
    return np.load(fpath)


def cc_analysis_dir(submodule):
    """Return analysis output directory for a canonical-circuits submodule."""
    d = os.path.join(analysis_dir(), 'canonical_circuits', submodule)
    os.makedirs(d, exist_ok=True)
    return d


def cc_figures_dir(submodule):
    """Return figures directory for a canonical-circuits submodule."""
    d = os.path.join(figures_dir(), 'canonical_circuits', submodule)
    os.makedirs(d, exist_ok=True)
    return d
```

---

## Script 1 — `corticocerebellar_motor/cc_motor_trees.py`

**Purpose:** Build and visualize CL trees for the cortico-cerebellar motor circuit.

### Step-by-step logic

1. Parse CLI args: subjects, atlas_subdir, task, num_bins, suffix, output flags.
2. For each subject, load MI matrix and subset to `CC_FULL_IDX` (all motor cortex + all cerebellum).
3. Build a CL tree via `mi_to_graph` + `chow_liu_tree`.
4. Save the adjacency matrix to `analysis/canonical_circuits/corticocerebellar_motor/`.
5. Generate three figure types (see below).
6. Build a consensus tree across subjects.

### CLI

```python
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='CL trees for cortico-cerebellar motor circuit.')
    p.add_argument('--subjects', nargs='+',
                   default=[f'MSC{i:02d}' for i in range(1, 11)])
    p.add_argument('--atlas-subdir', default='glasser360_SUIT_Thalamus')
    p.add_argument('--task', default='motor',
                   help='Task label; falls back to rest if motor MI not found.')
    p.add_argument('--num-bins', type=int, default=100)
    p.add_argument('--suffix', default='func-01-10')
    p.add_argument('--roi-set', choices=['motor_only', 'motor_cereb', 'full'],
                   default='motor_cereb',
                   help='motor_only=cortex only; motor_cereb=motor lobules; full=all cereb.')
    p.add_argument('--no-figures', action='store_true')
    return p.parse_args()
```

### CL tree construction

```python
from canonical_utils import (CC_MOTOR_IDX, CC_FULL_IDX, MOTOR_CORTEX_ALL,
                              CEREB_MOTOR_IDX, CEREB_ALL_IDX,
                              load_mi_matrix, load_combined_labels, subset_mi,
                              cc_analysis_dir, cc_figures_dir)
from msc_chow_liu import mi_to_graph, chow_liu_tree, draw_hierarchical_tree

def build_cc_tree(subject, roi_indices, atlas_subdir, task, num_bins, suffix):
    mi_full = load_mi_matrix(subject, atlas_subdir, task, num_bins, suffix)
    mi_sub  = subset_mi(mi_full, roi_indices)           # (n_rois, n_rois)
    G       = mi_to_graph(mi_sub)
    _, tree = chow_liu_tree(G)                          # nx.Graph MST
    adj     = nx.to_numpy_array(tree, weight='weight')  # (n_rois, n_rois)
    return tree, adj, mi_sub
```

### Figure 1 — Hierarchical tree with cortex/cerebellum coloring

Extend `draw_hierarchical_tree` with two-tier color logic: motor cortex parcels colored by their
functional subregion (M1/S1/SMA/PMd/PMv), cerebellar parcels colored by lobule group.

```python
def _cc_node_color(label):
    """Color scheme for cortico-cerebellar motor tree nodes."""
    COLOR = {
        'M1':    '#e41a1c',   # Primary Motor
        'S1':    '#ff7f00',   # Somatosensory (3a/3b/1/2)
        'SMA':   '#377eb8',   # SMA + pre-SMA + SCEF
        'PMd':   '#4daf4a',   # Dorsal premotor (6d/6r/6a)
        'PMv':   '#984ea3',   # Ventral premotor (6v/55b)
        'CbMotor': '#a65628', # Cerebellar lobule IV/V/VI/VIII
        'CbAssoc': '#999999', # Other cerebellar
    }
    l = label.upper()
    if '4' in l and 'CEREB' not in l:          return COLOR['M1']
    if any(x in l for x in ('3A','3B','_1_','_2_','AREA_1','AREA_2')): return COLOR['S1']
    if any(x in l for x in ('6MP','6MA','SCEF')): return COLOR['SMA']
    if any(x in l for x in ('6D','6R','6A')):     return COLOR['PMd']
    if any(x in l for x in ('6V','55B')):         return COLOR['PMv']
    # Cerebellar: label is "Cereb_N"; lobules IV/V = Cereb_1,2; VI = Cereb_6,7; VIII = Cereb_21,22
    if 'CEREB' in l:
        n = int(l.split('_')[1])
        if n in (1, 2, 6, 7, 21, 22):         return COLOR['CbMotor']
        return COLOR['CbAssoc']
    return '#cccccc'

def plot_cc_tree(tree, roi_labels, subject, out_dir):
    fig, ax = plt.subplots(figsize=(14, 9))
    # Find the highest-degree cortical node as root (prefer M1)
    m1_nodes = [i for i, l in enumerate(roi_labels) if '4' in l and 'Cereb' not in l]
    degrees = dict(tree.degree())
    root = max(m1_nodes, key=lambda i: degrees.get(i, 0)) if m1_nodes else \
           max(degrees, key=degrees.get)
    draw_hierarchical_tree(tree, roi_labels, root=root,
                           title=f'{subject} — Cortico-Cerebellar Motor CL Tree', ax=ax)
    # Override colors with cc-specific scheme
    # (pass color_fn parameter — add this parameter to draw_hierarchical_tree)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{subject}_cc_motor_tree.pdf'), bbox_inches='tight')
    plt.close(fig)
```

### Figure 2 — Cortico-cerebellar edge matrix (sorted by region type)

A heatmap of the CL adjacency matrix, with rows/columns sorted as:
M1 → S1 → SMA → PMd → PMv → Cereb_Motor → Cereb_Other.

```python
def plot_cc_adjacency(adj, roi_labels, subject, out_dir):
    order = _sort_order(roi_labels)   # indices sorted by region type
    adj_sorted = adj[np.ix_(order, order)]
    labels_sorted = [roi_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(adj_sorted, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='MI (CL edge weight)')
    ax.set_title(f'{subject} — CC Motor CL Adjacency')
    # Add region-type dividers
    _draw_matrix_dividers(ax, roi_labels, order)
    fig.savefig(os.path.join(out_dir, f'{subject}_cc_motor_adj.pdf'), bbox_inches='tight')
    plt.close(fig)
```

### Figure 3 — Consensus tree + edge vote matrix

Reuse `msc_chow_liu.plot_consensus_tree` logic but restricted to the CC motor ROI subset:

```python
def build_consensus(cl_trees, roi_labels, out_dir, threshold=0.5):
    n = len(roi_labels)
    vote_matrix = np.zeros((n, n))
    for tree in cl_trees.values():
        adj = nx.to_numpy_array(tree, weight=None, nodelist=range(n))
        adj_bin = (adj > 0).astype(float)
        vote_matrix += adj_bin
    vote_matrix /= len(cl_trees)
    np.save(os.path.join(out_dir, 'cc_motor_edge_votes.npy'), vote_matrix)

    # Threshold and build consensus MST
    consensus_mi = vote_matrix.copy()
    consensus_mi[vote_matrix < threshold] = 0
    G_cons = mi_to_graph(consensus_mi)
    _, tree_cons = chow_liu_tree(G_cons)

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_hierarchical_tree(tree_cons, roi_labels, title='Consensus CC Motor Tree', ax=ax)
    fig.savefig(os.path.join(out_dir, 'consensus_cc_motor_tree.pdf'), bbox_inches='tight')
    plt.close(fig)

    # Edge vote heatmap
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(vote_matrix, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction of subjects with edge')
    ax.set_title('CC Motor Edge Vote Matrix')
    fig.savefig(os.path.join(out_dir, 'cc_motor_edge_votes.pdf'), bbox_inches='tight')
    plt.close(fig)
```

---

## Script 2 — `corticocerebellar_motor/cc_motor_clustering.py`

**Purpose:** Hierarchical clustering of the CC motor circuit using Correlation, MI, and CL-derived
distances; produce dendrograms and clustering maps.

### Design notes

This mirrors `msc_clustering.py` but is restricted to the CC motor ROI set. The key addition is a
**clustering map** that projects cluster assignments back onto an anatomical layout for interpretability.

### Clustering pipeline

```python
from msc_clustering import (condensed_mi_dist, condensed_corr_dist, compute_linkages,
                              network_coherence, mean_coherence_vs_k, compute_agreement)

def run_clustering(subjects, roi_indices, atlas_subdir, task, num_bins, suffix,
                   linkage_method='ward', k_max=20):
    all_linkages = {}
    for subject in subjects:
        mi_full  = load_mi_matrix(subject, atlas_subdir, task, num_bins, suffix)
        mi_sub   = subset_mi(mi_full, roi_indices)
        corr_sub = load_corr_matrix(subject, atlas_subdir, task)  # see below
        cl_sub   = load_cl_adjacency(subject, roi_indices)        # from cc_motor_trees output

        matrices = {'mi': mi_sub, 'corr': corr_sub, 'cl': cl_sub}
        linkages = compute_linkages(matrices, linkage_method=linkage_method)
        all_linkages[subject] = linkages
    return all_linkages
```

Loading covariance/correlation matrix (from `msc_covariance.py` output):

```python
def load_corr_matrix(subject, atlas_subdir, task):
    fname = f'{task}_100bins.npy'
    fpath = os.path.join(cov_dir(subject, atlas_subdir), fname)
    if not os.path.exists(fpath):
        fpath = fpath.replace('motor', 'rest')
    corr = np.load(fpath)
    return corr
```

### Figure 4 — Colored dendrograms (three methods)

Reuse `msc_clustering.plot_colored_dendrogram` with a CC-specific `leaf_color_fn`:

```python
def cc_leaf_colors(roi_labels):
    """Return a color per leaf for the CC motor circuit dendrogram."""
    return [_cc_node_color(l) for l in roi_labels]
```

Output: `figures/canonical_circuits/corticocerebellar_motor/{subject}_cc_dendrograms.pdf`
(3-panel: Correlation | MI | CL, leaves colored by region type)

### Figure 5 — Tanglegrams

Reuse `msc_clustering.plot_tanglegram` for the three method pairs.

Output: `figures/canonical_circuits/corticocerebellar_motor/{subject}_cc_tanglegram_{m1}_vs_{m2}.pdf`

### Figure 6 — Clustering map: cerebellum lobule × cortical region co-cluster heatmap

A novel figure not in `msc_clustering.py`: for each cut level k, compute the fraction of
cortex–cerebellum pairs assigned to the same cluster. This directly tests whether the CL tree
respects the known cortico-cerebellar somatotopic map.

```python
def plot_coassignment_map(linkages, roi_labels, k, out_dir, subject):
    """
    For each cortical ROI × cerebellar ROI pair, fraction assigned to same cluster at cut k.
    Rows = cortical regions (sorted M1/S1/SMA/PMd/PMv).
    Cols = cerebellar lobules (sorted IV/V → VI → VIII → Crus I/II → others).
    """
    from scipy.cluster.hierarchy import fcluster

    cortex_idx  = [i for i, l in enumerate(roi_labels) if 'Cereb' not in l]
    cereb_idx   = [i for i, l in enumerate(roi_labels) if 'Cereb' in l]
    cortex_lbls = [roi_labels[i] for i in cortex_idx]
    cereb_lbls  = [roi_labels[i] for i in cereb_idx]

    maps = {}
    for method, Z in linkages.items():
        labels_k = fcluster(Z, k, criterion='maxclust')
        # Co-assignment matrix: cortex rows × cerebellum cols
        co = np.zeros((len(cortex_idx), len(cereb_idx)))
        for ci, cx in enumerate(cortex_idx):
            for cj, cr in enumerate(cereb_idx):
                co[ci, cj] = float(labels_k[cx] == labels_k[cr])
        maps[method] = co

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (method, co) in zip(axes, maps.items()):
        im = ax.imshow(co, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'{method.upper()} (k={k})')
        ax.set_yticks(range(len(cortex_lbls)))
        ax.set_yticklabels(cortex_lbls, fontsize=6)
        ax.set_xticks(range(len(cereb_lbls)))
        ax.set_xticklabels(cereb_lbls, rotation=90, fontsize=6)
        plt.colorbar(im, ax=ax)

    fig.suptitle(f'{subject} — Cortico-Cerebellar Co-Cluster Map')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{subject}_coassignment_k{k}.pdf'), bbox_inches='tight')
    plt.close(fig)
```

### Figure 7 — Network coherence across k (CC-specific)

Adapted from `msc_clustering.plot_network_coherence`. Networks are defined as the 5 functional
subgroups (M1+S1, SMA, PMd, PMv, Cereb_Motor) so that coherence at each k measures how well each
method groups anatomically related ROIs.

```python
def cc_network_membership(roi_labels):
    """Return a list of network names parallel to roi_labels."""
    groups = []
    for l in roi_labels:
        col = _cc_node_color(l)
        if col == '#e41a1c' or col == '#ff7f00': groups.append('M1_S1')
        elif col == '#377eb8':                    groups.append('SMA')
        elif col == '#4daf4a':                    groups.append('PMd')
        elif col == '#984ea3':                    groups.append('PMv')
        elif col == '#a65628':                    groups.append('Cereb_Motor')
        else:                                     groups.append('Cereb_Assoc')
    return groups
```

---

## Script 3 — `motor_cortex/mc_effector_trees.py`

**Purpose:** Build separate CL trees for foot, hand, and mouth motor circuits; compare their
hierarchical structure across subjects.

### Design notes

Following Gordon et al. (2023), each effector has distinct cortical topology. This script:

1. Defines three overlapping ROI sets from `EFFECTOR_ROIS` in `canonical_utils.py`.
2. For each effector, computes a sub-matrix of the full MI matrix restricted to that effector's
   cortical ROIs + motor cerebellar lobules (lobule IV/V for foot, VI for hand, VIII for secondary).
3. Builds per-effector CL trees.
4. Produces a 3-column figure comparing trees side by side.
5. Computes inter-effector Jaccard similarity to measure how much the trees overlap.

### Effector tree construction

```python
from canonical_utils import EFFECTOR_ROIS, CEREB_MOTOR_IDX

def build_effector_trees(subject, atlas_subdir, task, num_bins, suffix):
    mi_full = load_mi_matrix(subject, atlas_subdir, task, num_bins, suffix)
    trees = {}
    for effector, spec in EFFECTOR_ROIS.items():
        roi_idx = sorted(set(spec['cortex'] + CEREB_MOTOR_IDX))
        mi_sub  = subset_mi(mi_full, roi_idx)
        G, tree = chow_liu_tree(mi_to_graph(mi_sub))
        trees[effector] = {'tree': tree, 'roi_idx': roi_idx, 'mi': mi_sub}
    return trees
```

### Figure 8 — Three-panel effector CL tree comparison

```python
def plot_effector_trees(effector_trees, roi_labels_fn, subject, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    for ax, (effector, data) in zip(axes, effector_trees.items()):
        tree     = data['tree']
        roi_idx  = data['roi_idx']
        e_labels = [roi_labels_fn(i) for i in roi_idx]
        root     = _best_root(tree, e_labels)
        draw_hierarchical_tree(tree, e_labels, root=root,
                               title=f'{subject} — {effector.capitalize()} Motor Tree', ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{subject}_effector_trees.pdf'), bbox_inches='tight')
    plt.close(fig)
```

### Figure 9 — Inter-effector Jaccard matrix

Quantifies how similar the trees are across effectors and across subjects. Uses the same
`edge_set` + `jaccard` helpers from `noise_sim.py`:

```python
def edge_set(tree):
    return frozenset(frozenset(e) for e in tree.edges())

def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0

def plot_effector_jaccard(effector_trees_by_subject, out_dir):
    effectors = list(next(iter(effector_trees_by_subject.values())).keys())
    subjects  = list(effector_trees_by_subject.keys())
    # Average Jaccard across subjects for each effector pair
    J = np.zeros((len(effectors), len(effectors)))
    for subj_trees in effector_trees_by_subject.values():
        for i, e1 in enumerate(effectors):
            for j, e2 in enumerate(effectors):
                J[i, j] += jaccard(edge_set(subj_trees[e1]['tree']),
                                    edge_set(subj_trees[e2]['tree']))
    J /= len(subjects)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(J, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(effectors))); ax.set_xticklabels(effectors)
    ax.set_yticks(range(len(effectors))); ax.set_yticklabels(effectors)
    for i in range(len(effectors)):
        for j in range(len(effectors)):
            ax.text(j, i, f'{J[i,j]:.2f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax, label='Mean Jaccard similarity')
    ax.set_title('Inter-Effector Tree Similarity')
    fig.savefig(os.path.join(out_dir, 'effector_jaccard.pdf'), bbox_inches='tight')
    plt.close(fig)
```

---

## Script 4 — `motor_cortex/mc_effector_clustering.py`

**Purpose:** Clustering analysis for each effector's ROI set; produces dendrograms and
somatotopic layout figures.

### Figure 10 — Somatotopic co-cluster heatmap

Extends the co-assignment map idea from `cc_motor_clustering.py`. For the motor cortex only
(no cerebellum), sort ROIs by their y-coordinate (superior–inferior in MNI space from
`HCP-MMP1_UniqueRegionList.csv`), which approximates the somatotopic foot–hand–mouth axis.
Then plot which ROIs cluster together at cut k=3.

```python
def load_glasser_coords():
    """Return dict {region_name: (x, y, z)} from HCP-MMP1_UniqueRegionList.csv."""
    csv_path = os.path.join(ATLAS_DIR, 'glasser360', 'HCP-MMP1_UniqueRegionList.csv')
    df = pd.read_csv(csv_path, header=None,
                     names=['regionName','regionLongName','regionIdLabel','LR','region',
                            'Lobe','cortex','regionID','Cortex_ID','x','y','z','vol'])
    return {row.regionName: (row.x, row.y, row.z) for _, row in df.iterrows()}

def plot_somatotopic_clustering(linkage_Z, roi_labels, k, subject, out_dir):
    """
    Sort motor-cortex ROIs by MNI z-coordinate (superior–inferior) and display
    cluster assignments at k levels as a color strip along the somatotopic axis.
    """
    from scipy.cluster.hierarchy import fcluster
    coords = load_glasser_coords()
    # Get z-coordinate for each ROI (use short region name, strip LH/RH prefix)
    def get_z(label):
        # label is like 'Right_4' or 'Left_6d'
        hemi, area = label.split('_', 1)
        key = area + ('_R' if hemi == 'Right' else '_L')
        return coords.get(key, (0, 0, 100))[2]  # z = inf-sup axis
    z_coords = np.array([get_z(l) for l in roi_labels])
    sort_order = np.argsort(z_coords)[::-1]   # superior first

    cluster_labels = fcluster(linkage_Z, k, criterion='maxclust')
    sorted_clusters = cluster_labels[sort_order]
    sorted_labels   = [roi_labels[i] for i in sort_order]
    sorted_z        = z_coords[sort_order]

    cmap = plt.cm.get_cmap('tab10', k)
    fig, ax = plt.subplots(figsize=(2, 10))
    for row_i, (cl, label, z) in enumerate(zip(sorted_clusters, sorted_labels, sorted_z)):
        ax.barh(row_i, 1, color=cmap(cl - 1))
        ax.text(1.05, row_i, f'{label} (z={z:.0f})', va='center', fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title(f'{subject} Somatotopic CL Clusters (k={k})')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{subject}_somatotopic_k{k}.pdf'), bbox_inches='tight')
    plt.close(fig)
```

---

## Figure Assembly (`cc_motor_figures.py` / `mc_effector_figures.py`)

These thin wrapper scripts load saved analysis outputs (adjacency NPY, linkage NPZ) and regenerate
all figures without recomputing, following the pattern of `code/simulations/_regen_figures.py`.

```python
# cc_motor_figures.py
def main():
    args = parse_args()
    for subject in args.subjects:
        adj = np.load(os.path.join(cc_analysis_dir('corticocerebellar_motor'),
                                   f'{subject}_cc_motor_adj.npy'))
        linkages = np.load(os.path.join(cc_analysis_dir('corticocerebellar_motor'),
                                         f'{subject}_cc_linkages.npz'), allow_pickle=True)
        roi_labels = load_combined_labels(args.atlas_subdir)
        roi_subset = [roi_labels[i] for i in CC_FULL_IDX]
        out = cc_figures_dir('corticocerebellar_motor')

        tree = _adj_to_tree(adj, roi_subset)
        plot_cc_tree(tree, roi_subset, subject, out)
        plot_cc_adjacency(adj, roi_subset, subject, out)
        plot_coassignment_map(dict(linkages), roi_subset, k=args.k, out_dir=out,
                              subject=subject)
```

---

## Implementation Order

1. **`canonical_utils.py`** — ROI index constants, `load_mi_matrix`, `subset_mi`, path helpers.
   No external dependencies beyond existing `msc_paths` / `msc_chow_liu`.

2. **`cc_motor_trees.py`** — depends on `canonical_utils`. Run after MI matrices are computed with
   `msc_mutual_info.py` (motor task or resting state). Produces the adjacency NPY files needed by
   `cc_motor_clustering.py`.

3. **`cc_motor_clustering.py`** — depends on adjacency NPY from step 2 and on covariance NPY from
   `msc_covariance.py`.

4. **`cc_motor_figures.py`** — regeneration wrapper; depends on steps 2–3.

5. **`mc_effector_trees.py`** — same MI inputs as step 2; can run in parallel.

6. **`mc_effector_clustering.py`** — depends on `mc_effector_trees.py` outputs.

7. **`mc_effector_figures.py`** — regeneration wrapper.

---

## Data Requirements

| Data | Source script | Output path |
|---|---|---|
| Motor task MI matrices (`.npy`) | `msc_mutual_info.py --task motor` | `output/mutual_information/{subject}/glasser360_SUIT_Thalamus/motor_100bins_func-01-10.npy` |
| Covariance matrices (`.npy`) | `msc_covariance.py` | `output/covariance/{subject}/glasser360_SUIT_Thalamus/motor_100bins.npy` |
| Time series (`.csv`) | `msc_extract_timeseries.py` | `output/roi_time_series/{subject}/{session}/glasser360_SUIT_Thalamus/...` |

If motor-task MI matrices are not yet computed, `load_mi_matrix` will fall back to resting-state MI.
The `--task` flag should be set to `motor` once motor-task extraction is complete.

---

## Key Design Decisions

1. **Shared `canonical_utils.py`** avoids duplicating ROI index logic across the four scripts.
   Any future canonical circuit module (e.g., language, visual) imports from the same file.

2. **Reuse existing functions** (`chow_liu_tree`, `draw_hierarchical_tree`, `compute_linkages`,
   `plot_tanglegram`, `compute_agreement`) from `msc_chow_liu.py` and `msc_clustering.py` rather
   than copying them. The `canonical_circuits` scripts are thin orchestration layers.

3. **ROI indices are global 0-based indices** into the combined `glasser360_SUIT_Thalamus` atlas
   array (0–359 = Glasser, 360–393 = SUIT). This is consistent with how `msc_chow_liu.py` and
   `msc_mutual_info.py` store data.

4. **Motor task vs. resting state**: Motor-task MI should be preferred for effector-specific
   analysis since resting-state connectivity is less effector-discriminating. The fallback to rest
   ensures scripts run even when motor MI is not yet computed.

5. **Effector ROIs overlap** (area 4 appears in foot, hand, and mouth sets) because Glasser360
   does not sub-parcellate M1 by effector. The analysis is still meaningful — the *tree structure*
   within each effector set will differ even with shared nodes because the MI weights to cerebellar
   and premotor partners differ by effector.

6. **`draw_hierarchical_tree` needs a `color_fn` parameter** to support the CC-specific coloring
   scheme. Add `color_fn=None` (defaults to `infer_network`) to its signature in `msc_chow_liu.py`
   so that `canonical_circuits` scripts can pass `_cc_node_color` without modifying the core script.

7. **No SUIT label file available**: `atlases/SUIT/` contains only the NIfTI parcellation, not the
   `.tsv` LUT. Labels for SUIT parcels are generated programmatically as `Cereb_1` through `Cereb_34`,
   matching the convention in `msc_chow_liu.load_node_labels()`. Mapping `Cereb_N` to lobule names
   requires the `SUIT_MOTOR_LOBULES` dict defined in `canonical_utils.py`.

---

## Extensions (Future Work)

- **Nilearn brain-surface overlay**: Project cluster assignments back onto the MNI surface using
  `nilearn.plotting.plot_surf_stat_map` with the Glasser360 volumetric atlas as a label image.
- **Thalamic relay nodes**: Add Morel thalamic nuclei (VLa, VLpd/pv, VPLa for motor relay) from
  `atlases/MorelAtlasMNI152/` to the CC motor circuit analysis, creating a
  cortex → thalamus → cerebellum triad.
- **Task vs. rest comparison**: Run `cc_motor_trees.py` with both `--task motor` and `--task rest`
  and compute edge-level differences in MI to identify task-specific circuit elements.
- **Subject-level clustering maps**: Use `plot_somatotopic_clustering` across all 10 MSC subjects
  to quantify individual variability in motor circuit clustering, tying into research question #7.
