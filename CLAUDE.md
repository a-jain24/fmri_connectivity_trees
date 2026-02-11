# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neuroscience research codebase for building and analyzing **Chow-Liu trees** (maximum spanning trees) as sparse graphical models of brain functional connectivity from fMRI data. The core workflow: extract ROI time series → compute mutual information matrices → construct Chow-Liu trees → visualize and compare connectivity patterns.

## Environment Setup

```bash
conda env create -f environments/dynamric1.yml
conda activate dynamric
```

Python 3.11. Key libraries: nilearn, nibabel, numpy, scipy, scikit-learn, torch, networkx, plotly, matplotlib, pandas.

## Architecture

### Datasets
Three fMRI datasets are processed with parallel directory structures under `code/functional_connectivity/` and `code/jupyter_notebooks/`:
- **Midnight Scan Club (MSC)**: 10 subjects (MSC01–MSC10), 10 sessions each. Primary dataset.
- **ABIDE**: Autism Brain Imaging Data Exchange.
- **LISTEN**: Audio processing study.

### Brain Atlases (`atlases/`)
Parcellations used to define ROIs: **Glasser360** (cortex), **SUIT** (cerebellum), **Thalamus** (Morel nuclei), **Brainstem**. Scripts often combine multiple atlases (e.g., `glasser360_SUIT_Thalamus`).

### Processing Pipeline (`code/functional_connectivity/<dataset>/`)
Each dataset has standalone Python scripts for each processing step:
- `*_extract_timeseries.py` — Uses nilearn maskers to extract ROI time series from NIfTI files. Handles confound removal and standardization.
- `*_mutual_info.py` — Discretizes time series, computes joint probabilities and mutual information between all ROI pairs. Uses PyTorch with optional CUDA acceleration.
- `*_covariance.py` — Computes covariance/correlation matrices across sessions.

Computed results (`.npy`, `.csv`) are saved under `code/functional_connectivity/<dataset>/output/`.

### Analysis Notebooks (`code/jupyter_notebooks/<dataset>/`)
Jupyter notebooks for tree construction, visualization, and figure generation. Key notebook: `msc_chow_liu.ipynb` (and root-level `msc_chow_liu.ipynb`) — contains `chow_liu_tree()`, `mi_to_graph()`, `cl_tree_traversal()`, tree visualization, and selective network analysis functions.

### Multi-Environment Path Convention
All scripts define multiple base directory paths for different machines (local Mac, lab workstation, UTD cluster, BioHPC). Scripts detect which environment they're running on. The active base dir for this machine is:
```
/Users/ajjain/Downloads/Code/fmri_connectivity_trees
```

## Key Functions (in `msc_chow_liu.ipynb`)
- `load_data()` — Load precomputed MI matrices for subjects
- `mi_to_graph()` — Convert MI matrix to NetworkX weighted graph
- `chow_liu_tree()` — Build maximum spanning tree (Chow-Liu tree) from MI graph
- `cl_tree_traversal()` — Reconstruct full MI matrix from tree edges
- `selective_chow_liu_tree()` — Build trees for specific brain network subsets
- `draw_hierarchical_tree()` — Visualize trees with region-based coloring
