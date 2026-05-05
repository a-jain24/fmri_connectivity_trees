# Synthetic dataset `20260421_102949_grid`

## How this data was generated

This run was produced by **`code/simulations/grid_sim.py`** (whole-brain–style network simulations using the TVB **Generic2dOscillator** model, integrated with **Heun stochastic** stepping on **PyTorch**, see `code/simulations/dynamics_simulation.py`).

Recorded metadata (`timeseries/params.json`):

| Field | Value |
|--------|--------|
| Timestamp | 2026-04-21T10:29:49 |
| Git commit | `c2b2842f` |
| Device | CUDA |
| Performance | `fast` (float32; TF32 matmul allowed on CUDA) |
| Base seed | `42` |

### Parameter grid

The script sweeps a **3×3×3** grid of structural settings:

- **N** (number of nodes): `10`, `20`, `50`
- **G** (global coupling strength): `0.1`, `0.3`, `0.5`
- **Connectivity** (one random graph per cell, mode fixed by spec):
  - `erdos_renyi:0.1` → Erdős–Rényi undirected graph with edge probability **0.1** (saved under folder tag `er_0p10`)
  - `hierarchical` → BFS-style hierarchical tree with default branching **3** (`hier_b3`)
  - `hierarchical:2` → same with branching **2** (`hier_b2`)

Within each **(N, G, connectivity)** cell, the code draws **one** ground-truth adjacency/weight matrix **C**, **one** assignment of per-edge coupling types, and **one** draw of **heterogeneous** oscillator parameters per node (TVB defaults with Gaussian variability). Those are **shared** across all “individuals” in that cell.

### Noise (“individuals”)

There are **5** individuals per cell, indexed only by additive noise amplitude **D** (no new graph or new node parameters per individual). Noise levels are:

`1e-5`, `5e-5`, `2e-4`, `8e-4`, `3e-3`

### Coupling and integration

- **Edge mode:** `uniform` — every active edge uses the same coupling type.
- **Coupling type:** `linear` (standard linear coupling into the target node).
- **Simulation length:** `10000` ms; internal step **`dt = 0.5`** ms (fixed in `grid_sim.py`).
- **Delays:** no tract lengths — effectively **1-step** coupling delays (see `run_sim` with `tract_lengths=None`).

### Random seeds

Each grid cell uses **`cell_seed = 42 + cell_index`** (0-based order: nested loops over **sorted N**, then **G**, then connectivity specs). That makes networks and heterogeneous parameters **reproducible** and **comparable across D** within the same cell.

### Output layout

Under `timeseries/`:

```text
timeseries/N{N}/G{G}/<conn_tag>/individual_{k}_D{D}.npz
```

Example tags: `er_0p10`, `hier_b3`, `hier_b2`. Each `.npz` stores the monitor time axis, temporally averaged **V** traces, **C**, **G**, **D**, coupling-type matrix, node parameters, **dt**, and **simlen** (see `save_data` in `dynamics_simulation.py`).

**Note:** This checkout of the folder currently contains **`timeseries/params.json`** only; large `.npz` outputs may live elsewhere or were removed after the run.
