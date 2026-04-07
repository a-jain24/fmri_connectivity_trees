"""
grid_sim.py — Grid sweep over (N, G, connectivity, individual) dimensions.

For each (N, G, connectivity) cell a single ground-truth network is drawn and
heterogeneous node parameters are sampled once.  That same network and those
same parameters are reused across all individuals; only the noise amplitude D
differs between individuals.  This isolates the effect of noise from structural
variation, matching the design of noise_sim.py but across a full parameter grid.

No analysis is performed — only raw timeseries are saved so the grid can be
re-analysed with different methods later.

Directory layout:
    datasets/synthetic/<run_id>/
        params.json
        timeseries/
            N{n}/G{g:.2f}/{conn_tag}/
                individual_{k+1}_D{D:.0e}.npz   — V, conn, coupling_types, D, params

Connectivity specs  (--connectivity flag):
    tree                     — uniformly random labeled tree
    hierarchical             — BFS branching tree (default branching=3)
    hierarchical:2           — BFS branching tree with branching=2
    erdos_renyi:0.1          — Erdős–Rényi with density 0.1
    erdos_renyi:0.5          — Erdős–Rényi with density 0.5
    dense                    — legacy asymmetric ~75% fill

Usage:
    # Quick test
    python code/simulations/grid_sim.py \\
        --N-values 10 15 \\
        --G-values 0.1 0.3 \\
        --connectivity erdos_renyi:0.1 erdos_renyi:0.5 tree \\
        --noise-levels 1e-5 1e-4 1e-3 \\
        --simlen 5000 --seed 42

    # Full grid
    python code/simulations/grid_sim.py \\
        --N-values 10 20 50 \\
        --G-values 0.1 0.3 0.5 \\
        --connectivity erdos_renyi:0.1 erdos_renyi:0.3 tree hierarchical hierarchical:2 \\
        --noise-levels 1e-5 5e-5 2e-4 8e-4 3e-3 \\
        --simlen 600000 --seed 42 --device cuda
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_utils import detect_device, make_run_id, save_ts_params, timeseries_dir
from dynamics_simulation import (
    NAME_TO_TYPE,
    build_connectivity,
    build_coupling_types,
    build_heterogeneous_params,
    build_symmetric_coupling_types,
    run_sim,
    save_data,
)

# TVB Generic2dOscillator defaults and per-node variability sigmas
_DEFAULTS = {
    "a": 0.0, "b": -10.0, "c": 0.0, "d": 0.02, "e": 3.0,
    "f": 1.0,  "g": 0.0,  "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "tau": 1.0,
}
_SIGMAS = {
    "a": 0.05, "b": 0.5, "c": 0.01, "d": 0.005, "e": 0.3,
    "f": 0.1,  "g": 0.05, "alpha": 0.1, "beta": 0.1, "gamma": 0.1, "tau": 0.15,
}

_SYMMETRIC_MODES = ("erdos_renyi", "tree", "hierarchical")


# ---------------------------------------------------------------------------
# Connectivity spec parsing
# ---------------------------------------------------------------------------

def parse_conn_specs(specs: list[str]) -> list[tuple[str, dict, str]]:
    """Parse connectivity spec strings into (mode, kwargs, dir_tag) tuples.

    Accepted formats:
        tree                  → mode='tree',         kwargs={},              tag='tree'
        hierarchical          → mode='hierarchical', kwargs={branching:3},   tag='hier_b3'
        hierarchical:2        → mode='hierarchical', kwargs={branching:2},   tag='hier_b2'
        erdos_renyi:0.1       → mode='erdos_renyi',  kwargs={density:0.1},   tag='er_0p10'
        dense                 → mode='dense',         kwargs={},              tag='dense'
    """
    results = []
    for spec in specs:
        parts = spec.split(":")
        mode = parts[0]
        if mode == "erdos_renyi":
            density = float(parts[1]) if len(parts) > 1 else 0.5
            tag = f"er_{density:.2f}".replace(".", "p")
            results.append((mode, {"density": density}, tag))
        elif mode == "hierarchical":
            branching = int(parts[1]) if len(parts) > 1 else 3
            tag = f"hier_b{branching}"
            results.append((mode, {"branching": branching}, tag))
        elif mode == "tree":
            results.append(("tree", {}, "tree"))
        elif mode == "dense":
            results.append(("dense", {}, "dense"))
        else:
            raise ValueError(
                f"Unknown connectivity spec '{spec}'. "
                "Expected: tree, hierarchical[:branching], erdos_renyi:density, dense"
            )
    return results


# ---------------------------------------------------------------------------
# Single cell simulation
# ---------------------------------------------------------------------------

def simulate_cell(N, G, mode, conn_kwargs, conn_tag, noise_levels, args, cell_seed):
    """Simulate all individuals for one (N, G, connectivity) cell.

    The ground-truth connectivity and heterogeneous node parameters are drawn
    once from cell_seed and shared across all individuals.  Only D varies.
    """
    rng = np.random.default_rng(cell_seed)
    np.random.seed(cell_seed)   # for build_heterogeneous_params (uses np.random.normal)

    # --- Shared ground truth ---
    C = build_connectivity(N, mode, rng, **conn_kwargs)

    uniform_type  = NAME_TO_TYPE.get(args.coupling_type)  if args.coupling_type  else None
    allowed_types = [NAME_TO_TYPE[t] for t in args.coupling_types] if args.coupling_types else None

    if mode in _SYMMETRIC_MODES:
        coupling_types = build_symmetric_coupling_types(
            C, args.edge_mode, rng,
            uniform_type=uniform_type, allowed_types=allowed_types,
        )
    else:
        coupling_types = build_coupling_types(
            C, args.edge_mode, rng,
            uniform_type=uniform_type, allowed_types=allowed_types,
        )

    params = build_heterogeneous_params(N, _DEFAULTS, _SIGMAS)

    n_edges = int(np.count_nonzero(np.triu(C, k=1)))
    print(f"  Ground truth: {n_edges} edges  "
          f"(N={N}, G={G}, conn={conn_tag}, seed={cell_seed})")

    # --- Per-individual simulations ---
    cell_dir = os.path.join(
        timeseries_dir(args.run_id),
        f"N{N}",
        f"G{G:.2f}",
        conn_tag,
    )
    os.makedirs(cell_dir, exist_ok=True)

    for k, D in enumerate(noise_levels):
        print(f"    Individual {k+1}/{len(noise_levels)}  D={D:.1e} ...", end=" ", flush=True)
        t0 = time.time()

        t, V = run_sim(
            C,
            G=G,
            D=D,
            coupling_types=coupling_types,
            tract_lengths=None,
            params=params,
            dt=0.5,
            simlen=args.simlen,
            device=args.device,
        )

        elapsed = time.time() - t0

        if not np.isfinite(V).all():
            print(f"WARNING: non-finite values (unstable?)", end=" ")

        out_path = os.path.join(cell_dir, f"individual_{k+1}_D{D:.0e}.npz")
        save_data(
            out_path,
            t, V, C, G, D, coupling_types,
            params=params, dt=0.5, simlen=args.simlen,
        )
        print(f"done ({elapsed:.1f}s)  V∈[{V.min():.3f}, {V.max():.3f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grid sweep over (N, G, connectivity) cells. "
            "Within each cell, one ground-truth network and one set of heterogeneous "
            "node parameters are drawn and shared across all individuals; only noise D varies."
        )
    )
    parser.add_argument(
        "--N-values", type=int, nargs="+", default=[10, 20],
        help="Node counts to sweep (default: 10 20)",
    )
    parser.add_argument(
        "--G-values", type=float, nargs="+", default=[0.1, 0.3, 0.5],
        help="Global coupling strengths to sweep (default: 0.1 0.3 0.5)",
    )
    parser.add_argument(
        "--connectivity", nargs="+",
        default=["erdos_renyi:0.1", "erdos_renyi:0.5", "tree"],
        metavar="SPEC",
        help=(
            "Connectivity specs to sweep. Each is TYPE[:PARAM], e.g.: "
            "tree  hierarchical  hierarchical:2  erdos_renyi:0.1  dense  "
            "(default: erdos_renyi:0.1 erdos_renyi:0.5 tree)"
        ),
    )
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3],
        metavar="D",
        help="Noise amplitude D for each individual (default: 1e-5 1e-4 1e-3)",
    )
    parser.add_argument(
        "--simlen", type=float, default=600_000,
        help="Simulation length in ms (default: 600000)",
    )
    parser.add_argument(
        "--edge-mode", choices=["random", "uniform"], default="random",
        help="Coupling type assignment per edge (default: random)",
    )
    parser.add_argument(
        "--coupling-type", choices=list(NAME_TO_TYPE.keys()), default=None,
        help="Coupling type for --edge-mode uniform",
    )
    parser.add_argument(
        "--coupling-types", nargs="+", choices=list(NAME_TO_TYPE.keys()), default=None,
        help="Restrict random coupling to this subset of types",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device: cpu / cuda / mps (default: auto)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help=(
            "Run identifier used as directory name under datasets/synthetic/. "
            "Auto-generated as YYYYMMDD_HHMMSS_grid if omitted."
        ),
    )
    args = parser.parse_args()

    if args.edge_mode == "uniform" and args.coupling_type is None:
        parser.error("--coupling-type is required when --edge-mode uniform")

    args.device = detect_device(args.device)

    if args.run_id is None:
        args.run_id = make_run_id("grid")
    print(f"Run ID: {args.run_id}")

    conn_specs = parse_conn_specs(args.connectivity)

    # Summary
    n_cells = len(args.N_values) * len(args.G_values) * len(conn_specs)
    n_sims  = n_cells * len(args.noise_levels)
    print(f"\nGrid dimensions:")
    print(f"  N values    : {args.N_values}")
    print(f"  G values    : {args.G_values}")
    print(f"  Connectivity: {[tag for _, _, tag in conn_specs]}")
    print(f"  Individuals : {len(args.noise_levels)}  (D = {args.noise_levels})")
    print(f"  Total cells : {n_cells}   Total simulations: {n_sims}")
    print(f"  Timeseries  → {timeseries_dir(args.run_id)}\n")

    # Serialise args for params.json (convert lists to plain lists for JSON)
    args_dict = vars(args).copy()
    args_dict["connectivity"] = args.connectivity   # raw spec strings
    save_ts_params(args.run_id, "grid_sim.py", args_dict)

    # --- Main grid loop ---
    cell_idx = 0
    t_total = time.time()

    for N in sorted(args.N_values):
        for G in args.G_values:
            for mode, conn_kwargs, conn_tag in conn_specs:
                print(f"\n[{cell_idx+1}/{n_cells}] N={N}  G={G}  conn={conn_tag}")
                cell_seed = args.seed + cell_idx
                simulate_cell(N, G, mode, conn_kwargs, conn_tag,
                              args.noise_levels, args, cell_seed)
                cell_idx += 1

    elapsed_total = time.time() - t_total
    print(f"\nDone — {n_sims} simulations in {elapsed_total/60:.1f} min")
    print(f"Timeseries saved to: {timeseries_dir(args.run_id)}")


if __name__ == "__main__":
    main()
