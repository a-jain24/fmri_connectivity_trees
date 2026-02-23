"""
sweep_sim.py — Sweep over N values, run multiple trials per N, collect and average metrics.

Usage:
    python sweep_sim.py \
        --N-values 5 10 15 20 25 --trials 10 --simlen 600000 \
        --connectivity dense --edge-mode random --no-delays \
        --outdir output/sweep_dense --seed 42

    # Quick test
    python code/simulations/sweep_sim.py \
        --N-values 5 10 --trials 2 --simlen 5000 \
        --connectivity tree --no-delays --seed 42 \
        --outdir /tmp/sweep_test
"""

import argparse
import json
import os
import time

import numpy as np

from dynamics_simulation import (
    NAME_TO_TYPE,
    build_connectivity,
    build_coupling_types,
    build_heterogeneous_params,
    build_symmetric_coupling_types,
    run_sim,
    save_data,
)
from analyze_sim import (
    compute_chow_liu,
    compute_fc,
    compute_mi,
    density_threshold_matrix,
    evaluate,
)


# Methods and metrics tracked
METHODS = [
    "fc_significance", "fc_density", "fc_matched",
    "mi_raw", "mi_density", "mi_matched",
    "cl_adjacency",
]
METRIC_KEYS = [
    "pearson", "pearson_pvalue", "f1",
    "sensitivity", "specificity", "precision", "mse",
]

# Map from metric key to the key in evaluate()'s output dict
_EVAL_KEY_MAP = {
    "pearson": "pearson_correlation",
    "pearson_pvalue": "pearson_pvalue",
    "f1": "f1",
    "sensitivity": "sensitivity",
    "specificity": "specificity",
    "precision": "precision",
    "mse": "mean_squared_error",
}

# Default heterogeneous parameter sigmas (same as dynamics_simulation.py main)
_DEFAULTS = {
    "a": 0.0, "b": -10.0, "c": 0.0, "d": 0.02, "e": 3.0,
    "f": 1.0, "g": 0.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "tau": 1.0,
}
_SIGMAS = {
    "a": 0.05, "b": 0.5, "c": 0.01, "d": 0.005, "e": 0.3,
    "f": 0.1, "g": 0.05, "alpha": 0.1, "beta": 0.1, "gamma": 0.1, "tau": 0.15,
}


def run_single_trial(N, trial_idx, args, trial_dir):
    """Run one simulation + analysis trial. Returns metrics dict."""
    seed = args.seed + trial_idx if args.seed is not None else None
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    # --- Connectivity ---
    C = build_connectivity(N, args.connectivity, rng,
                           extra_edges=args.extra_edges,
                           branching=args.branching,
                           density=args.density)

    tract_lengths = None
    if not args.no_delays:
        tract_lengths = rng.uniform(10.0, 150.0, size=(N, N))
        tract_lengths = (tract_lengths + tract_lengths.T) / 2
        np.fill_diagonal(tract_lengths, 0.0)

    # --- Coupling types ---
    uniform_type = NAME_TO_TYPE.get(args.coupling_type) if args.coupling_type else None
    if args.connectivity in ("erdos_renyi", "sparse", "tree", "hierarchical"):
        coupling_types = build_symmetric_coupling_types(
            C, args.edge_mode, rng, uniform_type=uniform_type
        )
    else:
        coupling_types = build_coupling_types(
            C, args.edge_mode, rng, uniform_type=uniform_type
        )

    # --- Heterogeneous params ---
    params = build_heterogeneous_params(N, _DEFAULTS, _SIGMAS)

    # --- Simulate ---
    t, V = run_sim(
        C, G=args.G, D=args.D,
        coupling_types=coupling_types,
        tract_lengths=tract_lengths,
        params=params,
        conduction_speed=3.0, dt=0.5, simlen=args.simlen,
    )

    # --- Save simulation ---
    os.makedirs(trial_dir, exist_ok=True)
    save_data(
        os.path.join(trial_dir, "dynamics_sim.npz"),
        t, V, C, args.G, args.D, coupling_types,
        tract_lengths=tract_lengths,
        params=params, conduction_speed=3.0, dt=0.5, simlen=args.simlen,
    )

    # --- Analysis ---
    fc_raw, fc_significance, fc_density = compute_fc(V)
    mi_raw, mi_density = compute_mi(V, num_bins=args.num_bins)
    cl_adjacency = compute_chow_liu(mi_raw)

    # Density-matched
    n_gt_edges = np.count_nonzero(np.triu(C, k=1))
    n_match = n_gt_edges if n_gt_edges <= 4 * N else N - 1
    fc_matched = density_threshold_matrix(fc_raw, n_edges=n_match)
    mi_matched = density_threshold_matrix(mi_raw, n_edges=n_match)

    matrices_dict = {
        "fc_significance": fc_significance,
        "fc_density": fc_density,
        "fc_matched": fc_matched,
        "mi_raw": mi_raw,
        "mi_density": mi_density,
        "mi_matched": mi_matched,
        "cl_adjacency": cl_adjacency,
    }
    metrics = evaluate(C, matrices_dict)

    # --- Save analysis matrices ---
    np.savez_compressed(
        os.path.join(trial_dir, "analysis.npz"),
        ground_truth_weights=C,
        fc_raw=fc_raw,
        fc_significance=fc_significance,
        fc_density=fc_density,
        fc_matched=fc_matched,
        mi_raw=mi_raw,
        mi_density=mi_density,
        mi_matched=mi_matched,
        cl_adjacency=cl_adjacency,
    )
    print(f"Saved → {os.path.join(trial_dir, 'analysis.npz')}")

    return metrics


def build_results_arrays(N_values, n_trials):
    """Pre-allocate result arrays: {method}_{metric} → shape (len(N_values), n_trials)."""
    arrays = {}
    n_N = len(N_values)
    for method in METHODS:
        for metric in METRIC_KEYS:
            key = f"{method}_{metric}"
            arrays[key] = np.full((n_N, n_trials), np.nan)
    return arrays


def store_trial_metrics(arrays, ni, ti, trial_metrics):
    """Fill in arrays for one trial from evaluate() output."""
    for method in METHODS:
        if method not in trial_metrics:
            continue
        m = trial_metrics[method]
        for metric in METRIC_KEYS:
            eval_key = _EVAL_KEY_MAP[metric]
            arrays[f"{method}_{metric}"][ni, ti] = m[eval_key]


def build_summary(N_values, arrays, config):
    """Build human-readable summary dict."""
    summary = {"config": config, "results": {}}
    for ni, N in enumerate(N_values):
        N_key = str(N)
        summary["results"][N_key] = {}
        for method in METHODS:
            method_summary = {}
            for metric in METRIC_KEYS:
                vals = arrays[f"{method}_{metric}"][ni]
                method_summary[metric] = {
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals)),
                }
            summary["results"][N_key][method] = method_summary
    return summary


def print_sweep_table(N_values, arrays):
    """Print a summary table with mean +/- std for key metrics across methods."""
    key_metrics = ["pearson", "f1"]

    for ni, N in enumerate(N_values):
        print(f"\n{'='*80}")
        print(f"  N = {N}")
        print(f"{'='*80}")

        header_parts = [f"{'Method':<20}"]
        for metric in key_metrics:
            header_parts.append(f"{metric:>18}")
        header = " ".join(header_parts)
        print(header)
        print("-" * len(header))

        for method in METHODS:
            parts = [f"{method:<20}"]
            for metric in key_metrics:
                vals = arrays[f"{method}_{metric}"][ni]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                parts.append(f"{mean:>8.4f}±{std:<7.4f}")
            print(" ".join(parts))

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep over N values, run multiple trials, collect metrics."
    )
    parser.add_argument("--N-values", type=int, nargs="+", default=[5, 10, 15, 20, 25],
                        help="List of N values to sweep (default: 5 10 15 20 25)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials per N (default: 10)")
    parser.add_argument("--simlen", type=float, default=600000,
                        help="Simulation length in ms (default: 600000)")
    parser.add_argument("--connectivity",
                        choices=["erdos_renyi", "dense", "sparse", "tree", "hierarchical"],
                        default="erdos_renyi", help="Connectivity mode (default: erdos_renyi)")
    parser.add_argument("--density", type=float, default=0.5,
                        help="Edge probability for erdos_renyi mode (0 < density ≤ 1, default: 0.5)")
    parser.add_argument("--branching", type=int, default=3,
                        help="Max children per node for hierarchical mode (default: 3)")
    parser.add_argument("--edge-mode", choices=["random", "uniform"], default="random",
                        help="Coupling type assignment mode (default: random)")
    parser.add_argument("--coupling-type", choices=list(NAME_TO_TYPE.keys()), default=None,
                        help="Coupling type for uniform edge-mode")
    parser.add_argument("--extra-edges", type=int, default=0,
                        help="Extra edges for tree/hierarchical mode (default: 0)")
    parser.add_argument("--no-delays", action="store_true",
                        help="Disable conduction delays")
    parser.add_argument("--G", type=float, default=0.3,
                        help="Global coupling strength (default: 0.3)")
    parser.add_argument("--D", type=float, default=2e-4,
                        help="Noise amplitude (default: 2e-4)")
    parser.add_argument("--num-bins", type=int, default=100,
                        help="MI discretization bins (default: 100)")
    parser.add_argument("--outdir", default="output/sweep",
                        help="Output directory (default: output/sweep)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (each trial uses seed + trial_idx)")
    args = parser.parse_args()

    if args.edge_mode == "uniform" and args.coupling_type is None:
        parser.error("--coupling-type is required when --edge-mode is uniform")

    N_values = sorted(args.N_values)
    n_trials = args.trials

    print(f"Sweep: N = {N_values}, {n_trials} trials each")
    print(f"Connectivity: {args.connectivity}, edge-mode: {args.edge_mode}, "
          f"simlen: {args.simlen} ms")
    print(f"Output: {args.outdir}\n")

    arrays = build_results_arrays(N_values, n_trials)

    for ni, N in enumerate(N_values):
        for ti in range(n_trials):
            print(f"\n{'#'*60}")
            print(f"  N={N}, trial {ti+1}/{n_trials}")
            print(f"{'#'*60}")

            trial_dir = os.path.join(args.outdir, f"N{N}", f"trial{ti}")
            t0 = time.time()
            trial_metrics = run_single_trial(N, ti, args, trial_dir)
            elapsed = time.time() - t0
            print(f"Trial completed in {elapsed:.1f}s")

            store_trial_metrics(arrays, ni, ti, trial_metrics)

    # --- Save results ---
    os.makedirs(args.outdir, exist_ok=True)

    # sweep_results.npz
    save_dict = {"N_values": np.array(N_values), "n_trials": n_trials}
    save_dict.update(arrays)
    npz_path = os.path.join(args.outdir, "sweep_results.npz")
    np.savez_compressed(npz_path, **save_dict)
    print(f"\nSaved → {npz_path}")

    # sweep_summary.json
    config = {
        "N_values": N_values,
        "trials": n_trials,
        "simlen": args.simlen,
        "connectivity": args.connectivity,
        "density": args.density,
        "branching": args.branching,
        "edge_mode": args.edge_mode,
        "coupling_type": args.coupling_type,
        "extra_edges": args.extra_edges,
        "no_delays": args.no_delays,
        "G": args.G,
        "D": args.D,
        "num_bins": args.num_bins,
        "seed": args.seed,
    }
    summary = build_summary(N_values, arrays, config)
    json_path = os.path.join(args.outdir, "sweep_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {json_path}")

    # --- Print summary table ---
    print_sweep_table(N_values, arrays)


if __name__ == "__main__":
    main()
