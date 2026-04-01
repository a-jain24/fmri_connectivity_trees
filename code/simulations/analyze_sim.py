"""
analyze_sim.py — Consolidated analysis pipeline for Generic2dOscillator simulations.

Load simulation .npz → compute FC, MI, Chow-Liu tree → evaluate recovery of
ground-truth input connectivity.

Usage:
    python code/simulations/analyze_sim.py \
        --input code/simulations/output/try1_2_11_26/generic_2doscillator_tvb_heterogeneous.npz \
        --outdir code/simulations/output/try1_2_11_26/analysis

    python code/simulations/analyze_sim.py \
        --input code/simulations/output/testing2_2_11_26/generic_2doscillator_sim.npz
"""

import argparse
import json
import os
import re
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection


# ---------------------------------------------------------------------------
# 1. Load simulation
# ---------------------------------------------------------------------------

def load_simulation(npz_path):
    """Load V (T, N) and weights (N, N) from an .npz file.

    Auto-detects two formats:
      - 'weights' key  (TVB heterogeneous output)
      - 'conn' key     (from-scratch simulation output)
    """
    data = np.load(npz_path, allow_pickle=True)

    V = data["V"]  # (T, N) in both formats

    if "weights" in data:
        weights = data["weights"]
    elif "conn" in data:
        weights = data["conn"]
    else:
        raise KeyError("Neither 'weights' nor 'conn' found in .npz file")

    print(f"Loaded V {V.shape} and weights {weights.shape} from {npz_path}")
    return V, weights


# ---------------------------------------------------------------------------
# 2. Functional connectivity
# ---------------------------------------------------------------------------

def fc_significance_threshold(X, alpha=0.05, fdr=True):
    """Pearson correlation with Fisher-z significance thresholding.

    Parameters
    ----------
    X : ndarray (T, N)
    alpha : float
    fdr : bool — apply FDR correction

    Returns
    -------
    FC_thr, C, pvals
    """
    T, N = X.shape

    C = np.corrcoef(X, rowvar=False)
    C = np.clip(C, -0.999999, 0.999999)

    Z = np.arctanh(C)
    Z_stat = Z * np.sqrt(T - 3)

    pvals = 2 * (1 - norm.cdf(np.abs(Z_stat)))
    np.fill_diagonal(pvals, 1.0)

    if fdr:
        mask = np.triu(np.ones_like(pvals, dtype=bool), k=1)
        rejected, _ = fdrcorrection(pvals[mask], alpha=alpha)
        sig = np.zeros_like(C, dtype=bool)
        sig[mask] = rejected
        sig = sig | sig.T
    else:
        sig = pvals < alpha

    FC_thr = C * sig
    return FC_thr, C, pvals


def density_threshold_matrix(C, n_edges, use_abs=True):
    """Keep only the top *n_edges* edges by magnitude.

    Parameters
    ----------
    C : ndarray (N, N)
    n_edges : int — number of edges to retain
    use_abs : bool — rank by absolute value
    """
    if torch.is_tensor(C):
        C = C.cpu().numpy()

    C = C.copy()
    np.fill_diagonal(C, 0.0)

    W = np.abs(C) if use_abs else C

    iu = np.triu_indices_from(W, k=1)
    weights = W[iu]

    if n_edges > len(weights):
        raise ValueError("n_edges exceeds number of possible edges")

    idx = np.argpartition(weights, -n_edges)[-n_edges:]
    mask = np.zeros_like(weights, dtype=bool)
    mask[idx] = True

    C_thr = np.zeros_like(C)
    C_thr[iu[0][mask], iu[1][mask]] = C[iu[0][mask], iu[1][mask]]
    C_thr = C_thr + C_thr.T
    return C_thr


def compute_fc(V, alpha=0.05):
    """Compute FC matrices from time series V (T, N).

    Returns
    -------
    fc_raw : full correlation matrix
    fc_significance : significance-thresholded FC
    fc_density : density-thresholded FC (top N-1 edges)
    """
    N = V.shape[1]
    fc_significance, fc_raw, _ = fc_significance_threshold(V, alpha=alpha)
    fc_density = density_threshold_matrix(fc_raw, n_edges=N - 1)
    print(f"FC computed — raw range [{fc_raw.min():.3f}, {fc_raw.max():.3f}]")
    return fc_raw, fc_significance, fc_density


# ---------------------------------------------------------------------------
# 3. Mutual information (torch-accelerated)
# ---------------------------------------------------------------------------

def discretize_time_series(timeseries, num_bins=100):
    """Discretize tensor (N, T) into bins."""
    timeseries = timeseries.contiguous()
    bin_edges = torch.linspace(
        timeseries.min(), timeseries.max(),
        steps=num_bins + 1, device=timeseries.device
    )[1:-1]
    return torch.bucketize(timeseries, bin_edges)


def joint_prob(a, b, discretized_timeseries, num_bins=100):
    """Joint probability matrix for ROIs a and b."""
    a_vals = discretized_timeseries[a, :]
    b_vals = discretized_timeseries[b, :]
    joint_indices = num_bins * a_vals + b_vals
    joint_prob_mat = torch.bincount(
        joint_indices, minlength=num_bins * num_bins
    ).reshape(num_bins, num_bins).float()
    joint_prob_mat /= joint_prob_mat.sum()
    return joint_prob_mat


def get_joint_probs(discretized_timeseries, num_bins=100):
    """Joint probability matrices for all ROI pairs."""
    num_rois = discretized_timeseries.shape[0]
    joint_probs = torch.zeros(
        (num_rois, num_rois, num_bins, num_bins),
        dtype=torch.float32, device=discretized_timeseries.device,
    )
    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            jp = joint_prob(i, j, discretized_timeseries, num_bins)
            joint_probs[i, j] = jp
            joint_probs[j, i] = jp.T
    return joint_probs


def get_product_of_marginals(joint_probs, num_bins=100):
    """Product of marginals for all ROI pairs."""
    num_rois = joint_probs.shape[0]
    product_marginals = torch.zeros(
        (num_rois, num_rois, num_bins, num_bins),
        dtype=torch.float32, device=joint_probs.device,
    )
    for i in range(num_rois):
        for j in range(num_rois):
            if i != j:
                jp = joint_probs[i, j]
                marginal_a = jp.sum(dim=1)
                marginal_b = jp.sum(dim=0)
                product_marginals[i, j] = torch.outer(marginal_a, marginal_b)
    return product_marginals


def get_mutual_information(joint_probs, product_marginals, num_bins=100):
    """Mutual information matrix for all ROI pairs."""
    num_rois = joint_probs.shape[0]
    mi_matrix = torch.zeros(
        (num_rois, num_rois),
        dtype=torch.float32, device=joint_probs.device,
    )
    for i in range(num_rois):
        for j in range(num_rois):
            if i != j:
                jp = joint_probs[i, j].clone()
                pm = product_marginals[i, j].clone()
                jp[jp == 0] = 1e-10
                pm[pm == 0] = 1e-10
                mi_matrix[i, j] = torch.sum(jp * torch.log(jp / pm))
    return mi_matrix


def pairwise_roi_mutual_information(timeseries_tensor, num_bins=100):
    """Full MI pipeline: discretize → joint probs → marginals → MI matrix."""
    disc = discretize_time_series(timeseries_tensor, num_bins=num_bins)
    jp = get_joint_probs(disc, num_bins=num_bins)
    pm = get_product_of_marginals(jp, num_bins=num_bins)
    mi = get_mutual_information(jp, pm, num_bins=num_bins)
    return mi


def compute_mi(V, num_bins=100):
    """Compute MI from V (T, N). Returns mi_raw, mi_density."""
    N = V.shape[1]
    V_tensor = torch.tensor(V.T, dtype=torch.float32)  # (N, T)
    mi_raw = pairwise_roi_mutual_information(V_tensor, num_bins=num_bins)
    mi_raw = mi_raw.cpu().numpy()
    mi_density = density_threshold_matrix(mi_raw, n_edges=N - 1)
    print(f"MI computed — raw range [{mi_raw.min():.4f}, {mi_raw.max():.4f}]")
    return mi_raw, mi_density


# ---------------------------------------------------------------------------
# 4. Chow-Liu tree
# ---------------------------------------------------------------------------

def mi_to_graph(mi):
    """Convert MI matrix to weighted NetworkX graph."""
    G = nx.Graph()
    num_variables = mi.shape[0]
    for i in range(num_variables):
        G.add_node(i)
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            G.add_edge(i, j, weight=mi[i, j])
    return G


def compute_chow_liu(mi_raw):
    """Build Chow-Liu tree (maximum spanning tree) from MI matrix.

    Returns adjacency matrix (N, N) with original MI weights on tree edges.
    """
    G = mi_to_graph(mi_raw)
    cl_tree = nx.maximum_spanning_tree(G, algorithm="kruskal", weight="weight")
    cl_adjacency = nx.to_numpy_array(cl_tree, nodelist=sorted(cl_tree.nodes()))
    print(f"Chow-Liu tree — {cl_tree.number_of_edges()} edges")
    return cl_adjacency


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def calculate_confusion_matrix_elements(true_matrix, predicted_matrix):
    """TP, FP, TN, FN from binarised (nonzero vs zero) matrices."""
    true_flat = np.abs(true_matrix.flatten())
    predicted_flat = np.abs(predicted_matrix.flatten())

    TP = int(np.sum((true_flat > 0) & (predicted_flat > 0)))
    FP = int(np.sum((true_flat == 0) & (predicted_flat > 0)))
    TN = int(np.sum((true_flat == 0) & (predicted_flat == 0)))
    FN = int(np.sum((true_flat > 0) & (predicted_flat == 0)))
    return TP, FP, TN, FN


def evaluate(weights, matrices_dict):
    """Evaluate each estimated matrix against ground-truth weights.

    Auto-detects sparse ground truth (≤ 4N nonzero upper-triangle edges) and
    evaluates against full nonzero structure. For dense ground truth, falls back
    to density-thresholding to N-1 edges.

    Parameters
    ----------
    weights : ndarray (N, N) — ground-truth connectivity
    matrices_dict : dict[str, ndarray] — name → estimated matrix

    Returns
    -------
    dict[str, dict] — metrics per method
    """
    N = weights.shape[0]
    n_upper_nonzero = np.count_nonzero(np.triu(weights, k=1))

    if n_upper_nonzero <= 4 * N:
        # Sparse ground truth — use full nonzero structure
        weights_dt = weights.copy()
        np.fill_diagonal(weights_dt, 0.0)
        print(f"Ground truth is sparse ({n_upper_nonzero} edges) — evaluating against full structure")
    else:
        # Dense ground truth — density-threshold to N-1 edges
        weights_dt = density_threshold_matrix(weights, n_edges=N - 1)
        print(f"Ground truth is dense ({n_upper_nonzero} edges) — density-thresholding to {N - 1} edges")

    weights_flat = np.abs(weights_dt.flatten())

    metrics = {}
    for name, mat in matrices_dict.items():
        sim_mat = np.abs(mat)
        sim_flat = sim_mat.flatten()

        # Upper triangle for MSE
        iu = np.triu_indices_from(sim_mat, k=1)
        sim_vals = sim_mat[iu]
        orig_vals = np.abs(weights_dt)[iu]

        corr, corr_pval = pearsonr(weights_flat, sim_flat)
        mse = float(np.mean((sim_vals - orig_vals) ** 2))

        TP, FP, TN, FN = calculate_confusion_matrix_elements(weights_dt, sim_mat)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
               if (precision + sensitivity) > 0 else 0.0)

        metrics[name] = {
            "pearson_correlation": float(corr),
            "pearson_pvalue": float(corr_pval),
            "mean_squared_error": mse,
            "confusion_matrix": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
        }

    return metrics


# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------

def print_summary_table(metrics):
    """Pretty-print metrics as a table."""
    header = f"{'Method':<20} {'Pearson':>8} {'p-value':>10} {'MSE':>10} {'Sens':>6} {'Spec':>6} {'Prec':>6} {'F1':>6}  {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        cm = m["confusion_matrix"]
        print(
            f"{name:<20} {m['pearson_correlation']:>8.4f} {m['pearson_pvalue']:>10.2e} {m['mean_squared_error']:>10.6f} "
            f"{m['sensitivity']:>6.3f} {m['specificity']:>6.3f} {m['precision']:>6.3f} {m['f1']:>6.3f}  "
            f"{cm['TP']:>5d} {cm['FP']:>5d} {cm['TN']:>5d} {cm['FN']:>5d}"
        )
    print("=" * len(header) + "\n")


def save_results(outdir, stem, weights, fc_raw, fc_significance, fc_density,
                 fc_matched, mi_raw, mi_density, mi_matched, cl_adjacency, metrics):
    """Save .npz arrays and .json metrics."""
    os.makedirs(outdir, exist_ok=True)

    npz_path = os.path.join(outdir, f"{stem}_analysis.npz")
    np.savez_compressed(
        npz_path,
        ground_truth_weights=weights,
        fc_raw=fc_raw,
        fc_significance=fc_significance,
        fc_density=fc_density,
        fc_matched=fc_matched,
        mi_raw=mi_raw,
        mi_density=mi_density,
        mi_matched=mi_matched,
        cl_adjacency=cl_adjacency,
    )
    print(f"Saved arrays → {npz_path}")

    json_path = os.path.join(outdir, f"{stem}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics → {json_path}")


# ---------------------------------------------------------------------------
# 7. Sweep-mode helpers
# ---------------------------------------------------------------------------

# Mirror of constants in sweep_sim.py (kept here to avoid circular imports)
_SWEEP_METHODS = [
    "fc_significance", "fc_density", "fc_matched",
    "mi_raw", "mi_density", "mi_matched",
    "cl_adjacency",
]
_SWEEP_METRIC_KEYS = [
    "pearson", "pearson_pvalue", "f1",
    "sensitivity", "specificity", "precision", "mse",
]
_EVAL_KEY_MAP = {
    "pearson": "pearson_correlation",
    "pearson_pvalue": "pearson_pvalue",
    "f1": "f1",
    "sensitivity": "sensitivity",
    "specificity": "specificity",
    "precision": "precision",
    "mse": "mean_squared_error",
}


def discover_sweep(sweep_dir):
    """Scan sweep_dir for N{N}/trial{t}/dynamics_sim.npz files.

    Returns
    -------
    all_N_values : sorted list of ints
    n_trials_per_N : dict  {N: int}  — number of valid trials found
    """
    all_N_values = []
    n_trials_per_N = {}
    for entry in sorted(os.listdir(sweep_dir), key=lambda d: int(d[1:]) if re.match(r"^N\d+$", d) else -1):
        if not re.match(r"^N\d+$", entry):
            continue
        N = int(entry[1:])
        n_dir = os.path.join(sweep_dir, entry)
        if not os.path.isdir(n_dir):
            continue
        trial_dirs = sorted(
            [t for t in os.listdir(n_dir)
             if re.match(r"^trial\d+$", t) and
             os.path.exists(os.path.join(n_dir, t, "dynamics_sim.npz"))],
            key=lambda t: int(t[5:]),
        )
        if trial_dirs:
            all_N_values.append(N)
            n_trials_per_N[N] = len(trial_dirs)
    return all_N_values, n_trials_per_N


def analyze_trial(trial_dir, num_bins=100, alpha=0.05):
    """Load dynamics_sim.npz, run full analysis pipeline, save analysis.npz.

    Returns the metrics dict from evaluate().
    """
    sim_path = os.path.join(trial_dir, "dynamics_sim.npz")
    V, weights = load_simulation(sim_path)
    N = weights.shape[0]

    fc_raw, fc_significance, fc_density = compute_fc(V, alpha=alpha)
    mi_raw, mi_density = compute_mi(V, num_bins=num_bins)
    cl_adjacency = compute_chow_liu(mi_raw)

    n_gt_edges = np.count_nonzero(np.triu(weights, k=1))
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
    metrics = evaluate(weights, matrices_dict)

    np.savez_compressed(
        os.path.join(trial_dir, "analysis.npz"),
        ground_truth_weights=weights,
        fc_raw=fc_raw,
        fc_significance=fc_significance,
        fc_density=fc_density,
        fc_matched=fc_matched,
        mi_raw=mi_raw,
        mi_density=mi_density,
        mi_matched=mi_matched,
        cl_adjacency=cl_adjacency,
    )
    print(f"  Saved → {os.path.join(trial_dir, 'analysis.npz')}")
    return metrics


def merge_and_save_sweep_results(sweep_dir, N_values, new_arrays, n_trials):
    """Merge new metric arrays into sweep_results.npz and sweep_summary.json.

    If sweep_results.npz already exists, rows for N values not being updated
    are preserved. Rows for updated N values are overwritten. New N values are
    appended. The final file is always sorted by N.

    Parameters
    ----------
    sweep_dir  : str
    N_values   : list of int — N values that were just analyzed
    new_arrays : dict  — {"{method}_{metric}": ndarray shape (len(N_values), n_trials)}
    n_trials   : int
    """
    npz_path = os.path.join(sweep_dir, "sweep_results.npz")

    if os.path.exists(npz_path):
        existing = dict(np.load(npz_path, allow_pickle=True))
        ex_N_list = existing["N_values"].tolist()
        ex_n_trials = int(existing["n_trials"])
        n_cols = max(ex_n_trials, n_trials)

        # Union of N values, sorted
        merged_N_list = sorted(set(ex_N_list) | set(N_values))

        # Build merged arrays, initialising from existing data
        merged_arrays = {}
        all_keys = set(existing.keys()) | set(new_arrays.keys())
        all_keys -= {"N_values", "n_trials"}
        for key in all_keys:
            arr = np.full((len(merged_N_list), n_cols), np.nan)
            if key in existing and existing[key].ndim == 2:
                for old_i, N in enumerate(ex_N_list):
                    new_i = merged_N_list.index(N)
                    cols = min(existing[key].shape[1], n_cols)
                    arr[new_i, :cols] = existing[key][old_i, :cols]
            merged_arrays[key] = arr

        # Overwrite rows for newly analyzed N values
        for local_i, N in enumerate(N_values):
            merged_i = merged_N_list.index(N)
            for key, val in new_arrays.items():
                cols = min(val.shape[1], n_cols)
                merged_arrays[key][merged_i, :cols] = val[local_i, :cols]

        final_N = np.array(merged_N_list)
        final_n_trials = n_cols
        final_arrays = merged_arrays
    else:
        final_N = np.array(N_values)
        final_n_trials = n_trials
        final_arrays = new_arrays

    save_dict = {"N_values": final_N, "n_trials": final_n_trials}
    save_dict.update(final_arrays)
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved → {npz_path}")

    # Update sweep_summary.json (preserve existing config if present)
    json_path = os.path.join(sweep_dir, "sweep_summary.json")
    config = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            config = json.load(f).get("config", {})

    results = {}
    for ni, N in enumerate(final_N.tolist()):
        results[str(N)] = {}
        for method in _SWEEP_METHODS:
            method_summary = {}
            for metric in _SWEEP_METRIC_KEYS:
                key = f"{method}_{metric}"
                if key in final_arrays:
                    vals = final_arrays[key][ni]
                    method_summary[metric] = {
                        "mean": float(np.nanmean(vals)),
                        "std": float(np.nanstd(vals)),
                    }
            results[str(N)][method] = method_summary

    with open(json_path, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)
    print(f"Saved → {json_path}")


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Generic2dOscillator simulation output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Single-file mode:\n"
            "  python analyze_sim.py --input output/trial0/dynamics_sim.npz\n\n"
            "Sweep mode (re-analyze subset of a sweep directory):\n"
            "  python analyze_sim.py --sweep-dir output/sweep --N 10 20\n"
            "  python analyze_sim.py --sweep-dir output/sweep  # all N values\n"
        ),
    )

    # --- Single-file mode ---
    parser.add_argument(
        "--input", default=None,
        help="Path to a single simulation .npz file",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory for single-file mode (default: <input_dir>/analysis)",
    )

    # --- Sweep mode ---
    parser.add_argument(
        "--sweep-dir", default=None,
        help="Path to sweep output directory (produced by sweep_sim.py)",
    )
    parser.add_argument(
        "--N", type=int, nargs="+", metavar="N",
        help="Subset of N values to re-analyze (e.g. --N 10 20). "
             "Defaults to all N values found in the sweep directory.",
    )

    # --- Shared options ---
    parser.add_argument(
        "--num-bins", type=int, default=100,
        help="Number of bins for MI discretization (default: 100)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for FC thresholding (default: 0.05)",
    )
    args = parser.parse_args()

    if args.input is None and args.sweep_dir is None:
        parser.error("Provide either --input (single file) or --sweep-dir (sweep mode).")
    if args.input is not None and args.sweep_dir is not None:
        parser.error("--input and --sweep-dir are mutually exclusive.")

    # ── Single-file mode ────────────────────────────────────────────────────
    if args.input is not None:
        if args.N is not None:
            parser.error("--N is only valid in sweep mode (--sweep-dir).")

        input_path = args.input
        stem = Path(input_path).stem
        outdir = args.outdir or os.path.join(os.path.dirname(input_path), "analysis")

        V, weights = load_simulation(input_path)

        print("\n--- Computing FC ---")
        fc_raw, fc_significance, fc_density = compute_fc(V, alpha=args.alpha)

        print("\n--- Computing MI ---")
        mi_raw, mi_density = compute_mi(V, num_bins=args.num_bins)

        print("\n--- Computing Chow-Liu Tree ---")
        cl_adjacency = compute_chow_liu(mi_raw)

        N = weights.shape[0]
        n_gt_edges = np.count_nonzero(np.triu(weights, k=1))
        n_match = n_gt_edges if n_gt_edges <= 4 * N else N - 1
        fc_matched = density_threshold_matrix(fc_raw, n_edges=n_match)
        mi_matched = density_threshold_matrix(mi_raw, n_edges=n_match)
        print(f"Density-matched FC and MI at {n_match} edges")

        print("\n--- Evaluating ---")
        matrices_dict = {
            "fc_significance": fc_significance,
            "fc_density": fc_density,
            "fc_matched": fc_matched,
            "mi_raw": mi_raw,
            "mi_density": mi_density,
            "mi_matched": mi_matched,
            "cl_adjacency": cl_adjacency,
        }
        metrics = evaluate(weights, matrices_dict)
        print_summary_table(metrics)

        save_results(
            outdir, stem, weights,
            fc_raw, fc_significance, fc_density, fc_matched,
            mi_raw, mi_density, mi_matched, cl_adjacency, metrics,
        )
        return

    # ── Sweep mode ───────────────────────────────────────────────────────────
    sweep_dir = args.sweep_dir
    all_N_values, n_trials_per_N = discover_sweep(sweep_dir)

    if not all_N_values:
        print(f"No N{{N}}/trial*/dynamics_sim.npz files found in {sweep_dir}")
        return

    print(f"Found N values in sweep: {all_N_values}")

    if args.N is not None:
        invalid = sorted(set(args.N) - set(all_N_values))
        if invalid:
            parser.error(
                f"Requested N values not found in sweep: {invalid}. "
                f"Available: {sorted(all_N_values)}"
            )
        N_values = sorted(set(args.N))
    else:
        N_values = all_N_values

    # Require consistent n_trials across selected N values
    n_trials_list = [n_trials_per_N[N] for N in N_values]
    if len(set(n_trials_list)) > 1:
        print(f"Warning: trial counts differ across N values: "
              f"{dict(zip(N_values, n_trials_list))}. "
              f"Using max ({max(n_trials_list)}) and padding with NaN.")
    n_trials = max(n_trials_list)

    # Pre-allocate metric arrays: shape (len(N_values), n_trials)
    new_arrays = {
        f"{method}_{metric}": np.full((len(N_values), n_trials), np.nan)
        for method in _SWEEP_METHODS
        for metric in _SWEEP_METRIC_KEYS
    }

    for ni, N in enumerate(N_values):
        for ti in range(n_trials_per_N[N]):
            trial_dir = os.path.join(sweep_dir, f"N{N}", f"trial{ti}")
            print(f"\n{'#'*50}")
            print(f"  N={N}, trial {ti}")
            print(f"{'#'*50}")

            metrics = analyze_trial(trial_dir, num_bins=args.num_bins, alpha=args.alpha)
            print_summary_table(metrics)

            for method in _SWEEP_METHODS:
                if method not in metrics:
                    continue
                m = metrics[method]
                for metric in _SWEEP_METRIC_KEYS:
                    eval_key = _EVAL_KEY_MAP[metric]
                    new_arrays[f"{method}_{metric}"][ni, ti] = m[eval_key]

    print(f"\n--- Saving sweep results to {sweep_dir} ---")
    merge_and_save_sweep_results(sweep_dir, N_values, new_arrays, n_trials)
    print("\nDone.")


if __name__ == "__main__":
    main()
