"""
noise_sim.py — Noise-level sweep on a shared ground-truth tree network.

One tree network (N=15, all-LINEAR coupling) is shared across 5 "individuals".
Each individual gets different heterogeneous intrinsic oscillator parameters
and a different noise amplitude D (low → high).  CL trees and FC density trees
are recovered from each individual's time series, then compared.

Hypothesis (from notes/noise.md):
  Chow-Liu trees should be more consistent across individuals than
  density-matched FC (top N-1 correlation edges), even when intrinsic
  dynamics and noise levels differ.

Output layout (determined by --run-id):
  datasets/synthetic/<run_id>/timeseries/individual_<k>_D<level>.npz   — V + connectivity
  code/simulations/output/<run_id>/individual_<k>_D<level>_analysis.npz — FC / MI / CL results
  code/simulations/output/<run_id>/params.json                           — full run parameters
  code/simulations/figures/<run_id>/noise_sim_networks.pdf               — GT + CL + FC panels
  code/simulations/figures/<run_id>/noise_sim_consistency.pdf            — Jaccard heatmaps

Usage:
    python code/simulations/noise_sim.py
    python code/simulations/noise_sim.py --simlen 200000 --seed 42 --run-id my_noise_run
"""

import argparse
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_utils import analysis_dir, detect_device, figures_dir, make_run_id, save_params, timeseries_dir
from dynamics_simulation import (
    LINEAR,
    build_connectivity,
    build_heterogeneous_params,
    build_symmetric_coupling_types,
    run_sim,
)
from analyze_sim import (
    compute_chow_liu,
    compute_fc,
    compute_mi,
)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

N_NODES       = 15
N_INDIVIDUALS = 5
NOISE_LEVELS  = [1e-5, 5e-5, 2e-4, 8e-4, 3e-3]   # low → high
G_COUPLING    = 0.3
DEFAULT_SIMLEN = 600_000   # ms
DT            = 0.5

# TVB Generic2dOscillator defaults (shared starting point)
_DEFAULTS = {
    "a": 0.0, "b": -10.0, "c": 0.0, "d": 0.02, "e": 3.0,
    "f": 1.0,  "g": 0.0,  "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "tau": 1.0,
}
# Per-node parameter variability — moderate sigmas to create realistic
# between-individual heterogeneity without destabilising the oscillators.
_SIGMAS = {
    "a": 0.05, "b": 0.5, "c": 0.01, "d": 0.005, "e": 0.3,
    "f": 0.1,  "g": 0.05, "alpha": 0.1, "beta": 0.1, "gamma": 0.1, "tau": 0.15,
}

# Edge colour palette
_TP_COLOR   = "#2196F3"   # blue  — correct edge
_FP_COLOR   = "#F44336"   # red   — spurious edge
_FN_COLOR   = "#BDBDBD"   # gray  — missed edge (drawn dashed)
_NODE_COLOR = "#F5F5F5"
_EDGE_GT    = "#555555"   # neutral gray for the GT panel


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def edge_set(adj: np.ndarray, tol: float = 1e-12) -> frozenset:
    """Return frozenset of (i, j) pairs with i < j for all nonzero edges."""
    N = adj.shape[0]
    return frozenset(
        (i, j)
        for i in range(N)
        for j in range(i + 1, N)
        if abs(adj[i, j]) > tol
    )


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard index |a∩b| / |a∪b|.  Returns 1.0 if both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def gt_corr(C: np.ndarray, adj: np.ndarray) -> float:
    """Pearson correlation between binary upper-triangle vectors of C and adj."""
    N = C.shape[0]
    iu = np.triu_indices(N, k=1)
    gt_vec  = (np.abs(C[iu])   > 1e-12).astype(float)
    rec_vec = (np.abs(adj[iu]) > 1e-12).astype(float)
    if gt_vec.std() == 0 or rec_vec.std() == 0:
        return 0.0
    return float(np.corrcoef(gt_vec, rec_vec)[0, 1])


def pairwise_jaccard_matrix(edge_sets: list) -> np.ndarray:
    """Compute symmetric n×n Jaccard similarity matrix."""
    n = len(edge_sets)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = jaccard(edge_sets[i], edge_sets[j])
            J[i, j] = J[j, i] = v
    return J


# ---------------------------------------------------------------------------
# Layout helpers  (copied from visualize_sweep.py)
# ---------------------------------------------------------------------------

def _hierarchy_pos(G, root, width=2.0, vert_gap=0.6):
    """Hierarchical layout for a directed BFS tree."""
    def _count_desc(node):
        children = list(G.successors(node))
        return 1 if not children else sum(_count_desc(c) for c in children)

    def _recurse(node, left, right, y, pos):
        pos[node] = ((left + right) / 2, y)
        children = list(G.successors(node))
        if children:
            total = sum(_count_desc(c) for c in children)
            dx = (right - left) / total
            x = left
            for child in children:
                sz = _count_desc(child)
                _recurse(child, x, x + dx * sz, y - vert_gap, pos)
                x += dx * sz
        return pos

    return _recurse(root, 0, width, 0, {})


def _pub_layout(G):
    """Hierarchical layout if G is a tree, else Kamada-Kawai."""
    if nx.is_tree(G) and G.number_of_nodes() > 0:
        root = max(G.nodes, key=lambda n: G.degree(n))
        directed = nx.bfs_tree(G, source=root)
        return _hierarchy_pos(directed, root)
    return nx.kamada_kawai_layout(G) if G.number_of_edges() > 0 else nx.spring_layout(G, seed=0)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_network(ax, adj: np.ndarray, pos: dict, title: str,
                 edge_color: str = _EDGE_GT) -> None:
    """Draw a network on *ax* using the shared node layout *pos*."""
    N = adj.shape[0]
    rec_edges = edge_set(adj)

    for i, j in rec_edges:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color=edge_color, lw=1.6, alpha=0.85, zorder=1)

    ax.set_title(title, fontsize=8, pad=3)

    # Nodes
    xs = [pos[n][0] for n in range(N)]
    ys = [pos[n][1] for n in range(N)]
    ax.scatter(xs, ys, c=_NODE_COLOR, s=160, zorder=4,
               edgecolors="#444444", linewidths=0.8)
    for n in range(N):
        ax.annotate(str(n), (pos[n][0], pos[n][1]),
                    fontsize=6, ha="center", va="center",
                    color="#222222", fontweight="bold", zorder=5)

    ax.axis("off")
    ax.set_facecolor("white")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(C: np.ndarray, results: list, fig_dir: str) -> None:
    """Generate both output figures."""
    n = len(results)
    N = C.shape[0]
    noise_labels = [f"D={r['D']:.0e}" for r in results]

    # Shared node layout from the ground-truth graph
    G_gt = nx.Graph()
    G_gt.add_nodes_from(range(N))
    gt_es = edge_set(C)
    for i, j in gt_es:
        G_gt.add_edge(i, j)
    pos = _pub_layout(G_gt)

    # ── Figure 1: Network panels ──────────────────────────────────────────────
    # Pre-compute GT similarity metrics per individual
    cl_gt_corr = [gt_corr(C, r["cl_adj"])    for r in results]
    fc_gt_corr = [gt_corr(C, r["fc_density"]) for r in results]
    cl_gt_j    = [r["cl_gt_jaccard"]          for r in results]
    fc_gt_j    = [r["fc_gt_jaccard"]          for r in results]

    fig1 = plt.figure(figsize=(3.5 * (n + 1), 7), facecolor="white")
    fig1.suptitle(
        "Recovered networks per individual  (linear coupling, heterogeneous intrinsics)\n"
        "Row 1: Chow-Liu tree   |   Row 2: FC density tree (top N−1 edges)",
        fontsize=9, y=1.0,
    )

    gs = gridspec.GridSpec(
        2, n + 1,
        figure=fig1,
        wspace=0.15, hspace=0.55,
        left=0.02, right=0.98, top=0.88, bottom=0.04,
    )

    # Ground-truth panel spans both rows
    ax_gt = fig1.add_subplot(gs[:, 0])
    draw_network(ax_gt, C, pos, f"Ground Truth\n({len(gt_es)} edges)",
                 edge_color=_EDGE_GT)

    for k, res in enumerate(results):
        col = k + 1
        lbl = noise_labels[k]

        ax_cl = fig1.add_subplot(gs[0, col])
        draw_network(ax_cl, res["cl_adj"], pos,
                     f"CL  {lbl}\nJ={cl_gt_j[k]:.2f}  r={cl_gt_corr[k]:.2f}",
                     edge_color=_TP_COLOR)

        ax_fc = fig1.add_subplot(gs[1, col])
        draw_network(ax_fc, res["fc_density"], pos,
                     f"FC  {lbl}\nJ={fc_gt_j[k]:.2f}  r={fc_gt_corr[k]:.2f}",
                     edge_color="#FF9800")

    os.makedirs(fig_dir, exist_ok=True)
    path1 = os.path.join(fig_dir, "noise_sim_networks.pdf")
    fig1.savefig(path1, bbox_inches="tight", dpi=150)
    plt.close(fig1)
    print(f"Saved → {path1}")

    # ── Figure 2: Consistency analysis ────────────────────────────────────────
    cl_edge_sets = [r["cl_edges"] for r in results]
    fc_edge_sets = [r["fc_edges"] for r in results]

    J_cl = pairwise_jaccard_matrix(cl_edge_sets)
    J_fc = pairwise_jaccard_matrix(fc_edge_sets)

    # Off-diagonal mean (excluding self-similarity)
    mask = ~np.eye(n, dtype=bool)
    mean_cl = J_cl[mask].mean()
    mean_fc = J_fc[mask].mean()

    cl_gt_j = [r["cl_gt_jaccard"] for r in results]
    fc_gt_j = [r["fc_gt_jaccard"] for r in results]

    fig2, axes = plt.subplots(1, 3, figsize=(15, 5),
                              constrained_layout=True, facecolor="white")
    fig2.suptitle(
        "Consistency across individuals: Chow-Liu vs FC\n"
        f"Mean pairwise Jaccard — CL: {mean_cl:.3f}   FC: {mean_fc:.3f}",
        fontsize=10,
    )

    # Heatmap helper
    def _heatmap(ax, J, title, labels):
        im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(n):
            for j in range(n):
                txt_color = "white" if J[i, j] < 0.55 else "black"
                ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _heatmap(axes[0], J_cl,
             f"Chow-Liu pairwise Jaccard\n(mean off-diag = {mean_cl:.3f})",
             noise_labels)
    _heatmap(axes[1], J_fc,
             f"FC density pairwise Jaccard\n(mean off-diag = {mean_fc:.3f})",
             noise_labels)

    # Bar chart: GT recovery per individual
    x = np.arange(n)
    w = 0.35
    axes[2].bar(x - w / 2, cl_gt_j, w, color=_TP_COLOR, alpha=0.85, label="CL tree")
    axes[2].bar(x + w / 2, fc_gt_j, w, color="#FF9800", alpha=0.85, label="FC density")

    # Annotate each bar with its value
    for k in range(n):
        axes[2].text(x[k] - w / 2, cl_gt_j[k] + 0.01, f"{cl_gt_j[k]:.2f}",
                     ha="center", va="bottom", fontsize=7, color=_TP_COLOR)
        axes[2].text(x[k] + w / 2, fc_gt_j[k] + 0.01, f"{fc_gt_j[k]:.2f}",
                     ha="center", va="bottom", fontsize=7, color="#E65100")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(noise_labels, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Jaccard vs Ground Truth")
    axes[2].set_ylim(0, 1.15)
    axes[2].set_title("GT recovery per individual", fontsize=9)
    axes[2].legend(fontsize=9)
    axes[2].spines[["top", "right"]].set_visible(False)

    path2 = os.path.join(fig_dir, "noise_sim_consistency.pdf")
    fig2.savefig(path2, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Saved → {path2}")

    # Console summary
    print("\n" + "=" * 55)
    print("CONSISTENCY SUMMARY")
    print("=" * 55)
    print(f"  Mean pairwise Jaccard  CL : {mean_cl:.4f}")
    print(f"  Mean pairwise Jaccard  FC : {mean_fc:.4f}")
    print(f"\n  {'Individual':<12} {'D':>8}  {'CL-GT':>7}  {'FC-GT':>7}")
    print(f"  {'-'*40}")
    for k, r in enumerate(results):
        print(f"  {k+1:<12} {r['D']:>8.1e}  {r['cl_gt_jaccard']:>7.3f}  {r['fc_gt_jaccard']:>7.3f}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Simulation pipeline
# ---------------------------------------------------------------------------

def run_noise_sim(args):
    """Build shared network, simulate 5 individuals, return (C, results)."""

    # ── Shared ground-truth tree ──────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)   # controls gaussian_param inside build_heterogeneous_params

    print(f"Building tree connectivity  N={N_NODES}  seed={args.seed}")
    C = build_connectivity(N_NODES, "tree", rng)
    coupling_types = build_symmetric_coupling_types(
        C, mode="uniform", rng=rng, uniform_type=LINEAR
    )

    gt_es = edge_set(C)
    print(f"Ground-truth tree: {len(gt_es)} edges  (expected N−1 = {N_NODES - 1})")

    ts_dir = timeseries_dir(args.run_id)
    an_dir = analysis_dir(args.run_id)
    os.makedirs(ts_dir, exist_ok=True)
    os.makedirs(an_dir, exist_ok=True)
    results = []

    # If --shared-params, sample once and reuse for all individuals
    if args.shared_params:
        np.random.seed(args.seed + 1)
        shared_params = build_heterogeneous_params(N_NODES, _DEFAULTS, _SIGMAS)
        print("Shared intrinsic params (same for all individuals)")
    else:
        shared_params = None

    for idx, D in enumerate(NOISE_LEVELS):
        ind_seed = args.seed + idx + 1

        if shared_params is not None:
            params = shared_params
        else:
            # Different intrinsic params per individual.
            # NOTE: build_heterogeneous_params uses np.random.normal (global RNG),
            # so we seed np.random directly before each call.
            np.random.seed(ind_seed)
            params = build_heterogeneous_params(N_NODES, _DEFAULTS, _SIGMAS)

        print(f"\n{'='*60}")
        print(f"  Individual {idx + 1}/{N_INDIVIDUALS}   D = {D:.1e}   seed = {ind_seed}")
        print(f"{'='*60}")

        t, V = run_sim(
            C,
            G=G_COUPLING,
            D=D,
            coupling_types=coupling_types,
            tract_lengths=None,   # no conduction delays
            params=params,
            dt=DT,
            simlen=args.simlen,
            device=args.device,
        )

        if not np.isfinite(V).all():
            print(f"  WARNING: non-finite values in V — simulation may be unstable")

        print(f"  V: shape={V.shape}  range=[{V.min():.3f}, {V.max():.3f}]")

        # Analysis
        fc_raw, _, fc_density = compute_fc(V)
        mi_raw, _             = compute_mi(V, num_bins=args.num_bins)
        cl_adj                = compute_chow_liu(mi_raw)

        cl_es = edge_set(cl_adj)
        fc_es = edge_set(fc_density)

        cl_gt_j = jaccard(cl_es, gt_es)
        fc_gt_j = jaccard(fc_es, gt_es)
        print(f"  CL GT-Jaccard = {cl_gt_j:.3f}   FC GT-Jaccard = {fc_gt_j:.3f}")

        tag = f"individual_{idx+1}_D{D:.0e}"

        # Save timeseries to datasets/synthetic/<run_id>/timeseries/
        ts_path = os.path.join(ts_dir, f"{tag}.npz")
        np.savez_compressed(
            ts_path,
            conn=C,
            coupling_types=coupling_types,
            D=D,
            V=V,
        )
        print(f"  Saved timeseries → {ts_path}")

        # Save analysis to code/simulations/output/<run_id>/
        an_path = os.path.join(an_dir, f"{tag}_analysis.npz")
        np.savez_compressed(
            an_path,
            fc_raw=fc_raw,
            fc_density=fc_density,
            mi_raw=mi_raw,
            cl_adj=cl_adj,
        )
        print(f"  Saved analysis  → {an_path}")

        results.append({
            "D":             D,
            "cl_adj":        cl_adj,
            "fc_density":    fc_density,
            "cl_edges":      cl_es,
            "fc_edges":      fc_es,
            "cl_gt_jaccard": cl_gt_j,
            "fc_gt_jaccard": fc_gt_j,
        })

    return C, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Noise-level sweep: shared tree network, 5 individuals with "
            "heterogeneous intrinsic params and varying noise D. "
            "Compares CL vs FC network recovery and consistency."
        )
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--simlen", type=float, default=DEFAULT_SIMLEN,
        help=f"Simulation length in ms (default: {DEFAULT_SIMLEN})",
    )
    parser.add_argument(
        "--num-bins", type=int, default=100,
        help="Bins for MI discretisation (default: 100)",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device: cpu / cuda / mps or omit for auto-select",
    )
    parser.add_argument(
        "--run-id", default=None,
        help=(
            "Identifier for this run, used as the directory name under "
            "datasets/synthetic/, simulations/output/, and simulations/figures/. "
            "Auto-generated as YYYYMMDD_HHMMSS_noise_sim[_shared_params] if omitted."
        ),
    )
    parser.add_argument(
        "--shared-params", action="store_true",
        help="Use the same intrinsic NMM parameters for all individuals (only noise varies)",
    )
    args = parser.parse_args()

    args.device = detect_device(args.device)

    if args.run_id is None:
        tag = "noise_sim_shared_params" if args.shared_params else "noise_sim"
        args.run_id = make_run_id(tag)
    print(f"Run ID: {args.run_id}")

    save_params(args.run_id, "noise_sim.py", vars(args))

    C, results = run_noise_sim(args)
    print("\n--- Generating figures ---")
    make_figures(C, results, figures_dir(args.run_id))
    print("Done.")


if __name__ == "__main__":
    main()
