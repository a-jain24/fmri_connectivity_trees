"""
visualize_sweep.py — Visualize sweep results from sweep_sim.py.

Produces:
  1. metrics_vs_N.png           — Line plots of each metric vs N (mean ± std)
  2. matrices_N{N}.png          — Heatmap panels for a representative trial
                                  (each panel independently normalized to [0,1])
  3. graphs_N{N}.png            — Graph drawings for a representative trial
  4. coupling_graphs_N{N}.png   — Edges colored by coupling type (TP) / FP / FN
  5. metrics_table.png          — Summary table (also printed to stdout)

Usage:
    python visualize_sweep.py --sweep-dir output/sweep
    python visualize_sweep.py --sweep-dir /tmp/sweep_test --no-show
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np

# Optional: import evaluate from analyze_sim to reconstruct metrics when
# sweep_results.npz is absent.  Falls back gracefully if unavailable.
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from analyze_sim import evaluate, density_threshold_matrix
    _ANALYZE_AVAILABLE = True
except Exception:
    _ANALYZE_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────

METHODS = [
    "fc_significance", "fc_density", "fc_matched",
    "mi_raw", "mi_density", "mi_matched",
    "cl_adjacency",
]

METHOD_LABELS = {
    "fc_significance": "FC Significance",
    "fc_density": "FC Density",
    "fc_matched": "FC Matched",
    "mi_raw": "MI Raw",
    "mi_density": "MI Density",
    "mi_matched": "MI Matched",
    "cl_adjacency": "CL Tree",
}

DISPLAY_METRICS = ["pearson", "f1", "sensitivity", "specificity", "precision", "mse"]

METRIC_LABELS = {
    "pearson": "Pearson r",
    "f1": "F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "mse": "MSE",
}

# Consistent colors for each method across all figures
_COLORS = plt.cm.tab10.colors
METHOD_COLORS = {m: _COLORS[i] for i, m in enumerate(METHODS)}

# Coupling-type colors for plot_coupling_type_graphs
TYPE_NAMES = {1: "linear", 2: "quadrature", 3: "rectified", 4: "squared", 5: "PAC"}
TYPE_COLORS = {
    1: "#2196F3",   # linear      → blue
    2: "#4CAF50",   # quadrature  → green
    3: "#FF9800",   # rectified   → orange
    4: "#F44336",   # squared     → red
    5: "#9C27B0",   # PAC         → purple
}
FP_COLOR = "#FF6B6B"   # false positive (spurious edge)  → salmon
FN_COLOR = "#BBBBBB"   # false negative (missed edge)    → light gray


# ── Loading ────────────────────────────────────────────────────────────────

_METRIC_KEYS = ["pearson", "pearson_pvalue", "f1", "sensitivity", "specificity", "precision", "mse"]
_EVAL_KEY_MAP = {
    "pearson": "pearson_correlation",
    "pearson_pvalue": "pearson_pvalue",
    "f1": "f1",
    "sensitivity": "sensitivity",
    "specificity": "specificity",
    "precision": "precision",
    "mse": "mean_squared_error",
}


def _build_sweep_from_trials(sweep_dir):
    """Reconstruct sweep arrays from per-trial analysis.npz files.

    Scans sweep_dir for N{N}/trial{t}/analysis.npz, runs evaluate() on each,
    builds the same arrays dict that load_sweep() would return, then saves
    sweep_results.npz and sweep_summary.json so future runs load instantly.

    Returns the same (N_values, n_trials, arrays, config) tuple as load_sweep().
    """
    if not _ANALYZE_AVAILABLE:
        raise RuntimeError(
            "sweep_results.npz not found and analyze_sim could not be imported. "
            "Run sweep_sim.py or analyze_sim.py first to generate sweep_results.npz."
        )

    # Discover N values and trial counts
    N_dirs = sorted(
        [d for d in os.listdir(sweep_dir)
         if re.match(r"^N\d+$", d) and os.path.isdir(os.path.join(sweep_dir, d))],
        key=lambda d: int(d[1:]),
    )
    N_values_list = []
    n_trials_per_N = {}
    for d in N_dirs:
        N = int(d[1:])
        n_dir = os.path.join(sweep_dir, d)
        trial_dirs = sorted(
            [t for t in os.listdir(n_dir)
             if re.match(r"^trial\d+$", t) and
             os.path.exists(os.path.join(n_dir, t, "analysis.npz"))],
            key=lambda t: int(t[5:]),
        )
        if trial_dirs:
            N_values_list.append(N)
            n_trials_per_N[N] = len(trial_dirs)

    if not N_values_list:
        raise FileNotFoundError(
            f"No sweep_results.npz and no N{{N}}/trial*/analysis.npz found in {sweep_dir}"
        )

    n_trials = max(n_trials_per_N.values())
    print(f"Reconstructing sweep results from per-trial analysis.npz files: "
          f"N={N_values_list}, up to {n_trials} trials each")

    # Pre-allocate metric arrays
    arrays = {
        f"{method}_{metric}": np.full((len(N_values_list), n_trials), np.nan)
        for method in METHODS
        for metric in _METRIC_KEYS
    }

    for ni, N in enumerate(N_values_list):
        for ti in range(n_trials_per_N[N]):
            trial_dir = os.path.join(sweep_dir, f"N{N}", f"trial{ti}")
            an = np.load(os.path.join(trial_dir, "analysis.npz"))
            weights = an["ground_truth_weights"]
            matrices_dict = {m: an[m] for m in METHODS if m in an}
            metrics = evaluate(weights, matrices_dict)

            for method in METHODS:
                if method not in metrics:
                    continue
                m = metrics[method]
                for metric in _METRIC_KEYS:
                    arrays[f"{method}_{metric}"][ni, ti] = m[_EVAL_KEY_MAP[metric]]

    N_values = np.array(N_values_list)

    # Save so future runs skip this step
    npz_path = os.path.join(sweep_dir, "sweep_results.npz")
    save_dict = {"N_values": N_values, "n_trials": n_trials}
    save_dict.update(arrays)
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved → {npz_path}")

    results = {}
    for ni, N in enumerate(N_values_list):
        results[str(N)] = {}
        for method in METHODS:
            method_summary = {}
            for metric in _METRIC_KEYS:
                key = f"{method}_{metric}"
                vals = arrays[key][ni]
                method_summary[metric] = {
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals)),
                }
            results[str(N)][method] = method_summary

    json_path = os.path.join(sweep_dir, "sweep_summary.json")
    with open(json_path, "w") as f:
        json.dump({"config": {}, "results": results}, f, indent=2)
    print(f"Saved → {json_path}")

    return N_values, n_trials, arrays, {}


def load_sweep(sweep_dir):
    """Load sweep results and config.

    Falls back to reconstructing from per-trial analysis.npz files if
    sweep_results.npz / sweep_summary.json are absent, then saves them.

    Returns
    -------
    N_values : ndarray
    n_trials : int
    arrays : dict  — keys like 'fc_significance_pearson', values shape (n_N, n_trials)
    config : dict
    """
    npz_path = os.path.join(sweep_dir, "sweep_results.npz")

    if not os.path.exists(npz_path):
        return _build_sweep_from_trials(sweep_dir)

    data = np.load(npz_path, allow_pickle=True)
    N_values = data["N_values"]
    n_trials = int(data["n_trials"])

    arrays = {}
    for key in data.files:
        if key not in ("N_values", "n_trials"):
            arrays[key] = data[key]

    json_path = os.path.join(sweep_dir, "sweep_summary.json")
    config = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            config = json.load(f).get("config", {})

    return N_values, n_trials, arrays, config


# ── Figure 1: Metrics vs N ────────────────────────────────────────────────

def plot_metrics_vs_N(N_values, arrays, figdir):
    """2x3 grid of line plots: each metric vs N, one line per method."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.flat

    for ax, metric in zip(axes, DISPLAY_METRICS):
        for method in METHODS:
            key = f"{method}_{metric}"
            if key not in arrays:
                continue
            vals = arrays[key]  # (n_N, n_trials)
            mean = np.nanmean(vals, axis=1)
            color = METHOD_COLORS[method]
            ax.plot(N_values, mean, "o-", color=color, label=METHOD_LABELS[method])

        ax.set_xlabel("N (nodes)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(N_values)
        ax.set_xticklabels(
            [str(n) if n == 5 or n >= 25 else "" for n in N_values],
            fontsize=9,
        )

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    path = os.path.join(figdir, "metrics_vs_N.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
    return fig


# ── Figure 2: Adjacency matrix heatmaps ───────────────────────────────────

def plot_matrices(N, trial_dir, figdir):
    """Row of heatmaps for one trial: GT | FC Sig | FC Matched | MI Raw | MI Matched | CL.

    Each panel is normalized independently so structure is equally visible across
    methods regardless of absolute scale (FC ~[-1,1] vs MI in nats vs binary CL).
    FC panels use a diverging colormap (RdBu_r) centered at zero.
    All other panels are min-max normalized to [0,1] with viridis.
    Colorbar tick labels display the original value range.
    """
    analysis_path = os.path.join(trial_dir, "analysis.npz")
    if not os.path.exists(analysis_path):
        print(f"  Skipping matrices for {trial_dir} — analysis.npz not found")
        return None

    data = np.load(analysis_path)

    # (title, array_key, colormap, symmetric_zero)
    # symmetric_zero=True  → FC: center at 0 using ±abs_max, preserve sign color
    # symmetric_zero=False → normalize to [0,1] independently
    panels = [
        ("Ground Truth",       "ground_truth_weights", "viridis", False),
        ("FC Significance",    "fc_significance",      "RdBu_r",  True),
        ("FC Density Matched", "fc_density",           "RdBu_r",  True),
        ("MI Raw",             "mi_raw",               "viridis", False),
        ("MI Density Matched", "mi_density",           "viridis", False),
        ("CL Tree",            "cl_adjacency",         "viridis", False),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.5),
                             constrained_layout=True)
    fig.suptitle(f"Adjacency Matrices — N = {N}", fontsize=13)

    for ax, (title, key, cmap, symmetric) in zip(axes, panels):
        mat = data[key].astype(float)
        raw_min, raw_max = float(mat.min()), float(mat.max())

        if symmetric:
            # Centre at zero so negatives → blue, positives → red
            abs_max = max(abs(raw_min), abs(raw_max))
            abs_max = abs_max if abs_max > 1e-12 else 1.0
            mat_disp = mat
            vmin, vmax = -abs_max, abs_max
            cbar_ticks = [-abs_max, 0.0, abs_max]
            cbar_labels = [f"{-abs_max:.2f}", "0", f"{abs_max:.2f}"]
        else:
            # Min-max normalize to [0, 1] so every panel uses the full colormap
            span = raw_max - raw_min
            if span < 1e-12:
                mat_disp = np.zeros_like(mat)
                span = 1.0
            else:
                mat_disp = (mat - raw_min) / span
            vmin, vmax = 0.0, 1.0
            mid = raw_min + span / 2.0
            cbar_ticks = [0.0, 0.5, 1.0]
            cbar_labels = [f"{raw_min:.3g}", f"{mid:.3g}", f"{raw_max:.3g}"]

        im = ax.imshow(mat_disp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_labels)
        cbar.ax.tick_params(labelsize=7)

    path = os.path.join(figdir, f"matrices_N{N}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
    return fig


# ── Graph drawing helpers (publication style) ─────────────────────────────

def _graph_from_matrix(mat):
    """Build nx.Graph from adjacency matrix, keeping nonzero edges."""
    G = nx.Graph()
    N = mat.shape[0]
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            w = mat[i, j]
            if abs(w) > 1e-12:
                G.add_edge(i, j, weight=float(w))
    return G


def _edge_binary(mat, tol=1e-12):
    """Return frozenset of (i, j) pairs (i < j) where mat[i, j] != 0."""
    N = mat.shape[0]
    return frozenset(
        (i, j)
        for i in range(N)
        for j in range(i + 1, N)
        if abs(mat[i, j]) > tol
    )


def _hierarchy_pos(G, root, width=2.0, vert_gap=0.6):
    """Hierarchical layout for a directed BFS tree (from chow_liu_tree_visualization.ipynb)."""
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
    """Hierarchical layout if G is a tree, else Kamada-Kawai.

    All panels share the GT layout so nodes stay in the same position
    across subplots for easy visual comparison.
    """
    if nx.is_tree(G) and G.number_of_nodes() > 0:
        root = max(G.nodes, key=lambda n: G.degree(n))
        directed = nx.bfs_tree(G, source=root)
        return _hierarchy_pos(directed, root)
    return nx.kamada_kawai_layout(G) if G.number_of_edges() > 0 else nx.spring_layout(G, seed=0)


def _draw_nodes(ax, pos, nodes, colors, size=160):
    """Scatter-based node drawing (white edge border, publication style)."""
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    ax.scatter(xs, ys, c=colors, s=size, zorder=3,
               edgecolors="white", linewidths=0.8)


def _draw_edge(ax, pos, i, j, color, lw=1.4, alpha=0.85, linestyle="-"):
    """Draw a single edge as a line (ax.plot), enabling per-edge color/alpha)."""
    x0, y0 = pos[i]
    x1, y1 = pos[j]
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
            alpha=alpha, linestyle=linestyle, zorder=1)


def _draw_labels(ax, pos, nodes, fontsize=9):
    """Annotate each node with its integer index."""
    for n in nodes:
        x, y = pos[n]
        ax.annotate(str(n), (x, y), fontsize=fontsize, ha="center", va="center",
                    color="#222222", fontweight="bold", zorder=4)


def _edge_alphas(edges, weights_dict, lo=0.25, hi=0.95):
    """Map edge weights linearly to alpha range [lo, hi]."""
    if not edges:
        return {}
    vals = [abs(weights_dict.get((i, j), weights_dict.get((j, i), 0.0))) for (i, j) in edges]
    w_min, w_max = min(vals), max(vals)
    span = w_max - w_min if w_max > w_min else 1.0
    return {e: lo + (hi - lo) * (v - w_min) / span for e, v in zip(edges, vals)}


# ── Figure 3: Graph drawings ──────────────────────────────────────────────

def plot_graphs(N, trial_dir, figdir):
    """Row of publication-style graph drawings: GT | FC Matched | MI Matched | CL Tree.

    Uses hierarchical layout (BFS from highest-degree node) when the GT is a tree,
    Kamada-Kawai otherwise. All panels share the GT node positions. Edge alpha
    scales with edge weight so strong connections are more prominent.
    """
    analysis_path = os.path.join(trial_dir, "analysis.npz")
    if not os.path.exists(analysis_path):
        print(f"  Skipping graphs for {trial_dir} — analysis.npz not found")
        return None

    data = np.load(analysis_path)

    panels = [
        ("Ground Truth",       data["ground_truth_weights"]),
        ("FC Density Matched", data["fc_density"]),
        ("MI Density Matched", data["mi_density"]),
        ("CL Tree",            data["cl_adjacency"]),
    ]

    gt_graph = _graph_from_matrix(data["ground_truth_weights"])
    pos = _pub_layout(gt_graph)
    node_list = list(range(N))
    node_colors = ["#4C9BE8"] * N   # steel-blue for all nodes

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.5),
                             constrained_layout=True, facecolor="white")
    fig.suptitle(f"Graph Visualization — N = {N}", fontsize=14, fontweight="bold")

    for ax, (title, mat) in zip(axes, panels):
        G = _graph_from_matrix(mat)
        edges = list(G.edges())
        w_dict = {(i, j): abs(G[i][j]["weight"]) for i, j in edges}
        alphas = _edge_alphas(edges, w_dict)

        for (i, j) in edges:
            _draw_edge(ax, pos, i, j, color="#555555", lw=1.4, alpha=alphas[(i, j)])

        _draw_nodes(ax, pos, node_list, node_colors)
        _draw_labels(ax, pos, node_list, fontsize=max(6, 10 - N // 5))
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis("off")
        ax.set_facecolor("white")

    path = os.path.join(figdir, f"graphs_N{N}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path}")
    return fig


# ── Figure 4: Coupling-type graph drawings ────────────────────────────────

def plot_coupling_type_graphs(N, trial_dir, figdir):
    """Graph drawings that reveal which coupling types each method captures.

    Panels: Ground Truth | FC Density Matched | MI Density Matched | CL Tree

    Ground Truth edges are colored by coupling type (linear / quadrature /
    rectified / squared / PAC) with alpha proportional to coupling weight.
    For each recovered method, edges are drawn as:
      • TP (edge present in both GT and recovered) — GT coupling-type color
      • FN (edge in GT but not recovered)          — light gray dashed (missed)
      • FP (edge in recovered but not GT)          — very faint gray (de-emphasized)

    Requires dynamics_sim.npz (produced by sweep_sim.py) alongside analysis.npz.
    """
    analysis_path = os.path.join(trial_dir, "analysis.npz")
    dynamics_path = os.path.join(trial_dir, "dynamics_sim.npz")

    if not os.path.exists(analysis_path):
        print(f"  Skipping coupling graphs for {trial_dir} — analysis.npz not found")
        return None
    if not os.path.exists(dynamics_path):
        print(f"  Skipping coupling graphs for {trial_dir} — dynamics_sim.npz not found")
        return None

    an  = np.load(analysis_path)
    dyn = np.load(dynamics_path, allow_pickle=True)

    gt_weights    = an["ground_truth_weights"]
    coupling_types = dyn["coupling_types"].astype(int)   # (N, N)

    gt_edges  = _edge_binary(gt_weights)
    gt_graph  = _graph_from_matrix(gt_weights)
    pos       = _pub_layout(gt_graph)
    node_list = list(range(N))

    # GT weight dict for alpha scaling
    gt_w_dict = {(i, j): float(gt_weights[i, j]) for (i, j) in gt_edges}

    recovered_panels = [
        ("FC Density Matched", an["fc_density"]),
        ("MI Density Matched", an["mi_density"]),
        ("CL Tree",            an["cl_adjacency"]),
    ]

    n_panels = 1 + len(recovered_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6.5),
                             constrained_layout=True, facecolor="white")
    fig.suptitle(f"Coupling Types — N = {N}", fontsize=14, fontweight="bold")

    gt_alphas = _edge_alphas(list(gt_edges), gt_w_dict)

    # ── Ground Truth panel ──────────────────────────────────────────────────
    ax = axes[0]
    for (i, j) in gt_edges:
        ct    = int(coupling_types[i, j])
        color = TYPE_COLORS.get(ct, "#888888")
        _draw_edge(ax, pos, i, j, color=color, lw=1.8, alpha=gt_alphas[(i, j)])
    _draw_nodes(ax, pos, node_list, ["#EEEEEE"] * N)
    _draw_labels(ax, pos, node_list, fontsize=max(6, 10 - N // 5))
    ax.set_title("Ground Truth", fontsize=11, pad=8)
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Recovered method panels ─────────────────────────────────────────────
    for ax, (title, mat) in zip(axes[1:], recovered_panels):
        rec_edges = _edge_binary(mat)
        tp_edges  = gt_edges & rec_edges
        fp_edges  = rec_edges - gt_edges
        fn_edges  = gt_edges - rec_edges

        # FN first (drawn below TP so TP is on top)
        for (i, j) in fn_edges:
            _draw_edge(ax, pos, i, j, color=FN_COLOR, lw=1.0, alpha=0.5, linestyle="--")
        # FP: very faint gray, thin
        for (i, j) in fp_edges:
            _draw_edge(ax, pos, i, j, color="#CCCCCC", lw=0.8, alpha=0.35)
        # TP: coupling-type color, alpha from GT weight
        for (i, j) in tp_edges:
            ct    = int(coupling_types[i, j])
            color = TYPE_COLORS.get(ct, "#888888")
            _draw_edge(ax, pos, i, j, color=color, lw=1.8, alpha=gt_alphas.get((i, j), 0.8))

        _draw_nodes(ax, pos, node_list, ["#EEEEEE"] * N)
        _draw_labels(ax, pos, node_list, fontsize=max(6, 10 - N // 5))
        ax.set_title(f"{title}\nTP={len(tp_edges)}  FP={len(fp_edges)}  FN={len(fn_edges)}",
                     fontsize=10, pad=8)
        ax.axis("off")
        ax.set_facecolor("white")

    # ── Shared legend ────────────────────────────────────────────────────────
    legend_elements = [
        mlines.Line2D([0], [0], color=TYPE_COLORS[ct], linewidth=2.0,
                      label=f"TP: {name}")
        for ct, name in TYPE_NAMES.items()
    ] + [
        mlines.Line2D([0], [0], color="#CCCCCC", linewidth=1.0, alpha=0.5,
                      label="FP (spurious)"),
        mlines.Line2D([0], [0], color=FN_COLOR, linewidth=1.0,
                      linestyle="--", label="FN (missed)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    path = os.path.join(figdir, f"coupling_graphs_N{N}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path}")
    return fig


# ── Figure 5: Summary table ───────────────────────────────────────────────

def print_and_save_tables(N_values, arrays, figdir):
    """Print summary tables to stdout and save as a figure."""
    all_tables = []  # (N, cell_text, row_labels, col_labels)

    for ni, N in enumerate(N_values):
        print(f"\n{'='*90}")
        print(f"  N = {N}")
        print(f"{'='*90}")

        header = f"{'Method':<20}"
        for metric in DISPLAY_METRICS:
            header += f"  {METRIC_LABELS[metric]:>16}"
        print(header)
        print("-" * len(header))

        cell_text = []
        row_labels = []
        for method in METHODS:
            row_labels.append(METHOD_LABELS[method])
            row = []
            parts = [f"{METHOD_LABELS[method]:<20}"]
            for metric in DISPLAY_METRICS:
                key = f"{method}_{metric}"
                vals = arrays[key][ni]
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                cell = f"{mean:.3f}\u00b1{std:.3f}"
                row.append(cell)
                parts.append(f"  {mean:>7.3f}\u00b1{std:<7.3f}")
            cell_text.append(row)
            print("".join(parts))

        col_labels = [METRIC_LABELS[m] for m in DISPLAY_METRICS]
        all_tables.append((N, cell_text, row_labels, col_labels))

    print()

    # Save as figure — one table per N, stacked vertically
    n_tables = len(all_tables)
    fig, axes = plt.subplots(n_tables, 1, figsize=(14, 2.2 * n_tables),
                             constrained_layout=True)
    if n_tables == 1:
        axes = [axes]

    for ax, (N, cell_text, row_labels, col_labels) in zip(axes, all_tables):
        ax.axis("off")
        ax.set_title(f"N = {N}", fontsize=12, fontweight="bold", pad=8)
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        # Color row labels to match method colors
        for i, method in enumerate(METHODS):
            cell = table[i + 1, -1]  # row label cell
            cell.set_facecolor((*METHOD_COLORS[method][:3], 0.25))

    path = os.path.join(figdir, "metrics_table.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
    return fig


# ── Figure 6: Cross-type comparison ───────────────────────────────────────

def plot_metrics_comparison(type_data, figdir, transpose=False):
    """Grid of line plots comparing multiple connectivity types.

    Default layout  : rows = metrics,          columns = connectivity types.
    transpose=True  : rows = connectivity types, columns = metrics.

    Parameters
    ----------
    type_data  : list of (label, N_values, arrays)
    figdir     : str
    transpose  : bool
    """
    n_metrics = len(DISPLAY_METRICS)
    n_types = len(type_data)

    if transpose:
        nrows, ncols = n_types, n_metrics
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 3.2 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        # Column headers = metric names
        for col, metric in enumerate(DISPLAY_METRICS):
            axes[0, col].set_title(METRIC_LABELS[metric], fontsize=12,
                                   fontweight="bold", pad=10)

        for row, (label, N_values, arrays) in enumerate(type_data):
            # Row label on the left y-axis of the first column
            axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold",
                                    labelpad=8)
            for col, metric in enumerate(DISPLAY_METRICS):
                ax = axes[row, col]
                for method in METHODS:
                    key = f"{method}_{metric}"
                    if key not in arrays:
                        continue
                    mean = np.nanmean(arrays[key], axis=1)
                    ax.plot(N_values, mean, "o-", color=METHOD_COLORS[method],
                            linewidth=1.4, label=METHOD_LABELS[method])
                if row == nrows - 1:
                    ax.set_xlabel("N (nodes)", fontsize=9)
                ax.set_xticks(N_values)
                ax.set_xticklabels(
                    [str(n) if n == 5 or n >= 25 else "" for n in N_values],
                    fontsize=8,
                )
    else:
        nrows, ncols = n_metrics, n_types
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 3.2 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        for col, (label, N_values, arrays) in enumerate(type_data):
            axes[0, col].set_title(label, fontsize=13, fontweight="bold", pad=10)
            for row, metric in enumerate(DISPLAY_METRICS):
                ax = axes[row, col]
                for method in METHODS:
                    key = f"{method}_{metric}"
                    if key not in arrays:
                        continue
                    mean = np.nanmean(arrays[key], axis=1)
                    ax.plot(N_values, mean, "o-", color=METHOD_COLORS[method],
                            linewidth=1.4, label=METHOD_LABELS[method])
                if col == 0:
                    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
                if row == nrows - 1:
                    ax.set_xlabel("N (nodes)", fontsize=9)
                ax.set_xticks(N_values)
                ax.set_xticklabels(
                    [str(n) if n == 5 or n >= 25 else "" for n in N_values],
                    fontsize=8,
                )

    handles, label_list = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, label_list, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    path = os.path.join(figdir, "metrics_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
    return fig


# ── Multi-sweep helpers ────────────────────────────────────────────────────

def _parse_sweep_spec(spec):
    """Parse 'path/to/sweep[:N1,N2,...]' into (sweep_dir, N_filter_or_None)."""
    if ":" in spec:
        # Split on the LAST colon so Windows absolute paths (C:\...) aren't broken
        idx = spec.rfind(":")
        sweep_dir = spec[:idx]
        N_filter = [int(x) for x in spec[idx + 1:].split(",") if x.strip()]
    else:
        sweep_dir = spec
        N_filter = None
    return sweep_dir, N_filter


def _load_and_filter(sweep_dir, N_filter):
    """Load a sweep and apply an optional N filter.

    If sweep_results.npz exists but is missing requested N values (e.g. it was
    cached from a partial previous run), rebuilds it from per-trial analysis.npz.

    Returns (N_values ndarray, arrays dict).
    """
    N_values, _n_trials, arrays, _config = load_sweep(sweep_dir)
    if N_filter is not None:
        missing = sorted(set(N_filter) - set(N_values.tolist()))
        if missing:
            print(f"  N values {missing} missing from cached sweep_results.npz "
                  f"in {sweep_dir} — rebuilding from trial files...")
            N_values, _n_trials, arrays, _config = _build_sweep_from_trials(sweep_dir)
            still_missing = sorted(set(N_filter) - set(N_values.tolist()))
            if still_missing:
                raise ValueError(
                    f"Requested N values {still_missing} not found in {sweep_dir} "
                    f"(available: {sorted(N_values.tolist())})"
                )
        keep = np.array(sorted(set(N_filter)))
        indices = [int(np.where(N_values == n)[0][0]) for n in keep]
        N_values = keep
        arrays = {k: v[indices] if v.ndim >= 1 else v for k, v in arrays.items()}
    return N_values, arrays


def combine_sweeps(sweep_specs):
    """Load and merge multiple (sweep_dir, N_filter) specs into one dataset.

    N values must be unique across all sweeps. If trial counts differ, shorter
    arrays are NaN-padded to match the longest.

    Parameters
    ----------
    sweep_specs : list of (sweep_dir, N_filter_or_None)

    Returns
    -------
    N_values : sorted ndarray of all combined N values
    arrays   : dict with shape (total_N, max_trials), NaN-padded where needed
    """
    parts = []  # list of (N_values_part, arrays_part)
    for sweep_dir, N_filter in sweep_specs:
        print(f"Loading {sweep_dir}" + (f" N={N_filter}" if N_filter else ""))
        N_part, arr_part = _load_and_filter(sweep_dir, N_filter)
        parts.append((N_part, arr_part))

    # Validate uniqueness
    all_N_flat = [n for N_part, _ in parts for n in N_part.tolist()]
    seen, dupes = set(), set()
    for n in all_N_flat:
        (dupes if n in seen else seen).add(n)
    if dupes:
        raise ValueError(f"Duplicate N values across sweep dirs: {sorted(dupes)}")

    # Sort everything by N
    sort_order = np.argsort(all_N_flat)
    combined_N = np.array(sorted(all_N_flat))

    # Pad trial axes to uniform width and concatenate
    max_trials = max(
        next(iter(arr.values())).shape[1]
        for _, arr in parts
        if arr
    )

    all_keys = set(k for _, arr in parts for k in arr)
    combined_arrays = {}
    for key in all_keys:
        rows = []
        for N_part, arr in parts:
            if key in arr:
                v = arr[key]   # (n_N_part, n_trials_part)
                if v.shape[1] < max_trials:
                    pad = np.full((v.shape[0], max_trials - v.shape[1]), np.nan)
                    v = np.hstack([v, pad])
            else:
                v = np.full((len(N_part), max_trials), np.nan)
            rows.append(v)
        combined_arrays[key] = np.vstack(rows)[sort_order]

    return combined_N, combined_arrays


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize sweep results from sweep_sim.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Single-sweep mode:\n"
            "  python visualize_sweep.py --sweep-dir output/sweep [--N 10 20]\n\n"
            "Multi-sweep mode (metrics_vs_N and metrics_table only):\n"
            "  python visualize_sweep.py --sweeps output/sweep1 output/sweep2\n"
            "  python visualize_sweep.py --sweeps output/sweep1:5,10 output/sweep2:20,25\n\n"
            "Comparison mode (one column per connectivity type):\n"
            "  python visualize_sweep.py \\\n"
            "    --type sparse output/sweep_sparse:5,10 output/sweep_sparse_30-90:30,40 \\\n"
            "    --type tree   output/sweep_tree:5,10   output/sweep_tree_30-90:30,40 \\\n"
            "    --figdir output/comparison\n"
        ),
    )
    # Single-sweep mode
    parser.add_argument("--sweep-dir", default=None,
                        help="Path to a single sweep output directory")
    parser.add_argument("--N", type=int, nargs="+", metavar="N",
                        help="Subset of N values (single-sweep mode only)")
    parser.add_argument("--trial", type=int, default=0,
                        help="Trial index for matrix/graph figures (default: 0)")

    # Comparison mode
    parser.add_argument("--transpose", action="store_true",
                        help="In --type mode, use types as rows and metrics as "
                             "columns instead of the default (metrics as rows).")
    parser.add_argument("--type", nargs="+", action="append",
                        metavar="LABEL_OR_DIR",
                        help="Define one connectivity type for comparison. "
                             "First token is the label, remaining tokens are "
                             "sweep-dir specs (same format as --sweeps). "
                             "Repeat --type for each connectivity type.")

    # Multi-sweep mode
    parser.add_argument("--sweeps", nargs="+", metavar="DIR[:N,...]",
                        help="One or more sweep dirs, each optionally followed by "
                             "':N1,N2,...' to select specific N values. "
                             "Produces metrics_vs_N and metrics_table only.")

    # Shared
    parser.add_argument("--figdir", default=None,
                        help="Where to save figures. Required when using --sweeps "
                             "(default for --sweep-dir: <sweep-dir>/figures)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't call plt.show() (for headless use)")
    args = parser.parse_args()

    modes = sum([args.sweep_dir is not None, args.sweeps is not None,
                 args.type is not None])
    if modes == 0:
        parser.error("Provide one of --sweep-dir, --sweeps, or --type.")
    if modes > 1:
        parser.error("--sweep-dir, --sweeps, and --type are mutually exclusive.")

    # ── Comparison mode ──────────────────────────────────────────────────────
    if args.type is not None:
        figdir = args.figdir or "figures"
        os.makedirs(figdir, exist_ok=True)

        type_data = []
        for type_spec in args.type:
            if len(type_spec) < 2:
                parser.error(f"--type needs a label followed by at least one "
                             f"sweep-dir spec, got: {type_spec}")
            label = type_spec[0]
            sweep_specs = [_parse_sweep_spec(s) for s in type_spec[1:]]
            print(f"\nLoading type '{label}'...")
            N_values, arrays = combine_sweeps(sweep_specs)
            print(f"  Combined N values: {N_values.tolist()}")
            type_data.append((label, N_values, arrays))

        print("\n--- Metrics comparison ---")
        plot_metrics_comparison(type_data, figdir, transpose=args.transpose)

        if not args.no_show:
            plt.show()
        print("\nDone.")
        return

    # ── Multi-sweep mode ─────────────────────────────────────────────────────
    if args.sweeps is not None:
        if args.N is not None:
            parser.error("--N is not valid with --sweeps; embed N values in the "
                         "spec instead, e.g. --sweeps dir:10,20")
        figdir = args.figdir or "figures"
        os.makedirs(figdir, exist_ok=True)

        specs = [_parse_sweep_spec(s) for s in args.sweeps]
        N_values, arrays = combine_sweeps(specs)
        print(f"Combined N values: {N_values.tolist()}")

        print("\n--- Metrics vs N ---")
        plot_metrics_vs_N(N_values, arrays, figdir)

        print("\n--- Summary Tables ---")
        print_and_save_tables(N_values, arrays, figdir)

        if not args.no_show:
            plt.show()
        print("\nDone.")
        return

    # ── Single-sweep mode ────────────────────────────────────────────────────
    sweep_dir = args.sweep_dir
    figdir = args.figdir or os.path.join(sweep_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print(f"Loading sweep results from {sweep_dir}")
    N_values, n_trials, arrays, _config = load_sweep(sweep_dir)
    print(f"N values: {N_values}, trials: {n_trials}")

    if args.N is not None:
        invalid = sorted(set(args.N) - set(N_values.tolist()))
        if invalid:
            parser.error(f"Requested N values not found in sweep: {invalid}. "
                         f"Available: {sorted(N_values.tolist())}")
        keep = np.array(sorted(set(args.N)))
        indices = [int(np.where(N_values == n)[0][0]) for n in keep]
        N_values = keep
        arrays = {k: v[indices] if v.ndim >= 1 else v for k, v in arrays.items()}

    # 1. Metrics vs N
    print("\n--- Metrics vs N ---")
    plot_metrics_vs_N(N_values, arrays, figdir)

    # 2 & 3. Per-N matrix heatmaps and graph drawings
    for N in N_values:
        trial_dir = os.path.join(sweep_dir, f"N{N}", f"trial{args.trial}")
        if not os.path.isdir(trial_dir):
            print(f"  Trial dir not found: {trial_dir}, skipping")
            continue

        print(f"\n--- Matrices N={N} (trial {args.trial}) ---")
        plot_matrices(N, trial_dir, figdir)

        print(f"\n--- Graphs N={N} (trial {args.trial}) ---")
        plot_graphs(N, trial_dir, figdir)

        print(f"\n--- Coupling-type graphs N={N} (trial {args.trial}) ---")
        plot_coupling_type_graphs(N, trial_dir, figdir)

    # 4. Summary table
    print("\n--- Summary Tables ---")
    print_and_save_tables(N_values, arrays, figdir)

    if not args.no_show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
