"""
grid_visualize.py — Recreate visualize_sweep figures for a grid sim run.

Produces (into <run_dir>/analysis/):
  metrics_vs_noise.png        — metrics vs noise level D (all methods, per conn type)
  metrics_vs_N.png            — metrics vs N (all methods, per conn type)
  matrices_N{N}_G{G}_{c}.png  — heatmap panel per condition (low-noise rep.)
  graphs_N{N}_G{G}_{c}.png    — graph drawings per condition
  coupling_N{N}_G{G}_{c}.png  — coupling-type edge drawings per condition
  metrics_table.png           — summary table

Usage:
    python code/simulations/grid_visualize.py --run-id 20260407_133706_grid
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "code", "simulations"))

# Re-use drawing helpers and evaluate() from existing scripts
from visualize_sweep import (
    _graph_from_matrix, _edge_binary, _pub_layout,
    _draw_nodes, _draw_edge, _draw_labels, _edge_alphas,
    TYPE_NAMES, TYPE_COLORS, FP_COLOR, FN_COLOR,
)
from analyze_sim import evaluate, density_threshold_matrix

# ---------------------------------------------------------------------------
# Constants — adapted method set for the grid
# ---------------------------------------------------------------------------

METHODS = ["fc_matched", "mi_matched", "cl_adjacency"]
METHOD_LABELS = {
    "fc_matched":   "FC Matched",
    "mi_matched":   "MI Matched",
    "cl_adjacency": "CL Tree",
}
_COLORS = plt.cm.tab10.colors
METHOD_COLORS = {
    "fc_matched":   _COLORS[2],   # green
    "mi_matched":   _COLORS[3],   # red
    "cl_adjacency": _COLORS[4],   # purple
}

DISPLAY_METRICS = ["pearson", "f1", "sensitivity", "specificity", "precision", "mse"]
METRIC_LABELS = {
    "pearson":     "Pearson r",
    "f1":          "F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision":   "Precision",
    "mse":         "MSE",
}
_EVAL_KEY_MAP = {
    "pearson":     "pearson_correlation",
    "f1":          "f1",
    "sensitivity": "sensitivity",
    "specificity": "specificity",
    "precision":   "precision",
    "mse":         "mean_squared_error",
}

CONN_LABELS = {"er_0p10": "ER p=0.10", "er_0p50": "ER p=0.50", "tree": "Tree"}
CONN_ORDER  = ["er_0p10", "er_0p50", "tree"]
CONN_COLORS = {"er_0p10": "#2166ac", "er_0p50": "#d6604d", "tree": "#1a9850"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cache(run_id):
    cache_path = os.path.join(
        _REPO_ROOT, "datasets", "synthetic", run_id,
        "analysis", "results_cache.npz"
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"results_cache.npz not found at {cache_path}. "
            f"Run grid_analysis.py first."
        )
    data = np.load(cache_path, allow_pickle=True)
    return dict(data["results"].item())


def timeseries_base(run_id):
    return os.path.join(_REPO_ROOT, "datasets", "synthetic", run_id, "timeseries")


def load_gt_weights_and_coupling(run_id, N, G, conn_tag, D):
    """Load ground-truth weighted matrix and coupling_types from original NPZ."""
    base = timeseries_base(run_id)
    # Find the file for this individual
    cell_dir = os.path.join(base, f"N{N}", f"G{G:.2f}", conn_tag)
    for fname in os.listdir(cell_dir):
        d_val = float(fname.split("_D")[1].replace(".npz", ""))
        if abs(d_val - D) / max(D, 1e-30) < 0.01:
            d = np.load(os.path.join(cell_dir, fname), allow_pickle=True)
            return d["conn"], d["coupling_types"].astype(int)
    raise FileNotFoundError(f"No file for D={D} in {cell_dir}")


# ---------------------------------------------------------------------------
# Build sweep-style metric arrays from cache
# ---------------------------------------------------------------------------

def build_metric_arrays(results, run_id):
    """
    Compute pearson/f1/sensitivity/specificity/precision/mse for every cell
    using analyze_sim.evaluate(), return dicts organized for plotting.

    Returns
    -------
    metric_data : dict
        metric_data[(N, G, conn_tag)][D][method][metric] = float
    """
    metric_data = {}
    total = sum(len(cell) for cell in results.values())
    done = 0

    for (N, G, conn_tag), cell in sorted(results.items()):
        metric_data[(N, G, conn_tag)] = {}
        for D, m in sorted(cell.items()):
            print(f"  [{done+1}/{total}] N={N} G={G} {conn_tag} D={D:.0e}")
            gt_weights, _ = load_gt_weights_and_coupling(run_id, N, G, conn_tag, D)
            fc_raw  = m["fc_raw"]
            mi_raw  = m["mi_raw"]
            cl_adj  = m["cl_adj"].astype(float)

            n_gt = int(m["n_gt_edges"])
            fc_matched = density_threshold_matrix(fc_raw, n_edges=n_gt)
            mi_matched = density_threshold_matrix(mi_raw, n_edges=n_gt)

            matrices = {
                "fc_matched":   fc_matched,
                "mi_matched":   mi_matched,
                "cl_adjacency": cl_adj,
            }
            ev = evaluate(gt_weights, matrices)
            metric_data[(N, G, conn_tag)][D] = ev
            done += 1

    return metric_data


# ---------------------------------------------------------------------------
# Figure 1: metrics vs noise level D
# ---------------------------------------------------------------------------

def plot_metrics_vs_noise(metric_data, D_vals, figdir):
    """2x3 grid: each panel = one metric, x=D, lines = (method, conn_tag)."""
    conn_tags = CONN_ORDER

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Metrics vs Noise Level", fontsize=14, fontweight="bold")
    axes_flat = axes.flat

    ls_per_conn = {"er_0p10": "-", "er_0p50": "--", "tree": ":"}

    for ax, metric in zip(axes_flat, DISPLAY_METRICS):
        for method in METHODS:
            for conn_tag in conn_tags:
                # Collect values for each D, averaging across N and G
                means, stds = [], []
                for D in D_vals:
                    vals = []
                    for (N, G, ct), cell in metric_data.items():
                        if ct != conn_tag or D not in cell:
                            continue
                        v = cell[D].get(method, {}).get(_EVAL_KEY_MAP[metric], np.nan)
                        vals.append(v)
                    means.append(np.nanmean(vals))
                    stds.append(np.nanstd(vals))

                x = np.log10(D_vals)
                means = np.array(means)
                stds  = np.array(stds)
                label = f"{METHOD_LABELS[method]} / {CONN_LABELS.get(conn_tag, conn_tag)}"
                ax.plot(x, means, "o" + ls_per_conn[conn_tag],
                        color=METHOD_COLORS[method], linewidth=1.6, label=label)
                ax.fill_between(x, means - stds, means + stds,
                                color=METHOD_COLORS[method], alpha=0.10)

        exp_ticks = [int(np.log10(D)) for D in D_vals]
        ax.set_xticks(exp_ticks)
        ax.set_xticklabels([rf"$10^{{{e}}}$" for e in exp_ticks], fontsize=9)
        ax.set_xlabel("Noise level D", fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.grid(True, alpha=0.3)

    # Shared legend — methods
    method_handles = [
        mlines.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2,
                      label=METHOD_LABELS[m])
        for m in METHODS
    ]
    conn_handles = [
        mlines.Line2D([0], [0], color="gray", ls=ls_per_conn[c], linewidth=2,
                      label=CONN_LABELS.get(c, c))
        for c in conn_tags
    ]
    fig.legend(handles=method_handles + conn_handles,
               loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    path = os.path.join(figdir, "metrics_vs_noise.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: metrics vs N
# ---------------------------------------------------------------------------

def plot_metrics_vs_N(metric_data, N_vals, figdir):
    """2x3 grid: each panel = one metric, x=N, lines = (method, conn_tag).
       Averages across D and G (treating them as repeated measurements).
    """
    conn_tags = CONN_ORDER
    ls_per_conn = {"er_0p10": "-", "er_0p50": "--", "tree": ":"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Metrics vs Network Size (N)", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes.flat, DISPLAY_METRICS):
        for method in METHODS:
            for conn_tag in conn_tags:
                means, stds = [], []
                for N in N_vals:
                    vals = []
                    for (Nc, G, ct), cell in metric_data.items():
                        if Nc != N or ct != conn_tag:
                            continue
                        for D, ev in cell.items():
                            v = ev.get(method, {}).get(_EVAL_KEY_MAP[metric], np.nan)
                            vals.append(v)
                    means.append(np.nanmean(vals))
                    stds.append(np.nanstd(vals))

                means = np.array(means)
                stds  = np.array(stds)
                ax.plot(N_vals, means, "o" + ls_per_conn[conn_tag],
                        color=METHOD_COLORS[method], linewidth=1.6)
                ax.fill_between(N_vals, means - stds, means + stds,
                                color=METHOD_COLORS[method], alpha=0.10)

        ax.set_xticks(N_vals)
        ax.set_xlabel("N (nodes)", fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.grid(True, alpha=0.3)

    method_handles = [
        mlines.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2,
                      label=METHOD_LABELS[m])
        for m in METHODS
    ]
    conn_handles = [
        mlines.Line2D([0], [0], color="gray", ls=ls_per_conn[c], linewidth=2,
                      label=CONN_LABELS.get(c, c))
        for c in conn_tags
    ]
    fig.legend(handles=method_handles + conn_handles,
               loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    path = os.path.join(figdir, "metrics_vs_N.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Adjacency matrix heatmaps
# ---------------------------------------------------------------------------

def plot_matrices(results, run_id, N, G, conn_tag, D_rep, figdir):
    """Heatmap row: GT | FC Matched | MI Raw | MI Matched | CL Tree."""
    m = results.get((N, G, conn_tag), {}).get(D_rep)
    if m is None:
        return

    gt_weights, _ = load_gt_weights_and_coupling(run_id, N, G, conn_tag, D_rep)
    fc_raw  = m["fc_raw"]
    mi_raw  = m["mi_raw"]
    cl_adj  = m["cl_adj"].astype(float)
    n_gt    = int(m["n_gt_edges"])
    fc_mat  = density_threshold_matrix(fc_raw, n_edges=n_gt)
    mi_mat  = density_threshold_matrix(mi_raw, n_edges=n_gt)

    panels = [
        ("Ground Truth",    gt_weights, "viridis", False),
        ("FC Matched",      fc_mat,     "RdBu_r",  True),
        ("MI Raw",          mi_raw,     "viridis", False),
        ("MI Matched",      mi_mat,     "viridis", False),
        ("CL Tree",         cl_adj,     "viridis", False),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.5),
                             constrained_layout=True)
    fig.suptitle(
        f"Adjacency Matrices — N={N}  G={G}  {CONN_LABELS.get(conn_tag, conn_tag)}"
        f"  D=1e{int(np.log10(D_rep))}",
        fontsize=12,
    )

    for ax, (title, mat, cmap, symmetric) in zip(axes, panels):
        mat = mat.astype(float)
        raw_min, raw_max = float(mat.min()), float(mat.max())
        if symmetric:
            abs_max = max(abs(raw_min), abs(raw_max), 1e-12)
            mat_disp, vmin, vmax = mat, -abs_max, abs_max
            cbar_ticks  = [-abs_max, 0.0, abs_max]
            cbar_labels = [f"{-abs_max:.2f}", "0", f"{abs_max:.2f}"]
        else:
            span = raw_max - raw_min
            if span < 1e-12:
                mat_disp, span = np.zeros_like(mat), 1.0
            else:
                mat_disp = (mat - raw_min) / span
            vmin, vmax = 0.0, 1.0
            mid = raw_min + span / 2
            cbar_ticks  = [0.0, 0.5, 1.0]
            cbar_labels = [f"{raw_min:.3g}", f"{mid:.3g}", f"{raw_max:.3g}"]

        im = ax.imshow(mat_disp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(cbar_ticks); cbar.set_ticklabels(cbar_labels)
        cbar.ax.tick_params(labelsize=7)

    tag = f"N{N}_G{str(G).replace('.','p')}_{conn_tag}"
    path = os.path.join(figdir, f"matrices_{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Graph drawings
# ---------------------------------------------------------------------------

def plot_graphs(results, run_id, N, G, conn_tag, D_rep, figdir):
    """Publication-style graph row: GT | FC Matched | MI Matched | CL Tree."""
    m = results.get((N, G, conn_tag), {}).get(D_rep)
    if m is None:
        return

    gt_weights, _ = load_gt_weights_and_coupling(run_id, N, G, conn_tag, D_rep)
    fc_raw = m["fc_raw"]
    mi_raw = m["mi_raw"]
    cl_adj = m["cl_adj"].astype(float)
    n_gt   = int(m["n_gt_edges"])
    fc_mat = density_threshold_matrix(fc_raw, n_edges=n_gt)
    mi_mat = density_threshold_matrix(mi_raw, n_edges=n_gt)

    panels = [
        ("Ground Truth",  gt_weights),
        ("FC Matched",    fc_mat),
        ("MI Matched",    mi_mat),
        ("CL Tree",       cl_adj),
    ]

    gt_graph = _graph_from_matrix(gt_weights)
    pos = _pub_layout(gt_graph)
    node_list = list(range(N))
    node_colors = ["#4C9BE8"] * N

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.5),
                             constrained_layout=True, facecolor="white")
    fig.suptitle(
        f"Graph Visualization — N={N}  G={G}  "
        f"{CONN_LABELS.get(conn_tag, conn_tag)}  D=1e{int(np.log10(D_rep))}",
        fontsize=13, fontweight="bold",
    )

    for ax, (title, mat) in zip(axes, panels):
        G_nx = _graph_from_matrix(mat)
        edges = list(G_nx.edges())
        w_dict = {(i, j): abs(G_nx[i][j]["weight"]) for i, j in edges}
        alphas = _edge_alphas(edges, w_dict)
        for (i, j) in edges:
            _draw_edge(ax, pos, i, j, color="#555555", lw=1.4,
                       alpha=alphas.get((i, j), 0.7))
        _draw_nodes(ax, pos, node_list, node_colors)
        _draw_labels(ax, pos, node_list, fontsize=max(6, 10 - N // 5))
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis("off"); ax.set_facecolor("white")

    tag = f"N{N}_G{str(G).replace('.','p')}_{conn_tag}"
    path = os.path.join(figdir, f"graphs_{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Coupling-type graph drawings
# ---------------------------------------------------------------------------

def plot_coupling_graphs(results, run_id, N, G, conn_tag, D_rep, figdir):
    """GT edges colored by coupling type; recovered edges as TP/FP/FN."""
    m = results.get((N, G, conn_tag), {}).get(D_rep)
    if m is None:
        return

    gt_weights, coupling_types = load_gt_weights_and_coupling(
        run_id, N, G, conn_tag, D_rep
    )
    fc_raw = m["fc_raw"]
    mi_raw = m["mi_raw"]
    cl_adj = m["cl_adj"].astype(float)
    n_gt   = int(m["n_gt_edges"])
    fc_mat = density_threshold_matrix(fc_raw, n_edges=n_gt)
    mi_mat = density_threshold_matrix(mi_raw, n_edges=n_gt)

    gt_edges = _edge_binary(gt_weights)
    gt_graph  = _graph_from_matrix(gt_weights)
    pos       = _pub_layout(gt_graph)
    node_list = list(range(N))
    gt_w_dict = {(i, j): float(gt_weights[i, j]) for (i, j) in gt_edges}
    gt_alphas = _edge_alphas(list(gt_edges), gt_w_dict)

    recovered_panels = [
        ("FC Matched",  fc_mat),
        ("MI Matched",  mi_mat),
        ("CL Tree",     cl_adj),
    ]

    n_panels = 1 + len(recovered_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6.5),
                             constrained_layout=True, facecolor="white")
    fig.suptitle(
        f"Coupling Types — N={N}  G={G}  "
        f"{CONN_LABELS.get(conn_tag, conn_tag)}  D=1e{int(np.log10(D_rep))}",
        fontsize=13, fontweight="bold",
    )

    # Ground Truth panel
    ax = axes[0]
    for (i, j) in gt_edges:
        ct = int(coupling_types[i, j])
        color = TYPE_COLORS.get(ct, "#888888")
        _draw_edge(ax, pos, i, j, color=color, lw=1.8,
                   alpha=gt_alphas.get((i, j), 0.8))
    _draw_nodes(ax, pos, node_list, ["#EEEEEE"] * N)
    _draw_labels(ax, pos, node_list, fontsize=max(6, 10 - N // 5))
    ax.set_title("Ground Truth", fontsize=11, pad=8)
    ax.axis("off"); ax.set_facecolor("white")

    # Recovered panels
    for ax, (title, mat) in zip(axes[1:], recovered_panels):
        rec_edges = _edge_binary(mat)
        tp = gt_edges & rec_edges
        fp = rec_edges - gt_edges
        fn = gt_edges - rec_edges

        for (i, j) in fn:
            _draw_edge(ax, pos, i, j, color=FN_COLOR, lw=1.0, alpha=0.5,
                       linestyle="--")
        for (i, j) in fp:
            _draw_edge(ax, pos, i, j, color="#CCCCCC", lw=0.8, alpha=0.35)
        for (i, j) in tp:
            ct = int(coupling_types[i, j])
            color = TYPE_COLORS.get(ct, "#888888")
            _draw_edge(ax, pos, i, j, color=color, lw=1.8,
                       alpha=gt_alphas.get((i, j), 0.8))
        _draw_nodes(ax, pos, node_list, ["#EEEEEE"] * N)
        _draw_labels(ax, pos, node_list, fontsize=max(6, 10 - N // 5))
        ax.set_title(
            f"{title}\nTP={len(tp)}  FP={len(fp)}  FN={len(fn)}",
            fontsize=10, pad=8,
        )
        ax.axis("off"); ax.set_facecolor("white")

    legend_handles = [
        mlines.Line2D([0], [0], color=TYPE_COLORS[ct], lw=2,
                      label=f"TP: {name}")
        for ct, name in TYPE_NAMES.items()
    ] + [
        mlines.Line2D([0], [0], color="#CCCCCC", lw=1.0, alpha=0.5,
                      label="FP (spurious)"),
        mlines.Line2D([0], [0], color=FN_COLOR, lw=1.0, ls="--",
                      label="FN (missed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    tag = f"N{N}_G{str(G).replace('.','p')}_{conn_tag}"
    path = os.path.join(figdir, f"coupling_{tag}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Summary metrics table
# ---------------------------------------------------------------------------

def plot_metrics_table(metric_data, N_vals, D_vals, figdir):
    """One table per N: rows = methods, cols = metrics.
       Values = mean +- std across all (G, conn_tag, D).
    """
    n_tables = len(N_vals)
    fig, axes = plt.subplots(n_tables, 1,
                             figsize=(14, 2.2 * n_tables),
                             constrained_layout=True)
    if n_tables == 1:
        axes = [axes]
    fig.suptitle("Metrics Summary (mean +- std across conditions)",
                 fontsize=12, fontweight="bold")

    print()
    for ax, N in zip(axes, N_vals):
        print(f"\n{'='*80}  N = {N}  {'='*80}")
        header = f"{'Method':<16}"
        for m in DISPLAY_METRICS:
            header += f"  {METRIC_LABELS[m]:>18}"
        print(header); print("-" * len(header))

        cell_text, row_labels = [], []
        for method in METHODS:
            row_labels.append(METHOD_LABELS[method])
            row = []
            parts = [f"{METHOD_LABELS[method]:<16}"]
            for metric in DISPLAY_METRICS:
                vals = []
                for (Nc, G, ct), cell in metric_data.items():
                    if Nc != N:
                        continue
                    for D, ev in cell.items():
                        v = ev.get(method, {}).get(_EVAL_KEY_MAP[metric], np.nan)
                        vals.append(v)
                mean = np.nanmean(vals)
                std  = np.nanstd(vals)
                row.append(f"{mean:.3f}\u00b1{std:.3f}")
                parts.append(f"  {mean:>7.3f}\u00b1{std:<7.3f}")
            cell_text.append(row)
            print("".join(parts))

        col_labels = [METRIC_LABELS[m] for m in DISPLAY_METRICS]
        ax.axis("off")
        ax.set_title(f"N = {N}", fontsize=12, fontweight="bold", pad=8)
        tbl = ax.table(
            cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)
        for i, method in enumerate(METHODS):
            tbl[i + 1, -1].set_facecolor((*METHOD_COLORS[method][:3], 0.25))

    print()
    path = os.path.join(figdir, "metrics_table.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20260407_133706_grid")
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory (default: datasets/synthetic/<run_id>/analysis)",
    )
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(
        _REPO_ROOT, "datasets", "synthetic", args.run_id, "analysis"
    )
    os.makedirs(outdir, exist_ok=True)

    print(f"Run ID : {args.run_id}")
    print(f"Output : {outdir}")
    print("Loading cache ...")
    results = load_cache(args.run_id)

    N_vals    = sorted({k[0] for k in results})
    G_vals    = sorted({k[1] for k in results})
    conn_tags = sorted({k[2] for k in results},
                       key=lambda c: CONN_ORDER.index(c) if c in CONN_ORDER else 99)
    D_vals    = sorted({D for cell in results.values() for D in cell})
    D_rep     = min(D_vals)   # representative = lowest noise

    print(f"N={N_vals}  G={G_vals}  conn={conn_tags}  D={D_vals}")
    print(f"Representative D = {D_rep:.0e}")

    # ── Compute metrics from raw matrices ──────────────────────────────────
    print("\nComputing metrics via evaluate() ...")
    metric_data = build_metric_arrays(results, args.run_id)

    # ── Figure 1: metrics vs noise ─────────────────────────────────────────
    print("\n--- Metrics vs Noise ---")
    plot_metrics_vs_noise(metric_data, D_vals, outdir)

    # ── Figure 2: metrics vs N ─────────────────────────────────────────────
    print("\n--- Metrics vs N ---")
    plot_metrics_vs_N(metric_data, N_vals, outdir)

    # ── Figures 3/4/5: per-condition matrix, graph, coupling ───────────────
    for N in N_vals:
        for G in G_vals:
            for conn_tag in conn_tags:
                if (N, G, conn_tag) not in results:
                    continue
                tag = f"N={N} G={G} {conn_tag}"
                print(f"\n--- {tag} ---")
                print(f"  Matrices ...")
                plot_matrices(results, args.run_id, N, G, conn_tag, D_rep, outdir)
                print(f"  Graphs ...")
                plot_graphs(results, args.run_id, N, G, conn_tag, D_rep, outdir)
                print(f"  Coupling graphs ...")
                plot_coupling_graphs(results, args.run_id, N, G, conn_tag, D_rep, outdir)

    # ── Figure 6: summary table ────────────────────────────────────────────
    print("\n--- Metrics Table ---")
    plot_metrics_table(metric_data, N_vals, D_vals, outdir)

    print(f"\nDone. Figures in {outdir}")


if __name__ == "__main__":
    main()
