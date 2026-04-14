"""
grid_analysis.py — CL vs. FC comparison for a grid simulation run.

Produces figures:
    fig_a_recovery.png          – edge & structure recovery vs noise level
    fig_b_consistency.png       – per-condition inter-individual consistency
    fig_b_consistency_summary.png – consistency summary across conditions
    fig_structure_heatmaps.png  – adjacency heatmaps (GT vs CL vs FC)
    fig_summary_bars.png        – aggregate bar chart across all conditions

Usage:
    python code/simulations/grid_analysis.py --run-id 20260407_133706_grid
    python code/simulations/grid_analysis.py   # default run ID
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONN_LABELS = {"er_0p10": "ER p=0.10", "er_0p50": "ER p=0.50", "tree": "Tree"}
CONN_ORDER  = ["er_0p10", "er_0p50", "tree"]
CONN_COLORS = {"er_0p10": "#2166ac", "er_0p50": "#d6604d", "tree": "#1a9850"}

METHOD_COLORS = {"CL": "#5e3c99", "FC": "#e66101"}
METHOD_LS     = {"CL": "-", "FC": "--"}


# ---------------------------------------------------------------------------
# MI utilities (self-contained, GPU-accelerated)
# ---------------------------------------------------------------------------

def _discretize(ts: torch.Tensor, num_bins: int) -> torch.Tensor:
    ts = ts.contiguous()
    edges = torch.linspace(float(ts.min()), float(ts.max()),
                           steps=num_bins + 1, device=ts.device)[1:-1]
    return torch.bucketize(ts, edges)


def compute_mi_matrix(V: np.ndarray, num_bins: int = 50) -> np.ndarray:
    """Return full MI matrix (N, N) for timeseries V (T, N)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = torch.tensor(V.T, dtype=torch.float32, device=device)  # (N, T)
    N = ts.shape[0]
    disc = _discretize(ts, num_bins)

    mi = torch.zeros(N, N, dtype=torch.float32, device=device)
    for i in range(N):
        for j in range(i + 1, N):
            idx = num_bins * disc[i] + disc[j]
            jp = torch.bincount(idx, minlength=num_bins * num_bins).float()
            jp = jp.reshape(num_bins, num_bins)
            jp /= jp.sum()
            ma = jp.sum(dim=1)
            mb = jp.sum(dim=0)
            pm = torch.outer(ma, mb)
            jp_s = jp.clone(); jp_s[jp_s == 0] = 1e-10
            pm_s = pm.clone(); pm_s[pm_s == 0] = 1e-10
            val = (jp_s * torch.log(jp_s / pm_s)).sum()
            mi[i, j] = val
            mi[j, i] = val
    return mi.cpu().numpy()


# ---------------------------------------------------------------------------
# FC utilities
# ---------------------------------------------------------------------------

def compute_fc_matrix(V: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix (T, N) -> (N, N)."""
    C = np.corrcoef(V, rowvar=False)
    np.fill_diagonal(C, 0.0)
    return C


def density_threshold(C: np.ndarray, n_edges: int) -> np.ndarray:
    """Keep the top n_edges by |value| in the upper triangle."""
    C = C.copy()
    np.fill_diagonal(C, 0.0)
    W = np.abs(C)
    iu = np.triu_indices_from(W, k=1)
    w = W[iu]
    if n_edges <= 0 or n_edges > len(w):
        return C
    thresh_idx = np.argpartition(w, -n_edges)[-n_edges:]
    mask = np.zeros(len(w), dtype=bool)
    mask[thresh_idx] = True
    out = np.zeros_like(C)
    out[iu[0][mask], iu[1][mask]] = C[iu[0][mask], iu[1][mask]]
    return out + out.T


# ---------------------------------------------------------------------------
# Chow-Liu tree
# ---------------------------------------------------------------------------

def chow_liu_tree(score_matrix: np.ndarray) -> np.ndarray:
    """Maximum spanning tree of score_matrix. Returns binary adjacency (N, N)."""
    N = score_matrix.shape[0]
    G = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=float(score_matrix[i, j]))
    tree = nx.maximum_spanning_tree(G, algorithm="kruskal", weight="weight")
    A = nx.to_numpy_array(tree, nodelist=sorted(tree.nodes()))
    return (A > 0).astype(int)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def binarize(conn: np.ndarray) -> np.ndarray:
    A = (np.abs(conn) > 0).astype(int)
    np.fill_diagonal(A, 0)
    return A


def upper_tri(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def auc_roc(score: np.ndarray, gt: np.ndarray) -> float:
    s = upper_tri(np.abs(score))
    y = upper_tri(gt)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def auc_pr(score: np.ndarray, gt: np.ndarray) -> float:
    s = upper_tri(np.abs(score))
    y = upper_tri(gt)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def f1_prec_rec(pred: np.ndarray, gt: np.ndarray):
    p = upper_tri(pred).astype(bool)
    g = upper_tri(gt).astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def degree_corr(pred: np.ndarray, gt: np.ndarray) -> float:
    r, _ = spearmanr(pred.sum(1), gt.sum(1))
    return float(r) if not np.isnan(r) else 0.0


def jaccard(adj1: np.ndarray, adj2: np.ndarray) -> float:
    e1 = upper_tri(adj1).astype(bool)
    e2 = upper_tri(adj2).astype(bool)
    inter = int((e1 & e2).sum())
    union = int((e1 | e2).sum())
    return inter / union if union > 0 else 1.0


def matrix_corr(M1: np.ndarray, M2: np.ndarray) -> float:
    v1 = upper_tri(M1)
    v2 = upper_tri(M2)
    if v1.std() == 0 or v2.std() == 0:
        return float("nan")
    r, _ = pearsonr(v1, v2)
    return float(r)


# ---------------------------------------------------------------------------
# Per-individual analysis
# ---------------------------------------------------------------------------

def analyze_individual(V: np.ndarray, conn_gt: np.ndarray, num_bins: int = 50) -> dict:
    gt = binarize(conn_gt)
    n_gt = int(upper_tri(gt).sum())

    fc  = compute_fc_matrix(V)
    mi  = compute_mi_matrix(V, num_bins=num_bins)

    cl_adj     = chow_liu_tree(mi)           # N-1 edges (tree)
    fc_matched = binarize(density_threshold(fc, n_gt))

    return dict(
        fc_raw     = fc,
        mi_raw     = mi,
        cl_adj     = cl_adj,
        fc_matched = fc_matched,
        gt_binary  = gt,
        n_gt_edges = n_gt,
        # --- AUC (continuous scores) ---
        auc_cl     = auc_roc(mi, gt),
        auc_fc     = auc_roc(fc, gt),
        auc_pr_cl  = auc_pr(mi, gt),
        auc_pr_fc  = auc_pr(fc, gt),
        # --- F1 at matched edge count ---
        f1_cl      = f1_prec_rec(cl_adj,     gt)[0],
        prec_cl    = f1_prec_rec(cl_adj,     gt)[1],
        rec_cl     = f1_prec_rec(cl_adj,     gt)[2],
        f1_fc      = f1_prec_rec(fc_matched, gt)[0],
        prec_fc    = f1_prec_rec(fc_matched, gt)[1],
        rec_fc     = f1_prec_rec(fc_matched, gt)[2],
        # --- Degree sequence correlation ---
        deg_corr_cl = degree_corr(cl_adj,     gt),
        deg_corr_fc = degree_corr(fc_matched, gt),
    )


# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------

def timeseries_base(run_id: str) -> str:
    return os.path.join(_REPO_ROOT, "datasets", "synthetic", run_id, "timeseries")


def collect_results(run_id: str, num_bins: int = 50, cache_path: str = None) -> dict:
    """
    Returns nested dict:  results[(N, G, conn_tag)][D] = metrics_dict
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        return dict(np.load(cache_path, allow_pickle=True)["results"].item())

    base = timeseries_base(run_id)

    # Enumerate files
    cells: dict = {}
    for N_dir in sorted(os.listdir(base)):
        if not N_dir.startswith("N"):
            continue
        N = int(N_dir[1:])
        for G_dir in sorted(os.listdir(os.path.join(base, N_dir))):
            if not G_dir.startswith("G"):
                continue
            G = float(G_dir[1:])
            for conn_tag in sorted(os.listdir(os.path.join(base, N_dir, G_dir))):
                cell_dir = os.path.join(base, N_dir, G_dir, conn_tag)
                for fname in sorted(os.listdir(cell_dir)):
                    if not fname.endswith(".npz"):
                        continue
                    D = float(fname.split("_D")[1].replace(".npz", ""))
                    cells.setdefault((N, G, conn_tag), []).append(
                        (D, os.path.join(cell_dir, fname))
                    )

    total = sum(len(v) for v in cells.values())
    results = {}
    done = 0

    for cell_key, individuals in sorted(cells.items()):
        N, G, conn_tag = cell_key
        results[cell_key] = {}
        for D, path in sorted(individuals):
            print(f"  [{done+1}/{total}] N={N} G={G} conn={conn_tag} D={D:.0e} ...",
                  end=" ", flush=True)
            d = np.load(path, allow_pickle=True)
            V, conn_gt = d["V"], d["conn"]
            m = analyze_individual(V, conn_gt, num_bins=num_bins)
            print(f"F1_CL={m['f1_cl']:.3f} F1_FC={m['f1_fc']:.3f} "
                  f"AUC_CL={m['auc_cl']:.3f} AUC_FC={m['auc_fc']:.3f}")
            results[cell_key][D] = m
            done += 1

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, results=results)
        print(f"\nCached -> {cache_path}")

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lognoise(d):
    exp = int(round(np.log10(d)))
    return rf"$10^{{{exp}}}$"


def get_noise_levels(results):
    Ds = set()
    for cell in results.values():
        Ds.update(cell.keys())
    return sorted(Ds)


# ---------------------------------------------------------------------------
# Figure A: Recovery metrics vs noise level
# ---------------------------------------------------------------------------

def make_figure_recovery(results, out_path):
    conn_tags = CONN_ORDER
    D_vals    = get_noise_levels(results)
    ng_list   = sorted({(k[0], k[1]) for k in results})

    metrics = [
        ("f1",       "F1 Score"),
        ("auc",      "AUC-ROC"),
        ("auc_pr",   "AUC-PR"),
        ("deg_corr", "Degree Corr. (ρ)"),
    ]
    n_conn = len(conn_tags)
    n_met  = len(metrics)

    fig, axes = plt.subplots(n_conn, n_met, figsize=(3.8 * n_met, 3.0 * n_conn),
                             sharex=True)
    fig.suptitle("CL vs. FC — Network Recovery", fontsize=13, fontweight="bold", y=1.01)

    markers = ["o", "s", "^", "D"]
    ng_markers = {ng: markers[i % len(markers)] for i, ng in enumerate(ng_list)}

    for row, conn_tag in enumerate(conn_tags):
        for col, (mkey, mlabel) in enumerate(metrics):
            ax = axes[row, col]

            for (N, G), mk in ng_markers.items():
                key = (N, G, conn_tag)
                if key not in results:
                    continue
                cell = results[key]
                Ds = sorted(cell.keys())

                for method in ["CL", "FC"]:
                    mstr = method.lower()
                    vals = [cell[D].get(f"{mkey}_{mstr}", np.nan) for D in Ds]
                    ax.plot(
                        np.log10(Ds), vals,
                        color=METHOD_COLORS[method],
                        ls=METHOD_LS[method],
                        marker=mk, markersize=5,
                        label=f"{method} N={N} G={G}",
                    )

            if row == 0:
                ax.set_title(mlabel, fontsize=10)
            if col == 0:
                ax.set_ylabel(CONN_LABELS.get(conn_tag, conn_tag),
                              fontsize=10, fontweight="bold")
            if row == n_conn - 1:
                ax.set_xlabel("Noise level D", fontsize=9)

            ax.set_xticks([int(np.log10(D)) for D in D_vals])
            ax.set_xticklabels([_lognoise(D) for D in D_vals], fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
            ax.grid(True, alpha=0.3)

    # Shared legend
    from matplotlib.lines import Line2D
    handles = []
    for method in ["CL", "FC"]:
        handles.append(Line2D([0], [0], color=METHOD_COLORS[method],
                               ls=METHOD_LS[method], lw=2, label=method))
    for (N, G), mk in ng_markers.items():
        handles.append(Line2D([0], [0], color="gray", marker=mk, ls="-",
                               markersize=5, lw=1, label=f"N={N}, G={G}"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.04), frameon=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B: Per-condition pairwise consistency (bar chart)
# ---------------------------------------------------------------------------

def make_figure_consistency(results, out_path):
    conn_tags = CONN_ORDER
    D_vals    = get_noise_levels(results)
    ng_list   = sorted({(k[0], k[1]) for k in results})
    D_pairs   = list(combinations(D_vals, 2))
    pair_labels = [f"{_lognoise(d1)} vs {_lognoise(d2)}" for d1, d2 in D_pairs]

    n_conn = len(conn_tags)
    bw = 0.8 / len(ng_list)
    x  = np.arange(len(D_pairs))
    colors = plt.cm.Set2(np.linspace(0, 1, len(ng_list)))

    fig, axes = plt.subplots(n_conn, 2, figsize=(10, 3.0 * n_conn), sharey="row")
    fig.suptitle("Inter-individual Consistency (per condition)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, conn_tag in enumerate(conn_tags):
        ax_cl, ax_fc = axes[row, 0], axes[row, 1]
        for ni, (N, G) in enumerate(ng_list):
            key = (N, G, conn_tag)
            if key not in results:
                continue
            cell = results[key]
            jacc_vals = []
            fc_corr_vals = []
            for d1, d2 in D_pairs:
                if d1 in cell and d2 in cell:
                    jacc_vals.append(jaccard(cell[d1]["cl_adj"], cell[d2]["cl_adj"]))
                    fc_corr_vals.append(matrix_corr(cell[d1]["fc_raw"], cell[d2]["fc_raw"]))
                else:
                    jacc_vals.append(np.nan)
                    fc_corr_vals.append(np.nan)

            offset = (ni - len(ng_list) / 2 + 0.5) * bw
            ax_cl.bar(x + offset, jacc_vals, bw, color=colors[ni],
                      label=f"N={N}, G={G}", alpha=0.85)
            ax_fc.bar(x + offset, fc_corr_vals, bw, color=colors[ni], alpha=0.85)

        for ax, title in [(ax_cl, "CL Jaccard Similarity"),
                          (ax_fc, "FC Matrix Correlation (r)")]:
            ax.set_xticks(x)
            ax.set_xticklabels(pair_labels, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", alpha=0.3)
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")

        axes[row, 0].set_ylabel(CONN_LABELS.get(conn_tag, conn_tag),
                                fontsize=10, fontweight="bold")

    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B2: Consistency summary (aggregated across N, G per conn type)
# ---------------------------------------------------------------------------

def make_figure_consistency_summary(results, out_path):
    conn_tags = CONN_ORDER
    D_vals  = get_noise_levels(results)
    D_pairs = list(combinations(D_vals, 2))
    pair_labels = [f"{_lognoise(d1)}\nvs {_lognoise(d2)}" for d1, d2 in D_pairs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Inter-individual Consistency — Aggregated by Connectivity Type",
                 fontsize=12, fontweight="bold")

    bw = 0.8 / len(conn_tags)
    x  = np.arange(len(D_pairs))

    for ax, (method_key, ylabel) in zip(axes, [("cl", "CL Jaccard"), ("fc", "FC matrix r")]):
        for ci, conn_tag in enumerate(conn_tags):
            means, stds = [], []
            for d1, d2 in D_pairs:
                vals = []
                for (N, G, ct), cell in results.items():
                    if ct != conn_tag or d1 not in cell or d2 not in cell:
                        continue
                    if method_key == "cl":
                        vals.append(jaccard(cell[d1]["cl_adj"], cell[d2]["cl_adj"]))
                    else:
                        vals.append(matrix_corr(cell[d1]["fc_raw"], cell[d2]["fc_raw"]))
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))

            offset = (ci - len(conn_tags) / 2 + 0.5) * bw
            ax.bar(x + offset, means, bw, yerr=stds, capsize=4,
                   color=CONN_COLORS[conn_tag], alpha=0.85,
                   label=CONN_LABELS.get(conn_tag, conn_tag))

        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=9)
        ax.set_xlabel("Individual pair", fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure C: Adjacency heatmaps (N=10, G=0.1, all conn types)
# ---------------------------------------------------------------------------

def make_figure_heatmaps(results, out_path):
    N, G = 10, 0.1
    D_vals = get_noise_levels(results)
    D_low, D_high = min(D_vals), max(D_vals)

    conn_tags = CONN_ORDER
    col_titles = [
        "Ground Truth",
        f"CL  D={_lognoise(D_low)}",
        f"FC  D={_lognoise(D_low)}",
        f"CL  D={_lognoise(D_high)}",
        f"FC  D={_lognoise(D_high)}",
    ]
    ncols = len(col_titles)

    fig, axes = plt.subplots(len(conn_tags), ncols,
                             figsize=(3.0 * ncols, 2.8 * len(conn_tags)))
    fig.suptitle(f"Adjacency Heatmaps  N={N}, G={G}",
                 fontsize=12, fontweight="bold", y=1.01)

    for row, conn_tag in enumerate(conn_tags):
        key = (N, G, conn_tag)
        if key not in results:
            for c in range(ncols):
                axes[row, c].axis("off")
            continue
        cell = results[key]
        gt = cell[D_low]["gt_binary"].astype(float)

        panels = [
            gt,
            cell[D_low]["cl_adj"].astype(float),
            cell[D_low]["fc_matched"].astype(float),
            cell[D_high]["cl_adj"].astype(float),
            cell[D_high]["fc_matched"].astype(float),
        ]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, cmap="Blues", vmin=0, vmax=1,
                      aspect="equal", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(CONN_LABELS.get(conn_tag, conn_tag),
                              fontsize=10, fontweight="bold")
            n_e = int(panel.sum()) // 2
            ax.text(0.97, 0.03, f"{n_e}e", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color="darkred")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure D: Summary bar chart — mean metrics across all conditions
# ---------------------------------------------------------------------------

def make_figure_summary_bars(results, out_path):
    D_vals = get_noise_levels(results)
    metrics = [
        ("f1",       "F1 Score"),
        ("auc",      "AUC-ROC"),
        ("auc_pr",   "AUC-PR"),
        ("deg_corr", "Degree Corr. (ρ)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    fig.suptitle("Summary: CL vs. FC across all conditions",
                 fontsize=12, fontweight="bold")

    x  = np.arange(len(D_vals))
    bw = 0.35

    for ax, (mkey, mlabel) in zip(axes, metrics):
        cl_m, cl_s, fc_m, fc_s = [], [], [], []
        for D in D_vals:
            cl_v, fc_v = [], []
            for cell in results.values():
                if D in cell:
                    cl_v.append(cell[D].get(f"{mkey}_cl", np.nan))
                    fc_v.append(cell[D].get(f"{mkey}_fc", np.nan))
            cl_m.append(np.nanmean(cl_v)); cl_s.append(np.nanstd(cl_v))
            fc_m.append(np.nanmean(fc_v)); fc_s.append(np.nanstd(fc_v))

        ax.bar(x - bw / 2, cl_m, bw, yerr=cl_s, label="CL",
               color=METHOD_COLORS["CL"], alpha=0.85, capsize=4)
        ax.bar(x + bw / 2, fc_m, bw, yerr=fc_s, label="FC",
               color=METHOD_COLORS["FC"], alpha=0.85, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_lognoise(D) for D in D_vals], fontsize=9)
        ax.set_xlabel("Noise level D", fontsize=9)
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(results):
    D_vals = get_noise_levels(results)
    hdr = (f"{'Condition':<30} {'D':>8}  "
           f"{'F1_CL':>6} {'F1_FC':>6}  "
           f"{'AUC_CL':>7} {'AUC_FC':>7}  "
           f"{'PR_CL':>6} {'PR_FC':>6}  "
           f"{'Deg_CL':>7} {'Deg_FC':>7}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(results.keys()):
        N, G, conn_tag = key
        for D in D_vals:
            if D not in results[key]:
                continue
            m = results[key][D]
            print(
                f"N={N} G={G} {conn_tag:<15} {D:>8.0e}  "
                f"{m.get('f1_cl',np.nan):>6.3f} {m.get('f1_fc',np.nan):>6.3f}  "
                f"{m.get('auc_cl',np.nan):>7.3f} {m.get('auc_fc',np.nan):>7.3f}  "
                f"{m.get('auc_pr_cl',np.nan):>6.3f} {m.get('auc_pr_fc',np.nan):>6.3f}  "
                f"{m.get('deg_corr_cl',np.nan):>7.3f} {m.get('deg_corr_fc',np.nan):>7.3f}"
            )
    print("=" * len(hdr) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20260407_133706_grid")
    parser.add_argument("--num-bins", type=int, default=50,
                        help="MI discretization bins (default: 50)")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(
        _REPO_ROOT, "datasets", "synthetic", args.run_id, "analysis"
    )
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(outdir, "results_cache.npz")
    if args.no_cache and os.path.exists(cache):
        os.remove(cache); print("Cache cleared.")

    print(f"Run ID  : {args.run_id}")
    print(f"Output  : {outdir}")

    results = collect_results(args.run_id, num_bins=args.num_bins, cache_path=cache)

    print_summary(results)

    make_figure_recovery(     results, os.path.join(outdir, "fig_a_recovery.png"))
    make_figure_consistency(  results, os.path.join(outdir, "fig_b_consistency.png"))
    make_figure_consistency_summary(results, os.path.join(outdir, "fig_b_consistency_summary.png"))
    make_figure_heatmaps(     results, os.path.join(outdir, "fig_structure_heatmaps.png"))
    make_figure_summary_bars( results, os.path.join(outdir, "fig_summary_bars.png"))

    print(f"\nDone. Figures in {outdir}")


if __name__ == "__main__":
    main()
