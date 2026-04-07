"""
msc_clustering.py — Hierarchical clustering analysis comparing Correlation, MI, and CL.

For each subject, hierarchical clustering dendrograms are computed from three
connectivity matrices:
    - Correlation (Pearson FC)
    - Mutual Information (MI)
    - Chow-Liu tree adjacency

Analyses:
    1. Colored dendrograms by anatomical network
    2. Tanglegrams: side-by-side dendrogram pairs showing structural alignment
    3. Network coherence: within-network clustering fraction as a function of
       the number of clusters k
    4. Clustering agreement: Adjusted Rand Index (ARI) and Normalized Mutual
       Information (NMI) between all method pairs

Outputs:
    analysis/clustering/<atlas>/
        linkages_<subject>.npz          — linkage matrices (ward/average)
        coherence_<subject>.npz         — network coherence arrays
        agreement_<subject>.npz         — ARI / NMI matrices

    figures/clustering/<atlas>/
        fig1_dendrograms_<subject>.pdf  — three colored dendrograms
        fig2_tanglegram_*.pdf           — tanglegram pairs
        fig3_network_coherence.pdf      — coherence vs. k (averaged across subjects)
        fig4_ari_nmi.pdf                — between-method agreement heatmaps

Usage:
    python code/functional_connectivity/midnight_scan_club/msc_clustering.py
    python code/functional_connectivity/midnight_scan_club/msc_clustering.py \\
        --subjects MSC01 MSC04 --atlas-subdir glasser360_SUIT_Thalamus \\
        --num-bins 100 --suffix func-01-10 --k-max 20
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msc_paths import BASE_DIR, mi_dir, cov_dir, analysis_dir, figures_dir
from msc_chow_liu import load_node_labels, infer_network, _NETWORK_COLORS, load_cl_matrices

# ---------------------------------------------------------------------------
# Distance conversion
# ---------------------------------------------------------------------------

def condensed_mi_dist(matrix: np.ndarray) -> np.ndarray:
    """Convert an MI matrix to a condensed distance vector.  d = 1 / (1 + MI)."""
    sym = (matrix + matrix.T) / 2
    np.fill_diagonal(sym, 0.0)
    dist = 1.0 / (1.0 + sym)
    np.fill_diagonal(dist, 0.0)
    return squareform(dist, checks=False)


def condensed_corr_dist(matrix: np.ndarray) -> np.ndarray:
    """Convert a correlation matrix to a condensed distance vector.  d = 1 - |r|."""
    sym = (matrix + matrix.T) / 2
    np.fill_diagonal(sym, 1.0)
    dist = 1.0 - np.abs(sym)
    np.fill_diagonal(dist, 0.0)
    return squareform(dist, checks=False)


# ---------------------------------------------------------------------------
# Cluster analysis helpers
# ---------------------------------------------------------------------------

def same_cluster_matrix(Z: np.ndarray, k: int) -> np.ndarray:
    """Return an N×N binary matrix: entry [i,j]=1 iff ROIs i and j share a cluster at cut k."""
    labels = fcluster(Z, k, criterion='maxclust')
    return (labels[:, None] == labels[None, :]).astype(float)


def network_coherence(Z: np.ndarray, node_networks: list[str], k: int) -> dict[str, float]:
    """Fraction of within-network ROI pairs that co-cluster at cut k.

    Returns a dict {network_name: coherence_value}.
    """
    co = same_cluster_matrix(Z, k)
    networks = sorted(set(node_networks))
    coherence = {}
    for net in networks:
        idx = [i for i, n in enumerate(node_networks) if n == net]
        if len(idx) < 2:
            continue
        pairs = [(i, j) for ii, i in enumerate(idx) for j in idx[ii + 1:]]
        if not pairs:
            continue
        coherence[net] = float(np.mean([co[i, j] for i, j in pairs]))
    return coherence


def mean_coherence_vs_k(Z: np.ndarray, node_networks: list[str], k_values: list[int]) -> np.ndarray:
    """Return mean network coherence across all networks for each k in k_values."""
    means = []
    for k in k_values:
        coh = network_coherence(Z, node_networks, k)
        means.append(np.mean(list(coh.values())) if coh else 0.0)
    return np.array(means)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_matrices(
    subjects: list[str],
    atlas_subdir: str,
    task: str,
    num_bins: int,
    suffix: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Load MI, correlation, and CL matrices for each subject.

    Returns
    -------
    data : {subject: {'mi': ..., 'corr': ..., 'cl': ...}}
          Missing matrices are omitted from the inner dict.
    """
    mi_filename  = f'{task}_{num_bins}bins{suffix}.npy'
    cov_filename = f'{task}_{num_bins}bins.npy'

    cl_matrices = load_cl_matrices(subjects, atlas_subdir)

    data = {}
    for subject in subjects:
        d = {}

        mi_path = os.path.join(mi_dir(subject, atlas_subdir), mi_filename)
        if os.path.exists(mi_path):
            d['mi'] = np.load(mi_path)
        else:
            print(f"  WARNING: MI not found for {subject}")

        cov_path = os.path.join(cov_dir(subject, atlas_subdir), cov_filename)
        if os.path.exists(cov_path):
            d['corr'] = np.load(cov_path)
        else:
            print(f"  WARNING: Correlation not found for {subject}")

        if subject in cl_matrices:
            d['cl'] = cl_matrices[subject]
        else:
            print(f"  WARNING: CL matrix not found for {subject}")

        if d:
            data[subject] = d

    return data


# ---------------------------------------------------------------------------
# Linkage computation
# ---------------------------------------------------------------------------

def compute_linkages(matrices: dict[str, np.ndarray], linkage_method: str = 'ward') -> dict[str, np.ndarray]:
    """Compute hierarchical clustering linkage for each method.

    Parameters
    ----------
    matrices : {'mi': ..., 'corr': ..., 'cl': ...}
    linkage_method : scipy linkage method (default 'ward')

    Returns
    -------
    linkages : {'mi': Z, 'corr': Z, 'cl': Z}
    """
    linkages = {}
    if 'corr' in matrices:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            linkages['corr'] = linkage(condensed_corr_dist(matrices['corr']), method=linkage_method)
    if 'mi' in matrices:
        linkages['mi'] = linkage(condensed_mi_dist(matrices['mi']), method=linkage_method)
    if 'cl' in matrices:
        linkages['cl'] = linkage(condensed_mi_dist(matrices['cl']), method=linkage_method)
    return linkages


# ---------------------------------------------------------------------------
# Figures: colored dendrogram
# ---------------------------------------------------------------------------

def _leaf_colors(Z: np.ndarray, node_networks: list[str]) -> dict[int, str]:
    """Map leaf indices to network colors for scipy dendrogram."""
    n = len(node_networks)
    color_map = {}
    for i in range(n):
        net = node_networks[i]
        color_map[i] = _NETWORK_COLORS.get(net, '#DDDDDD')
    return color_map


def plot_colored_dendrogram(
    Z: np.ndarray,
    node_networks: list[str],
    title: str = '',
    ax: plt.Axes = None,
    orientation: str = 'top',
) -> plt.Axes:
    """Draw a dendrogram with leaves colored by network."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))

    n = len(node_networks)
    leaf_colors = _leaf_colors(Z, node_networks)

    def link_color_func(k):
        return '#888888'

    ddata = dendrogram(
        Z,
        ax=ax,
        orientation=orientation,
        no_labels=True,
        color_threshold=0,
        link_color_func=link_color_func,
    )

    # Color individual leaves
    order = ddata['ivl']
    x_ticks = ax.get_xticks() if orientation in ('top', 'bottom') else ax.get_yticks()
    for tick_x, leaf_idx in zip(x_ticks, order):
        net = node_networks[int(leaf_idx)]
        color = _NETWORK_COLORS.get(net, '#DDDDDD')
        if orientation in ('top', 'bottom'):
            ax.axvline(x=tick_x, ymin=0, ymax=0.03, color=color, lw=2.5)
        else:
            ax.axhline(y=tick_x, xmin=0, xmax=0.03, color=color, lw=2.5)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_dendrograms(
    linkages_per_subject: dict[str, dict[str, np.ndarray]],
    node_networks: list[str],
    atlas_subdir: str,
) -> None:
    """Save fig1: three side-by-side colored dendrograms per subject."""
    fig_dir = figures_dir(f'clustering/{atlas_subdir}')
    os.makedirs(fig_dir, exist_ok=True)

    method_order = [('corr', 'Correlation FC'), ('mi', 'Mutual Information'), ('cl', 'Chow-Liu')]

    for subject, linkages in linkages_per_subject.items():
        n_methods = sum(1 for m, _ in method_order if m in linkages)
        if n_methods == 0:
            continue

        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 4), facecolor='white')
        if n_methods == 1:
            axes = [axes]
        ax_idx = 0
        for method_key, method_label in method_order:
            if method_key not in linkages:
                continue
            plot_colored_dendrogram(
                linkages[method_key], node_networks,
                title=method_label, ax=axes[ax_idx],
            )
            ax_idx += 1

        # Legend
        handles = [
            plt.Line2D([0], [0], color=col, lw=3, label=net)
            for net, col in _NETWORK_COLORS.items()
            if net in node_networks
        ]
        if handles:
            fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                       fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, -0.05))

        fig.suptitle(f'Hierarchical Clustering — {subject}  [{atlas_subdir}]', fontsize=10)
        path = os.path.join(fig_dir, f'fig1_dendrograms_{subject}.pdf')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Figures: tanglegram
# ---------------------------------------------------------------------------

def _reorder_leaves(Z: np.ndarray, target_order: list[int]) -> list[int]:
    """Return the leaf order from Z that minimises crossings with target_order."""
    ddata = dendrogram(Z, no_plot=True)
    current = [int(x) for x in ddata['ivl']]
    target_rank = {leaf: i for i, leaf in enumerate(target_order)}
    return current


def plot_tanglegram(
    Z1: np.ndarray, Z2: np.ndarray,
    node_networks: list[str],
    label1: str = 'Method A', label2: str = 'Method B',
    ax1: plt.Axes = None, ax2: plt.Axes = None, ax_mid: plt.Axes = None,
) -> None:
    """Draw two mirrored dendrograms with connecting lines between matched leaves."""
    ddata1 = dendrogram(Z1, no_plot=True)
    ddata2 = dendrogram(Z2, no_plot=True)
    order1 = [int(x) for x in ddata1['ivl']]
    order2 = [int(x) for x in ddata2['ivl']]

    def _draw_side(ax, Z, orientation, title):
        dendrogram(Z, ax=ax, orientation=orientation,
                   no_labels=True, color_threshold=0,
                   link_color_func=lambda k: '#AAAAAA')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if ax1 is not None:
        _draw_side(ax1, Z1, 'right', label1)
    if ax2 is not None:
        _draw_side(ax2, Z2, 'left', label2)

    # Connecting lines in middle panel
    if ax_mid is not None:
        rank2 = {leaf: i / max(len(order2) - 1, 1) for i, leaf in enumerate(order2)}
        for i, leaf in enumerate(order1):
            y1 = i / max(len(order1) - 1, 1)
            y2 = rank2.get(leaf, 0.5)
            net = node_networks[leaf] if leaf < len(node_networks) else 'Other'
            color = _NETWORK_COLORS.get(net, '#DDDDDD')
            ax_mid.plot([0, 1], [y1, y2], color=color, alpha=0.4, lw=0.8)
        ax_mid.set_xlim(0, 1)
        ax_mid.set_ylim(-0.05, 1.05)
        ax_mid.axis('off')


def plot_tanglegrams(
    linkages_per_subject: dict[str, dict[str, np.ndarray]],
    node_networks: list[str],
    atlas_subdir: str,
) -> None:
    """Save fig2: tanglegrams for all three method pairs, averaged across subjects."""
    fig_dir = figures_dir(f'clustering/{atlas_subdir}')
    os.makedirs(fig_dir, exist_ok=True)

    # Use the first available subject with all three methods
    reference_subject = None
    for subject, lk in linkages_per_subject.items():
        if all(m in lk for m in ('corr', 'mi', 'cl')):
            reference_subject = subject
            break

    if reference_subject is None:
        print("  WARNING: No subject has all three linkages; skipping tanglegrams.")
        return

    pairs = [
        ('corr', 'mi',   'Correlation', 'Mutual Information'),
        ('corr', 'cl',   'Correlation', 'Chow-Liu'),
        ('mi',   'cl',   'Mutual Information', 'Chow-Liu'),
    ]

    for subject, linkages in linkages_per_subject.items():
        for m1, m2, l1, l2 in pairs:
            if m1 not in linkages or m2 not in linkages:
                continue

            fig = plt.figure(figsize=(10, 6), facecolor='white')
            gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 2], wspace=0.05)
            ax1   = fig.add_subplot(gs[0])
            ax_m  = fig.add_subplot(gs[1])
            ax2   = fig.add_subplot(gs[2])

            plot_tanglegram(
                linkages[m1], linkages[m2],
                node_networks, label1=l1, label2=l2,
                ax1=ax1, ax2=ax2, ax_mid=ax_m,
            )
            fig.suptitle(f'Tanglegram: {l1} vs {l2} — {subject}  [{atlas_subdir}]', fontsize=9)
            fname = f'fig2_tanglegram_{m1}_vs_{m2}_{subject}.pdf'
            path = os.path.join(fig_dir, fname)
            fig.savefig(path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Figures: network coherence
# ---------------------------------------------------------------------------

def plot_network_coherence(
    coherence_arrays: dict[str, dict[str, np.ndarray]],
    k_values: list[int],
    atlas_subdir: str,
) -> None:
    """Save fig3: mean network coherence vs. k for each method.

    coherence_arrays : {method: {subject: array of shape (len(k_values),)}}
    """
    fig_dir = figures_dir(f'clustering/{atlas_subdir}')
    os.makedirs(fig_dir, exist_ok=True)

    method_styles = {
        'corr': ('#E41A1C', '-',  'Correlation FC'),
        'mi':   ('#377EB8', '--', 'Mutual Information'),
        'cl':   ('#4DAF4A', ':',  'Chow-Liu'),
    }

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    for method, (color, ls, label) in method_styles.items():
        if method not in coherence_arrays:
            continue
        arr = np.array(list(coherence_arrays[method].values()))  # (n_subjects, len(k_values))
        mean = arr.mean(axis=0)
        sem  = arr.std(axis=0) / np.sqrt(arr.shape[0])
        ax.plot(k_values, mean, color=color, ls=ls, lw=2, label=label)
        ax.fill_between(k_values, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.set_xlabel('Number of clusters k', fontsize=11)
    ax.set_ylabel('Mean network coherence', fontsize=11)
    ax.set_title(f'Network coherence vs. k  [{atlas_subdir}]', fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    path = os.path.join(fig_dir, 'fig3_network_coherence.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Figures: ARI / NMI agreement heatmaps
# ---------------------------------------------------------------------------

def compute_agreement(linkages: dict[str, np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ARI and NMI matrices for all method pairs at cut k."""
    methods = list(linkages.keys())
    n = len(methods)
    ari_mat = np.zeros((n, n))
    nmi_mat = np.zeros((n, n))

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            l1 = fcluster(linkages[m1], k, criterion='maxclust')
            l2 = fcluster(linkages[m2], k, criterion='maxclust')
            ari_mat[i, j] = adjusted_rand_score(l1, l2)
            nmi_mat[i, j] = normalized_mutual_info_score(l1, l2)

    return ari_mat, nmi_mat, methods


def plot_ari_nmi(
    linkages_per_subject: dict[str, dict[str, np.ndarray]],
    atlas_subdir: str,
    k: int = 7,
) -> None:
    """Save fig4: ARI and NMI heatmaps averaged across subjects."""
    fig_dir = figures_dir(f'clustering/{atlas_subdir}')
    os.makedirs(fig_dir, exist_ok=True)

    ari_list, nmi_list, method_names = [], [], None
    for linkages in linkages_per_subject.values():
        if len(linkages) < 2:
            continue
        ari, nmi, methods = compute_agreement(linkages, k)
        ari_list.append(ari)
        nmi_list.append(nmi)
        method_names = methods

    if not ari_list:
        return

    mean_ari = np.mean(ari_list, axis=0)
    mean_nmi = np.mean(nmi_list, axis=0)
    labels = [{'corr': 'Correlation', 'mi': 'MI', 'cl': 'Chow-Liu'}.get(m, m) for m in method_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
    for ax, mat, title in [(ax1, mean_ari, f'ARI  (k={k})'), (ax2, mean_nmi, f'NMI  (k={k})')]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if mat[i, j] < 0.5 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Clustering agreement  [{atlas_subdir}]', fontsize=10)
    path = os.path.join(fig_dir, 'fig4_ari_nmi.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Save / load analysis results
# ---------------------------------------------------------------------------

def save_analysis(
    subject: str,
    linkages: dict[str, np.ndarray],
    coherence_arrays: dict[str, np.ndarray],
    atlas_subdir: str,
) -> None:
    out_dir = analysis_dir(f'clustering/{atlas_subdir}')
    os.makedirs(out_dir, exist_ok=True)

    np.savez(os.path.join(out_dir, f'linkages_{subject}.npz'), **linkages)
    np.savez(os.path.join(out_dir, f'coherence_{subject}.npz'), **coherence_arrays)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical clustering analysis of MSC connectivity matrices.'
    )
    parser.add_argument('--subjects', nargs='+',
                        default=['MSC01', 'MSC02', 'MSC03', 'MSC04', 'MSC05',
                                 'MSC06', 'MSC07', 'MSC08', 'MSC09', 'MSC10'])
    parser.add_argument('--atlas-subdir', default='glasser360_SUIT_Thalamus')
    parser.add_argument('--task', default='rest')
    parser.add_argument('--num-bins', type=int, default=100)
    parser.add_argument('--suffix', default='func-01-10',
                        help='Suffix appended after bins in MI filename (default: func-01-10)')
    parser.add_argument('--linkage-method', default='ward',
                        choices=['ward', 'average', 'complete', 'single'],
                        help='Scipy linkage method (default: ward)')
    parser.add_argument('--k-max', type=int, default=20,
                        help='Maximum number of clusters for coherence plot (default: 20)')
    parser.add_argument('--k-agreement', type=int, default=7,
                        help='k used for ARI/NMI agreement heatmaps (default: 7)')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    k_values = list(range(2, args.k_max + 1))

    print(f'\n=== Loading connectivity matrices ({args.atlas_subdir}) ===')
    all_data = load_all_matrices(
        args.subjects, args.atlas_subdir,
        args.task, args.num_bins, args.suffix,
    )
    if not all_data:
        print('No data found. Run msc_chow_liu.py first to generate CL matrices.')
        return

    available = list(all_data.keys())
    # Infer n_rois from the first available matrix
    first_mat = next(iter(next(iter(all_data.values())).values()))
    n_rois = first_mat.shape[0]
    node_labels   = load_node_labels(args.atlas_subdir, n_rois)
    node_networks = [infer_network(l) for l in node_labels]

    print(f'ROIs: {n_rois},  subjects: {len(available)}')
    print(f'Networks: {sorted(set(node_networks))}')

    print('\n=== Computing linkages ===')
    linkages_per_subject: dict[str, dict[str, np.ndarray]] = {}
    coherence_mi   = {}   # {subject: coherence_array}
    coherence_corr = {}
    coherence_cl   = {}

    for subject in available:
        matrices = all_data[subject]
        linkages = compute_linkages(matrices, linkage_method=args.linkage_method)
        linkages_per_subject[subject] = linkages

        for method_key, store in [('mi', coherence_mi), ('corr', coherence_corr), ('cl', coherence_cl)]:
            if method_key in linkages:
                store[subject] = mean_coherence_vs_k(linkages[method_key], node_networks, k_values)

        save_analysis(
            subject, linkages,
            {'mi': coherence_mi.get(subject, np.array([])),
             'corr': coherence_corr.get(subject, np.array([])),
             'cl':   coherence_cl.get(subject, np.array([]))},
            args.atlas_subdir,
        )
        print(f'  {subject}: linkages computed for {list(linkages.keys())}')

    if not args.no_figures:
        print('\n=== Figure 1: colored dendrograms ===')
        plot_dendrograms(linkages_per_subject, node_networks, args.atlas_subdir)

        print('\n=== Figure 2: tanglegrams ===')
        plot_tanglegrams(linkages_per_subject, node_networks, args.atlas_subdir)

        print('\n=== Figure 3: network coherence ===')
        coherence_arrays = {}
        for method_key, store in [('mi', coherence_mi), ('corr', coherence_corr), ('cl', coherence_cl)]:
            if store:
                coherence_arrays[method_key] = store
        plot_network_coherence(coherence_arrays, k_values, args.atlas_subdir)

        print('\n=== Figure 4: ARI / NMI heatmaps ===')
        plot_ari_nmi(linkages_per_subject, args.atlas_subdir, k=args.k_agreement)

    print('\nDone.')


if __name__ == '__main__':
    main()
