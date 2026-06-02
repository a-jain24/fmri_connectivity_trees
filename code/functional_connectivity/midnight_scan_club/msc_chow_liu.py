"""
msc_chow_liu.py — Build and visualize Chow-Liu trees from precomputed MI matrices.

A Chow-Liu tree is the maximum spanning tree of the complete weighted graph
whose edge weights are pairwise mutual information values.  It is the best
tree-structured approximation (in KL-divergence) to the full joint distribution.

Pipeline:
    1. Load MI matrices from output/mutual_information/<subject>/<atlas>/
    2. Build one Chow-Liu tree (NetworkX MST) per subject
    3. Save adjacency matrices to analysis/chow_liu/<atlas>/
    4. Save hierarchical tree visualizations to figures/chow_liu/<atlas>/

Usage:
    python code/functional_connectivity/midnight_scan_club/msc_chow_liu.py
    python code/functional_connectivity/midnight_scan_club/msc_chow_liu.py \\
        --subjects MSC01 MSC04 MSC07 \\
        --atlas-subdir glasser360_SUIT_Thalamus \\
        --num-bins 100 --suffix func-01-10
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msc_paths import (
    BASE_DIR, ATLAS_DIR,
    mi_dir, jp_dir,
    analysis_dir, figures_dir,
    glasser360_labels_path,
)

# ---------------------------------------------------------------------------
# Atlas label loading
# ---------------------------------------------------------------------------

def load_glasser360_labels() -> list[str]:
    """Return the 360 Glasser parcel names (one per line in the .txt file)."""
    path = glasser360_labels_path()
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_node_labels(atlas_subdir: str, n_rois: int) -> list[str]:
    """Return ROI labels for the given atlas combination.

    For atlas combinations not fully resolved here, fall back to
    index strings so the rest of the pipeline still works.
    """
    labels = []
    if 'glasser360' in atlas_subdir:
        labels += load_glasser360_labels()          # 360 cortical parcels
    if 'SUIT' in atlas_subdir:
        # SUIT has 34 cerebellar parcels (indices vary by version)
        n_suit = 34
        labels += [f'Cereb_{i+1}' for i in range(n_suit)]
    if 'Thalamus' in atlas_subdir:
        n_thal = n_rois - len(labels)
        labels += [f'Thal_{i+1}' for i in range(max(n_thal, 0))]

    # Pad or truncate to exactly n_rois
    if len(labels) < n_rois:
        labels += [f'ROI_{i}' for i in range(len(labels), n_rois)]
    return labels[:n_rois]


def infer_network(label: str) -> str:
    """Infer a broad network category from a Glasser ROI name."""
    l = label.lower()
    if l.startswith('cereb'):
        return 'Cerebellum'
    if l.startswith('thal'):
        return 'Thalamus'
    # Motor / somatomotor
    if any(x in l for x in ['_4', '_3b', '_3a', '_1', '_2', 'fef', 'pef', 'sma', 'premotor', '6']):
        return 'Somatomotor'
    # Visual
    if any(x in l for x in ['v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'mst', 'mt', 'lo1', 'lo2', 'ffc', 'pit', 'vvc', 'lingual']):
        return 'Visual'
    # Default mode
    if any(x in l for x in ['pcc', 'rsc', 'pgp', 'pgm', 'mpfc', 'a24', 'v23', 'd23', '31pd', 'ofc', 'tpoj']):
        return 'DefaultMode'
    # Attention
    if any(x in l for x in ['ips', 'spl', 'aip', 'mip', '7', 'fef']):
        return 'DorsAttn'
    # Language
    if any(x in l for x in ['stp', 'stg', 'a5', 'a4', 'ta2', 'ph', 'ip1', 'ip2', 'stgs', 'tpoj']):
        return 'Language'
    # Frontoparietal
    if any(x in l for x in ['46', '9-46', 'dlpfc', 'ifg', 'if', 'ips', 'lpc', 'a9', 'a10']):
        return 'Frontoparietal'
    return 'Other'


# Network → color mapping
_NETWORK_COLORS = {
    'Visual':         '#E41A1C',
    'Somatomotor':    '#377EB8',
    'DorsAttn':       '#4DAF4A',
    'Frontoparietal': '#984EA3',
    'DefaultMode':    '#FF7F00',
    'Language':       '#A65628',
    'Cerebellum':     '#F781BF',
    'Thalamus':       '#999999',
    'Other':          '#DDDDDD',
}


# ---------------------------------------------------------------------------
# Graph construction and Chow-Liu tree
# ---------------------------------------------------------------------------

def mi_to_graph(mi_matrix: np.ndarray) -> nx.Graph:
    """Convert an (N, N) MI matrix to a fully connected weighted NetworkX graph."""
    n = mi_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(mi_matrix[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)
    return G


def chow_liu_tree(mi_matrix: np.ndarray) -> tuple[nx.Graph, nx.Graph]:
    """Build the Chow-Liu tree (maximum spanning tree) from a MI matrix.

    Returns
    -------
    G : full weighted graph
    tree : maximum spanning tree (NetworkX Graph)
    """
    G = mi_to_graph(mi_matrix)
    tree = nx.maximum_spanning_tree(G, algorithm='kruskal', weight='weight')
    return G, tree


def build_cl_trees(
    mi_data: dict[str, np.ndarray],
    subjects: list[str],
) -> tuple[dict, dict]:
    """Build Chow-Liu trees for each subject.

    Returns
    -------
    cl_trees    : {subject: nx.Graph}  — tree structure
    cl_matrices : {subject: np.ndarray} — sparse adjacency matrix
    """
    cl_trees, cl_matrices = {}, {}
    for subject in subjects:
        mi = mi_data[subject]
        _, tree = chow_liu_tree(mi)
        cl_trees[subject] = tree
        cl_matrices[subject] = nx.to_numpy_array(tree, weight='weight')
        n_edges = tree.number_of_edges()
        n_nodes = tree.number_of_nodes()
        print(f"  {subject}: {n_nodes} nodes, {n_edges} edges (expected {n_nodes - 1})")
    return cl_trees, cl_matrices


# ---------------------------------------------------------------------------
# Hierarchical tree layout and visualization
# ---------------------------------------------------------------------------

def _hierarchy_pos(G: nx.DiGraph, root: int, width: float = 2.0, vert_gap: float = 0.5) -> dict:
    """Compute a top-down hierarchical layout for a BFS-rooted directed tree."""
    def _count(node):
        children = list(G.successors(node))
        return 1 if not children else sum(_count(c) for c in children)

    def _recurse(node, left, right, y, pos):
        pos[node] = ((left + right) / 2, y)
        children = list(G.successors(node))
        if children:
            total = sum(_count(c) for c in children)
            dx = (right - left) / total
            x = left
            for child in children:
                sz = _count(child)
                _recurse(child, x, x + dx * sz, y - vert_gap, pos)
                x += dx * sz
        return pos

    return _recurse(root, 0, width, 0, {})


def draw_hierarchical_tree(
    tree: nx.Graph,
    node_labels: list[str],
    root: int = 0,
    title: str = '',
    ax: plt.Axes = None,
    color_fn=None,
    label_fn=None,
    label_fontsize: int = 6,
) -> plt.Axes:
    """Draw a Chow-Liu tree with a top-down hierarchical layout.

    Parameters
    ----------
    color_fn : callable or None
        If provided, called as ``color_fn(label) -> hex_color_str`` for each
        node label.  Overrides the default ``infer_network`` coloring.
    label_fn : callable or None
        If provided, called as ``label_fn(label) -> str`` to produce the
        display text for each node.  Labels are drawn rotated 45° above each
        node.  Pass ``None`` to suppress labels entirely.
    label_fontsize : int
        Font size for node labels when ``label_fn`` is set.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))

    # Pick root as highest-degree node for a balanced layout
    if tree.number_of_nodes() > 0:
        root = max(tree.degree, key=lambda x: x[1])[0]

    directed = nx.bfs_tree(tree, source=root)
    pos = _hierarchy_pos(directed, root)

    if color_fn is not None:
        colors = [color_fn(node_labels[n]) for n in tree.nodes()]
        legend_items = {}
    else:
        networks = [infer_network(node_labels[n]) for n in tree.nodes()]
        colors = [_NETWORK_COLORS.get(net, '#DDDDDD') for net in networks]
        legend_items = {net: col for net, col in _NETWORK_COLORS.items()
                        if net in networks}

    edge_weights = [tree[u][v]['weight'] for u, v in tree.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    widths = [1.0 + 3.0 * (w / max_w) for w in edge_weights]

    nx.draw_networkx_edges(tree, pos, width=widths, alpha=0.6, edge_color='#555555', ax=ax)
    nx.draw_networkx_nodes(tree, pos, node_color=colors, node_size=120, ax=ax)

    if label_fn is not None:
        # Estimate a small vertical offset in data units so labels sit just above nodes
        y_vals = [y for _, y in pos.values()]
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1.0
        offset = y_range * 0.04
        for node in tree.nodes():
            x, y = pos[node]
            text = label_fn(node_labels[node])
            ax.text(
                x, y + offset, text,
                fontsize=label_fontsize,
                ha='left', va='bottom',
                rotation=45, rotation_mode='anchor',
                clip_on=False,
            )

    ax.set_title(title, fontsize=10, pad=4)
    ax.axis('off')

    if legend_items:
        for net, col in legend_items.items():
            ax.scatter([], [], c=col, s=60, label=net)
        ax.legend(loc='lower right', fontsize=6, ncol=2, framealpha=0.7)

    return ax


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_mi_data(
    subjects: list[str],
    atlas_subdir: str,
    task: str = 'rest',
    num_bins: int = 100,
    suffix: str = '',
) -> dict[str, np.ndarray]:
    """Load MI matrices for all subjects from output/mutual_information/."""
    mi_data = {}
    filename = f'{task}_{num_bins}bins{suffix}.npy'
    for subject in subjects:
        path = os.path.join(mi_dir(subject, atlas_subdir), filename)
        if not os.path.exists(path):
            print(f"  WARNING: MI file not found for {subject}: {path}")
            continue
        mi_data[subject] = np.load(path)
        print(f"  Loaded MI {subject}: {mi_data[subject].shape}")
    return mi_data


def save_cl_matrices(
    cl_matrices: dict[str, np.ndarray],
    atlas_subdir: str,
) -> None:
    """Save CL adjacency matrices to analysis/chow_liu/<atlas>/."""
    out_dir = analysis_dir(f'chow_liu/{atlas_subdir}')
    os.makedirs(out_dir, exist_ok=True)
    for subject, mat in cl_matrices.items():
        path = os.path.join(out_dir, f'{subject}_cl_adjacency.npy')
        np.save(path, mat)
        print(f"  Saved → {path}")


def load_cl_matrices(
    subjects: list[str],
    atlas_subdir: str,
) -> dict[str, np.ndarray]:
    """Load previously saved CL adjacency matrices."""
    cl_matrices = {}
    in_dir = analysis_dir(f'chow_liu/{atlas_subdir}')
    for subject in subjects:
        path = os.path.join(in_dir, f'{subject}_cl_adjacency.npy')
        if not os.path.exists(path):
            print(f"  WARNING: CL matrix not found for {subject}: {path}")
            continue
        cl_matrices[subject] = np.load(path)
    return cl_matrices


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_cl_trees(
    cl_trees: dict[str, nx.Graph],
    node_labels: list[str],
    atlas_subdir: str,
    subjects: list[str] | None = None,
) -> None:
    """Save a per-subject Chow-Liu tree figure to figures/chow_liu/<atlas>/."""
    if subjects is None:
        subjects = list(cl_trees.keys())

    fig_dir = figures_dir(f'chow_liu/{atlas_subdir}')
    os.makedirs(fig_dir, exist_ok=True)

    for subject in subjects:
        if subject not in cl_trees:
            continue
        fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
        draw_hierarchical_tree(
            cl_trees[subject],
            node_labels,
            title=f'Chow-Liu Tree — {subject}  ({atlas_subdir})',
            ax=ax,
        )
        path = os.path.join(fig_dir, f'{subject}_cl_tree.pdf')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")


def plot_consensus_tree(
    cl_trees: dict[str, nx.Graph],
    node_labels: list[str],
    atlas_subdir: str,
    threshold: float = 0.5,
) -> None:
    """Build and plot a consensus tree: edges present in ≥ threshold fraction of subjects.

    The consensus adjacency is used as edge weights, then MST is taken.
    Saved to figures/chow_liu/<atlas>/consensus_tree.pdf.
    """
    subjects = list(cl_trees.keys())
    n = cl_trees[subjects[0]].number_of_nodes()
    vote_matrix = np.zeros((n, n))

    for tree in cl_trees.values():
        adj = nx.to_numpy_array(tree, nodelist=range(n), weight=None)
        vote_matrix += adj

    vote_matrix /= len(subjects)

    # Build consensus tree from vote matrix (edges present in >= threshold subjects)
    consensus_adj = np.where(vote_matrix >= threshold, vote_matrix, 0.0)
    G_cons = nx.from_numpy_array(consensus_adj)
    if G_cons.number_of_edges() == 0:
        print(f"  WARNING: No edges survive threshold={threshold:.0%}; skipping consensus tree.")
        return

    consensus_tree = nx.maximum_spanning_tree(G_cons, weight='weight')

    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    draw_hierarchical_tree(
        consensus_tree, node_labels,
        title=f'Consensus Chow-Liu Tree (threshold={threshold:.0%}, N={len(subjects)} subjects)  [{atlas_subdir}]',
        ax=ax,
    )
    fig_dir = figures_dir(f'chow_liu/{atlas_subdir}')
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, 'consensus_tree.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # Also save vote matrix
    vote_path = os.path.join(analysis_dir(f'chow_liu/{atlas_subdir}'), 'edge_vote_matrix.npy')
    os.makedirs(os.path.dirname(vote_path), exist_ok=True)
    np.save(vote_path, vote_matrix)
    print(f"  Saved → {vote_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build Chow-Liu trees from MSC MI matrices.'
    )
    parser.add_argument(
        '--subjects', nargs='+',
        default=['MSC01', 'MSC02', 'MSC03', 'MSC04', 'MSC05',
                 'MSC06', 'MSC07', 'MSC08', 'MSC09', 'MSC10'],
        help='Subject IDs (default: all 10)',
    )
    parser.add_argument(
        '--atlas-subdir', default='glasser360_SUIT_Thalamus',
        help='Atlas subdirectory under output/mutual_information/ (default: glasser360_SUIT_Thalamus)',
    )
    parser.add_argument(
        '--task', default='rest',
        help='Task name (default: rest)',
    )
    parser.add_argument(
        '--num-bins', type=int, default=100,
        help='Number of MI discretization bins (default: 100)',
    )
    parser.add_argument(
        '--suffix', default='func-01-10',
        help='Filename suffix after bins (default: func-01-10)',
    )
    parser.add_argument(
        '--consensus-threshold', type=float, default=0.5,
        help='Fraction of subjects an edge must appear in for the consensus tree (default: 0.5)',
    )
    parser.add_argument(
        '--no-figures', action='store_true',
        help='Skip figure generation (only save adjacency matrices)',
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    print(f'\n=== Loading MI matrices ({args.atlas_subdir}) ===')
    mi_data = load_mi_data(
        args.subjects, args.atlas_subdir,
        task=args.task, num_bins=args.num_bins, suffix=args.suffix,
    )
    if not mi_data:
        print('No MI data found. Check --atlas-subdir and --suffix.')
        return

    available = list(mi_data.keys())
    n_rois = next(iter(mi_data.values())).shape[0]
    node_labels = load_node_labels(args.atlas_subdir, n_rois)
    print(f'ROIs: {n_rois},  subjects loaded: {len(available)}')

    print('\n=== Building Chow-Liu trees ===')
    cl_trees, cl_matrices = build_cl_trees(mi_data, available)

    print('\n=== Saving adjacency matrices ===')
    save_cl_matrices(cl_matrices, args.atlas_subdir)

    if not args.no_figures:
        print('\n=== Generating per-subject tree figures ===')
        plot_cl_trees(cl_trees, node_labels, args.atlas_subdir)

        print('\n=== Generating consensus tree ===')
        plot_consensus_tree(
            cl_trees, node_labels, args.atlas_subdir,
            threshold=args.consensus_threshold,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
