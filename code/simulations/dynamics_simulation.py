"""
dynamics_simulation.py — Generic2dOscillator network with edge-specific coupling types.

Each edge applies a different transform to the source node's state before
injecting into the target's ODE, enabling nonlinear coupling that MI can
detect but correlation may miss.

Coupling types:
    1 (linear)     : V_j                     — standard linear coupling
    2 (quadrature) : W_j                     — ~90° phase-shifted recovery variable
    3 (rectified)  : max(V_j, 0)             — burst coupling (threshold gating)
    4 (squared)    : V_j²                    — nonlinear energy injection (double frequency)
    5 (pac)        : (0.5 + 0.4*V_j) * W_j  — phase-amplitude coupling (V modulates W envelope)

Usage:
    python code/simulations/dynamics_simulation.py \
        --N 15 --simlen 600000 --edge-mode random \
        --outdir code/simulations/output/dynamics_test \
        --no-delays \
        --connectivity sparse

    python code/simulations/dynamics_simulation.py \
        --N 15 --simlen 600000 --edge-mode uniform --coupling-type quadrature \
        --outdir code/simulations/output/dynamics_test_quad
"""

import argparse
import os

import networkx as nx
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Coupling type constants
# ---------------------------------------------------------------------------
LINEAR = 1
QUADRATURE = 2
RECTIFIED = 3
SQUARED = 4
PAC = 5

TYPE_NAMES = {
    LINEAR: "linear",
    QUADRATURE: "quadrature",
    RECTIFIED: "rectified",
    SQUARED: "squared",
    PAC: "pac",
}

NAME_TO_TYPE = {v: k for k, v in TYPE_NAMES.items()}


# ---------------------------------------------------------------------------
# Connectivity generation
# ---------------------------------------------------------------------------

def build_connectivity(N, mode, rng, extra_edges=0, branching=3, density=0.5):
    """Generate a connectivity weight matrix.

    Parameters
    ----------
    N : int
        Number of nodes.
    mode : str
        'erdos_renyi'  — G(N, p) with p = density.  Each upper-triangle pair is
                         independently included with probability `density`, giving a
                         continuous sparsity scale.  Connectivity is guaranteed by
                         bridging isolated components.  Symmetric, weights in [0.5, 3.0].
        'dense'        — Legacy: rng.integers(0, 4), asymmetric, ~75% fill.
        'sparse'       — Legacy: Erdos-Renyi with fixed p = min(4/(N-1), 0.8) ~ avg degree 4.
        'tree'         — Uniformly random labeled tree via Prüfer sequence (NetworkX).
        'hierarchical' — BFS-style random branching tree; each internal node gets
                         1..branching children, producing explicit multi-level hierarchy.
    rng : numpy.random.Generator
    extra_edges : int
        Additional non-tree edges appended after tree/hierarchical construction.
    branching : int
        Maximum children per node in 'hierarchical' mode (default 3).
    density : float
        Edge probability for 'erdos_renyi' mode (0 < density ≤ 1, default 0.5).
        Expected fraction of all possible edges that will be present.

    Returns
    -------
    C : ndarray (N, N), symmetric for erdos_renyi/sparse/tree/hierarchical,
        asymmetric for dense.
    """
    if mode == "dense":
        C = rng.integers(0, 4, size=(N, N)).astype(float)
        np.fill_diagonal(C, 0.0)
        return C

    if mode == "tree":
        # Prüfer-sequence random tree: uniform distribution over all N^(N-2) labeled
        # trees, producing natural hub-and-spoke branching rather than a path graph.
        C = np.zeros((N, N))
        seed_int = int(rng.integers(0, 2**31))
        T = nx.random_labeled_tree(N, seed=seed_int)
        for i, j in T.edges():
            w = rng.uniform(0.5, 3.0)
            C[i, j] = C[j, i] = w
        # Add extra non-tree edges
        added = 0
        attempts = 0
        max_attempts = max(extra_edges * 100, 1)
        while added < extra_edges and attempts < max_attempts:
            i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
            if i != j and C[i, j] == 0:
                w = rng.uniform(0.5, 3.0)
                C[i, j] = C[j, i] = w
                added += 1
            attempts += 1
        if added < extra_edges:
            print(f"Warning: only added {added}/{extra_edges} extra edges (graph may be near-complete)")
        return C

    if mode == "hierarchical":
        # BFS construction: pick a random root, then at each level assign 1..branching
        # children per parent.  The BFS queue never empties before all nodes are placed
        # (each parent adds ≥1 child while unassigned nodes remain), guaranteeing a
        # connected tree with explicit depth structure.
        from collections import deque
        C = np.zeros((N, N))
        if N <= 1:
            return C
        nodes = list(rng.permutation(N))
        queue = deque([nodes[0]])
        pos = 1
        while pos < N and queue:
            parent = queue.popleft()
            n_children = min(int(rng.integers(1, branching + 1)), N - pos)
            for _ in range(n_children):
                child = nodes[pos]
                pos += 1
                w = rng.uniform(0.5, 3.0)
                C[parent, child] = C[child, parent] = w
                queue.append(child)
        # Add extra non-tree edges
        added = 0
        attempts = 0
        max_attempts = max(extra_edges * 100, 1)
        while added < extra_edges and attempts < max_attempts:
            i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
            if i != j and C[i, j] == 0:
                w = rng.uniform(0.5, 3.0)
                C[i, j] = C[j, i] = w
                added += 1
            attempts += 1
        if added < extra_edges:
            print(f"Warning: only added {added}/{extra_edges} extra edges (graph may be near-complete)")
        return C

    if mode == "sparse":
        C = np.zeros((N, N))
        p = min(4.0 / (N - 1), 0.8)
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < p:
                    w = rng.uniform(0.5, 3.0)
                    C[i, j] = C[j, i] = w
        # Ensure connected via NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if C[i, j] > 0:
                    G.add_edge(i, j)
        components = list(nx.connected_components(G))
        while len(components) > 1:
            # Connect first node of each component to first node of next
            for ci in range(len(components) - 1):
                a = min(components[ci])
                b = min(components[ci + 1])
                w = rng.uniform(0.5, 3.0)
                C[a, b] = C[b, a] = w
            # Recompute components
            G = nx.Graph()
            G.add_nodes_from(range(N))
            for i in range(N):
                for j in range(i + 1, N):
                    if C[i, j] > 0:
                        G.add_edge(i, j)
            components = list(nx.connected_components(G))
        return C

    if mode == "erdos_renyi":
        if not (0 < density <= 1):
            raise ValueError(f"density must be in (0, 1], got {density}")
        C = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < density:
                    w = rng.uniform(0.5, 3.0)
                    C[i, j] = C[j, i] = w
        # Ensure connected by bridging isolated components
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if C[i, j] > 0:
                    G.add_edge(i, j)
        components = list(nx.connected_components(G))
        while len(components) > 1:
            for ci in range(len(components) - 1):
                a = min(components[ci])
                b = min(components[ci + 1])
                w = rng.uniform(0.5, 3.0)
                C[a, b] = C[b, a] = w
            G = nx.Graph()
            G.add_nodes_from(range(N))
            for i in range(N):
                for j in range(i + 1, N):
                    if C[i, j] > 0:
                        G.add_edge(i, j)
            components = list(nx.connected_components(G))
        return C

    raise ValueError(f"Unknown connectivity mode: {mode}")


def build_symmetric_coupling_types(weights, mode, rng, uniform_type=None):
    """Assign coupling types symmetrically for symmetric weight matrices.

    For each upper-triangle edge, assigns one type and mirrors it.

    Parameters
    ----------
    weights : ndarray (N, N)
    mode : str — 'random' or 'uniform'
    rng : numpy.random.Generator
    uniform_type : int or None

    Returns
    -------
    coupling_types : ndarray (N, N), dtype int
    """
    N = weights.shape[0]
    coupling_types = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            if weights[i, j] != 0:
                if mode == "random":
                    ct = int(rng.integers(1, len(TYPE_NAMES) + 1))
                elif mode == "uniform":
                    if uniform_type is None:
                        raise ValueError("uniform_type must be specified for mode='uniform'")
                    ct = uniform_type
                else:
                    raise ValueError(f"Unknown edge mode: {mode}")
                coupling_types[i, j] = ct
                coupling_types[j, i] = ct

    return coupling_types


# ---------------------------------------------------------------------------
# Default parameters  (reused from generic_2doscillator.py)
# ---------------------------------------------------------------------------

def tvb_default_params(N):
    """Return homogeneous TVB Generic2dOscillator defaults for N nodes."""
    return {
        "a":     np.full(N, 0.0),
        "b":     np.full(N, -10.0),
        "c":     np.full(N, 0.0),
        "d":     np.full(N, 0.02),
        "e":     np.full(N, 3.0),
        "f":     np.full(N, 1.0),
        "g":     np.full(N, 0.0),
        "alpha": np.full(N, 1.0),
        "beta":  np.full(N, 1.0),
        "gamma": np.full(N, 1.0),
        "tau":   np.full(N, 1.0),
    }


# ---------------------------------------------------------------------------
# Heterogeneous parameter generation  (reused from generic_2doscillator.py)
# ---------------------------------------------------------------------------

def gaussian_param(default, sigma, N, clip=None):
    """Sample N values from a Gaussian around a default."""
    values = np.random.normal(default, sigma, N)
    if clip is not None:
        values = np.clip(values, clip[0], clip[1])
    return values


def build_heterogeneous_params(N, defaults, sigmas):
    """Build per-node parameter arrays by sampling around TVB defaults."""
    return {
        "a":     gaussian_param(defaults["a"],     sigmas["a"],     N),
        "b":     gaussian_param(defaults["b"],     sigmas["b"],     N),
        "c":     gaussian_param(defaults["c"],     sigmas["c"],     N),
        "d":     gaussian_param(defaults["d"],     sigmas["d"],     N, clip=(1e-4, None)),
        "e":     gaussian_param(defaults["e"],     sigmas["e"],     N),
        "f":     gaussian_param(defaults["f"],     sigmas["f"],     N, clip=(0.1, None)),
        "g":     gaussian_param(defaults["g"],     sigmas["g"],     N),
        "alpha": gaussian_param(defaults["alpha"], sigmas["alpha"], N),
        "beta":  gaussian_param(defaults["beta"],  sigmas["beta"],  N, clip=(0.1, None)),
        "gamma": gaussian_param(defaults["gamma"], sigmas["gamma"], N),
        "tau":   gaussian_param(defaults["tau"],   sigmas["tau"],   N, clip=(0.1, None)),
    }


# ---------------------------------------------------------------------------
# Model equations  (reused from generic_2doscillator.py)
# ---------------------------------------------------------------------------

def rhs_generic_2d(state, params, coupling_input):
    """
    Right-hand side of the Generic2dOscillator model.

    Parameters
    ----------
    state : ndarray, shape (N, 2)
        Current [V, W] for each node.
    params : dict of ndarrays
        Per-node model parameters.
    coupling_input : ndarray, shape (N,)
        Pre-computed delayed coupling term.
    """
    V = state[:, 0]
    W = state[:, 1]

    dV = params["d"] * params["tau"] * (
        -params["f"] * V**3
        + params["e"] * V**2
        + params["g"] * V
        + params["alpha"] * W
        + params["gamma"] * coupling_input
    )

    dW = (params["d"] / params["tau"]) * (
        params["c"] * V**2
        + params["b"] * V
        - params["beta"] * W
        + params["a"]
    )

    return np.column_stack([dV, dW])


# ---------------------------------------------------------------------------
# Edge-specific coupling
# ---------------------------------------------------------------------------

def coupling_transform(V_source, W_source, coupling_type):
    """Apply per-edge transform to source state variables.

    Parameters
    ----------
    V_source : float or ndarray
        V state of the source node(s).
    W_source : float or ndarray
        W state of the source node(s).
    coupling_type : int or ndarray of ints
        Coupling type code(s).

    Returns
    -------
    transformed : same shape as V_source
    """
    result = np.zeros_like(V_source, dtype=float)

    mask_lin = coupling_type == LINEAR
    mask_quad = coupling_type == QUADRATURE
    mask_rect = coupling_type == RECTIFIED
    mask_sq = coupling_type == SQUARED
    mask_pac = coupling_type == PAC

    result[mask_lin] = V_source[mask_lin]
    result[mask_quad] = W_source[mask_quad]
    result[mask_rect] = np.maximum(V_source[mask_rect], 0.0)
    result[mask_sq] = V_source[mask_sq] ** 2
    # PAC: V acts as a slow envelope that amplitude-modulates the fast W signal.
    # Mirrors the oscillator demo where theta modulates gamma amplitude.
    result[mask_pac] = (0.5 + 0.4 * V_source[mask_pac]) * W_source[mask_pac]

    return result


def compute_typed_coupling(history, step, idelays, weights, coupling_types, G):
    """
    Compute coupling with edge-specific transforms using delayed state.

    Parameters
    ----------
    history : ndarray, shape (horizon, N, 2)
        Circular buffer of past [V, W] values.
    step : int
        Current integration step.
    idelays : ndarray, shape (N, N), dtype int
        Delay for each connection in integration steps.
    weights : ndarray, shape (N, N)
        Connectivity weight matrix.
    coupling_types : ndarray, shape (N, N), dtype int
        Per-edge coupling type (0 = no connection).
    G : float
        Global coupling strength.

    Returns
    -------
    coupling : ndarray, shape (N,)
    """
    N = weights.shape[0]
    horizon = history.shape[0]

    # Gather delayed V and W for all source nodes, per target
    delayed_V = np.empty((N, N))
    delayed_W = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            idx = (step - idelays[i, j]) % horizon
            delayed_V[i, j] = history[idx, j, 0]
            delayed_W[i, j] = history[idx, j, 1]

    # Apply per-edge transforms
    transformed = coupling_transform(delayed_V, delayed_W, coupling_types)

    # Zero out where there is no connection
    transformed[coupling_types == 0] = 0.0

    coupling = G * np.sum(weights * transformed, axis=1)
    return coupling


def build_coupling_types(weights, mode, rng, uniform_type=None):
    """Assign coupling types to nonzero edges.

    Parameters
    ----------
    weights : ndarray (N, N)
        Connectivity weight matrix.
    mode : str
        'random' — each nonzero edge gets a random type (1-4).
        'uniform' — all nonzero edges get the same type.
    rng : numpy.random.Generator
        Random number generator.
    uniform_type : int or None
        Required if mode == 'uniform'. The coupling type to use.

    Returns
    -------
    coupling_types : ndarray (N, N), dtype int
    """
    N = weights.shape[0]
    coupling_types = np.zeros((N, N), dtype=int)
    nonzero = weights != 0

    if mode == "random":
        n_edges = int(nonzero.sum())
        coupling_types[nonzero] = rng.integers(1, len(TYPE_NAMES) + 1, size=n_edges)
    elif mode == "uniform":
        if uniform_type is None:
            raise ValueError("uniform_type must be specified for mode='uniform'")
        coupling_types[nonzero] = uniform_type
    else:
        raise ValueError(f"Unknown edge mode: {mode}")

    # Zero diagonal
    np.fill_diagonal(coupling_types, 0)

    return coupling_types


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sim(conn, G, D, coupling_types, tract_lengths=None, params=None,
            conduction_speed=3.0, dt=0.5, simlen=1000):
    """
    Simulate Generic2dOscillator network with edge-specific coupling types,
    Heun stochastic integration, and temporal-average monitoring.

    Parameters
    ----------
    conn : ndarray, shape (N, N)
        Connectivity weight matrix.
    G : float
        Global coupling strength.
    D : float
        Noise amplitude.
    coupling_types : ndarray, shape (N, N), dtype int
        Per-edge coupling type (0=none, 1=linear, 2=quadrature, 3=rectified, 4=squared, 5=pac).
    tract_lengths : ndarray, shape (N, N), or None
        Fiber tract lengths in mm. If None, all delays are 1 step (instantaneous).
    params : dict of ndarrays or None
        Per-node model parameters. If None, uses homogeneous TVB defaults.
    conduction_speed : float
        Signal propagation speed in mm/ms (only used when tract_lengths is provided).
    dt : float
        Integration time step in ms.
    simlen : float
        Total simulation length in ms.

    Returns
    -------
    t : ndarray
        Monitor time points in ms.
    V_avg : ndarray, shape (n_monitor_steps, N)
        Temporally averaged V for each node.
    """
    N = conn.shape[0]
    n_steps = int(simlen / dt)

    if params is None:
        params = tvb_default_params(N)

    # Convert tract lengths to delays in integration steps
    if tract_lengths is not None:
        delays_ms = tract_lengths / conduction_speed
        idelays = np.round(delays_ms / dt).astype(int)
        idelays = np.clip(idelays, 1, None)
        np.fill_diagonal(idelays, 0)
        print(f"Max delay: {delays_ms.max():.1f} ms = {idelays.max()} steps, "
              f"history buffer size: {int(idelays.max()) + 1}")
    else:
        idelays = np.ones((N, N), dtype=int)
        np.fill_diagonal(idelays, 0)
        print("No tract lengths — using 1-step delays (instantaneous coupling)")

    # Circular history buffer sized to max delay + 1, storing both V and W
    horizon = int(idelays.max()) + 1

    # Initialize history buffer with small random state (V, W)
    history = np.random.normal(0.0, 0.1, (horizon, N, 2))

    # Current state
    curr_state = np.random.normal(0.0, 0.1, (N, 2))
    history[:] = curr_state[np.newaxis, :, :]

    # TemporalAverage monitor (period=5.0 ms)
    monitor_period = 5.0
    skip_step = int(monitor_period / dt)
    buffer_V = np.zeros((skip_step, N))

    out_t = []
    out_V = []

    for k in tqdm(range(n_steps), desc="Simulating", miniters=n_steps // 100):
        # Store current state into the circular history buffer
        history[k % horizon] = curr_state

        # Compute delayed coupling from history with edge-specific transforms
        coupling_input = compute_typed_coupling(
            history, k, idelays, conn, coupling_types, G
        )

        # Heun stochastic integration step
        noise = D * np.sqrt(dt) * np.random.randn(N, 2)

        k1 = rhs_generic_2d(curr_state, params, coupling_input)
        inter_state = curr_state + dt * k1 + noise

        # Recompute coupling at predicted state
        history[k % horizon] = inter_state
        coupling_pred = compute_typed_coupling(
            history, k, idelays, conn, coupling_types, G
        )

        k2 = rhs_generic_2d(inter_state, params, coupling_pred)
        curr_state = curr_state + 0.5 * dt * (k1 + k2) + noise

        # Restore actual state into history
        history[k % horizon] = curr_state

        # TemporalAverage monitor logic
        idx = k % skip_step
        buffer_V[idx] = curr_state[:, 0]

        if idx == skip_step - 1:
            out_t.append(k * dt)
            out_V.append(buffer_V.mean(axis=0))

    return np.array(out_t), np.array(out_V)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_data(filename, t, V, conn, G, D, coupling_types,
              tract_lengths=None, params=None, conduction_speed=3.0,
              dt=0.5, simlen=1000):
    """Save monitor output to a .npz file (compatible with analyze_sim.py)."""
    save_dict = dict(
        t=t, V=V, conn=conn,
        G=G, D=D, coupling_types=coupling_types,
        params=params, conduction_speed=conduction_speed,
        dt=dt, simlen=simlen,
    )
    if tract_lengths is not None:
        save_dict["tract_lengths"] = tract_lengths
    np.savez(filename, **save_dict)
    print(f"Saved → {filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate Generic2dOscillator network with edge-specific coupling types."
    )
    parser.add_argument("--N", type=int, default=15, help="Number of nodes (default: 15)")
    parser.add_argument("--G", type=float, default=0.3, help="Global coupling strength (default: 0.3)")
    parser.add_argument("--D", type=float, default=2e-4, help="Noise amplitude (default: 2e-4)")
    parser.add_argument("--simlen", type=float, default=600000,
                        help="Simulation length in ms (default: 600000 = 10 min)")
    parser.add_argument("--edge-mode", choices=["random", "uniform"], default="random",
                        help="How to assign coupling types to edges (default: random)")
    parser.add_argument("--coupling-type", choices=list(NAME_TO_TYPE.keys()), default=None,
                        help="Coupling type for uniform mode (required if --edge-mode uniform)")
    parser.add_argument("--connectivity",
                        choices=["erdos_renyi", "dense", "sparse", "tree", "hierarchical"],
                        default="erdos_renyi",
                        help="Connectivity generation mode (default: erdos_renyi)")
    parser.add_argument("--density", type=float, default=0.5,
                        help="Edge probability for --connectivity erdos_renyi "
                             "(0 < density ≤ 1, default: 0.5). "
                             "density≈0.1 is sparse, density≈0.9 is near-complete.")
    parser.add_argument("--extra-edges", type=int, default=0,
                        help="Extra non-tree edges appended after tree/hierarchical construction (default: 0)")
    parser.add_argument("--branching", type=int, default=3,
                        help="Max children per node for --connectivity hierarchical (default: 3)")
    parser.add_argument("--no-delays", action="store_true",
                        help="Disable conduction delays (no tract lengths, instantaneous coupling)")
    parser.add_argument("--outdir", default="output/dynamics_sim",
                        help="Output directory (default: output/dynamics_sim)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.edge_mode == "uniform" and args.coupling_type is None:
        parser.error("--coupling-type is required when --edge-mode is uniform")

    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)

    N = args.N

    # --- Connectivity ---
    C = build_connectivity(N, args.connectivity, rng,
                           extra_edges=args.extra_edges, branching=args.branching,
                           density=args.density)
    n_edges = np.count_nonzero(np.triu(C, k=1))
    max_edges = N * (N - 1) // 2
    achieved_density = n_edges / max_edges
    is_symmetric = np.allclose(C, C.T)
    print(f"Connectivity mode: {args.connectivity} — {n_edges}/{max_edges} edges, "
          f"density={achieved_density:.3f}, symmetric={is_symmetric}")

    tract_lengths = None
    if not args.no_delays:
        tract_lengths = rng.uniform(10.0, 150.0, size=(N, N))
        tract_lengths = (tract_lengths + tract_lengths.T) / 2
        np.fill_diagonal(tract_lengths, 0.0)

    # --- Coupling types ---
    uniform_type = NAME_TO_TYPE.get(args.coupling_type) if args.coupling_type else None
    if args.connectivity in ("erdos_renyi", "sparse", "tree", "hierarchical"):
        coupling_types = build_symmetric_coupling_types(C, args.edge_mode, rng, uniform_type=uniform_type)
    else:
        coupling_types = build_coupling_types(C, args.edge_mode, rng, uniform_type=uniform_type)

    # Print coupling type distribution
    nonzero_types = coupling_types[coupling_types > 0]
    print(f"Coupling type distribution ({args.edge_mode} mode):")
    for ct, name in TYPE_NAMES.items():
        count = int((nonzero_types == ct).sum())
        print(f"  {name}: {count} edges")

    # --- Heterogeneous node parameters ---
    defaults = {
        "a": 0.0, "b": -10.0, "c": 0.0, "d": 0.02, "e": 3.0,
        "f": 1.0, "g": 0.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "tau": 1.0,
    }
    sigmas = {
        "a": 0.05, "b": 0.5, "c": 0.01, "d": 0.005, "e": 0.3,
        "f": 0.1, "g": 0.05, "alpha": 0.1, "beta": 0.1, "gamma": 0.1, "tau": 0.15,
    }
    params = build_heterogeneous_params(N, defaults, sigmas)

    for key in ["a", "d", "e", "tau"]:
        vals = params[key]
        print(f"{key:>6s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")

    # --- Run ---
    conduction_speed = 3.0
    dt = 0.5

    t, V = run_sim(
        C, G=args.G, D=args.D,
        coupling_types=coupling_types,
        tract_lengths=tract_lengths,
        params=params,
        conduction_speed=conduction_speed, dt=dt, simlen=args.simlen,
    )

    # --- Save ---
    os.makedirs(args.outdir, exist_ok=True)
    save_data(
        os.path.join(args.outdir, "dynamics_sim.npz"),
        t, V, C, args.G, args.D, coupling_types,
        tract_lengths=tract_lengths,
        params=params, conduction_speed=conduction_speed, dt=dt, simlen=args.simlen,
    )


if __name__ == "__main__":
    main()
