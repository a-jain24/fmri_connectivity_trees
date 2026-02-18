import numpy as np
from tqdm.notebook import tqdm
import os


# ---------------------------------------------------------------------------
# Default parameters
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
# Heterogeneous parameter generation
# ---------------------------------------------------------------------------

def gaussian_param(default, sigma, N, clip=None):
    """Sample N values from a Gaussian around a default."""
    values = np.random.normal(default, sigma, N)
    if clip is not None:
        values = np.clip(values, clip[0], clip[1])
    return values


def build_heterogeneous_params(N, defaults, sigmas):
    """
    Build per-node parameter arrays by sampling around TVB defaults.

    Each parameter is drawn independently per node from
    N(default, sigma), with clipping where needed to keep
    values physically meaningful.
    """
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
# Model equations
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
        Pre-computed delayed coupling term: G * sum_j(C_ij * V_j(t - delay_ij))
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
# Delayed coupling
# ---------------------------------------------------------------------------

def compute_delayed_coupling(history, step, idelays, weights, G):
    """
    Compute the coupling term using delayed state variables from a
    circular history buffer, matching TVB's Linear coupling.

    Parameters
    ----------
    history : ndarray, shape (horizon, N)
        Circular buffer of past V values.
    step : int
        Current integration step (used to index into circular buffer).
    idelays : ndarray, shape (N, N), dtype int
        Delay for each connection in integration steps.
    weights : ndarray, shape (N, N)
        Connectivity weight matrix.
    G : float
        Global coupling strength (TVB Linear coupling 'a' parameter).

    Returns
    -------
    coupling : ndarray, shape (N,)
    """
    N = weights.shape[0]
    horizon = history.shape[0]

    delayed_V = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            delayed_V[i, j] = history[(step - idelays[i, j]) % horizon, j]

    coupling = G * np.sum(weights * delayed_V, axis=1)
    return coupling


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sim(conn, tract_lengths, G, D, params=None, conduction_speed=3.0,
            dt=0.5, simlen=1000):
    """
    Simulate Generic2dOscillator network with conduction delays,
    Heun stochastic integration, and temporal-average monitoring.

    Parameters
    ----------
    conn : ndarray, shape (N, N)
        Connectivity weight matrix.
    tract_lengths : ndarray, shape (N, N)
        Fiber tract lengths in mm between each pair of regions.
    G : float
        Global coupling strength (Linear coupling 'a' parameter).
    D : float
        Noise amplitude (nsig for Additive noise on both state vars).
    params : dict of ndarrays or None
        Per-node model parameters. If None, uses homogeneous TVB defaults.
    conduction_speed : float
        Signal propagation speed in mm/ms (TVB default: 3.0).
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
    delays_ms = tract_lengths / conduction_speed
    idelays = np.round(delays_ms / dt).astype(int)
    idelays = np.clip(idelays, 1, None)
    np.fill_diagonal(idelays, 0)

    # Circular history buffer sized to max delay + 1
    horizon = int(idelays.max()) + 1
    print(f"Max delay: {delays_ms.max():.1f} ms = {idelays.max()} steps, "
          f"history buffer size: {horizon}")

    # Initialize history buffer with small random state (V only)
    history = np.random.normal(0.0, 0.1, (horizon, N))

    # Current state
    curr_state = np.random.normal(0.0, 0.1, (N, 2))
    history[:] = curr_state[:, 0]

    # TemporalAverage monitor (period=5.0 ms)
    monitor_period = 5.0
    skip_step = int(monitor_period / dt)
    buffer_V = np.zeros((skip_step, N))

    out_t = []
    out_V = []

    for k in tqdm(range(n_steps), desc="Simulating", miniters=n_steps // 100):
        # Store current V into the circular history buffer
        history[k % horizon] = curr_state[:, 0]

        # Compute delayed coupling from history
        coupling_input = compute_delayed_coupling(
            history, k, idelays, conn, G
        )

        # Heun stochastic integration step
        noise = D * np.sqrt(dt) * np.random.randn(N, 2)

        k1 = rhs_generic_2d(curr_state, params, coupling_input)
        inter_state = curr_state + dt * k1 + noise

        # Recompute coupling at predicted state
        history[k % horizon] = inter_state[:, 0]
        coupling_pred = compute_delayed_coupling(
            history, k, idelays, conn, G
        )

        k2 = rhs_generic_2d(inter_state, params, coupling_pred)
        curr_state = curr_state + 0.5 * dt * (k1 + k2) + noise

        # Restore actual state into history
        history[k % horizon] = curr_state[:, 0]

        # TemporalAverage monitor logic
        idx = k % skip_step
        buffer_V[idx] = curr_state[:, 0]

        if idx == skip_step - 1:
            out_t.append(k * dt)
            out_V.append(buffer_V.mean(axis=0))

    return np.array(out_t), np.array(out_V)

def save_data(filename, t, V, conn, tract_lengths, G, D, params=None, conduction_speed=3.0,
            dt=0.5, simlen=1000):
    """Save monitor output to a .npz file."""
    np.savez(filename, t=t, V=V, conn=conn, tract_lengths=tract_lengths, G=G, D=D,
             params=params, conduction_speed=conduction_speed, dt=dt, simlen=simlen)
    

def main():
    N = 15

    # --- Connectivity (reuse from above or regenerate) ---
    C = np.random.randint(0, 4, size=(N, N)).astype(float)
    np.fill_diagonal(C, 0.0)

    tract_lengths = np.random.uniform(10.0, 150.0, size=(N, N))
    tract_lengths = (tract_lengths + tract_lengths.T) / 2
    np.fill_diagonal(tract_lengths, 0.0)

    # --- TVB defaults as center values ---
    defaults = {
        "a":     0.0,
        "b":    -10.0,
        "c":     0.0,
        "d":     0.02,
        "e":     3.0,
        "f":     1.0,
        "g":     0.0,
        "alpha": 1.0,
        "beta":  1.0,
        "gamma": 1.0,
        "tau":   1.0,
    }

    # --- Per-parameter spread (sigma) ---
    # Kept small enough that every node stays in a qualitatively
    # similar dynamical regime (excitable), but large enough that
    # amplitudes, phases, and intrinsic frequencies differ visibly.
    sigmas = {
        "a":     0.05,   # slight shifts in excitability threshold
        "b":     0.5,    # moderate spread in W->V linear coupling
        "c":     0.01,   # V^2 in dW is 0 by default, keep tight
        "d":     0.005,  # overall time-scale — small spread
        "e":     0.3,    # V^2 in dV — affects intrinsic frequency
        "f":     0.1,    # V^3 in dV — affects amplitude ceiling
        "g":     0.05,   # linear V in dV
        "alpha": 0.1,    # W coupling into dV
        "beta":  0.1,    # W self-decay
        "gamma": 0.1,    # network coupling gain
        "tau":   0.15,   # V/W time-scale separation — key for frequency spread
    }

    params = build_heterogeneous_params(N, defaults, sigmas)

    # Quick sanity check: print a few params across nodes
    for key in ["a", "d", "e", "tau"]:
        vals = params[key]
        print(f"{key:>6s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}")

    G = 0.3
    D = 2e-4
    conduction_speed = 3.0
    dt = 0.5
    simlen = 10 * 60e3  # 10 minutes in ms

    # --- Run ---
    t, V = run_sim(
        C, tract_lengths, G=G, D=D,
        params=params,
        conduction_speed=conduction_speed, dt=dt, simlen=simlen
    )

    run_name = "testing2_2_11_26"

    if not os.path.exists(f"output/{run_name}"):
        os.makedirs(f"output/{run_name}")

    save_data(f"output/{run_name}/generic_2doscillator_sim.npz", t, V, C, tract_lengths, G, D, params=params, conduction_speed=conduction_speed, dt=dt, simlen=simlen)


if __name__ == "__main__":
    main()