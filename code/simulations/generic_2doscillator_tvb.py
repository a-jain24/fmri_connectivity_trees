"""
Generic2dOscillator simulation using The Virtual Brain API.

Mirrors generic_2doscillator.py but uses TVB's Simulator, connectivity,
integrator, and monitors rather than a from-scratch implementation.
Supports random connectivity matrices and heterogeneous per-node parameters
sampled from Gaussians around the TVB defaults.
"""

import numpy as np
import os
import time
import argparse

from tvb.simulator.lab import (
    simulator, models, coupling, integrators, noise, monitors, connectivity
)


# ---------------------------------------------------------------------------
# Heterogeneous parameter generation
# ---------------------------------------------------------------------------

def gaussian_param(default, sigma, N, clip=None):
    """Sample N values from a Gaussian around a default."""
    values = np.random.normal(default, sigma, N)
    if clip is not None:
        values = np.clip(values, clip[0], clip[1])
    return values


def build_heterogeneous_model(N, defaults, sigmas):
    """
    Create a Generic2dOscillator with per-node heterogeneous parameters.

    Each parameter is drawn from N(default, sigma) independently per node,
    then reshaped to the TVB convention for per-node arrays.

    Parameters
    ----------
    N : int
        Number of nodes.
    defaults : dict
        Default (mean) value for each model parameter.
    sigmas : dict
        Standard deviation for each model parameter.

    Returns
    -------
    model : models.Generic2dOscillator
    params_dict : dict of ndarrays (flat, for saving)
    """
    params_dict = {
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

    model = models.Generic2dOscillator(
        a=np.array(params_dict["a"]),
        b=np.array(params_dict["b"]),
        c=np.array(params_dict["c"]),
        d=np.array(params_dict["d"]),
        e=np.array(params_dict["e"]),
        f=np.array(params_dict["f"]),
        g=np.array(params_dict["g"]),
        alpha=np.array(params_dict["alpha"]),
        beta=np.array(params_dict["beta"]),
        gamma=np.array(params_dict["gamma"]),
        tau=np.array(params_dict["tau"]),
    )

    return model, params_dict


# ---------------------------------------------------------------------------
# Custom random connectivity
# ---------------------------------------------------------------------------

def build_random_connectivity(N, weight_range=(0, 4), tract_length_range=(10.0, 150.0),
                              conduction_speed=3.0):
    """
    Build a TVB Connectivity object with random weights and tract lengths.

    Parameters
    ----------
    N : int
        Number of regions.
    weight_range : tuple
        (low, high) for random integer weights.
    tract_length_range : tuple
        (low, high) in mm for uniform random tract lengths.
    conduction_speed : float
        Conduction speed in mm/ms (stored on the Connectivity).

    Returns
    -------
    conn : connectivity.Connectivity
    """
    weights = np.random.randint(weight_range[0], weight_range[1], size=(N, N)).astype(float)
    np.fill_diagonal(weights, 0.0)

    tract_lengths = np.random.uniform(tract_length_range[0], tract_length_range[1], size=(N, N))
    tract_lengths = (tract_lengths + tract_lengths.T) / 2  # symmetric
    np.fill_diagonal(tract_lengths, 0.0)

    # Minimal required fields for TVB Connectivity
    region_labels = np.array([f"R{i:03d}" for i in range(N)])
    centres = np.random.randn(N, 3) * 50  # arbitrary 3D coordinates

    conn = connectivity.Connectivity(
        weights=weights,
        tract_lengths=tract_lengths,
        region_labels=region_labels,
        centres=centres,
        speed=np.array([conduction_speed]),
    )
    conn.configure()

    return conn


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sim(conn, model, G, D, dt=0.5, simlen=1e3):
    """
    Run a TVB simulation with the given connectivity and model.

    Parameters
    ----------
    conn : connectivity.Connectivity
    model : models.Generic2dOscillator
    G : float
        Global coupling strength (Linear coupling 'a').
    D : float
        Noise amplitude (Additive noise nsig).
    dt : float
        Integration step in ms.
    simlen : float
        Simulation length in ms.

    Returns
    -------
    t : ndarray, shape (n_timepoints,)
        Time in ms.
    y : ndarray, shape (n_timepoints, N)
        Temporally averaged V for each node.
    """
    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupling.Linear(a=np.array([G])),
        integrator=integrators.HeunStochastic(
            dt=dt,
            noise=noise.Additive(nsig=np.array([D]))
        ),
        monitors=(monitors.TemporalAverage(period=5.0),),  # 200 Hz
    )
    sim.configure()

    tic = time.time()
    (t, y), = sim.run(simulation_length=simlen)
    elapsed = time.time() - tic
    print(f"Simulation completed in {elapsed:.1f}s  "
          f"({simlen/1e3:.0f}s simulated, {y.shape[0]} monitor steps)")

    # y shape from TVB: (time, state_vars, nodes, modes) → extract V
    return t, y[:, 0, :, 0]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_data(filename, t, V, conn, G, D, params_dict=None,
              conduction_speed=3.0, dt=0.5, simlen=1000):
    """Save simulation output and metadata to .npz."""
    save_dict = dict(
        t=t, V=V,
        weights=conn.weights, tract_lengths=conn.tract_lengths,
        G=G, D=D,
        conduction_speed=conduction_speed, dt=dt, simlen=simlen,
    )
    if params_dict is not None:
        for k, v in params_dict.items():
            save_dict[f"param_{k}"] = v

    np.savez(filename, **save_dict)
    print(f"Saved to {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TVB Generic2dOscillator simulation with random connectivity "
                    "and heterogeneous per-node parameters."
    )
    parser.add_argument("--N", type=int, default=76, help="Number of nodes")
    parser.add_argument("--G", type=float, default=0.3, help="Global coupling strength")
    parser.add_argument("--D", type=float, default=2e-4, help="Noise amplitude")
    parser.add_argument("--speed", type=float, default=3.0, help="Conduction speed (mm/ms)")
    parser.add_argument("--dt", type=float, default=0.5, help="Integration step (ms)")
    parser.add_argument("--simlen", type=float, default=10*60e3,
                        help="Simulation length (ms), default 10 min")
    parser.add_argument("--outdir", type=str, default="output/tvb_hetero",
                        help="Output directory")
    parser.add_argument("--homogeneous", action="store_true",
                        help="Use homogeneous (default) parameters instead of heterogeneous")
    args = parser.parse_args()

    N = args.N

    # --- Connectivity ---
    conn = build_random_connectivity(N, conduction_speed=args.speed)
    print(f"Connectivity: {N} nodes, weight range "
          f"[{conn.weights.min():.0f}, {conn.weights.max():.0f}], "
          f"tract length range [{conn.tract_lengths[conn.tract_lengths > 0].min():.1f}, "
          f"{conn.tract_lengths.max():.1f}] mm")

    # --- Model ---
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

    sigmas = {
        "a":     0.05,
        "b":     0.5,
        "c":     0.01,
        "d":     0.005,
        "e":     0.3,
        "f":     0.1,
        "g":     0.05,
        "alpha": 0.1,
        "beta":  0.1,
        "gamma": 0.1,
        "tau":   0.15,
    }

    if args.homogeneous:
        model = models.Generic2dOscillator(a=np.array([defaults["a"]]))
        params_dict = None
        print("Using homogeneous TVB default parameters")
    else:
        model, params_dict = build_heterogeneous_model(N, defaults, sigmas)
        for key in ["a", "d", "e", "tau"]:
            vals = params_dict[key]
            print(f"  {key:>6s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")

    # --- Simulate ---
    t, V = run_sim(conn, model, G=args.G, D=args.D, dt=args.dt, simlen=args.simlen)

    # --- Save ---
    os.makedirs(args.outdir, exist_ok=True)
    tag = "homogeneous" if args.homogeneous else "heterogeneous"
    filename = os.path.join(args.outdir, f"generic_2doscillator_tvb_{tag}.npz")
    save_data(filename, t, V, conn, args.G, args.D,
              params_dict=params_dict, conduction_speed=args.speed,
              dt=args.dt, simlen=args.simlen)


if __name__ == "__main__":
    main()
