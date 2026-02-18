"""
mi_vs_correlation_demo.py — Demonstrate that MI captures nonlinear dependencies
that Pearson correlation misses.

Two demo modes:

  1. Abstract: paired samples with known math relationships (linear, quadratic,
     sinusoidal, circular).
  2. Neural oscillators: coupled oscillator time series with neurophysiologically
     motivated relationships (in-phase sync, quadrature phase shift,
     cross-frequency phase-amplitude coupling, nonlinear burst coupling).

Usage:
    python code/simulations/mi_vs_correlation_demo.py                          # abstract demo
    python code/simulations/mi_vs_correlation_demo.py --mode oscillator        # neural demo
    python code/simulations/mi_vs_correlation_demo.py --mode oscillator --outdir code/simulations/output/mi_vs_corr
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def generate_pairs(n=2000, noise_std=0.15, seed=42):
    """Generate (X, Y) pairs for each dependency type."""
    rng = np.random.default_rng(seed)

    pairs = {}

    # 1. Linear: Y = X + noise
    x = rng.standard_normal(n)
    pairs["Linear\n$Y = X + \\epsilon$"] = (
        x, x + rng.normal(0, noise_std, n)
    )

    # 2. Quadratic: Y = X^2 + noise
    x = rng.standard_normal(n)
    pairs["Quadratic\n$Y = X^2 + \\epsilon$"] = (
        x, x**2 + rng.normal(0, noise_std, n)
    )

    # 3. Sinusoidal: Y = sin(4X) + noise
    x = rng.uniform(-np.pi, np.pi, n)
    pairs["Sinusoidal\n$Y = \\sin(4X) + \\epsilon$"] = (
        x, np.sin(4 * x) + rng.normal(0, noise_std, n)
    )

    # 4. Circular: (cos θ, sin θ) + noise
    theta = rng.uniform(0, 2 * np.pi, n)
    pairs["Circular\n$X = \\cos\\theta,\\ Y = \\sin\\theta + \\epsilon$"] = (
        np.cos(theta) + rng.normal(0, noise_std, n),
        np.sin(theta) + rng.normal(0, noise_std, n),
    )

    return pairs


# ---------------------------------------------------------------------------
# Neural oscillator generators
# ---------------------------------------------------------------------------

def _oscillator(freq, t, phase=0.0, amp=1.0):
    """Single sinusoidal oscillator."""
    return amp * np.sin(2 * np.pi * freq * t + phase)


def generate_oscillator_pairs(duration=10.0, fs=500.0, noise_std=0.15, seed=42):
    """Generate coupled neural oscillator time series.

    Returns
    -------
    pairs : dict[str, (x, y, t)]
        Each value is (x_timeseries, y_timeseries, time_vector).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    n = len(t)
    noise = lambda: rng.normal(0, noise_std, n)

    pairs = {}

    # 1. In-phase synchronization (both at 8 Hz, same phase)
    #    Linear coupling → high corr AND high MI
    freq = 8.0
    x = _oscillator(freq, t) + noise()
    y = _oscillator(freq, t) + noise()
    pairs["In-phase sync\n(same freq, same phase)"] = (x, y, t)

    # 2. Quadrature phase shift (8 Hz, 90° offset)
    #    Signals are perfectly synchronized but orthogonal → corr ≈ 0, MI high
    x = _oscillator(freq, t, phase=0.0) + noise()
    y = _oscillator(freq, t, phase=np.pi / 2) + noise()
    pairs["Quadrature coupling\n(same freq, 90° shift)"] = (x, y, t)

    # 3. Cross-frequency phase-amplitude coupling (PAC)
    #    Slow theta (5 Hz) modulates the amplitude of fast gamma (40 Hz)
    #    Nonlinear relationship → corr ≈ 0, MI > 0
    f_slow, f_fast = 5.0, 40.0
    x = _oscillator(f_slow, t) + noise()
    theta_phase = 2 * np.pi * f_slow * t
    # Gamma amplitude peaks when theta is at its peak (modulation depth = 0.8)
    gamma_envelope = 0.5 + 0.4 * np.cos(theta_phase)
    y = gamma_envelope * _oscillator(f_fast, t) + noise()
    pairs["Phase-amplitude coupling\n(theta modulates gamma)"] = (x, y, t)

    # 4. Nonlinear burst coupling
    #    X oscillates; Y bursts only when X exceeds a threshold (rectified)
    #    Like an excitatory neuron driving a downstream neuron past threshold
    freq_x = 6.0
    x = _oscillator(freq_x, t) + noise()
    x_rect = np.clip(x, 0, None)  # rectify: only positive phases
    y = x_rect * _oscillator(35.0, t) + noise()
    pairs["Burst coupling\n(X threshold gates Y)"] = (x, y, t)

    # 5. Squared / frequency-doubling coupling
    #    Y receives X² — energy appears at 2× the source frequency
    #    Symmetric nonlinearity → corr ≈ 0, MI > 0
    freq_sq = 8.0
    x = _oscillator(freq_sq, t) + noise()
    y = x ** 2 + noise()
    pairs["Squared coupling\n($Y = X^2$)"] = (x, y, t)

    return pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pearson_corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def compute_mi_binned(x, y, num_bins=30):
    """Compute MI using the same discretization approach as the fMRI pipeline."""
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, T)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, T)

    ts = torch.cat([x_t, y_t], dim=0)  # (2, T)
    ts = ts.contiguous()

    bin_edges = torch.linspace(ts.min(), ts.max(), steps=num_bins + 1)[1:-1]
    disc = torch.bucketize(ts, bin_edges)

    # Joint probability
    a_vals = disc[0]
    b_vals = disc[1]
    joint_idx = num_bins * a_vals + b_vals
    jp = torch.bincount(joint_idx, minlength=num_bins * num_bins).reshape(
        num_bins, num_bins
    ).float()
    jp /= jp.sum()

    # Marginals and their product
    marginal_a = jp.sum(dim=1)
    marginal_b = jp.sum(dim=0)
    pm = torch.outer(marginal_a, marginal_b)

    # MI
    jp_safe = jp.clone().clamp(min=1e-10)
    pm_safe = pm.clone().clamp(min=1e-10)
    mi = float(torch.sum(jp_safe * torch.log(jp_safe / pm_safe)))
    return mi


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(pairs, outdir=None):
    names = list(pairs.keys())
    n_pairs = len(names)

    corrs = [abs(pearson_corr(*pairs[n])) for n in names]
    mis = [compute_mi_binned(*pairs[n]) for n in names]

    # Normalize MI to [0, 1] for visual comparison (divide by max)
    mi_max = max(mis)
    mis_norm = [m / mi_max for m in mis]

    fig, axes = plt.subplots(2, n_pairs, figsize=(3.5 * n_pairs, 6),
                             gridspec_kw={"height_ratios": [2, 1]})

    # --- Top row: scatter plots ---
    for i, name in enumerate(names):
        ax = axes[0, i]
        x, y = pairs[name]
        ax.scatter(x, y, s=1.5, alpha=0.4, color="steelblue", rasterized=True)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("X")
        if i == 0:
            ax.set_ylabel("Y")
        ax.tick_params(labelsize=8)

    # --- Bottom row: grouped bars ---
    bar_width = 0.3
    x_pos = np.arange(n_pairs)

    # Create a single axis spanning the bottom row
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]:
        ax.remove()
    ax_bar = fig.add_subplot(gs[1, :])

    bars_corr = ax_bar.bar(x_pos - bar_width / 2, corrs, bar_width,
                           label="|Pearson $r$|", color="salmon", edgecolor="k",
                           linewidth=0.5)
    bars_mi = ax_bar.bar(x_pos + bar_width / 2, mis_norm, bar_width,
                         label="MI (normalized)", color="steelblue", edgecolor="k",
                         linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars_corr, corrs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_mi, mis_norm):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    short_labels = ["Linear", "Quadratic", "Sinusoidal", "Circular"]
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(short_labels, fontsize=10)
    ax_bar.set_ylabel("Magnitude", fontsize=11)
    ax_bar.set_ylim(0, 1.3)
    ax_bar.legend(loc="upper right", fontsize=10)
    ax_bar.axhline(0, color="k", linewidth=0.5)

    fig.suptitle("MI captures nonlinear dependencies that correlation misses",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "mi_vs_correlation.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Oscillator figure
# ---------------------------------------------------------------------------

def make_oscillator_figure(pairs, outdir=None, t_window=(0.0, 2.0)):
    """5-column figure: one column per coupling type.

    Rows: X(t) | Y(t) | X-vs-Y scatter | grouped bar chart (corr vs MI).
    """
    names = list(pairs.keys())
    n_pairs = len(names)

    # Compute metrics (use full time series, not just display window)
    corrs = [abs(pearson_corr(pairs[n][0], pairs[n][1])) for n in names]
    mis = [compute_mi_binned(pairs[n][0], pairs[n][1], num_bins=50) for n in names]
    mi_max = max(mis)
    mis_norm = [m / mi_max for m in mis]

    fig, axes = plt.subplots(4, n_pairs,
                             figsize=(4 * n_pairs, 10),
                             gridspec_kw={"height_ratios": [1, 1, 1.5, 1]})

    colors_x = "tab:blue"
    colors_y = "tab:orange"

    for i, name in enumerate(names):
        x, y, t = pairs[name]

        # Time window mask for plotting (show a readable snippet)
        mask = (t >= t_window[0]) & (t < t_window[1])
        t_plot = t[mask]

        # --- Row 0: X(t) ---
        ax = axes[0, i]
        ax.plot(t_plot, x[mask], linewidth=0.6, color=colors_x)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(t_window)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("X(t)", fontsize=10)
        ax.set_xticklabels([])

        # --- Row 1: Y(t) ---
        ax = axes[1, i]
        ax.plot(t_plot, y[mask], linewidth=0.6, color=colors_y)
        ax.set_xlim(t_window)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("Y(t)", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)

        # --- Row 2: X vs Y scatter ---
        ax = axes[2, i]
        ax.scatter(x, y, s=0.8, alpha=0.25, color="steelblue", rasterized=True)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("Y", fontsize=10)
        ax.set_xlabel("X", fontsize=8)

        # Annotate with raw values
        ax.text(0.05, 0.95, f"|r| = {corrs[i]:.3f}\nMI = {mis[i]:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # --- Row 3: grouped bar chart spanning all columns ---
    gs = axes[3, 0].get_gridspec()
    for ax in axes[3, :]:
        ax.remove()
    ax_bar = fig.add_subplot(gs[3, :])

    bar_width = 0.3
    x_pos = np.arange(n_pairs)

    bars_corr = ax_bar.bar(x_pos - bar_width / 2, corrs, bar_width,
                           label="|Pearson $r$|", color="salmon", edgecolor="k",
                           linewidth=0.5)
    bars_mi = ax_bar.bar(x_pos + bar_width / 2, mis_norm, bar_width,
                         label="MI (normalized)", color="steelblue", edgecolor="k",
                         linewidth=0.5)

    for bar, val in zip(bars_corr, corrs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_mi, mis_norm):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    short_labels = ["In-phase", "Quadrature", "PAC", "Burst", "Squared"]
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(short_labels, fontsize=10)
    ax_bar.set_ylabel("Magnitude", fontsize=11)
    ax_bar.set_ylim(0, 1.35)
    ax_bar.legend(loc="upper right", fontsize=10)

    fig.suptitle("MI captures nonlinear coupling between neural oscillators",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "mi_vs_correlation_oscillators.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Edge-specific coupling types figure
# ---------------------------------------------------------------------------

def generate_coupling_type_signals(duration=4.0, fs=500.0, noise_std=0.05, seed=42):
    """Generate a Generic2dOscillator-like source (V, W) and apply each coupling transform.

    Returns
    -------
    data : dict with keys 'V', 'W', 't', and a sub-dict 'transforms' mapping
           type name → transformed signal.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    n = len(t)

    # Simulate a Generic2dOscillator-like pair (V, W) where W is the
    # slow recovery variable ~90° behind V
    freq = 6.0  # Hz — theta-range oscillation
    V = np.sin(2 * np.pi * freq * t) + rng.normal(0, noise_std, n)
    W = np.cos(2 * np.pi * freq * t) + rng.normal(0, noise_std, n)  # ~90° shifted

    transforms = {
        "Linear\n$c_{{ij}} = V_j$": V.copy(),
        "Quadrature\n$c_{{ij}} = W_j$": W.copy(),
        "Rectified\n$c_{{ij}} = \\max(V_j, 0)$": np.maximum(V, 0.0),
        "Squared\n$c_{{ij}} = V_j^2$": V ** 2,
        "PAC\n$c_{{ij}} = (0.5{+}0.4 V_j) W_j$": (0.5 + 0.4 * V) * W,
    }

    return {"V": V, "W": W, "t": t, "transforms": transforms}


def make_coupling_types_figure(data, outdir=None, t_window=(0.0, 2.0)):
    """5-column figure illustrating edge-specific coupling transforms.

    Rows:
        0 — Source V(t) with the transform highlighted
        1 — Transformed coupling signal c_ij(t)
        2 — V vs c_ij scatter (shows the functional relationship)
        3 — Grouped bar chart: |Pearson r| vs MI(normalized) for each transform
    """
    V = data["V"]
    W = data["W"]
    t = data["t"]
    transforms = data["transforms"]
    names = list(transforms.keys())
    n_types = len(names)

    mask = (t >= t_window[0]) & (t < t_window[1])
    t_plot = t[mask]

    # Compute metrics
    corrs = [abs(pearson_corr(V, transforms[n])) for n in names]
    mis = [compute_mi_binned(V, transforms[n], num_bins=50) for n in names]
    mi_max = max(mis)
    mis_norm = [m / mi_max for m in mis]

    fig, axes = plt.subplots(4, n_types,
                             figsize=(4 * n_types, 10),
                             gridspec_kw={"height_ratios": [1, 1, 1.5, 1]})

    type_colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for i, name in enumerate(names):
        c_ij = transforms[name]
        color = type_colors[i]

        # --- Row 0: Source V(t) and W(t) ---
        ax = axes[0, i]
        ax.plot(t_plot, V[mask], linewidth=0.7, color="tab:blue", alpha=0.6, label="V(t)")
        ax.plot(t_plot, W[mask], linewidth=0.7, color="tab:orange", alpha=0.4, label="W(t)")
        ax.set_title(name, fontsize=10)
        ax.set_xlim(t_window)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("Source node", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xticklabels([])

        # --- Row 1: Transformed signal c_ij(t) ---
        ax = axes[1, i]
        ax.plot(t_plot, c_ij[mask], linewidth=0.7, color=color)
        ax.set_xlim(t_window)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("Coupling $c_{ij}(t)$", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)

        # --- Row 2: V vs c_ij scatter ---
        ax = axes[2, i]
        ax.scatter(V, c_ij, s=0.6, alpha=0.2, color=color, rasterized=True)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("$c_{ij}$", fontsize=10)
        ax.set_xlabel("$V_j$", fontsize=8)

        ax.text(0.05, 0.95, f"|r| = {corrs[i]:.3f}\nMI = {mis[i]:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # --- Row 3: grouped bar chart spanning all columns ---
    gs = axes[3, 0].get_gridspec()
    for ax in axes[3, :]:
        ax.remove()
    ax_bar = fig.add_subplot(gs[3, :])

    bar_width = 0.3
    x_pos = np.arange(n_types)

    bars_corr = ax_bar.bar(x_pos - bar_width / 2, corrs, bar_width,
                           label="|Pearson $r$|", color="salmon", edgecolor="k",
                           linewidth=0.5)
    bars_mi = ax_bar.bar(x_pos + bar_width / 2, mis_norm, bar_width,
                         label="MI (normalized)", color="steelblue", edgecolor="k",
                         linewidth=0.5)

    for bar, val in zip(bars_corr, corrs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_mi, mis_norm):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    short_labels = ["Linear", "Quadrature", "Rectified", "Squared", "PAC"]
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(short_labels, fontsize=10)
    ax_bar.set_ylabel("Magnitude", fontsize=11)
    ax_bar.set_ylim(0, 1.35)
    ax_bar.legend(loc="upper right", fontsize=10)

    fig.suptitle("Edge-specific coupling types in dynamics simulation",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "coupling_types.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Demo: MI vs correlation on nonlinear dependencies."
    )
    parser.add_argument("--mode", choices=["abstract", "oscillator", "coupling-types"],
                        default="abstract",
                        help="'abstract' for math relationships, 'oscillator' for neural signals, "
                             "'coupling-types' for dynamics_simulation edge transforms")
    parser.add_argument("--outdir", default=None,
                        help="Directory to save figure (default: display only)")
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of samples per pair (abstract mode)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in seconds (oscillator mode)")
    parser.add_argument("--fs", type=float, default=500.0,
                        help="Sampling rate in Hz (oscillator mode)")
    parser.add_argument("--noise", type=float, default=0.15,
                        help="Noise std")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "abstract":
        pairs = generate_pairs(n=args.n, noise_std=args.noise, seed=args.seed)
        make_figure(pairs, outdir=args.outdir)
    elif args.mode == "oscillator":
        pairs = generate_oscillator_pairs(
            duration=args.duration, fs=args.fs,
            noise_std=args.noise, seed=args.seed,
        )
        make_oscillator_figure(pairs, outdir=args.outdir)
    elif args.mode == "coupling-types":
        data = generate_coupling_type_signals(
            duration=args.duration, fs=args.fs,
            noise_std=args.noise, seed=args.seed,
        )
        make_coupling_types_figure(data, outdir=args.outdir)


if __name__ == "__main__":
    main()
