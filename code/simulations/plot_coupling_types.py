"""
plot_coupling_types.py — Show V timeseries for each of the 5 coupling types.

Panel A (top): The source node's V and W signals (free oscillator, noise-driven).
Panel B (rows 1-5): The coupling signal injected into the target for each type,
                    plus the target node's V response.

Setup: 2-node network, node 0 → node 1 only (unidirectional) so node 1's
behaviour reflects the coupling type without feedback.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dynamics_simulation import (
    run_sim, tvb_default_params,
    LINEAR, QUADRATURE, RECTIFIED, SQUARED, PAC, TYPE_NAMES,
)

# ---------------------------------------------------------------------------
# Parameters chosen to produce sustained stochastic oscillations
# ---------------------------------------------------------------------------
SEED = 7
np.random.seed(SEED)

N = 2
G = 0.8          # stronger coupling so target response is visible
D = 0.015        # noise level — sustains oscillations
DT = 0.5
SIMLEN = 15_000  # 15 s

# Unidirectional: 0 → 1 only (so 1 doesn't feed back into 0)
C = np.array([[0.0, 0.0],
              [1.5, 0.0]])   # only C[1,0] is nonzero

COUPLING_ORDER = [LINEAR, QUADRATURE, RECTIFIED, SQUARED, PAC]

COLORS = {
    LINEAR:     "#4C72B0",
    QUADRATURE: "#DD8452",
    RECTIFIED:  "#55A868",
    SQUARED:    "#C44E52",
    PAC:        "#8172B3",
}

DESCRIPTIONS = {
    LINEAR:     r"$\mathbf{1\ –\ Linear}$" + "\n" + r"$V_j$",
    QUADRATURE: r"$\mathbf{2\ –\ Quadrature}$" + "\n" + r"$W_j$",
    RECTIFIED:  r"$\mathbf{3\ –\ Rectified}$" + "\n" + r"$\max(V_j,\,0)$",
    SQUARED:    r"$\mathbf{4\ –\ Squared}$" + "\n" + r"$V_j^2$",
    PAC:        r"$\mathbf{5\ –\ PAC}$" + "\n" + r"$(0.5 + 0.4\,V_j)\,W_j$",
}

# ---------------------------------------------------------------------------
# Run one simulation per coupling type (same seed → same source node)
# ---------------------------------------------------------------------------
results = {}
for ct in COUPLING_ORDER:
    np.random.seed(SEED)
    coupling_types = np.array([[0, 0],
                                [ct, 0]], dtype=int)   # 1→0 only
    params = tvb_default_params(N)
    t, V = run_sim(C, G=G, D=D, coupling_types=coupling_types,
                   tract_lengths=None, params=params, dt=DT, simlen=SIMLEN)
    results[ct] = (t, V)
    print(f"Done: {TYPE_NAMES[ct]}")

# ---------------------------------------------------------------------------
# Compute coupling signals analytically from the *linear* run's source trace
# (all runs share the same seed so node 0 is nearly identical; use LINEAR run)
# ---------------------------------------------------------------------------
t_ref, V_ref = results[LINEAR]
V_src = V_ref[:, 0]   # node 0 V  (same across runs to good approximation)

# We also need W from the source node — re-run a single-node simulation to get W
np.random.seed(SEED)
# Use a 1-node uncoupled sim to get clean W
C1 = np.array([[0.0]])
ct1 = np.array([[0]], dtype=int)
p1 = tvb_default_params(1)
t1, V1 = run_sim(C1, G=0.0, D=D, coupling_types=ct1,
                 tract_lengths=None, params=p1, dt=DT, simlen=SIMLEN)
# That only gives V; we need W — pull it from the LINEAR simulation node0
# Since node 0 is uncoupled in the unidirectional setup, we can reconstruct W
# via the model equation at each step.  The simpler approach: show everything
# from the saved V traces and derive coupling signals analytically where possible.

# For W we use a proxy: W ≈ low-pass filtered integral of V (phase-quadrature)
# More precisely: extract W by band-pass + 90° shift of V_src
from scipy.signal import butter, filtfilt, hilbert

def bandpass(sig, lo=0.01, hi=2.0, fs=200.0):
    """Butterworth bandpass filter."""
    nyq = fs / 2
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)

fs = 1000.0 / (DT * 10)   # monitor period = 5 ms → 200 Hz
V_filt = bandpass(V_src, lo=0.05, hi=5.0, fs=fs)
# W approximation: 90° phase-shifted version via Hilbert
analytic = hilbert(V_filt)
W_src = np.imag(analytic)
W_src = W_src * (np.std(V_filt) / (np.std(W_src) + 1e-12))  # match amplitude

# Coupling signals for each type
coupling_signals = {
    LINEAR:     V_src,
    QUADRATURE: W_src,
    RECTIFIED:  np.maximum(V_src, 0.0),
    SQUARED:    V_src ** 2,
    PAC:        (0.5 + 0.4 * V_src) * W_src,
}

# ---------------------------------------------------------------------------
# Plot window: skip first 2 s (transient), show 8 s of dynamics
# ---------------------------------------------------------------------------
t_s = t_ref / 1000.0
mask = (t_s >= 2.0) & (t_s <= 10.0)
t_plot = t_s[mask]

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
n_rows = 1 + len(COUPLING_ORDER)   # source panel + 5 coupling panels
fig = plt.figure(figsize=(13, 10))
fig.suptitle(
    "Generic2dOscillator — coupling types (2-node, node 0 → node 1 unidirectional)\n"
    "Top: source node signals.  Rows 1-5: coupling signal injected (left axis, shaded)\n"
    "  and target node V response (right axis, solid line).",
    fontsize=10, y=0.995
)

gs = gridspec.GridSpec(n_rows, 1, hspace=0.7,
                       left=0.13, right=0.97, top=0.92, bottom=0.05)

# --- Row 0: source node V and W ---
ax0 = fig.add_subplot(gs[0])
ax0.plot(t_plot, V_src[mask], color="#333333", lw=1.2, label=r"$V_0$ (source V)")
ax0.plot(t_plot, W_src[mask], color="#888888", lw=1.0, ls="--", alpha=0.8,
         label=r"$W_0$ (approx. recovery)")
ax0.set_ylabel("Source\nnode 0", fontsize=9)
ax0.set_title("Source node signals", fontsize=9, pad=2)
ax0.legend(fontsize=8, loc="upper right", framealpha=0.7, ncol=2)
ax0.spines[["top", "right"]].set_visible(False)
ax0.set_xticklabels([])
ax0.set_xlim(t_plot[0], t_plot[-1])

# --- Rows 1-5: coupling signal + target response ---
for row, ct in enumerate(COUPLING_ORDER, start=1):
    t_r, V_r = results[ct]
    t_r_s = t_r / 1000.0
    mask_r = (t_r_s >= 2.0) & (t_r_s <= 10.0)
    t_r_plot = t_r_s[mask_r]

    csig = coupling_signals[ct][mask]
    target_V = V_r[mask_r, 1]

    ax = fig.add_subplot(gs[row])
    color = COLORS[ct]

    # Coupling signal (left axis, shaded)
    ax.fill_between(t_plot, csig, alpha=0.25, color=color)
    ax.plot(t_plot, csig, color=color, lw=1.0, alpha=0.7, label="coupling signal")

    # Target response (right axis)
    ax2 = ax.twinx()
    ax2.plot(t_r_plot, target_V, color=color, lw=1.4, ls="-",
             label="node 1 V (target)")
    ax2.set_ylabel("target V", fontsize=7, color=color)
    ax2.tick_params(axis="y", labelsize=7, labelcolor=color)
    ax2.spines[["top"]].set_visible(False)

    ax.set_ylabel(DESCRIPTIONS[ct], fontsize=8.5, rotation=0,
                  labelpad=100, va="center", ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.tick_params(labelsize=8)

    if row < n_rows - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (s)", fontsize=9)

    # Combined legend
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7,
              loc="upper right", framealpha=0.6)

outpath = os.path.join(os.path.dirname(__file__), "coupling_types_timeseries.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nSaved → {outpath}")
plt.show()
