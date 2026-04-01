"""Reload saved .npz results and regenerate figures without re-simulating."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise_sim import edge_set, jaccard, make_figures

outdir = "code/simulations/output/noise_sim"
noise_levels = [1e-5, 5e-5, 2e-4, 8e-4, 3e-3]

d0 = np.load(os.path.join(outdir, "individual_1_D1e-05.npz"), allow_pickle=True)
C = d0["conn"]
gt_es = edge_set(C)

results = []
for i, D in enumerate(noise_levels):
    fname = os.path.join(outdir, f"individual_{i+1}_D{D:.0e}.npz")
    d = np.load(fname, allow_pickle=True)
    cl_adj     = d["cl_adj"]
    fc_density = d["fc_density"]
    cl_es = edge_set(cl_adj)
    fc_es = edge_set(fc_density)
    results.append({
        "D":             D,
        "cl_adj":        cl_adj,
        "fc_density":    fc_density,
        "cl_edges":      cl_es,
        "fc_edges":      fc_es,
        "cl_gt_jaccard": jaccard(cl_es,  gt_es),
        "fc_gt_jaccard": jaccard(fc_es,  gt_es),
    })

make_figures(C, results, outdir)
