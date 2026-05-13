# GCSS Motor Network — Publication Figure Plan

**Last updated:** 2026-05-12  
**Based on:** `report.md` (group analysis, MSC01–MSC10)

The figures below are ordered as they would appear in a methods/results paper. Each entry specifies the message the figure must land, the data it requires, the visual form, and the implementation notes needed to produce it.

---

## Figure 1 — GCSS Pipeline Overview (Methods schematic)

**Message:** How subject-specific motor fROIs are derived — three-step procedure from group probability map through cluster segmentation to subject intersection.

**Form:** Three-panel horizontal schematic.
- Panel A: Axial slice montage showing the group probability map for one representative contrast (`combined_motor`), coloured by overlap probability (0.6–1.0), with the retained cluster boundaries overlaid as white contours.
- Panel B: Same slices showing the segmented cluster map (each cluster a distinct colour), labelled with semantic ROI name.
- Panel C: Same slices for one example subject (MSC01) showing their z-map thresholded at z > 3.7 (hot colourmap), then the resulting subject-specific fROI mask (filled outline) constrained within the group cluster boundary.

**Data:**
- Group probability map: `analysis/canonical_circuits/motor_cortex/gcss/group_maps/combined_motor_prob.nii.gz`
- Cluster map: `analysis/canonical_circuits/motor_cortex/gcss/cluster_maps/combined_motor_clusters.nii.gz`
- Subject z-map: `derivatives/firstLevel/…/sub-MSC01_task-motor_contrast-combined_motor_zmap.nii.gz` (odd-run average)
- Subject fROI masks: `froi_masks/MSC01/MSC01_gcss_{SMA,L_PMd,R_PMv,L_S1S2,R_S1S2}_mask.nii.gz`
- MNI template for background.

**Slices:** Axial z = 28, 42, 56, 66 (captures tongue/hand/foot/SMA across four cuts). Add one coronal cut (y = −12) to show the tongue/S2 cluster.

**Colour scheme:** Probability map = sequential (white→red); cluster map = `ALL_ROI_DEFS` colours; z-map = hot; fROI fill = semi-transparent match to cluster colour, black outline.

**Implementation:** Extend `gcss_viz_group_maps.py` — new function `fig_pipeline_schematic(subject='MSC01', contrast='combined_motor')`. Use `nilearn.plotting.plot_stat_map` with `cut_coords` and manual axes via `display.axes_img_`.

---

## Figure 2 — Group fROI Atlas (11 ROIs on MNI brain)

**Message:** Spatial layout of the 11 group-level cluster masks on the MNI brain — where each canonical motor ROI sits anatomically.

**Form:** Four-row brain figure.
- Row 1: Glass brain (lyrz views), all 11 ROIs, each coloured by `ALL_ROI_DEFS` colour scheme.
- Row 2: Axial montage (z = 24, 32, 44, 56, 66) — captures tongue/S2, hand, foot, SMA/PMd.
- Row 3: Coronal montage (y = −22, −12, −2, 8) — captures foot/hand M1 and S2/opercular.
- Row 4: Legend (11 coloured patches, ROI name + anatomical descriptor).

**Data:** Group cluster maps (binary masks derived from `*_clusters.nii.gz`). **Not** subject masks — this is the group template.

**Notes:** This is already partially implemented as `fig_all_rois_labeled()` in `gcss_viz_group_maps.py`. That function needs minor polish: increase label font size, move legend to its own row with two-column layout, add MNI coordinate annotations for each ROI centroid.

---

## Figure 3 — Subject fROI Mosaic (variability across individuals)

**Message:** The 11 fROI masks are spatially consistent across subjects but show individual variability in extent and position.

**Form:** 10-column × 11-row panel.
- Each column = one subject (MSC01–MSC10).
- Each row = one ROI (same order and colour as Fig 2).
- Each cell = a single-slice thumbnail of that subject's fROI mask on the MNI brain (one fixed axial cut per ROI, chosen to bisect the group centroid).

**Optimal slice per ROI (from group centroids):**

| ROI | Axial z | Rationale |
|-----|---------|-----------|
| `R_M1_foot` / `L_M1_foot` | z = 68 | Paracentral lobule |
| `R_M1_hand` / `L_M1_hand` | z = 56 | Hand knob |
| `R_M1_tongue` / `L_M1_tongue` | z = 28 | Lateral face M1 |
| `SMA` | z = 62 | Medial SMA/pre-SMA |
| `L_PMd` | z = 58 | Dorsal premotor |
| `R_PMv` | z = 34 | Ventral premotor |
| `L_S1S2` / `R_S1S2` | z = 30 | Parietal operculum |

**Data:** `froi_masks/{sub}/{sub}_gcss_{sem_name}_mask.nii.gz` for all 10 subjects × 11 ROIs.

**Implementation:** New function `fig_subject_froi_mosaic()` in `gcss_viz_group_maps.py`. 110 small axes (~1.5 × 1.5 inches each). Use `nilearn.plotting.plot_roi` with `display_mode='z'`, `cut_coords=[z]`. Add row labels (ROI name, left) and column headers (subject ID, top).

---

## Figure 4 — Group FC and MI Matrices

**Message:** The motor network has a structured connectivity profile: bilateral M1 pairs dominate, with a secondary orofacial–premotor cluster.

**Form:** Two-panel (side by side).
- Left: Group-mean FC matrix (11×11 Pearson r), sorted by somatotopic group (foot → hand → tongue → premotor), coloured RdBu_r (−1 to +1), with ±1 SD error bars not shown in matrix but reported in supplementary table.
- Right: Group-mean MI matrix (same ordering), viridis colourmap.
- Both panels: thin white dividers between somatotopic groups, ROI labels on both axes (rotated 45° on x-axis), colourbar.

**ROI ordering for matrices:** `R_M1_foot, L_M1_foot, R_M1_hand, L_M1_hand, R_M1_tongue, L_M1_tongue, R_PMv, R_S1S2, L_S1S2, L_PMd, SMA` — sorted to place bilateral pairs adjacent and the premotor/SMA block together.

**Data:** `fc_arr.mean(axis=0)` and `mi_arr.mean(axis=0)` from `connectivity/{fc,mi}/MSC{01..10}/all_sessions/gcss_motor/`.

**Implementation:** New script `gcss_group_figures.py`. Function `fig_group_matrices()`. Include CL edge markers: overlay a dot or asterisk at each cell that is a CL edge in ≥ 7/10 subjects.

---

## Figure 5 — CL Tree Edge Frequency (Group Consensus)

**Message:** Three bilateral M1 pairs are universal; the orofacial–S2 and premotor edges are nearly so; more distal premotor connections are individually variable.

**Form:** Single panel — the group consensus CL tree drawn as a weighted graph.
- Nodes: 11 ROIs, positioned to reflect anatomy (foot nodes at top, tongue nodes at bottom-left, SMA/PMd/PMv at right). Use a fixed anatomical layout (not force-directed).
- Edges: Draw all edges that appear in ≥ 1/10 subjects. Edge thickness ∝ CL frequency (1–10). Edge opacity ∝ frequency. Label each edge with its frequency (e.g., "10/10" for the three universal pairs).
- Node colour: `_SEMANTIC_COLORS` (same as all other figures).
- Node size ∝ mean degree.

**Anatomical node positions (approximate, in figure space):**

```
           SMA
          /   \
       L_PMd   R_PMv
         |       |
    L_M1_hand  R_M1_tongue -- L_M1_tongue
         |       |                 |
    R_M1_hand  R_S1S2 ---- L_S1S2
         |
    R_M1_foot -- L_M1_foot
```

**Data:** `edge_freq[i,j]` matrix; `mi_mean[i,j]`; `mean_deg` per node.

**Implementation:** `fig_consensus_tree()` in `gcss_group_figures.py`. Use `networkx` with fixed `pos` dict; `nx.draw_networkx` with width and alpha scaled to frequency. Annotate edges with frequency string. This is the key summary figure for the paper.

---

## Figure 6 — Per-Subject CL Trees (10-panel mosaic)

**Message:** Individual CL trees share the bilateral M1 core but differ in premotor branch structure; L_S1S2 hub is consistent but MSC05/MSC07 show alternative topologies.

**Form:** 2-row × 5-column panel, one CL tree per subject. All trees use the same fixed anatomical node layout as Fig 5. Edge colour encodes whether the edge is canonical (in ≥ 7/10 subjects: dark grey), intermediate (4–6/10: medium grey), or subject-specific (1–3/10: light grey with dashed line). Each tree panel labelled with subject ID and total all-sessions MI mean.

**Data:** Per-subject `cl_arr[si]`; per-subject `mi_arr[si]`; `edge_freq` from group for edge classification.

**Implementation:** `fig_subject_trees()` in `gcss_group_figures.py`. Fixed layout, loop over subjects. Highlight L_S1S2 node with a bold border in subjects where it is a hub (degree ≥ 3).

---

## Figure 7 — Hub Node Degree Distribution

**Message:** L_S1S2 is the dominant hub (mean degree 3.4), followed by R_M1_hand (2.7); all other nodes have mean degree < 2.4 — the tree is not uniformly connected but has clear structural hierarchy.

**Form:** Two-panel.
- Left: Bar chart of mean degree ± SD across 10 subjects, one bar per ROI, sorted descending. Bars coloured by `_SEMANTIC_COLORS`. Horizontal dashed line at degree = 2 (threshold between leaf/peripheral and hub territory). Individual subject degree values overlaid as jittered dots.
- Right: Heatmap (11 ROIs × 10 subjects) showing each ROI's degree per subject. Colour scale 0–4. Row order = mean degree descending (same as bar chart). Column order = subjects sorted by mean network MI (MSC05 → MSC01 → MSC09 → MSC03 → MSC02 → MSC01 → MSC07 → MSC04 → MSC10 → MSC06 → MSC08).

**Data:** `cl_binary[si].sum(axis=1)` per subject and ROI.

**Implementation:** `fig_hub_degrees()` in `gcss_group_figures.py`.

---

## Figure 8 — FC vs MI Scatter with CL Edge Annotation (Group)

**Message:** FC (Pearson r) and MI are strongly correlated, but MI provides additional discriminability — particularly for the CL tree edges, which cluster at high MI values regardless of FC rank.

**Form:** Scatter plot, one point per ROI pair (55 pairs × 10 subjects = 550 points, or just group means as 55 points with error bars).
- Use **group-mean** version (55 points) for clarity; plot individual-subject points as transparent grey in the background.
- Colour CL-canonical edges (≥ 7/10 subjects) in dark red; intermediate edges (4–6/10) in orange; non-CL pairs in grey.
- Label the 6 canonical CL edges by name.
- Add regression line (full dataset) with r and p.

**Data:** `fc_arr`, `mi_arr`, `edge_freq`.

**Implementation:** `fig_group_fc_vs_mi()` in `gcss_group_figures.py`.

---

## Figure 9 — CL Tree Drawn on the Brain (Connectome-Style)

**Message:** The Chow-Liu tree edges are not abstract graph relationships — they connect specific anatomical locations. Rendering edges as curves on a glass brain makes the spatial structure of the motor network immediately legible.

**Form:** Single figure, three views of a glass brain (left lateral, superior axial, posterior coronal), with all 11 ROI centroids plotted as filled circles and the 6 canonical CL edges (≥ 7/10 subjects) drawn as arcs between them. Use a representative single subject (MSC01, or whichever has the cleanest all-sessions tree) rather than the group average, so every edge shown is a real tree edge rather than a consensus construct.

- **Background:** MNI152 glass brain via `nilearn.plotting.plot_connectome` or `plot_markers` + manual arc drawing.
- **Nodes:** Filled circles at each ROI's MNI centroid. Size ∝ mean degree (same scale as Fig 5). Colour = `_SEMANTIC_COLORS`. Labelled with ROI name (offset to avoid overlap).
- **Edges:** Curved arcs (bezier) connecting ROI centroids. Line width ∝ MI weight on that edge. Colour = edge MI mapped to a sequential colourmap (e.g. `plasma`, low→high). Only draw edges present in the subject's actual CL tree. Optionally overlay canonical group edges (≥ 7/10) as a slightly thicker black outline to distinguish universal from subject-specific edges.
- **Colourbar:** MI (nats) scale for edge colour.

**Preferred implementation path:**  
Use `nilearn.plotting.plot_connectome(adjacency_matrix, node_coords, node_color, node_size, edge_cmap, edge_vmin, edge_vmax, display_mode='lzr', axes=ax)` where `adjacency_matrix` is the subject's CL adjacency with MI weights. This handles the 3-view layout and arc rendering automatically.

**Node coordinates (group centroids, MNI mm):**

| ROI | x | y | z |
|-----|---|---|---|
| `R_M1_foot` | 5.8 | −25.0 | 69.7 |
| `L_M1_foot` | −5.8 | −22.8 | 68.9 |
| `R_M1_hand` | 39.5 | −20.9 | 57.7 |
| `L_M1_hand` | −39.4 | −23.3 | 57.4 |
| `R_M1_tongue` | 57.6 | −5.1 | 28.6 |
| `L_M1_tongue` | −57.5 | −8.7 | 30.0 |
| `SMA` | −1.8 | −4.6 | 61.7 |
| `L_PMd` | −45.4 | −8.8 | 58.8 |
| `R_PMv` | 57.1 | 1.5 | 35.1 |
| `L_S1S2` | −58.8 | −11.5 | 32.4 |
| `R_S1S2` | 58.0 | −13.9 | 27.1 |

**Data:** Subject CL adjacency (`cl_arr[si]`) with MI weights (`mi_arr[si]` masked to CL edges); ROI centroids table above; MNI template.

**Implementation:** `fig_cl_tree_on_brain(subject='MSC01')` in `gcss_group_figures.py`. To show the group picture instead of a single subject, call with the mean CL adjacency thresholded at the canonical edge set (≥ 7/10). To generate all subjects, add a loop over `SUBJECTS` in addition to the existing two variants.

**Variants to generate:**
1. `fig9_cl_tree_on_brain_MSC01.pdf` through `fig9_cl_tree_on_brain_MSC10.pdf` — one per subject, all CL edges MI-weighted
2. `fig9_cl_tree_on_brain_group_canonical.pdf` — canonical edges only (≥ 7/10), MI-weighted mean

**Per-subject loop (add to function):**
```python
for si, sub in enumerate(SUBJECTS):
    adj_mi = _make_adj(cl_arr[si] * mi_arr[si])
    label  = f'{sub} — GCSS Motor CL Tree (all sessions)'
    # … same plot_connectome call, save as fig9_cl_tree_on_brain_{sub}
```

Node size should reflect *that subject's* degree rather than the group mean for subject-specific panels:
```python
deg_s      = (cl_arr[si] > 0).sum(axis=1)
node_size_s = 20 + 80 * (deg_s - deg_s.min()) / max(deg_s.max() - deg_s.min(), 1e-3)
```

---

## Figure 10 — Aesthetic CL Tree (Presentation / Poster Version)

**Message:** Same content as Fig 5 (group consensus tree) but optimised for visual impact — large nodes, readable labels, and clean layout suitable for a poster, talk slide, or graphical abstract.

**Form:** Single panel, ~10 × 8 inches. Fixed anatomical layout (same node positions as Fig 5), but all visual elements scaled up:

- **Nodes:** Diameter 60–80 pt (vs. ~30 pt in Fig 5). Filled with `_SEMANTIC_COLORS`, white label centred inside the node. Font: Arial Bold, 11 pt minimum.
- **Edges:** Width ∝ CL frequency (range: 2 pt for 1/10 → 10 pt for 10/10). Only draw edges ≥ 4/10 to reduce clutter (omit the 1–3/10 idiosyncratic edges). Edge colour = frequency mapped to a single-hue ramp (e.g. dark grey for 10/10 → light grey for 4/10).
- **Edge labels:** Frequency fraction ("10/10", "9/10", etc.) in small text (9 pt) centred on the edge midpoint, on a white background patch to ensure legibility.
- **No axis frame:** `ax.axis('off')`. Maximise figure area for the graph.
- **Subtitle:** Small italic caption below the figure: *"Edge thickness = fraction of subjects (N=10) in which the edge appears in the Chow-Liu tree. Node colour = anatomical region."*

**Style deviations from the standard constants** (these are intentional for the poster version):

```python
NODE_RADIUS      = 0.08     # in normalised axes units
NODE_FONT_SIZE   = 12       # pt, bold
EDGE_LABEL_SIZE  = 9        # pt
EDGE_WIDTH_SCALE = (2, 10)  # (min, max) pt mapped to freq (1, 10)
FIGURE_SIZE      = (11, 9)  # inches
BACKGROUND_COLOR = '#f8f8f8'  # off-white
```

**Anatomical node layout** (same x, y positions as Fig 5 — reproduced here for convenience):

```
y=0.90   SMA
y=0.75   L_PMd         R_PMv
y=0.60   L_M1_hand     R_M1_tongue   L_M1_tongue
y=0.45   R_M1_hand     R_S1S2        L_S1S2
y=0.20   R_M1_foot     L_M1_foot
         x=0.25        x=0.55         x=0.80
```

**Data:** Same as Fig 5 — `edge_freq`, `mi_mean`, `mean_deg`.

**Implementation:** `fig_aesthetic_tree()` in `gcss_group_figures.py`. Use `matplotlib.patches.Circle` for nodes and `matplotlib.patches.FancyArrowPatch` (connectionstyle `arc3,rad=0.1`) for curved edges, rather than `networkx.draw_networkx`, to allow precise visual control. Draw nodes last so they sit on top of edge lines.

**Output:** `pub_figures/fig10_aesthetic_cl_tree.pdf` (vector, for scaling) + `.png` at 300 DPI.

**Per-subject variants:** Also generate one aesthetic tree per subject showing that subject's actual CL edges. Same fixed node layout and colours; edges are those present in the subject's tree. Colour each edge by its group-frequency category (canonical ≥ 7/10 = `#222222`; intermediate 4–6/10 = `#777777`; idiosyncratic 1–3/10 = `#bbbbbb`) so the viewer can see which of the subject's edges are typical vs. rare. Replace frequency label with the MI value for that edge (`f"MI={mi:.3f}"`). Node size ∝ that subject's degree.

```python
# In fig_aesthetic_tree(), add parameter: subject=None
# If subject is not None, use per-subject cl_bin[si] + mi_arr[si] instead of ef + mi_mean.
# Edge colour: group-frequency category (same palette as Fig 6 edges).
# Edge label: MI weight formatted as "MI=0.XXX".
# Output: fig10_aesthetic_cl_tree_{subject}.pdf/png
```

**Outputs (per-subject loop):** `fig10_aesthetic_cl_tree_MSC01.pdf` → `fig10_aesthetic_cl_tree_MSC10.pdf`

---

## Figure 11 — Per-Subject CL Networks on the Brain (Brain Mosaic)

**Message:** The same individual variability shown in the abstract graph mosaic (Fig 6) is visible anatomically — most subjects share the bilateral hand/foot/tongue M1 edges, with idiosyncratic variation in premotor connectivity.

**Form:** 2-row × 5-column mosaic, one brain connectome panel per subject. Each panel shows that subject's CL tree edges as arcs on a glass brain, using the same MNI centroid coordinates and colour scheme as Fig 9. Matches the layout of Fig 6 exactly, but replaces the abstract graph with the brain view.

- **Each panel:** Axial montage (`display_mode='z'`, three cuts at z = 28, 55, 68) from `nilearn.plotting.plot_connectome`. Three cuts capture tongue / hand / foot somatotopy and fit in a panel ~4.2 × 1.8 inches.
- **Edges:** MI-weighted CL adjacency (same as individual Fig 9 panels). Plasma colourmap; `edge_vmin=0.1`; `edge_vmax` = 95th percentile of non-zero CL MI values across all subjects (shared scale).
- **Nodes:** Filled circles, `SEMANTIC_COLORS`, size ∝ that subject's degree.
- **Panel label:** Subject ID + `MĪ=X.XX`, top-left of each panel (same format as Fig 6).
- **Shared colourbar:** One colourbar for edge MI placed at the right margin; individual panels use `colorbar=False`.

**Implementation:** `fig_subject_brain_mosaic()` in `gcss_group_figures.py` (new function, Fig 11).

Because `plot_connectome` manages its own figure layout, compositing 10 panels requires the raster-stitch approach from the Fig 1 fix — render each panel into a `BytesIO` buffer, then arrange images in a plain matplotlib grid:

```python
import io

def _connectome_to_img(adj, node_coords, node_color, node_size, mi_vmax):
    disp = plotting.plot_connectome(
        adj, node_coords,
        node_color=node_color, node_size=node_size,
        edge_cmap='plasma', edge_vmin=0.1, edge_vmax=mi_vmax,
        display_mode='z', cut_coords=[28, 55, 68],
        colorbar=False,
    )
    buf = io.BytesIO()
    disp.frame_axes.figure.savefig(buf, format='png', dpi=150,
                                   bbox_inches='tight', facecolor='black')
    plt.close(disp.frame_axes.figure)
    buf.seek(0)
    return plt.imread(buf)


def fig_subject_brain_mosaic(fc_arr, mi_arr, cl_arr, keys, d):
    node_coords = np.array([NODE_COORDS_MNI[k] for k in keys])
    node_color  = [SEMANTIC_COLORS[k] for k in keys]
    mi_vmax     = float(np.percentile(mi_arr[cl_arr > 0], 95))

    fig, axes = plt.subplots(2, 5, figsize=(13.6, 5.6))
    axes = axes.ravel()

    for si, (sub, ax) in enumerate(zip(SUBJECTS, axes)):
        deg_s       = (cl_arr[si] > 0).sum(axis=1).astype(float)
        node_size_s = 20 + 80 * (deg_s - deg_s.min()) / max(
                          deg_s.max() - deg_s.min(), 1e-3)
        adj_mi = cl_arr[si] * mi_arr[si]
        np.fill_diagonal(adj_mi, 0)
        img = _connectome_to_img(adj_mi, node_coords, node_color,
                                  node_size_s.tolist(), mi_vmax)
        ax.imshow(img)
        ax.axis('off')
        n = len(keys)
        mi_mean_s = mi_arr[si][np.triu_indices(n, k=1)].mean()
        ax.set_title(f'{sub}\n(MĪ={mi_mean_s:.2f})', fontsize=7.5, pad=2)

    sm = plt.cm.ScalarMappable(
        cmap='plasma', norm=matplotlib.colors.Normalize(vmin=0.1, vmax=mi_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.01,
                        label='MI (nats)')
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle('GCSS Motor Network — Per-Subject CL Trees on the Brain (all-sessions)',
                 fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig11_subject_brain_mosaic')
```

**Output:** `pub_figures/fig11_subject_brain_mosaic.{pdf,png}`

---

## Table 1 — Group-Mean Connectivity for All ROI Pairs

**Message:** Full numerical reference for the connectivity profile.

**Form:** Supplementary table (11×11 lower triangle), cells showing `FC (MI)` as `r = X.XX (MI = X.XX nats)`, with CL frequency shown in a separate column. Rows/columns sorted in the same anatomical order as Fig 4.

**Implementation:** `pandas.DataFrame` → LaTeX `longtable` or CSV. Export both.

---

## Table 2 — Per-Subject Summary Statistics

**Message:** Subject-level overview for reproducibility.

**Columns:** Subject | TRs (all sessions) | ROI voxel counts (mean ± SD across 11 ROIs) | MI mean | MI max | Strongest pair | Hub nodes (degree ≥ 3) | CL tree canonical edges (n present out of 6)

**Implementation:** Compile from `roi_metadata.json` (voxel counts) and from the per-subject `mi_arr`, `cl_binary` computations already done.

---

---

## Rendering Issues and Fixes (v1 run — 2026-05-12)

### Fig 1 — Only Panel C renders (subfigures incompatible with nilearn)

**Problem:** `fig.subfigures()` creates matplotlib `SubFigure` objects. Nilearn's `plot_stat_map` / `plot_roi` accept a `figure` keyword but internally call `plt.figure(figure, ...)` — if given a `SubFigure` rather than a real `Figure`, nilearn either ignores it or spawns a fresh figure. The upshot is that only the last nilearn call (Panel C, the MSC01 z-map) is written into the saved file; Panels A and B are lost or overwritten.

**Fix:** Render each panel into a separate in-memory PNG buffer via `io.BytesIO`, read the buffers back with `plt.imread`, and composite them with `ax.imshow` in a plain three-row figure. Each panel is a self-contained call to nilearn with no shared figure object:

```python
import io

def _nilearn_panel_to_img(disp):
    """Rasterise a nilearn display to an RGBA array."""
    buf = io.BytesIO()
    disp.frame_axes.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(disp.frame_axes.figure)
    buf.seek(0)
    return plt.imread(buf)

# Then in fig_pipeline_schematic():
img_a = _nilearn_panel_to_img(disp_a)   # probability map
img_b = _nilearn_panel_to_img(disp_b)   # cluster/ROI map
img_c = _nilearn_panel_to_img(disp_c)   # subject z-map + fROI outlines

fig, axes = plt.subplots(3, 1, figsize=(13.6, 9.6))
for ax, img, tag in zip(axes, [img_a, img_b, img_c], ['A', 'B', 'C']):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.01, 0.97, tag, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', color='white')
```

---

### Fig 5 & 6 — Node labels overflow circles

**Problem:** `nx.draw_networkx_nodes` maps `node_size` to the *area* of a circle in pt². At `node_size=200` the circle diameter is approximately 16 pt. The full ROI name "L_M1_tongue" at 5.5 pt font is ~33 pt wide — over twice the circle diameter, so the text extends far beyond the node boundary and is clipped or unreadably overlaps edges. In Fig 6 the problem is worse: scatter markers with `s=60` produce ~8 pt dots that cannot contain any label.

**Fix (Fig 5):** Replace `nx.draw_networkx_nodes` + `nx.draw_networkx_labels` with `matplotlib.patches.Circle` + `ax.text`, using a fixed radius in *axes-coordinate* units (not networkx point units). This is exactly the approach already used in Fig 10. Radius `0.065` in axes units (~55 pt on an 8.5-inch figure at 150 DPI) comfortably contains the two-line `_SHORT` labels at 8–9 pt:

```python
NODE_R = 0.065   # axes-coordinate radius, same as Fig 10

for ki, roi in enumerate(keys):
    x, y = _TREE_POS[roi]
    circle = Circle((x, y), radius=NODE_R,
                    color=SEMANTIC_COLORS[roi], zorder=5,
                    ec='white', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, _SHORT[roi],
            ha='center', va='center',
            fontsize=8, fontweight='bold', color='white',
            zorder=6, linespacing=1.2)
```

Remove all `nx.draw_networkx_*` calls from `fig_consensus_tree()`; draw edges manually as `ax.plot(...)` between the `_TREE_POS` coordinates (already done for edges — only the node drawing needs changing).

**Fix (Fig 6):** Drop in-node text entirely. The 10-panel mosaic (each subplot ~2.7 × 2.8 inches) has no room for readable per-node text. Replace with a small shared ROI-colour inset legend appended to the figure after the loop:

```python
# Replace ax.text(…) calls inside the per-subject loop with nothing.
# After the 10 subplots, add a colour legend:
patches_roi = [mpatches.Patch(color=SEMANTIC_COLORS[r], label=_SHORT[r].replace('\n', ' '))
               for r in keys]
fig.legend(handles=patches_roi, loc='lower center', ncol=6,
           fontsize=6.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.06),
           title='ROI', title_fontsize=7)
```

Increase node scatter size to `s=120 + 100 * min(deg_s[ki], 4) / 4` so coloured dots are legible without labels.

---

### Fig 9 — ROI labels invisible (white on white background)

**Problem:** The annotation code sets `color='white'` for ROI name labels, but nilearn's glass brain uses a light grey / white background. White text on white is invisible. In addition, the annotation call targets `disp.axes['z'].ax` using MNI millimetre coordinates — nilearn's internal axes use data-space transforms that don't map 1:1 to screen position without the display's own transform applied, so even with a visible colour the labels land in the wrong place.

**Fix:** Remove the in-figure annotation block entirely (the `try: axial_ax.annotate(…)` section). Instead, add a coloured `matplotlib.patches` legend below the brain panels. Nilearn's `plot_connectome` already colour-codes nodes; the legend provides the name key:

```python
# After both variants are saved, or inside the variant loop after plt.subplots():
patches = [mpatches.Patch(color=SEMANTIC_COLORS[r],
                           label=_SHORT[r].replace('\n', ' '))
           for r in keys]
fig.legend(handles=patches, loc='lower center', ncol=6,
           fontsize=7, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.04),
           title='ROI (node colour)', title_fontsize=7)
fig.subplots_adjust(bottom=0.12)
```

---

## Implementation Notes

### New script: `gcss_group_figures.py`

Location: `code/functional_connectivity/midnight_scan_club/canonical_circuits/motor_cortex/gcss/`

Functions to implement:
```
fig_pipeline_schematic(subject, contrast)   → Fig 1
fig_group_matrices()                         → Fig 4
fig_consensus_tree()                         → Fig 5
fig_subject_trees()                          → Fig 6
fig_hub_degrees()                            → Fig 7
fig_group_fc_vs_mi()                         → Fig 8
fig_cl_tree_on_brain(subject)                → Fig 9  (group + per-subject loop MSC01–MSC10)
fig_aesthetic_tree(subject=None)             → Fig 10 (group + per-subject loop MSC01–MSC10)
fig_subject_brain_mosaic()                   → Fig 11
table_pairwise_connectivity()                → Table 1
table_subject_summary()                      → Table 2
```

**CLI choices update:** Add `11` to `--figs` argument choices.  
**Per-subject flags:** `--all-subjects` (generate per-subject Fig 9 and Fig 10 variants for all 10 subjects, in addition to the group figures).

Figures 2 and 3 extend `gcss_viz_group_maps.py` (polish `fig_all_rois_labeled()` and add `fig_subject_froi_mosaic()`).

### Style constants (apply to all figures)

```python
FONT_FAMILY  = 'Arial'
FONT_SIZE_LABEL  = 8      # axis labels, tick labels
FONT_SIZE_TITLE  = 9      # panel titles
FONT_SIZE_ANNOT  = 7      # in-figure annotations
FIGURE_DPI   = 300
PANEL_WIDTH  = 3.3        # inches (single column = 3.3, double = 6.8)
```

Use `matplotlib.rcParams` to set globally at the top of each figure script.

### Output directory

All publication figures → `figures/canonical_circuits/motor_cortex/gcss/analysis/pub_figures/`  
All tables → `figures/canonical_circuits/motor_cortex/gcss/analysis/tables/`

### Priority order

For a first submission, the minimum set is:

1. **Fig 2** (group ROI atlas) — spatial grounding for the reader
2. **Fig 4** (group FC + MI matrices) — core connectivity result
3. **Fig 5** (consensus CL tree) — main structural finding
4. **Fig 9** (CL tree on brain) — connects abstract graph to anatomy; the most intuitive result figure
5. **Fig 7** (hub degrees) — quantifies L_S1S2 hub claim
6. **Table 1** (full pairwise table) — numerical support for all claims

**Fig 10** (aesthetic tree) is the graphical abstract / poster version of Fig 5 — produce alongside Fig 5 since they share all the same data and layout logic.  
**Fig 11** (brain mosaic) sits alongside Fig 6 as the anatomical counterpart to the abstract-graph mosaic; both are supplementary.

Figures 1, 3, 6, 8, 11 and Table 2 are supplementary material.  
Per-subject variants of Figs 9 and 10 are supplementary individual-subject panels.
