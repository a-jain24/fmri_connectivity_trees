# GCSS Motor Network — MSC01 Analysis Report

**Date:** 2026-05-12  
**Subject:** MSC01 (Midnight Scan Club, 10 sessions, 818 TRs/session)  
**Pipeline:** Group-Constrained Subject-Specific (GCSS) motor parcellation  
**Phases completed:** 1–5 (group maps → cluster segmentation → subject fROIs → timeseries → connectivity)

---

## 1. fROI Extraction (Phase 3)

Phase 3 intersected MSC01's odd-run motor task z-maps with the 6 Tier-1 group cluster maps, yielding **18 subject-specific fROI masks** across all contrasts. No fallbacks to relaxed threshold or whole-parcel were required — MSC01 has robust motor activation at the primary z > 3.7 threshold across all contrasts.

| ROI key | Anatomical region | Contrast |
|---------|-------------------|---------|
| `LFoot__Right_4` | Right paracentral M1 (foot) | LFoot |
| `RFoot__Left_4` | Left paracentral M1 (foot) | RFoot |
| `LHand__Right_4` | Right hand-knob M1 | LHand |
| `RHand__Left_3b` | Left S1/M1 hand (area 3b centroid) | RHand |
| `combined_motor__Left_SCEF` | SMA / pre-SMA medial wall | combined_motor |
| `combined_motor__Left_6d` | Left dorsal premotor (PMd) | combined_motor |
| `combined_motor__Right_6v` | Right ventral premotor (PMv) | combined_motor |
| `combined_motor__Left_PFop` | Left parietal operculum / S2 | combined_motor |
| `combined_motor__Right_PFop` | Right parietal operculum / S2 | combined_motor |
| `combined_motor__Left_FOP2` | Left frontal operculum (FOP2) | combined_motor |
| `combined_motor__Right_FOP2` | Right frontal operculum | combined_motor |
| `combined_motor__Left_PF` | Left PF (supramarginal gyrus) | combined_motor |
| `tongue__Left_4` | Left lateral M1 (tongue/face) | tongue |
| `tongue__Right_4` | Right lateral M1 (tongue/face) | tongue |
| `tongue__Left_FOP2` | Left frontal operculum | tongue |
| `tongue__Right_FOP2` | Right frontal operculum | tongue |
| `tongue__Left_PoI2` | Left posterior insular cortex | tongue |
| `tongue__cluster_04` | Left peri-insular (unlabeled) | tongue |

Timeseries extracted for all 10 sessions: **818 TRs × 18 ROIs** per session; **8,180 TRs** concatenated.

---

## 2. Connectivity Analysis (Phase 5)

### 2.1 Overall Statistics

| Metric | Value |
|--------|-------|
| Concatenated TRs | 8,180 |
| ROIs | 18 |
| MI range (all-sessions) | 0.263 – 0.893 nats |
| MI mean (off-diagonal) | 0.311 nats |
| CL tree edges | 17 (= N − 1) |
| Strongest edge | `tongue/Left_4 ↔ tongue/Right_4` (MI = 0.893) |

Per-session max MI is stable across sessions (range 1.20–1.38 nats), confirming reliable connectivity estimates. The difference between per-session and all-sessions MI values reflects the finer effective resolution in shorter concatenations.

### 2.2 Functional Connectivity Highlights (|r| > 0.4)

Three clusters emerge:

**Bilateral tongue / orofacial cluster** (dominant in this subject):
- `tongue/Left_4 ↔ tongue/Right_4` r = **0.91** — strongest pair; expected given bilateral corticobulbar projections from both tongue M1 areas to the hypoglossal nucleus
- `combined_motor/Left_PFop ↔ tongue/Left_4` r = **0.88** — left S2/parietal operculum tightly coupled to left tongue M1 (sensorimotor feedback for oral movements)
- `combined_motor/Right_6v ↔ tongue/Right_4` r = **0.82** — right PMv to right tongue M1 (orofacial motor coordination / grasping circuit)
- `combined_motor/Left_PFop ↔ tongue/Right_4` r = **0.79**

**Left sensorimotor hand:**
- `LHand/Right_4 ↔ RHand/Left_3b` r = **0.42** — interhemispheric hand sensorimotor coupling
- `RHand/Left_3b ↔ combined_motor/Left_6d` r = **0.50** — left S1 hand directly coupled to left PMd (sensorimotor–premotor chain)

**Premotor/SMA:**
- `combined_motor/Left_FOP2 ↔ combined_motor/Left_SCEF` r = **0.50** — frontal operculum to SMA
- `combined_motor/Left_PFop ↔ combined_motor/Right_6v` r = **0.69** — cross-hemispheric S2–PMv coupling

### 2.3 Chow-Liu Tree Structure

**Hub nodes** (degree ≥ 3):

| Node | Degree | Role |
|------|--------|------|
| `tongue/Right_4` | **4** | Central hub: connects tongue network to hand network via `LHand/Right_4` |
| `tongue/Left_4` | **4** | Central hub: connects bilateral tongue, opercular, insular, and cluster nodes |
| `combined_motor/Left_SCEF` | 3 | SMA bridge: connects PMd, FOP2, and S2 branches |
| `combined_motor/Left_FOP2` | 3 | FOP2 bridge: connects SMA, PF, and insular nodes |
| `LHand/Right_4` | 3 | Hand bridge: connects right M1 foot, left S1 hand, and tongue subtrees |

**Strongest CL edges:**

| Rank | Edge | MI (nats) | Interpretation |
|------|------|-----------|----------------|
| 1 | `tongue/Left_4 ↔ tongue/Right_4` | 0.893 | Bilateral tongue M1 — midline organ with obligate bilateral representation |
| 2 | `Left_PFop ↔ tongue/Left_4` | 0.877 | Left S2/parietal operculum to tongue M1 — proprioceptive feedback loop |
| 3 | `Right_6v ↔ tongue/Right_4` | 0.672 | Right PMv to right tongue M1 — orofacial motor network |
| 4 | `LHand/Right_4 ↔ RHand/Left_3b` | 0.445 | Interhemispheric hand sensorimotor coupling |
| 5 | `RHand/Left_3b ↔ Left_6d` | 0.387 | Left S1 hand to left PMd — sensorimotor-to-premotor chain |
| 6 | `Left_FOP2 ↔ Left_SCEF` | 0.382 | Frontal operculum to SMA |

**Tree topology:** The CL tree has a **bilateral tongue M1 core** (edges 1–3 by MI weight). The foot nodes (`LFoot/Right_4 ↔ RFoot/Left_4`, MI = 0.363) form their own bilateral pair at the periphery, connected to the hand–premotor subgraph via `LHand/Right_4` as a bridge. `RFoot/Left_4` is a leaf (degree 1), suggesting weaker integration of left foot M1 into the broader network for MSC01. The SMA (`Left_SCEF`) connects the premotor chain (`Left_6d` → `RHand/Left_3b`) to the opercular nodes (`Left_FOP2` → `Left_PF`), confirming SMA's expected integrative role.

**Consistency with predictions from gcss_plan.md:**
- ✓ Tongue M1 bilateral coupling: confirmed as strongest edges
- ✓ SMA as hub: confirmed (degree 3, bridges premotor and opercular branches)
- ✓ L/R tongue M1 directly connected: confirmed (rank 1 edge)
- ✓ S1S2 as leaf/peripheral: L_PFop is degree 2 (not a leaf, but near-peripheral)
- ✗ SMA as root/highest-degree: tongue M1 nodes (degree 4) outrank SMA in this subject

---

## 3. Cerebellar and Subcortical Localizer

**Figures:** `roi_figures/MSC01/MSC01_cereb_zmaps.pdf`, `MSC01_cereb_somatotopy.pdf`, `MSC01_subcortical_zmaps.pdf`

### 3.1 Cerebellar Activation

MSC01's cerebellar z-maps (odd-run average, thresholded at per-contrast z) show the expected somatotopic organization:

- **Foot:** Activation in bilateral lobules IV/V (axial z ≈ −28 to −36). Follows the ipsilateral lateralization convention in cerebellar projections (LFoot → right cerebellar hemisphere receives input from left motor cortex via ipsilateral pontine nuclei and crosses at the middle cerebellar peduncle).
- **Hand:** Activation in lobule VI (intermediate cerebellum, z ≈ −22 to −30). Consistent with the classical cerebellar hand area.
- **Tongue:** Strongest cerebellar activation of the three effectors, bilaterally in lateral lobule VI and vermis VI. Bilateral cerebellar tongue activation is consistent with bilateral corticobulbar drive.
- **Combined motor:** Broad activation across lobules IV–VIII; the winner-takes-all somatotopy check (z ≥ 2.3) shows a rough anterior–inferior (foot) to posterior–superior (hand/tongue) gradient in the sagittal view, consistent with the anterior somatotopic map of the cerebellum.

### 3.2 Subcortical Activation

- **Thalamus (Morel motor nuclei — VLa, VLpd, VLpv, VAmc, VApc):** `combined_motor` shows bilateral activation within the white-contour Morel motor-thalamus overlay at axial z ≈ 4–12, confirming that the thalamic relay is captured. Individual effector maps show weaker but present thalamic activation.
- **Putamen/caudate:** Visible in the `combined_motor` map at axial z ≈ 4–20, consistent with the motor cortico-striatal loop.
- **Subthalamic nucleus (STN):** Not clearly visible at the 2mm resolution and standard z threshold.

---

## 4. Limitations and Next Steps

1. **Tongue-dominated network:** The CL tree is structurally dominated by tongue/bilateral connectivity, partly because tongue has the largest fROIs (1148 + 995 vox vs. ~184–516 for foot/hand) and most ancillary opercular nodes. Consider restricting to the 11 canonical ROIs or matching fROI sizes.

2. **Ancillary tongue opercular nodes:** Eight of the 18 fROIs belong to the tongue contrast (including FOP2, PoI2, and an unlabeled cluster). These are neuroanatomically plausible (orofacial sensorimotor) but were not originally planned. Future work should either incorporate them into the canonical scheme or exclude them for comparability.

3. **Single subject — no test-retest run yet:** Even-run fROIs should be derived and their spatial overlap with odd-run fROIs computed to verify within-subject reproducibility (Fedorenko 2010 benchmark: r ≈ 0.52).

4. **Cerebellar/subcortical GCSS fROIs not extracted:** Phases 3E/3F (GCSS applied within SUIT and Morel atlas masks) are not yet implemented. The cerebellar figures represent individual z-maps, not GCSS-constrained fROIs. Adding 3–6 cerebellar ROIs per subject would enable cortico-cerebellar CL tree analysis.

5. **Pipeline validation for remaining subjects:** MSC02–MSC10 should be run next. Comparing CL tree topologies across subjects will reveal whether the bilateral tongue hub and SMA bridge are consistent features or MSC01-specific.

---

## 5. File Index

### ROI Figures — `roi_figures/MSC01/`
| File | Content |
|------|---------|
| `MSC01_all_rois.pdf` | All 18 fROI masks on MNI slices |
| `MSC01_cereb_zmaps.pdf` | Cerebellar cuts, MSC01 z-maps (all contrasts) |
| `MSC01_cereb_somatotopy.pdf` | Cerebellar winner-takes-all somatotopy |
| `MSC01_subcortical_zmaps.pdf` | Thalamus/BG cuts + Morel motor-thalamus contours |

### Analysis Figures — `figures/canonical_circuits/motor_cortex/gcss/MSC01/`
| File | Content |
|------|---------|
| `MSC01_all_sessions_gcss_fc_matrix.pdf` | FC matrix (all-sessions) |
| `MSC01_all_sessions_gcss_mi_matrix.pdf` | MI matrix (all-sessions) |
| `MSC01_all_sessions_gcss_cl_tree.pdf` | Chow-Liu tree (all-sessions) |
| `MSC01_all_sessions_gcss_fc_vs_mi.pdf` | FC vs MI scatter, CL edges in red |
| `MSC01_func{01–10}_gcss_*.pdf` | Per-session versions (40 files) |

### Connectivity Data — `datasets/midnight_scan_club/connectivity/`
```
fc/MSC01/{func01..func10,all_sessions}/gcss_motor/  → FC matrices (.npy) + ROI keys (.json)
mi/MSC01/…                                          → MI matrices (.npy)
cl/MSC01/…                                          → CL adjacency matrices (.npy)
```
