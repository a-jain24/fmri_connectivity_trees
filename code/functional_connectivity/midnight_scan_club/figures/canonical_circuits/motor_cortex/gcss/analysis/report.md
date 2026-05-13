# GCSS Motor Network — Group Analysis Report (MSC01–MSC10)

**Date:** 2026-05-12  
**Dataset:** Midnight Scan Club (10 subjects, 10 sessions each, 818 TRs/session)  
**Pipeline:** Group-Constrained Subject-Specific (GCSS) motor parcellation  
**ROI scheme:** 11 semantic fROIs (bilateral foot/hand/tongue M1 + SMA + L_PMd + R_PMv + bilateral S1S2)  
**Phases completed:** 1–5 (group maps → segmentation → subject fROIs → timeseries → connectivity)

---

## 1. ROI Scheme

Each subject has 11 subject-specific fROIs derived by intersecting their odd-run motor task z-maps (z > 3.7) with 6 group cluster maps. Multiple Glasser-labeled sub-clusters are merged per hemisphere into a single anatomically-named ROI.

| ROI | Anatomy | Source contrast |
|-----|---------|----------------|
| `R_M1_foot` | Right paracentral M1 (foot) | LFoot |
| `L_M1_foot` | Left paracentral M1 (foot) | RFoot |
| `R_M1_hand` | Right hand-knob M1 | LHand |
| `L_M1_hand` | Left S1/M1 hand | RHand |
| `R_M1_tongue` | Right lateral M1 (tongue/face) | tongue |
| `L_M1_tongue` | Left lateral M1 (tongue/face) + opercular | tongue |
| `SMA` | SMA / pre-SMA (medial wall) | combined_motor |
| `L_PMd` | Left dorsal premotor (PMd) | combined_motor |
| `R_PMv` | Right ventral premotor (PMv) | combined_motor |
| `L_S1S2` | Left parietal operculum / S2 | combined_motor |
| `R_S1S2` | Right parietal operculum / S2 | combined_motor |

All 10 subjects yielded exactly 11 fROIs with no fallbacks to relaxed threshold or whole-parcel, confirming robust motor activation across the cohort.

---

## 2. Overall Connectivity Statistics

### 2.1 Per-subject summary (all-sessions concatenated, 8180 TRs)

| Subject | MI mean | MI max | FC mean | FC max | Strongest pair |
|---------|---------|--------|---------|--------|----------------|
| MSC01 | 0.356 | 0.883 | 0.294 | 0.908 | `L_M1_tongue ↔ R_M1_tongue` |
| MSC02 | 0.376 | 0.869 | 0.350 | 0.873 | `L_M1_tongue ↔ L_S1S2` |
| MSC03 | 0.282 | 0.832 | 0.406 | 0.875 | `L_M1_tongue ↔ L_S1S2` |
| MSC04 | 0.438 | 1.138 | 0.460 | 0.934 | `L_M1_tongue ↔ L_S1S2` |
| MSC05 | 0.272 | 0.656 | 0.262 | 0.820 | `L_M1_tongue ↔ R_M1_tongue` |
| MSC06 | 0.567 | 0.888 | 0.250 | 0.846 | `L_M1_tongue ↔ R_M1_tongue` |
| MSC07 | 0.506 | 0.986 | 0.527 | 0.892 | `L_M1_tongue ↔ R_M1_tongue` |
| MSC08 | 0.636 | 1.003 | 0.463 | 0.874 | `L_M1_tongue ↔ L_S1S2` |
| MSC09 | 0.281 | 0.796 | 0.276 | 0.867 | `L_M1_tongue ↔ R_M1_tongue` |
| MSC10 | 0.515 | 1.133 | 0.409 | 0.918 | `L_M1_tongue ↔ R_M1_tongue` |
| **Group mean** | **0.423** | **0.898** | **0.370** | **0.881** | |

MI values span a wide range across subjects (mean 0.27–0.64 nats), reflecting individual differences in motor network coupling strength and/or data quality. The strongest pair in every subject involves `L_M1_tongue`, either with `R_M1_tongue` (6/10 subjects) or `L_S1S2` (4/10 subjects).

---

## 3. Group-Level Pairwise Connectivity

### 3.1 Top pairs by group-mean MI

| Rank | Pair | MI (mean ± SD) | FC (mean ± SD) | CL frequency |
|------|------|----------------|----------------|--------------|
| 1 | `L_M1_tongue ↔ R_M1_tongue` | 0.876 ± 0.131 | 0.871 ± 0.031 | **10/10** |
| 2 | `L_M1_tongue ↔ L_S1S2` | 0.759 ± 0.250 | 0.769 ± 0.148 | 9/10 |
| 3 | `L_M1_hand ↔ R_M1_hand` | 0.608 ± 0.151 | 0.670 ± 0.115 | **10/10** |
| 4 | `L_S1S2 ↔ R_M1_tongue` | 0.585 ± 0.165 | 0.669 ± 0.131 | 0/10 |
| 5 | `R_M1_tongue ↔ R_PMv` | 0.556 ± 0.120 | 0.642 ± 0.117 | 7/10 |
| 6 | `L_M1_foot ↔ R_M1_foot` | 0.512 ± 0.135 | 0.574 ± 0.133 | **10/10** |
| 7 | `L_S1S2 ↔ R_PMv` | 0.496 ± 0.133 | 0.574 ± 0.113 | 3/10 |
| 8 | `L_S1S2 ↔ R_S1S2` | 0.493 ± 0.136 | 0.577 ± 0.092 | 8/10 |
| 9 | `L_S1S2 ↔ SMA` | 0.433 ± 0.122 | 0.482 ± 0.059 | 4/10 |
| 10 | `R_PMv ↔ SMA` | 0.432 ± 0.123 | 0.468 ± 0.086 | 6/10 |

**Note on rank 4:** `L_S1S2 ↔ R_M1_tongue` has high group-mean MI (0.585) but appears in no CL tree. It is consistently strong but superseded in every subject by the stronger `L_M1_tongue ↔ L_S1S2` and `R_M1_tongue ↔ R_PMv` edges that already cover its two nodes in the MST.

---

## 4. Chow-Liu Tree Structure

### 4.1 Canonical edges (≥ 7/10 subjects)

| Edge | CL frequency | MI (mean ± SD) | Interpretation |
|------|-------------|----------------|----------------|
| `L_M1_tongue ↔ R_M1_tongue` | **10/10** | 0.876 ± 0.131 | Bilateral tongue M1 — obligate bilateral representation of midline organ |
| `L_M1_hand ↔ R_M1_hand` | **10/10** | 0.608 ± 0.151 | Bilateral hand M1 — interhemispheric motor coupling via corpus callosum |
| `L_M1_foot ↔ R_M1_foot` | **10/10** | 0.512 ± 0.135 | Bilateral foot M1 — consistent but weakest of the three bilateral M1 pairs |
| `L_M1_tongue ↔ L_S1S2` | 9/10 | 0.759 ± 0.250 | Left tongue M1 → left S2/operculum — orofacial sensorimotor feedback |
| `L_S1S2 ↔ R_S1S2` | 8/10 | 0.493 ± 0.136 | Bilateral S2 coupling — cross-hemispheric somatosensory integration |
| `R_M1_tongue ↔ R_PMv` | 7/10 | 0.556 ± 0.120 | Right tongue M1 → right PMv — orofacial motor coordination |

The three bilateral M1 pairs form a **conserved somatotopic core** present in every subject. Their rank order (tongue > hand > foot) mirrors expected interhemispheric coupling strength.

### 4.2 Variable edges (4–6/10 subjects)

| Edge | CL frequency | MI (mean ± SD) | Notes |
|------|-------------|----------------|-------|
| `R_PMv ↔ SMA` | 6/10 | 0.432 ± 0.123 | Premotor–SMA chain; present when SMA is not connected via L_S1S2 |
| `R_M1_foot ↔ R_M1_hand` | 6/10 | 0.416 ± 0.138 | Ipsilateral M1 coupling; likely reflects coactivation in right hemisphere |
| `L_PMd ↔ L_S1S2` | 5/10 | 0.400 ± 0.126 | Left PMd anchored through S2; sensorimotor-to-premotor chain |
| `L_S1S2 ↔ SMA` | 4/10 | 0.433 ± 0.122 | S2–SMA path; competes with `R_PMv ↔ SMA` for the SMA connection slot |

### 4.3 Hub nodes

| Node | Mean degree | Max degree (any subject) | Role |
|------|-------------|--------------------------|------|
| `L_S1S2` | **3.40** | 4 | Dominant group hub; anchors tongue, hand, and premotor branches |
| `R_M1_hand` | 2.70 | 4 | Secondary hub; connects hand M1 to foot and S2 branches |
| `L_M1_tongue` | 2.30 | 4 | Tongue core; connects bilateral tongue and L_S1S2 |
| `R_M1_tongue` | 1.90 | 3 | Tongue core; connects bilateral tongue and R_PMv |
| `R_PMv` | 1.70 | 3 | Premotor bridge; links tongue and SMA |
| `SMA` | 1.50 | 3 | Premotor anchor; connects PMv/PMd branches |
| `R_S1S2` | 1.00 | 2 | Leaf or near-peripheral in most subjects |

`L_S1S2` is the highest-degree node in 7/10 subjects (mean degree 3.40), making it the most consistent hub in the group. `R_M1_hand` is a hub in 6/10 subjects.

---

## 5. Individual Differences

### 5.1 Cross-subject variability in coupling strength

The mean off-diagonal MI ranges from 0.272 (MSC05) to 0.636 (MSC08), a 2.3× spread. This likely reflects a combination of:
- **Data quality**: motion censoring, signal-to-noise, number of censored frames
- **Genuine individual differences**: some individuals may have tighter motor network coupling at rest

The most reliable pair — `L_M1_tongue ↔ R_M1_tongue` — has the lowest relative variability (SD/mean = 15%), making it the best candidate for a normalization anchor or reliability benchmark. The `L_M1_tongue ↔ L_S1S2` pair has 33% coefficient of variation, explaining its absence in 1 subject.

### 5.2 Per-subject hub structure

| Subject | Hub nodes (degree ≥ 3) | Notable features |
|---------|----------------------|-----------------|
| MSC01 | `L_S1S2` | Sparse premotor branch |
| MSC02 | `L_S1S2`, `R_M1_hand`, `R_M1_tongue` | Elaborated orofacial and hand networks |
| MSC03 | `L_S1S2` | Single hub; weakest orofacial coupling |
| MSC04 | `L_M1_tongue`, `L_S1S2`, `R_M1_hand` | Three hubs; highest MI subject |
| MSC05 | `R_M1_hand`, `R_M1_tongue`, `SMA` | SMA prominent; L_S1S2 less central |
| MSC06 | `L_S1S2`, `R_M1_hand` | Canonical two-hub structure |
| MSC07 | `L_M1_tongue` | Tongue-dominated; SMA connects via tongue not S2 |
| MSC08 | `L_S1S2`, `R_M1_hand` | Canonical two-hub; highest MI subject |
| MSC09 | `L_S1S2`, `R_M1_hand` | Canonical two-hub |
| MSC10 | `L_S1S2`, `R_M1_hand`, `SMA` | SMA bridge prominent |

The most common topology is the **L_S1S2 + R_M1_hand two-hub structure** (4/10 subjects: MSC06, MSC08, MSC09, MSC10), suggesting this as the modal motor CL tree configuration in this cohort.

---

## 6. Interpretation

### 6.1 Conserved somatotopic core

The three bilateral M1 pairs (tongue, hand, foot) are present in every subject, forming a conserved backbone of the motor CL tree. Their MI rank order — tongue (0.876) > hand (0.608) > foot (0.512) — is consistent with known neuroanatomy: tongue M1 has obligate bilateral corticobulbar representation; hand M1 has extensive callosal connections and mirror activity; foot M1 has the weakest interhemispheric coupling reflecting more independent lower-limb control.

### 6.2 L_S1S2 as the dominant integration hub

The emergence of `L_S1S2` (left parietal operculum / SII cortex) as the group's highest-degree hub is neuroanatomically coherent. Area OP1/OP4 receives direct thalamocortical input from VPLc, projects to M1, PMd, and SMA, and is activated by passive touch, active grasping, and orofacial movements across effectors. Its role as a bridge between the tongue M1 cluster and the premotor/SMA branch suggests it mediates cross-effector sensorimotor integration rather than effector-specific processing.

### 6.3 Orofacial network dominance

The strongest MI edges and the dominant hub all involve the orofacial subnetwork (`L_M1_tongue`, `R_M1_tongue`, `L_S1S2`). This is consistent with the large size of the tongue fROI — tongue activates more cortex than foot or hand in this dataset, and the merged `L_M1_tongue` ROI includes frontal and insular opercular territory beyond primary M1. ROI-size equalization would be needed to confirm whether orofacial dominance reflects stronger connectivity or simply larger masks.

### 6.4 Premotor branch variability

SMA, PMd, and PMv edges show substantially higher inter-subject variability than the M1 pairs. `R_PMv ↔ SMA` appears in 6/10 subjects and `L_PMd ↔ L_S1S2` in 5/10, but neither is universal. This suggests premotor connectivity is more individually idiosyncratic, consistent with PMd/PMv's role in action selection and motor planning where individual strategy differences are expected.

---

## 7. Predictions from gcss_plan.md — Validation

| Prediction | Outcome |
|-----------|---------|
| Bilateral tongue M1 coupling is the strongest edge | **Confirmed** — rank 1 by group-mean MI (0.876), 10/10 subjects |
| SMA as integrative hub | **Partially confirmed** — SMA is a hub (degree ≥ 3) in 3/10 subjects; L_S1S2 outranks it as the dominant hub |
| Foot M1 bilateral pair weaker than hand and tongue | **Confirmed** — foot MI (0.512) < hand (0.608) < tongue (0.876) |
| L_S1S2 / parietal operculum as near-peripheral | **Refuted** — L_S1S2 is the dominant hub (mean degree 3.40), not peripheral |
| Consistent topology across subjects | **Partially confirmed** — 3 universal edges (bilateral M1 pairs); premotor branches variable |

---

## 8. Limitations and Next Steps

1. **ROI size confound for orofacial dominance.** The `L_M1_tongue` ROI merges primary motor cortex with frontal and insular opercular territory, producing larger masks than foot/hand ROIs. MI is sensitive to ROI size through effective resolution. A size-matched analysis is needed to disentangle connectivity from ROI extent.

2. **Bilateral M1 pairs exhaust tree capacity.** The three bilateral pairs consume 6 of the 10 tree edges in most subjects, leaving only 4 edges to describe premotor connectivity. A partial-correlation or conditioned-MI tree that removes bilateral M1 structure would expose the premotor sub-network.

3. **L_S1S2 merges heterogeneous anatomy.** Left_PFop (S2), Left_PF (supramarginal), and Left_FOP2 are distinct areas that are merged here. Splitting these into separate ROIs would test whether hub behavior is driven by one specific sub-region.

4. **No within-subject test-retest reliability.** Even-run fROIs have not yet been extracted. Odd/even spatial overlap (benchmark r ≈ 0.52, Fedorenko 2010) would confirm fROI stability.

5. **No group-level consensus tree.** Per-subject CL trees were analyzed independently. A consensus tree via majority-vote edges or Karcher mean on spanning-tree space would provide a single group-level structural summary.

6. **Cross-subject MI variability not corrected for motion.** The 2.3× spread in mean MI (0.27–0.64 nats) warrants a check against per-subject motion statistics (mean FD, number of censored frames) to separate data-quality effects from genuine individual differences.

---

## 9. File Index

### Per-subject figures — `figures/canonical_circuits/motor_cortex/gcss/{sub}/`
| Pattern | Content |
|---------|---------|
| `{sub}_all_sessions_gcss_fc_matrix.pdf` | FC matrix (all-sessions) |
| `{sub}_all_sessions_gcss_mi_matrix.pdf` | MI matrix (all-sessions) |
| `{sub}_all_sessions_gcss_cl_tree.pdf` | Chow-Liu tree (all-sessions) |
| `{sub}_all_sessions_gcss_fc_vs_mi.pdf` | FC vs MI scatter, CL edges highlighted |
| `{sub}_func{01–10}_gcss_*.pdf` | Per-session versions (40 files per subject) |

### Connectivity data — `datasets/midnight_scan_club/connectivity/`
```
{fc,mi,cl}/{sub}/{func01..func10,all_sessions}/gcss_motor/
  {sub}_{ses}_gcss_{fc,mi,cl}.npy    — (11×11) matrices
  {sub}_{ses}_gcss_roi_keys.json     — ordered ROI key list
```

### fROI masks — `analysis/canonical_circuits/motor_cortex/gcss/froi_masks/{sub}/`
```
{sub}_gcss_{sem_name}_mask.nii.gz   — 11 files per subject (2 mm MNI)
```
