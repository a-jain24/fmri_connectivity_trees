# Individual Differences Analysis Plan: CL Tree vs FC Conservation

**Hypothesis:** Chow-Liu trees show greater conservation of organization across individuals compared to functional connectivity matrices. CL trees extract the backbone of sparse connectivity (37 edges for 38 ROIs), while FC captures all pairwise correlations (703 edges), making CL more robust to individual noise and task-related variability.

**Rationale:** The maximum spanning tree enforces a unique sparse structure that captures only the strongest dependency relationships. This constraint may filter out spurious correlations while preserving canonical circuit topology. FC, being dense and noisy, may reflect individual differences more strongly than true organizational principles.

---

## Analysis Strategy

### Phase 1: Quantify Conservation Within Each Connectivity Type

#### A. FC Conservation (Pairwise Correlation Stability)

**Metric 1: FC Matrix Similarity Across Subjects**
- Compute all pairwise Pearson correlations between FC matrices (vectorized upper triangle)
- Result: 10×10 subject similarity matrix for FC
- Expected: mean r ≈ 0.5–0.7 if moderately conserved

```python
fc_pairs = []
for i in range(10):
    for j in range(i+1, 10):
        fc_i = fc_matrices[subjects[i]]
        fc_j = fc_matrices[subjects[j]]
        vec_i = fc_i[np.triu_indices_from(fc_i, k=1)]
        vec_j = fc_j[np.triu_indices_from(fc_j, k=1)]
        r = np.corrcoef(vec_i, vec_j)[0, 1]
        fc_pairs.append(r)
mean_fc_conservation = np.mean(fc_pairs)
```

**Metric 2: Edge Stability (Top-N FC Edges)**
- For each subject, identify top-K edges by |r| (K = 37, matching CL edge count)
- Compute Jaccard similarity of top-K edge sets across all subject pairs
- Result: fraction of overlapping high-FC edges (0 = no overlap, 1 = perfect overlap)

```python
fc_top_edges = {subject: set top 37 edges for subject}
jaccard_fc = []
for i, j in subject_pairs:
    overlap = len(fc_top_edges[i] & fc_top_edges[j])
    union = len(fc_top_edges[i] | fc_top_edges[j])
    jaccard = overlap / union
    jaccard_fc.append(jaccard)
mean_jaccard_fc = np.mean(jaccard_fc)
```

#### B. CL Tree Conservation (Topology Stability)

**Metric 1: CL Edge Overlap Across Subjects**
- For each subject pair, compute Jaccard similarity of CL edge sets
- Result: 0–1 score (1 = identical trees)
- Expected: higher than FC top-edges if hypothesis true

```python
cl_edges = {subject: set(CL adjacency edges) for subject}
jaccard_cl = []
for i, j in subject_pairs:
    overlap = len(cl_edges[i] & cl_edges[j])
    union = len(cl_edges[i] | cl_edges[j])
    jaccard = overlap / union
    jaccard_cl.append(jaccard)
mean_jaccard_cl = np.mean(jaccard_cl)
```

**Metric 2: Tree Structure Similarity (Graph Isomorphism)**
- Compute tree edit distance (TED) between CL trees
  - Normalized by maximum possible distance
  - Smaller = more similar structure
- Also compute graph correlation coefficient (GCC) from NetworkX

```python
from scipy.spatial.distance import jaccardacond
ted_normalized = []
for i, j in subject_pairs:
    tree_i = nx.Graph(cl_adjacency[i])
    tree_j = nx.Graph(cl_adjacency[j])
    # Compute tree edit distance (requires graph_metric library or manual implementation)
    ted = tree_edit_distance_normalized(tree_i, tree_j)
    ted_normalized.append(ted)
mean_ted = np.mean(ted_normalized)
```

**Metric 3: Node Degree Preservation**
- For each node, compute its degree (number of edges) across subjects
- Correlate degree sequences across subject pairs
- Result: correlation of node centrality structure

```python
degrees_cl = {subject: [degree(node) for each node in CL tree]}
corr_degrees = []
for i, j in subject_pairs:
    r = np.corrcoef(degrees_cl[i], degrees_cl[j])[0, 1]
    corr_degrees.append(r)
mean_degree_corr = np.mean(corr_degrees)
```

---

### Phase 2: Direct Comparison

#### Test 1: Edge Stability Comparison

**Null Hypothesis:** Mean Jaccard(CL) = Mean Jaccard(FC top-37)

```python
# Paired t-test: CL vs FC stability
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(jaccard_cl, jaccard_fc)
effect_size = np.mean(jaccard_cl) - np.mean(jaccard_fc)

print(f"CL edge Jaccard:   {np.mean(jaccard_cl):.3f} ± {np.std(jaccard_cl):.3f}")
print(f"FC top-37 Jaccard: {np.mean(jaccard_fc):.3f} ± {np.std(jaccard_fc):.3f}")
print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")
print(f"Effect size (Δ Jaccard): {effect_size:.3f}")
```

**Expected outcome:** CL Jaccard > FC Jaccard, p < 0.05 (supports hypothesis)

#### Test 2: Degree Sequence Stability

**Compare node degree correlation:**
```python
# Paired t-test: CL degree correlation vs FC degree correlation
t_stat, p_val = ttest_rel(corr_degrees, fc_degree_corr)
print(f"CL degree correlation:   {np.mean(corr_degrees):.3f}")
print(f"FC degree correlation:   {np.mean(fc_degree_corr):.3f}")
print(f"t-test: p={p_val:.4f}")
```

---

### Phase 3: Anatomical Organization Stability

#### Test 3: Somatotopic Organization Preservation

**Hypothesis:** CL trees preserve somatotopic structure (foot → hand → tongue) more consistently than FC.

**Metrics:**
1. **Foot-Hand-Tongue Cluster Stability**
   - For each subject, extract sub-graphs of CL tree connecting only {foot, hand, tongue} ROIs
   - Compute cluster coefficients and average path lengths
   - Compare across subjects: are these metrics stable?

2. **Hierarchical Organization**
   - Compute 2D MDS embedding of each CL tree (graph distance)
   - Measure if somatotopic order (foot superior → tongue inferior) is preserved in the embedding
   - Use Spearman rank correlation of MNI z-coordinates with MDS 2nd principal axis

```python
# For each subject's CL tree
embeddings_cl = {}
somatotopic_order = []
for subject in subjects:
    tree = nx.Graph(cl_adjacency[subject])
    dist_matrix = nx.all_pairs_shortest_path_length(tree)
    # Compute MDS
    embedding = MDS(n_components=2).fit_transform(dist_matrix)
    embeddings_cl[subject] = embedding
    
    # Get somatotopic order from MNI z-coordinates
    z_coords = np.array([centroids[roi_key][2] for roi_key in roi_keys])
    
    # Correlate z-coords with embedding second axis
    r = spearmanr(z_coords, embedding[:, 1])[0]
    somatotopic_order.append(r)

print(f"CL somatotopic preservation (Spearman r): {np.mean(somatotopic_order):.3f}")
```

**Compare to FC:** Repeat for top-37 FC edges.

---

### Phase 4: Consensus Structure Analysis

#### Building Consensus Graphs

**CL Consensus Tree:**
1. Overlay all 10 CL trees
2. Compute edge frequency (# subjects where edge appears)
3. Extract high-frequency edges (appearing in ≥5 subjects) → consensus tree
4. Measure: what fraction of edges are conserved across ≥50% of subjects?

```python
edge_counts = {}
for subject in subjects:
    for edge in cl_edges[subject]:
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

# Consensus: edges in ≥50% of subjects
consensus_cl_edges = [e for e, count in edge_counts.items() if count >= 5]
conservation_ratio_cl = len(consensus_cl_edges) / 37
print(f"CL consensus edges (≥50% subjects): {conservation_ratio_cl:.1%}")
```

**FC Consensus (Top-37 edges):**
Repeat for FC top-N edges.

```python
# Expected: CL consensus_ratio >> FC consensus_ratio
```

---

### Phase 5: Noise Robustness Test

#### Bootstrap Stability

For each subject:
1. Subsample ROI timeseries: 90%, 80%, 70% of TRs
2. Recompute FC and CL tree for each subsample
3. Measure edge/tree similarity between subsamples
4. Compare: which (CL vs FC) shows lower sensitivity to data reduction?

```python
for subject in subjects:
    ts_full = load_timeseries(subject)
    n_tr = ts_full.shape[0]
    
    fc_stability = []
    cl_stability = []
    
    for frac in [0.9, 0.8, 0.7]:
        indices = np.random.choice(n_tr, int(n_tr * frac), replace=False)
        ts_sub = ts_full[indices]
        
        fc_sub = compute_fc(ts_sub)
        cl_sub = compute_cl_tree(ts_sub)
        
        # Compare to full
        fc_jaccard = jaccard_similarity(top_37_edges(fc_full), top_37_edges(fc_sub))
        cl_jaccard = jaccard_similarity(cl_edges_full, cl_edges_sub)
        
        fc_stability.append(fc_jaccard)
        cl_stability.append(cl_jaccard)
    
    print(f"{subject}: FC drop = {1 - np.mean(fc_stability):.1%}, "
          f"CL drop = {1 - np.mean(cl_stability):.1%}")
```

---

## Output Files

Create results file: `motor_cortex_conservation_analysis.json`

```json
{
  "conservation_metrics": {
    "fc_matrix_correlation": 0.62,
    "fc_top37_jaccard": 0.31,
    "cl_edge_jaccard": 0.52,
    "cl_degree_correlation": 0.71,
    "cl_somatotopic_preservation": 0.68
  },
  "statistical_tests": {
    "jaccard_comparison_t": 4.23,
    "jaccard_comparison_p": 0.0012,
    "effect_size": 0.21,
    "interpretation": "CL trees show significantly higher edge conservation"
  },
  "consensus_analysis": {
    "cl_consensus_ratio": 0.68,
    "fc_consensus_ratio": 0.28
  },
  "bootstrap_robustness": {
    "fc_stability_drop_at_70percent": 0.15,
    "cl_stability_drop_at_70percent": 0.08
  }
}
```

---

## Visualization Plan

1. **Figure 1: Conservation Heatmaps**
   - 10×10 Jaccard matrices: CL edges vs FC top-37 edges
   - Side-by-side comparison with dendrograms

2. **Figure 2: Statistical Comparison**
   - Box plots: CL Jaccard vs FC Jaccard (with paired lines)
   - t-test results annotated

3. **Figure 3: Somatotopic Preservation**
   - MDS embeddings of CL trees from 4 subjects
   - Color-coded by effector (foot/hand/tongue)
   - Plot: does somatotopic order emerge consistently?

4. **Figure 4: Consensus Structure**
   - Heatmap of edge frequency across subjects (CL and FC)
   - Highlight high-frequency edges (conserved structure)

5. **Figure 5: Bootstrap Robustness**
   - Line plot: Jaccard similarity vs. % of data retained
   - Separate lines for CL and FC across all subjects

---

## Expected Findings & Interpretation

### If Hypothesis Is Supported:
- CL edge Jaccard: 0.45–0.60 (mean)
- FC top-37 Jaccard: 0.25–0.35 (mean)
- t-test p < 0.01, effect size > 0.15

**Interpretation:** CL trees extract a more universal backbone of motor organization. Individual FC variation reflects task-dependent, subject-specific fluctuations, while CL captures canonical structure.

### Alternative Outcome:
If CL and FC show similar conservation, conclude that individual variability is fundamental to motor connectivity, not an artifact of FC's density.

---

## Next Steps (Post-Analysis)

1. **Cortical Gradient Analysis:** Do conserved CL edges align with cortical gradients (e.g., somatosensory → motor)?
2. **Cross-Dataset Validation:** Repeat analysis on ABIDE or LISTEN datasets
3. **Clinical Implications:** Is CL tree topology disrupted in motor disorders (Parkinson's, stroke)?
4. **Temporal Dynamics:** Does CL tree structure change across learning/adaptation?

---

**Status:** Ready for implementation
**Date Created:** 2026-05-04
**Implementation Start:** After approval
