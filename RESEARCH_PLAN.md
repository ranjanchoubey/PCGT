# Research Plan: Honest Assessment and Execution Roadmap

## Section 0: Should You Even Do This?

### Honest Risk Assessment

**What you have going for you:**
- M.Tech from ISI Kolkata — you understand statistics, math, and ML fundamentals
- You already know graph partitioning, METIS, subgraph transformers deeply from your thesis work
- You have PhD scholar contacts willing to help if you show real work
- You're willing to invest (Colab Pro subscription)
- You have a working codebase and familiarity with the problem domain

**What's working against you:**
- No active advisor — nobody to catch mistakes early
- Working alone — no daily research discussions
- Not a proof-writing researcher — deep theoretical contributions will be hard
- The graph Transformer space is crowded — many strong labs publishing here
- You need results that actually beat SOTA, not just match

**Honest probability of success:**
- Getting a workshop paper (NeurIPS/ICML workshop, LoG): ~60% if you execute well
- Getting a main conference paper (AAAI, IJCAI): ~30-40%
- Getting a top venue (NeurIPS, ICML, ICLR main): ~10-15% (very competitive)

**When to back out:**
- After Week 2 (Phase 1 done): If you cannot reproduce SGFormer's numbers, stop. The foundation is broken.
- After Week 4 (Phase 2 done): If PCGT does NOT outperform SGFormer on at least 4 out of 7 medium datasets, the idea doesn't work. Stop.
- These are hard go/no-go gates. Respect them.

---

## Section 1: The Idea (Stripped to What Matters)

### One-paragraph summary
SGFormer uses one-layer linear attention (approximate, all-pair) + GCN. It's fast but its attention is an approximation — it loses expressiveness compared to exact softmax attention. We exploit graph structure through METIS partitioning to enable **exact softmax attention within partitions** at the same linear cost. Nodes that actually matter to each other (structurally close) get exact attention. Cross-partition information flows through boundary nodes. No approximation where it counts.

### What makes this different from existing work

| Existing Method | What They Do | How We Differ |
|---|---|---|
| SGFormer | Linear approximate attention over all nodes | We use exact attention within partitions — no approximation error for local interactions |
| FairGP (AAAI 2025) | Graph partitioning for fairness in GTs | We optimize for accuracy and scalability, not fairness. Different attention mechanism, different goal, different experiments |
| HSGT | Graph coarsening → hierarchical Transformer | Coarsening merges nodes (loses identity). We partition but keep every node. Fundamentally different |
| ClusterGCN | Graph partitioning for scalable GNN training | GNN only, no Transformer, no attention mechanism |
| NAGphormer | Per-node hop tokenization | Different tokenization strategy, doesn't use partitioning |

### The key claim (what reviewers will judge)
> "Structure-preserving partitioning enables exact local attention within subgraphs at linear overall cost, outperforming approximate global attention methods like SGFormer on standard benchmarks."

**This claim requires NO deep theory.** It's an empirical claim verified by experiments. If the numbers are good, the paper stands.

---

## Section 2: Architecture Specification

### Components

```
PCGT Model:
├── Input Layer: MLP(X) → Z₀ ∈ R^{N×d}
├── Preprocessing (computed ONCE before training):
│   ├── METIS partition: G → {S₁, S₂, ..., Sₖ}
│   ├── Boundary node identification
│   ├── Partition membership embedding (learnable, one per partition)
│   └── Boundary indicator feature (binary: is this node on a cut edge?)
├── PCGT Layer (stack 1-2 of these):
│   ├── Intra-Partition Attention:
│   │   ├── For each partition Sₖ: standard multi-head softmax attention
│   │   ├── Input: nodes in Sₖ + partition PE
│   │   ├── This is EXACT attention, not approximate
│   │   └── Complexity per partition: O(|Sₖ|²), total: O(N²/K)
│   ├── Cross-Partition Communication:
│   │   ├── For each boundary node: aggregate from boundary neighbors in other partitions
│   │   ├── Mechanism: simple attention or weighted message passing
│   │   └── Complexity: O(|boundary edges|) which is O(E_cut)
│   └── Combine: LayerNorm(intra_output + cross_output) → FFN → output
├── GCN Branch (same as SGFormer): 
│   ├── 1-2 layer GCN on full graph
│   └── Captures explicit structural info
├── Combine: α × PCGT_output + (1-α) × GCN_output
└── Output Layer: Linear → prediction
```

### Why this should work (intuition, not proof)
- METIS partitioning puts structurally close nodes together
- Most useful attention is between structurally close nodes (especially in homophilic graphs)
- Exact softmax captures this better than SGFormer's linear approximation
- Cross-partition communication handles the rest
- GCN branch provides structural grounding (same as SGFormer does)

### Complexity math (straightforward, not deep theory)
- K partitions of size N/K each
- Intra-partition attention: K × O((N/K)²) = O(N²/K)
- With K = √N: this becomes O(N^{1.5}) — better than O(N²), but not linear
- With K = N/c for constant c (e.g., partition size 2000): O(N × c) = O(N) — LINEAR
- Cross-partition: O(E_cut) — bounded by total edges
- GCN: O(N + E) — standard
- **Total: O(N + E) — same as SGFormer**

**Important**: For medium graphs (N < 30K), you can use K such that each partition is ~500-2000 nodes. This easily fits in GPU memory and is fast.

---

## Section 3: What You Need to Build

### Codebase: Fork SGFormer (not start from scratch)

SGFormer repo: https://github.com/qitianwu/SGFormer

**Why fork SGFormer:**
- Same data loading, same splits, same evaluation protocol
- Direct comparison is fair and no reviewer can question it
- You only need to add ONE new model class

### Files you need to write (estimated lines)

| File | What | Estimated Lines | Difficulty |
|---|---|---|---|
| `model/pcgt.py` | PCGT model class | ~200-300 | Medium |
| `partition.py` | METIS partitioning + boundary detection | ~100 | Easy (you already know this!) |
| `train_pcgt.py` | Training loop adapted for PCGT | ~50 lines change from SGFormer's train.py | Easy |
| Config files | Hyperparameters per dataset | ~50 | Easy |

**Total new code: ~400-500 lines.** The rest is reused from SGFormer.

### Key implementation details

**Intra-partition attention:**
```python
# Pseudocode — this is standard PyTorch
for k in range(num_partitions):
    nodes_k = partition_indices[k]  # indices of nodes in partition k
    Z_k = Z[nodes_k]               # extract their embeddings
    # Standard multi-head attention (PyTorch nn.MultiheadAttention)
    Z_k_out = self.attention(Z_k, Z_k, Z_k)  
    Z_out[nodes_k] = Z_k_out       # scatter back
```

This is straightforward. No custom CUDA kernels. Standard PyTorch.

**Cross-partition communication:**
```python
# For each boundary node, aggregate from cross-partition neighbors
for b in boundary_nodes:
    cross_neighbors = get_cross_partition_neighbors(b)
    msg = aggregate(Z[cross_neighbors])  # mean, attention, or weighted sum
    Z_out[b] = Z_out[b] + msg
```

Also straightforward. Can use DGL or PyG message passing for efficiency.

**Partition preprocessing (you already know this):**
```python
import dgl
partition_labels = dgl.metis_partition(graph, num_partitions=K)
```

---

## Section 4: Experiments Plan

### Datasets (exactly matching SGFormer's Table 2)

| Dataset | Nodes | Edges | Type | Splits |
|---|---|---|---|---|
| Cora | 2,708 | 5,278 | Homophilic | Semi-supervised (GCN splits) |
| Citeseer | 3,327 | 4,552 | Homophilic | Semi-supervised (GCN splits) |
| Pubmed | 19,717 | 44,324 | Homophilic | Semi-supervised (GCN splits) |
| Actor | 7,600 | 29,926 | Heterophilic | Random splits [Lim et al. 2021] |
| Squirrel | 2,223 | 46,998 | Heterophilic | Filtered splits [Platonov et al. 2023] |
| Chameleon | 890 | 8,854 | Heterophilic | Filtered splits [Platonov et al. 2023] |
| Deezer | 28,281 | 92,752 | Heterophilic | Random 50/25/25 |

**CRITICAL**: Use EXACTLY the same splits as SGFormer. Their code has this built in. Do not use your old dataset splits.

### Large graphs (stretch goal — do these last)

| Dataset | Nodes | Where |
|---|---|---|
| ogbn-arxiv | 169K | Colab Pro (T4 sufficient) |
| Amazon2M | 2.4M | Colab Pro (A100 needed) |

### Evaluation protocol
- 5 independent runs with different random seeds
- Report mean ± std
- Same as SGFormer paper

### Target numbers to beat

| Dataset | SGFormer | NodeFormer | Your target (minimum) |
|---|---|---|---|
| Cora | 84.5 ± 0.8 | 82.2 ± 0.9 | > 84.5 |
| Citeseer | 72.6 ± 0.2 | 72.5 ± 1.1 | > 72.6 |
| Pubmed | 80.3 ± 0.6 | 79.9 ± 1.0 | > 80.3 |
| Actor | 37.9 ± 1.1 | 36.9 ± 1.0 | > 37.9 |
| Squirrel | 41.8 ± 2.2 | 38.5 ± 1.5 | > 41.8 |
| Chameleon | 44.9 ± 3.9 | 34.7 ± 4.1 | > 44.9 |
| Deezer | 67.1 ± 1.1 | 66.4 ± 0.7 | > 67.1 |

**Reality check**: You probably won't beat SGFormer on ALL 7. Beating on 5+ with competitive on the rest is enough for a paper. Beating on 4 or fewer likely means the approach needs rethinking.

### Hyperparameter search space (keep small and focused)

| Param | Search Space | Notes |
|---|---|---|
| Number of partitions K | {5, 10, 20, 50, 100} | Tune per dataset based on graph size |
| Hidden dim | {64, 128, 256} | Same as SGFormer's search |
| Attention heads | {1, 4, 8} | Standard |
| PCGT layers | {1, 2} | SGFormer uses 1; more may overfit |
| GCN layers | {1, 2} | Same as SGFormer |
| α (GCN weight) | {0.5, 0.8} | Same as SGFormer |
| Learning rate | {0.001, 0.005, 0.01} | Standard |
| Weight decay | {1e-5, 5e-4, 1e-3} | Standard |
| Dropout | {0, 0.2, 0.5} | Standard |

### Required ablations (reviewers WILL ask for these)

1. **PCGT full** vs **PCGT without cross-partition comm** — shows boundary communication matters
2. **PCGT full** vs **SGFormer with METIS batching** (no intra-partition attention, just better batching) — shows the attention mechanism matters, not just partitioning
3. **Impact of K**: sweep K values, plot accuracy vs K — shows robustness
4. **Random partition vs METIS partition** — shows structure-preserving partitioning matters
5. **Attention visualization**: show that intra-partition attention weights are more structured/meaningful than SGFormer's linear attention

---

## Section 5: About Theory (Honest)

### What you CAN include (no formal proof needed)

1. **Complexity analysis** — straightforward math showing O(N+E) total cost. This is counting operations, not a theorem.

2. **Informal analysis of attention quality** — "Most edges (X%) are intra-partition after METIS, therefore most useful attention interactions are captured exactly." Back this up with a simple empirical measurement: for each dataset, count what fraction of edges are within vs across partitions.

3. **Connection to ClusterGCN** — ClusterGCN showed partitioned training works for GNNs. You extend that insight to Transformers.

### What you should NOT attempt without help

- Formal expressiveness proofs (WL-test style)
- Tight approximation error bounds
- Convergence guarantees

### What your PhD scholar could help with (if they're willing)

- Formalizing the "attention energy within partitions" argument as a proper proposition
- Reviewing the complexity analysis for correctness
- Suggesting relevant theoretical references to cite

**Bottom line**: A clean empirical paper with good ablations and an informal analysis section is publishable at LoG, AAAI, IJCAI. You don't need proofs if the experiments are thorough and the analysis is insightful.

---

## Section 6: Detailed Week-by-Week Plan

### Phase 1: Foundation (Week 1-2)

**Goal**: Reproduce SGFormer's published numbers on all 7 datasets.

**Week 1:**
- Day 1-2: Clone SGFormer repo. Set up environment. Understand their code structure.
- Day 3-4: Run SGFormer on Cora, Citeseer, Pubmed (small, fast, CPU is fine).
- Day 5-7: Run SGFormer on Actor, Squirrel, Chameleon, Deezer. Compare to Table 2.

**Week 2:**
- Day 1-3: If numbers don't match, debug. Check splits, hyperparameters.
- Day 4-5: Document reproduced numbers in a table. 
- Day 6-7: Read SGFormer code deeply. Identify EXACTLY where to inject partitioning.

**GO/NO-GO Gate 1**: Can you reproduce SGFormer numbers within ±1% of published? 
- YES → Proceed to Phase 2
- NO → Debug. If still can't after 3 more days, the foundation is unreliable. Reconsider.

**Deliverable**: Table of reproduced SGFormer numbers.

---

### Phase 2: Core Implementation (Week 3-4)

**Goal**: Implement PCGT and get first results.

**Week 3:**
- Day 1-2: Implement METIS partitioning preprocessing. Store partition labels, boundary nodes.
- Day 3-4: Implement intra-partition attention module. Test on Cora (smallest graph).
- Day 5-7: Implement cross-partition boundary communication. Wire into training loop.

**Week 4:**
- Day 1-2: Debug. Run PCGT on Cora. Does it train? Does loss decrease? Does accuracy improve?
- Day 3-4: Run PCGT on all 7 datasets with default hyperparameters (no tuning yet).
- Day 5-7: First comparison table: PCGT (untuned) vs SGFormer.

**GO/NO-GO Gate 2**: With default hyperparameters, does PCGT beat SGFormer on at least 2 datasets?
- YES → Proceed. Hyperparameter tuning will improve more.
- MIXED (beats 1, close on others) → Proceed cautiously. May need architecture tweaks.
- NO (worse than SGFormer everywhere) → The idea may not work. Take 3 days to analyze WHY. Check: Is the partitioning actually capturing useful structure? Is the attention learning? Share results with PhD scholar for their opinion. If no clear fix, consider pivoting.

**Deliverable**: Working PCGT implementation + first results table.

---

### Phase 3: Hyperparameter Tuning + Full Results (Week 5-6)

**Goal**: Get best numbers across all datasets.

**Week 5:**
- Day 1-3: Tune K (number of partitions) per dataset. This is likely the most impactful hyperparameter.
- Day 4-5: Tune learning rate, hidden dim, dropout per dataset.
- Day 6-7: 5-seed runs on best configs for all 7 datasets.

**Week 6:**
- Day 1-3: Run ablation experiments (Section 4 ablation list).
- Day 4-5: Attention visualization: extract attention weights, plot heatmaps.
- Day 6-7: Compile ALL results into tables and figures.

**GO/NO-GO Gate 3**: After tuning, does PCGT beat SGFormer on at least 4 out of 7 datasets (with competitive on rest)?
- YES → This is a paper. Proceed to writing.
- NO → Honest discussion needed. Consider:
  - Can you add one more component to close the gap?
  - Is the story strong enough for a workshop paper even without full SOTA?
  - Discuss with PhD scholar.

**Deliverable**: Complete results table + ablation tables + figures.

---

### Phase 4: Large Graph Experiments (Week 7, optional but recommended)

**Goal**: Show scalability on ogbn-arxiv and optionally Amazon2M.

- Day 1-2: Run PCGT on ogbn-arxiv (fits in Colab Pro T4).
- Day 3-4: Run SGFormer on ogbn-arxiv for comparison.
- Day 5-7: If results are promising, run Amazon2M on A100 (Colab Pro).

**This phase is optional.** If medium-graph results are strong, the paper works without large graphs. But large-graph results make the scalability claim concrete.

**Deliverable**: Large-graph results table.

---

### Phase 5: Paper Writing (Week 8-10)

**Week 8:**
- Day 1-2: Write Introduction and Related Work.
- Day 3-5: Write Method section (architecture, complexity analysis).
- Day 6-7: Write Experiment Setup section.

**Week 9:**
- Day 1-3: Write Results and Analysis sections.
- Day 4-5: Create all figures (architecture diagram, attention visualization, K-sensitivity plot).
- Day 6-7: Write Abstract and Conclusion.

**Week 10:**
- Day 1-3: Internal review. Read the whole paper 3 times. Fix gaps.
- Day 4-5: Send to PhD scholar for review.
- Day 6-7: Incorporate feedback. Final polish.

**Deliverable**: Complete paper draft.

---

## Section 7: Target Venues and Deadlines

| Venue | Type | Typical Deadline | Fit |
|---|---|---|---|
| LoG 2026 (Learning on Graphs) | Conference | Usually Sep-Oct | **Best fit** — dedicated graph ML venue |
| AAAI 2027 | Conference | Usually Aug 2026 | Good fit — accepts empirical graph papers |
| IJCAI 2027 | Conference | Usually Jan 2027 | Good fit |
| NeurIPS 2026 GLFrontiers Workshop | Workshop | Usually Sep 2026 | Lower bar, good for first submission |
| ICML 2026 Workshop | Workshop | Usually May 2026 | Tight but possible if you start now |

**Recommended strategy**: Target LoG 2026 as primary. If rejected, revise and submit to AAAI 2027.

---

## Section 8: What Could Go Wrong (And What To Do)

| Risk | Likelihood | Mitigation |
|---|---|---|
| PCGT doesn't beat SGFormer on enough datasets | Medium | Tune K extensively. Try adding a learnable partition refinement step. If still failing, pivot to workshop paper showing interesting analysis of WHY partition attention behaves differently. |
| METIS partitioning is too slow as preprocessing | Low | METIS is very fast (< 1 sec for graphs under 1M nodes). Not a concern for target datasets. |
| Reviewers say "just ClusterGCN with attention" | Medium | Your ablations will show that random partitioning doesn't help — it's the combination of structure-preserving partitioning + exact attention that works. Also ClusterGCN doesn't use any Transformer/attention. |
| Reviewers ask for theoretical analysis | High | Include the complexity analysis (O(N+E)) and empirical analysis (fraction of edges within partitions). Acknowledge formal analysis as future work. Many empirical papers get accepted without formal theory. |
| Can't reproduce SGFormer numbers | Low | Their code is well-maintained and widely used. If you can't reproduce, check PyTorch/CUDA versions. |
| Heterophilic datasets don't improve | Medium | For heterophilic graphs, partition boundaries carry important signal. Strengthen the cross-partition communication mechanism. Allocate more attention heads to boundary nodes. |

---

## Section 9: Summary Checklist

Before you start, confirm you have:
- [ ] Colab Pro subscription (or access to GPU)
- [ ] GitHub account to fork SGFormer
- [ ] Python environment with PyTorch, DGL, PyG
- [ ] Understanding of SGFormer paper (read it fully, not just abstract)
- [ ] Contact info for PhD scholar (for review at key stages)
- [ ] Realistic time commitment (2-3 hours per day minimum for 10 weeks)

At each gate, honestly answer:
- [ ] Gate 1 (Week 2): Can I reproduce SGFormer? → If NO, stop.
- [ ] Gate 2 (Week 4): Does PCGT show any promise? → If NO everywhere, reconsider.
- [ ] Gate 3 (Week 6): Does tuned PCGT beat SGFormer on 4+ datasets? → If NO, discuss with PhD scholar before investing more.

---

## Section 10: Final Honest Word

This plan is achievable for a solo M.Tech graduate from ISI Kolkata. The implementation is ~500 lines on top of an existing codebase. The experiments use standard datasets with established protocols. No custom CUDA kernels. No formal proofs required.

The hardest part is not the code — it's the discipline to follow the go/no-go gates honestly. If you reach Gate 2 and the numbers are bad, don't spend 3 months hoping they'll improve. Either find the concrete reason why and fix it, or pivot.

If the idea works (Gate 2 and 3 pass), you will have a clean, publishable paper. Not because of fancy theory, but because you found a practical insight (exact local attention via partitioning) that demonstrably improves a strong baseline (SGFormer) across standard benchmarks.

That is what gets papers accepted.
