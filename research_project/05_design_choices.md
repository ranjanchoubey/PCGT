# Design Choices - PCGT Configuration

<!-- 🧠 PHASE 4: DESIGN RATIONALE (Why Each Choice?) -->
<!--
Use this prompt when documenting design choices:

For each design decision, explain:

- Why was this decision made? (connect to patterns/failures)
- Alternatives considered
- Trade-offs between options
- Pros and cons of your choice
- Evidence supporting this choice
- What would break or fail?

Do NOT just list decisions. Teach why each matters.

Ask yourself:
- Is the rationale clear?
- Would a reader understand why you chose this over alternatives?
- What assumptions underlie this choice?

Reference: See 00_prompt_system.md Phase 4 for full guidance.
-->

## 1. Multi-Resolution Attention (Local + Global)

### Decision
Split attention into two components:
- **Local (fine-grained)**: Exact softmax within partitions → O(N²/K)
- **Global (coarse)**: Learned seed pooling across partitions → O(N·M·K)
- **Blend**: Learnable weight α per layer

### Alternatives Considered
1. **Pure local attention**: Faster but misses long-range dependencies
2. **Pure global attention**: Can lose fine-grained local patterns
3. **Global kernel trick** (SGFormer): Cheaper but less expressive (Softmax(QK^T) ≠ learned pooling)

### Trade-offs
| Approach | Complexity | Heterophily | Homophily |
|----------|-----------|-----------|-----------|
| Pure Local | O(N²/K) | ❌ Misses long-range | ✅ Respects communities |
| Pure Global | O(N²) or O(NMK) | ✅ Long-range | ⚠️ Over-smoothing |
| **PCGT Blend** | O(N²/K + NMK) | ✅ **+3.19, +3.34** | ✅ **+0.84, +0.16** |

### Why This Works
- **Heterophilic graphs**: α learns to weight global heavily, enabling non-neighbor connections
- **Homophilic graphs**: α balances both, respecting communities while allowing long-range
- **Empirical validation**: Chameleon (+3.19) and Squirrel (+3.34) confirm heterophilic gains

---

## 2. Learnable Self-Connection Weight (β)

### Decision
Make self-connection weight learnable, not fixed at 1.0.

```
out_i = β · feature_i + (1-β) · aggregated_neighbors_i
```

### Alternatives Considered
1. **Fixed β = 1.0**: Classical GCN (no learning)
2. **Fixed β = 0.5**: Balanced default
3. **β ∈ [0, 1]**: Constrained learning
4. **β ∈ ℝ**: Unconstrained learning (current)

### Trade-offs
| β Range | Pros | Cons | Use Case |
|---------|------|------|----------|
| Fixed 1.0 | Simple, stable | Cannot suppress self | Homophilic only |
| [0, 1] | Interpretable (convex) | May underfit heterophilic | Safe default |
| ℝ (unconstrained) | **Discovers negative β** | Unstable if extreme | Heterophilic |

### Empirical Behavior
- **Homophilic** (Cora, CiteSeer, PubMed): β ≈ 0.4–0.7 (moderate, non-edge-centric)
- **Heterophilic** (Chameleon, Squirrel): β ≈ 0.2–0.5 (lower, more edge-centric)
- **Extreme case** (Film): β ≈ -2.01 (learned strong self-suppression)

### Why This Works
- **Heterophilic signal**: Edges informative; self-suppression (β < 0) reduces feature bias
- **Flexibility**: Model autonomously decides whether to trust self vs neighbors
- **Validation**: Negative β discovery confirms PCGT learning sensible task-specific patterns

---

## 3. Partition-to-Seed Pooling (vs Standard Softmax)

### Decision
Learn K×M representative vectors (seeds) per layer. Global attention via:
```
global_out = softmax(Q @ pool_seeds) @ pool_seeds^T @ V
```

### Alternatives Considered
1. **Fixed pooling** (e.g., mean/sum): Deterministic, non-learnable
2. **Random projections**: Cheap but non-parametric
3. **Learned seeds** (current): Parametric, train with backprop
4. **Kernel trick** (SGFormer): Kernel-based approximation to softmax

### Trade-offs
| Method | Complexity | Expressiveness | Learnability |
|--------|-----------|---|---|
| Fixed mean | O(NK) | ❌ Limited | ❌ None |
| Random proj | O(NMK) | ⚠️ Fixed | ❌ None |
| **Learned seeds** | O(NMK) | ✅ **High** | ✅ **Backprop** |
| Kernel trick | O(NK) | ⚠️ Kernel space | ⚠️ Limited |

### Why This Works
- **Parameterization**: Seeds become problem-specific; model learns where to "attend globally"
- **Gradient flow**: Entire attention mechanism is differentiable
- **Empirical validation**: PCGT substantially outperforms SGFormer kernel trick on heterophilic graphs

---

## 4. GCN Backbone Blending

### Decision
Linearly interpolate PCGT output with GCN output via learnable `graph_weight`:
```
output = gw · GCN(X, A) + (1-gw) · PCGT(X, A)
```

### Alternatives Considered
1. **Pure PCGT**: Ignore classical message-passing
2. **Pure GCN**: Ignore partition attention
3. **Adaptive blending**: Different gw per layer (more parameters)
4. **Serial stacking**: PCGT → GCN (vs linear blend)

### Trade-offs
| Approach | Parameters | Homophily | Heterophily | Stability |
|----------|-----------|-----------|-----------|-----------|
| Pure PCGT | Lower | ⚠️ Moderate | ✅ Strong | ⚠️ Can overfit |
| Pure GCN | Lower | ✅ Stable | ❌ Weak | ✅ Robust |
| **Linear blend** | +1 per layer | ✅ **Best both** | ✅ **Best both** | ✅ **Stable** |
| Per-layer gw | 3× params | ⚠️ Overkill | ⚠️ Overkill | ❌ Overfit |

### Empirical Behavior
- **Standard config** (gw=0.5): Balanced, works across all datasets
- **GCN-heavy** (gw=0.9): Better on homophilic; can hurt heterophilic
- **PCGT-heavy** (gw=0.3): Better on heterophilic; can underfit homophilic
- **Pure GCN** (gw=1.0): Baseline for Deezer attack (what's the ceiling?)

### Why This Works
- **Unified framework**: Single architecture handles both paradigms
- **Interpretability**: gw shows which mechanism dominates per task
- **Robustness**: Blend prevents architecture from being too specialized
- **Practical**: Simple linear interpolation adds minimal computational cost

---

## 5. Partition Method Selection

### Decision
Default: **KMeans** (or METIS for graph-aware partitioning)
- KMeans: Fast, deterministic, feature-space clustering
- METIS: Slower but respects graph structure (minimizes edge cuts)

### Alternatives Considered
1. **Random**: Uniform random assignment (baseline ablation)
2. **Spectral**: Eigenvector-based partitioning
3. **Louvain**: Community detection
4. **Geometric**: k-d tree partitioning

### Trade-offs
| Method | Speed | Structure-Aware | Deterministic | Scalability |
|--------|-------|-----------------|---------------|-------------|
| Random | ✅ O(N) | ❌ No | ✅ Yes | ✅ Linear |
| KMeans | ✅ O(NK log K) | ⚠️ Feature-space | ✅ Yes (seeded) | ✅ Linear |
| **METIS** | ❌ O(N log N) | ✅ **Yes** | ✅ Yes | ⚠️ Slower |
| Spectral | ❌ O(N³) | ✅ Graph-aware | ✅ Yes | ❌ Expensive |

### Empirical Validation
- **Standard baseline**: KMeans K=50 (good trade-off)
- **Deezer attack** (B10): Random partition ablation → check if structure matters
- **Noted behavior**: METIS vs KMeans show minimal accuracy diff (both ~84% Cora); KMeans faster

### Why This Works
- **KMeans default**: Fast, stable across datasets, minimizes partition-size variance
- **METIS option**: Available for graph-structure-savvy partitioning
- **Random ablation**: Validates that learned structure (not random luck) drives gains

---

## 6. Standard Hyperparameter Choices

### Model Capacity
```
--hidden_channels 64    # 32–128 range; 64 balances expressiveness + memory
--num_layers 3          # 2–4; deeper = more aggregation, more overfitting
--ours_layers 3         # Typically equals num_layers
--num_reps 4            # 1, 2, 4; trade representatives ↔ global cost
```

### Rationale
- **hidden_channels=64**: Standard GNN choice; larger for image/large datasets; smaller for small graphs
- **num_layers=3**: Avoids over-smoothing (deep GNNs); 3 achieves good expressiveness
- **num_reps=4**: M=4 with K=50 → O(200N) global cost (acceptable linear factor)

### Regularization
```
--dropout 0.6           # Typical aggressive dropout (moderate to prevent overfitting)
--weight_decay 5e-4     # Standard L2; prevents weight explosion
--ours_dropout 0.6      # Same as dropout (could differ for ablation)
--ours_weight_decay 5e-4
```

### Why These Values
- **dropout=0.6**: Moderate-aggressive (overfitting observed on small graphs)
- **weight_decay=5e-4**: Classical value balancing generalization without underfit

### Deezer Attack Variants
- **B1–B3**: Reduce capacity (binary task → smaller model)
- **B4–B6**: Increase regularization (97% train overfitting → heavy dropout/wd)
- **B7–B8**: BatchNorm + graph_weight extremes (stabilize vs blend)
- **B11-B12**: Feature norm ablation + pure GCN ceiling

---

## 7. Training Procedure

### Learning Rate & Optimizer
```
--lr 0.01               # Adam optimizer, standard choice
--epochs 200            # Max; usually converge earlier
--early_stopping 50     # Stop after 50 inactive epochs
```

### Why These Values
- **lr=0.01**: Standard for GNNs; allows convergence without divergence
- **epochs=200**: Conservative upper bound; early stopping activates ~epoch 50–100
- **early_stopping=50**: Prevents overfitting on small graphs (Cora converges by epoch 100)

### Loss Function
```
CrossEntropyLoss(logits, labels)  # Standard for classification
```

### No Auxiliary Losses
- No contrastive learning (unlike BGRL)
- No knowledge distillation
- No self-supervised pretraining
- **Simplicity choice**: Validate pure supervised PCGT vs SGFormer baseline

---

## Summary Table

| Design | Choice | Why | Evidence |
|--------|--------|-----|----------|
| Attention | Multi-resolution (local + global) | Handles both homophilic + heterophilic | Chameleon +3.19, CiteSeer +0.84 |
| Self-weight | Learnable β (unconstrained) | Discover task-specific self-importance | Film β=-2.01 (heterophilic pattern) |
| Pooling | Learned seeds | Parametric global attention | > SGFormer kernel trick |
| Blending | Linear GCN interpolation | Unified framework for both paradigms | Works on all 7 datasets |
| Partition | KMeans (default) + METIS option | Speed + structure awareness | Consistent 83%–85% Cora |
| Model | hidden=64, layers=3, reps=4 | Balance capacity + memory + speed | Standard GNN recipe |
| Regularization | dropout=0.6, wd=5e-4 | Prevent overfitting on small graphs | Effective on 7 datasets |
| Training | Adam, lr=0.01, early_stopping=50 | Stable convergence + prevent overfit | Cora ~epoch 50–100 |
