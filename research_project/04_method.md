# PCGT: Partition-Conditioned Graph Transformer

<!-- 🧠 PHASE 4: METHOD FORMATION (Formalization) -->
<!--
Use this prompt when writing the method section:

Based on discovered patterns (from 08_analysis.md), define the method:

- What is the core insight?
- Step-by-step process (each step explains one insight)
- Why does each step exist? (connect to failure/success patterns)
- What problem does each step solve?
- Assumptions behind each step (list explicitly)
- Edge cases where it might fail

Explain like teaching someone who will build it from scratch.

Ask yourself:
- Is each design choice justified?
- Could someone understand this without knowing the paper?
- Where are you least confident?

Reference: See 00_prompt_system.md Phase 4 for full guidance.
-->

## Overview
PCGT is a Graph Neural Network that combines **partition-aware attention** with **GCN backbone blending** to handle both homophilic and heterophilic graphs. The core innovation is multi-resolution attention: exact softmax within partitions (fine-grained local patterns) + learned seed-based pooling for global context.

---

## Architecture

### 1. Graph Partitioning
**Goal**: Divide N nodes into K disjoint partitions to reduce attention complexity.

**Methods**:
- **METIS** (graph-aware, spectral): Minimizes edge cuts (preserves local structure)
- **KMeans** (spatial): Cluster nodes in feature space
- **Random** (baseline): Uniform random assignment

**Output**: 
- `partition_indices` (node → partition_id mapping)
- `partition_labels` (one-hot encoding, shape N×K)

### 2. Partition Structural Encoding (PSE)
**Goal**: Encode which partition each node belongs to.

**Implementation**:
```
partition_pe = Embedding(K, D) 
                           ↓
node_embedding += partition_pe[partition_id[i]]
```

- K = number of partitions (typically 20–50)
- D = hidden dimension (typically 64)
- Added per layer (adaptively reweighted)

### 3. Multi-Resolution Attention (MRA)

#### Standard Attention (Baseline)
```
Attention(Q, K, V) = softmax(QK^T / √d) V
Complexity: O(N²) globally
```

#### PCGT Multi-Resolution Attention
Split into two components:

**A) Local (Fine-Grained) Attention**
- Exact softmax within each partition only
- Nodes in partition p attend to nodes in partition p
- Complexity: O(N²/K) where K = num_partitions
- Captures local/fine-grained patterns

**B) Global (Coarse) Attention via Learned Pooling**
- Each partition p computes M representative vectors (learned seed vectors)
- pool_seeds ∈ ℝ^(K×M×D): learnable seeds per partition
- Cross-partition attention: all nodes attend to all partition-representatives
- Complexity: O(N·M·K) where M = num_reps (typically 4)
- Captures global/long-range patterns

**Combined Output**:
```
out = α · local_attention + (1-α) · global_attention
α = sigmoid(α_logit)  # learned blending weight, per layer
```

### 4. Self-Connection Adaptation
**Goal**: Learn whether to amplify or suppress self-loops (important for heterophilic graphs).

**Implementation**:
```
self_connection_weight = β  # learnable, per layer
out_i = β · feature_i + (1-β) · attention_output_i
```

- Homophilic graphs: β ≈ 0.5–0.7 (moderate self-weighting)
- Heterophilic graphs: β can become negative (learned self-suppression)
  - Example: Film dataset, β ≈ -2.01 (strong self-suppression)

### 5. GCN Backbone Blending
**Goal**: Combine partition-aware attention with classical GCN to handle both paradigms.

**Architecture**:
```
output = graph_weight · GCN_output + (1 - graph_weight) · PCGT_output
graph_weight ∈ [0, 1]  # learnable or fixed per layer
```

- **graph_weight = 1.0**: Pure GCN (no PCGT)
- **graph_weight = 0.5**: Balanced blend
- **graph_weight = 0.0**: Pure PCGT (no GCN)

**GCN Branch**: Standard `AX̃` where Ã is normalized adjacency.

### 6. Full Layer Description

```python
# PCGTConvLayer(in_dim, out_dim, num_partitions, num_reps, ...)

input: X ∈ ℝ^(N×D_in)
       partition_labels ∈ ℝ^(N×K)

# Step 1: Add partition structural encoding
partition_pe = PartitionEmbedding(K, D_in)
X = X + partition_pe(partition_labels)

# Step 2: Compute Q, K, V projections
Q = Linear(D_in, D_out)(X)
K = Linear(D_in, D_out)(X)
V = Linear(D_in, D_out)(X)

# Step 3: Local attention (intra-partition)
Attn_local = []
for partition p in range(K):
    mask = (partition_labels[:, p] == 1)
    Attn_local.append(softmax(Q[mask] @ K[mask].T / √d) @ V[mask])
local_out = concatenate(Attn_local)  # O(N²/K)

# Step 4: Global attention (via learned seeds)
pool_seeds ∈ ℝ^(K×M×D_out)  # learnable
global_representative = pool(V, pool_seeds, weights)  # K×M×D_out
Attn_global = softmax(Q @ global_representative.T / √d) @ global_representative  # O(N·M·K)

# Step 5: Blend local + global
α = sigmoid(α_logit)
attention_out = α · local_out + (1-α) · global_out

# Step 6: Add self-connection weighting
β_weighted = β · X + (1-β) · attention_out

# Step 7: GCN blending
gcn_out = GCN(X, graph_weight)
output = graph_weight · gcn_out + (1 - graph_weight) · β_weighted
output = LayerNorm(output + X)  # residual connection

# Step 8: MLP readout
output = Dropout(output)
output = Linear(D_out, D_out)(output)
output = ReLU(output)
output = Dropout(output)
```

---

## Training Procedure

### Objective
Minimize classification loss:
```
L = CrossEntropyLoss(model(X, A), y_train)
```

### Hyperparameter Groups

**Model Capacity**:
- `hidden_channels`: Dimension of hidden layers (32, 48, 64)
- `num_layers`: Number of PCGT layers (2–4)
- `ours_layers`: Number of transformer layers (same or different)
- `num_reps`: Representatives per partition (1, 2, 4)

**Regularization**:
- `dropout`: Attention dropout (0.5–0.8)
- `weight_decay`: L2 regularization (1e-5 to 1e-2)
- `ours_dropout`: Transformer-specific dropout
- `ours_weight_decay`: Transformer-specific weight decay

**Graph Partition**:
- `partition_method`: METIS, KMeans, random
- `num_partitions`: Number of partitions (5–50)

**Optimization**:
- `lr`: Learning rate (0.001–0.01)
- `optimizer`: Adam (default)
- `epochs`: Max epochs (200–500)
- `early_stopping`: Stop after N inactive epochs (50–100)

---

## Why PCGT Works

### For Homophilic Graphs
- Partition attention respects natural community structure
- GCN blend (gw=0.5) provides classical message-passing baseline
- Self-weighting β ≈ 0.5–0.7 maintains self-connections
- Local attention captures tight node clusters

### For Heterophilic Graphs
- Global attention (via seeds) allows non-neighboring nodes to "see" each other
- Learned α blending prioritizes global context over local
- Negative β discovered automatically (self-suppression)
- Multi-resolution avoids over-smoothing from deep aggregation
- Examples:
  - **Chameleon** (h=0.46): +3.19 vs SGFormer (multi-resolution > kernel trick)
  - **Squirrel** (h=0.22): +3.34 vs SGFormer (strong heterophilic signal)

### Complexity Analysis
| Component | Complexity | Notes |
|-----------|-----------|-------|
| Local attention | O(N²/K) | K partitions reduce cost |
| Global attention | O(N·M·K) | M=4, K=50 → O(200N) ≈ linear |
| GCN baseline | O(\|E\|·D) | Sparse adjacency |
| Total per-layer | O(N²/K + NMK) | Subquadratic for K>√N |

---

## Implementation Details

### Code Files
- **pcgt.py**: Core classes (PCGTConvLayer, PCGTConv, PCGT)
- **models.py**: Model factory (PCGT + other baselines)
- **main.py**: Training loop + evaluation
- **data_utils.py**: Dataset loading + partitioning
- **parse.py**: Argument parser for all hyperparameters

### Device Support
- GPU (CUDA): Primary (Colab GPUs T4/A100)
- CPU: Fallback for debugging (slower)
- Auto-detection: `torch.cuda.is_available()`
- Flag override: `--cpu` to force CPU mode

### Reproducibility
- Fixed random seeds for all experiments (per `--runs N`)
- METIS produces deterministic partitions
- Dropout only active during training, disabled at test time
- Early stopping checkpoint preserves best model
