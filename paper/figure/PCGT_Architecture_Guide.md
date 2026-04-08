# PCGT Architecture — Detailed Drawing Guide

Use this as a visual reference to draw in draw.io. Every box, arrow, and label is described.

---

## OVERALL LAYOUT (Left to Right)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  INPUT STAGE          PCGT ATTENTION LAYER                              OUTPUT STAGE    │
│  ───────────         ─────────────────────────                          ────────────    │
│  Graph + Feat   →    PSE → Q,K,V → [Local + Global] → Blend → Self → LN → Fusion → ŷ │
│                                                                           ↑            │
│                      GCN Branch ─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## SECTION 1: INPUT (leftmost)

**Box: "Input Graph"**
- Color: Light blue
- Content: `X ∈ ℝ^{N×d_in}`, `A ∈ ℝ^{N×N}`
- Shape: Rounded rectangle
- Also show a small graph icon (5-6 nodes with edges)

**Arrow →**

**Box: "METIS Partition"**
- Color: Light purple
- Content: `→ K disjoint sets S₁...Sₖ`
- Show nodes colored by partition (e.g., 3 colors for K=3)
- This is a PREPROCESSING step (dashed border to indicate "offline")

---

## SECTION 2: PCGT ATTENTION LAYER (main block, enclosed in dashed blue box)

### Step 2a: Partition Structural Encoding (PSE)

**Box: "+ PSE"**
- Color: Light blue
- Content: `xᵢ ← xᵢ + PE(kᵢ)`, `PE: K×D embedding table`
- NOTE: `PE` is a learned `nn.Embedding(K, hidden_dim)`
- Each node gets its partition ID's embedding added

**Arrow →**

### Step 2b: Input Projection

**Box: "Linear Projection"**  
- Color: Light blue
- Content: `x ← Linear(x)` then `LayerNorm → ReLU → Dropout`
- This is `self.fcs[0]` in the code

**Arrow →**

### Step 2c: Q, K, V Projection

**Box: "Q, K, V"**
- Color: Light blue  
- Content: `Q = Wq·x`, `K = Wk·x`, `V = Wv·x`
- Each is `nn.Linear(hidden, hidden * num_heads)`
- Reshape to `[N, H, D]` (H heads, D dim per head)

**Arrow splits into TWO parallel branches (fork) →**

---

### BRANCH A (TOP): LOCAL ATTENTION

**Box: "Local Attention"**
- Color: Deeper blue (e.g., blue!20)
- Content: `Per partition p:`
  ```
  attn_p = softmax(Q[Sp] · K[Sp]ᵀ / √d)
  out_local[Sp] = attn_p · V[Sp]
  ```
- Cost label: `O(N²/K)`
- Key detail: This is EXACT softmax, NOT approximate
- **Inside the same for-loop** as seed pooling (shown below)

### BRANCH B (BOTTOM): GLOBAL ATTENTION (two sub-steps)

**Sub-box B1: "Seed Pooling"**
- Color: Cyan/teal
- Content: `For each partition p:`
  ```
  pool_attn = softmax(Seeds · K[Sp]ᵀ / √d)    ← Seeds: [M,H,D] learned
  Rk[p] = pool_attn · K[Sp]                     ← M representative keys
  Rv[p] = pool_attn · V[Sp]                     ← M representative values
  ```
- Show: `Seeds` as a small learned parameter block (M=4 vectors)
- Stack across all K partitions → `Rk, Rv ∈ ℝ^{KM × H × D}`

**Arrow →**

**Sub-box B2: "Global Cross-Attention"**
- Color: Cyan/teal
- Content:
  ```
  cross_attn = softmax(Q · Rk.ᵀ / √d)    [N, H, K*M]
  out_global = cross_attn · Rv             [N, H, D]
  ```
- Cost label: `O(N·M·K)`
- Key: EVERY node attends to ALL K×M representatives

---

### MERGE: α-BLEND

**Box: "α-Blend"**
- Color: Orange
- Content:
  ```
  α = sigmoid(α_logit)     ← learnable scalar, init 0.0
  x_context = α · x_local + (1-α) · x_global
  ```
- Both branches merge here
- α is a single scalar (not per-node, not per-head)

**Arrow →**

### SELF-CONNECTION

**Box: "+ β · V"**
- Color: Orange
- Content:
  ```
  x_out = x_context + β · Vᵢ    ← β: learnable scalar, init 1.0
                                    β is UNCONSTRAINED (can go negative!)
  ```
- Key: β goes negative on heterophilic graphs (e.g., β≈-2.0 on Film)
- V here is `V.mean(dim=1)` — the node's own value, averaged over heads

**Arrow →**

### RESIDUAL + LAYERNORM

**Box: "LayerNorm + Residual"**
- Color: Light blue
- Content:
  ```
  x = α_res · x_out + (1-α_res) · x_prev    ← α_res=0.5 fixed residual
  x = LayerNorm(x)
  x = Dropout(x)
  ```
- NOTE: The code uses a fixed `self.alpha` (default 0.5) for residual mixing, NOT sigmoid
- LayerNorm (not BatchNorm) is used

---

## SECTION 3: GCN BRANCH (runs in parallel, shown BELOW the main attention path)

**Box: "GCN Branch"**
- Color: Green
- Content:
  ```
  GCN(X, A) — standard multi-layer GCN
  x₂ = GCN_L₂(...GCN_L₁(X, edge_index)...)
  Uses residual connections + BatchNorm
  ```
- The GCN takes ORIGINAL features X and adjacency A
- Number of layers: L_gcn (2 or 4 depending on dataset)
- Uses PyG `GCNConv` layers

---

## SECTION 4: FUSION + OUTPUT (rightmost)

**Box: "λ_gw Fusion"**
- Color: Red/salmon
- Content:
  ```
  x = λ_gw · x_GCN + (1 - λ_gw) · x_PCGT
  ```
- λ_gw is a FIXED hyperparameter (not learned), typically 0.5-0.8
- Higher λ_gw = more GCN weight (used on homophilic graphs)

**Arrow →**

**Box: "Classifier"**
- Color: Yellow
- Content: `ŷ = Linear(x)` → `[N, C]` logits
- Single linear layer, no activation

---

## COMPLETE DATA FLOW (for the technical person)

```
Input: X ∈ ℝ^{N×d_in}, edge_index ∈ ℝ^{2×E}

Preprocessing (once):
  partition_labels = METIS(edge_index, K)
  partition_indices = [nodes in partition p, for p in 0..K-1]

Forward pass:
  ┌─ PCGT Branch ─────────────────────────────────────────────┐
  │  x = Linear(X)             # [N, d_in] → [N, hidden]     │
  │  x = LayerNorm(x)                                        │
  │  x = ReLU(x)                                             │
  │  x = Dropout(x, p)                                       │
  │  x = x + PSE(partition_labels)  # add partition embedding │
  │                                                           │
  │  ┌─ PCGTConvLayer (repeated L_pcgt times) ──────────────┐ │
  │  │  Q = Wq(x).reshape(N,H,D)                           │ │
  │  │  K = Wk(x).reshape(N,H,D)                           │ │
  │  │  V = Wv(x).reshape(N,H,D)                           │ │
  │  │                                                      │ │
  │  │  for each partition p:                               │ │
  │  │    # Local attention                                 │ │
  │  │    attn = softmax(Q[Sp]·K[Sp]ᵀ / √d)   [H,np,np]  │ │
  │  │    out_local[Sp] = attn · V[Sp]                     │ │
  │  │                                                      │ │
  │  │    # Seed pooling                                    │ │
  │  │    pool = softmax(seeds·K[Sp]ᵀ / √d)    [M,H,np]   │ │
  │  │    Rk[p] = pool · K[Sp]                 [M,H,D]    │ │
  │  │    Rv[p] = pool · V[Sp]                 [M,H,D]    │ │
  │  │                                                      │ │
  │  │  # Global cross-attention                            │ │
  │  │  cross = softmax(Q·Rk_allᵀ / √d)       [N,H,K*M]  │ │
  │  │  out_global = cross · Rv_all            [N,H,D]    │ │
  │  │                                                      │ │
  │  │  x_local  = out_local.mean(dim=1)       [N,D]      │ │
  │  │  x_global = out_global.mean(dim=1)      [N,D]      │ │
  │  │  x_self   = V.mean(dim=1)               [N,D]      │ │
  │  │                                                      │ │
  │  │  α = sigmoid(α_logit)                               │ │
  │  │  x = α·x_local + (1-α)·x_global + β·x_self        │ │
  │  │                                                      │ │
  │  │  x = 0.5·x + 0.5·x_prev   # residual              │ │
  │  │  x = LayerNorm(x)                                   │ │
  │  │  x = Dropout(x)                                     │ │
  │  └─────────────────────────────────────────────────────┘ │
  │  x1 = x                        # [N, hidden]            │
  └──────────────────────────────────────────────────────────┘

  ┌─ GCN Branch ──────────────────────────────────────────────┐
  │  x = GCN_layer1(X, edge_index)   # raw features          │
  │  x = BatchNorm(x)                                        │
  │  x = ReLU(x)                                             │
  │  x = Dropout(x)                                          │
  │  ...repeat L_gcn layers...                                │
  │  x2 = GCN_layerL(x, edge_index)  # [N, hidden]          │
  └──────────────────────────────────────────────────────────┘

  # Fusion
  x = λ_gw · x2 + (1 - λ_gw) · x1

  # Classifier
  ŷ = Linear(x)                       # [N, num_classes]
```

---

## DIMENSIONS CHEAT SHEET

| Symbol | Meaning | Typical values |
|--------|---------|---------------|
| N | Number of nodes | 890 - 169K |
| d_in | Input feature dim | 500 - 8415 |
| hidden | Hidden dim | 64 (medium), 256 (arxiv) |
| H | Attention heads | 1 |
| D | Dim per head | = hidden/H = hidden |
| K | Number of partitions | 5 - 256 |
| M | Seeds per partition | 4 |
| L_pcgt | PCGT attention layers | 1 |
| L_gcn | GCN backbone layers | 2 or 4 |
| C | Number of classes | 2 - 40 |

---

## COLOR SCHEME FOR DRAW.IO

| Component | Suggested Color | Hex |
|-----------|----------------|-----|
| Input/Output | Light yellow | #FFF2CC |
| PCGT Attention (main) | Light blue | #DAE8FC |
| Local Attention | Medium blue | #B4C7E7 |
| Global Attention / Seeds | Teal/Cyan | #D5E8D4 |
| Blend / Self-conn | Orange | #FFE6CC |
| GCN Branch | Green | #D5E8D4 |
| Fusion | Salmon/Red | #F8CECC |
| Preprocessing | Light purple (dashed border) | #E1D5E7 |
| Learnable params (α,β,seeds) | Bold border or star icon | — |

---

## WHAT MAKES THIS DIAGRAM DIFFERENT FROM THE CURRENT TIKZ

The current TikZ is **correct but simplified**. For the draw.io version, add:

1. **Show the for-loop**: Local attention AND seed pooling happen inside the SAME loop over partitions
2. **Show tensor shapes**: Add `[N,H,D]`, `[M,H,D]`, `[KM,H,D]` at each arrow
3. **Show the head averaging**: After attention, `x = x.mean(dim=1)` averages over heads
4. **Show the input projection**: Before PSE, there's a `Linear → LayerNorm → ReLU → Dropout`
5. **Show seeds as a parameter**: Draw the `pool_seeds` as a small learned parameter box with `[M,H,D]`
6. **Highlight what's learned**: α_logit, β, PSE embedding, pool_seeds, Wq/Wk/Wv all have gradients
7. **Show residual**: The fixed 0.5 residual mixing after the attention layer

---

## STEP-BY-STEP DRAW.IO INSTRUCTIONS

1. **Canvas**: Set to landscape, ~1200×500px
2. **Draw Input box** (left) with graph icon
3. **Draw METIS box** with partition coloring → arrow from Input
4. **Draw large dashed rectangle** labeled "PCGT Attention Layer" containing steps 2a-2f
5. **Inside**: PSE → Input Proj → Q,K,V → fork
6. **Top fork**: Local Attention box (blue)
7. **Bottom fork**: Seed Pool box → Global Cross-Attention box (teal)
8. **Merge**: α-Blend box (orange) ← arrows from both branches
9. **After merge**: + β·V box → LayerNorm box
10. **Below main path**: GCN Branch (green, separate row)
11. **Rightmost**: λ_gw Fusion box ← arrows from LayerNorm AND GCN
12. **Output**: Classifier → ŷ
13. **Add tensor shape annotations** on key arrows
14. **Add "Learned" badges** to α, β, seeds, PSE
