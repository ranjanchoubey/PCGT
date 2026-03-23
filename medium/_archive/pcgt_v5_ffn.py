"""
PCGT v5: Partition-Conditioned Graph Transformer — Proper Transformer Block

First-principles redesign fixing 5 fundamental issues in v4:

  1. ADD FFN BLOCK after attention (standard Transformer: Attn → FFN)
     v4 has no FFN — severely limits representational power.

  2. PER-NODE α via GATING NETWORK (not a single frozen scalar)
     v4's sigmoid(0.0) = 0.5 never moves.  Replace with:
       α_i = σ(W_gate · x_i)   — node-dependent local/global blend.

  3. CONCAT HEADS instead of averaging before combine
     v4 averages heads BEFORE mixing local/global → destroys head diversity.
     v5 concatenates heads, then projects.

  4. PRE-NORM layout (like GPT-2, modern Transformers)
     v4 does: Attn → residual → LN → dropout.
     v5 does: LN → Attn → residual → LN → FFN → residual.
     Proven more stable for training.

  5. LEARNABLE GRAPH FUSION via gating (not static graph_weight)
     v4: fixed scalar blend.
     v5: g_i = σ(W_fuse · [x_gcn; x_pcgt]) — per-node adaptive fusion.

Architecture per layer (Pre-Norm Transformer Block):
  x → LN → MultiResAttn → + residual
    → LN → FFN (up → GELU → down) → + residual

MultiResAttn:
  Q, K, V (multi-head)
  ├─ IntraPartition softmax attention ── h_local  [N, H*D]
  ├─ Seed-pooled reps → Cross-attention ── h_global [N, H*D]
  └─ Gate: α_i = σ(W_gate · x_i) ∈ [0,1] per node
     out = α_i * h_local + (1-α_i) * h_global

Fusion:
  g_i = σ(W_fuse · [x_gcn_i ; x_pcgt_i])
  x_i = g_i * x_gcn_i + (1 - g_i) * x_pcgt_i
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """Feed-Forward Network: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult)
        self.w2 = nn.Linear(dim * mult, dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.w1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.w2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MultiResAttention(nn.Module):
    """Multi-Resolution Partition Attention with:
    - Multi-head Q/K/V with head concatenation (not averaging)
    - Learned pool seeds for partition representatives
    - Per-node gating for local/global blend
    """

    def __init__(self, dim, num_heads=1, num_reps=4, attn_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_reps = num_reps
        self.scale = math.sqrt(self.head_dim)
        self.attn_dropout = attn_dropout

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)  # output projection after head concat

        # Learned pool seeds: [M, H, head_dim]
        self.pool_seeds = nn.Parameter(
            torch.randn(num_reps, num_heads, self.head_dim) * 0.02)

        # Per-node gating: projects node features to scalar α per node
        self.gate = nn.Linear(dim, 1)

    def reset_parameters(self):
        for module in [self.Wq, self.Wk, self.Wv, self.Wo, self.gate]:
            module.reset_parameters()
        nn.init.normal_(self.pool_seeds, std=0.02)

    def forward(self, x, partition_indices):
        N = x.size(0)
        H = self.num_heads
        D = self.head_dim
        M = self.num_reps

        Q = self.Wq(x).reshape(N, H, D)
        K = self.Wk(x).reshape(N, H, D)
        V = self.Wv(x).reshape(N, H, D)

        if partition_indices is None:
            # Fallback: no partitions, just return projected values
            out = V.reshape(N, H * D)
            return self.Wo(out)

        num_parts = len(partition_indices)

        # ─── LOCAL: exact intra-partition attention ───
        out_local = torch.zeros(N, H, D, device=x.device)

        # ─── BUILD REPRESENTATIVES ───
        reps_k = torch.zeros(num_parts * M, H, D, device=x.device)
        reps_v = torch.zeros(num_parts * M, H, D, device=x.device)

        for p, indices in enumerate(partition_indices):
            q_p = Q[indices]   # [n_p, H, D]
            k_p = K[indices]
            v_p = V[indices]

            # Intra-partition exact softmax attention
            attn = torch.einsum('phd,qhd->hpq', q_p, k_p) / self.scale
            attn = F.softmax(attn, dim=-1)
            if self.training and self.attn_dropout > 0:
                attn = F.dropout(attn, p=self.attn_dropout, training=True)
            out_local[indices] = torch.einsum('hpq,qhd->phd', attn, v_p)

            # Attention-pooled representatives
            pool_attn = torch.einsum('mhd,nhd->mhn', self.pool_seeds, k_p)
            pool_attn = pool_attn / self.scale
            pool_attn = F.softmax(pool_attn, dim=-1)
            reps_k[p*M:(p+1)*M] = torch.einsum('mhn,nhd->mhd', pool_attn, k_p)
            reps_v[p*M:(p+1)*M] = torch.einsum('mhn,nhd->mhd', pool_attn, v_p)

        # ─── GLOBAL: cross-partition attention ───
        cross_attn = torch.einsum('nhd,rhd->nhr', Q, reps_k) / self.scale
        cross_attn = F.softmax(cross_attn, dim=-1)
        if self.training and self.attn_dropout > 0:
            cross_attn = F.dropout(cross_attn, p=self.attn_dropout, training=True)
        out_global = torch.einsum('nhr,rhd->nhd', cross_attn, reps_v)

        # ─── PER-NODE GATING (not a single scalar) ───
        alpha = torch.sigmoid(self.gate(x))  # [N, 1]

        # Concat heads (not average), then gate and project
        h_local = out_local.reshape(N, H * D)    # [N, dim]
        h_global = out_global.reshape(N, H * D)  # [N, dim]

        h = alpha * h_local + (1 - alpha) * h_global  # [N, dim]
        return self.Wo(h)


class PCGTBlock(nn.Module):
    """Pre-Norm Transformer Block: LN → MultiResAttn → residual → LN → FFN → residual."""

    def __init__(self, dim, num_heads=1, num_reps=4, ffn_mult=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiResAttention(dim, num_heads, num_reps, attn_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mult=ffn_mult, dropout=dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.attn.reset_parameters()
        self.norm2.reset_parameters()
        for p in self.ffn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, partition_indices):
        # Pre-norm attention
        x = x + F.dropout(self.attn(self.norm1(x), partition_indices),
                          p=self.dropout, training=self.training)
        # Pre-norm FFN
        x = x + F.dropout(self.ffn(self.norm2(x)),
                          p=self.dropout, training=self.training)
        return x


class PCGTConv(nn.Module):
    """Stack of PCGTBlocks with input projection and PSE."""

    def __init__(self, in_channels, hidden_channels, num_layers=1, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_act=False, num_partitions=10,
                 num_reps=4):
        super().__init__()
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)

        # Partition Structural Encoding
        self.partition_pe = nn.Embedding(num_partitions, hidden_channels)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PCGTBlock(hidden_channels, num_heads, num_reps, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.input_norm.reset_parameters()
        nn.init.normal_(self.partition_pe.weight, std=0.02)
        for block in self.blocks:
            block.reset_parameters()
        self.final_norm.reset_parameters()

    def forward(self, data, partition_indices, partition_labels=None):
        x = data.graph['node_feat']

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Add partition structural encoding
        if partition_labels is not None:
            x = x + self.partition_pe(partition_labels)

        # Transformer blocks (each has its own residual connections)
        for block in self.blocks:
            x = block(x, partition_indices)

        x = self.final_norm(x)
        return x


class PCGT_V5(nn.Module):
    """PCGT v5: Proper Transformer + Adaptive Fusion.

    Changes from v4:
    1. FFN after attention (standard Transformer block)
    2. Per-node α gating (not frozen scalar)
    3. Head concatenation (not averaging)
    4. Pre-norm layout
    5. Learnable per-node fusion with GCN branch
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=1, num_heads=1, alpha=0.5, dropout=0.5,
                 use_bn=True, use_residual=True, use_weight=True,
                 use_graph=True, use_act=False,
                 graph_weight=0.8, gnn=None, aggregate='add',
                 num_partitions=10, num_reps=4):
        super().__init__()

        self.pcgt_conv = PCGTConv(
            in_channels, hidden_channels, num_layers, num_heads,
            alpha, dropout, use_bn, use_residual, use_weight, use_act,
            num_partitions=num_partitions, num_reps=num_reps)

        self.gnn = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight  # fallback, used if aggregate='add' without gating

        self.aggregate = aggregate
        if aggregate == 'add':
            # Learnable per-node fusion gate
            self.fusion_gate = nn.Linear(hidden_channels * 2, 1)
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type: {aggregate}')

        self.partition_indices = None
        self.partition_labels = None

        # Parameter groups for differential learning rates
        self.params1 = list(self.pcgt_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))
        if aggregate == 'add' and use_graph:
            self.params2.extend(list(self.fusion_gate.parameters()))

    def set_partition_info(self, partition_indices, partition_labels):
        device = next(self.parameters()).device
        self.partition_indices = [idx.to(device) for idx in partition_indices]
        self.partition_labels = torch.LongTensor(partition_labels).to(device)

    def forward(self, data):
        x1 = self.pcgt_conv(data, self.partition_indices, self.partition_labels)

        if self.use_graph:
            x2 = self.gnn(data)
            if self.aggregate == 'add':
                # Learnable per-node fusion
                g = torch.sigmoid(self.fusion_gate(torch.cat([x1, x2], dim=-1)))
                x = g * x2 + (1 - g) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1

        x = self.fc(x)
        return x

    def reset_parameters(self):
        self.pcgt_conv.reset_parameters()
        if self.use_graph and self.gnn is not None:
            self.gnn.reset_parameters()
        self.fc.reset_parameters()
        if hasattr(self, 'fusion_gate'):
            self.fusion_gate.reset_parameters()

    def get_gamma_values(self):
        """Return per-layer attention gate stats for monitoring."""
        vals = []
        for block in self.pcgt_conv.blocks:
            # Report mean gate value (would need forward pass data for actual per-node stats)
            gate_bias = block.attn.gate.bias.item() if block.attn.gate.bias is not None else 0
            gate_alpha = torch.sigmoid(torch.tensor(gate_bias)).item()
            vals.append(f"α≈{gate_alpha:.2f}")
        return vals
