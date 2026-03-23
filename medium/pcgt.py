"""
PCGT: Partition-Conditioned Graph Transformer (v4)
Multi-Resolution Partition Attention — a novel attention mechanism for graphs.

Key idea: Graph partitions define a natural two-level hierarchy (nodes and
communities). We build attention that operates at BOTH resolutions:

  FINE (Local):   Exact softmax attention within each partition.
                  Captures detailed intra-community patterns.
                  Cost: O(N²/K)

  COARSE (Global): Learnable "seed" vectors attention-pool each partition
                  into M representative vectors. Each node then cross-attends
                  to K×M partition representatives for global context.
                  Cost: O(NKM) = O(N) since K,M are small constants.

  Total: O(N²/K + NKM) — subquadratic, topology-aware.

This is NOT an add-on to SGFormer. It completely replaces the attention
mechanism. No kernel approximation is used.

Why this differs from prior work:
  - SGFormer:       Single-resolution O(N) kernel trick (lossy)
  - NodeFormer:     Gumbel-softmax kernelized attention (single resolution)
  - NAGphormer:     Hop-based tokenization (no partition structure)
  - Set Transformer: Learned inducing points (not topology-aware)
  - PCGT:           Two-resolution attention via graph topology
                    with LEARNED partition representatives (not mean pooling)

Architecture per layer:
  x → Q, K, V
      ├─→ IntraPartitionAttn(Q,K,V) per partition ──── x_local
      │
      ├─→ seeds · K[p] → attn → weighted V[p] ─→ reps (K×M vectors)
      └─→ CrossAttn(Q, reps_k, reps_v) ────────────── x_global
                                                        │
      α * x_local + (1-α) * x_global + β * x_self    ◄───┘
      α = sigmoid(learnable scalar)
      β = learnable self-connection weight

  + Partition Structural Encoding (learnable embeddings per partition)
  + GCN branch for edge-level structural features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCGTConvLayer(nn.Module):
    """Multi-Resolution Partition Attention Layer.

    Fine resolution:  Exact softmax attention within each partition  O(N²/K)
    Coarse resolution: Attention-pooled partition representatives + cross-attn
                      M learned seeds per partition → K×M representatives
                      Each node cross-attends to representatives → O(NKM)
    Combined via learnable scalars α (local vs global) and β (self-connection).
    """

    def __init__(self, in_channels, out_channels, num_heads=1,
                 use_weight=True, attn_dropout=0.0, num_reps=4,
                 local_only=False, global_only=False):
        super().__init__()
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        # Learnable pool seeds: each extracts a different "aspect" from partitions
        # Shape: [M, H, D] — M seeds, each a query vector per head
        self.pool_seeds = nn.Parameter(
            torch.randn(num_reps, num_heads, out_channels) * 0.02)

        self.num_reps = num_reps
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.scale = math.sqrt(out_channels)
        self.attn_dropout = attn_dropout

        # Ablation flags
        self.local_only = local_only
        self.global_only = global_only

        # α blends local (intra-partition) and global (cross-partition)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        # β: learnable self-connection weight
        self.beta = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()
        nn.init.normal_(self.pool_seeds, std=0.02)
        nn.init.constant_(self.alpha_logit, 0.0)
        nn.init.constant_(self.beta, 1.0)

    def forward(self, x, partition_indices):
        N = x.size(0)
        H = self.num_heads
        D = self.out_channels
        M = self.num_reps

        Q = self.Wq(x).reshape(N, H, D)
        K = self.Wk(x).reshape(N, H, D)
        if self.use_weight:
            V = self.Wv(x).reshape(N, H, D)
        else:
            V = x.reshape(N, 1, D).expand(N, H, D)

        if partition_indices is None:
            return V.mean(dim=1)

        num_parts = len(partition_indices)

        # ─── LOCAL: exact intra-partition attention ───
        out_local = torch.zeros(N, H, D, device=x.device)

        # ─── BUILD REPRESENTATIVES: attention-pooled, M per partition ───
        reps_k = torch.zeros(num_parts * M, H, D, device=x.device)
        reps_v = torch.zeros(num_parts * M, H, D, device=x.device)

        for p, indices in enumerate(partition_indices):
            q_p = Q[indices]  # [n_p, H, D]
            k_p = K[indices]
            v_p = V[indices]

            # Exact softmax attention within this partition
            attn_local = torch.einsum('phd,qhd->hpq', q_p, k_p) / self.scale
            attn_local = F.softmax(attn_local, dim=-1)
            if self.training and self.attn_dropout > 0:
                attn_local = F.dropout(attn_local, p=self.attn_dropout,
                                       training=True)
            out_local[indices] = torch.einsum('hpq,qhd->phd', attn_local, v_p)

            # Attention-pooled representatives using learned seeds
            # pool_seeds: [M,H,D] queries, k_p: [n_p,H,D] keys
            pool_attn = torch.einsum('mhd,nhd->mhn', self.pool_seeds, k_p)
            pool_attn = pool_attn / self.scale
            pool_attn = F.softmax(pool_attn, dim=-1)  # [M, H, n_p]

            # Extract M representatives from this partition
            reps_k[p*M:(p+1)*M] = torch.einsum('mhn,nhd->mhd', pool_attn, k_p)
            reps_v[p*M:(p+1)*M] = torch.einsum('mhn,nhd->mhd', pool_attn, v_p)

        x_local = out_local.mean(dim=1)  # [N, D]

        # ─── GLOBAL: cross-partition attention to representatives ───
        # Q: [N,H,D] attends to reps_k: [K*M,H,D]
        cross_attn = torch.einsum('nhd,rhd->nhr', Q, reps_k) / self.scale
        cross_attn = F.softmax(cross_attn, dim=-1)  # [N, H, K*M]
        if self.training and self.attn_dropout > 0:
            cross_attn = F.dropout(cross_attn, p=self.attn_dropout,
                                   training=True)

        out_global = torch.einsum('nhr,rhd->nhd', cross_attn, reps_v)
        x_global = out_global.mean(dim=1)  # [N, D]

        # ─── SELF: each node's own transformed value ───
        x_self = V.mean(dim=1)  # [N, D]

        # ─── COMBINE: local + global + self ───
        if self.local_only:
            x_context = x_local
        elif self.global_only:
            x_context = x_global
        else:
            alpha = torch.sigmoid(self.alpha_logit)
            x_context = alpha * x_local + (1 - alpha) * x_global
        return x_context + self.beta * x_self


class PCGTConv(nn.Module):
    """Multi-layer multi-resolution partition attention."""

    def __init__(self, in_channels, hidden_channels, num_layers=1, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_act=False, num_partitions=10,
                 num_reps=4, no_pse=False, local_only=False, global_only=False):
        super().__init__()

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))

        # Partition Structural Encoding
        self.no_pse = no_pse
        self.partition_pe = nn.Embedding(num_partitions, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                PCGTConvLayer(hidden_channels, hidden_channels,
                              num_heads=num_heads, use_weight=use_weight,
                              attn_dropout=dropout, num_reps=num_reps,
                              local_only=local_only, global_only=global_only))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.normal_(self.partition_pe.weight, std=0.02)

    def forward(self, data, partition_indices, partition_labels=None):
        x = data.graph['node_feat']
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if partition_labels is not None and not self.no_pse:
            x = x + self.partition_pe(partition_labels)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, partition_indices)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x


class PCGT(nn.Module):
    """Partition-Conditioned Graph Transformer (v4).

    Multi-resolution partition attention + GCN for node classification.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=1, num_heads=1, alpha=0.5, dropout=0.5,
                 use_bn=True, use_residual=True, use_weight=True,
                 use_graph=True, use_act=False,
                 graph_weight=0.8, gnn=None, aggregate='add',
                 num_partitions=10, num_reps=4,
                 no_pse=False, local_only=False, global_only=False):
        super().__init__()

        self.pcgt_conv = PCGTConv(
            in_channels, hidden_channels, num_layers, num_heads,
            alpha, dropout, use_bn, use_residual, use_weight, use_act,
            num_partitions=num_partitions, num_reps=num_reps,
            no_pse=no_pse, local_only=local_only, global_only=global_only)

        self.gnn = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.use_act = use_act

        self.aggregate = aggregate
        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type: {aggregate}')

        self.partition_indices = None
        self.partition_labels = None

        self.params1 = list(self.pcgt_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def set_partition_info(self, partition_indices, partition_labels):
        device = next(self.parameters()).device
        self.partition_indices = [idx.to(device) for idx in partition_indices]
        self.partition_labels = torch.LongTensor(partition_labels).to(device)

    def forward(self, data):
        x1 = self.pcgt_conv(data, self.partition_indices, self.partition_labels)
        if self.use_graph:
            x2 = self.gnn(data)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
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

    def get_gamma_values(self):
        """Return α (local vs global) and β (self-connection) for monitoring."""
        vals = []
        for c in self.pcgt_conv.convs:
            a = torch.sigmoid(c.alpha_logit).item()
            b = c.beta.item()
            vals.append(f"α={a:.2f},β={b:.2f}")
        return vals
