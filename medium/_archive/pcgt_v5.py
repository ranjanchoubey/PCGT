"""
PCGT v5: Boundary-Routed Multi-Resolution Partition Attention

Key improvements over v4:
  1. Boundary-Aware Routing: Per-node α based on boundary score.
     Interior nodes → more local attention (intra-community)
     Boundary nodes → more global attention (cross-community)
     Only 1 extra scalar parameter (boundary_weight).

  2. Constrained Self-Connection: β ∈ [0, 2] via sigmoid scaling.
     Prevents the model from bypassing attention via unbounded β.

Architecture per layer:
  x → Q, K, V
      ├─→ IntraPartitionAttn(Q,K,V) per partition ──── x_local
      │
      ├─→ seeds · K[p] → attn → weighted V[p] ─→ reps (K×M vectors)
      └─→ CrossAttn(Q, reps_k, reps_v) ────────────── x_global
                                                        │
      α_i * x_local + (1-α_i) * x_global + β * x_self ◄┘
      α_i = sigmoid(α_logit + w * boundary_score_i)   (per-node)
      β   = sigmoid(β_logit) * 2                       (bounded [0,2])
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCGTConvLayer(nn.Module):
    """Boundary-Routed Multi-Resolution Partition Attention Layer."""

    def __init__(self, in_channels, out_channels, num_heads=1,
                 use_weight=True, attn_dropout=0.0, num_reps=4):
        super().__init__()
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.pool_seeds = nn.Parameter(
            torch.randn(num_reps, num_heads, out_channels) * 0.02)

        self.num_reps = num_reps
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.scale = math.sqrt(out_channels)
        self.attn_dropout = attn_dropout

        # Base α: local vs global blend
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        # Boundary routing: shifts α for boundary nodes (positive w → boundary gets more global)
        self.boundary_weight = nn.Parameter(torch.tensor(-1.0))

        # Constrained self-connection: β ∈ [0, 2]
        self.beta_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 → β=1.0

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()
        nn.init.normal_(self.pool_seeds, std=0.02)
        nn.init.constant_(self.alpha_logit, 0.0)
        nn.init.constant_(self.boundary_weight, -1.0)
        nn.init.constant_(self.beta_logit, 0.0)

    def forward(self, x, partition_indices, boundary_scores=None):
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
            q_p = Q[indices]
            k_p = K[indices]
            v_p = V[indices]

            attn_local = torch.einsum('phd,qhd->hpq', q_p, k_p) / self.scale
            attn_local = F.softmax(attn_local, dim=-1)
            if self.training and self.attn_dropout > 0:
                attn_local = F.dropout(attn_local, p=self.attn_dropout,
                                       training=True)
            out_local[indices] = torch.einsum('hpq,qhd->phd', attn_local, v_p)

            pool_attn = torch.einsum('mhd,nhd->mhn', self.pool_seeds, k_p)
            pool_attn = pool_attn / self.scale
            pool_attn = F.softmax(pool_attn, dim=-1)

            reps_k[p*M:(p+1)*M] = torch.einsum('mhn,nhd->mhd', pool_attn, k_p)
            reps_v[p*M:(p+1)*M] = torch.einsum('mhn,nhd->mhd', pool_attn, v_p)

        x_local = out_local.mean(dim=1)  # [N, D]

        # ─── GLOBAL: cross-partition attention to representatives ───
        cross_attn = torch.einsum('nhd,rhd->nhr', Q, reps_k) / self.scale
        cross_attn = F.softmax(cross_attn, dim=-1)
        if self.training and self.attn_dropout > 0:
            cross_attn = F.dropout(cross_attn, p=self.attn_dropout,
                                   training=True)

        out_global = torch.einsum('nhr,rhd->nhd', cross_attn, reps_v)
        x_global = out_global.mean(dim=1)  # [N, D]

        # ─── SELF: each node's own transformed value ───
        x_self = V.mean(dim=1)  # [N, D]

        # ─── BOUNDARY-ROUTED COMBINE ───
        # Per-node α: boundary nodes shifted toward more global attention
        if boundary_scores is not None:
            # boundary_scores: [N] in [0,1], higher = more boundary
            # Negative boundary_weight → boundary nodes get LOWER α → more global
            alpha = torch.sigmoid(
                self.alpha_logit + self.boundary_weight * boundary_scores
            ).unsqueeze(-1)  # [N, 1]
        else:
            alpha = torch.sigmoid(self.alpha_logit)  # scalar fallback

        x_context = alpha * x_local + (1 - alpha) * x_global

        # Constrained β ∈ [0, 2]
        beta = torch.sigmoid(self.beta_logit) * 2.0

        return x_context + beta * x_self


class PCGTConv(nn.Module):
    """Multi-layer boundary-routed partition attention."""

    def __init__(self, in_channels, hidden_channels, num_layers=1, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_act=False, num_partitions=10,
                 num_reps=4):
        super().__init__()

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))

        self.partition_pe = nn.Embedding(num_partitions, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                PCGTConvLayer(hidden_channels, hidden_channels,
                              num_heads=num_heads, use_weight=use_weight,
                              attn_dropout=dropout, num_reps=num_reps))
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

    def forward(self, data, partition_indices, partition_labels=None,
                boundary_scores=None):
        x = data.graph['node_feat']
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if partition_labels is not None:
            x = x + self.partition_pe(partition_labels)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, partition_indices, boundary_scores)
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
    """PCGT v5: Boundary-Routed Partition-Conditioned Graph Transformer."""

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
        self.boundary_scores = None

        self.params1 = list(self.pcgt_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def set_partition_info(self, partition_indices, partition_labels,
                          boundary_scores=None):
        device = next(self.parameters()).device
        self.partition_indices = [idx.to(device) for idx in partition_indices]
        self.partition_labels = torch.LongTensor(partition_labels).to(device)
        if boundary_scores is not None:
            self.boundary_scores = boundary_scores.to(device)

    def forward(self, data):
        x1 = self.pcgt_conv(data, self.partition_indices,
                            self.partition_labels, self.boundary_scores)
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
        """Return α_base, β, w for monitoring."""
        vals = []
        for c in self.pcgt_conv.convs:
            a = torch.sigmoid(c.alpha_logit).item()
            b = (torch.sigmoid(c.beta_logit) * 2.0).item()
            w = c.boundary_weight.item()
            vals.append(f"α={a:.2f},β={b:.2f},w={w:.2f}")
        return vals
