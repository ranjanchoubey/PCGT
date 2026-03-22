"""
PCGT v6: Partition-Conditioned Graph Transformer
with Topology-Grounded Global Representatives.

Key change from v4: replaces learned pool_seeds [M, H, D] with
partition centroids — the mean of node features per partition.

Why: pool_seeds are arbitrary learned vectors with no structural meaning.
Centroids are topology-grounded: each one summarizes a real region of the
graph.  This removes M×H×D parameters and ties global context directly to
graph structure.

Architecture per layer:
  x → Q, K, V
      ├─→ IntraPartitionAttn(Q,K,V) per partition ──── x_local
      │
      ├─→ centroid_k = mean(K[p]),  centroid_v = mean(V[p])  per partition
      └─→ CrossAttn(Q, centroids_k, centroids_v) ──── x_global
                                                        │
      α * x_local + (1-α) * x_global + β * x_self    ◄───┘
      α = sigmoid(learnable scalar)
      β = learnable self-connection weight

Complexity: O(N²/K + NKD) — same as v4, but K naturally replaces M
and the global reps are grounded in topology rather than learned from scratch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCGTConvLayer(nn.Module):
    """Multi-Resolution Partition Attention with Centroid Global Reps.

    Fine:   Exact softmax attention within each partition  O(N²/K)
    Coarse: Partition centroids (mean-pooled K/V) as global representatives
            Each node cross-attends to K centroids → O(NKD)
    Combined via learnable α (local vs global) and β (self-connection).
    """

    def __init__(self, in_channels, out_channels, num_heads=1,
                 use_weight=True, attn_dropout=0.0):
        super().__init__()
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.scale = math.sqrt(out_channels)
        self.attn_dropout = attn_dropout

        # α blends local (intra-partition) and global (cross-partition)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        # β: learnable self-connection weight
        self.beta = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()
        nn.init.constant_(self.alpha_logit, 0.0)
        nn.init.constant_(self.beta, 1.0)

    def forward(self, x, partition_indices):
        N = x.size(0)
        H = self.num_heads
        D = self.out_channels

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

        # ─── BUILD CENTROIDS: mean-pooled K/V per partition ───
        centroids_k = torch.zeros(num_parts, H, D, device=x.device)
        centroids_v = torch.zeros(num_parts, H, D, device=x.device)

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

            # Partition centroids: simple mean pooling
            centroids_k[p] = k_p.mean(dim=0)  # [H, D]
            centroids_v[p] = v_p.mean(dim=0)  # [H, D]

        x_local = out_local.mean(dim=1)  # [N, D]

        # ─── GLOBAL: cross-attention to partition centroids ───
        # Q: [N,H,D] attends to centroids_k: [K,H,D]
        cross_attn = torch.einsum('nhd,khd->nhk', Q, centroids_k) / self.scale
        cross_attn = F.softmax(cross_attn, dim=-1)  # [N, H, K]
        if self.training and self.attn_dropout > 0:
            cross_attn = F.dropout(cross_attn, p=self.attn_dropout,
                                   training=True)

        out_global = torch.einsum('nhk,khd->nhd', cross_attn, centroids_v)
        x_global = out_global.mean(dim=1)  # [N, D]

        # ─── SELF: each node's own transformed value ───
        x_self = V.mean(dim=1)  # [N, D]

        # ─── COMBINE: local + global + self ───
        alpha = torch.sigmoid(self.alpha_logit)
        x_context = alpha * x_local + (1 - alpha) * x_global
        return x_context + self.beta * x_self


class PCGTConv(nn.Module):
    """Multi-layer multi-resolution partition attention with centroids."""

    def __init__(self, in_channels, hidden_channels, num_layers=1, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_act=False, num_partitions=10):
        super().__init__()

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))

        # Partition Structural Encoding
        self.partition_pe = nn.Embedding(num_partitions, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                PCGTConvLayer(hidden_channels, hidden_channels,
                              num_heads=num_heads, use_weight=use_weight,
                              attn_dropout=dropout))
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

        if partition_labels is not None:
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
    """Partition-Conditioned Graph Transformer (v6).

    Topology-grounded centroid global reps + GCN for node classification.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=1, num_heads=1, alpha=0.5, dropout=0.5,
                 use_bn=True, use_residual=True, use_weight=True,
                 use_graph=True, use_act=False,
                 graph_weight=0.8, gnn=None, aggregate='add',
                 num_partitions=10):
        super().__init__()

        self.pcgt_conv = PCGTConv(
            in_channels, hidden_channels, num_layers, num_heads,
            alpha, dropout, use_bn, use_residual, use_weight, use_act,
            num_partitions=num_partitions)

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
