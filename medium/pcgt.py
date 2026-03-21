"""
PCGT: Partition-Conditioned Graph Transformer (v3)

Key improvements over SGFormer:

1. **Shared Q/K/V projections**: Both global and partition attention use the
   SAME query/key/value projections. The only difference: O(N) kernel-approx
   (global) vs exact softmax within partitions (local). Partition attention
   is a principled correction for what the kernel approximation misses.

2. **Partition Structural Encoding (PSE)**: Learnable partition embeddings
   injected into features before attention. Global attention becomes
   partition-aware without a separate branch.

3. **Feature-based partitioning**: For heterophilic graphs, K-means on
   features groups similar nodes regardless of connectivity.

Architecture per layer (shared Q/K/V):
  x + PSE → Q, K, V ─→ GlobalLinearAttn(Q,K,V) ──┐
                    └──→ PartitionExactAttn(Q,K,V)─┤
                                                    ├→ (1-γ)*global + γ*partition
                                                   γ = sigmoid(learnable scalar)

Combined with GCN exactly like SGFormer:
  output = graph_weight * x_gnn + (1 - graph_weight) * x_attn → fc
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ours import full_attention_conv


class PCGTConvLayer(nn.Module):
    """Single PCGT attention layer with shared Q/K/V.

    Global branch: O(N) kernel-approximated attention (SGFormer's mechanism)
    Partition branch: Exact softmax attention within each partition
    Both share Q/K/V projections — the only difference is the attention fn.
    Mixed by a learnable scalar γ per layer.
    """

    def __init__(self, in_channels, out_channels, num_heads=1,
                 use_weight=True, attn_dropout=0.0):
        super().__init__()
        # Shared Q/K/V projections for both branches
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.scale = math.sqrt(out_channels)
        self.attn_dropout = attn_dropout

        # Learnable scalar γ: sigmoid(-1.0) ≈ 0.27
        self.gamma_logit = nn.Parameter(torch.tensor(-1.0))

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()
        nn.init.constant_(self.gamma_logit, -1.0)

    def forward(self, x, partition_indices):
        N = x.size(0)
        H = self.num_heads
        D = self.out_channels

        # Shared Q/K/V
        Q = self.Wq(x).reshape(N, H, D)
        K = self.Wk(x).reshape(N, H, D)
        if self.use_weight:
            V = self.Wv(x).reshape(N, H, D)
        else:
            V = x.reshape(N, 1, D).expand(N, H, D)

        # Global: O(N) kernel-approximated attention (same as SGFormer)
        x_global = full_attention_conv(Q, K, V).mean(dim=1)  # [N, D]

        if partition_indices is not None:
            # Partition: exact softmax attention within each partition
            out_part = torch.zeros(N, H, D, device=x.device)
            for indices in partition_indices:
                q = Q[indices]
                k = K[indices]
                v = V[indices]
                attn = torch.einsum('phd,qhd->hpq', q, k) / self.scale
                attn = F.softmax(attn, dim=-1)
                if self.training and self.attn_dropout > 0:
                    attn = F.dropout(attn, p=self.attn_dropout, training=True)
                out_part[indices] = torch.einsum('hpq,qhd->phd', attn, v)
            x_part = out_part.mean(dim=1)  # [N, D]

            gamma = torch.sigmoid(self.gamma_logit)
            return (1 - gamma) * x_global + gamma * x_part

        return x_global


class PCGTConv(nn.Module):
    """Multi-layer PCGT attention. Mirrors TransConv structure exactly,
    but each layer combines global + partition attention."""

    def __init__(self, in_channels, hidden_channels, num_layers=1, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_act=False, num_partitions=10):
        super().__init__()

        # Shared input MLP (same as TransConv)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))

        # Partition Structural Encoding: learnable embedding per partition
        self.partition_pe = nn.Embedding(num_partitions, hidden_channels)

        # PCGT layers (global + partition attention combined)
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

        # Shared input MLP
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Add Partition Structural Encoding
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
    """
    Partition-Conditioned Graph Transformer (v3).

    Three improvements over SGFormer:
      1. Partition Structural Encoding (learnable embeddings per partition)
      2. Node-adaptive gating between global and partition attention
      3. Supports feature-based (kmeans) partitioning for heterophilic graphs
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=1, num_heads=1, alpha=0.5, dropout=0.5,
                 use_bn=True, use_residual=True, use_weight=True,
                 use_graph=True, use_act=False,
                 graph_weight=0.8, gnn=None, aggregate='add',
                 num_partitions=10):
        super().__init__()

        # Augmented transformer: global + partition attention (shared embedding)
        self.pcgt_conv = PCGTConv(
            in_channels, hidden_channels, num_layers, num_heads,
            alpha, dropout, use_bn, use_residual, use_weight, use_act,
            num_partitions=num_partitions)

        # GCN branch (identical to SGFormer)
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

        # Partition info (set before training)
        self.partition_indices = None
        self.partition_labels = None

        # Optimizer param groups (same interface as SGFormer)
        self.params1 = list(self.pcgt_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def set_partition_info(self, partition_indices, partition_labels):
        """Set partition indices and labels (call after model.to(device))."""
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
        """Return current gamma values for monitoring."""
        gammas = []
        for conv in self.pcgt_conv.convs:
            gammas.append(torch.sigmoid(conv.gamma_logit).item())
        return gammas
