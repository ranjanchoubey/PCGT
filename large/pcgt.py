"""
PCGT for Large-Scale Graphs (adapted from medium/pcgt.py v4)

Matches the SGFormer interface in large/ours.py:
  - forward(x, edge_index) — raw tensors, no data object
  - Uses large/ours.py GraphConv for GNN branch
  - params1 / params2 for differential weight decay
  - set_partition_info() called from main.py after partition computation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ours import GraphConv


class PCGTConvLayer(nn.Module):
    """Multi-Resolution Partition Attention Layer.

    Fine:   Exact softmax within each partition  O(N²/K)
    Coarse: Learned seeds pool each partition → cross-attend  O(NKM)
    """

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

        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
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
        out_local = torch.zeros(N, H, D, device=x.device)
        reps_k = torch.zeros(num_parts * M, H, D, device=x.device)
        reps_v = torch.zeros(num_parts * M, H, D, device=x.device)

        for p, indices in enumerate(partition_indices):
            q_p = Q[indices]
            k_p = K[indices]
            v_p = V[indices]

            # Exact softmax attention within partition
            attn_local = torch.einsum('phd,qhd->hpq', q_p, k_p) / self.scale
            attn_local = F.softmax(attn_local, dim=-1)
            if self.training and self.attn_dropout > 0:
                attn_local = F.dropout(attn_local, p=self.attn_dropout, training=True)
            out_local[indices] = torch.einsum('hpq,qhd->phd', attn_local, v_p)

            # Learned seed pooling → partition representatives
            pool_attn = torch.einsum('mhd,nhd->mhn', self.pool_seeds, k_p) / self.scale
            pool_attn = F.softmax(pool_attn, dim=-1)
            reps_k[p*M:(p+1)*M] = torch.einsum('mhn,nhd->mhd', pool_attn, k_p)
            reps_v[p*M:(p+1)*M] = torch.einsum('mhn,nhd->mhd', pool_attn, v_p)

        x_local = out_local.mean(dim=1)

        # Cross-partition attention to all representatives
        cross_attn = torch.einsum('nhd,rhd->nhr', Q, reps_k) / self.scale
        cross_attn = F.softmax(cross_attn, dim=-1)
        if self.training and self.attn_dropout > 0:
            cross_attn = F.dropout(cross_attn, p=self.attn_dropout, training=True)
        out_global = torch.einsum('nhr,rhd->nhd', cross_attn, reps_v)
        x_global = out_global.mean(dim=1)

        x_self = V.mean(dim=1)

        alpha = torch.sigmoid(self.alpha_logit)
        x_context = alpha * x_local + (1 - alpha) * x_global
        return x_context + self.beta * x_self


class PCGTConv(nn.Module):
    """Multi-layer partition attention with PSE."""

    def __init__(self, in_channels, hidden_channels, num_layers=1, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_act=True, num_partitions=500, num_reps=4):
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
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.normal_(self.partition_pe.weight, std=0.02)

    def forward(self, x, partition_indices, partition_labels=None):
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
            # If partition_indices is None (e.g. batch has no valid partitions),
            # skip partition attention and just use identity
            x = conv(x, partition_indices)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x


class PCGTFormer(nn.Module):
    """PCGT for large-scale graphs.

    Matches SGFormer interface: forward(x, edge_index), params1/params2.
    Uses large/ours.py GraphConv for GNN branch.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 graph_weight=0.8, aggregate='add',
                 trans_num_layers=1, trans_dropout=0.5, trans_num_heads=1,
                 trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=2, gnn_dropout=0.5,
                 gnn_use_bn=True, gnn_use_residual=True,
                 gnn_use_weight=True, gnn_use_init=False, gnn_use_act=True,
                 use_graph=True, num_partitions=500, num_reps=4):
        super().__init__()

        self.pcgt_conv = PCGTConv(
            in_channels, hidden_channels,
            num_layers=trans_num_layers, num_heads=trans_num_heads,
            dropout=trans_dropout, use_bn=trans_use_bn,
            use_residual=trans_use_residual, use_weight=trans_use_weight,
            use_act=trans_use_act, num_partitions=num_partitions,
            num_reps=num_reps)

        self.graph_conv = GraphConv(
            in_channels, hidden_channels,
            num_layers=gnn_num_layers, dropout=gnn_dropout,
            use_bn=gnn_use_bn, use_residual=gnn_use_residual,
            use_weight=gnn_use_weight, use_init=gnn_use_init,
            use_act=gnn_use_act)

        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type: {aggregate}')

        # Partition info set by main.py after model creation
        self.partition_indices = None
        self.partition_labels = None

        # Differential weight decay groups (same convention as SGFormer)
        self.params1 = list(self.pcgt_conv.parameters())
        self.params2 = list(self.graph_conv.parameters())
        self.params2.extend(list(self.fc.parameters()))

    def set_partition_info(self, partition_indices, partition_labels):
        device = next(self.parameters()).device
        self.partition_indices = [idx.to(device) for idx in partition_indices]
        self.partition_labels = torch.LongTensor(partition_labels).to(device)

    def forward(self, x, edge_index, node_idx=None):
        # For batch mode: remap partition info to local batch indices
        if node_idx is not None and self.partition_labels is not None:
            batch_labels = self.partition_labels[node_idx]
            # Build batch-local partition_indices
            # Create reverse map: global_id -> local_id for this batch
            local_map = torch.full((self.partition_labels.size(0),), -1,
                                   dtype=torch.long, device=x.device)
            local_map[node_idx] = torch.arange(len(node_idx), device=x.device)
            batch_part_indices = []
            for part_idx in self.partition_indices:
                local_ids = local_map[part_idx]
                valid = local_ids >= 0
                if valid.any():
                    batch_part_indices.append(local_ids[valid])
            # If no valid partitions in batch, fall back to single partition
            if not batch_part_indices:
                batch_part_indices = [torch.arange(len(node_idx), device=x.device)]
        else:
            batch_labels = self.partition_labels
            batch_part_indices = self.partition_indices

        x1 = self.pcgt_conv(x, batch_part_indices, batch_labels)
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
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
        self.graph_conv.reset_parameters()
        self.fc.reset_parameters()

    def get_gamma_values(self):
        vals = []
        for c in self.pcgt_conv.convs:
            a = torch.sigmoid(c.alpha_logit).item()
            b = c.beta.item()
            vals.append(f"α={a:.2f},β={b:.2f}")
        return vals
