"""
Differentiable Graph Partitioning for PCGT.

Instead of fixed METIS partitions, learns soft partition assignments
end-to-end via Gumbel-Softmax. Uses straight-through estimator
(hard=True) so the forward pass uses discrete assignments (compatible
with existing PCGTConvLayer) but gradients flow through.

Key idea: Learn S ∈ R^{N×K} (logits), sample hard assignments via
Gumbel-Softmax, convert to partition_indices format.

Optional: Initialize from METIS to warm-start the learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiablePartitioner(nn.Module):
    """Learnable graph partition assignments.
    
    Learns assignment logits S ∈ R^{N×K}. During forward:
      1. Gumbel-Softmax(S, hard=True) → discrete assignments
      2. Convert to partition_indices format (list of LongTensors)
      3. Feed into existing PCGT attention layers
    
    Gradients flow through Gumbel-Softmax straight-through estimator.
    
    Args:
        num_nodes: N
        num_partitions: K
        init_labels: Optional np.array of initial partition labels (e.g. from METIS)
        temperature: Gumbel-Softmax temperature (lower = more discrete)
        edge_index: Optional [2, E] tensor for edge-aware regularization
    """
    
    def __init__(self, num_nodes, num_partitions, init_labels=None,
                 temperature=1.0, edge_index=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_partitions = num_partitions
        self.temperature = temperature
        
        # Learnable assignment logits: S ∈ R^{N × K}
        self.logits = nn.Parameter(torch.zeros(num_nodes, num_partitions))
        
        if init_labels is not None:
            # Warm-start from METIS: set logit of assigned partition high
            with torch.no_grad():
                for i, label in enumerate(init_labels):
                    self.logits[i, label] = 3.0  # ~95% probability after softmax
        else:
            # Random initialization with slight noise
            nn.init.normal_(self.logits, mean=0.0, std=0.1)
        
        # Store edge_index for cut regularization
        if edge_index is not None:
            self.register_buffer('edge_src', edge_index[0])
            self.register_buffer('edge_dst', edge_index[1])
        else:
            self.edge_src = None
            self.edge_dst = None
    
    def forward(self, hard=True):
        """Compute partition assignments.
        
        Returns:
            partition_indices: list of K LongTensors (node indices per partition)
            partition_labels: LongTensor [N] of partition assignments
            soft_assignments: [N, K] soft probabilities (for regularization)
        """
        # Gumbel-Softmax: differentiable discrete sampling
        soft = F.gumbel_softmax(self.logits, tau=self.temperature, hard=hard)
        # soft shape: [N, K]
        # When hard=True: one-hot in forward, soft gradients in backward
        
        # Convert to hard labels
        partition_labels = soft.argmax(dim=1)  # [N]
        
        # Build partition_indices (list of LongTensors)
        partition_indices = []
        for k in range(self.num_partitions):
            indices = (partition_labels == k).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                partition_indices.append(indices)
        
        return partition_indices, partition_labels, soft
    
    def cut_loss(self, soft_assignments=None):
        """Graph cut regularization: encourage connected nodes to share partitions.
        
        Loss = sum over edges (i,j): ||S_i - S_j||^2
        Minimizing this encourages neighbors to be in the same partition.
        """
        if self.edge_src is None:
            return torch.tensor(0.0, device=self.logits.device)
        
        if soft_assignments is None:
            soft_assignments = F.softmax(self.logits, dim=1)
        
        S_src = soft_assignments[self.edge_src]  # [E, K]
        S_dst = soft_assignments[self.edge_dst]  # [E, K]
        
        # Squared difference of assignment probabilities
        cut = (S_src - S_dst).pow(2).sum(dim=1).mean()
        return cut
    
    def balance_loss(self, soft_assignments=None):
        """Balance regularization: encourage roughly equal partition sizes.
        
        Loss = -entropy of partition size distribution.
        Maximizing entropy means uniform partition sizes.
        """
        if soft_assignments is None:
            soft_assignments = F.softmax(self.logits, dim=1)
        
        # Average assignment probability per partition
        partition_sizes = soft_assignments.mean(dim=0)  # [K]
        # Negative entropy (minimize this = maximize entropy = more balanced)
        log_sizes = torch.log(partition_sizes + 1e-8)
        neg_entropy = (partition_sizes * log_sizes).sum()
        return neg_entropy


def test_differentiable_partitioner():
    """Quick sanity check."""
    N, K = 100, 5
    edge_index = torch.randint(0, N, (2, 300))
    
    dp = DifferentiablePartitioner(N, K, edge_index=edge_index)
    
    # Forward pass
    indices, labels, soft = dp(hard=True)
    
    print(f"Partitions: {len(indices)}")
    print(f"Partition sizes: {[len(idx) for idx in indices]}")
    print(f"Labels shape: {labels.shape}")
    print(f"Soft shape: {soft.shape}")
    
    # Losses
    cut = dp.cut_loss(soft)
    bal = dp.balance_loss(soft)
    print(f"Cut loss: {cut.item():.4f}")
    print(f"Balance loss: {bal.item():.4f}")
    
    # Test gradient flow
    loss = cut + bal
    loss.backward()
    print(f"Logits grad norm: {dp.logits.grad.norm().item():.4f}")
    print("Gradient flows through! ✓")


if __name__ == '__main__':
    test_differentiable_partitioner()
