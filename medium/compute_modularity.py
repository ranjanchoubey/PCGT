"""
Compute Newman modularity Q and edge-cut ratio for METIS partitions.
Reproduces the values in Section 6 (Discussion).

Usage:
    cd medium && python compute_modularity.py
"""
import numpy as np
import networkx as nx
from dataset import load_wiki_new
from partition import compute_partitions


def compute_stats(dataset_name, K=10):
    """Load dataset, partition with METIS, compute Q and edge-cut ratio."""
    dataset = load_wiki_new(dataset_name, no_feat_norm=True)
    graph, label = dataset[0]

    edge_index = graph['edge_index']  # [2, E] tensor
    N = graph['num_nodes']

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    ei_np = edge_index.numpy()
    edges = list(zip(ei_np[0], ei_np[1]))
    G.add_edges_from(edges)

    # METIS partition
    partition_indices, boundary_nodes, partition_labels = compute_partitions(
        edge_index, N, K, method='metis'
    )

    # Newman modularity
    communities = []
    for k in range(K):
        members = [i for i, p in enumerate(partition_labels) if p == k]
        communities.append(set(members))
    Q = nx.community.modularity(G, communities)

    # Edge-cut ratio
    total_edges = ei_np.shape[1] // 2  # undirected
    cut_edges = 0
    for src, dst in zip(ei_np[0], ei_np[1]):
        if partition_labels[src] != partition_labels[dst]:
            cut_edges += 1
    cut_edges //= 2  # undirected
    edge_cut_ratio = cut_edges / total_edges

    # Boundary nodes
    boundary = len(boundary_nodes)

    avg_degree = ei_np.shape[1] / N

    print(f"{dataset_name} (K={K}):")
    print(f"  Newman modularity Q = {Q:.4f}")
    print(f"  Edge-cut ratio      = {edge_cut_ratio:.4f}")
    print(f"  Boundary nodes      = {boundary}/{N} ({100*boundary/N:.1f}%)")
    print(f"  Avg degree          = {avg_degree:.1f}")
    print()
    return Q, edge_cut_ratio


if __name__ == '__main__':
    for ds in ['chameleon', 'squirrel']:
        compute_stats(ds, K=10)
