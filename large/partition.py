import numpy as np
import torch
from collections import defaultdict


def edge_index_to_adj_list(edge_index, num_nodes):
    """Convert edge_index [2, E] to adjacency list (list of lists).
    Uses scipy CSR for memory efficiency on large graphs."""
    from scipy.sparse import coo_matrix

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    # Build symmetric CSR (METIS needs undirected)
    data = np.ones(len(src) * 2, dtype=np.int32)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    # Deduplicate by converting back (CSR automatically sums duplicates)
    adj.data = np.ones_like(adj.data)
    # Convert to list of lists
    return [adj.indices[adj.indptr[i]:adj.indptr[i+1]].tolist() for i in range(num_nodes)]


def metis_partition(edge_index, num_nodes, num_partitions):
    """Partition graph using METIS via pymetis."""
    import pymetis
    adjacency = edge_index_to_adj_list(edge_index, num_nodes)
    _, partition_labels = pymetis.part_graph(num_partitions, adjacency=adjacency)
    return np.array(partition_labels)


def spectral_partition(edge_index, num_nodes, num_partitions):
    """Partition graph using spectral clustering (scipy eigsh + KMeans)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    from sklearn.cluster import KMeans

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    data = np.ones(len(src))
    adj = coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes)).tocsr()

    # Normalized Laplacian
    L = laplacian(adj, normed=True)

    # Compute k smallest eigenvectors (Fiedler vectors)
    k = min(num_partitions, num_nodes - 1)
    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM', tol=1e-4)

    # KMeans on eigenvectors
    kmeans = KMeans(n_clusters=num_partitions, random_state=42, n_init=10)
    partition_labels = kmeans.fit_predict(eigenvectors)
    return partition_labels


def random_partition(num_nodes, num_partitions):
    """Random partition baseline (for ablation)."""
    return np.random.randint(0, num_partitions, size=num_nodes)


def feature_kmeans_partition(features, num_partitions, num_iters=100, seed=42):
    """Partition nodes by K-means clustering on features (PyTorch, no sklearn).

    Useful for heterophilic graphs where graph structure groups dissimilar nodes.
    Feature-based clustering groups nodes with similar attributes regardless of
    graph connectivity.
    """
    torch.manual_seed(seed)
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    features = features.float()
    N, D = features.shape

    # Initialize centroids from random data points
    indices = torch.randperm(N)[:num_partitions]
    centroids = features[indices].clone()

    for _ in range(num_iters):
        dists = torch.cdist(features, centroids)  # [N, K]
        labels = dists.argmin(dim=1)               # [N]
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_partitions):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = features[mask].mean(dim=0)
            else:
                new_centroids[k] = centroids[k]
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels.numpy()


def compute_partitions(edge_index, num_nodes, num_partitions, method='metis', features=None):
    """
    Compute graph partitions.

    Args:
        edge_index: [2, E] tensor (CPU)
        num_nodes: int
        num_partitions: int
        method: 'metis', 'spectral', 'random', or 'kmeans'
        features: [N, D] tensor (required for 'kmeans')

    Returns:
        partition_indices: list of LongTensors, node indices per partition
        boundary_nodes: set of boundary node indices
        partition_labels: np.array of partition assignments
    """
    edge_index_cpu = edge_index.cpu()
    num_partitions = min(num_partitions, num_nodes)

    if method == 'metis':
        try:
            partition_labels = metis_partition(edge_index_cpu, num_nodes, num_partitions)
        except (ImportError, Exception) as e:
            print(f"METIS unavailable ({e}), falling back to spectral clustering")
            partition_labels = spectral_partition(edge_index_cpu, num_nodes, num_partitions)
    elif method == 'spectral':
        partition_labels = spectral_partition(edge_index_cpu, num_nodes, num_partitions)
    elif method == 'random':
        partition_labels = random_partition(num_nodes, num_partitions)
    elif method == 'kmeans':
        assert features is not None, "Features required for kmeans partitioning"
        partition_labels = feature_kmeans_partition(features, num_partitions)
    else:
        raise ValueError(f"Unknown partition method: {method}")

    # Build partition index lists
    partition_indices = []
    for k in range(num_partitions):
        indices = np.where(partition_labels == k)[0]
        if len(indices) > 0:
            partition_indices.append(torch.LongTensor(indices))

    # Identify boundary nodes (nodes with cross-partition edges)
    src, dst = edge_index_cpu[0].numpy(), edge_index_cpu[1].numpy()
    boundary_nodes = set()
    for s, d in zip(src, dst):
        if partition_labels[s] != partition_labels[d]:
            boundary_nodes.add(int(s))
            boundary_nodes.add(int(d))

    return partition_indices, boundary_nodes, partition_labels
