import numpy as np
import torch
from collections import defaultdict


def edge_index_to_adj_list(edge_index, num_nodes):
    """Convert edge_index [2, E] to adjacency list (list of lists)."""
    adj = defaultdict(set)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for s, d in zip(src, dst):
        if s != d:
            adj[int(s)].add(int(d))
            adj[int(d)].add(int(s))
    # Return as list of sorted lists (METIS format)
    return [sorted(adj[i]) for i in range(num_nodes)]


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
    """Partition nodes by K-means clustering on features.

    Uses sklearn MiniBatchKMeans for efficiency on high-dimensional data.
    For very high-dim features (>1000), applies TruncatedSVD first.
    """
    from sklearn.cluster import MiniBatchKMeans
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    features = features.astype(np.float32)
    N, D = features.shape

    # Reduce dimensionality for very high-dim features
    if D > 1000:
        from sklearn.decomposition import TruncatedSVD
        n_components = min(128, N - 1, D - 1)
        print(f"  KMeans: reducing {D}-dim features to {n_components}-dim via TruncatedSVD...")
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        features = svd.fit_transform(features)

    kmeans = MiniBatchKMeans(
        n_clusters=num_partitions, random_state=seed,
        batch_size=min(1024, N), max_iter=num_iters, n_init=3)
    labels = kmeans.fit_predict(features)
    return labels


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
