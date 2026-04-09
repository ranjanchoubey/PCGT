"""
Benchmark inference time and GPU memory for SGFormer vs PCGT.
Measures:
  1. Inference time (ms) — mean of 50 forward passes after 10 warmup
  2. Peak GPU memory (MB) — torch.cuda.max_memory_allocated
"""
import argparse, os, sys, time, warnings, gc
import numpy as np
import torch
import torch.nn.functional as F
from dataset import load_nc_dataset
from parse import parse_method, parser_add_default_args, parser_add_main_args
from torch_geometric.utils import to_undirected

warnings.filterwarnings('ignore')

# Configs: (dataset, method, K, hidden, gcn_layers, graph_weight, extra_args)
CONFIGS = [
    # Small
    dict(dataset='cora', method='sgformer', num_partitions=0, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.2, dropout=0.4, weight_decay=5e-4,
         ours_weight_decay=1e-3, protocol='semi', rand_split_class=True,
         label_num_per_class=20, valid_num=500, test_num=1000),
    dict(dataset='cora', method='pcgt', num_partitions=10, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.2, dropout=0.4, weight_decay=5e-4,
         ours_weight_decay=1e-3, protocol='semi', rand_split_class=True,
         label_num_per_class=20, valid_num=500, test_num=1000, partition_method='metis'),
    # Medium
    dict(dataset='pubmed', method='sgformer', num_partitions=0, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.2, dropout=0.3, weight_decay=5e-4,
         ours_weight_decay=1e-3, protocol='semi', rand_split_class=True,
         label_num_per_class=20, valid_num=500, test_num=1000),
    dict(dataset='pubmed', method='pcgt', num_partitions=50, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.2, dropout=0.3, weight_decay=5e-4,
         ours_weight_decay=1e-3, protocol='semi', rand_split_class=True,
         label_num_per_class=20, valid_num=500, test_num=1000, partition_method='metis'),
    # Large-medium
    dict(dataset='deezer-europe', method='sgformer', num_partitions=0, hidden_channels=96,
         num_layers=2, graph_weight=0.8, ours_dropout=0.4, dropout=0.4, weight_decay=5e-5,
         ours_weight_decay=5e-5, train_prop=0.5, valid_prop=0.25),
    dict(dataset='deezer-europe', method='pcgt', num_partitions=20, hidden_channels=96,
         num_layers=2, graph_weight=0.5, ours_dropout=0.4, dropout=0.4, weight_decay=5e-5,
         ours_weight_decay=5e-5, train_prop=0.5, valid_prop=0.25, partition_method='metis',
         no_feat_norm=False, ours_use_residual=True),
    # Additional: Chameleon, Squirrel, Film, CiteSeer
    dict(dataset='chameleon', method='sgformer', num_partitions=0, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.3, dropout=0.5, weight_decay=1e-3,
         ours_weight_decay=1e-2, train_prop=0.5, valid_prop=0.25),
    dict(dataset='chameleon', method='pcgt', num_partitions=10, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.3, dropout=0.5, weight_decay=1e-3,
         ours_weight_decay=1e-2, train_prop=0.5, valid_prop=0.25, partition_method='metis'),
    dict(dataset='squirrel', method='sgformer', num_partitions=0, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.3, dropout=0.5, weight_decay=1e-3,
         ours_weight_decay=1e-2, train_prop=0.5, valid_prop=0.25),
    dict(dataset='squirrel', method='pcgt', num_partitions=10, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.3, dropout=0.5, weight_decay=1e-3,
         ours_weight_decay=1e-2, train_prop=0.5, valid_prop=0.25, partition_method='metis'),
    dict(dataset='film', method='sgformer', num_partitions=0, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.3, dropout=0.5, weight_decay=1e-3,
         ours_weight_decay=1e-3, train_prop=0.5, valid_prop=0.25),
    dict(dataset='film', method='pcgt', num_partitions=5, hidden_channels=64,
         num_layers=2, graph_weight=0.8, ours_dropout=0.3, dropout=0.5, weight_decay=1e-3,
         ours_weight_decay=1e-3, train_prop=0.5, valid_prop=0.25, partition_method='metis'),
    dict(dataset='citeseer', method='sgformer', num_partitions=0, hidden_channels=64,
         num_layers=2, graph_weight=0.7, ours_dropout=0.2, dropout=0.3, weight_decay=5e-4,
         ours_weight_decay=1e-3, protocol='semi', rand_split_class=True,
         label_num_per_class=20, valid_num=500, test_num=1000),
    dict(dataset='citeseer', method='pcgt', num_partitions=20, hidden_channels=64,
         num_layers=2, graph_weight=0.7, ours_dropout=0.2, dropout=0.3, weight_decay=5e-4,
         ours_weight_decay=1e-3, protocol='semi', rand_split_class=True,
         label_num_per_class=20, valid_num=500, test_num=1000, partition_method='metis'),
]

WARMUP = 10
MEASURE = 50

def build_args(cfg, data_dir):
    """Build an args namespace from config dict."""
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    # Start with defaults
    args = parser.parse_args([])
    parser_add_default_args(args)
    # Override with config
    for k, v in cfg.items():
        setattr(args, k, v)
    args.data_dir = data_dir
    args.device = 0
    args.runs = 1
    args.epochs = 1
    args.use_graph = True
    args.aggregate = 'add'
    args.alpha = 0.5
    args.ours_layers = 1 if cfg['method'] == 'pcgt' else 1
    if not hasattr(args, 'no_feat_norm'):
        args.no_feat_norm = False
    if not hasattr(args, 'partition_method'):
        args.partition_method = 'metis'
    if not hasattr(args, 'ours_use_residual'):
        args.ours_use_residual = False
    return args

def benchmark_one(cfg, data_dir, device):
    args = build_args(cfg, data_dir)
    ds_name = cfg['dataset']
    method = cfg['method']
    K = cfg.get('num_partitions', 0)

    # Load dataset
    dataset = load_nc_dataset(args)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    if ds_name not in ('deezer-europe',):
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    dataset.label = dataset.label.to(device)

    # Partitioning for PCGT
    partition_indices = None
    if method == 'pcgt':
        from partition import compute_partitions
        edge_index_cpu = dataset.graph['edge_index'].cpu()
        features_cpu = dataset.graph['node_feat'].cpu()
        partition_indices, boundary_nodes, partition_labels = compute_partitions(
            edge_index_cpu, n, args.num_partitions, method=args.partition_method,
            features=features_cpu)

    # Build model
    model = parse_method(method, args, c, d, device)
    if method == 'pcgt' and partition_indices is not None:
        model.set_partition_info(partition_indices, partition_labels)
    model.eval()

    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dataset)
    torch.cuda.synchronize()

    # Record peak memory after warmup (includes model + data)
    torch.cuda.reset_peak_memory_stats(device)

    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(MEASURE):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dataset)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    avg_time = np.mean(times)
    std_time = np.std(times)

    # Cleanup
    del model, dataset
    torch.cuda.empty_cache()
    gc.collect()

    return dict(dataset=ds_name, method=method, K=K, N=n,
                inf_time_ms=avg_time, inf_std_ms=std_time,
                peak_mem_mb=peak_mem)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='../data')
    ap.add_argument('--datasets', type=str, default='all',
                    help='Comma-separated list of datasets or "all"')
    run_args = ap.parse_args()

    device = torch.device('cuda:0')
    results = []
    
    target_datasets = None
    if run_args.datasets != 'all':
        target_datasets = set(run_args.datasets.split(','))

    for cfg in CONFIGS:
        if target_datasets and cfg['dataset'] not in target_datasets:
            continue
        print(f"\n{'='*60}")
        print(f"Benchmarking: {cfg['dataset']} / {cfg['method']} (K={cfg.get('num_partitions',0)})")
        print(f"{'='*60}")
        try:
            r = benchmark_one(cfg, run_args.data_dir, device)
            results.append(r)
            print(f"  Inference: {r['inf_time_ms']:.2f} ± {r['inf_std_ms']:.2f} ms")
            print(f"  Peak GPU Memory: {r['peak_mem_mb']:.1f} MB")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Dataset':<15} {'Method':<10} {'K':>4} {'N':>8} {'Inf(ms)':>10} {'Mem(MB)':>10}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['dataset']:<15} {r['method']:<10} {r['K']:>4} {r['N']:>8} {r['inf_time_ms']:>9.2f} {r['peak_mem_mb']:>9.1f}")
    
    # Save to CSV
    import csv
    csv_path = 'benchmark_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset','method','K','N','inf_time_ms','inf_std_ms','peak_mem_mb'])
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to {csv_path}")
