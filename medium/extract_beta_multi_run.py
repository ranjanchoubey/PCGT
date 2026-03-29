"""
Re-run β extraction for Chameleon and Squirrel with 5 seeds
to verify whether positive β is robust or single-run variance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits
from dataset import load_nc_dataset
from parse import parse_method, parser_add_main_args
from torch_geometric.utils import to_undirected

warnings.filterwarnings('ignore')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_homophily(edge_index, labels, num_nodes):
    src, dst = edge_index[0], edge_index[1]
    if labels.dim() > 1:
        labels = labels.squeeze()
    same = (labels[src] == labels[dst]).float().sum().item()
    return same / src.size(0)

SEEDS = [42, 123, 256, 512, 1024]

CONFIGS = [
    ('chameleon', 10, ['--lr', '0.01', '--num_layers', '2', '--weight_decay', '0.001', '--dropout', '0.5',
                        '--graph_weight', '0.8', '--ours_dropout', '0.3', '--ours_weight_decay', '0.01', '--no_feat_norm']),
    ('squirrel',  10, ['--lr', '0.01', '--num_layers', '4', '--weight_decay', '5e-4', '--dropout', '0.5',
                        '--graph_weight', '0.8', '--ours_dropout', '0.3', '--ours_weight_decay', '0.01', '--no_feat_norm']),
]

def run_one(dataset_name, num_parts, extra, seed):
    base_args = [
        '--data_dir', '../data',
        '--method', 'pcgt',
        '--dataset', dataset_name,
        '--backbone', 'gcn',
        '--hidden_channels', '64',
        '--ours_layers', '1',
        '--use_graph', '--use_residual',
        '--alpha', '0.5',
        '--num_partitions', str(num_parts),
        '--partition_method', 'metis',
        '--seed', str(seed),
        '--runs', '1',
        '--epochs', '300',
        '--cpu',
        '--display_step', '100',
    ]
    all_args = base_args + extra

    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    args = parser.parse_args(all_args)

    if not hasattr(args, 'patience'):
        args.patience = 300
    for attr, default in [('gat_heads', 8), ('out_heads', 1), ('hops', 1),
                          ('encoder_emdim', 80), ('display_step', 100),
                          ('ours_use_weight', True), ('ours_use_act', False),
                          ('ours_use_residual', True), ('num_heads', 1),
                          ('num_reps', 4), ('aggregate', 'add')]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    fix_seed(seed)
    device = torch.device("cpu")

    dataset = load_nc_dataset(args)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    h = compute_homophily(dataset.graph['edge_index'], dataset.label.squeeze(), n)

    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    dataset.label = dataset.label.to(device)

    from partition import compute_partitions
    edge_index_cpu = dataset.graph['edge_index'].cpu()
    features_cpu = dataset.graph['node_feat'].cpu()
    partition_indices, boundary_nodes, partition_labels = compute_partitions(
        edge_index_cpu, n, num_parts, method='metis', features=features_cpu)

    model = parse_method('pcgt', args, c, d, device)
    model.set_partition_info(partition_indices, partition_labels)

    splits = load_fixed_splits(dataset, name=dataset_name, protocol=args.protocol)
    split_idx = splits[0]
    train_idx = split_idx['train'].to(device)
    criterion = torch.nn.NLLLoss()

    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay}
    ], lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            gammas = model.get_gamma_values()
            print(f"    Epoch {epoch:3d}: Loss={loss.item():.4f} {gammas}")

    for conv in model.pcgt_conv.convs:
        alpha_val = torch.sigmoid(conv.alpha_logit).item()
        beta_val = conv.beta.item()

    return h, alpha_val, beta_val


def main():
    for dataset_name, num_parts, extra in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} — {len(SEEDS)} seeds")
        print(f"{'='*60}")

        betas = []
        alphas = []
        h_val = None

        for i, seed in enumerate(SEEDS):
            print(f"\n  --- Run {i+1}/{len(SEEDS)}, seed={seed} ---")
            h, alpha, beta = run_one(dataset_name, num_parts, extra, seed)
            h_val = h
            betas.append(beta)
            alphas.append(alpha)
            print(f"  => β={beta:.4f}, α={alpha:.4f}")

        betas_arr = np.array(betas)
        alphas_arr = np.array(alphas)
        print(f"\n  SUMMARY for {dataset_name} (h={h_val:.4f}):")
        print(f"    β values: {[f'{b:.4f}' for b in betas]}")
        print(f"    β mean ± std: {betas_arr.mean():.4f} ± {betas_arr.std():.4f}")
        print(f"    α mean ± std: {alphas_arr.mean():.4f} ± {alphas_arr.std():.4f}")
        print(f"    All β > 0? {'YES' if all(b > 0 for b in betas) else 'NO'}")
        print(f"    All β < 0? {'YES' if all(b < 0 for b in betas) else 'NO'}")

    print(f"\n{'='*60}")
    print("Done. If β is consistently positive for heterophilic graphs,")
    print("this is a genuine exception to Proposition 3 (not variance).")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
