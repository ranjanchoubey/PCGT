"""
Thorough β extraction for Film (and Deezer for comparison) using
the EXACT paper config from run.sh, 10 fixed splits, 500 epochs.
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


# EXACT config from run.sh (the paper config)
CONFIGS = [
    ('film', 5, [
        '--lr', '0.05', '--num_layers', '2', '--weight_decay', '0.0005',
        '--dropout', '0.5', '--graph_weight', '0.5',
        '--ours_dropout', '0.3', '--ours_weight_decay', '0.01',
        '--no_feat_norm',
        # paper used 10 runs with 10 fixed splits
    ]),
    ('deezer-europe', 20, [
        '--rand_split',
        '--lr', '0.01', '--num_layers', '2', '--hidden_channels', '96',
        '--weight_decay', '5e-05', '--dropout', '0.4',
        '--graph_weight', '0.5', '--ours_dropout', '0.4', '--ours_use_residual',
        '--ours_weight_decay', '5e-05',
    ]),
]


def run_one(dataset_name, num_parts, extra, seed, split_idx_override=None):
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
        '--epochs', '500',
        '--cpu',
        '--display_step', '100',
    ]
    if '--hidden_channels' in extra:
        base_args = [a for i, a in enumerate(base_args)
                    if not (a == '--hidden_channels' or (i > 0 and base_args[i-1] == '--hidden_channels'))]

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

    if dataset_name not in {'deezer-europe'}:
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

    # Get split
    if split_idx_override is not None:
        split_idx = split_idx_override
    elif args.rand_split:
        split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
    else:
        splits = load_fixed_splits(dataset, name=dataset_name, protocol=args.protocol)
        split_idx = splits[0]

    train_idx = split_idx['train'].to(device)
    test_idx = split_idx['test'].to(device)
    criterion = torch.nn.NLLLoss()

    # Check training set size
    if train_idx.dtype == torch.bool:
        n_train = train_idx.sum().item()
    else:
        n_train = train_idx.size(0)
    print(f"    Train size: {n_train}, Test size: {test_idx.sum().item() if test_idx.dtype == torch.bool else test_idx.size(0)}")

    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay}
    ], lr=args.lr)

    best_test = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                out_eval = model(dataset)
                out_eval = F.log_softmax(out_eval, dim=1)
            pred = out_eval.argmax(dim=-1, keepdim=True)
            correct = pred[test_idx].eq(dataset.label[test_idx]).sum().item()
            if test_idx.dtype == torch.bool:
                test_acc = correct / test_idx.sum().item()
            else:
                test_acc = correct / test_idx.size(0)
            best_test = max(best_test, test_acc)
            gammas = model.get_gamma_values()
            print(f"    Epoch {epoch:3d}: Loss={loss.item():.4f} Test={100*test_acc:.1f}% {gammas}")

    for conv in model.pcgt_conv.convs:
        alpha_val = torch.sigmoid(conv.alpha_logit).item()
        beta_val = conv.beta.item()

    return h, alpha_val, beta_val, best_test


def main():
    for dataset_name, num_parts, extra in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        betas = []
        alphas = []
        test_accs = []
        h_val = None

        if dataset_name == 'film':
            # Use 10 fixed splits like the paper
            print("  Using 10 fixed splits (same as paper)")

            # Load dataset once
            parser = argparse.ArgumentParser()
            parser_add_main_args(parser)
            base_args = [
                '--data_dir', '../data', '--method', 'pcgt', '--dataset', 'film',
                '--backbone', 'gcn', '--hidden_channels', '64', '--ours_layers', '1',
                '--use_graph', '--use_residual', '--alpha', '0.5',
                '--num_partitions', '5', '--partition_method', 'metis',
                '--seed', '123', '--runs', '1', '--epochs', '500', '--cpu',
            ] + extra
            args_tmp = parser.parse_args(base_args)

            from dataset import load_nc_dataset as ld
            dataset_tmp = ld(args_tmp)
            splits_lst = load_fixed_splits(dataset_tmp, name='film', protocol='semi')
            print(f"  Loaded {len(splits_lst)} fixed splits")

            for i in range(min(10, len(splits_lst))):
                seed = 123 + i  # Different seed per split
                print(f"\n  --- Split {i}, seed={seed} ---")
                h, alpha, beta, test_acc = run_one(dataset_name, num_parts, extra, seed, split_idx_override=splits_lst[i])
                h_val = h
                betas.append(beta)
                alphas.append(alpha)
                test_accs.append(test_acc)
                print(f"  => β={beta:.4f}, α={alpha:.4f}, test={100*test_acc:.1f}%")

        else:
            # Deezer: random splits with 5 seeds
            seeds = [42, 123, 256, 512, 1024]
            print(f"  Using {len(seeds)} random seeds")
            for i, seed in enumerate(seeds):
                print(f"\n  --- Run {i+1}/{len(seeds)}, seed={seed} ---")
                h, alpha, beta, test_acc = run_one(dataset_name, num_parts, extra, seed)
                h_val = h
                betas.append(beta)
                alphas.append(alpha)
                test_accs.append(test_acc)
                print(f"  => β={beta:.4f}, α={alpha:.4f}, test={100*test_acc:.1f}%")

        betas_arr = np.array(betas)
        alphas_arr = np.array(alphas)
        accs_arr = np.array(test_accs) * 100

        print(f"\n  {'='*50}")
        print(f"  SUMMARY for {dataset_name} (h={h_val:.4f}):")
        print(f"    β values: {[f'{b:.4f}' for b in betas]}")
        print(f"    β mean ± std: {betas_arr.mean():.4f} ± {betas_arr.std():.4f}")
        print(f"    α mean ± std: {alphas_arr.mean():.4f} ± {alphas_arr.std():.4f}")
        print(f"    Test acc: {accs_arr.mean():.2f} ± {accs_arr.std():.2f}%")
        print(f"    All β > 0? {'YES' if all(b > 0 for b in betas) else 'NO'}")
        print(f"    All β < 0? {'YES' if all(b < 0 for b in betas) else 'NO'}")
        n_pos = sum(1 for b in betas if b > 0)
        n_neg = sum(1 for b in betas if b < 0)
        print(f"    Positive: {n_pos}/{len(betas)}, Negative: {n_neg}/{len(betas)}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
