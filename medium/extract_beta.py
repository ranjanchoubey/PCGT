"""
Extract learned β (self-connection) and α (local-global blend) values
from PCGT across datasets with different homophily levels.

This validates Proposition 3: β tracks graph homophily.
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
    """Edge homophily: fraction of edges connecting same-class nodes."""
    src, dst = edge_index[0], edge_index[1]
    if labels.dim() > 1:
        labels = labels.squeeze()
    same = (labels[src] == labels[dst]).float().sum().item()
    return same / src.size(0)

# Datasets to test with their PCGT configs
CONFIGS = [
    # (dataset, num_partitions, extra_args)
    ('cora',              10, ['--rand_split_class', '--valid_num', '500', '--test_num', '1000',
                               '--lr', '0.01', '--num_layers', '2', '--weight_decay', '5e-4', '--dropout', '0.4',
                               '--graph_weight', '0.8', '--ours_dropout', '0.2', '--ours_weight_decay', '0.001', '--no_feat_norm']),
    ('citeseer',          20, ['--rand_split_class', '--valid_num', '500', '--test_num', '1000',
                               '--lr', '0.01', '--num_layers', '2', '--weight_decay', '0.01', '--dropout', '0.5',
                               '--graph_weight', '0.7', '--ours_dropout', '0.3', '--ours_weight_decay', '0.01', '--no_feat_norm']),
    ('pubmed',            50, ['--rand_split_class', '--valid_num', '500', '--test_num', '1000',
                               '--lr', '0.01', '--num_layers', '2', '--weight_decay', '0.0005', '--dropout', '0.5',
                               '--graph_weight', '0.8', '--ours_dropout', '0.3', '--ours_weight_decay', '0.01', '--no_feat_norm']),
    ('chameleon',         10, ['--lr', '0.01', '--num_layers', '2', '--weight_decay', '0.001', '--dropout', '0.5',
                               '--graph_weight', '0.8', '--ours_dropout', '0.3', '--ours_weight_decay', '0.01', '--no_feat_norm']),
    ('squirrel',          10, ['--lr', '0.01', '--num_layers', '4', '--weight_decay', '5e-4', '--dropout', '0.5',
                               '--graph_weight', '0.8', '--ours_dropout', '0.3', '--ours_weight_decay', '0.01', '--no_feat_norm']),
    ('film',               5, ['--lr', '0.05', '--num_layers', '2', '--weight_decay', '0.0005', '--dropout', '0.5',
                               '--graph_weight', '0.5', '--ours_dropout', '0.3', '--ours_weight_decay', '0.01', '--no_feat_norm']),
    ('deezer-europe',     20, ['--rand_split', '--lr', '0.01', '--num_layers', '2', '--hidden_channels', '96',
                               '--weight_decay', '5e-05', '--dropout', '0.4',
                               '--graph_weight', '0.5', '--ours_dropout', '0.4', '--ours_use_residual',
                               '--ours_weight_decay', '5e-05']),
    ('coauthor-cs',       15, ['--rand_split', '--lr', '0.01', '--num_layers', '4', '--weight_decay', '5e-4', '--dropout', '0.5',
                               '--graph_weight', '0.8', '--ours_dropout', '0.2', '--ours_weight_decay', '0.001', '--no_feat_norm']),
    ('coauthor-physics',  20, ['--rand_split', '--lr', '0.01', '--num_layers', '4', '--weight_decay', '5e-4', '--dropout', '0.5',
                               '--graph_weight', '0.8', '--ours_dropout', '0.2', '--ours_weight_decay', '0.001', '--no_feat_norm']),
    ('amazon-computers',  10, ['--rand_split', '--lr', '0.01', '--num_layers', '4', '--weight_decay', '5e-4', '--dropout', '0.5',
                               '--graph_weight', '0.8', '--ours_dropout', '0.2', '--ours_weight_decay', '0.001', '--no_feat_norm']),
    ('amazon-photo',      10, ['--rand_split', '--lr', '0.01', '--num_layers', '4', '--weight_decay', '5e-4', '--dropout', '0.5',
                               '--graph_weight', '0.8', '--ours_dropout', '0.2', '--ours_weight_decay', '0.001', '--no_feat_norm']),
]

def main():
    results = []
    
    for dataset_name, num_parts, extra in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}, K={num_parts}")
        print(f"{'='*60}")
        
        # Build args
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
            '--seed', '123',
            '--runs', '1',
            '--epochs', '300',  # Enough to converge
            '--cpu',
            '--display_step', '50',
        ]
        # Override hidden_channels if in extra
        if '--hidden_channels' in extra:
            base_args = [a for i, a in enumerate(base_args) 
                        if not (a == '--hidden_channels' or (i > 0 and base_args[i-1] == '--hidden_channels'))]
        
        all_args = base_args + extra
        
        parser = argparse.ArgumentParser()
        parser_add_main_args(parser)
        args = parser.parse_args(all_args)
        
        # Add default args
        if not hasattr(args, 'patience'):
            args.patience = 300
        for attr, default in [('gat_heads', 8), ('out_heads', 1), ('hops', 1),
                              ('encoder_emdim', 80), ('display_step', 50),
                              ('ours_use_weight', True), ('ours_use_act', False),
                              ('ours_use_residual', True), ('num_heads', 1),
                              ('num_reps', 4), ('aggregate', 'add')]:
            if not hasattr(args, attr):
                setattr(args, attr, default)
        
        fix_seed(args.seed)
        device = torch.device("cpu")
        
        # Load data
        dataset = load_nc_dataset(args)
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        
        n = dataset.graph['num_nodes']
        c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
        d = dataset.graph['node_feat'].shape[1]
        
        if dataset_name not in {'deezer-europe'}:
            dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        
        # Compute homophily
        h = compute_homophily(dataset.graph['edge_index'], dataset.label.squeeze(), n)
        print(f"Homophily h = {h:.4f}")
        
        dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
        dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
        dataset.label = dataset.label.to(device)
        
        # Partition
        from partition import compute_partitions
        edge_index_cpu = dataset.graph['edge_index'].cpu()
        features_cpu = dataset.graph['node_feat'].cpu()
        partition_indices, boundary_nodes, partition_labels = compute_partitions(
            edge_index_cpu, n, num_parts, method='metis', features=features_cpu)
        
        # Build model
        model = parse_method('pcgt', args, c, d, device)
        model.set_partition_info(partition_indices, partition_labels)
        
        # Train
        if args.rand_split:
            split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
        elif args.rand_split_class:
            split_idx = class_rand_splits(dataset.label, args.label_num_per_class, args.valid_num, args.test_num)
        else:
            splits = load_fixed_splits(dataset, name=dataset_name, protocol=args.protocol)
            split_idx = splits[0]
        
        train_idx = split_idx['train'].to(device)
        criterion = torch.nn.NLLLoss()
        
        optimizer = torch.optim.Adam([
            {'params': model.params1, 'weight_decay': args.ours_weight_decay},
            {'params': model.params2, 'weight_decay': args.weight_decay}
        ], lr=args.lr)
        
        best_val = float('-inf')
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(dataset)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0 or epoch == args.epochs - 1:
                gammas = model.get_gamma_values()
                model.eval()
                with torch.no_grad():
                    out = model(dataset)
                    out = F.log_softmax(out, dim=1)
                pred = out.argmax(dim=-1, keepdim=True)
                test_idx = split_idx['test'].to(device)
                correct = pred[test_idx].eq(dataset.label[test_idx]).sum().item()
                test_acc = correct / test_idx.size(0)
                print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f} Test={100*test_acc:.1f}% {gammas}")
        
        # Extract final β and α
        gammas = model.get_gamma_values()
        for conv in model.pcgt_conv.convs:
            alpha_val = torch.sigmoid(conv.alpha_logit).item()
            beta_val = conv.beta.item()
        
        results.append({
            'dataset': dataset_name,
            'homophily': h,
            'alpha': alpha_val,
            'beta': beta_val,
            'nodes': n,
        })
        
        print(f"\n  FINAL: h={h:.4f}, α={alpha_val:.4f}, β={beta_val:.4f}")
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Dataset':<20} {'h':>8} {'β':>10} {'α':>10} {'Nodes':>8}")
    print(f"{'='*70}")
    for r in sorted(results, key=lambda x: x['homophily'], reverse=True):
        print(f"{r['dataset']:<20} {r['homophily']:>8.4f} {r['beta']:>10.4f} {r['alpha']:>10.4f} {r['nodes']:>8d}")
    print(f"{'='*70}")
    
    # Correlation
    hs = [r['homophily'] for r in results]
    betas = [r['beta'] for r in results]
    if len(hs) > 2:
        corr = np.corrcoef(hs, betas)[0, 1]
        print(f"\nPearson correlation(h, β) = {corr:.4f}")

if __name__ == '__main__':
    main()
