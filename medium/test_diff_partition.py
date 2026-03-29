"""
Test: Differentiable Partitions for PCGT
=========================================
Compares three modes on Cora:
  1. Fixed METIS partitions (current PCGT)
  2. Fixed Random partitions (ablation baseline) 
  3. Learned partitions via Gumbel-Softmax (NEW)
     - Initialized from METIS (warm-start)
     - Jointly trained with PCGT

The learned partitioner has its own optimizer with cut_loss + balance_loss
regularization to encourage topology-aware and balanced partitions.

Usage: cd medium && python test_diff_partition.py
Logs to: experiments/logs/phase4_metis_ablation/diff_partition_test.log
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits
from dataset import load_nc_dataset
from parse import parse_method, parser_add_main_args
from partition import compute_partitions
from diff_partition import DifferentiablePartitioner

warnings.filterwarnings('ignore')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_experiment(mode, dataset, args, device, split_idx, c, d, n,
                   edge_index_cpu, features_cpu, num_epochs=300, 
                   num_runs=3, display_step=50):
    """Run PCGT with a specific partition mode.
    
    mode: 'metis', 'random', or 'learned'
    """
    results = []
    
    for run in range(num_runs):
        fix_seed(args.seed + run)
        
        # === Compute initial partitions ===
        if mode == 'learned':
            # Start from METIS, then learn
            initial_indices, boundary, initial_labels = compute_partitions(
                edge_index_cpu, n, args.num_partitions, method='metis',
                features=features_cpu)
            
            # Create differentiable partitioner
            partitioner = DifferentiablePartitioner(
                num_nodes=n,
                num_partitions=args.num_partitions,
                init_labels=initial_labels,
                temperature=0.5,  # Low temp = more discrete
                edge_index=edge_index_cpu
            ).to(device)
        else:
            partition_indices, boundary, partition_labels = compute_partitions(
                edge_index_cpu, n, args.num_partitions, method=mode,
                features=features_cpu)
            partitioner = None
        
        # === Build model ===
        model = parse_method('pcgt', args, c, d, device)
        
        if mode != 'learned':
            model.set_partition_info(partition_indices, partition_labels)
        
        # === Optimizers ===
        model_optimizer = torch.optim.Adam([
            {'params': model.params1, 'weight_decay': args.ours_weight_decay},
            {'params': model.params2, 'weight_decay': args.weight_decay}
        ], lr=args.lr)
        
        if partitioner is not None:
            part_optimizer = torch.optim.Adam(
                partitioner.parameters(), lr=0.01, weight_decay=1e-4)
        
        train_idx = split_idx['train'].to(device)
        criterion = nn.NLLLoss()
        
        best_val = float('-inf')
        best_test = 0
        patience_count = 0
        
        for epoch in range(num_epochs):
            model.train()
            
            # === Update partitions if learned ===
            if partitioner is not None:
                partitioner.train()
                part_indices, part_labels, soft = partitioner(hard=True)
                model.set_partition_info(part_indices, part_labels)
            
            # === Forward pass ===
            model_optimizer.zero_grad()
            if partitioner is not None:
                part_optimizer.zero_grad()
            
            out = model(dataset)
            out = F.log_softmax(out, dim=1)
            cls_loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
            
            # === Partition regularization ===
            if partitioner is not None:
                cut = partitioner.cut_loss(soft)
                bal = partitioner.balance_loss(soft)
                loss = cls_loss + 0.1 * cut + 0.01 * bal
            else:
                loss = cls_loss
            
            loss.backward()
            model_optimizer.step()
            if partitioner is not None:
                part_optimizer.step()
            
            # === Eval ===
            if epoch % display_step == 0 or epoch == num_epochs - 1:
                model.eval()
                with torch.no_grad():
                    if partitioner is not None:
                        partitioner.eval()
                        p_idx, p_lbl, _ = partitioner(hard=True)
                        model.set_partition_info(p_idx, p_lbl)
                    
                    out = model(dataset)
                    out = F.log_softmax(out, dim=1)
                    pred = out.argmax(dim=-1)
                    
                    train_acc = pred[split_idx['train']].eq(
                        dataset.label.squeeze()[split_idx['train']]).float().mean().item()
                    val_acc = pred[split_idx['valid']].eq(
                        dataset.label.squeeze()[split_idx['valid']]).float().mean().item()
                    test_acc = pred[split_idx['test']].eq(
                        dataset.label.squeeze()[split_idx['test']]).float().mean().item()
                
                gamma_str = ''
                if hasattr(model, 'get_gamma_values'):
                    gamma_str = f' {model.get_gamma_values()}'
                
                part_info = ''
                if partitioner is not None:
                    sizes = [len(idx) for idx in p_idx]
                    part_info = f' parts={len(p_idx)} sizes={min(sizes)}-{max(sizes)}'
                    
                    # Check how many assignments changed from METIS init
                    current_labels = p_lbl.cpu().numpy()
                    changed = (current_labels != initial_labels).sum()
                    part_info += f' changed={changed}/{n}'
                
                print(f'  [{mode}] Run {run+1} Epoch {epoch:3d}: '
                      f'Train={100*train_acc:.1f}% Val={100*val_acc:.1f}% '
                      f'Test={100*test_acc:.1f}%{gamma_str}{part_info}')
                
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= 6:  # 6 * display_step epochs
                        break
        
        results.append(best_test)
        print(f'  [{mode}] Run {run+1} BEST: Val={100*best_val:.1f}% Test={100*best_test:.1f}%')
    
    mean = np.mean(results) * 100
    std = np.std(results) * 100
    return mean, std, results


def main():
    # === Setup ===
    base_args = [
        '--data_dir', '../data',
        '--method', 'pcgt',
        '--dataset', 'cora',
        '--backbone', 'gcn',
        '--hidden_channels', '64',
        '--ours_layers', '1',
        '--use_graph', '--use_residual',
        '--alpha', '0.5',
        '--lr', '0.01',
        '--num_layers', '2',
        '--weight_decay', '5e-4',
        '--dropout', '0.4',
        '--graph_weight', '0.8',
        '--ours_dropout', '0.2',
        '--ours_weight_decay', '0.001',
        '--no_feat_norm',
        '--num_partitions', '10',
        '--partition_method', 'metis',
        '--rand_split_class', '--valid_num', '500', '--test_num', '1000',
        '--seed', '123',
        '--runs', '3',
        '--epochs', '300',
        '--cpu',
    ]
    
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    args = parser.parse_args(base_args)
    
    # Set defaults
    for attr, default in [('patience', 300), ('gat_heads', 8), ('out_heads', 1),
                          ('hops', 1), ('encoder_emdim', 80), ('display_step', 50),
                          ('ours_use_weight', True), ('ours_use_act', False),
                          ('ours_use_residual', True), ('num_heads', 1),
                          ('num_reps', 4), ('aggregate', 'add')]:
        if not hasattr(args, attr):
            setattr(args, attr, default)
    
    fix_seed(args.seed)
    device = torch.device("cpu")
    
    # === Load data ===
    dataset = load_nc_dataset(args)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    
    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    edge_index_cpu = dataset.graph['edge_index'].cpu()
    features_cpu = dataset.graph['node_feat'].cpu()
    
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    dataset.label = dataset.label.to(device)
    
    split_idx = class_rand_splits(dataset.label, args.label_num_per_class,
                                   args.valid_num, args.test_num)
    
    print(f"Dataset: cora, N={n}, d={d}, C={c}")
    print(f"Train: {len(split_idx['train'])}, Val: {len(split_idx['valid'])}, Test: {len(split_idx['test'])}")
    print(f"K={args.num_partitions}")
    print(f"{'='*70}")
    
    NUM_RUNS = 3
    NUM_EPOCHS = 300
    DISPLAY = 50
    
    # === Mode 1: Fixed METIS ===
    print(f"\n{'='*70}")
    print("MODE 1: Fixed METIS partitions")
    print(f"{'='*70}")
    metis_mean, metis_std, _ = run_experiment(
        'metis', dataset, args, device, split_idx, c, d, n,
        edge_index_cpu, features_cpu, NUM_EPOCHS, NUM_RUNS, DISPLAY)
    print(f"\n>>> METIS: {metis_mean:.2f} ± {metis_std:.2f}")
    
    # === Mode 2: Fixed Random ===
    print(f"\n{'='*70}")
    print("MODE 2: Fixed Random partitions")
    print(f"{'='*70}")
    rand_mean, rand_std, _ = run_experiment(
        'random', dataset, args, device, split_idx, c, d, n,
        edge_index_cpu, features_cpu, NUM_EPOCHS, NUM_RUNS, DISPLAY)
    print(f"\n>>> Random: {rand_mean:.2f} ± {rand_std:.2f}")
    
    # === Mode 3: Learned (initialized from METIS) ===
    print(f"\n{'='*70}")
    print("MODE 3: Learned partitions (Gumbel-Softmax, METIS init)")
    print(f"{'='*70}")
    learn_mean, learn_std, _ = run_experiment(
        'learned', dataset, args, device, split_idx, c, d, n,
        edge_index_cpu, features_cpu, NUM_EPOCHS, NUM_RUNS, DISPLAY)
    print(f"\n>>> Learned: {learn_mean:.2f} ± {learn_std:.2f}")
    
    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY: Differentiable Partition Test (Cora)")
    print(f"{'='*70}")
    print(f"  Fixed METIS:     {metis_mean:.2f} ± {metis_std:.2f}")
    print(f"  Fixed Random:    {rand_mean:.2f} ± {rand_std:.2f}")
    print(f"  Learned (ours):  {learn_mean:.2f} ± {learn_std:.2f}")
    print(f"{'='*70}")
    
    gap_metis = learn_mean - metis_mean
    gap_random = learn_mean - rand_mean
    print(f"  Learned vs METIS:  {gap_metis:+.2f}%")
    print(f"  Learned vs Random: {gap_random:+.2f}%")
    
    if gap_metis > 0:
        print(f"\n  ✓ Learned partitions BEAT fixed METIS by {gap_metis:.2f}%!")
        print(f"    → Integrate into main PCGT model")
    elif gap_metis > -0.5:
        print(f"\n  ~ Learned partitions comparable to METIS ({gap_metis:.2f}%)")
        print(f"    → Could still be valuable for removing METIS dependency")
    else:
        print(f"\n  ✗ Learned partitions worse than METIS by {-gap_metis:.2f}%")
        print(f"    → Discard, keep fixed METIS")


if __name__ == '__main__':
    main()
