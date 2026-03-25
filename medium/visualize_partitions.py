"""
Partition-wise Actual vs Predicted Visualization for PCGT.

Shows a side-by-side comparison of actual and predicted labels for each
partition, printing a clear table and optionally saving a matplotlib figure.

Usage:
    cd medium
    python visualize_partitions.py --dataset cora --num_partitions 10 \
        --hidden_channels 64 --ours_layers 1 --num_layers 2 \
        --graph_weight 0.8 --lr 0.01 --dropout 0.4 --ours_dropout 0.2 \
        --use_bn --use_residual --ours_use_weight --ours_use_residual \
        --use_graph --aggregate add --backbone gcn \
        --epochs 500 --runs 1 --protocol semi

    Or simply:
    python visualize_partitions.py --dataset cora --num_partitions 10 \
        --quick   (uses sensible defaults)
"""

import argparse
import os
import sys
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from data_utils import (class_rand_splits, eval_acc, evaluate,
                        load_fixed_splits, to_sparse_tensor)
from dataset import load_nc_dataset
from parse import parse_method, parser_add_main_args, parser_add_default_args
from partition import compute_partitions
from torch_geometric.utils import to_undirected

warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_class_name(label_id, dataset_name):
    """Return a short readable class name if available."""
    class_maps = {
        'cora': {0: 'CaseB', 1: 'GA', 2: 'HCI', 3: 'IR', 4: 'ML', 5: 'Net', 6: 'PM'},
        'citeseer': {0: 'Agen', 1: 'AI', 2: 'DB', 3: 'IR', 4: 'ML', 5: 'HCI'},
    }
    if dataset_name in class_maps and label_id in class_maps[dataset_name]:
        return class_maps[dataset_name][label_id]
    return str(label_id)


def print_partition_comparison(partition_indices, partition_labels, true_labels,
                               pred_labels, split_mask, dataset_name, num_classes):
    """Print a clear partition-by-partition actual vs predicted comparison."""

    num_parts = len(partition_indices)
    total_correct = 0
    total_nodes = 0

    print("\n" + "=" * 80)
    print(f"  PARTITION-WISE ACTUAL vs PREDICTED  |  Dataset: {dataset_name}")
    print(f"  {num_parts} partitions  |  {num_classes} classes  |  {split_mask.sum().item()} eval nodes")
    print("=" * 80)

    for p_idx, indices in enumerate(partition_indices):
        indices_np = indices.cpu().numpy()
        n_p = len(indices_np)

        # Filter to only nodes in the eval split
        mask = split_mask[indices].cpu().numpy().astype(bool)
        eval_indices = indices_np[mask]

        if len(eval_indices) == 0:
            print(f"\n  Partition {p_idx:>3d} ({n_p:>5d} nodes) — no eval nodes")
            continue

        true_p = true_labels[eval_indices]
        pred_p = pred_labels[eval_indices]
        correct = (true_p == pred_p).sum()
        acc = 100.0 * correct / len(eval_indices)
        total_correct += correct
        total_nodes += len(eval_indices)

        # Class distribution
        true_counts = Counter(true_p.tolist())
        pred_counts = Counter(pred_p.tolist())

        print(f"\n  Partition {p_idx:>3d}  |  {n_p:>5d} total nodes  |  "
              f"{len(eval_indices):>5d} eval nodes  |  Acc: {acc:5.1f}%")
        print("  " + "-" * 70)

        # Header
        cls_names = [get_class_name(c, dataset_name) for c in range(num_classes)]
        header = "  {:>10s}".format("")
        for c in range(num_classes):
            header += f" | {cls_names[c]:>5s}"
        header += " | Total"
        print(header)
        print("  " + "-" * 70)

        # Actual row
        actual_row = "  {:>10s}".format("Actual")
        for c in range(num_classes):
            actual_row += f" | {true_counts.get(c, 0):>5d}"
        actual_row += f" | {len(eval_indices):>5d}"
        print(actual_row)

        # Predicted row
        pred_row = "  {:>10s}".format("Predicted")
        for c in range(num_classes):
            pred_row += f" | {pred_counts.get(c, 0):>5d}"
        pred_row += f" | {len(eval_indices):>5d}"
        print(pred_row)

        # Per-class accuracy
        acc_row = "  {:>10s}".format("Acc%")
        for c in range(num_classes):
            c_mask = true_p == c
            if c_mask.sum() > 0:
                c_acc = 100.0 * ((true_p == pred_p) & c_mask).sum() / c_mask.sum()
                acc_row += f" | {c_acc:>4.0f}%"
            else:
                acc_row += f" |    - "
        acc_row += f" | {acc:>4.1f}%"
        print(acc_row)

        # Show some individual misclassifications (up to 5)
        wrong_mask = true_p != pred_p
        wrong_indices_local = np.where(wrong_mask)[0]
        if len(wrong_indices_local) > 0:
            n_show = min(5, len(wrong_indices_local))
            print(f"  Misclassified ({wrong_mask.sum()}/{len(eval_indices)}):", end="")
            for i in range(n_show):
                ni = wrong_indices_local[i]
                node_id = eval_indices[ni]
                t = get_class_name(true_p[ni], dataset_name)
                p = get_class_name(pred_p[ni], dataset_name)
                print(f"  node{node_id}({t}→{p})", end="")
            if len(wrong_indices_local) > n_show:
                print(f"  ...+{len(wrong_indices_local)-n_show} more", end="")
            print()

    overall_acc = 100.0 * total_correct / total_nodes if total_nodes > 0 else 0
    print("\n" + "=" * 80)
    print(f"  OVERALL: {total_correct}/{total_nodes} correct = {overall_acc:.2f}%")
    print("=" * 80 + "\n")


def save_partition_figure(partition_indices, true_labels, pred_labels,
                          split_mask, dataset_name, num_classes, num_parts_to_show=6):
    """Save a matplotlib figure showing actual vs predicted per partition."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping figure.")
        return

    num_parts = len(partition_indices)
    show_parts = min(num_parts_to_show, num_parts)

    # Pick partitions with most eval nodes
    part_eval_counts = []
    for p_idx, indices in enumerate(partition_indices):
        n_eval = split_mask[indices].sum().item()
        part_eval_counts.append((p_idx, n_eval))
    part_eval_counts.sort(key=lambda x: -x[1])
    selected_parts = [p[0] for p in part_eval_counts[:show_parts]]
    selected_parts.sort()

    fig, axes = plt.subplots(show_parts, 2, figsize=(14, 3 * show_parts))
    if show_parts == 1:
        axes = axes.reshape(1, 2)

    for row, p_idx in enumerate(selected_parts):
        indices = partition_indices[p_idx]
        indices_np = indices.cpu().numpy()
        mask = split_mask[indices].cpu().numpy().astype(bool)
        eval_indices = indices_np[mask]

        true_p = true_labels[eval_indices]
        pred_p = pred_labels[eval_indices]

        true_counts = [0] * num_classes
        pred_counts = [0] * num_classes
        for c in range(num_classes):
            true_counts[c] = (true_p == c).sum()
            pred_counts[c] = (pred_p == c).sum()

        x = np.arange(num_classes)
        width = 0.35
        acc = 100.0 * (true_p == pred_p).sum() / len(eval_indices) if len(eval_indices) > 0 else 0

        # Actual
        axes[row, 0].bar(x, true_counts, width, color='steelblue', alpha=0.8)
        axes[row, 0].set_title(f'Partition {p_idx} — Actual (n={len(eval_indices)})')
        axes[row, 0].set_xticks(x)
        axes[row, 0].set_xticklabels([get_class_name(c, dataset_name) for c in range(num_classes)],
                                      rotation=45, fontsize=8)
        axes[row, 0].set_ylabel('Count')

        # Predicted
        colors = ['forestgreen' if pred_counts[c] == true_counts[c] else
                  ('orange' if abs(pred_counts[c] - true_counts[c]) <= 2 else 'tomato')
                  for c in range(num_classes)]
        axes[row, 1].bar(x, pred_counts, width, color=colors, alpha=0.8)
        axes[row, 1].set_title(f'Partition {p_idx} — Predicted (acc={acc:.1f}%)')
        axes[row, 1].set_xticks(x)
        axes[row, 1].set_xticklabels([get_class_name(c, dataset_name) for c in range(num_classes)],
                                      rotation=45, fontsize=8)
        axes[row, 1].set_ylabel('Count')

    plt.tight_layout()
    out_path = f'results/partition_viz_{dataset_name}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='PCGT Partition Visualization')
    parser_add_main_args(parser)
    parser.add_argument('--quick', action='store_true',
                        help='Use sensible defaults for quick demo')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Which split to visualize')
    parser.add_argument('--no_figure', action='store_true',
                        help='Skip saving matplotlib figure')
    args = parser.parse_args()

    # Override method to pcgt
    args.method = 'pcgt'
    parser_add_default_args(args)

    if args.quick:
        if not hasattr(args, 'use_graph') or not args.use_graph:
            args.use_graph = True
        args.use_bn = True
        args.ours_use_weight = True
        args.ours_use_residual = True
        args.aggregate = 'add'
        args.backbone = 'gcn'
        args.runs = 1
        args.epochs = 300
        args.patience = 200

    fix_seed(args.seed)

    device = torch.device("cpu") if args.cpu else (
        torch.device(f"cuda:{args.device}") if torch.cuda.is_available()
        else torch.device("cpu"))

    # Load dataset
    dataset = load_nc_dataset(args)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(
            train_prop=args.train_prop, valid_prop=args.valid_prop)]
    elif args.rand_split_class:
        split_idx_lst = [class_rand_splits(
            dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
    else:
        split_idx_lst = load_fixed_splits(dataset, name=args.dataset, protocol=args.protocol)

    split_idx = split_idx_lst[0]

    dataset.label = dataset.label.to(device)
    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    if args.dataset not in {'deezer-europe'}:
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    # Compute partitions
    print(f"Computing {args.num_partitions} partitions ({args.partition_method})...")
    edge_index_cpu = dataset.graph['edge_index'].cpu()
    features_cpu = dataset.graph['node_feat'].cpu()
    partition_indices, boundary_nodes, partition_labels = compute_partitions(
        edge_index_cpu, n, args.num_partitions,
        method=args.partition_method, features=features_cpu)
    print(f"  {len(partition_indices)} partitions, {len(boundary_nodes)} boundary nodes")

    # Build model
    model = parse_method('pcgt', args, c, d, device)
    model.set_partition_info(partition_indices, partition_labels)

    # Train
    if args.dataset in ('deezer-europe',):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay}
    ], lr=args.lr)

    train_idx = split_idx['train'].to(device)
    model.reset_parameters()

    print(f"Training PCGT for {args.epochs} epochs...")
    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset)
        out_log = F.log_softmax(out, dim=1)
        loss = criterion(out_log[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                out = model(dataset)
            preds = out.argmax(dim=1)
            test_correct = (preds[split_idx['test'].to(device)] ==
                            dataset.label.squeeze(1)[split_idx['test'].to(device)]).float().mean()
            val_correct = (preds[split_idx['valid'].to(device)] ==
                           dataset.label.squeeze(1)[split_idx['valid'].to(device)]).float().mean()
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, "
                  f"val={100*val_correct:.1f}%, test={100*test_correct:.1f}%")

            if val_correct > best_val:
                best_val = val_correct
                patience = 0
            else:
                patience += 50
                if patience >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    # Final predictions
    model.eval()
    with torch.no_grad():
        out = model(dataset)
    pred_labels = out.argmax(dim=1).cpu().numpy()
    true_labels = dataset.label.squeeze(1).cpu().numpy()

    # Build split mask
    split_mask = torch.zeros(n, dtype=torch.bool)
    eval_key = args.split
    split_mask[split_idx[eval_key]] = True

    # Move partition_indices to CPU for comparison
    partition_indices_cpu = [idx.cpu() for idx in partition_indices]

    # Print comparison
    print_partition_comparison(
        partition_indices_cpu, partition_labels, true_labels,
        pred_labels, split_mask, args.dataset, c)

    # Save figure
    if not args.no_figure:
        save_partition_figure(
            partition_indices_cpu, true_labels, pred_labels,
            split_mask, args.dataset, c)

    # Print α and β values
    if hasattr(model, 'get_gamma_values'):
        print("Learned parameters:")
        for v in model.get_gamma_values():
            print(f"  {v}")


if __name__ == '__main__':
    main()
