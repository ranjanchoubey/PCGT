"""
PCGT Partition-wise Visualization: Actual vs Predicted Labels.

Trains PCGT on a dataset, then produces:
  1. Terminal table — per-partition class distribution & accuracy
  2. Figure — side-by-side bar charts for EVERY partition (up to --max_parts, default 20)

Outputs are saved to visualization/outputs/<dataset>/.

Usage examples:
  # Quick run with defaults on Cora (10 partitions):
  python visualization/visualize_partitions.py --dataset cora --num_partitions 10 --quick

  # Full hyper-param run on Chameleon:
  python visualization/visualize_partitions.py --dataset chameleon \
      --num_partitions 10 --hidden_channels 64 --ours_layers 1 \
      --num_layers 2 --graph_weight 0.8 --lr 0.01 --dropout 0.5 \
      --ours_dropout 0.3 --use_bn --ours_use_weight --ours_use_residual \
      --use_graph --backbone gcn --epochs 500

  # Show only first 5 partitions in the figure:
  python visualization/visualize_partitions.py --dataset cora --num_partitions 10 --quick --max_parts 5
"""

import argparse
import os
import sys
import warnings
import random

# -- Path setup: add medium/ to sys.path so imports work --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MEDIUM_DIR = os.path.join(REPO_ROOT, 'medium')
sys.path.insert(0, MEDIUM_DIR)

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

# ── Output directory ──
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'outputs')


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Terminal table
# ═══════════════════════════════════════════════════════════════════════════════

def print_partition_comparison(partition_indices, partition_labels, true_labels,
                               pred_labels, split_mask, dataset_name, num_classes):
    """Print a partition-by-partition actual vs predicted comparison."""

    num_parts = len(partition_indices)
    total_correct = 0
    total_nodes = 0

    print("\n" + "=" * 80)
    print(f"  PARTITION-WISE ACTUAL vs PREDICTED  |  Dataset: {dataset_name}")
    print(f"  {num_parts} partitions  |  {num_classes} classes  |  "
          f"{split_mask.sum().item()} eval nodes")
    print("=" * 80)

    for p_idx, indices in enumerate(partition_indices):
        indices_np = indices.cpu().numpy()
        n_p = len(indices_np)

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

        true_counts = Counter(true_p.tolist())
        pred_counts = Counter(pred_p.tolist())

        print(f"\n  Partition {p_idx:>3d}  |  {n_p:>5d} total nodes  |  "
              f"{len(eval_indices):>5d} eval nodes  |  Acc: {acc:5.1f}%")
        print("  " + "-" * 70)

        cls_names = [get_class_name(c, dataset_name) for c in range(num_classes)]
        header = "  {:>10s}".format("")
        for c in range(num_classes):
            header += f" | {cls_names[c]:>5s}"
        header += " | Total"
        print(header)
        print("  " + "-" * 70)

        actual_row = "  {:>10s}".format("Actual")
        for c in range(num_classes):
            actual_row += f" | {true_counts.get(c, 0):>5d}"
        actual_row += f" | {len(eval_indices):>5d}"
        print(actual_row)

        pred_row = "  {:>10s}".format("Predicted")
        for c in range(num_classes):
            pred_row += f" | {pred_counts.get(c, 0):>5d}"
        pred_row += f" | {len(eval_indices):>5d}"
        print(pred_row)

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

        wrong_mask = true_p != pred_p
        wrong_indices_local = np.where(wrong_mask)[0]
        if len(wrong_indices_local) > 0:
            n_show = min(5, len(wrong_indices_local))
            print(f"  Misclassified ({wrong_mask.sum()}/{len(eval_indices)}):",
                  end="")
            for i in range(n_show):
                ni = wrong_indices_local[i]
                node_id = eval_indices[ni]
                t = get_class_name(true_p[ni], dataset_name)
                p = get_class_name(pred_p[ni], dataset_name)
                print(f"  node{node_id}({t}→{p})", end="")
            if len(wrong_indices_local) > n_show:
                print(f"  ...+{len(wrong_indices_local) - n_show} more", end="")
            print()

    overall_acc = 100.0 * total_correct / total_nodes if total_nodes > 0 else 0
    print("\n" + "=" * 80)
    print(f"  OVERALL: {total_correct}/{total_nodes} correct = {overall_acc:.2f}%")
    print("=" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure — graph drawing: left = Actual, right = Predicted per partition
# ═══════════════════════════════════════════════════════════════════════════════

def _build_color_palette(num_classes):
    """Return a list of distinct colors for up to num_classes classes."""
    # Curated palette (tab10 + extras) for up to 20 classes
    base = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    ]
    return base[:num_classes]


def _get_partition_subgraph(edge_index_np, indices_np):
    """Extract edges whose BOTH endpoints are in the partition."""
    idx_set = set(indices_np.tolist())
    src, dst = edge_index_np[0], edge_index_np[1]
    mask = np.array([s in idx_set and d in idx_set for s, d in zip(src, dst)])
    return src[mask], dst[mask]


def save_partition_figure(partition_indices, true_labels, pred_labels,
                          split_mask, dataset_name, num_classes,
                          edge_index_np, max_parts=20, out_dir=None):
    """Save one image per partition: left=Actual graph, right=Predicted graph."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import networkx as nx
        from matplotlib.patches import Patch, FancyBboxPatch
        from matplotlib.lines import Line2D
    except ImportError:
        print("  matplotlib/networkx not installed — skipping figures.")
        return

    palette = _build_color_palette(num_classes)
    num_parts = len(partition_indices)
    show_parts = min(max_parts, num_parts)

    # Pick partitions with most eval nodes
    part_eval_counts = []
    for p_idx, indices in enumerate(partition_indices):
        n_eval = split_mask[indices].sum().item()
        part_eval_counts.append((p_idx, int(n_eval)))
    part_eval_counts.sort(key=lambda x: -x[1])
    selected = [p for p, n in part_eval_counts[:show_parts] if n > 0]
    selected.sort()

    if not selected:
        print("  No partitions with eval nodes — skipping figures.")
        return

    if out_dir is None:
        out_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Shared legend elements
    cls_names = [get_class_name(c, dataset_name) for c in range(num_classes)]
    legend_handles = [Patch(facecolor=palette[c], edgecolor='#555555',
                            linewidth=0.6, label=cls_names[c])
                      for c in range(num_classes)]
    # Separator + structural legends
    legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='#888888',
                                 markeredgecolor='#555555', markeredgewidth=0.5,
                                 markersize=9, alpha=1.0,
                                 label='Test node (bold)'))
    legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='#CCCCCC',
                                 markeredgecolor='#CCCCCC', markeredgewidth=0.3,
                                 markersize=6, alpha=0.4,
                                 label='Train/Val (faded)'))
    legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='white', markeredgecolor='red',
                                 markeredgewidth=2.0, markersize=9,
                                 label='Misclassified'))

    saved = []
    for p_idx in selected:
        indices = partition_indices[p_idx]
        indices_np = indices.cpu().numpy()
        n_p = len(indices_np)

        # Build networkx subgraph
        sub_src, sub_dst = _get_partition_subgraph(edge_index_np, indices_np)
        G = nx.Graph()
        G.add_nodes_from(indices_np.tolist())
        for s, d in zip(sub_src, sub_dst):
            if s < d:
                G.add_edge(int(s), int(d))

        # Layout computed once, reused for both panels
        if n_p <= 300:
            pos = nx.spring_layout(G, seed=42, k=1.5 / max(np.sqrt(n_p), 1),
                                   iterations=50)
        else:
            pos = nx.kamada_kawai_layout(G)

        node_size = max(50, min(300, 7000 // n_p))
        edge_alpha = max(0.12, min(0.45, 180 / max(G.number_of_edges(), 1)))
        edge_width = max(0.5, min(1.2, 250 / max(G.number_of_edges(), 1)))

        # Eval mask
        mask = split_mask[indices].cpu().numpy().astype(bool)
        eval_indices_set = set(indices_np[mask].tolist())
        n_eval = len(eval_indices_set)
        true_p = true_labels[indices_np[mask]]
        pred_p = pred_labels[indices_np[mask]]
        n_correct = int((true_p == pred_p).sum())
        acc = (100.0 * n_correct / n_eval if n_eval > 0 else 0)

        # Split nodes into test vs train/val
        test_nodes = [n for n in indices_np if n in eval_indices_set]
        trainval_nodes = [n for n in indices_np if n not in eval_indices_set]

        # ── Figure with GridSpec: [graph_left | divider | graph_right] ──
        fig = plt.figure(figsize=(14, 6), facecolor='white')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.01, 1],
                               wspace=0.02, figure=fig)
        ax_actual = fig.add_subplot(gs[0, 0])
        ax_div = fig.add_subplot(gs[0, 1])
        ax_pred = fig.add_subplot(gs[0, 2])

        # ── Divider axis: vertical line ──
        ax_div.set_xlim(0, 1)
        ax_div.set_ylim(0, 1)
        ax_div.axvline(x=0.5, color='#888888', linewidth=1.2, linestyle='-')
        ax_div.axis('off')

        # ── Panel background styling ──
        for ax in [ax_actual, ax_pred]:
            ax.set_facecolor('#FAFAFA')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

        # ── Left: Actual labels ──
        nx.draw_networkx_edges(G, pos, ax=ax_actual, alpha=edge_alpha,
                               edge_color='#AAAAAA', width=edge_width)
        # Faded train/val nodes first (background)
        if trainval_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=trainval_nodes, ax=ax_actual,
                                  node_size=node_size,
                                  node_color=[palette[true_labels[n] % len(palette)]
                                              for n in trainval_nodes],
                                  alpha=0.15, edgecolors='none', linewidths=0)
        # Bold test nodes on top
        if test_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=test_nodes, ax=ax_actual,
                                  node_size=node_size,
                                  node_color=[palette[true_labels[n] % len(palette)]
                                              for n in test_nodes],
                                  alpha=0.95, edgecolors='#555555',
                                  linewidths=0.5)
        ax_actual.set_title('Actual Labels',
                            fontsize=13, fontweight='bold', color='#222222',
                            pad=10)
        ax_actual.text(0.5, 0.97,
                       f'n = {n_p}  |  test = {n_eval}  |  train/val = {n_p - n_eval} (faded)',
                       transform=ax_actual.transAxes, ha='center', va='top',
                       fontsize=8, color='#666666')

        # ── Right: Predicted labels ──
        nx.draw_networkx_edges(G, pos, ax=ax_pred, alpha=edge_alpha,
                               edge_color='#AAAAAA', width=edge_width)
        # Faded train/val nodes first
        if trainval_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=trainval_nodes, ax=ax_pred,
                                  node_size=node_size,
                                  node_color=[palette[pred_labels[n] % len(palette)]
                                              for n in trainval_nodes],
                                  alpha=0.15, edgecolors='none', linewidths=0)
        # Bold test nodes on top
        if test_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=test_nodes, ax=ax_pred,
                                  node_size=node_size,
                                  node_color=[palette[pred_labels[n] % len(palette)]
                                              for n in test_nodes],
                                  alpha=0.95, edgecolors='#555555',
                                  linewidths=0.5)
        # Red rings on misclassified test nodes
        misclassified = [n for n in test_nodes
                         if pred_labels[n] != true_labels[n]]
        if misclassified:
            nx.draw_networkx_nodes(G, pos, nodelist=misclassified, ax=ax_pred,
                                  node_size=int(node_size * 1.6),
                                  node_color='none',
                                  edgecolors='red', linewidths=2.2, alpha=0.9)
        # Accuracy color
        acc_color = '#2e7d32' if acc >= 80 else ('#e65100' if acc >= 60 else '#c62828')
        ax_pred.set_title('Predicted Labels',
                          fontsize=13, fontweight='bold', color='#222222',
                          pad=10)
        ax_pred.text(0.5, 0.97,
                     f'acc = {acc:.1f}%    ({n_correct}/{n_eval} correct)',
                     transform=ax_pred.transAxes, ha='center', va='top',
                     fontsize=9, color=acc_color, fontweight='bold')

        # ── Suptitle ──
        fig.suptitle(f'Partition {p_idx}  \u2014  {dataset_name}',
                     fontsize=15, fontweight='bold', color='#111111', y=0.98)

        # ── Legend at bottom ──
        fig.legend(handles=legend_handles, loc='lower center',
                   ncol=min(num_classes + 1, 10), fontsize=9,
                   frameon=True, fancybox=True, shadow=False,
                   edgecolor='#CCCCCC', facecolor='white',
                   bbox_to_anchor=(0.5, -0.01))

        plt.subplots_adjust(top=0.90, bottom=0.08)

        out_path = os.path.join(out_dir, f'partition_{p_idx}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        saved.append(out_path)

    print(f"  Saved {len(saved)} partition images \u2192 {out_dir}/")
    for p in saved:
        print(f"    {os.path.basename(p)}")
    return saved


def save_summary_figure(partition_indices, true_labels, pred_labels,
                        split_mask, dataset_name, num_classes, out_dir=None):
    """Save a single-page summary: overall confusion + per-partition accuracy bar."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    num_parts = len(partition_indices)
    part_accs = []
    part_sizes = []
    for p_idx, indices in enumerate(partition_indices):
        indices_np = indices.cpu().numpy()
        mask = split_mask[indices].cpu().numpy().astype(bool)
        eval_indices = indices_np[mask]
        if len(eval_indices) == 0:
            part_accs.append(0)
            part_sizes.append(0)
            continue
        true_p = true_labels[eval_indices]
        pred_p = pred_labels[eval_indices]
        part_accs.append(100.0 * (true_p == pred_p).sum() / len(eval_indices))
        part_sizes.append(len(eval_indices))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: per-partition accuracy bar chart ──
    colors = ['forestgreen' if a >= 80 else ('orange' if a >= 60 else 'tomato')
              for a in part_accs]
    bars = ax1.bar(range(num_parts), part_accs, color=colors, alpha=0.85,
                   edgecolor='gray', linewidth=0.5)
    ax1.set_xlabel('Partition ID', fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=10)
    ax1.set_title(f'Per-Partition Accuracy — {dataset_name}', fontsize=12,
                  fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=np.mean([a for a, s in zip(part_accs, part_sizes) if s > 0]),
                color='navy', linestyle='--', linewidth=1.5, label='Mean Acc')
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    # Add value labels on bars
    for i, (a, s) in enumerate(zip(part_accs, part_sizes)):
        if s > 0:
            ax1.text(i, a + 1.5, f'{a:.0f}', ha='center', fontsize=7)

    # ── Right: confusion matrix ──
    all_eval_mask = split_mask.cpu().numpy().astype(bool)
    all_true = true_labels[all_eval_mask]
    all_pred = pred_labels[all_eval_mask]
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_true, all_pred):
        conf[t][p] += 1

    cls_names = [get_class_name(c, dataset_name) for c in range(num_classes)]
    im = ax2.imshow(conf, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels(cls_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(num_classes))
    ax2.set_yticklabels(cls_names, fontsize=8)
    ax2.set_xlabel('Predicted', fontsize=10)
    ax2.set_ylabel('Actual', fontsize=10)
    ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    # Annotate cells
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if conf[i, j] > conf.max() * 0.5 else 'black'
            ax2.text(j, i, str(conf[i, j]), ha='center', va='center',
                     fontsize=8, color=color)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if out_dir is None:
        out_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'summary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='PCGT Partition Visualization — Actual vs Predicted')
    parser_add_main_args(parser)
    parser.add_argument('--quick', action='store_true',
                        help='Use sensible defaults for a quick demo')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Which split to visualize (default: test)')
    parser.add_argument('--max_parts', type=int, default=20,
                        help='Max partitions to show in the figure (default: 20)')
    parser.add_argument('--no_figure', action='store_true',
                        help='Skip saving figures')
    args = parser.parse_args()

    args.method = 'pcgt'
    parser_add_default_args(args)

    if args.quick:
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

    # Convert data_dir to absolute path before chdir so it still resolves
    args.data_dir = os.path.abspath(args.data_dir)

    # chdir to medium/ so hardcoded '../data' paths in data_utils resolve
    original_cwd = os.getcwd()
    os.chdir(MEDIUM_DIR)

    device = torch.device("cpu") if args.cpu else (
        torch.device(f"cuda:{args.device}") if torch.cuda.is_available()
        else torch.device("cpu"))

    # ── Load dataset ──
    dataset = load_nc_dataset(args)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(
            train_prop=args.train_prop, valid_prop=args.valid_prop)]
    elif args.rand_split_class:
        split_idx_lst = [class_rand_splits(
            dataset.label, args.label_num_per_class,
            args.valid_num, args.test_num)]
    else:
        split_idx_lst = load_fixed_splits(
            dataset, name=args.dataset, protocol=args.protocol)

    split_idx = split_idx_lst[0]

    dataset.label = dataset.label.to(device)
    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    if args.dataset not in {'deezer-europe'}:
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    # ── Partitions ──
    print(f"Computing {args.num_partitions} {args.partition_method} partitions...")
    edge_index_cpu = dataset.graph['edge_index'].cpu()
    features_cpu = dataset.graph['node_feat'].cpu()
    partition_indices, boundary_nodes, partition_labels = compute_partitions(
        edge_index_cpu, n, args.num_partitions,
        method=args.partition_method, features=features_cpu)
    print(f"  → {len(partition_indices)} partitions, "
          f"{len(boundary_nodes)} boundary nodes ({100*len(boundary_nodes)/n:.1f}%)")

    # ── Build & train model ──
    model = parse_method('pcgt', args, c, d, device)
    model.set_partition_info(partition_indices, partition_labels)

    criterion = (nn.BCEWithLogitsLoss() if args.dataset in ('deezer-europe',)
                 else nn.NLLLoss())

    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay}
    ], lr=args.lr)

    train_idx = split_idx['train'].to(device)
    model.reset_parameters()

    print(f"Training PCGT for up to {args.epochs} epochs...")
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
                val_out = model(dataset)
            preds = val_out.argmax(dim=1)
            val_acc = (preds[split_idx['valid'].to(device)] ==
                       dataset.label.squeeze(1)[split_idx['valid'].to(device)]
                       ).float().mean()
            test_acc = (preds[split_idx['test'].to(device)] ==
                        dataset.label.squeeze(1)[split_idx['test'].to(device)]
                        ).float().mean()
            print(f"  Epoch {epoch+1:>4d}: loss={loss.item():.4f}  "
                  f"val={100*val_acc:.1f}%  test={100*test_acc:.1f}%")

            if val_acc > best_val:
                best_val = val_acc
                patience = 0
            else:
                patience += 50
                if patience >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    # ── Predictions ──
    model.eval()
    with torch.no_grad():
        out = model(dataset)
    pred_labels = out.argmax(dim=1).cpu().numpy()
    true_labels = dataset.label.squeeze(1).cpu().numpy()

    split_mask = torch.zeros(n, dtype=torch.bool)
    split_mask[split_idx[args.split]] = True

    partition_indices_cpu = [idx.cpu() for idx in partition_indices]

    # Restore original cwd so output paths resolve correctly
    os.chdir(original_cwd)

    # ── Terminal output ──
    print_partition_comparison(
        partition_indices_cpu, partition_labels, true_labels,
        pred_labels, split_mask, args.dataset, c)

    # ── Figures ──
    if not args.no_figure:
        out_dir = os.path.join(OUTPUT_ROOT, args.dataset)
        edge_index_np = dataset.graph['edge_index'].cpu().numpy()
        save_partition_figure(
            partition_indices_cpu, true_labels, pred_labels, split_mask,
            args.dataset, c, edge_index_np,
            max_parts=args.max_parts, out_dir=out_dir)
        save_summary_figure(
            partition_indices_cpu, true_labels, pred_labels, split_mask,
            args.dataset, c, out_dir=out_dir)

    # ── Learned α, β ──
    if hasattr(model, 'get_gamma_values'):
        print("Learned parameters:")
        for v in model.get_gamma_values():
            print(f"  {v}")


if __name__ == '__main__':
    main()
