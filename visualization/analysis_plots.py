"""
PCGT Analysis Plots — 4 impressive visualizations for thesis defense.

1. t-SNE: Raw features vs PCGT embeddings (cluster quality comparison)
2. Confusion Matrix: Heatmap showing per-class prediction patterns
3. Per-Class F1 Scores: Horizontal bar chart of precision, recall, F1 per class
4. Training Curves: Loss and accuracy dynamics over epochs

Usage:
  python visualization/analysis_plots.py --dataset cora --num_partitions 10 \\
      --lr 0.01 --num_layers 2 --hidden_channels 64 --dropout 0.4 \\
      --ours_dropout 0.2 --graph_weight 0.8 --weight_decay 5e-4 \\
      --ours_weight_decay 0.001 --use_graph --use_bn --ours_use_weight \\
      --ours_use_residual --backbone gcn --aggregate add \\
      --partition_method metis --no_feat_norm \\
      --rand_split_class --valid_num 500 --test_num 1000 \\
      --seed 123 --epochs 500

  # Or use the shell helper:
  bash visualization/run_analysis.sh cora
"""

import argparse
import os
import sys
import warnings
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MEDIUM_DIR = os.path.join(REPO_ROOT, 'medium')
sys.path.insert(0, MEDIUM_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import (class_rand_splits, eval_acc, evaluate,
                        load_fixed_splits, to_sparse_tensor)
from dataset import load_nc_dataset
from parse import parse_method, parser_add_main_args, parser_add_default_args
from partition import compute_partitions
from torch_geometric.utils import to_undirected

warnings.filterwarnings('ignore')

OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'outputs')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_class_name(label_id, dataset_name):
    class_maps = {
        'cora': {0: 'CaseB', 1: 'GA', 2: 'HCI', 3: 'IR', 4: 'ML', 5: 'Net', 6: 'PM'},
        'citeseer': {0: 'Agen', 1: 'AI', 2: 'DB', 3: 'IR', 4: 'ML', 5: 'HCI'},
    }
    if dataset_name in class_maps and label_id in class_maps[dataset_name]:
        return class_maps[dataset_name][label_id]
    return str(label_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1: t-SNE — Raw Features vs PCGT Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tsne(raw_feats, pcgt_embeds, labels, num_classes, dataset_name,
              out_dir, split_mask=None):
    """Side-by-side t-SNE: raw input features vs learned PCGT embeddings."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from matplotlib.patches import Patch

    # Use test nodes only if split_mask provided, otherwise sample
    if split_mask is not None:
        idx = np.where(split_mask)[0]
    else:
        idx = np.arange(len(labels))
    if len(idx) > 3000:
        idx = np.random.choice(idx, 3000, replace=False)

    raw_sub = raw_feats[idx]
    emb_sub = pcgt_embeds[idx]
    lab_sub = labels[idx]

    print("  Computing t-SNE for raw features...")
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42,
                    max_iter=1000).fit_transform(raw_sub)
    print("  Computing t-SNE for PCGT embeddings...")
    tsne_emb = TSNE(n_components=2, perplexity=30, random_state=42,
                    max_iter=1000).fit_transform(emb_sub)

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
               '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    colors = [palette[l % len(palette)] for l in lab_sub]
    cls_names = [get_class_name(c, dataset_name) for c in range(num_classes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

    ax1.scatter(tsne_raw[:, 0], tsne_raw[:, 1], c=colors, s=12, alpha=0.7,
                edgecolors='none')
    ax1.set_title('Raw Input Features', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_facecolor('#FAFAFA')
    for sp in ax1.spines.values(): sp.set_visible(False)

    ax2.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=colors, s=12, alpha=0.7,
                edgecolors='none')
    ax2.set_title('PCGT Learned Embeddings', fontsize=14, fontweight='bold',
                  pad=12)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_facecolor('#FAFAFA')
    for sp in ax2.spines.values(): sp.set_visible(False)

    legend_handles = [Patch(facecolor=palette[c], label=cls_names[c])
                      for c in range(num_classes)]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=min(num_classes, 10), fontsize=10, frameon=True,
               fancybox=True, edgecolor='#CCCCCC', facecolor='white',
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f't-SNE Visualization — {dataset_name}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    path = os.path.join(out_dir, 'tsne_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 2: Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(true_labels, pred_labels, split_mask,
                          num_classes, dataset_name, out_dir):
    """Heatmap confusion matrix on test nodes — shows per-class patterns."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    test_idx = np.where(split_mask)[0]
    y_true = true_labels[test_idx]
    y_pred = pred_labels[test_idx]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    # Normalize each row to percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = 100.0 * cm / row_sums

    cls_names = [get_class_name(c, dataset_name) for c in range(num_classes)]

    fig, ax = plt.subplots(figsize=(max(7, num_classes * 0.9),
                                    max(6, num_classes * 0.8)),
                           facecolor='white')

    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=100)

    # Annotate cells
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = 'white' if val > 60 else '#333333'
            if count > 0:
                ax.text(j, i, f'{val:.0f}%\n({count})',
                        ha='center', va='center', fontsize=8,
                        color=color, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(cls_names, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(cls_names, fontsize=10)
    ax.set_xlabel('Predicted Class', fontsize=12, labelpad=8)
    ax.set_ylabel('True Class', fontsize=12, labelpad=8)

    overall_acc = 100.0 * (y_true == y_pred).sum() / len(y_true)
    ax.set_title(f'Confusion Matrix — {dataset_name}\n'
                 f'Test Accuracy: {overall_acc:.1f}%',
                 fontsize=14, fontweight='bold', pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Row-Normalized (%)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 3: Per-Class F1 Score Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_per_class_f1(true_labels, pred_labels, split_mask,
                      num_classes, dataset_name, out_dir):
    """Horizontal grouped bar chart: precision, recall, F1 per class."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    test_idx = np.where(split_mask)[0]
    y_true = true_labels[test_idx]
    y_pred = pred_labels[test_idx]

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0)

    cls_names = [get_class_name(c, dataset_name) for c in range(num_classes)]

    # Sort classes by F1 descending for visual impact
    order = np.argsort(f1)  # ascending so barh reads top-to-bottom nicely
    cls_names_sorted = [cls_names[i] for i in order]
    prec_s = prec[order]
    rec_s = rec[order]
    f1_s = f1[order]
    support_s = support[order]

    fig, ax = plt.subplots(figsize=(10, max(5, num_classes * 0.7)),
                           facecolor='white')

    y_pos = np.arange(num_classes)
    bar_h = 0.25

    bars_p = ax.barh(y_pos + bar_h, prec_s, bar_h, label='Precision',
                     color='#42A5F5', alpha=0.85, edgecolor='white', linewidth=0.8)
    bars_r = ax.barh(y_pos, rec_s, bar_h, label='Recall',
                     color='#66BB6A', alpha=0.85, edgecolor='white', linewidth=0.8)
    bars_f = ax.barh(y_pos - bar_h, f1_s, bar_h, label='F1 Score',
                     color='#FFA726', alpha=0.9, edgecolor='white', linewidth=0.8)

    # Value labels on F1 bars
    for bar, val, sup in zip(bars_f, f1_s, support_s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}  (n={sup})', va='center', fontsize=9,
                color='#333333', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cls_names_sorted, fontsize=11)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    macro_f1 = f1.mean()
    ax.set_title(f'Per-Class Metrics — {dataset_name}\n'
                 f'Macro F1: {macro_f1:.3f}',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout()
    path = os.path.join(out_dir, 'per_class_f1.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 4: Training Curves (Loss + Accuracy)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, dataset_name, out_dir):
    """Dual-axis plot: training loss + val/test accuracy over epochs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]
    val_accs = [h['val_acc'] for h in history]
    test_accs = [h['test_acc'] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')

    # Loss on left axis
    color_loss = '#E53935'
    ax1.plot(epochs, losses, color=color_loss, linewidth=2, alpha=0.8,
             label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12, color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_facecolor('#FAFAFA')
    for sp in ['top', 'right']:
        ax1.spines[sp].set_visible(False)

    # Accuracy on right axis
    ax2 = ax1.twinx()
    color_val = '#1E88E5'
    color_test = '#43A047'
    ax2.plot(epochs, val_accs, color=color_val, linewidth=2.2, alpha=0.9,
             label='Val Accuracy', linestyle='-')
    ax2.plot(epochs, test_accs, color=color_test, linewidth=2.2, alpha=0.9,
             label='Test Accuracy', linestyle='--')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, color='#333333')
    ax2.tick_params(axis='y', labelcolor='#333333')
    ax2.spines['top'].set_visible(False)

    # Mark best val epoch
    best_idx = np.argmax(val_accs)
    ax2.scatter([epochs[best_idx]], [val_accs[best_idx]], s=100,
               color=color_val, zorder=5, edgecolors='white', linewidths=2)
    ax2.annotate(f'Best: {val_accs[best_idx]:.1f}%\n(ep {epochs[best_idx]})',
                 xy=(epochs[best_idx], val_accs[best_idx]),
                 xytext=(15, -10), textcoords='offset points',
                 fontsize=9, fontweight='bold', color=color_val,
                 arrowprops=dict(arrowstyle='->', color=color_val, lw=1.2))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
              fontsize=10, frameon=True, fancybox=True, edgecolor='#CCCCCC')

    ax1.xaxis.grid(True, alpha=0.2, linestyle='--')
    ax1.set_axisbelow(True)

    final_test = test_accs[-1]
    ax1.set_title(f'Training Dynamics — {dataset_name}\n'
                  f'Final Test Acc: {final_test:.1f}%',
                  fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='PCGT Analysis Plots — t-SNE, Confusion, F1, Training')
    parser_add_main_args(parser)
    parser.add_argument('--plots', type=str, default='all',
                        help='Comma-separated: tsne,confusion,f1,curves,all')
    args = parser.parse_args()

    args.method = 'pcgt'
    parser_add_default_args(args)

    fix_seed(args.seed)

    # Convert data_dir to absolute before chdir
    args.data_dir = os.path.abspath(args.data_dir)

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
    print(f"Computing {args.num_partitions} partitions...")
    edge_index_cpu = dataset.graph['edge_index'].cpu()
    features_cpu = dataset.graph['node_feat'].cpu()
    partition_indices, boundary_nodes, partition_labels = compute_partitions(
        edge_index_cpu, n, args.num_partitions,
        method=args.partition_method, features=features_cpu)

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
    patience_counter = 0
    training_history = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset)
        out_log = F.log_softmax(out, dim=1)
        loss = criterion(out_log[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        # Record history every 5 epochs for smooth curves
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(dataset)
            preds = val_out.argmax(dim=1)
            val_acc = (preds[split_idx['valid'].to(device)] ==
                       dataset.label.squeeze(1)[split_idx['valid'].to(device)]
                       ).float().mean().item()
            test_acc = (preds[split_idx['test'].to(device)] ==
                        dataset.label.squeeze(1)[split_idx['test'].to(device)]
                        ).float().mean().item()
            training_history.append({
                'epoch': epoch + 1,
                'loss': loss.item(),
                'val_acc': 100 * val_acc,
                'test_acc': 100 * test_acc,
            })

        if (epoch + 1) % 50 == 0:
            h = training_history[-1]
            print(f"  Epoch {epoch+1:>4d}: loss={h['loss']:.4f}  "
                  f"val={h['val_acc']:.1f}%  test={h['test_acc']:.1f}%")

            if h['val_acc'] / 100 > best_val:
                best_val = h['val_acc'] / 100
                patience_counter = 0
            else:
                patience_counter += 50
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    # ── Extract predictions & embeddings ──
    model.eval()
    with torch.no_grad():
        logits = model(dataset)
        pred_labels = logits.argmax(dim=1).cpu().numpy()

        # PCGT embedding (before classification head)
        x_pcgt = model.pcgt_conv(dataset, model.partition_indices,
                                  model.partition_labels)
        if model.use_graph:
            x_gnn = model.gnn(dataset)
            x_fused = (model.graph_weight * x_gnn +
                       (1 - model.graph_weight) * x_pcgt)
        else:
            x_fused = x_pcgt
        pcgt_embeds = x_fused.cpu().numpy()

    true_labels = dataset.label.squeeze(1).cpu().numpy()
    raw_feats = dataset.graph['node_feat'].cpu().numpy()

    split_mask = np.zeros(n, dtype=bool)
    split_mask[split_idx['test'].numpy()] = True

    # Print learned params
    gamma_vals = model.get_gamma_values()
    print(f"\nLearned parameters: {', '.join(gamma_vals)}")

    # Restore cwd before saving
    os.chdir(original_cwd)

    # ── Output directory ──
    out_dir = os.path.join(OUTPUT_ROOT, args.dataset, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    # ── Which plots to generate ──
    plots = args.plots.lower().split(',')
    do_all = 'all' in plots

    edge_index_np = dataset.graph['edge_index'].cpu().numpy()
    partition_indices_cpu = [idx.cpu() for idx in partition_indices]
    split_mask_tensor = torch.zeros(n, dtype=torch.bool)
    split_mask_tensor[split_idx['test']] = True

    print(f"\n{'='*60}")
    print(f"  Generating analysis plots → {out_dir}/")
    print(f"{'='*60}\n")

    if do_all or 'tsne' in plots:
        print("[1/4] t-SNE: Raw Features vs PCGT Embeddings")
        plot_tsne(raw_feats, pcgt_embeds, true_labels, c, args.dataset,
                  out_dir, split_mask=split_mask)

    if do_all or 'confusion' in plots:
        print("[2/4] Confusion Matrix")
        plot_confusion_matrix(true_labels, pred_labels, split_mask,
                              c, args.dataset, out_dir)

    if do_all or 'f1' in plots:
        print("[3/4] Per-Class F1 Scores")
        plot_per_class_f1(true_labels, pred_labels, split_mask,
                          c, args.dataset, out_dir)

    if do_all or 'curves' in plots:
        print("[4/4] Training Curves")
        plot_training_curves(training_history, args.dataset, out_dir)

    print(f"\nAll done! Check {out_dir}/")


if __name__ == '__main__':
    main()
