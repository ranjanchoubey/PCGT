# Visualization

Partition-wise analysis tools for PCGT. Run from the **repo root**.

## Quick Start

```bash
# Cora (10 partitions, all defaults)
python visualization/visualize_partitions.py --dataset cora --num_partitions 10 --quick

# Chameleon (heterophilic)
python visualization/visualize_partitions.py --dataset chameleon --num_partitions 10 --quick

# Show all 20 partitions in figure
python visualization/visualize_partitions.py --dataset cora --num_partitions 20 --quick --max_parts 20
```

## Full Hyper-parameter Run

```bash
python visualization/visualize_partitions.py --dataset cora \
    --num_partitions 10 --hidden_channels 64 --ours_layers 1 \
    --num_layers 2 --graph_weight 0.8 --lr 0.01 --dropout 0.4 \
    --ours_dropout 0.2 --use_bn --ours_use_weight --ours_use_residual \
    --use_graph --backbone gcn --epochs 500 --protocol semi --split test
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | — | Use sensible defaults (epochs=300, gcn backbone, etc.) |
| `--split` | test | Which split to visualize: train / valid / test |
| `--max_parts` | 20 | Maximum partitions shown in the figure |
| `--no_figure` | — | Skip figure generation, only print terminal table |

## Output Structure

```
visualization/
├── README.md
├── visualize_partitions.py
└── outputs/
    ├── cora/
    │   ├── partition_actual_vs_pred.png   # Side-by-side bars per partition
    │   └── summary.png                    # Accuracy bar chart + confusion matrix
    ├── chameleon/
    │   ├── partition_actual_vs_pred.png
    │   └── summary.png
    └── ...
```

## What the Figures Show

1. **partition_actual_vs_pred.png** — For each partition (up to `--max_parts`): grouped bar chart with **blue = actual** class distribution and **red = predicted**. Title shows partition ID, number of eval nodes, and accuracy.

2. **summary.png** — Two panels:
   - **Left:** Per-partition accuracy bar chart (green ≥ 80%, orange ≥ 60%, red < 60%) with mean line.
   - **Right:** Overall confusion matrix.
