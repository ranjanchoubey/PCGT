# Visualization

Partition-level and analysis tools for PCGT. **Run all commands from the repo root.**

## 1. Partition Visualization

Per-partition graph drawings showing actual vs predicted labels (side-by-side).

### Quick Start

```bash
# Best paper config for each dataset
bash visualization/run_visualize.sh cora
bash visualization/run_visualize.sh chameleon
bash visualization/run_visualize.sh all          # all 11 datasets
```

### Custom Run

```bash
python visualization/visualize_partitions.py --dataset cora \
    --num_partitions 10 --hidden_channels 64 --ours_layers 1 \
    --num_layers 2 --graph_weight 0.8 --lr 0.01 --dropout 0.4 \
    --ours_dropout 0.2 --use_bn --ours_use_weight --ours_use_residual \
    --use_graph --backbone gcn --epochs 500 --split test --max_parts 20
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--split` | test | Which split to visualize: train / valid / test |
| `--max_parts` | 20 | Maximum partitions shown |

### Output

```
visualization/outputs/<dataset>/
├── partition_0.png            # Actual vs Predicted graph for partition 0
├── partition_1.png
├── ...
└── summary.png                # Per-partition accuracy bars + confusion matrix
```

Each `partition_X.png` shows a graph drawing with colored nodes — **left panel: actual labels**, **right panel: predicted labels**. Bold nodes are test nodes; faded nodes are train/val. Red rings mark misclassified nodes.

---

## 2. Analysis Plots

Four thesis-quality analysis visualizations.

### Quick Start

```bash
bash visualization/run_analysis.sh cora
bash visualization/run_analysis.sh chameleon
```

Available datasets: `cora`, `citeseer`, `chameleon`, `squirrel`, `film`

### Output

```
visualization/outputs/<dataset>/analysis/
├── tsne_comparison.png        # Raw features vs PCGT embeddings (t-SNE)
├── confusion_matrix.png       # Row-normalized confusion matrix with counts
├── per_class_f1.png           # Precision, Recall, F1 per class
└── training_curves.png        # Loss + Val/Test accuracy over epochs
```

### Run Individual Plots

```bash
# Only t-SNE and confusion matrix
PLOTS="tsne,confusion" bash visualization/run_analysis.sh cora
```

Plot options: `tsne`, `confusion`, `f1`, `curves`, `all`
