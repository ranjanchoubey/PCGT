# PCGT: Partition-Conditioned Graph Transformer

This repository implements **PCGT (Partition-Conditioned Graph Transformer)**, a graph transformer that replaces the expensive global attention in standard graph transformers with **multi-resolution partition-aware attention**. PCGT computes exact attention within graph partitions and cross-partition attention via learned representative nodes, achieving **linear complexity** while preserving structural awareness.

Built upon and extending [SGFormer](https://arxiv.org/pdf/2306.10759.pdf) (NeurIPS 2023).

## Key Idea

Standard graph transformers compute $O(N^2)$ global attention, which is expensive and structure-agnostic. PCGT partitions the graph using METIS and computes:

1. **Local attention**: Exact softmax within each partition — $O(N^2/K)$
2. **Global attention**: Cross-partition communication via learned seed vectors — $O(NKM)$  
3. **Partition Structural Encoding (PSE)**: Learnable embeddings that encode partition membership

The total complexity is $O(N \cdot (N/K + KM))$, which is linear in $N$ for fixed $K$ and $M$.

## Repository Structure

```
PCGT/
├── medium/          # Medium-scale experiments (Cora, CiteSeer, PubMed, Chameleon, etc.)
│   ├── main.py      # Training & evaluation entry point
│   ├── pcgt.py      # PCGT model implementation
│   ├── ours.py      # SGFormer baseline (original code, method='sgformer')
│   ├── partition.py  # Graph partitioning utilities (METIS, spectral, random)
│   ├── run.sh       # Example run commands
│   └── results/     # Experiment result logs
├── large/           # Large-scale experiments (ogbn-arxiv, ogbn-proteins, Pokec)
│   ├── main.py      # Full-batch training
│   ├── main-batch.py # Mini-batch training for large graphs
│   ├── pcgt.py      # PCGT model (large-scale version)
│   ├── ours.py      # SGFormer baseline
│   └── run.sh       # Example run commands
├── 100M/            # ogbn-papers100M (SGFormer baseline only)
├── data/            # Datasets (auto-downloaded or manual)
├── requirements.txt # Python dependencies
└── reproduce_paper_results.sh  # Commands to reproduce all results
```

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- Python >= 3.10
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- OGB >= 1.3.1
- pymetis (for METIS graph partitioning)

For GPU experiments (large-scale), install matching CUDA versions of torch-scatter, torch-sparse, torch-cluster:
```bash
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
```

## Datasets

**Auto-downloaded**: OGB datasets (ogbn-arxiv, ogbn-proteins), PyG datasets (Cora, CiteSeer, PubMed, Coauthor-CS, Coauthor-Physics, Amazon-Computers, Amazon-Photo)

**Manual download**: Chameleon, Squirrel, Film, Deezer, Pokec — download from:
https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link

Or use the download script:
```bash
bash download_data.sh
```

For Chameleon and Squirrel, we use the [filtered splits](https://github.com/yandex-research/heterophilous-graphs/tree/main) that remove overlapping nodes.

## Running Experiments

### Method Names

| Method | Flag | Description |
|--------|------|-------------|
| SGFormer (baseline) | `--method sgformer` | Original SGFormer all-pair attention |
| PCGT (ours) | `--method pcgt` | Partition-conditioned graph transformer |

### Medium-Scale (Cora, CiteSeer, PubMed, Chameleon, Squirrel, Film, Deezer)

```bash
cd medium/

# SGFormer baseline on Cora
python main.py --method sgformer --dataset cora --backbone gcn \
    --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 5e-4 \
    --dropout 0.5 --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.2 --use_residual --no_feat_norm \
    --seed 123 --runs 10 --epochs 500

# PCGT on Cora (K=10 partitions)
python main.py --method pcgt --dataset cora --backbone gcn \
    --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 5e-4 \
    --dropout 0.5 --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.2 --use_residual --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500
```

### Large-Scale (ogbn-arxiv, ogbn-proteins, Pokec)

```bash
cd large/

# SGFormer baseline on ogbn-arxiv
python main.py --method sgformer --dataset ogbn-arxiv --metric acc \
    --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
    --gnn_num_layers 3 --trans_num_layers 1 \
    --seed 123 --runs 5 --epochs 1000 --eval_step 9

# PCGT on ogbn-arxiv (K=256 partitions)
python main.py --method pcgt --dataset ogbn-arxiv --metric acc \
    --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
    --gnn_num_layers 3 --trans_num_layers 1 \
    --num_partitions 256 --partition_method metis \
    --seed 123 --runs 3 --epochs 1000 --eval_step 9

# For large graphs (proteins, pokec), use mini-batch training:
python main-batch.py --method pcgt --dataset ogbn-proteins --metric rocauc \
    --num_partitions 256 --batch_size 10000 \
    --seed 123 --runs 3 --epochs 1000 --eval_step 9
```

### PCGT-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_partitions` | 10 (medium) / 100 (large) | Number of METIS graph partitions $K$ |
| `--partition_method` | `metis` | Partitioning algorithm (`metis`, `random`) |
| `--num_reps` | 4 | Number of representative nodes per partition (medium only) |
| `--no_pse` | False | Disable Partition Structural Encoding |
| `--local_only` | False | Ablation: only intra-partition attention |
| `--global_only` | False | Ablation: only cross-partition attention |

### Reproducing Paper Results

See `reproduce_paper_results.sh` for exact commands to reproduce all results, or run:
```bash
# Large-scale experiments (requires GPU)
cd large && bash run_all_large.sh pcgt
```

## Results

### Medium-Scale Node Classification (Accuracy %)

| Dataset | SGFormer | PCGT | $\Delta$ |
|---------|----------|------|---------|
| Cora | 84.50 | **84.80** | +0.30 |
| CiteSeer | 72.60 | **73.44** | +0.84 |
| PubMed | 80.30 | **80.46** | +0.16 |
| Chameleon | 44.90 | **48.09** | +3.19 |
| Squirrel | 41.80 | **45.14** | +3.34 |
| Deezer | 67.10 | **67.24** | +0.14 |
| Film | **37.90** | 37.69 | -0.21 |
| Coauthor-Physics | 96.45 | **96.61** | +0.16 |
| Amazon-Computers | 87.21 | **88.27** | +1.06 |
| Amazon-Photo | 94.68 | **94.90** | +0.22 |

### Large-Scale Node Classification

| Dataset | Metric | SGFormer | PCGT |
|---------|--------|----------|------|
| ogbn-arxiv (169K) | Accuracy | 72.63 ± 0.13 | 72.36 ± 0.20 |
| ogbn-proteins (132K) | ROC-AUC | 79.53 ± 0.38 | — |
| Pokec (1.6M) | Accuracy | 73.76 ± 0.24 | — |

> Large-scale results are being finalized. PCGT shows strongest gains on heterophilic graphs (Chameleon +3.19%, Squirrel +3.34%).

## Acknowledgements

This codebase is built upon [SGFormer](https://github.com/qitianwu/SGFormer) by Qitian Wu et al. (NeurIPS 2023). We thank the authors for their excellent open-source implementation.

```bibtex
@inproceedings{wu2023sgformer,
    title={SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations},
    author={Qitian Wu and Wentao Zhao and Chenxiao Yang and Hengrui Zhang and Fan Nie and Haitian Jiang and Yatao Bian and Junchi Yan},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2023}
}
```

