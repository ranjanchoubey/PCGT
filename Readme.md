# PCGT: Partition-Conditioned Graph Transformer

<p align="center">
  <img src="assets/architecture.png" alt="PCGT Architecture" width="100%">
</p>

<p align="center">
  <b>PCGT</b> replaces expensive O(N²) global attention with <b>multi-resolution partition-aware attention</b>.<br>
  Local attention within partitions · Global attention via learned seeds · Linear complexity O(N·(N/K + KM))
</p>

<p align="center">
  <a href="#results">Results</a> •
  <a href="#visualizations">Visualizations</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#running-experiments">Experiments</a> •
  <a href="#acknowledgements">Acknowledgements</a>
</p>

---

## How It Works

1. **Partition** the graph into $K$ balanced groups using METIS
2. **Encode** each node with a learnable Partition Structural Encoding (PSE)
3. **Local attention** — exact softmax within each partition, $O(N^2/K)$
4. **Global attention** — cross-partition communication via $M$ learned seed vectors, $O(NMK)$
5. **Blend** local + global with learnable $\alpha$, add $\beta$-weighted self-connection
6. **Fuse** with a GCN branch via $\lambda_{\text{gw}}$ for the final prediction

Built upon [SGFormer](https://github.com/qitianwu/SGFormer) (Wu et al., NeurIPS 2023).

---

## Results

### Medium-Scale — 11 Benchmarks (Accuracy %)

| Dataset | SGFormer | **PCGT** | Δ |
|:--------|:--------:|:--------:|:-:|
| Cora | **84.50** ± 0.8 | 84.30 ± 0.4 | -0.20 |
| CiteSeer | 72.60 ± 0.2 | **73.10** ± 0.4 | +0.50 |
| PubMed | 80.30 ± 0.6 | **81.00** ± 0.6 | +0.70 |
| Film | 37.90 ± 1.1 | **38.00** ± 0.9 | +0.10 |
| Squirrel | 41.80 ± 2.2 | **45.50** ± 2.7 | **+3.70** |
| Chameleon | 44.90 ± 3.9 | **49.00** ± 2.8 | **+4.10** |
| Deezer | 67.10 ± 1.1 | **67.20** ± 0.7 | +0.10 |
| Coauthor-CS | 94.60 ± 0.5 | **95.10** ± 0.3 | +0.50 |
| Coauthor-Physics | 96.50 ± 0.2 | **96.80** ± 0.2 | +0.30 |
| Amazon-Computers | 87.20 ± 0.8 | **88.80** ± 0.7 | **+1.60** |
| Amazon-Photo | 94.70 ± 0.4 | **95.30** ± 0.4 | +0.60 |

> **PCGT wins 10/11 benchmarks.** Largest gains on heterophilic graphs: Chameleon +4.10%, Squirrel +3.70%.

### Large-Scale

| Dataset | Metric | SGFormer | **PCGT** |
|:--------|:------:|:--------:|:--------:|
| ogbn-arxiv (169K) | Accuracy | 72.63 ± 0.13 | 72.36 ± 0.20 |
| ogbn-proteins (132K) | ROC-AUC | 79.53 ± 0.38 | **80.47** ± 0.55 |
| Pokec (1.6M) | Accuracy | 73.76 ± 0.24 | **76.68** ± 0.24 |

> On **Pokec (1.6M nodes)**, PCGT outperforms SGFormer by **+2.92%**.

### Runtime (H100 GPU)

<p align="center">
  <img src="assets/runtime_comparison.png" alt="Runtime" width="80%">
</p>

---

## Visualizations

### t-SNE Embeddings — Cora

<p align="center">
  <img src="assets/cora_tsne.png" alt="Cora t-SNE" width="70%">
</p>

> t-SNE of raw features (left) vs PCGT learned embeddings (right) — PCGT produces well-separated clusters.

### Per-Class F1 Score — Cora

<p align="center">
  <img src="assets/cora_f1.png" alt="Cora Per-Class F1" width="55%">
</p>

### Partition-Level Analysis

<table>
<tr>
<td width="50%">

**Cora** (homophilic, $h=0.81$)

<img src="assets/cora_summary.png" alt="Cora Summary" width="100%">

Best partition: **88.2%** accuracy (90/102 test nodes correct)

<img src="assets/cora_partition_best.png" alt="Cora Best Partition" width="100%">
</td>
<td width="50%">

**Chameleon** (heterophilic, $h=0.23$)

<img src="assets/chameleon_summary.png" alt="Chameleon Summary" width="100%">

Best partition: **80.0%** accuracy (12/15 test nodes correct)

<img src="assets/chameleon_partition_best.png" alt="Chameleon Best Partition" width="100%">
</td>
</tr>
</table>

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ranjanchoubey/PCGT.git && cd PCGT
pip install -r requirements.txt

# Run PCGT on Cora (medium-scale)
cd medium
python main.py --method pcgt --dataset cora --backbone gcn \
    --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 5e-4 \
    --dropout 0.5 --use_graph --graph_weight 0.8 \
    --num_partitions 10 --seed 123 --runs 10 --epochs 500
```

---

## Repository Structure

```
PCGT/
├── medium/                # Medium-scale experiments
│   ├── main.py            # Entry point
│   ├── pcgt.py            # PCGT model
│   ├── sgformer.py        # SGFormer baseline
│   └── partition.py       # METIS partitioning
├── large/                 # Large-scale (arxiv, proteins, pokec)
│   ├── main.py            # Full-batch training
│   └── main-batch.py      # Mini-batch for >100K nodes
├── experiments/           # Logs, results, scripts
│   ├── logs/              # H100 GPU training logs
│   ├── results/           # Parsed results
│   └── MANIFEST.md        # Experiment documentation
├── visualization/         # Partition & analysis plots
├── data/                  # Datasets (auto-downloaded)
├── requirements.txt
├── SETUP.md               # Detailed setup & troubleshooting
└── reproduce_paper_results.sh
```

## Requirements

- Python ≥ 3.10 · PyTorch ≥ 2.0 · PyTorch Geometric ≥ 2.3 · pymetis · OGB ≥ 1.3

```bash
pip install -r requirements.txt
```

For GPU: install matching CUDA versions of torch-scatter, torch-sparse, torch-cluster from [PyG wheels](https://data.pyg.org/whl/).

## Datasets

**Auto-downloaded:** Cora, CiteSeer, PubMed, Coauthor-CS/Physics, Amazon-Computers/Photo, OGB datasets.

**Manual download:** Chameleon, Squirrel, Film, Deezer, Pokec → [Google Drive](https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link) or `bash download_data.sh`

---

## Running Experiments

### Method Flags

| Method | Flag | Description |
|--------|------|-------------|
| SGFormer (baseline) | `--method sgformer` | Original all-pair attention |
| **PCGT (ours)** | `--method pcgt` | Partition-conditioned attention |

### PCGT-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_partitions` | 10 / 100 | Number of METIS partitions $K$ |
| `--partition_method` | `metis` | Partitioning algorithm |
| `--num_reps` | 4 | Seed vectors per partition $M$ |
| `--no_pse` | False | Disable Partition Structural Encoding |
| `--local_only` | False | Ablation: local attention only |
| `--global_only` | False | Ablation: global attention only |

### Examples

```bash
# Medium-scale
cd medium && python main.py --method pcgt --dataset cora --backbone gcn \
    --num_partitions 10 --seed 123 --runs 10 --epochs 500

# Large-scale (mini-batch)
cd large && python main-batch.py --method pcgt --dataset ogbn-proteins \
    --num_partitions 256 --batch_size 10000 --runs 3 --epochs 1000
```

### Reproduce All Results

```bash
pip install -r requirements.txt && bash download_data.sh
cd medium && bash run.sh          # All 11 datasets
cd ../large && bash run.sh all    # arxiv, proteins, pokec
```

See `reproduce_paper_results.sh` for exact per-dataset hyperparameters.

---

## Acknowledgements

Built upon [SGFormer](https://github.com/qitianwu/SGFormer) by Qitian Wu et al. (NeurIPS 2023).

```bibtex
@inproceedings{wu2023sgformer,
    title={SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations},
    author={Qitian Wu and Wentao Zhao and Chenxiao Yang and Hengrui Zhang and Fan Nie and Haitian Jiang and Yatao Bian and Junchi Yan},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2023}
}
```

