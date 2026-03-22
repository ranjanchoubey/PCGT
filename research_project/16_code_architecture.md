# Code Architecture & Key Files Reference

**Last Updated**: March 22, 2026  
**Project Root**: `/Users/vn59a0h/thesis/PCGT/`

---

## Directory Structure

```
PCGT/
├── medium/                      # Main codebase (medium-scale experiments)
│   ├── main.py                  # Entry point, training loop
│   ├── pcgt.py                  # PCGT architecture implementation
│   ├── models.py                # Model factory + baselines
│   ├── dataset.py               # Dataset loading + preprocessing
│   ├── data_utils.py            # Utility functions (partitioning, normalization)
│   ├── parse.py                 # Argument parser definitions
│   ├── run.sh                   # Local bash runner
│   └── __pycache__/
│
├── large/                       # Large-scale experiments (ogbn-arxiv)
│   ├── main.py, ours.py, ...
│
├── 100M/                        # 100M edge experiments
│   ├── main.py, ours.py, ...
│
├── data/                        # Dataset directory
│   ├── Planetoid/               # Cora, CiteSeer, PubMed
│   ├── geom-gcn/               # Chameleon, Film, Squirrel
│   ├── ogb/                    # OGB datasets
│   ├── deezer/                 # Deezer-Europe
│   └── pokec/, wiki_new/, ...
│
├── write_paper/                 # Paper + research documentation
│   ├── setup_research.py        # Research setup tool (generates research_project/)
│
├── research_project/            # This folder! Research documentation
│   ├── 00_status_summary.md     # Project status (you are here)
│   ├── 04_method.md             # PCGT architecture
│   ├── 05_design_choices.md     # Design rationale
│   ├── 06_experiments_plan.md   # Experiment protocol
│   ├── 07_results_raw.md        # GPU results
│   ├── 11_experiment_logs.md    # Detailed run logs
│   ├── 15_reproducibility_guide.md  # How to reproduce
│   └── 01_problem.md, 02_background.md, ...  (other docs)
│
├── PCGT_Runner.ipynb            # Interactive Colab notebook (Deezer attack integrated)
├── colab_run.sh                 # One-command Colab runner (all 7 datasets)
├── FINAL_CONFIGS.sh             # Official hyperparameter reference + results table
├── SETUP.md                     # Installation guide
├── Readme.md                    # Project overview
├── requirements.txt             # Dependencies
└── LICENSE
```

---

## Core Code Files (medium/)

### 1. main.py — Training Entry Point
**What it does**: 
- Loads dataset and creates train/val/test splits
- Initializes model with parsed hyperparameters
- Runs training loop with early stopping
- Evaluates on test set with multiple random seeds (`--runs N`)
- Logs epoch-by-epoch progress and final results

**Key Functions**:
```python
def main():
    args = parse_args()                    # Parse command-line arguments
    device = set_device(args.device)       # GPU/CPU setup
    data = load_dataset(args.dataset)      # Load + split dataset
    model = get_model(args.method, ...)    # Initialize model (PCGT)
    
    for seed in range(args.runs):
        train_loop(model, data, args)      # Train with early stopping
        test_acc = evaluate(model, data)   # Evaluate on test set
        record_result(test_acc)
    
    print_final_statistics()               # Mean ± std over runs
```

**Hyperparameter Flow**:
```
parse.py (argparse definitions)
    ↓
main.py (--hidden_channels 64 --num_layers 3 ...)
    ↓
pcgt.py / models.py (Model(hidden_channels=64, num_layers=3, ...))
    ↓
Training loop
```

### 2. pcgt.py — Architecture Implementation
**What it does**: Implements PCGT layers and full model

**Key Classes**:

#### PCGTConvLayer
```python
class PCGTConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_partitions, num_reps, ...):
        self.local_attn = ...              # Local (within-partition) softmax
        self.pool_seeds = nn.Parameter()   # Learned global seeds K × M × D
        self.alpha_logit = nn.Parameter()  # Learnable blend weight
        self.beta = nn.Parameter()         # Learnable self-connection
        self.gcn = GCNConv(in_channels, out_channels)
        
    def forward(self, x, partition_labels, edge_index, batch=None):
        # Step 1: Local attention (intra-partition softmax)
        local_out = compute_local_attention(x, partition_labels)  # O(N²/K)
        
        # Step 2: Global attention (via learned seeds)
        global_out = compute_global_attention(x, self.pool_seeds)  # O(N·M·K)
        
        # Step 3: Blend
        alpha = sigmoid(self.alpha_logit)
        attn_out = alpha * local_out + (1 - alpha) * global_out
        
        # Step 4: Self-weighting
        out = self.beta * x + (1 - self.beta) * attn_out
        
        # Step 5: GCN blend
        gcn_out = self.gcn(x, edge_index)
        out = self.graph_weight * gcn_out + (1 - self.graph_weight) * out
        
        return LayerNorm(out + x)
```

#### PCGT (Full Model)
```python
class PCGT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers, num_partitions, num_reps, ...):
        self.convs = nn.ModuleList([
            PCGTConvLayer(...) for _ in range(num_layers)
        ])
        self.mlp = MLP(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, partition_indices, ...):
        for conv in self.convs:
            x = conv(x, partition_labels=self.partition_labels, edge_index=edge_index)
        
        logits = self.mlp(x)
        return logits
    
    def set_partition_info(self, partition_indices, partition_labels):
        """Called once per dataset"""
        self.partition_indices = partition_indices
        self.partition_labels = partition_labels
```

### 3. models.py — Model Factory
**What it does**: Creates PCGT or baseline models based on `--method` flag

**Key Function**:
```python
def get_model(method, in_channels, hidden_channels, out_channels, dataset_name, ...):
    if method == 'pcgt':
        return PCGT(in_channels, hidden_channels, out_channels,
                   num_layers=args.num_layers, ours_layers=args.ours_layers,
                   num_partitions=args.num_partitions, num_reps=args.num_reps,
                   partition_method=args.partition_method,
                   graph_weight=args.graph_weight)
    elif method == 'gcn':
        return GCN(...)
    elif method == 'sgformer':
        return SGFormer(...)
    ...
```

### 4. dataset.py — Dataset Loading
**What it does**: Loads datasets and creates train/val/test splits

**Supported Datasets**:
- Planetoid: Cora, CiteSeer, PubMed
- GeomGCN: Chameleon, Film, Squirrel
- Custom: Deezer-Europe
- OGB: ogbn-arxiv, ogbn-products

**Key Function**:
```python
def load_dataset(name, root='data/', split='0.6_0.2_0.2'):
    if 'planetoid' in name.lower():
        data = Planetoid(root, name)
    elif name == 'deezer-europe':
        data = load_deezer()
    ...
    
    # Create splits
    train_mask, val_mask, test_mask = create_splits(data, split_config)
    return data, train_mask, val_mask, test_mask
```

### 5. data_utils.py — Utilities
**What it does**: Graph partitioning, normalization, feature preprocessing

**Key Functions**:
```python
def compute_partitions(edge_index, num_nodes, num_partitions, method='kmeans', features=None):
    """
    Partition graph into K groups
    Returns: partition_indices, partition_labels
    """
    if method == 'kmeans':
        # Cluster nodes in feature space
        kmeans = KMeans(n_clusters=num_partitions)
        partition_ids = kmeans.fit_predict(features)
    elif method == 'metis':
        # Use METIS to minimize edge cuts
        partition_ids = metis_partition(...)
    elif method == 'random':
        partition_ids = np.random.randint(0, num_partitions, num_nodes)
    
    # Convert to one-hot labels (N × K)
    partition_labels = torch.eye(num_partitions)[partition_ids]
    return partition_ids, partition_labels

def normalize_features(features):
    """L2-normalize features per node"""
    return F.normalize(features, p=2, dim=1)
```

### 6. parse.py — Argument Definitions
**What it does**: Defines all command-line arguments with default values

**Key Arguments**:
```python
parser.add_argument('--method', choices=['pcgt', 'gcn', 'sgformer', ...], default='pcgt')
parser.add_argument('--dataset', default='cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--graph_weight', type=float, default=0.5)
parser.add_argument('--partition_method', choices=['kmeans', 'metis', 'random'], default='kmeans')
parser.add_argument('--num_partitions', type=int, default=50)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--device', default='0')
... (see FINAL_CONFIGS.sh for full list)
```

---

## Execution Files

### colab_run.sh — Colab One-Liner
**Purpose**: Self-contained script for Colab that does everything

**What it does**:
1. Clones/pulls PCGT repo
2. Installs dependencies (torch, torch_geometric, pymetis, scikit-learn)
3. Downloads datasets from internet
4. Runs all 7 experiments with GPU
5. Logs results to `results_gpu/` directory

**Usage**:
```bash
bash colab_run.sh
```

**Limitations**: Auto-downloads datasets (slow on Colab internet)

### PCGT_Runner.ipynb — Interactive Notebook
**Purpose**: Colab notebook for pre-uploaded Drive data (no downloads)

**Structure**:
- Cells 1–13: Setup (clone, install, mount Drive, verify GPU)
- Cells 15–25: Standard datasets (Cora through Squirrel)
- Cells 27–33: **Deezer attack section** (12 configs, summary parser, re-runner)

**Data Setup**:
```python
# Symlink instead of download
!ln -s /content/drive/MyDrive/PCGT_datasets/data /content/PCGT/data
```

**Deezer Attack Cells** (27–33):
- Cell 27: Markdown header
- Cells 28–30: DZ-B1 through DZ-B12 configs
- Cell 31: Summary parser (extract best config)
- Cells 32–33: Best config re-runner with 5 runs

### FINAL_CONFIGS.sh — Hyperparameter Reference
**Purpose**: Official hyperparameter table + results logging

**Contents**:
```bash
# Standard baseline (all 7 datasets)
HIDDEN_CHANNELS=64
NUM_LAYERS=3
DROPOUT=0.6
WEIGHT_DECAY=5e-4
...

# Results table
Cora: 83.80 ± 1.21 (5 GPU runs)
CiteSeer: 73.44 ± 0.21
PubMed: 80.46 ± 0.64
...

# Deezer attack log
# B1: (capacity reduction) → TBD (awaiting GPU results)
# B2: ...
```

**Usage**: Single source of truth for all hyperparameters across scripts

---

## Data Flow Diagram

```
Command-Line Arguments (parse.py)
    ↓
main.py
    ├→ Load Dataset (dataset.py) → data, splits
    │
    ├→ Compute Partitions (data_utils.py) 
    │   └→ partition_indices, partition_labels
    │
    ├→ Initialize Model (models.py → pcgt.py)
    │   └→ model.set_partition_info(partition_indices)
    │
    ├→ Training Loop (multiple seeds, early stopping)
    │   └→ for epoch in range(max_epochs):
    │       ├→ Forward pass: pcgt.py (PCGTConvLayer × 3)
    │       ├→ Compute loss: CrossEntropyLoss
    │       ├→ Backprop + optimize
    │       └→ Evaluate on validation set
    │
    └→ Final Evaluation on Test Set
        └→ Record accuracy, std dev
```

---

## Hyperparameter Mapping

| Feature | parse.py arg | pcgt.py usage | Purpose |
|---------|---|---|---|
| Model size | `--hidden_channels` | `out_channels` of PCGTConvLayer | Hidden dimension |
| Depth | `--num_layers` | `nn.ModuleList([PCGTConvLayer(...) for _ in range(num_layers)])` | Number of transformer blocks |
| Global pooling | `--num_reps` | `pool_seeds.shape[1]` (M) | Reps per partition |
| Partitions | `--num_partitions` | `pool_seeds.shape[0]` (K) | Number of partitions |
| Reg. (model) | `--dropout` | `Dropout(dropout)` in PCGTConvLayer | Attention dropout |
| Reg. (weights) | `--weight_decay` | Optimizer: `weight_decay=wd` | L2 regularization |
| Graph blend | `--graph_weight` | `gw * gcn_out + (1-gw) * pcgt_out` | GCN vs PCGT mix |
| Partition method | `--partition_method` | `data_utils.compute_partitions(method=...)` | How to partition graph |

---

## Key Concepts

### Partition Structural Encoding (PSE)
Each node's hidden representation includes partition ID embedding:
```python
x = x + partition_pe[partition_id]  # Add once per layer
```

### Learnable Seeds
Instead of fixed pooling, PCGT learns where to attend globally:
```python
pool_seeds: torch.nn.Parameter(K, M, D)  # Per-layer learnable
global_out = softmax(Q @ pool_seeds @ pool_seeds.T) V
```

### Alpha Blending (Local ↔ Global)
Model learns the balance:
```python
alpha = sigmoid(alpha_logit)  # Per layer
out = alpha * local_attention + (1-alpha) * global_attention
```

### Beta Self-Weighting (Feature ↔ Aggregate)
Model learns importance of self vs neighbors:
```python
beta = beta_param  # Per layer, can be unconstrained
out = beta * x + (1-beta) * aggregated
```

---

## Testing & Validation

### Single Dataset Test
```bash
cd /Users/vn59a0h/thesis/PCGT/medium
python main.py --dataset cora --runs 1  # Quick single-seed test
# Expected: ~84% accuracy in ~1 min
```

### Full Baseline Reproduction
```bash
python main.py --dataset cora --runs 5 --device 0
# Expected: 83.80 ± 1.21 (matches GPU results in results_raw.md)
```

### Deezer Attack (Batch 1 Example)
```bash
# B1: Capacity reduction
python main.py --dataset deezer-europe \
  --hidden_channels 32 --num_reps 2 --runs 3

# B4: Heavy regularization
python main.py --dataset deezer-europe \
  --dropout 0.7 --weight_decay 5e-3 --ours_weight_decay 0.02 --ours_dropout 0.5 --runs 3
```

---

## Modification Guide

### To change model capacity:
Edit `parse.py`:
```python
parser.add_argument('--hidden_channels', type=int, default=32)  # Was 64
```
Then run:
```bash
python main.py --dataset cora --hidden_channels 32
```

### To add new dataset:
Edit `dataset.py`:
```python
def load_dataset(name, ...):
    if name == 'my_new_dataset':
        # Load your data here
        data = MyCustomDataset(...)
    ...
```

### To change partition method:
Edit `parse.py`:
```python
parser.add_argument('--partition_method', 
                    choices=['kmeans', 'metis', 'random', 'spectral'],  # Add 'spectral'
                    default='kmeans')
```
Then implement in `data_utils.py`:
```python
elif method == 'spectral':
    partition_ids = spectral_clustering(...)
```

---

## Debugging Commands

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU only (debug)
python main.py --dataset cora --device cpu

# Single quick run (test pipeline)
python main.py --dataset cora --runs 1 --epochs 2 --early_stopping 1

# Verbose output
python main.py --dataset cora --runs 1  # Already prints per-epoch stats

# Profile memory
python -c "import torch; torch.cuda.memory_summary()"
```

---

## Reference: Command Templates

### Baseline (All Datasets)
```bash
python main.py \
  --method pcgt \
  --dataset DATASET_NAME \
  --hidden_channels 64 \
  --num_layers 3 \
  --ours_layers 3 \
  --num_reps 4 \
  --partition_method kmeans \
  --num_partitions 50 \
  --graph_weight 0.5 \
  --dropout 0.6 \
  --weight_decay 5e-4 \
  --ours_dropout 0.6 \
  --ours_weight_decay 5e-4 \
  --lr 0.01 \
  --runs 5 \
  --epochs 200 \
  --early_stopping 50 \
  --device 0
```

### Deezer Attack (Capacity Reduction - B1)
```bash
python main.py \
  --method pcgt \
  --dataset deezer-europe \
  --hidden_channels 32 \
  --num_reps 2 \
  --runs 3 \
  --device 0
```

### Deezer Attack (Heavy Regularization - B5)
```bash
python main.py \
  --method pcgt \
  --dataset deezer-europe \
  --dropout 0.8 \
  --weight_decay 0.01 \
  --ours_dropout 0.6 \
  --ours_weight_decay 0.02 \
  --runs 3 \
  --device 0
```

---

This reference captures the complete codebase structure and execution flow.