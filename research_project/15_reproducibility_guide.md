# Reproducibility Guide - PCGT GPU Validation

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (optional; CPU mode available)
- ~10GB disk space for datasets

### One-Command Setup (Colab)
```bash
# Copy from colab_run.sh and paste into notebook cell
bash colab_run.sh
```

This script:
1. Clones/pulls PCGT repo
2. Installs dependencies
3. Downloads datasets from internet
4. Runs all 7 experiments with GPU acceleration
5. Logs results to local `results_gpu/` directory

---

## Full Reproduction Steps

### Step 1: Environment Setup

#### Option A: Colab (Recommended)
```python
# In Colab notebook cell
!bash /path/to/colab_run.sh
```

#### Option B: Local GPU (CUDA 11.8+)
```bash
git clone https://github.com/your-repo/PCGT.git
cd PCGT/medium

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pymetis scikit-learn
pip install pandas numpy scipy

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### Option C: CPU (Slower, for debugging)
```bash
cd PCGT/medium
python main.py --dataset cora --device cpu
```

### Step 2: Prepare Datasets

#### Auto-Download (Internet required)
```bash
# colab_run.sh does this automatically
python -c "from dataset import get_dataset; get_dataset('cora')"
```

#### Manual Pre-Upload (For Drive-based Colab)
1. Download datasets manually:
   - [Cora, CiteSeer, PubMed](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html)
   - [Chameleon, Film, Squirrel](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikipediaNetwork.html)
   - [Deezer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.LastFMAsia.html)

2. Upload to Google Drive: `/PCGT_datasets/data/`

3. In PCGT_Runner.ipynb, mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Symlink
   !ln -s /content/drive/MyDrive/PCGT_datasets/data /content/PCGT/data
   ```

### Step 3: Run Baseline Experiments

#### Single Dataset (Serial)
```bash
cd PCGT/medium
python main.py \
  --method pcgt \
  --dataset cora \
  --hidden_channels 64 \
  --num_layers 3 \
  --ours_layers 3 \
  --num_reps 4 \
  --dropout 0.6 \
  --weight_decay 5e-4 \
  --ours_dropout 0.6 \
  --ours_weight_decay 5e-4 \
  --runs 5 \
  --lr 0.01 \
  --epochs 200 \
  --early_stopping 50 \
  --device 0
```

#### All 7 Datasets (Via Script)
```bash
bash colab_run.sh  # or manually run:

for dataset in cora citeseer pubmed chameleon film squirrel deezer-europe; do
  python main.py --method pcgt --dataset $dataset --runs 5 --device 0
done
```

### Step 4: Deezer Hyperparameter Attack

Run with Deezer-specific configurations (from PCGT_Runner.ipynb cells 27–30):

```bash
# Batch 1: Capacity Reduction
BASE="python -B main.py --method pcgt --dataset deezer-europe --device 0"

# B1: Aggressive capacity reduction
$BASE --hidden_channels 32 --num_reps 2 --runs 3

# B2: Moderate reduction
$BASE --hidden_channels 48 --num_reps 2 --runs 3

# B3: Extreme reduction
$BASE --hidden_channels 32 --num_reps 1 --runs 3

# Batch 2: Heavy regularization
# B4–B6: dropout 0.7–0.8, weight_decay 5e-3–1e-2

# ... (see PCGT_Runner.ipynb for all 12 configs)
```

---

## Expected Results

### Baseline (5 runs per dataset)

| Dataset | Expected Accuracy | Status |
|---------|------------------|--------|
| Cora | 83.80 ± 1.21 | ✅ Match SGFormer |
| CiteSeer | 73.44 ± 0.21 | ✅ Beat SGFormer (+0.84) |
| PubMed | 80.46 ± 0.64 | ✅ Beat SGFormer (+0.16) |
| Chameleon | 48.09 ± 2.39 | ✅ Beat SGFormer (+3.19) |
| Film | 37.69 ± 0.98 | ✅ Match SGFormer |
| Squirrel | 45.14 ± 2.29 | ✅ Beat SGFormer (+3.34) |
| Deezer | 64.94 ± 0.85 | ❌ Below SGFormer (-2.16) *Attack in progress* |

**Overall**: 4 wins, 2 matched, 1 loss → "6/7 datasets beat or match SGFormer"

### Deezer Attack (Top 3 Expected)
- **B1** (capacity ↓): Potentially +1–2% (simpler model for binary task)
- **B5** (heavy reg): Potentially +1–3% (reduce 97%→64% overfitting)
- **B12** (pure GCN): ~64.5% (establish GCN ceiling)

Success threshold: Any config > 65.5 closes the gap acceptably.

---

## Output Formats

### Logs (Per Run)
```
Epoch 001/200 | Train Loss 1.234 | Val Acc 75.3% | Test Acc 73.4%
...
Epoch 050/200 | Early Stopping! (Best Val Acc: 84.2% @ epoch 45)
Final Test Accuracy: 83.80%
```

### Results Summary (colab_run.sh Output)
```
=== FINAL RESULTS ===
Cora: 83.80 ± 1.21 (Seed 0-4)
CiteSeer: 73.44 ± 0.21
...
Overall Score: 4/7 datasets beat SGFormer
```

### Drive-Backed Logs (PCGT_Runner.ipynb)
- Location: `/content/drive/MyDrive/PCGT/results_colab/<timestamp>/`
- Files: `deezer_attack/DZ-B1.txt`, ..., `DZ-B12.txt`
- Auto-parsing cell extracts best config

---

## Debugging Tips

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False: Update NVIDIA drivers or use --device cpu
```

### Out of Memory
```bash
# Reduce batch size implicitly via early stopping + smaller hidden_channels
python main.py --hidden_channels 32 --num_reps 1
```

### Slow Training (CPU)
```bash
# Use --device 0 instead of --device cpu
# Or enable mixed precision (fp16) if available
```

### Dataset Not Found
```bash
# Ensure data/ folder structure matches expected layout:
# data/
#   Planetoid/
#     cora/
#     citeseer/
#     pubmed/
#   geom-gcn/
#     chameleon/
#     film/
#     squirrel/
#   deezer/
#     deezer-europe.mat
```

---

## Code Files Reference

| File | Purpose |
|------|---------|
| **main.py** | Entry point, training loop, evaluation |
| **pcgt.py** | PCGT architecture (PCGTConvLayer, PCGT) |
| **models.py** | Model factory (PCGT + baselines) |
| **dataset.py** | Dataset loading + train/val/test splits |
| **data_utils.py** | Utilities (graph partitioning, normalization) |
| **parse.py** | Argument parser definition |
| **colab_run.sh** | Colab automation script |
| **PCGT_Runner.ipynb** | Interactive notebook for Drive-based data |

---

## Hyperparameter Details

### Key Parameters
```
--hidden_channels       Hidden dimension (default: 64)
--num_layers            Number of PCGT layers (default: 3)
--ours_layers           Transformer layers (default: 3)
--num_reps              Representatives per partition (default: 4)
--partition_method      'kmeans', 'metis', 'random' (default: kmeans)
--num_partitions        Number of partitions (default: 50)
--graph_weight          GCN blend [0,1] (default: 0.5)
--dropout               Attention dropout (default: 0.6)
--weight_decay          L2 regularization (default: 5e-4)
--ours_dropout          Transformer dropout (default: 0.6)
--ours_weight_decay     Transformer L2 (default: 5e-4)
--lr                    Learning rate (default: 0.01)
--runs                  Number of random seeds (default: 5)
--epochs                Max training epochs (default: 200)
--early_stopping        Patience for early stopping (default: 50)
--device                0 (GPU) or 'cpu' (default: 0)
```

### Deezer Attack Variants

**Standard baseline** (reference):
```
--hidden_channels 64 --num_reps 4 --dropout 0.6 --weight_decay 5e-4
--graph_weight 0.5
```

**Capacity reduction** (B1–B3):
```
B1: --hidden_channels 32 --num_reps 2
B2: --hidden_channels 48 --num_reps 2
B3: --hidden_channels 32 --num_reps 1
```

**Heavy regularization** (B4–B6):
```
B4: --dropout 0.7 --weight_decay 5e-3 --ours_weight_decay 0.02 --ours_dropout 0.5
B5: --dropout 0.8 --weight_decay 0.01 --ours_dropout 0.6 --ours_weight_decay 0.02
B6: --dropout 0.6 --weight_decay 5e-4 --ours_dropout 0.8 --ours_weight_decay 0.03
```

**Architecture variants** (B7–B10):
```
B7: + --batch_norm + --graph_weight 0.9
B8: + --batch_norm + --graph_weight 0.3
B9: --num_partitions 20
B10: --partition_method random
```

**Feature & baseline** (B11–B12):
```
B11: + --no_feat_norm
B12: --graph_weight 1.0
```

---

## Validation Checklist

- [ ] CUDA available (if using GPU): `torch.cuda.is_available()` → True
- [ ] Datasets downloaded or symlinked to `data/`
- [ ] Dependencies installed (torch, torch_geometric, pymetis, scikit-learn)
- [ ] Baseline run on Cora completes without errors (should take 2–5 min GPU)
- [ ] Test accuracy in range ~83–84% (±1.5 std dev)
- [ ] Early stopping triggers around epoch 50–100
- [ ] All 7 datasets run to completion with results logged
- [ ] Deezer attack 12 configs execute (3 runs each)
- [ ] Best Deezer config identified + validated with 5 runs

---

## References

- **PCGT Paper**: [Link to paper]
- **SGFormer Baseline**: [SGFormer GitHub](https://github.com/...)
- **torch_geometric Docs**: https://pytorch-geometric.readthedocs.io
- **FINAL_CONFIGS.sh**: Official hyperparameter reference + results table