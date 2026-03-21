# PCGT Project Setup Guide

Complete setup guide for the PCGT (Simplified Graph Transformers) project on macOS.

## ⚡ Quick Start (5 minutes)

```bash
# 1. Create and activate virtual environment (Python 3.10)
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Install PyTorch and sparse libraries (IN THIS ORDER)
pip install torch torchvision torchaudio
pip install --no-build-isolation torch-scatter torch-sparse
pip install -r requirements.txt

# 3. Download datasets
python download_data.py

# 4. Run training
cd medium
python main.py --backbone gcn --dataset cora --epochs 10 --cpu --runs 1
```

---

## System Requirements

| Requirement | Details |
|---|---|
| **OS** | macOS 11+ (Intel or Apple Silicon) |
| **Python** | 3.10 or 3.11 (3.10 recommended) |
| **RAM** | 16GB minimum, 32GB recommended |
| **Disk** | 20GB free space |
| **⚠️ AVOID** | Python 3.12 (torch-sparse build failures) |

---

## Detailed Setup Steps

### Step 1: Install Python 3.10

#### Option A: Homebrew (Recommended)
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10
brew install python@3.10

# Verify
python3.10 --version
```

#### Option B: Conda
```bash
conda create -n pcgt python=3.10 -y
conda activate pcgt
```

### Step 2: Navigate to Project
```bash
cd /path/to/PCGT
```

### Step 3: Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

> **Note:** Virtual environment creation can take 1–3 minutes on macOS (normal behavior).

### Step 4: Install PyTorch

```bash
# MacOS automatically uses CPU version (optimal for all Macs)
pip install torch torchvision torchaudio
```

### Step 5: Install Sparse Libraries ⭐ CRITICAL

**This order and the `--no-build-isolation` flag are essential on macOS:**

```bash
pip install --no-build-isolation torch-scatter torch-sparse
```

**Why?** Build subprocesses can't access PyTorch without this flag. Omitting it causes build failures.

### Step 6: Install Remaining Dependencies

```bash
pip install torch-geometric
pip install -r requirements.txt
```

**If PyG installation fails:**
```bash
# Try pre-built wheels
pip install torch-geometric -f https://data.pyg.org/whl/torch_2.10.0+cpu.html

# Or use conda (most reliable on macOS)
conda install -c pytorch::pytorch-geometric
```

### Step 7: Verify Installation

```bash
python -c "import torch; import torch_geometric; import torch_sparse; print('✓ All imports successful')"
```

### Step 8: Download Datasets (Optional)

```bash
python download_data.py
# Downloads Cora, Citeseer, Pubmed (~100MB total)
```

---

## Running Training

### Quick Test (5 minutes, CPU)
```bash
cd medium
python main.py --backbone gcn --dataset cora --epochs 10 --cpu --runs 1
```

Expected output:
```
Final Test: 61.80%
```

### Full Training Run
```bash
cd medium
python main.py \
    --backbone gcn \
    --dataset cora \
    --lr 0.01 \
    --num_layers 4 \
    --hidden_channels 64 \
    --weight_decay 5e-4 \
    --dropout 0.5 \
    --method ours \
    --use_graph \
    --graph_weight 0.8 \
    --epochs 100 \
    --runs 5 \
    --cpu \
    --data_dir ../data/
```

### Using Run Scripts
```bash
# Edit medium/run.sh and remove GPU references (--device 3 -> --cpu)
cd medium
bash run.sh
```

---

## Troubleshooting

### ❌ torch_sparse Installation Fails

**Error:** `ModuleNotFoundError: No module named 'torch_sparse'`

**Solutions:**

1. **Ensure you install in the correct order:**
   ```bash
   pip install torch torchvision torchaudio
   pip install --no-build-isolation torch-scatter torch-sparse
   ```

2. **Install Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   # Then retry: pip install --no-build-isolation torch-scatter torch-sparse
   ```

3. **Use conda instead (most reliable):**
   ```bash
   conda install -c conda-forge torch-scatter torch-sparse
   ```

### ❌ "Virtual Environment Creation Interrupted"

On some macOS systems, `python3.10 -m venv venv` can stall for 1–3 minutes during the `ensurepip` phase. This is normal.

**Solution:** Simply recreate if interrupted:
```bash
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### ❌ Permission Denied Errors

```bash
chmod +x venv/bin/activate
chmod +x venv/bin/python
```

### ❌ GPU/Device Not Available

**This is expected and normal on macOS.** PyTorch doesn't support CUDA on macOS. Always use:
```bash
python main.py ... --cpu
```

### ❌ Data Download from Google Drive Fails

Try manual download:
1. Visit: https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link
2. Download and extract to `data/` directory

Or use gdown directly:
```bash
pip install gdown
gdown "https://drive.google.com/uc?id=FILE_ID"
```

### ❌ Memory Error During Training

Reduce model complexity:
```bash
python main.py ... --hidden_channels 32 --num_layers 2 --batch_size 32
```

---

## Why These Specific Dependencies?

### PyTorch & Sparse Libraries

- **torch (2.10.0+):** Deep learning framework with support for graphs
- **torch-scatter & torch-sparse:** Required for efficient GNN operations
  - Used throughout `models.py` and `ours.py`
  - Must build with PyTorch available (hence `--no-build-isolation`)
  - Pre-built wheels exist for Python 3.10 but not 3.12

### PyTorch Geometric

- Provides GNN layers (GCNConv, GATConv, etc.)
- Dataset utilities (Planetoid, OGB)
- Graph transformations

### Other Dependencies

- **scikit-learn, pandas:** Data processing
- **ogb:** Large-scale graph benchmarks
- **performer-pytorch:** Efficient transformer implementations
- **tensorboard:** Training visualization
- **matplotlib:** Plotting results

### Why Not Modify Original Code?

The project assumes all dependencies are installed. **Modifying `.py` files** to make imports optional would:
- Create runtime errors later when code tries to use those modules
- Make debugging harder
- Violate the principle of "setup tools should fix setup problems"

The **correct approach** is fixing the dependency installation process (which we did) rather than working around it in code.

---

---

## Available Datasets

The PCGT project supports multiple datasets across different categories. Here's what works out-of-the-box:

👉 **Full dataset guide**: See [DATASETS.md](DATASETS.md) for complete reference with all 20+ supported datasets.

### ✅ Planetoid Datasets (Recommended for Quick Start)

Automatically downloaded and work immediately:

| Dataset | Nodes | Edges | Classes | Command |
|---|---|---|---|---|
| **Cora** | 2,708 | 5,429 | 7 | `python main.py --dataset cora --epochs 100 --cpu` |
| **Citeseer** | 3,327 | 4,732 | 6 | `python main.py --dataset citeseer --epochs 100 --cpu` |
| **Pubmed** | 19,717 | 44,338 | 3 | `python main.py --dataset pubmed --epochs 100 --cpu` |

**Quick test all three:**
```bash
cd medium
for dataset in cora citeseer pubmed; do
  echo "Testing $dataset..."
  python main.py --dataset $dataset --epochs 10 --cpu --runs 1
done
```

### 📊 Other Supported Datasets

These require **manual data download**. See section below.

#### Wikipedia/Heterophilic Graphs
- **Chameleon**: `python main.py --dataset chameleon --epochs 100 --cpu`
- **Squirrel**: `python main.py --dataset squirrel --epochs 100 --cpu`

#### Heterophily Datasets
- **Roman Empire**: `python main.py --dataset roman-empire --epochs 100 --cpu`
- **Amazon Ratings**: `python main.py --dataset amazon-ratings --epochs 100 --cpu`
- **Minesweeper**: `python main.py --dataset minesweeper --epochs 100 --cpu`
- **Tolokers**: `python main.py --dataset tolokers --epochs 100 --cpu`
- **Questions**: `python main.py --dataset questions --epochs 100 --cpu`

#### Social Networks
- **Deezer Europe**: `python main.py --dataset deezer-europe --epochs 100 --cpu`

#### Large-Scale OGB Datasets
- **ogbn-arxiv**: `python main.py --dataset ogbn-arxiv --epochs 100 --cpu`
- **ogbn-products**: `python main.py --dataset ogbn-products --epochs 100 --cpu` (requires GPU)
- **ogbn-papers100M**: `python main.py --dataset ogbn-papers100M --epochs 100` (requires GPU)

---

## Downloading Additional Datasets

### Verify & Auto-Download Planetoid
```bash
# Verify all Planetoid datasets exist
python download_data.py --verify-only

# Re-download if needed
python download_data.py --datasets cora citeseer pubmed
```

### Manual Dataset Downloads

For non-Planetoid datasets, download from the official sources:

**1. Wiki/Heterophilic Datasets:**
```bash
# Download geom-gcn format datasets
mkdir -p data/geom-gcn/film
# Download from: https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/geom_gcn_datasets.py
```

**2. Heterophily Benchmark Datasets:**
```bash
# Download from: https://github.com/yushun-yuan/Heterophily-Benchmark
# Extract to: data/heterophilous-graphs/
```

**3. Deezer Social Network:**
```bash
# Download from: https://github.com/benedekrozemberczki/node2vec
# Save to: data/deezer/deezer-europe.mat
```

---

## Directory Structure Reference

Expected structure after download:

```
data/
├── Planetoid/               # Auto-downloaded
│   ├── cora/
│   │   ├── raw/
│   │   └── processed/
│   ├── citeseer/
│   └── pubmed/
├── geom-gcn/                # Manual (film dataset)
│   └── film/
├── wiki_new/                # Manual (chameleon, squirrel)
│   ├── chameleon/
│   └── squirrel/
├── heterophilous-graphs/    # Manual (heterophily datasets)
│   ├── roman_empire.npz
│   ├── amazon_ratings.npz
│   └── ...
└── deezer/                  # Manual (social networks)
    └── deezer-europe.mat
```

---

## FAQ: Datasets

**Q: Can I train on all datasets at once?**  
A: Yes. Use bash loop: `for ds in cora citeseer pubmed chameleon squirrel; do ... done`

**Q: Do I need all datasets?**  
A: No. Start with Cora (2.7k nodes) for quick prototyping, then try Pubmed (19.7k nodes) for scaling tests.

**Q: Which dataset should I use for my paper?**  
A: Typically researchers report on Cora, Citeseer, and Pubmed (standard citation benchmarks).

**Q: Can I use custom datasets?**  
A: Yes. Add your dataset loader in `medium/dataset.py` in the `load_nc_dataset()` function. Follow the `load_geom_gcn_dataset()` pattern.

**Q: Why can't I download non-Planetoid datasets automatically?**  
A: These datasets have different formats and license requirements. Manual download is safer for compliance.

**Q:How do I fix "Invalid dataname" error?**  
A: The dataset loader doesn't recognize your dataset name. Check spelling and make sure you've set up the data files in `data/` directory.

---

## Project Structure

After successful setup, your directory should look like:

```
PCGT/
├── venv/                    # Virtual environment
├── data/                    # Datasets (created by download_data.py)
│   └── Planetoid/
│       ├── cora/
│       ├── citeseer/       
│       └── pubmed/
├── medium/                  # Main training code
│   ├── main.py
│   ├── models.py
│   ├── dataset.py
│   ├── ours.py
│   └── run.sh
├── large/                   # Large-scale experiments
├── 100M/                    # 100M parameter experiments
├── requirements.txt
├── download_data.py
├── SETUP.md
├── Readme.md
└── RESEARCH_PLAN.md
```

---

## Verifying Everything Works

**Run this complete verification:**

```bash
# Activate venv
source venv/bin/activate

# Check imports
python -c "import torch, torch_geometric, torch_sparse; print('✓ Imports OK')"

# Download data
python download_data.py

# Test all three main datasets
cd medium
for dataset in cora citeseer pubmed; do
  echo "Testing $dataset..."
  python main.py --dataset $dataset --epochs 5 --cpu --runs 1
done

# Expected result: All three datasets train successfully
```

---

## FAQ

**Q: Can I use Python 3.12?**  
A: Not recommended. torch-sparse frequently fails to build on Python 3.12. Use 3.10 or 3.11.

**Q: Do I need GPU?**  
A: No. CPU mode works fine with `--cpu` flag. Training is slower but accurate.

**Q: How long does setup take?**  
A: 10–15 minutes total (most time spent downloading packages and building sparse libs).

**Q: Why does venv creation stall?**  
A: macOS ensurepip phase can take 1–3 minutes. It's not frozen; just wait.

**Q: Can I use conda instead of venv?**  
A: Yes. Replace venv steps with `conda create -n pcgt python=3.10` and `conda activate pcgt`.

---

## Still Having Issues?

1. Delete the venv and start fresh: `rm -rf venv`
2. Follow the Quick Start section exactly
3. Check that you're using Python 3.10: `python3.10 --version`
4. Ensure torch installs before sparse libs
5. Use `--no-build-isolation` for sparse libs

If issues persist, check the [Community Issues](https://github.com/qitianwu/SGFormer/issues) section or open a new issue with:
- Your Python version (`python --version`)
- Your macOS version (`uname -a`)
- The full error message
