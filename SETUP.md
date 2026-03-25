# Setup Guide

## Quick Setup (Recommended)

**Prerequisite:** Python 3.10, 3.11, or 3.12 must be installed.

```bash
# Install Python 3.10 if you don't have it:
#   macOS:  brew install python@3.10
#   Ubuntu: sudo apt install python3.10 python3.10-venv

git clone https://github.com/ranjanchoubey/PCGT.git
cd PCGT
bash setup.sh        # installs everything + downloads data
bash quick_test.sh   # runs a fast test on Cora (~1 min)
```

That's it. If you see `Final Test: ~84%`, everything works.

---

## Manual Setup (Step-by-Step)

### Requirements

- Python 3.10, 3.11, or 3.12 (3.10 recommended)
- macOS or Linux
- GPU optional (CPU works fine for testing)

### Step 1: Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Step 2: Install PyTorch

```bash
# CPU (macOS / Linux without GPU)
pip install torch torchvision

# GPU (Linux with CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Graph Libraries

```bash
pip install torch-scatter torch-sparse
```

### Step 4: Install All Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download Datasets

```bash
bash download_data.sh
```

### Step 6: Verify

```bash
python -c "import torch; import torch_geometric; import torch_sparse; print('OK')"
```

### Step 7: Quick Test

```bash
bash quick_test.sh
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `torch-sparse` build fails | Install PyTorch first, then: `pip install --no-build-isolation torch-scatter torch-sparse` |
| `No module named 'pymetis'` | `pip install pymetis` |
| macOS: no GPU detected | Expected — use `--cpu` flag (already set in quick_test.sh) |
| `python3` not found | Install Python 3.10 from python.org |
