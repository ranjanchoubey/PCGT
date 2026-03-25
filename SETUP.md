# Detailed Setup Guide

See the main [Readme.md](Readme.md) for a quick start. This guide covers platform-specific details.

## Requirements

- Python 3.10 or 3.11 (3.10 recommended; avoid 3.12)
- 16 GB RAM minimum
- GPU optional (CUDA 11.8+ for GPU acceleration)

## Installation

### 1. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch

```bash
# CPU
pip install torch torchvision torchaudio

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Sparse Libraries

```bash
pip install --no-build-isolation torch-scatter torch-sparse
```

> The `--no-build-isolation` flag is required so the build can find PyTorch.

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Datasets

```bash
bash download_data.sh
```

Cora, CiteSeer, PubMed, Coauthor, Amazon datasets are auto-downloaded.
Chameleon, Squirrel, Film, Deezer, Pokec require manual download (see Readme.md).

### 6. Verify

```bash
python -c "import torch; import torch_geometric; import torch_sparse; print('All imports OK')"
```

## Troubleshooting

### torch-sparse build fails

1. Ensure correct install order (PyTorch first, then sparse libs with `--no-build-isolation`)
2. On macOS: install Xcode CLI tools first (`xcode-select --install`)
3. Alternative: `conda install -c conda-forge torch-scatter torch-sparse`

### GPU not available

Expected on macOS. Always use `--cpu` flag. For GPU training, use a Linux machine with CUDA.
