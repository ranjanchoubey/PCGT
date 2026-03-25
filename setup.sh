#!/bin/bash
# ============================================================================
# PCGT — One-Click Setup
# Installs all dependencies and downloads datasets.
#
# Usage:
#   bash setup.sh
#
# Requirements: Python 3.10 or 3.11 installed on your system.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------- 1. Virtual environment ----------
if [ ! -d "venv" ]; then
    log "Creating virtual environment..."
    python3 -m venv venv
else
    log "Virtual environment already exists"
fi
source venv/bin/activate
log "Using Python: $(python --version) at $(which python)"

pip install --upgrade pip setuptools wheel -q

# ---------- 2. PyTorch ----------
log "Installing PyTorch..."
pip install torch torchvision -q

# ---------- 3. Sparse libraries ----------
log "Installing torch-scatter and torch-sparse..."
pip install torch-scatter torch-sparse -q

# ---------- 4. Remaining dependencies ----------
log "Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt -q

# ---------- 5. Verify imports ----------
log "Verifying installation..."
python -c "
import torch
import torch_geometric
import torch_sparse
import torch_scatter
import pymetis
print(f'  torch          {torch.__version__}')
print(f'  torch-geometric {torch_geometric.__version__}')
print(f'  pymetis         OK')
print('All imports OK!')
"

# ---------- 6. Download datasets ----------
log "Downloading datasets (this may take a few minutes)..."
bash download_data.sh

log "Setup complete! You can now run experiments."
echo ""
echo "Quick test:"
echo "  source venv/bin/activate"
echo "  bash quick_test.sh"
