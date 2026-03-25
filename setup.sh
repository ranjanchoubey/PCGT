#!/bin/bash
# ============================================================================
# PCGT — One-Click Setup
# Installs all dependencies and downloads datasets.
#
# Usage:
#   bash setup.sh
#
# Requirements: Python 3.10, 3.11, or 3.12.
#   macOS: brew install python@3.10
#   Ubuntu: sudo apt install python3.10 python3.10-venv
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[ERROR] $*" >&2; }

# ---------- 1. Find a compatible Python (3.10 or 3.11) ----------
find_python() {
    for cmd in python3.10 python3.11 python3; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local ver
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ "$ver" == "3.10" || "$ver" == "3.11" || "$ver" == "3.12" ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_CMD=$(find_python || true)

if [ -z "$PYTHON_CMD" ]; then
    err "Python 3.10, 3.11, or 3.12 is required but not found."
    err "Your system has: $(python3 --version 2>&1 || echo 'no python3')"
    err ""
    err "Install Python 3.10:"
    err "  macOS:  brew install python@3.10"
    err "  Ubuntu: sudo apt install python3.10 python3.10-venv"
    exit 1
fi

log "Found compatible Python: $($PYTHON_CMD --version)"

# ---------- 2. Virtual environment ----------
if [ ! -d "venv" ]; then
    log "Creating virtual environment..."
    "$PYTHON_CMD" -m venv venv
else
    log "Virtual environment already exists"
fi
source venv/bin/activate
log "Using: $(python --version) at $(which python)"

pip install --upgrade pip setuptools wheel -q

# ---------- 3. PyTorch ----------
log "Installing PyTorch..."
pip install torch torchvision -q

# ---------- 4. Sparse libraries ----------
log "Installing torch-scatter and torch-sparse..."
pip install --no-build-isolation torch-scatter torch-sparse -q

# ---------- 5. Remaining dependencies ----------
log "Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt -q

# ---------- 6. Verify imports ----------
log "Verifying installation..."
python -c "
import torch
import torch_geometric
import torch_sparse
import torch_scatter
import pymetis
print(f'  torch           {torch.__version__}')
print(f'  torch-geometric {torch_geometric.__version__}')
print(f'  pymetis         OK')
print('All imports OK!')
"

# ---------- 7. Download datasets ----------
log "Downloading datasets (this may take a few minutes)..."
bash download_data.sh

log "Setup complete! You can now run experiments."
echo ""
echo "Quick test:"
echo "  source venv/bin/activate"
echo "  bash quick_test.sh"
