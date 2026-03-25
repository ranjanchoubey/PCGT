#!/bin/bash
# ============================================================================
# PCGT — Quick Test
# Runs a fast sanity check on Cora to verify everything works.
# Expected result: ~84% test accuracy (1 run, 100 epochs).
#
# Usage:
#   source venv/bin/activate   # if not already activated
#   bash quick_test.sh
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------- Check venv ----------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "ERROR: No virtual environment found. Run 'bash setup.sh' first."
        exit 1
    fi
fi

echo "============================================"
echo "  PCGT Quick Test — Cora Dataset"
echo "============================================"
echo ""

# ---------- 1. Import check ----------
log "Step 1/3: Checking imports..."
python -c "
import torch, torch_geometric, torch_sparse, torch_scatter, pymetis
print('  All imports OK')
"

# ---------- 2. Run PCGT on Cora ----------
log "Step 2/3: Training PCGT on Cora (1 run, 100 epochs)..."
echo ""
cd medium
python main.py \
    --dataset cora \
    --method pcgt \
    --backbone gcn \
    --lr 0.01 \
    --num_layers 2 \
    --hidden_channels 64 \
    --weight_decay 5e-4 \
    --dropout 0.4 \
    --ours_layers 1 \
    --use_graph \
    --graph_weight 0.8 \
    --ours_dropout 0.2 \
    --use_residual \
    --alpha 0.5 \
    --ours_weight_decay 0.001 \
    --no_feat_norm \
    --num_partitions 10 \
    --partition_method metis \
    --rand_split_class \
    --valid_num 500 \
    --test_num 1000 \
    --seed 123 \
    --runs 1 \
    --epochs 100 \
    --cpu
cd ..

# ---------- 3. Summary ----------
echo ""
echo "============================================"
log "Step 3/3: Done!"
echo ""
echo "  If you see 'Final Test: ~84%' above, everything works correctly."
echo "  Paper reports: 84.3 ± 0.4% (with 10 runs, 500 epochs)"
echo ""
echo "  To reproduce full paper results:"
echo "    cd medium && bash run.sh cora"
echo "============================================"
