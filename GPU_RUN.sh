#!/bin/bash
# ============================================================
# GPU_RUN.sh — Single script to run on GPU machine
# ============================================================
# INSTRUCTIONS:
#   1. Clone the repo:     git clone https://github.com/ranjanchoubey/PCGT.git && cd PCGT
#   2. Run this script:    bash GPU_RUN.sh
#   3. When done, copy logs back:
#      scp -r PCGT/logs/large/ your_local:~/thesis/PCGT/logs/large/
#      OR just copy the logs/large/ folder to your local PCGT/logs/large/
#
# This script:
#   - Installs all dependencies
#   - Downloads OGB datasets (auto)
#   - Runs PCGT on proteins, pokec, amazon2m
#   - Saves all logs to logs/large/
#   - Prints final summary
#
# Expected time: ~3-4 hours on H100
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/large"
DATA_DIR="$SCRIPT_DIR/data"
DEVICE="${DEVICE:-0}"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ============================================================
# STEP 1: Install dependencies
# ============================================================
log "=== STEP 1: Installing dependencies ==="

# Detect python
if command -v python3 &>/dev/null; then
    PY="python3"
else
    PY="python"
fi
PYU="$PY -u"

echo "Python: $($PY --version)"
echo "PyTorch: $($PY -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# Install requirements
$PY -m pip install -q torch-geometric ogb pymetis scipy scikit-learn gdown 2>/dev/null || true

# Install torch-scatter/sparse (try PyG wheels first)
TORCH_VER=$($PY -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$($PY -c "import torch; print(torch.version.cuda.replace('.',''))" 2>/dev/null || echo "cpu")
$PY -m pip install -q torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu${CUDA_VER}.html" 2>/dev/null || \
$PY -m pip install -q torch-scatter torch-sparse 2>/dev/null || true

# Verify critical imports
$PY -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
from torch_geometric.utils import to_undirected
print('PyG OK')
from torch_scatter import scatter
print('torch-scatter OK')
import pymetis
print('pymetis OK')
from ogb.nodeproppred import NodePropPredDataset
print('OGB OK')
print()
print('ALL IMPORTS OK — ready to run')
" 2>&1 | tee "$LOG_DIR/setup.log"

if [ $? -ne 0 ]; then
    log "ERROR: Import check failed. Fix dependencies and re-run."
    exit 1
fi

# ============================================================
# STEP 2: Verify data (auto-downloads OGB if missing)
# ============================================================
log "=== STEP 2: Checking datasets ==="

$PY -c "
import os
data_dir = '$DATA_DIR'

# OGB datasets auto-download
from ogb.nodeproppred import NodePropPredDataset
for name in ['ogbn-proteins', 'ogbn-products']:
    try:
        d = NodePropPredDataset(name=name, root=os.path.join(data_dir, 'ogb'))
        print(f'{name}: OK ({d[0][0][\"num_nodes\"]} nodes)')
    except Exception as e:
        print(f'{name}: DOWNLOADING... (first run)')
        try:
            d = NodePropPredDataset(name=name, root=os.path.join(data_dir, 'ogb'))
            print(f'{name}: OK')
        except Exception as e2:
            print(f'{name}: ERROR - {e2}')

# Pokec check
pokec_path = os.path.join(data_dir, 'pokec', 'pokec.mat')
if os.path.exists(pokec_path):
    print(f'pokec: OK')
else:
    pokec_npy = os.path.join(data_dir, 'pokec', 'node_feat.npy')
    if os.path.exists(pokec_npy):
        print(f'pokec: OK (npy format)')
    else:
        print(f'pokec: MISSING — need pokec.mat in {data_dir}/pokec/')
" 2>&1 | tee -a "$LOG_DIR/setup.log"

# ============================================================
# STEP 3: Run experiments
# ============================================================

run_exp() {
    local name="$1"
    local script="$2"
    shift 2
    local logfile="$LOG_DIR/${name}.log"

    # Skip if already completed
    if [ -f "$logfile" ] && grep -q "Final Test:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        grep "Final Test:" "$logfile" | tail -1
        return 0
    fi

    log "START $name"
    echo "Log: $logfile"
    cd "$SCRIPT_DIR/large"
    if $PYU "$script" "$@" --data_dir "$DATA_DIR" --device $DEVICE 2>&1 | tee "$logfile"; then
        log "DONE $name"
    else
        log "DONE $name (exit code $?)"
    fi
    cd "$SCRIPT_DIR"
    echo ""
}

log "=== STEP 3: Running PCGT experiments ==="
log "Priority: proteins → pokec → amazon2m"
log "GPU device: $DEVICE"
echo ""

# ── Experiment 1: ogbn-proteins (132K nodes, ROC-AUC) ──
# SGFormer published: 79.53 ± 0.38
# Expected time: ~1.5-2h
run_exp "proteins_pcgt_k256" main-batch.py \
    --method pcgt --dataset ogbn-proteins --metric rocauc \
    --lr 0.01 --hidden_channels 64 \
    --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
    --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
    --trans_use_residual --trans_use_weight --trans_use_bn \
    --use_graph --graph_weight 0.5 \
    --num_partitions 256 --partition_method metis \
    --batch_size 10000 --seed 123 --runs 3 --epochs 1000 --eval_step 9

# ── Experiment 2: pokec (1.6M nodes, Accuracy) ──
# SGFormer published: 73.76 ± 0.24
# Expected time: ~45min-1h
run_exp "pokec_pcgt_k500" main-batch.py \
    --method pcgt --dataset pokec --rand_split --metric acc \
    --lr 0.01 --hidden_channels 64 \
    --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
    --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
    --trans_use_residual --trans_use_weight --trans_use_bn \
    --use_graph --graph_weight 0.5 \
    --num_partitions 500 --partition_method metis \
    --batch_size 100000 --seed 123 --runs 3 --epochs 1000 --eval_step 9

# ── Experiment 3: amazon2m (2.4M nodes, Accuracy) ──
# SGFormer published: 89.09 ± 0.10
# Expected time: ~1-1.5h (1 run only to save time)
run_exp "amazon2m_pcgt_k1000" main-batch.py \
    --method pcgt --dataset amazon2m --metric acc \
    --lr 0.01 --hidden_channels 256 \
    --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. \
    --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
    --trans_use_residual --trans_use_weight --trans_use_bn \
    --use_graph --graph_weight 0.5 \
    --num_partitions 1000 --partition_method metis \
    --batch_size 100000 --seed 123 --runs 1 --epochs 1000 --eval_step 9

# ============================================================
# STEP 4: Summary
# ============================================================
log "=== STEP 4: Final Summary ==="
echo ""
echo "======================================================================"
echo "RESULTS"
echo "======================================================================"
for logf in "$LOG_DIR"/*.log; do
    [ -f "$logf" ] || continue
    name=$(basename "$logf" .log)
    [[ "$name" == "setup" ]] && continue
    final=$(grep "Final Test:" "$logf" 2>/dev/null | tail -1)
    highest=$(grep "Highest Test:" "$logf" 2>/dev/null | tail -1)
    if [ -n "$final" ]; then
        echo "  $name:"
        echo "    $highest"
        echo "    $final"
    else
        echo "  $name: INCOMPLETE (check $logf)"
    fi
    echo ""
done
echo "======================================================================"
echo ""
echo "Logs saved in: $LOG_DIR/"
ls -la "$LOG_DIR/"
echo ""
log "=== ALL DONE ==="
echo ""
echo "NEXT: Copy logs/large/ back to your local machine:"
echo "  scp -r $(hostname):$(pwd)/logs/large/ ~/thesis/PCGT/logs/large/"
