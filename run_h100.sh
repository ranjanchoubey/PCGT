#!/bin/bash
# ============================================================================
# PCGT Large-Scale Experiments — H100 Parallel Runner
# Runs ALL 6 remaining experiments in parallel on H100 (80GB VRAM)
#
# Usage:
#   bash run_h100.sh setup      # Install deps + download data (~10 min)
#   bash run_h100.sh run        # Launch all 6 experiments in parallel
#   bash run_h100.sh status     # Check progress of all jobs
#   bash run_h100.sh results    # Print final results summary
#
# Expected total time: ~2.5-3 hrs on H100
# Expected VRAM peak: ~40-45 GB (fits in 80GB easily)
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LARGE_DIR="$REPO_DIR/large"
LOG_DIR="$REPO_DIR/logs/h100"
DATA_DIR="$REPO_DIR/data"
PID_DIR="$REPO_DIR/.pids"

# Auto-detect python
if command -v /system/conda/miniconda3/envs/cloudspace/bin/python &>/dev/null; then
    PY="/system/conda/miniconda3/envs/cloudspace/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    PY="python"
fi

mkdir -p "$LOG_DIR" "$PID_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================================
# SETUP
# ============================================================================
do_setup() {
    log "=== SETUP: Installing dependencies ==="
    cd "$REPO_DIR"

    $PY -m pip install --quiet torch-geometric ogb pymetis scipy scikit-learn networkx 2>&1 | tail -5

    # Install PyG scatter/sparse (try prebuilt wheels first)
    TORCH_VER=$($PY -c "import torch; print(torch.__version__.split('+')[0])")
    CUDA_VER=$($PY -c "import torch; print(torch.version.cuda.replace('.',''))")
    $PY -m pip install --quiet torch-scatter torch-sparse torch-cluster \
        -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu${CUDA_VER}.html" 2>&1 | tail -3 || \
    $PY -m pip install --quiet torch-scatter torch-sparse torch-cluster 2>&1 | tail -3

    log "=== Checking imports ==="
    $PY -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f'VRAM: {mem / 1e9:.1f} GB')
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from ogb.nodeproppred import NodePropPredDataset
import pymetis
print('All imports OK')
"
    if [ $? -ne 0 ]; then
        log "ERROR: Import check failed. Fix dependencies before running."
        exit 1
    fi

    log "=== Pre-downloading OGB datasets ==="
    $PY -c "
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched

from ogb.nodeproppred import NodePropPredDataset
import os
data_dir = '$DATA_DIR'
for name in ['ogbn-arxiv', 'ogbn-proteins', 'ogbn-products']:
    print(f'  {name}...', end=' ', flush=True)
    try:
        NodePropPredDataset(name=name, root=os.path.join(data_dir, 'ogb'))
        print('OK')
    except Exception as e:
        print(f'ERROR: {e}')
"

    # Check pokec data
    if [ -f "$DATA_DIR/pokec/pokec.mat" ]; then
        log "Pokec data found"
    else
        log "WARNING: Pokec data not found at $DATA_DIR/pokec/pokec.mat"
        log "Download it or copy from another machine"
    fi

    log "=== SETUP COMPLETE ==="
}

# ============================================================================
# RUN ONE EXPERIMENT (called internally)
# ============================================================================
run_one() {
    local name="$1"
    local script="$2"
    shift 2

    local logfile="$LOG_DIR/${name}.log"
    log "LAUNCH: $name"

    cd "$LARGE_DIR"
    $PY -u "$script" "$@" --data_dir "$DATA_DIR" --device 0 > "$logfile" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_DIR/${name}.pid"
    echo "  PID=$pid -> $logfile"
}

# ============================================================================
# LAUNCH ALL 6 EXPERIMENTS IN PARALLEL
# ============================================================================
do_run() {
    log "=========================================="
    log "LAUNCHING ALL EXPERIMENTS IN PARALLEL"
    log "=========================================="

    # Clean old PIDs
    rm -f "$PID_DIR"/*.pid

    # --- PROTEINS (small, hidden=64) ---
    run_one "proteins_sgformer" main-batch.py \
        --method sgformer --dataset ogbn-proteins --metric rocauc \
        --lr 0.01 --hidden_channels 64 \
        --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --batch_size 10000 --seed 123 --runs 5 --epochs 1000 --eval_step 9

    run_one "proteins_pcgt_k256" main-batch.py \
        --method pcgt --dataset ogbn-proteins --metric rocauc \
        --lr 0.01 --hidden_channels 64 \
        --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --num_partitions 256 --partition_method metis \
        --batch_size 10000 --seed 123 --runs 3 --epochs 1000 --eval_step 9

    # --- POKEC (medium, hidden=64) ---
    run_one "pokec_sgformer" main-batch.py \
        --method sgformer --dataset pokec --rand_split --metric acc \
        --lr 0.01 --hidden_channels 64 \
        --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9

    run_one "pokec_pcgt_k500" main-batch.py \
        --method pcgt --dataset pokec --rand_split --metric acc \
        --lr 0.01 --hidden_channels 64 \
        --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --num_partitions 500 --partition_method metis \
        --batch_size 100000 --seed 123 --runs 3 --epochs 1000 --eval_step 9

    # --- AMAZON2M (large, hidden=256) ---
    run_one "amazon2m_sgformer" main-batch.py \
        --method sgformer --dataset amazon2m --metric acc \
        --lr 0.01 --hidden_channels 256 \
        --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9

    run_one "amazon2m_pcgt_k1000" main-batch.py \
        --method pcgt --dataset amazon2m --metric acc \
        --lr 0.01 --hidden_channels 256 \
        --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --num_partitions 1000 --partition_method metis \
        --batch_size 100000 --seed 123 --runs 3 --epochs 1000 --eval_step 9

    log "=========================================="
    log "ALL 6 JOBS LAUNCHED — use 'bash run_h100.sh status' to monitor"
    log "=========================================="
    log ""
    log "Quick check:  watch -n 30 'bash run_h100.sh status'"
}

# ============================================================================
# STATUS: check all running jobs
# ============================================================================
do_status() {
    echo "======================================================================"
    echo "H100 EXPERIMENT STATUS — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"

    # GPU utilization
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
            awk -F', ' '{printf "  GPU: %s%% util | %s/%s MB VRAM\n", $1, $2, $3}'
        echo ""
    fi

    local all_done=true
    for name in proteins_sgformer proteins_pcgt_k256 pokec_sgformer pokec_pcgt_k500 amazon2m_sgformer amazon2m_pcgt_k1000; do
        local logfile="$LOG_DIR/${name}.log"
        local pidfile="$PID_DIR/${name}.pid"

        if [ ! -f "$logfile" ]; then
            printf "  %-25s  NOT STARTED\n" "$name"
            all_done=false
            continue
        fi

        # Check if has final aggregated result
        local final=$(grep -o "[0-9]* runs:.*" "$logfile" 2>/dev/null | tail -1)
        if [ -n "$final" ]; then
            printf "  %-25s  DONE   %s\n" "$name" "$final"
            continue
        fi

        # Count completed runs
        local runs_done=$(grep -c "^Run [0-9]*:" "$logfile" 2>/dev/null || echo 0)
        local last_epoch=$(grep -o "Epoch: [0-9]*" "$logfile" 2>/dev/null | tail -1)

        # Check if process is alive
        local status="RUNNING"
        if [ -f "$pidfile" ]; then
            local pid=$(cat "$pidfile")
            if ! kill -0 "$pid" 2>/dev/null; then
                status="DEAD?"
            fi
        fi

        printf "  %-25s  %-8s runs=%s  %s\n" "$name" "$status" "$runs_done" "$last_epoch"
        all_done=false
    done

    echo "======================================================================"
    if $all_done; then
        echo "  ALL EXPERIMENTS COMPLETE! Run: bash run_h100.sh results"
    fi
    echo ""
}

# ============================================================================
# RESULTS: print final summary for paper
# ============================================================================
do_results() {
    echo "======================================================================"
    echo "FINAL RESULTS — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
    echo ""

    for name in proteins_sgformer proteins_pcgt_k256 pokec_sgformer pokec_pcgt_k500 amazon2m_sgformer amazon2m_pcgt_k1000; do
        local logfile="$LOG_DIR/${name}.log"
        if [ ! -f "$logfile" ]; then
            printf "  %-25s  NO LOG\n" "$name"
            continue
        fi

        local highest=$(grep "Highest Test:" "$logfile" 2>/dev/null | tail -1)
        local final=$(grep "Final Test:" "$logfile" 2>/dev/null | tail -1)

        if [ -n "$highest" ]; then
            printf "  %-25s  %s  |  %s\n" "$name" "$highest" "$final"
        else
            printf "  %-25s  INCOMPLETE\n" "$name"
        fi
    done

    echo ""
    echo "======================================================================"
    echo "Copy logs to Mac:  scp -r <h100>:$(pwd)/logs/h100/ logs/h100/"
    echo "======================================================================"
}

# ============================================================================
# MAIN
# ============================================================================
CMD="${1:-help}"
case "$CMD" in
    setup)   do_setup ;;
    run)     do_run ;;
    status)  do_status ;;
    results) do_results ;;
    help|*)
        echo "Usage: bash run_h100.sh {setup|run|status|results}"
        echo ""
        echo "  setup   — Install deps, download OGB data (~10 min)"
        echo "  run     — Launch all 6 experiments in parallel"
        echo "  status  — Check progress of all jobs"
        echo "  results — Print final results summary"
        ;;
esac
