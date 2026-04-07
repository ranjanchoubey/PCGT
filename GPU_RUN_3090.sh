#!/bin/bash
# ============================================================================
# PCGT GPU Experiments — RTX 3090 (24GB VRAM)
# 
# Two experiments:
#   1. ogbn-arxiv 10 runs  (full-batch, ~4GB VRAM, ~1 hour)
#   2. amazon2m SGFormer+PCGT (mini-batch, ~10-14GB VRAM, ~3-5 hours)
#
# Usage:
#   bash GPU_RUN_3090.sh setup     # Download data (run once)
#   bash GPU_RUN_3090.sh arxiv     # Run ogbn-arxiv 10 runs
#   bash GPU_RUN_3090.sh amazon    # Run amazon2m (SGFormer + PCGT)
#   bash GPU_RUN_3090.sh all       # Run everything sequentially
#   bash GPU_RUN_3090.sh check     # Check log progress
#
# Requirements:
#   - NVIDIA RTX 3090 (24GB)
#   - ~32GB+ system RAM (amazon2m needs it for METIS)
#   - PyTorch 2.x, PyG, pymetis, OGB
#
# Logs go to: logs/gpu_3090/
# ============================================================================

set -uo pipefail

# === Config ===
DEVICE="${DEVICE:-0}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LARGE_DIR="$REPO_DIR/large"
LOG_DIR="$REPO_DIR/logs/gpu_3090"
DATA_DIR="$REPO_DIR/data/"

# Auto-detect python
if command -v python3 &>/dev/null; then
    PY="python3"
else
    PY="python"
fi

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# === GPU Check ===
check_gpu() {
    log "Checking GPU..."
    $PY -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'VRAM: {mem:.1f} GB')
" || { log "ERROR: PyTorch/CUDA check failed"; exit 1; }
}

# === Setup: Download data ===
do_setup() {
    log "=== Downloading datasets ==="
    cd "$LARGE_DIR"
    $PY -c "
from ogb.nodeproppred import NodePropPredDataset
print('Downloading ogbn-arxiv...')
NodePropPredDataset(name='ogbn-arxiv', root='$DATA_DIR/ogb')
print('Done: ogbn-arxiv')
print('Downloading ogbn-products (for amazon2m)...')
NodePropPredDataset(name='ogbn-products', root='$DATA_DIR/ogb')
print('Done: ogbn-products')
"
    log "=== Data download complete ==="
}

# ============================================================================
# EXPERIMENT 1: ogbn-arxiv — 10 runs (full-batch)
# Current paper: 3 runs → 72.63 ± 0.08. Goal: 10 runs for tighter CI.
# VRAM: ~4 GB. Time: ~1 hour.
# ============================================================================
run_arxiv() {
    log "=== Starting ogbn-arxiv experiments ==="
    cd "$LARGE_DIR"
    
    # --- SGFormer baseline (10 runs) ---
    log "Running SGFormer on ogbn-arxiv (10 runs)..."
    $PY -u main.py \
        --method sgformer --dataset ogbn-arxiv --metric acc \
        --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
        --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --device $DEVICE --seed 123 --runs 10 --epochs 1000 --eval_step 9 \
        --display_step 100 --data_dir "$DATA_DIR" \
        2>&1 | tee "$LOG_DIR/arxiv_sgformer_10run.log"
    
    log "SGFormer done. Exit code: $?"
    
    # --- PCGT (10 runs, K=256) ---
    log "Running PCGT on ogbn-arxiv (10 runs, K=256)..."
    $PY -u main.py \
        --method pcgt --dataset ogbn-arxiv --metric acc \
        --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
        --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --num_partitions 256 --partition_method metis \
        --device $DEVICE --seed 123 --runs 10 --epochs 1000 --eval_step 9 \
        --display_step 100 --data_dir "$DATA_DIR" \
        2>&1 | tee "$LOG_DIR/arxiv_pcgt_k256_10run.log"
    
    log "PCGT done. Exit code: $?"
    log "=== ogbn-arxiv complete ==="
}

# ============================================================================
# EXPERIMENT 2: amazon2m (ogbn-products) — mini-batch
# 2.4M nodes, 62M edges, 100-dim features, 47 classes.
# Uses batch_size=50K (conservative for 24GB).
# VRAM: ~10-14 GB. Time: ~3-5 hours total.
# ============================================================================
run_amazon() {
    log "=== Starting Amazon2M experiments ==="
    cd "$LARGE_DIR"
    
    # --- SGFormer baseline (5 runs) ---
    log "Running SGFormer on Amazon2M (5 runs, batch=50K)..."
    $PY -u main-batch.py \
        --method sgformer --dataset amazon2m --metric acc \
        --lr 0.01 --hidden_channels 256 \
        --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --batch_size 50000 --device $DEVICE --seed 123 --runs 5 --epochs 1000 \
        --eval_step 9 --display_step 100 --data_dir "$DATA_DIR" \
        2>&1 | tee "$LOG_DIR/amazon2m_sgformer_5run.log"
    
    log "SGFormer done. Exit code: $?"
    
    # --- PCGT (3 runs, K=1000) ---
    log "Running PCGT on Amazon2M (3 runs, K=1000, batch=50K)..."
    log "NOTE: METIS partitioning on 2.4M nodes takes ~5-15 min. Be patient."
    $PY -u main-batch.py \
        --method pcgt --dataset amazon2m --metric acc \
        --lr 0.01 --hidden_channels 256 \
        --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --num_partitions 1000 --partition_method metis \
        --batch_size 50000 --device $DEVICE --seed 123 --runs 3 --epochs 1000 \
        --eval_step 9 --display_step 100 --data_dir "$DATA_DIR" \
        2>&1 | tee "$LOG_DIR/amazon2m_pcgt_k1000_3run.log"
    
    log "PCGT done. Exit code: $?"
    log "=== Amazon2M complete ==="
}

# ============================================================================
# Log checker
# ============================================================================
do_check() {
    echo "=== GPU 3090 Experiment Logs ==="
    echo "Log directory: $LOG_DIR"
    echo ""
    for f in "$LOG_DIR"/*.log; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        lines=$(wc -l < "$f")
        # Extract final results if present
        final=$(grep -A5 "All runs:" "$f" 2>/dev/null | tail -5)
        echo "--- $fname ($lines lines) ---"
        if [ -n "$final" ]; then
            echo "$final"
        else
            echo "  (still running or no final results yet)"
            tail -1 "$f" 2>/dev/null
        fi
        echo ""
    done
}

# ============================================================================
# Main dispatch
# ============================================================================
case "${1:-help}" in
    setup)   check_gpu; do_setup ;;
    arxiv)   check_gpu; run_arxiv ;;
    amazon)  check_gpu; run_amazon ;;
    all)     check_gpu; run_arxiv; run_amazon ;;
    check)   do_check ;;
    *)
        echo "Usage: bash GPU_RUN_3090.sh {setup|arxiv|amazon|all|check}"
        echo ""
        echo "  setup   - Download ogbn-arxiv + ogbn-products data"
        echo "  arxiv   - Run ogbn-arxiv 10 runs (~1 hr, ~4GB VRAM)"
        echo "  amazon  - Run amazon2m SGFormer+PCGT (~3-5 hrs, ~12GB VRAM)"
        echo "  all     - Run everything sequentially (~4-6 hrs)"
        echo "  check   - Show log progress and final results"
        ;;
esac
