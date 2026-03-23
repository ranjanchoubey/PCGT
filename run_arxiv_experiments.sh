#!/bin/bash
# PCGT ogbn-arxiv experiment runner for Lightning Studio / any GPU server
# Usage: bash run_arxiv_experiments.sh [experiment_name]
# Examples:
#   bash run_arxiv_experiments.sh all          # run all experiments
#   bash run_arxiv_experiments.sh sgformer     # SGFormer baseline only
#   bash run_arxiv_experiments.sh a1           # PCGT A1 only
#   bash run_arxiv_experiments.sh a4           # PCGT A4 (random) only
#   bash run_arxiv_experiments.sh best         # Best config 5-run validation
#   bash run_arxiv_experiments.sh check        # Check status of running/completed experiments

set -euo pipefail

# ============================================================
# Configuration — adjust these for your environment
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
LARGE_DIR="$REPO_ROOT/large"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/}"
LOG_DIR="$REPO_ROOT/logs"

# Auto-detect Python with PyTorch
if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" 2>/dev/null; then
    PY=python3
elif [ -f "/system/conda/miniconda3/envs/cloudspace/bin/python" ]; then
    PY=/system/conda/miniconda3/envs/cloudspace/bin/python
else
    PY=python
fi

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# Common args (shared by all experiments)
# ============================================================
COMMON="--dataset ogbn-arxiv --metric acc \
    --lr 0.001 --hidden_channels 256 \
    --use_graph \
    --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
    --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
    --trans_use_residual --trans_use_weight --trans_use_bn \
    --seed 123 --epochs 1000 --eval_step 9 --device 0 \
    --data_dir $DATA_DIR"

# ============================================================
# Experiment definitions
# ============================================================
run_sgformer() {
    local logfile="$LOG_DIR/sgformer_baseline.log"
    log "Starting SGFormer baseline (5 runs) → $logfile"
    cd "$LARGE_DIR"
    $PY main.py --method sgformer $COMMON \
        --graph_weight 0.5 \
        --runs 5 2>&1 | tee "$logfile"
    log "SGFormer done. Results:"
    grep -E 'Final Test' "$logfile" | tail -1
}

run_a1() {
    local logfile="$LOG_DIR/pcgt_a1_k256_gw05.log"
    log "Starting PCGT A1: K=256, METIS, gw=0.5 (5 runs) → $logfile"
    cd "$LARGE_DIR"
    $PY main.py --method pcgt $COMMON \
        --graph_weight 0.5 \
        --num_partitions 256 --partition_method metis \
        --runs 5 2>&1 | tee "$logfile"
    log "PCGT A1 done. Results:"
    grep -E 'Final Test' "$logfile" | tail -1
}

run_a2() {
    local logfile="$LOG_DIR/pcgt_a2_k500_gw05.log"
    log "Starting PCGT A2: K=500, METIS, gw=0.5 (3 runs) → $logfile"
    cd "$LARGE_DIR"
    $PY main.py --method pcgt $COMMON \
        --graph_weight 0.5 \
        --num_partitions 500 --partition_method metis \
        --runs 3 2>&1 | tee "$logfile"
    log "PCGT A2 done. Results:"
    grep -E 'Final Test' "$logfile" | tail -1
}

run_a3() {
    local logfile="$LOG_DIR/pcgt_a3_k256_gw08.log"
    log "Starting PCGT A3: K=256, METIS, gw=0.8 (3 runs) → $logfile"
    cd "$LARGE_DIR"
    $PY main.py --method pcgt $COMMON \
        --graph_weight 0.8 \
        --num_partitions 256 --partition_method metis \
        --runs 3 2>&1 | tee "$logfile"
    log "PCGT A3 done. Results:"
    grep -E 'Final Test' "$logfile" | tail -1
}

run_a4() {
    local logfile="$LOG_DIR/pcgt_a4_random_k256.log"
    log "Starting PCGT A4: K=256, RANDOM, gw=0.5 (3 runs) → $logfile"
    cd "$LARGE_DIR"
    $PY main.py --method pcgt $COMMON \
        --graph_weight 0.5 \
        --num_partitions 256 --partition_method random \
        --runs 3 2>&1 | tee "$logfile"
    log "PCGT A4 done. Results:"
    grep -E 'Final Test' "$logfile" | tail -1
}

run_best() {
    # Edit these after finding the best config from probes
    local BEST_K=${BEST_K:-256}
    local BEST_GW=${BEST_GW:-0.5}
    local BEST_METHOD=${BEST_METHOD:-metis}
    local logfile="$LOG_DIR/pcgt_best_5runs.log"
    log "Starting PCGT best config: K=$BEST_K, $BEST_METHOD, gw=$BEST_GW (5 runs) → $logfile"
    cd "$LARGE_DIR"
    $PY main.py --method pcgt $COMMON \
        --graph_weight "$BEST_GW" \
        --num_partitions "$BEST_K" --partition_method "$BEST_METHOD" \
        --runs 5 2>&1 | tee "$logfile"
    log "PCGT best done. Results:"
    grep -E 'Final Test' "$logfile" | tail -1
}

check_status() {
    echo "============================================================"
    echo "OGBN-ARXIV EXPERIMENT STATUS"
    echo "============================================================"
    echo ""
    echo "=== Running processes ==="
    ps aux | grep 'main.py.*ogbn-arxiv' | grep -v grep || echo "  (none)"
    echo ""
    echo "=== Log files ==="
    ls -lh "$LOG_DIR"/*.log 2>/dev/null || echo "  (no logs yet)"
    echo ""
    echo "=== Completed results ==="
    for f in "$LOG_DIR"/*.log; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .log)
        result=$(grep -E '^\s*Final Test:' "$f" 2>/dev/null | tail -1)
        if [ -n "$result" ]; then
            printf "  %-25s %s\n" "$name:" "$result"
        else
            lines=$(wc -l < "$f")
            printf "  %-25s (in progress, %d lines)\n" "$name:" "$lines"
        fi
    done
    echo "============================================================"
}

# ============================================================
# Main dispatcher
# ============================================================
case "${1:-check}" in
    sgformer)  run_sgformer ;;
    a1)        run_a1 ;;
    a2)        run_a2 ;;
    a3)        run_a3 ;;
    a4)        run_a4 ;;
    best)      run_best ;;
    check)     check_status ;;
    all)
        run_sgformer
        run_a1
        run_a2
        run_a3
        run_a4
        ;;
    *)
        echo "Usage: $0 {sgformer|a1|a2|a3|a4|best|check|all}"
        exit 1
        ;;
esac
