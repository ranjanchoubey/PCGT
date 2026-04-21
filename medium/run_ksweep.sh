#!/bin/bash
# ============================================================================
# K-sweep for Table 5 (tab:ksweep) — Chameleon and Squirrel
# Varies K ∈ {5, 10, 15, 20, 30}, all other hyperparameters fixed at final
# paper configs. 10 runs each.
#
# Usage:
#   source ../venv/bin/activate
#   cd medium
#   bash run_ksweep.sh 2>&1 | tee ../logs/ksweep_$(date +%Y%m%d_%H%M%S).log
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

LOGDIR="../experiments/final_results/table6_ksweep"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------- Chameleon K-sweep ----------
# Base config: L=2, d=64, lr=0.01, wd=1e-3, dropout=0.5, gw=0.8,
#              attn_drop=0.3, attn_wd=0.01, seed=123, 10 runs, 500 epochs

for K in 5 10 15 20 30; do
    LOGFILE="$LOGDIR/chameleon_pcgt_K${K}.log"
    if [ -f "$LOGFILE" ]; then
        log "SKIP chameleon K=$K — log already exists"
        continue
    fi
    log "START chameleon K=$K"
    python main.py --dataset chameleon --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions "$K" --partition_method metis \
        --seed 123 --runs 10 --epochs 500 \
        2>&1 | tee "$LOGFILE"
    log "DONE chameleon K=$K"
done

# ---------- Squirrel K-sweep ----------
# Base config: L=4, d=64, lr=0.01, wd=5e-4, dropout=0.5, gw=0.8,
#              attn_drop=0.3, attn_wd=0.01, seed=123, 10 runs, 500 epochs

for K in 5 10 15 20 30; do
    LOGFILE="$LOGDIR/squirrel_pcgt_K${K}.log"
    if [ -f "$LOGFILE" ]; then
        log "SKIP squirrel K=$K — log already exists"
        continue
    fi
    log "START squirrel K=$K"
    python main.py --dataset squirrel --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions "$K" --partition_method metis \
        --seed 123 --runs 10 --epochs 500 \
        2>&1 | tee "$LOGFILE"
    log "DONE squirrel K=$K"
done

log "=== ALL K-SWEEP RUNS COMPLETE ==="
log "Logs saved to: $LOGDIR/"
log "Next: verify values match paper Table 5 (tab:ksweep)"
