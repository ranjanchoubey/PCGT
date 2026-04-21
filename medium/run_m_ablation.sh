#!/bin/bash
# ============================================================================
# M-ablation: vary num_reps (M) in {2, 4, 8} on Cora, Chameleon, Squirrel
# 5 runs each → 9 configs total
# Saves logs to experiments/final_results/m_ablation/
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
PYU="$PY -u"
DATA_DIR="${DATA_DIR:-../data}"
DEVICE="${DEVICE:-0}"
OUT_DIR="../experiments/final_results/m_ablation"
mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local name="$1"; shift
    local logfile="$OUT_DIR/${name}.log"
    log "START $name → $logfile"
    $PYU main.py "$@" --data_dir "$DATA_DIR" --device "$DEVICE" 2>&1 | tee "$logfile"
    log "DONE  $name"
}

for M in 2 4 8; do
    log "===== M=$M ====="

    run "cora_m${M}" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --num_reps "$M" \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 5

    run "chameleon_m${M}" --dataset chameleon --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --num_reps "$M" \
        --seed 123 --runs 5 --epochs 500

    run "squirrel_m${M}" --dataset squirrel --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --num_reps "$M" \
        --seed 123 --runs 5 --epochs 500
done

log "All M-ablation runs complete."
