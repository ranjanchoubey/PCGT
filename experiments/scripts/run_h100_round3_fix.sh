#!/bin/bash
# ============================================================================
# H100 Round 3 FIX: Re-run failed chameleon, squirrel, film experiments
# Must run from medium/ directory due to hardcoded ../data/ paths in data_utils.py
# ============================================================================

set -uo pipefail
cd /teamspace/studios/this_studio/PCGT/medium
source ../venv/bin/activate

PY="python -u"
DATA_DIR="../data"
DEVICE=0
LOG_DIR="../logs/h100_round3"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_exp() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    # Remove broken log files
    if [ -f "$logfile" ] && ! grep -q "Final Test:" "$logfile" 2>/dev/null; then
        log "REMOVING broken log: $logfile"
        rm "$logfile"
    fi
    if [ -f "$logfile" ] && grep -q "Final Test:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        return 0
    fi
    log "START $name → $logfile"
    $PY main.py "$@" --data_dir "$DATA_DIR" --device "$DEVICE" 2>&1 | tee "$logfile"
    log "DONE  $name"
}

# ============================================================================
# Timing: Chameleon (SGFormer + PCGT)
# ============================================================================
log "=== FIX: Chameleon ==="

run_exp "timing_chameleon_sgformer" --dataset chameleon --method sgformer \
    --backbone gcn --lr 0.001 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.6 --ours_layers 1 \
    --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5 --seed 123 --runs 3 --epochs 200

run_exp "timing_chameleon_pcgt" --dataset chameleon --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 3 --epochs 500

# ============================================================================
# Timing: Squirrel (SGFormer + PCGT)
# ============================================================================
log "=== FIX: Squirrel ==="

run_exp "timing_squirrel_sgformer" --dataset squirrel --method sgformer \
    --backbone gcn --lr 0.001 --num_layers 8 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.3 --ours_layers 1 \
    --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
    --alpha 0.5 --seed 123 --runs 3

run_exp "timing_squirrel_pcgt" --dataset squirrel --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 3 --epochs 500

# ============================================================================
# Timing: Film (SGFormer + PCGT)
# ============================================================================
log "=== FIX: Film ==="

run_exp "timing_film_sgformer" --dataset film --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --seed 123 --runs 3 --epochs 500

run_exp "timing_film_pcgt" --dataset film --method pcgt \
    --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 5 --partition_method metis \
    --seed 123 --runs 3 --epochs 500

# ============================================================================
# GCN Timing: Chameleon, Squirrel (GCN also uses hardcoded paths)
# ============================================================================
log "=== FIX: GCN Timing ==="

run_exp "timing_chameleon_gcn" --dataset chameleon --method gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 \
    --seed 123 --runs 3

run_exp "timing_squirrel_gcn" --dataset squirrel --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --seed 123 --runs 3

run_exp "timing_film_gcn" --dataset film --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --seed 123 --runs 3

log "=== FIX SCRIPT DONE ==="
for f in "$LOG_DIR"/timing_{chameleon,squirrel,film}_*.log; do
    name=$(basename "$f" .log)
    result=$(grep "Highest Test:" "$f" 2>/dev/null | tail -1)
    echo "$name | $result"
done
