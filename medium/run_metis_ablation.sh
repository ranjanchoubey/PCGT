#!/bin/bash
# METIS vs Random partition ablation on homophilic + heterophilic datasets
# Run from: cd medium/
# Usage: bash run_metis_ablation.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_DIR/experiments/logs/phase4_metis_ablation"
DATA_DIR="$REPO_DIR/data"

# Auto-detect python
if [ -f "$REPO_DIR/venv/bin/python" ]; then
    PY="$REPO_DIR/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    PY="python"
fi

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_ablation() {
    local name="$1"
    local logfile="$LOG_DIR/${name}.log"
    shift
    log "START $name"
    cd "$SCRIPT_DIR"
    $PY main.py --data_dir "$DATA_DIR" "$@" --cpu --display_step 100 2>&1 | tee "$logfile"
    log "DONE  $name"
    echo ""
}

# ============================================================================
# HOMOPHILIC DATASETS (expect bigger METIS vs random gap)
# ============================================================================

# --- Cora (h ≈ 0.81) ---
log "=== CORA: METIS ==="
run_ablation "cora_metis" --method pcgt --dataset cora --backbone gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --epochs 500

log "=== CORA: RANDOM ==="
run_ablation "cora_random" --method pcgt --dataset cora --backbone gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method random \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --epochs 500

# --- PubMed (h ≈ 0.80) ---
log "=== PUBMED: METIS ==="
run_ablation "pubmed_metis" --method pcgt --dataset pubmed --backbone gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0.0005 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --epochs 500

log "=== PUBMED: RANDOM ==="
run_ablation "pubmed_random" --method pcgt --dataset pubmed --backbone gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0.0005 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method random \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --epochs 500

# --- Coauthor-CS (h ≈ 0.81) ---
log "=== COAUTHOR-CS: METIS ==="
run_ablation "coauthor-cs_metis" --method pcgt --dataset coauthor-cs --backbone gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 15 --partition_method metis \
    --rand_split --seed 123 --runs 5 --epochs 500

log "=== COAUTHOR-CS: RANDOM ==="
run_ablation "coauthor-cs_random" --method pcgt --dataset coauthor-cs --backbone gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 15 --partition_method random \
    --rand_split --seed 123 --runs 5 --epochs 500

# --- Amazon-Computers (h ≈ 0.78) ---
log "=== AMAZON-COMPUTERS: METIS ==="
run_ablation "amazon-comp_metis" --method pcgt --dataset amazon-computers --backbone gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split --seed 123 --runs 5 --epochs 500

log "=== AMAZON-COMPUTERS: RANDOM ==="
run_ablation "amazon-comp_random" --method pcgt --dataset amazon-computers --backbone gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method random \
    --rand_split --seed 123 --runs 5 --epochs 500

# ============================================================================
# HETEROPHILIC DATASETS (expect smaller METIS vs random gap — for comparison)
# ============================================================================

# --- Chameleon (h ≈ 0.23) ---
log "=== CHAMELEON: METIS ==="
run_ablation "chameleon_metis" --method pcgt --dataset chameleon --backbone gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0.001 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 5 --epochs 500

log "=== CHAMELEON: RANDOM ==="
run_ablation "chameleon_random" --method pcgt --dataset chameleon --backbone gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 64 --weight_decay 0.001 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method random \
    --seed 123 --runs 5 --epochs 500

# --- Squirrel (h ≈ 0.22) ---
log "=== SQUIRREL: METIS ==="
run_ablation "squirrel_metis" --method pcgt --dataset squirrel --backbone gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 5 --epochs 500

log "=== SQUIRREL: RANDOM ==="
run_ablation "squirrel_random" --method pcgt --dataset squirrel --backbone gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method random \
    --seed 123 --runs 5 --epochs 500

# ============================================================================
# SUMMARY
# ============================================================================
log "=== ALL DONE ==="
log "Results saved in $LOG_DIR/"
echo ""
echo "=== METIS vs Random Ablation Summary ==="
for f in "$LOG_DIR"/*.log; do
    name=$(basename "$f" .log)
    final=$(grep -o "Final Test:.*" "$f" 2>/dev/null | tail -1)
    if [ -n "$final" ]; then
        echo "  $name: $final"
    fi
done
