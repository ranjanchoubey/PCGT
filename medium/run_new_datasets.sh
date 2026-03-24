#!/bin/bash
# Run PCGT + SGFormer on new datasets (Coauthor-CS, Coauthor-Physics, Amazon-Computers, Amazon-Photo)
# CPU only, runs sequentially, each experiment is independent (no set -e)

set -uo pipefail

cd "$(dirname "$0")"
LOG_DIR="../logs/medium_new"
mkdir -p "$LOG_DIR"

PY="python"
DATA_DIR="../data"
COMMON="--backbone gcn --use_graph --hidden_channels 64 --num_layers 2 --ours_layers 1 --epochs 500 --runs 10 --rand_split --cpu --data_dir $DATA_DIR --display_step 25"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    # Skip if already completed
    if [ -f "$logfile" ] && grep -q "Final Test:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        return 0
    fi
    log "START $name"
    $PY -u main.py "$@" 2>&1 | tee "$logfile"
    log "DONE  $name"
}

# ============================================================================
# Coauthor-CS (18333 nodes, 15 classes, homophilic)
# ============================================================================
run "coauthor-cs_sgformer" --dataset coauthor-cs --method ours \
    --graph_weight 0.5 $COMMON

run "coauthor-cs_pcgt" --dataset coauthor-cs --method pcgt \
    --graph_weight 0.5 --num_partitions 15 --partition_method metis $COMMON

# ============================================================================
# Coauthor-Physics (34493 nodes, 5 classes, homophilic)
# ============================================================================
run "coauthor-physics_sgformer" --dataset coauthor-physics --method ours \
    --graph_weight 0.5 $COMMON

run "coauthor-physics_pcgt" --dataset coauthor-physics --method pcgt \
    --graph_weight 0.5 --num_partitions 20 --partition_method metis $COMMON

# ============================================================================
# Amazon-Computers (13752 nodes, 10 classes, homophilic)
# ============================================================================
run "amazon-computers_sgformer" --dataset amazon-computers --method ours \
    --graph_weight 0.5 $COMMON

run "amazon-computers_pcgt" --dataset amazon-computers --method pcgt \
    --graph_weight 0.5 --num_partitions 10 --partition_method metis $COMMON

# ============================================================================
# Amazon-Photo (7650 nodes, 8 classes, homophilic)
# ============================================================================
run "amazon-photo_sgformer" --dataset amazon-photo --method ours \
    --graph_weight 0.5 $COMMON

run "amazon-photo_pcgt" --dataset amazon-photo --method pcgt \
    --graph_weight 0.5 --num_partitions 10 --partition_method metis $COMMON

log "=== ALL NEW DATASET EXPERIMENTS DONE ==="
echo ""
echo "=== RESULTS ==="
for f in "$LOG_DIR"/*.log; do
    name=$(basename "$f" .log)
    final=$(grep -o "Final Test: [0-9.]* ± [0-9.]*" "$f" 2>/dev/null | tail -1)
    [ -n "$final" ] && printf "  %-35s %s\n" "$name" "$final"
done
