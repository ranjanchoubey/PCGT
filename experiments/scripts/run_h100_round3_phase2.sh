#!/bin/bash
# ============================================================================
# H100 Round 3 Phase 2: Additional experiments
# GAT baselines for Table 3 + ogbn-arxiv timing
# Run AFTER fix script completes
# ============================================================================

set -uo pipefail
cd /teamspace/studios/this_studio/PCGT
source venv/bin/activate

PY="python -u"
DATA_DIR="data"
DEVICE=0
LOG_DIR="logs/h100_round3"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_med() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "Final Test:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        return 0
    fi
    log "START $name → $logfile"
    cd /teamspace/studios/this_studio/PCGT/medium
    $PY main.py "$@" --data_dir "../$DATA_DIR" --device "$DEVICE" 2>&1 | tee "$logfile"
    cd /teamspace/studios/this_studio/PCGT
    log "DONE  $name"
}

run_large() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "Final Test:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        return 0
    fi
    log "START $name → $logfile"
    cd /teamspace/studios/this_studio/PCGT/large
    $PY main.py "$@" --data_dir "../$DATA_DIR" --device "$DEVICE" 2>&1 | tee "$logfile"
    cd /teamspace/studios/this_studio/PCGT
    log "DONE  $name"
}

# ============================================================================
# SECTION 1: GAT Baselines for Table 3 (4 additional datasets, 10 runs each)
# Est: ~3-5 min each = ~15-20 min total
# ============================================================================
log "=== Phase 2 SECTION 1: GAT Baselines ==="

run_med "gat_coauthor-cs" --dataset coauthor-cs --method gat \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 10 --epochs 500

run_med "gat_coauthor-physics" --dataset coauthor-physics --method gat \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 10 --epochs 500

run_med "gat_amazon-computers" --dataset amazon-computers --method gat \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 10 --epochs 500

run_med "gat_amazon-photo" --dataset amazon-photo --method gat \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 10 --epochs 500

# ============================================================================
# SECTION 2: ogbn-arxiv timing (SGFormer + PCGT, fewer runs for timing)
# Est: ~10-15 min each = ~20-30 min total
# ============================================================================
log "=== Phase 2 SECTION 2: ogbn-arxiv Timing ==="

run_large "timing_arxiv_sgformer" \
    --method sgformer --dataset ogbn-arxiv --metric acc \
    --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
    --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
    --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
    --trans_use_residual --trans_use_weight --trans_use_bn \
    --seed 123 --runs 1 --epochs 500 --eval_step 9

run_large "timing_arxiv_pcgt" \
    --method pcgt --dataset ogbn-arxiv --metric acc \
    --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
    --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
    --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
    --trans_use_residual --trans_use_weight --trans_use_bn \
    --num_partitions 256 --partition_method metis \
    --seed 123 --runs 1 --epochs 500 --eval_step 9

# ============================================================================
# SECTION 3: Missing GCN Timing (citeseer, film — not in main or fix script)  
# Note: citeseer runs from root OK, film needs to be in medium/ dir
# ============================================================================
log "=== Phase 2 SECTION 3: Missing GCN Timing ==="

run_med "timing_citeseer_gcn" --dataset citeseer --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

# ============================================================================
# SUMMARY
# ============================================================================
log "=== Phase 2 ALL DONE ==="
echo ""
log "=== GAT RESULTS ==="
for f in "$LOG_DIR"/gat_*.log; do
    name=$(basename "$f" .log)
    result=$(grep "Highest Test:" "$f" 2>/dev/null | tail -1)
    echo "  $name | $result"
done
echo ""
log "=== ARXIV TIMING ==="
for f in "$LOG_DIR"/timing_arxiv_*.log; do
    name=$(basename "$f" .log)
    runtime=$(grep "Average epoch time:" "$f" 2>/dev/null | tail -1)
    result=$(grep "Highest Test:" "$f" 2>/dev/null | tail -1)
    echo "  $name | $runtime | $result"
done
