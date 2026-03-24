#!/bin/bash
# ============================================================================
# PCGT H100 Validation Script
# Run ALL medium-scale experiments from scratch on GPU to get final numbers.
# Results are saved to logs/h100_validation/ with clear naming.
#
# Usage:
#   bash doubts/validate_h100.sh              # Run everything
#   bash doubts/validate_h100.sh phase1       # HIGH priority only
#   bash doubts/validate_h100.sh phase2       # MEDIUM priority
#   bash doubts/validate_h100.sh phase3       # Reconfirmation
#   bash doubts/validate_h100.sh <dataset>    # Single dataset
# ============================================================================

set -uo pipefail

cd "$(dirname "$0")/../medium"

PY="${PY:-python}"
PYU="$PY -u"
DATA_DIR="${DATA_DIR:-../data}"
DEVICE="${DEVICE:-0}"
LOG_DIR="../logs/h100_validation"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    log "START $name → $logfile"
    $PYU main.py "$@" --data_dir "$DATA_DIR" --device "$DEVICE" 2>&1 | tee "$logfile"
    log "DONE  $name"
    echo ""
}

# ============================================================================
# PHASE 1: HIGH PRIORITY — Config investigation
# ============================================================================

phase1_citeseer() {
    log "=== CITESEER: Testing 3 configs ==="

    # Config A: layers=2, lr=0.01, gw=0.7, K=7 (old best single-run: 73.20)
    run "citeseer_pcgt_L2_gw07_K7" --dataset citeseer --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 7 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100

    # Config B: layers=2, lr=0.01, gw=0.7, K=20
    run "citeseer_pcgt_L2_gw07_K20" --dataset citeseer --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 20 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100

    # Config C: current run.sh (layers=4, lr=0.005, gw=0.8, K=20)
    run "citeseer_pcgt_L4_gw08_K20" --dataset citeseer --method pcgt \
        --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 20 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100
}

phase1_pubmed() {
    log "=== PUBMED: Testing 3 configs ==="

    # Config A: layers=2, lr=0.01, gw=0.8, K=50 (old best: 81.00±0.73)
    run "pubmed_pcgt_L2_gw08_K50" --dataset pubmed --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 50 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100

    # Config B: layers=2, lr=0.01, gw=0.9, K=10
    run "pubmed_pcgt_L2_gw09_K10" --dataset pubmed --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.9 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100

    # Config C: current run.sh (layers=4, lr=0.005, gw=0.9, K=10)
    run "pubmed_pcgt_L4_gw09_K10" --dataset pubmed --method pcgt \
        --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.9 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100
}

phase1_film() {
    log "=== FILM: Testing 3 configs ==="

    # Config A: old best 10-run (lr=0.05, layers=2, gw=0.5, K=5) → 38.04±0.84
    run "film_pcgt_lr005_L2_gw05_K5" --dataset film --method pcgt \
        --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 5 --partition_method metis \
        --seed 123 --runs 10 --epochs 500 --display_step 100

    # Config B: single-run best (lr=0.1, layers=2, gw=0.5, K=10, ours_decay=0.0001)
    run "film_pcgt_lr01_L2_gw05_K10" --dataset film --method pcgt \
        --backbone gcn --lr 0.1 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.6 --ours_layers 1 \
        --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.0001 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --seed 123 --runs 10 --epochs 500 --display_step 100

    # Config C: current run.sh (lr=0.1, layers=8, gw=0.6, K=5, ours_layers=2)
    run "film_pcgt_lr01_L8_gw06_K5" --dataset film --method pcgt \
        --backbone gcn --lr 0.1 --num_layers 8 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.6 --ours_layers 2 \
        --use_graph --graph_weight 0.6 --ours_dropout 0.6 \
        --ours_use_residual --ours_use_act \
        --alpha 0.5 --ours_weight_decay 0.0005 \
        --num_partitions 5 --partition_method metis \
        --seed 42 --runs 10 --epochs 500 --display_step 100
}

# ============================================================================
# PHASE 2: MEDIUM PRIORITY — Revalidation with current run.sh configs
# ============================================================================

phase2_chameleon() {
    run "chameleon_pcgt_final" --dataset chameleon --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --seed 123 --runs 10 --epochs 500 --display_step 100
}

phase2_squirrel() {
    run "squirrel_pcgt_final" --dataset squirrel --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --seed 123 --runs 10 --epochs 500 --display_step 100
}

phase2_deezer() {
    run "deezer_pcgt_final" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 20 --partition_method metis \
        --seed 42 --runs 5 --epochs 500 --display_step 100
}

# ============================================================================
# PHASE 3: LOW PRIORITY — Quick reconfirmation (10 runs)
# ============================================================================

phase3_cora() {
    run "cora_pcgt_final" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 7 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100
}

# ============================================================================
# SGFormer baselines (for our own consistency check)
# ============================================================================

sgformer_baselines() {
    run "cora_sgformer" --dataset cora --method sgformer \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500

    run "citeseer_sgformer" --dataset citeseer --method sgformer \
        --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500

    run "pubmed_sgformer" --dataset pubmed --method sgformer \
        --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500

    # Film SGFormer uses difformer method (same as SGFormer paper's Film baseline)
    run "film_sgformer" --dataset film --method difformer \
        --backbone gcn --lr 0.1 --num_layers 8 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.6 \
        --use_graph --graph_weight 0.5 --num_heads 1 \
        --ours_use_residual --ours_use_act \
        --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 \
        --seed 123 --runs 10 --epochs 500

    run "chameleon_sgformer" --dataset chameleon --method sgformer \
        --backbone gcn --lr 0.001 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.6 --ours_layers 1 \
        --use_graph --num_heads 1 --ours_use_residual \
        --alpha 0.5 --seed 123 --runs 10 --epochs 200

    run "squirrel_sgformer" --dataset squirrel --method sgformer \
        --backbone gcn --lr 0.001 --num_layers 8 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.3 --ours_layers 1 \
        --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
        --alpha 0.5 --seed 123 --runs 10

    run "deezer_sgformer" --dataset deezer-europe --method sgformer \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
        --alpha 0.5 --seed 123 --runs 5
}

# ============================================================================
# Dispatcher
# ============================================================================

case "${1:-all}" in
    phase1)
        phase1_citeseer
        phase1_pubmed
        phase1_film
        ;;
    phase2)
        phase2_chameleon
        phase2_squirrel
        phase2_deezer
        ;;
    phase3)
        phase3_cora
        ;;
    sgformer)
        sgformer_baselines
        ;;
    citeseer) phase1_citeseer ;;
    pubmed)   phase1_pubmed ;;
    film)     phase1_film ;;
    chameleon) phase2_chameleon ;;
    squirrel)  phase2_squirrel ;;
    deezer)    phase2_deezer ;;
    cora)      phase3_cora ;;
    all)
        log "=== FULL VALIDATION RUN ==="
        phase1_citeseer
        phase1_pubmed
        phase1_film
        phase2_chameleon
        phase2_squirrel
        phase2_deezer
        phase3_cora
        collect_results
        log "=== ALL DONE ==="
        ;;
    results) collect_results ;;
    *)
        echo "Usage: $0 {all|phase1|phase2|phase3|sgformer|<dataset>}"
        exit 1
        ;;
esac
