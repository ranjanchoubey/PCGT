#!/bin/bash
# ============================================================================
# PCGT H100 Round 2 — FIRE AND FORGET
# 
# Before running:
#   1. Activate your venv manually
#   2. cd to the PCGT directory  
#   3. Run: nohup bash run_sleep.sh > round2_output.log 2>&1 &
#      OR use screen: screen -S r2 bash run_sleep.sh
#
# Safe to disconnect — all output goes to logs/h100_round2/*.log
# If interrupted, re-run: it skips already-completed experiments.
# ============================================================================

set -uo pipefail

cd "$(dirname "$0")/medium" || exit 1

LOG_DIR="../logs/h100_round2"
mkdir -p "$LOG_DIR"
DEVICE=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"

    # Skip if already done
    if grep -q "runs:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        return 0
    fi

    log "START $name"
    python -u main.py "$@" --data_dir ../data --device $DEVICE > "$logfile" 2>&1
    if [ $? -eq 0 ]; then
        log "DONE  $name -> $(grep 'Highest Test:' "$logfile" | tail -1)"
    else
        log "FAIL  $name (check $logfile)"
        tail -5 "$logfile"
    fi
}

# ============================================================================
log "========== ROUND 2 START: $(date) =========="
log "Python: $(which python)"
log "PWD: $(pwd)"
# ============================================================================

# --- GROUP A: Cora sweeps (4 at a time) ---
log "=== GROUP A: Cora & Deezer config sweeps ==="

# Batch A1: Cora L2 with K=5, K=7, K=10 + Cora L4 K=10
run_exp "cora_pcgt_L2_K5" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 5 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

run_exp "cora_pcgt_L2_K7" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 7 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

run_exp "cora_pcgt_L2_K10" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

run_exp "cora_pcgt_L4_K10" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait

# Batch A2: Cora lr=0.005 + gw=0.9 + Deezer K=50 + Deezer random
run_exp "cora_pcgt_L2_K7_lr005" --dataset cora --method pcgt \
    --backbone gcn --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 7 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

run_exp "cora_pcgt_L2_K7_gw09" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.9 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 7 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

run_exp "deezer_pcgt_K50_metis" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 50 --partition_method metis \
    --seed 42 --runs 5 --epochs 500 --display_step 100 &

run_exp "deezer_pcgt_K20_random" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method random \
    --seed 42 --runs 5 --epochs 500 --display_step 100 &

wait

# Batch A3: Deezer gw=0.5, 0.8, 0.9 + K50 random
run_exp "deezer_pcgt_K20_gw05" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.5 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 5 --epochs 500 --display_step 100 &

run_exp "deezer_pcgt_K20_gw08" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 5 --epochs 500 --display_step 100 &

run_exp "deezer_pcgt_K20_gw09" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.9 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 5 --epochs 500 --display_step 100 &

run_exp "deezer_pcgt_K50_random" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 50 --partition_method random \
    --seed 42 --runs 5 --epochs 500 --display_step 100 &

wait
log "=== GROUP A DONE ==="

# --- GROUP B: 5-run protocol (matching SGFormer paper) ---
log "=== GROUP B: 5-run protocol ==="

# Batch B1: Cora, CiteSeer, PubMed, Film
run_exp "cora_pcgt_5run" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 7 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --epochs 500 --display_step 100 &

run_exp "citeseer_pcgt_5run" --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --epochs 500 --display_step 100 &

run_exp "pubmed_pcgt_5run" --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --epochs 500 --display_step 100 &

run_exp "film_pcgt_5run" --dataset film --method pcgt \
    --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 5 --partition_method metis \
    --seed 123 --runs 5 --epochs 500 --display_step 100 &

wait

# Batch B2: Chameleon, Squirrel, Deezer
run_exp "chameleon_pcgt_5run" --dataset chameleon --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 5 --epochs 500 --display_step 100 &

run_exp "squirrel_pcgt_5run" --dataset squirrel --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 5 --epochs 500 --display_step 100 &

run_exp "deezer_pcgt_5run" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 5 --epochs 500 --display_step 100 &

wait
log "=== GROUP B DONE ==="

# --- GROUP D: Efficiency timing ---
log "=== GROUP D: Efficiency timing ==="

# Batch D1: Cora + PubMed (PCGT vs SGFormer)
run_exp "timing_cora_pcgt" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 7 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 1 --epochs 200 --display_step 200 &

run_exp "timing_cora_sgformer" --dataset cora --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 1 --epochs 200 --display_step 200 &

run_exp "timing_pubmed_pcgt" --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 1 --epochs 200 --display_step 200 &

run_exp "timing_pubmed_sgformer" --dataset pubmed --method sgformer \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 1 --epochs 200 --display_step 200 &

wait

# Batch D2: Deezer + Chameleon (PCGT vs SGFormer)
run_exp "timing_deezer_pcgt" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 1 --epochs 200 --display_step 200 &

run_exp "timing_deezer_sgformer" --dataset deezer-europe --method sgformer \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5 \
    --seed 42 --runs 1 --epochs 200 --display_step 200 &

run_exp "timing_chameleon_pcgt" --dataset chameleon --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 1 --epochs 200 --display_step 200 &

run_exp "timing_chameleon_sgformer" --dataset chameleon --method sgformer \
    --backbone gcn --lr 0.001 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.6 --ours_layers 1 \
    --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5 --seed 123 --runs 1 --epochs 200 --display_step 200 &

wait
log "=== GROUP D DONE ==="

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "=========================================="
echo "ROUND 2 COMPLETE: $(date)"
echo "=========================================="

echo ""
echo "--- GROUP A: Config Sweeps ---"
for f in "$LOG_DIR"/cora_pcgt_L*.log "$LOG_DIR"/deezer_pcgt_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    result=$(grep "Highest Test:" "$f" | tail -1)
    [ -n "$result" ] && printf "%-42s %s\n" "$name" "$result"
done

echo ""
echo "--- GROUP B: 5-Run Protocol ---"
for f in "$LOG_DIR"/*_5run.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    result=$(grep "Highest Test:" "$f" | tail -1)
    [ -n "$result" ] && printf "%-42s %s\n" "$name" "$result"
done

echo ""
echo "--- GROUP D: Timing ---"
for f in "$LOG_DIR"/timing_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    result=$(grep "Highest Test:" "$f" | tail -1)
    timing=$(grep -iE "time|sec" "$f" | tail -1)
    printf "%-42s %s | %s\n" "$name" "$result" "$timing"
done

echo ""
echo "=========================================="
echo "Logs saved in: $LOG_DIR/"
echo "Total experiments: $(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)"
echo "=========================================="
