#!/bin/bash
# ============================================================================
# Parallel H100 Validation — runs 4 jobs at a time, tolerates failures
# Each experiment gets its own log file in logs/h100_validation/
# Skip already-completed experiments (checks for "runs:" in log)
# ============================================================================

cd ~/PCGT/medium
source ../venv/bin/activate

LOG_DIR="../logs/h100_validation"
mkdir -p "$LOG_DIR"
DEVICE=0
MAX_PARALLEL=4

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_one() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"

    if grep -q "runs:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        return 0
    fi

    log "START $name"
    python -u main.py "$@" --data_dir ../data --device $DEVICE > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        local result=$(grep "Highest Test:" "$logfile" | tail -1)
        log "DONE  $name -> $result"
    else
        log "FAIL  $name (exit=$rc)"
    fi
    return 0
}

wait_slot() {
    while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 2
    done
}

echo "=========================================="
echo "PARALLEL H100 VALIDATION (max $MAX_PARALLEL concurrent)"
echo "Started: $(date)"
echo "=========================================="

# ============================================================================
# GROUP 1: CiteSeer (3) + PubMed (1) — 4 parallel
# ============================================================================
log "=== GROUP 1: CiteSeer + PubMed-A + Cora ==="

wait_slot; run_one "citeseer_pcgt_L2_gw07_K7" --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 7 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "citeseer_pcgt_L2_gw07_K20" --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "citeseer_pcgt_L4_gw08_K20" --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "pubmed_pcgt_L2_gw08_K50" --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait
log "=== GROUP 1 DONE ==="

# ============================================================================
# GROUP 2: PubMed (2) + Cora + Deezer — 4 parallel
# ============================================================================
log "=== GROUP 2: PubMed B/C + Cora + Deezer ==="

wait_slot; run_one "pubmed_pcgt_L2_gw09_K10" --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.9 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "pubmed_pcgt_L4_gw09_K10" --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.9 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "cora_pcgt_final" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 7 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "deezer_pcgt_final" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 5 --epochs 500 --display_step 100 &

wait
log "=== GROUP 2 DONE ==="

# ============================================================================
# GROUP 3: Film (3) + Chameleon — 4 parallel
# ============================================================================
log "=== GROUP 3: Film configs + Chameleon ==="

wait_slot; run_one "film_pcgt_lr005_L2_gw05_K5" --dataset film --method pcgt \
    --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 5 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "film_pcgt_lr01_L2_gw05_K10" --dataset film --method pcgt \
    --backbone gcn --lr 0.1 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.6 --ours_layers 1 \
    --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.0001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "film_pcgt_lr01_L8_gw06_K5" --dataset film --method pcgt \
    --backbone gcn --lr 0.1 --num_layers 8 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.6 --ours_layers 2 \
    --use_graph --graph_weight 0.6 --ours_dropout 0.6 \
    --ours_use_residual --ours_use_act \
    --alpha 0.5 --ours_weight_decay 0.0005 \
    --num_partitions 5 --partition_method metis \
    --seed 42 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "chameleon_pcgt_final" --dataset chameleon --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait
log "=== GROUP 3 DONE ==="

# ============================================================================
# GROUP 4: Squirrel + SGFormer baselines (3 Planetoid)
# ============================================================================
log "=== GROUP 4: Squirrel + SGFormer Planetoid ==="

wait_slot; run_one "squirrel_pcgt_final" --dataset squirrel --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --display_step 100 &

wait_slot; run_one "cora_sgformer" --dataset cora --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 &

wait_slot; run_one "citeseer_sgformer" --dataset citeseer --method sgformer \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 &

wait_slot; run_one "pubmed_sgformer" --dataset pubmed --method sgformer \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 &

wait
log "=== GROUP 4 DONE ==="

# ============================================================================
# GROUP 5: SGFormer baselines (Film, Chameleon, Squirrel, Deezer)
# ============================================================================
log "=== GROUP 5: SGFormer non-Planetoid ==="

wait_slot; run_one "film_sgformer" --dataset film --method difformer \
    --backbone gcn --lr 0.1 --num_layers 8 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.6 \
    --use_graph --graph_weight 0.5 --num_heads 1 \
    --ours_use_residual --ours_use_act \
    --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 \
    --seed 123 --runs 10 --epochs 500 &

wait_slot; run_one "chameleon_sgformer" --dataset chameleon --method sgformer \
    --backbone gcn --lr 0.001 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.6 --ours_layers 1 \
    --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5 --seed 123 --runs 10 --epochs 200 &

wait_slot; run_one "squirrel_sgformer" --dataset squirrel --method sgformer \
    --backbone gcn --lr 0.001 --num_layers 8 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.3 --ours_layers 1 \
    --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
    --alpha 0.5 --seed 123 --runs 10 &

wait_slot; run_one "deezer_sgformer" --dataset deezer-europe --method sgformer \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5 --seed 123 --runs 5 &

wait
log "=== GROUP 5 DONE ==="

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "Generated: $(date)"
echo "=========================================="
for f in $LOG_DIR/*.log; do
    name=$(basename "$f" .log)
    result=$(grep -E "Highest Test:" "$f" | tail -1)
    if [ -n "$result" ]; then
        printf "%-40s %s\n" "$name" "$result"
    else
        printf "%-40s %s\n" "$name" "** NO RESULT **"
    fi
done
echo "==========================================" 
echo "ALL FINISHED: $(date)"
echo "=========================================="
