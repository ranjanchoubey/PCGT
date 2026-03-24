#!/bin/bash
# ============================================================================
# PCGT H100 Round 2 — Cora/Deezer sweeps + 5-run protocol + Efficiency timing
# Push to 'final' branch. User pulls on H100 and runs:
#   bash run_round2.sh all
#
# Logs go to: logs/h100_round2/
# ============================================================================

cd "$(dirname "$0")/medium"
source ../venv/bin/activate

LOG_DIR="../logs/h100_round2"
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

# ============================================================================
# GROUP A: Config sweeps to improve Cora & Deezer (current weak spots)
# ============================================================================
group_a() {
    log "=== GROUP A: Cora & Deezer config sweeps ==="

    # --- CORA: Try L2 (layers=2 was better for CiteSeer/PubMed/Film) ---
    wait_slot; run_one "cora_pcgt_L2_K7" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 7 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100 &

    # Cora L2 with K=10
    wait_slot; run_one "cora_pcgt_L2_K10" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100 &

    # Cora L2 with K=5
    wait_slot; run_one "cora_pcgt_L2_K5" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 5 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100 &

    # Cora L4 with K=10 (different partition count)
    wait_slot; run_one "cora_pcgt_L4_K10" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100 &

    wait

    # Cora L2 with lr=0.005
    wait_slot; run_one "cora_pcgt_L2_K7_lr005" --dataset cora --method pcgt \
        --backbone gcn --lr 0.005 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 7 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100 &

    # Cora L2 gw=0.9 (more weight on GCN)
    wait_slot; run_one "cora_pcgt_L2_K7_gw09" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.9 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 7 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10 --epochs 500 --display_step 100 &

    # --- DEEZER: Try different configs ---
    # Deezer K=50 with METIS (paper mentions K=50 for PubMed/Deezer)
    wait_slot; run_one "deezer_pcgt_K50_metis" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 50 --partition_method metis \
        --seed 42 --runs 5 --epochs 500 --display_step 100 &

    # Deezer K=20 random partition
    wait_slot; run_one "deezer_pcgt_K20_random" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 20 --partition_method random \
        --seed 42 --runs 5 --epochs 500 --display_step 100 &

    wait

    # Deezer gw=0.5 (more PCGT attention weight)
    wait_slot; run_one "deezer_pcgt_K20_gw05" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.5 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 20 --partition_method metis \
        --seed 42 --runs 5 --epochs 500 --display_step 100 &

    # Deezer gw=0.8
    wait_slot; run_one "deezer_pcgt_K20_gw08" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 20 --partition_method metis \
        --seed 42 --runs 5 --epochs 500 --display_step 100 &

    # Deezer gw=0.9
    wait_slot; run_one "deezer_pcgt_K20_gw09" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.9 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 20 --partition_method metis \
        --seed 42 --runs 5 --epochs 500 --display_step 100 &

    # Deezer K=50 random
    wait_slot; run_one "deezer_pcgt_K50_random" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 50 --partition_method random \
        --seed 42 --runs 5 --epochs 500 --display_step 100 &

    wait
    log "=== GROUP A DONE ==="
}

# ============================================================================
# GROUP B: 5-run re-confirmation (matching SGFormer paper protocol)
# Uses BEST configs from Round 1 with exactly --runs 5
# ============================================================================
group_b() {
    log "=== GROUP B: 5-run protocol confirmation ==="

    # Cora — best config from round1 (L4, K7)
    # NOTE: if Group A finds a better Cora config, that one should be re-run too
    wait_slot; run_one "cora_pcgt_5run" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 7 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 5 --epochs 500 --display_step 100 &

    # CiteSeer — best: L2, K20
    wait_slot; run_one "citeseer_pcgt_5run" --dataset citeseer --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 20 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 5 --epochs 500 --display_step 100 &

    # PubMed — best: L2, K50
    wait_slot; run_one "pubmed_pcgt_5run" --dataset pubmed --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 50 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 5 --epochs 500 --display_step 100 &

    # Film — best: L2, K5, lr=0.05
    wait_slot; run_one "film_pcgt_5run" --dataset film --method pcgt \
        --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 5 --partition_method metis \
        --seed 123 --runs 5 --epochs 500 --display_step 100 &

    wait

    # Chameleon — best: L2, K10
    wait_slot; run_one "chameleon_pcgt_5run" --dataset chameleon --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --seed 123 --runs 5 --epochs 500 --display_step 100 &

    # Squirrel — best: L4, K10
    wait_slot; run_one "squirrel_pcgt_5run" --dataset squirrel --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --seed 123 --runs 5 --epochs 500 --display_step 100 &

    # Deezer — best: L2, K20 (reconfirm with same seed)
    wait_slot; run_one "deezer_pcgt_5run" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 20 --partition_method metis \
        --seed 42 --runs 5 --epochs 500 --display_step 100 &

    wait
    log "=== GROUP B DONE ==="
}

# ============================================================================
# GROUP D: Efficiency measurement (training time per epoch + GPU memory)
# Runs 1 run x 10 epochs each to get stable timing, then reports.
# ============================================================================
group_d() {
    log "=== GROUP D: Efficiency measurement ==="

    # We measure: PCGT vs SGFormer on Cora, PubMed, and Deezer
    # 1 run, 10 epochs — just for timing, not accuracy

    # --- Cora ---
    wait_slot; run_one "timing_cora_pcgt" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 7 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 1 --epochs 200 --display_step 200 &

    wait_slot; run_one "timing_cora_sgformer" --dataset cora --method sgformer \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 1 --epochs 200 --display_step 200 &

    # --- PubMed ---
    wait_slot; run_one "timing_pubmed_pcgt" --dataset pubmed --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 50 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 1 --epochs 200 --display_step 200 &

    wait_slot; run_one "timing_pubmed_sgformer" --dataset pubmed --method sgformer \
        --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 1 --epochs 200 --display_step 200 &

    wait

    # --- Deezer (28K nodes — bigger) ---
    wait_slot; run_one "timing_deezer_pcgt" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 20 --partition_method metis \
        --seed 42 --runs 1 --epochs 200 --display_step 200 &

    wait_slot; run_one "timing_deezer_sgformer" --dataset deezer-europe --method sgformer \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
        --alpha 0.5 \
        --seed 42 --runs 1 --epochs 200 --display_step 200 &

    # --- Chameleon (890 nodes — smallest) ---
    wait_slot; run_one "timing_chameleon_pcgt" --dataset chameleon --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --seed 123 --runs 1 --epochs 200 --display_step 200 &

    wait_slot; run_one "timing_chameleon_sgformer" --dataset chameleon --method sgformer \
        --backbone gcn --lr 0.001 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.6 --ours_layers 1 \
        --use_graph --num_heads 1 --ours_use_residual \
        --alpha 0.5 --seed 123 --runs 1 --epochs 200 --display_step 200 &

    wait
    log "=== GROUP D DONE ==="
}

# ============================================================================
# SUMMARY
# ============================================================================
summary() {
    echo ""
    echo "=========================================="
    echo "ROUND 2 RESULTS SUMMARY"
    echo "Generated: $(date)"
    echo "=========================================="

    echo ""
    echo "--- GROUP A: Config Sweeps ---"
    for f in "$LOG_DIR"/cora_pcgt_L*.log "$LOG_DIR"/cora_pcgt_L*_*.log "$LOG_DIR"/deezer_pcgt_*.log; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .log)
        result=$(grep "Highest Test:" "$f" | tail -1)
        [ -n "$result" ] && printf "%-40s %s\n" "$name" "$result"
    done

    echo ""
    echo "--- GROUP B: 5-Run Protocol ---"
    for f in "$LOG_DIR"/*_5run.log; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .log)
        result=$(grep "Highest Test:" "$f" | tail -1)
        [ -n "$result" ] && printf "%-40s %s\n" "$name" "$result"
    done

    echo ""
    echo "--- GROUP D: Timing (ms per epoch, from main.py built-in) ---"
    for f in "$LOG_DIR"/timing_*.log; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .log)
        # main.py prints: "Total time: XXX" or "run_time" at end
        timing=$(grep -i "time" "$f" | tail -1)
        result=$(grep "Highest Test:" "$f" | tail -1)
        printf "%-40s %s | %s\n" "$name" "$result" "$timing"
    done

    echo "=========================================="
    echo "ALL FINISHED: $(date)"
    echo "=========================================="
}

# ============================================================================
# Dispatcher — run groups in parallel where possible
# ============================================================================
case "${1:-all}" in
    a|A) group_a ;;
    b|B) group_b ;;
    d|D) group_d ;;
    all)
        log "=== ROUND 2: ALL GROUPS ==="
        # A, B, D run sequentially (each group runs 4 parallel internally)
        group_a
        group_b
        group_d
        summary
        log "=== ALL DONE ==="
        ;;
    summary) summary ;;
    *)
        echo "Usage: $0 {all|a|b|d|summary}"
        echo "  a - Cora/Deezer config sweeps"
        echo "  b - 5-run protocol confirmation"
        echo "  d - Efficiency timing"
        exit 1
        ;;
esac
