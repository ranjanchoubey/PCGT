#!/bin/bash
# ============================================================
# PCGT ABLATION STUDY — Background Runner
# Run from: /Users/vn59a0h/thesis/PCGT/medium
# ============================================================
# 8 ablation variants × 3 datasets + K sweep (4 × 2 datasets) = 32 exps
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source ../venv/bin/activate

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="ablation_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/ablation_log.txt"

echo "============================================================" | tee "$LOG"
echo "PCGT ABLATION STUDY — Started $(date)" | tee -a "$LOG"
echo "Results directory: $RESULTS_DIR" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# ─── DATASET CONFIGS ───
# Each dataset has its own base command (matching FINAL_CONFIGS.sh exactly)

# Cora: rand_split_class, 3 runs (semi-supervised, small)
CORA_BASE="python -B main.py --method pcgt --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 3 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2"

# Chameleon: 10 pre-computed filtered splits, 10 runs
CHAM_BASE="python -B main.py --method pcgt --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3"

# Squirrel: 10 pre-computed filtered splits, 10 runs
SQRL_BASE="python -B main.py --method pcgt --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3"

run_experiment() {
    local name=$1
    local dataset=$2
    local cmd=$3
    local outfile="$RESULTS_DIR/${dataset}_${name}.txt"

    echo "" | tee -a "$LOG"
    echo ">>> [$dataset] $name — $(date '+%H:%M:%S')" | tee -a "$LOG"
    echo "    CMD: $cmd" >> "$LOG"

    eval "$cmd" 2>&1 | tee "$outfile"

    # Extract result
    local result=$(grep "Highest Test:" "$outfile" | tail -1)
    echo "    RESULT: $result" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

# ============================================================
# ABLATION 1: Full PCGT (baseline — same as production config)
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 1: Full PCGT (baseline) ===" | tee -a "$LOG"

run_experiment "01_full_pcgt" "cora" \
    "$CORA_BASE --partition_method metis"

run_experiment "01_full_pcgt" "chameleon" \
    "$CHAM_BASE --partition_method metis"

run_experiment "01_full_pcgt" "squirrel" \
    "$SQRL_BASE --partition_method metis"

# ============================================================
# ABLATION 2: w/o Global attention (local_only)
# Tests: Is cross-partition attention needed?
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 2: w/o Global (local_only) ===" | tee -a "$LOG"

run_experiment "02_local_only" "cora" \
    "$CORA_BASE --partition_method metis --local_only"

run_experiment "02_local_only" "chameleon" \
    "$CHAM_BASE --partition_method metis --local_only"

run_experiment "02_local_only" "squirrel" \
    "$SQRL_BASE --partition_method metis --local_only"

# ============================================================
# ABLATION 3: w/o Local attention (global_only)
# Tests: Is intra-partition attention needed?
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 3: w/o Local (global_only) ===" | tee -a "$LOG"

run_experiment "03_global_only" "cora" \
    "$CORA_BASE --partition_method metis --global_only"

run_experiment "03_global_only" "chameleon" \
    "$CHAM_BASE --partition_method metis --global_only"

run_experiment "03_global_only" "squirrel" \
    "$SQRL_BASE --partition_method metis --global_only"

# ============================================================
# ABLATION 4: w/o PSE (no partition structural encoding)
# Tests: Does knowing which partition helps?
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 4: w/o PSE ===" | tee -a "$LOG"

run_experiment "04_no_pse" "cora" \
    "$CORA_BASE --partition_method metis --no_pse"

run_experiment "04_no_pse" "chameleon" \
    "$CHAM_BASE --partition_method metis --no_pse"

run_experiment "04_no_pse" "squirrel" \
    "$SQRL_BASE --partition_method metis --no_pse"

# ============================================================
# ABLATION 5: w/o GCN branch (pure PCGT, graph_weight=0)
# Tests: Is the hybrid GCN design needed?
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 5: w/o GCN branch ===" | tee -a "$LOG"

# Remove --use_graph to disable GCN branch entirely
CORA_NOGRAPH=$(echo "$CORA_BASE" | sed 's/--use_graph//' | sed 's/--graph_weight 0.8/--graph_weight 0.0/')
CHAM_NOGRAPH=$(echo "$CHAM_BASE" | sed 's/--use_graph//' | sed 's/--graph_weight 0.8/--graph_weight 0.0/')
SQRL_NOGRAPH=$(echo "$SQRL_BASE" | sed 's/--use_graph//' | sed 's/--graph_weight 0.8/--graph_weight 0.0/')

run_experiment "05_no_gcn" "cora" \
    "$CORA_NOGRAPH --partition_method metis"

run_experiment "05_no_gcn" "chameleon" \
    "$CHAM_NOGRAPH --partition_method metis"

run_experiment "05_no_gcn" "squirrel" \
    "$SQRL_NOGRAPH --partition_method metis"

# ============================================================
# ABLATION 6: Random partitions (instead of METIS)
# Tests: Does topology-aware partitioning matter?
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 6: Random partitions ===" | tee -a "$LOG"

run_experiment "06_random_part" "cora" \
    "$CORA_BASE --partition_method random"

run_experiment "06_random_part" "chameleon" \
    "$CHAM_BASE --partition_method random"

run_experiment "06_random_part" "squirrel" \
    "$SQRL_BASE --partition_method random"

# ============================================================
# ABLATION 7: Pure GCN (no PCGT at all)
# Tests: What does the GCN backbone achieve alone?
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 7: Pure GCN baseline ===" | tee -a "$LOG"

# Use --method gcn with matched architecture
run_experiment "07_pure_gcn" "cora" \
    "python -B main.py --method gcn --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 3 --display_step 100 --dropout 0.4 --weight_decay 5e-4"

run_experiment "07_pure_gcn" "chameleon" \
    "python -B main.py --method gcn --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --data_dir ../data/ --dropout 0.5 --weight_decay 0.001"

run_experiment "07_pure_gcn" "squirrel" \
    "python -B main.py --method gcn --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --data_dir ../data/ --dropout 0.5 --weight_decay 5e-4"

# ============================================================
# ABLATION 8: SGFormer (matched config, for fair comparison)
# Tests: Fair SGFormer baseline under identical tuning
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 8: SGFormer baseline ===" | tee -a "$LOG"

run_experiment "08_sgformer" "cora" \
    "python -B main.py --method sgformer --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 3 --display_step 100 --aggregate add --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2"

run_experiment "08_sgformer" "chameleon" \
    "python -B main.py --method sgformer --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ --graph_weight 0.8 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3"

run_experiment "08_sgformer" "squirrel" \
    "python -B main.py --method sgformer --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3"

# ============================================================
# ABLATION 9: K sweep (varying num_partitions)
# Tests: Is there a sweet spot for K? (Chameleon/Squirrel only)
# ============================================================
echo "" | tee -a "$LOG"
echo "=== ABLATION 9: K sweep (Chameleon & Squirrel) ===" | tee -a "$LOG"

for K in 5 15 20 30; do
    CHAM_K=$(echo "$CHAM_BASE" | sed "s/--num_partitions 10/--num_partitions $K/")
    SQRL_K=$(echo "$SQRL_BASE" | sed "s/--num_partitions 10/--num_partitions $K/")

    run_experiment "09_K${K}" "chameleon" \
        "$CHAM_K --partition_method metis"

    run_experiment "09_K${K}" "squirrel" \
        "$SQRL_K --partition_method metis"
done

# ============================================================
# SUMMARY: Parse all results
# ============================================================
echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "ABLATION STUDY COMPLETE — $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "RESULTS SUMMARY:" | tee -a "$LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$LOG"
printf "%-25s %-20s %-20s %-20s\n" "Variant" "Cora" "Chameleon" "Squirrel" | tee -a "$LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$LOG"

for variant in 01_full_pcgt 02_local_only 03_global_only 04_no_pse 05_no_gcn 06_random_part 07_pure_gcn 08_sgformer 09_K5 09_K15 09_K20 09_K30; do
    cora_result="—"
    cham_result="—"
    sqrl_result="—"

    if [ -f "$RESULTS_DIR/cora_${variant}.txt" ]; then
        cora_result=$(grep "Highest Test:" "$RESULTS_DIR/cora_${variant}.txt" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+' | head -1)
    fi
    if [ -f "$RESULTS_DIR/chameleon_${variant}.txt" ]; then
        cham_result=$(grep "Highest Test:" "$RESULTS_DIR/chameleon_${variant}.txt" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+' | head -1)
    fi
    if [ -f "$RESULTS_DIR/squirrel_${variant}.txt" ]; then
        sqrl_result=$(grep "Highest Test:" "$RESULTS_DIR/squirrel_${variant}.txt" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+' | head -1)
    fi

    printf "%-25s %-20s %-20s %-20s\n" "$variant" "$cora_result" "$cham_result" "$sqrl_result" | tee -a "$LOG"
done

echo "─────────────────────────────────────────────────────────────" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Full logs saved in: $RESULTS_DIR/" | tee -a "$LOG"
echo "Summary log: $LOG" | tee -a "$LOG"
