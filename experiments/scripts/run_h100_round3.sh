#!/bin/bash
# ============================================================================
# H100 Round 3: Timing Benchmarks + GCN Baselines
# 
# Goals:
#   1. Clean timing data for ALL 11 medium datasets (PCGT + SGFormer)
#   2. GCN baselines on 4 additional datasets (Table 3)
#   3. All on same H100 GPU for consistent comparison
#
# Log structure: logs/h100_round3/{experiment_name}.log
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

run_exp() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "Final Test:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        return 0
    fi
    log "START $name → $logfile"
    $PY medium/main.py "$@" --data_dir "$DATA_DIR" --device "$DEVICE" 2>&1 | tee "$logfile"
    log "DONE  $name"
}

# ============================================================================
# SECTION 1: Timing benchmarks - 7 Original Datasets (SGFormer + PCGT)
# Using FINAL configs from medium/run.sh, 3 runs for timing
# ============================================================================

log "=== SECTION 1: Timing - 7 Original Datasets ==="

# --- Cora ---
run_exp "timing_cora_sgformer" --dataset cora --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

run_exp "timing_cora_pcgt" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

# --- CiteSeer ---
run_exp "timing_citeseer_sgformer" --dataset citeseer --method sgformer \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

run_exp "timing_citeseer_pcgt" --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

# --- PubMed ---
run_exp "timing_pubmed_sgformer" --dataset pubmed --method sgformer \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

run_exp "timing_pubmed_pcgt" --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

# --- Chameleon ---
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

# --- Squirrel ---
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

# --- Film ---
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

# --- Deezer ---
run_exp "timing_deezer_sgformer" --dataset deezer-europe --method sgformer \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5 --seed 123 --runs 3

run_exp "timing_deezer_pcgt" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.5 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 3 --epochs 500

# ============================================================================
# SECTION 2: Timing - 4 Additional Datasets (SGFormer + PCGT)
# ============================================================================

log "=== SECTION 2: Timing + Full runs - 4 Additional Datasets ==="

# --- Coauthor-CS ---
run_exp "timing_coauthor-cs_sgformer" --dataset coauthor-cs --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split --seed 123 --runs 3 --epochs 500

run_exp "timing_coauthor-cs_pcgt" --dataset coauthor-cs --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 15 --partition_method metis \
    --rand_split --seed 123 --runs 3 --epochs 500

# --- Coauthor-Physics ---
run_exp "timing_coauthor-physics_sgformer" --dataset coauthor-physics --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split --seed 123 --runs 3 --epochs 500

run_exp "timing_coauthor-physics_pcgt" --dataset coauthor-physics --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split --seed 123 --runs 3 --epochs 500

# --- Amazon-Computers ---
run_exp "timing_amazon-computers_sgformer" --dataset amazon-computers --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split --seed 123 --runs 3 --epochs 500

run_exp "timing_amazon-computers_pcgt" --dataset amazon-computers --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split --seed 123 --runs 3 --epochs 500

# --- Amazon-Photo ---
run_exp "timing_amazon-photo_sgformer" --dataset amazon-photo --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split --seed 123 --runs 3 --epochs 500

run_exp "timing_amazon-photo_pcgt" --dataset amazon-photo --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split --seed 123 --runs 3 --epochs 500

# ============================================================================
# SECTION 3: GCN Baselines for Table 3 (4 additional datasets)
# ============================================================================

log "=== SECTION 3: GCN Baselines for Table 3 ==="

run_exp "gcn_coauthor-cs" --dataset coauthor-cs --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 10 --epochs 500

run_exp "gcn_coauthor-physics" --dataset coauthor-physics --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 10 --epochs 500

run_exp "gcn_amazon-computers" --dataset amazon-computers --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 10 --epochs 500

run_exp "gcn_amazon-photo" --dataset amazon-photo --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 10 --epochs 500

# ============================================================================
# SECTION 4: GCN Timing (for runtime table comparison)
# ============================================================================

log "=== SECTION 4: GCN Timing (representative datasets) ==="

run_exp "timing_cora_gcn" --dataset cora --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

run_exp "timing_pubmed_gcn" --dataset pubmed --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 3

run_exp "timing_chameleon_gcn" --dataset chameleon --method gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 \
    --seed 123 --runs 3

run_exp "timing_squirrel_gcn" --dataset squirrel --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --seed 123 --runs 3

run_exp "timing_deezer_gcn" --dataset deezer-europe --method gcn \
    --lr 0.01 --num_layers 2 --hidden_channels 96 \
    --weight_decay 5e-05 --dropout 0.4 \
    --rand_split --seed 123 --runs 3

run_exp "timing_coauthor-cs_gcn" --dataset coauthor-cs --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 3

run_exp "timing_coauthor-physics_gcn" --dataset coauthor-physics --method gcn \
    --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 \
    --rand_split --seed 123 --runs 3

# ============================================================================
# SUMMARY
# ============================================================================

log "=== ALL EXPERIMENTS DONE ==="
log "Logs in: $LOG_DIR/"
ls -la "$LOG_DIR/"

log ""
log "=== TIMING SUMMARY ==="
for f in "$LOG_DIR"/timing_*.log; do
    name=$(basename "$f" .log)
    runtime=$(grep "run_time:" "$f" 2>/dev/null | tail -1 | grep -oP '[\d.]+' | tail -1)
    result=$(grep "Highest Test:" "$f" 2>/dev/null | tail -1)
    echo "$name | runtime=${runtime}ms | $result"
done

log ""
log "=== GCN BASELINE SUMMARY ==="
for f in "$LOG_DIR"/gcn_*.log; do
    name=$(basename "$f" .log)
    result=$(grep "Highest Test:" "$f" 2>/dev/null | tail -1)
    echo "$name | $result"
done
