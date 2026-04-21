#!/bin/bash
# ============================================================================
# Runtime benchmark: per-epoch training time (ms) on H100
# Matches Table 7 in the paper.
# Runs each config with --runs 1 --epochs 50 (enough for stable timing).
# main.py already prints mean ms/epoch at the end of each run.
# Saves logs to experiments/final_results/table7_runtime/
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
PYU="$PY -u"
DATA_DIR="${DATA_DIR:-../data}"
DEVICE="${DEVICE:-0}"
OUT_DIR="../experiments/final_results/table7_runtime"
mkdir -p "$OUT_DIR"

EPOCHS=50
RUNS=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local name="$1"; shift
    local logfile="$OUT_DIR/${name}.log"
    log "START $name → $logfile"
    $PYU main.py "$@" --data_dir "$DATA_DIR" --device "$DEVICE" \
        --epochs "$EPOCHS" --runs "$RUNS" 2>&1 | tee "$logfile"
    log "DONE  $name"
}

# --- GCN baseline (method=gcn uses the same main.py) ---
# For GCN, we use --method sgformer --use_graph (which uses GCN backbone only when graph_weight=1.0)
# Actually, we need a pure GCN. SGFormer with --no_feat_norm and no global attention ≈ GCN.
# The paper's GCN column was measured with method=gcn backbone runs.

# --- Cora ---
run "timing_cora_sgformer" --dataset cora --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 --seed 123

run "timing_cora_pcgt" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 --seed 123

# --- CiteSeer ---
run "timing_citeseer_sgformer" --dataset citeseer --method sgformer \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 --seed 123

run "timing_citeseer_pcgt" --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 --seed 123

# --- PubMed ---
run "timing_pubmed_sgformer" --dataset pubmed --method sgformer \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --rand_split_class --valid_num 500 --test_num 1000 --seed 123

run "timing_pubmed_pcgt" --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 --seed 123

# --- Chameleon ---
run "timing_chameleon_sgformer" --dataset chameleon --method sgformer \
    --backbone gcn --lr 0.001 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.6 --ours_layers 1 \
    --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5 --seed 123

run "timing_chameleon_pcgt" --dataset chameleon --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis --seed 123

# --- Squirrel ---
run "timing_squirrel_sgformer" --dataset squirrel --method sgformer \
    --backbone gcn --lr 0.001 --num_layers 8 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.3 --ours_layers 1 \
    --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
    --alpha 0.5 --seed 123

run "timing_squirrel_pcgt" --dataset squirrel --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis --seed 123

# --- Film ---
run "timing_film_sgformer" --dataset film --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm --seed 123

run "timing_film_pcgt" --dataset film --method pcgt \
    --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 5 --partition_method metis --seed 123

# --- Deezer ---
run "timing_deezer_sgformer" --dataset deezer-europe --method sgformer \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5 --seed 123

run "timing_deezer_pcgt" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.5 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis --seed 42

# --- Co-CS ---
run "timing_coauthor-cs_sgformer" --dataset coauthor-cs --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split --seed 123

run "timing_coauthor-cs_pcgt" --dataset coauthor-cs --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 15 --partition_method metis \
    --rand_split --seed 123

# --- Co-Physics ---
run "timing_coauthor-physics_sgformer" --dataset coauthor-physics --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split --seed 123

run "timing_coauthor-physics_pcgt" --dataset coauthor-physics --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split --seed 123

# --- Am-Comp ---
run "timing_amazon-computers_sgformer" --dataset amazon-computers --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split --seed 123

run "timing_amazon-computers_pcgt" --dataset amazon-computers --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split --seed 123

# --- Am-Photo ---
run "timing_amazon-photo_sgformer" --dataset amazon-photo --method sgformer \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --rand_split --seed 123

run "timing_amazon-photo_pcgt" --dataset amazon-photo --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split --seed 123

log "All runtime benchmarks complete."
echo ""
echo "=== Summary: grep run_time from all logs ==="
grep -h "run_time" "$OUT_DIR"/timing_*.log 2>/dev/null || echo "(no run_time lines found)"
