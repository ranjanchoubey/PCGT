#!/bin/bash
# Learnable graph_weight ablation — all 7 main datasets
# Runs sequentially on CPU, logs to logs/medium/learn_gw_*.log
set -uo pipefail
cd "$(dirname "$0")"
source ../venv/bin/activate

DATA_DIR="../data"
LOG_DIR="../logs/medium"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    log "START $name"
    python -B main.py "$@" --data_dir "$DATA_DIR" --cpu 2>&1 | tee "$logfile"
    log "DONE  $name"
}

# ---- Cora (10 runs, semi-supervised) ----
run "learn_gw_cora" --dataset cora --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --learn_graph_weight --ours_dropout 0.2 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 50

# ---- CiteSeer (10 runs, semi-supervised) ----
run "learn_gw_citeseer" --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --learn_graph_weight --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 50

# ---- PubMed (10 runs, semi-supervised) ----
run "learn_gw_pubmed" --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --learn_graph_weight --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 50

# ---- Chameleon (10 runs, pre-computed splits) ----
run "learn_gw_chameleon" --dataset chameleon --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --learn_graph_weight --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 50

# ---- Squirrel (10 runs, pre-computed splits) ----
run "learn_gw_squirrel" --dataset squirrel --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --learn_graph_weight --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 50

# ---- Film (10 runs, pre-computed splits) ----
run "learn_gw_film" --dataset film --method pcgt \
    --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.5 --learn_graph_weight --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 5 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 50

# ---- Deezer (10 runs, random split) ----
run "learn_gw_deezer" --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.5 --learn_graph_weight \
    --ours_dropout 0.4 --ours_use_residual \
    --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 10 --epochs 500 --patience 200 --display_step 50

log "ALL DONE"
echo ""
echo "=== SUMMARY ==="
for f in "$LOG_DIR"/learn_gw_*.log; do
    name=$(basename "$f" .log | sed 's/learn_gw_//')
    result=$(grep "runs:" "$f" 2>/dev/null | tail -1)
    gw=$(grep "gw=" "$f" 2>/dev/null | tail -1 | grep -oP 'gw=\K[0-9.]+')
    echo "$name: $result  (final gw=$gw)"
done
