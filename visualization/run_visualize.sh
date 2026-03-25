#!/bin/bash
# ============================================================================
# PCGT Partition Visualization — Run with Best Paper Configs
#
# Usage:
#   bash visualization/run_visualize.sh <dataset>
#   bash visualization/run_visualize.sh all
#
# Datasets: cora, citeseer, pubmed, chameleon, squirrel, film, deezer,
#           coauthor-cs, coauthor-physics, amazon-computers, amazon-photo
#
# Must be run from the repo root: /Users/vn59a0h/thesis/PCGT
# ============================================================================

set -uo pipefail

PY="${PY:-python}"
DEVICE="${DEVICE:-cpu}"
DATA_DIR="${DATA_DIR:-data}"
MAX_PARTS="${MAX_PARTS:-20}"
SPLIT="${SPLIT:-test}"

# Common PCGT flags
COMMON="--method pcgt --backbone gcn --ours_layers 1 \
  --use_graph --use_residual --alpha 0.5 \
  --use_bn --ours_use_weight --ours_use_residual --aggregate add \
  --partition_method metis --no_feat_norm \
  --runs 1 --split $SPLIT --max_parts $MAX_PARTS --data_dir $DATA_DIR"

if [[ "$DEVICE" == "cpu" ]]; then
    COMMON="$COMMON --cpu"
else
    COMMON="$COMMON --device $DEVICE"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================================
# Per-Dataset Commands (best configs from paper)
# ============================================================================

run_cora() {
    log "Visualizing: cora (K=10)"
    $PY visualization/visualize_partitions.py --dataset cora \
        $COMMON --num_partitions 10 \
        --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_dropout 0.2 \
        --graph_weight 0.8 --ours_weight_decay 0.001 \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --epochs 500
}

run_citeseer() {
    log "Visualizing: citeseer (K=20)"
    $PY visualization/visualize_partitions.py --dataset citeseer \
        $COMMON --num_partitions 20 \
        --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.7 --ours_weight_decay 0.01 \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --epochs 500
}

run_pubmed() {
    log "Visualizing: pubmed (K=50)"
    $PY visualization/visualize_partitions.py --dataset pubmed \
        $COMMON --num_partitions 50 \
        --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.8 --ours_weight_decay 0.01 \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --epochs 500
}

run_chameleon() {
    log "Visualizing: chameleon (K=10)"
    $PY visualization/visualize_partitions.py --dataset chameleon \
        $COMMON --num_partitions 10 \
        --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.8 --ours_weight_decay 0.01 \
        --seed 123 --epochs 500
}

run_squirrel() {
    log "Visualizing: squirrel (K=10)"
    $PY visualization/visualize_partitions.py --dataset squirrel \
        $COMMON --num_partitions 10 \
        --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.8 --ours_weight_decay 0.01 \
        --seed 123 --epochs 500
}

run_film() {
    log "Visualizing: film (K=5)"
    $PY visualization/visualize_partitions.py --dataset film \
        $COMMON --num_partitions 5 \
        --lr 0.05 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.5 --ours_weight_decay 0.01 \
        --seed 123 --epochs 500
}

run_deezer() {
    log "Visualizing: deezer-europe (K=20)"
    $PY visualization/visualize_partitions.py --dataset deezer-europe \
        $COMMON --num_partitions 20 \
        --lr 0.01 --num_layers 2 --hidden_channels 96 \
        --weight_decay 5e-05 --dropout 0.4 --ours_dropout 0.4 \
        --graph_weight 0.5 --ours_weight_decay 5e-05 \
        --rand_split --seed 42 --epochs 500
}

run_coauthor_cs() {
    log "Visualizing: coauthor-cs (K=15)"
    $PY visualization/visualize_partitions.py --dataset coauthor-cs \
        $COMMON --num_partitions 15 \
        --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_dropout 0.2 \
        --graph_weight 0.8 --ours_weight_decay 0.001 \
        --rand_split --seed 123 --epochs 500
}

run_coauthor_physics() {
    log "Visualizing: coauthor-physics (K=20)"
    $PY visualization/visualize_partitions.py --dataset coauthor-physics \
        $COMMON --num_partitions 20 \
        --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_dropout 0.2 \
        --graph_weight 0.8 --ours_weight_decay 0.001 \
        --rand_split --seed 123 --epochs 500
}

run_amazon_computers() {
    log "Visualizing: amazon-computers (K=10)"
    $PY visualization/visualize_partitions.py --dataset amazon-computers \
        $COMMON --num_partitions 10 \
        --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_dropout 0.2 \
        --graph_weight 0.8 --ours_weight_decay 0.001 \
        --rand_split --seed 123 --epochs 500
}

run_amazon_photo() {
    log "Visualizing: amazon-photo (K=10)"
    $PY visualization/visualize_partitions.py --dataset amazon-photo \
        $COMMON --num_partitions 10 \
        --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_dropout 0.2 \
        --graph_weight 0.8 --ours_weight_decay 0.001 \
        --rand_split --seed 123 --epochs 500
}

# ============================================================================
# MAIN
# ============================================================================
CMD="${1:-help}"

case "$CMD" in
    cora)             run_cora ;;
    citeseer)         run_citeseer ;;
    pubmed)           run_pubmed ;;
    chameleon)        run_chameleon ;;
    squirrel)         run_squirrel ;;
    film)             run_film ;;
    deezer)           run_deezer ;;
    coauthor-cs)      run_coauthor_cs ;;
    coauthor-physics) run_coauthor_physics ;;
    amazon-computers) run_amazon_computers ;;
    amazon-photo)     run_amazon_photo ;;
    all)
        log "=== Visualizing ALL 11 datasets ==="
        run_cora
        run_citeseer
        run_pubmed
        run_chameleon
        run_squirrel
        run_film
        run_deezer
        run_coauthor_cs
        run_coauthor_physics
        run_amazon_computers
        run_amazon_photo
        log "=== ALL DONE ==="
        ;;
    *)
        echo "PCGT Partition Visualization — Best Paper Configs"
        echo ""
        echo "Usage: bash visualization/run_visualize.sh <dataset>"
        echo ""
        echo "Datasets:"
        echo "  cora              (K=10, homophilic)"
        echo "  citeseer          (K=20, homophilic)"
        echo "  pubmed            (K=50, homophilic)"
        echo "  chameleon         (K=10, heterophilic)"
        echo "  squirrel          (K=10, heterophilic)"
        echo "  film              (K=5,  heterophilic)"
        echo "  deezer            (K=20, medium)"
        echo "  coauthor-cs       (K=15, large)"
        echo "  coauthor-physics  (K=20, large)"
        echo "  amazon-computers  (K=10, large)"
        echo "  amazon-photo      (K=10, large)"
        echo "  all               (run all 11 datasets)"
        echo ""
        echo "Environment variables:"
        echo "  DEVICE=0          GPU id (default: cpu)"
        echo "  DATA_DIR=data     Data directory (default: data)"
        echo "  MAX_PARTS=20      Max partitions to visualize (default: 20)"
        echo "  SPLIT=test        Which split: train/valid/test (default: test)"
        echo ""
        echo "Output: visualization/outputs/<dataset>/"
        ;;
esac
