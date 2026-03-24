#!/bin/bash
# ============================================================================
# PCGT Medium-Scale Experiments
# Runs SGFormer baseline + PCGT on all medium-scale datasets.
#
# Usage:
#   bash run.sh                    # Run all experiments sequentially
#   bash run.sh <dataset>          # Run one dataset (sgformer + pcgt)
#   bash run.sh ablation           # Run ablation study (calls run_ablation.sh)
#
# Datasets: cora, citeseer, pubmed, chameleon, squirrel, film, deezer,
#           coauthor-cs, coauthor-physics, amazon-computers, amazon-photo
#
# Environment variables:
#   DEVICE=0        GPU id (default: cpu)
#   DATA_DIR=../data
# ============================================================================

set -uo pipefail

cd "$(dirname "$0")"

PY="${PY:-python}"
PYU="$PY -u"
DATA_DIR="${DATA_DIR:-../data}"
DEVICE="${DEVICE:-cpu}"
LOG_DIR="../logs/medium"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "Final Test:" "$logfile" 2>/dev/null; then
        log "SKIP $name (already done)"
        return 0
    fi
    log "START $name"
    if [[ "$DEVICE" == "cpu" ]]; then
        $PYU main.py "$@" --data_dir "$DATA_DIR" --cpu 2>&1 | tee "$logfile"
    else
        $PYU main.py "$@" --data_dir "$DATA_DIR" --device "$DEVICE" 2>&1 | tee "$logfile"
    fi
    log "DONE  $name"
}

# ============================================================================
# SECTION 1: Original SGFormer Datasets (7 datasets from SGFormer paper)
# ============================================================================

run_cora() {
    run "cora_sgformer" --dataset cora --method sgformer \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 5

    run "cora_pcgt" --dataset cora --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10
}

run_citeseer() {
    run "citeseer_sgformer" --dataset citeseer --method sgformer \
        --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 5

    run "citeseer_pcgt" --dataset citeseer --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 20 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10
}

run_pubmed() {
    run "pubmed_sgformer" --dataset pubmed --method sgformer \
        --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 5

    run "pubmed_pcgt" --dataset pubmed --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 50 --partition_method metis \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10
}

run_chameleon() {
    run "chameleon_sgformer" --dataset chameleon --method sgformer \
        --backbone gcn --lr 0.001 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.6 --ours_layers 1 \
        --use_graph --num_heads 1 --ours_use_residual \
        --alpha 0.5 --seed 123 --runs 10 --epochs 200

    run "chameleon_pcgt" --dataset chameleon --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --seed 123 --runs 10 --epochs 500
}

run_squirrel() {
    run "squirrel_sgformer" --dataset squirrel --method sgformer \
        --backbone gcn --lr 0.001 --num_layers 8 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.3 --ours_layers 1 \
        --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
        --alpha 0.5 --seed 123 --runs 10

    run "squirrel_pcgt" --dataset squirrel --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --seed 123 --runs 10 --epochs 500
}

run_film() {
    run "film_sgformer" --dataset film --method sgformer \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --seed 123 --runs 10 --epochs 500

    run "film_pcgt" --dataset film --method pcgt \
        --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
        --num_partitions 5 --partition_method metis \
        --seed 123 --runs 10 --epochs 500
}

run_deezer() {
    run "deezer_sgformer" --dataset deezer-europe --method sgformer \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
        --alpha 0.5 --seed 123 --runs 5

    run "deezer_pcgt" --dataset deezer-europe --method pcgt \
        --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
        --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
        --ours_layers 1 --use_graph --graph_weight 0.5 --ours_dropout 0.4 \
        --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
        --num_partitions 20 --partition_method metis \
        --seed 42 --runs 10 --epochs 500
}

# ============================================================================
# SECTION 2: Additional Datasets (4 new datasets)
# ============================================================================

run_coauthor_cs() {
    run "coauthor-cs_sgformer" --dataset coauthor-cs --method sgformer \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --rand_split --seed 123 --runs 10 --epochs 500

    run "coauthor-cs_pcgt" --dataset coauthor-cs --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 15 --partition_method metis \
        --rand_split --seed 123 --runs 10 --epochs 500
}

run_coauthor_physics() {
    run "coauthor-physics_sgformer" --dataset coauthor-physics --method sgformer \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --rand_split --seed 123 --runs 10 --epochs 500

    run "coauthor-physics_pcgt" --dataset coauthor-physics --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 20 --partition_method metis \
        --rand_split --seed 123 --runs 10 --epochs 500
}

run_amazon_computers() {
    run "amazon-computers_sgformer" --dataset amazon-computers --method sgformer \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --rand_split --seed 123 --runs 10 --epochs 500

    run "amazon-computers_pcgt" --dataset amazon-computers --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --rand_split --seed 123 --runs 10 --epochs 500
}

run_amazon_photo() {
    run "amazon-photo_sgformer" --dataset amazon-photo --method sgformer \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --rand_split --seed 123 --runs 10 --epochs 500

    run "amazon-photo_pcgt" --dataset amazon-photo --method pcgt \
        --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
        --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual \
        --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
        --num_partitions 10 --partition_method metis \
        --rand_split --seed 123 --runs 10 --epochs 500
}

# ============================================================================
# MAIN
# ============================================================================
CMD="${1:-all}"

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
    ablation)
        log "Running ablation study..."
        bash run_ablation.sh
        ;;
    all)
        log "=== Running ALL medium-scale experiments ==="
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
        echo "Usage: bash run.sh [dataset|all|ablation]"
        echo ""
        echo "Datasets:"
        echo "  cora, citeseer, pubmed, chameleon, squirrel, film, deezer"
        echo "  coauthor-cs, coauthor-physics, amazon-computers, amazon-photo"
        echo ""
        echo "Special:"
        echo "  all       Run all 11 datasets (default)"
        echo "  ablation  Run ablation study (32 experiments)"
        echo ""
        echo "Environment: DEVICE=cpu (or GPU id), DATA_DIR=../data"
        ;;
esac

