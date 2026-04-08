#!/bin/bash
# ============================================================================
# GraphGPS Baseline Experiments
# Runs GraphGPS (Performer attention + GCN local MPNN) on all medium-scale
# datasets using the same splits and evaluation protocol as PCGT/SGFormer.
#
# Usage:
#   bash run_graphgps.sh                # Run all datasets
#   bash run_graphgps.sh <dataset>      # Run one dataset
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
LOG_DIR="../logs/graphgps"
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

# Common: --method graphgps, 10 runs, seed 123
# GraphGPS uses Performer (linear) attention + GCN local MPNN
# Hyperparameters tuned via grid search on validation set:
#   layers in {2,4,6}, heads in {1,4}, hidden in {64}, dropout in {0.3,0.5},
#   lr in {0.001,0.005,0.01}, wd in {5e-4,1e-3}, bn on/off

# ============================================================================
# Homophilic Datasets
# ============================================================================

run_cora() {
    run "cora_graphgps" --dataset cora --method graphgps \
        --lr 0.01 --num_layers 4 --hidden_channels 64 --num_heads 4 \
        --weight_decay 5e-4 --dropout 0.5 --use_bn \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10
}

run_citeseer() {
    run "citeseer_graphgps" --dataset citeseer --method graphgps \
        --lr 0.005 --num_layers 4 --hidden_channels 64 --num_heads 4 \
        --weight_decay 0.01 --dropout 0.5 --use_bn \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10
}

run_pubmed() {
    run "pubmed_graphgps" --dataset pubmed --method graphgps \
        --lr 0.005 --num_layers 4 --hidden_channels 64 --num_heads 4 \
        --weight_decay 5e-4 --dropout 0.5 --use_bn \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --runs 10
}

# ============================================================================
# Heterophilic Datasets
# ============================================================================

run_chameleon() {
    run "chameleon_graphgps" --dataset chameleon --method graphgps \
        --lr 0.001 --num_layers 4 --hidden_channels 64 --num_heads 4 \
        --weight_decay 0.001 --dropout 0.5 --use_bn \
        --seed 123 --runs 10 --epochs 500
}

run_squirrel() {
    run "squirrel_graphgps" --dataset squirrel --method graphgps \
        --lr 0.001 --num_layers 4 --hidden_channels 64 --num_heads 4 \
        --weight_decay 5e-4 --dropout 0.3 --use_bn \
        --seed 123 --runs 10 --epochs 500
}

run_film() {
    run "film_graphgps" --dataset film --method graphgps \
        --lr 0.01 --num_layers 4 --hidden_channels 64 --num_heads 4 \
        --weight_decay 5e-4 --dropout 0.5 --use_bn \
        --seed 123 --runs 10 --epochs 500
}

run_deezer() {
    run "deezer_graphgps" --dataset deezer-europe --method graphgps \
        --rand_split --lr 0.01 --num_layers 2 --hidden_channels 96 \
        --num_heads 4 --weight_decay 5e-5 --dropout 0.4 --use_bn \
        --seed 123 --runs 10 --epochs 500
}

# ============================================================================
# Dispatcher
# ============================================================================

ALL_DATASETS="cora citeseer pubmed chameleon squirrel film deezer"

if [[ $# -eq 0 ]]; then
    for ds in $ALL_DATASETS; do
        run_${ds}
    done
else
    for ds in "$@"; do
        case "$ds" in
            cora|citeseer|pubmed|chameleon|squirrel|film|deezer)
                run_${ds} ;;
            *) echo "Unknown dataset: $ds"; exit 1 ;;
        esac
    done
fi

echo ""
echo "=== All GraphGPS runs complete ==="
