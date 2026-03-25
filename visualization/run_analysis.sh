#!/bin/bash
# PCGT Analysis Plots — Best Paper Configs
# Usage: bash visualization/run_analysis.sh <dataset>
# Must be run from repo root.

set -uo pipefail

PY="${PY:-python}"
DEVICE="${DEVICE:-cpu}"
DATA_DIR="${DATA_DIR:-data}"
PLOTS="${PLOTS:-all}"

COMMON="--method pcgt --backbone gcn --ours_layers 1 \
  --use_graph --use_residual --alpha 0.5 \
  --use_bn --ours_use_weight --ours_use_residual --aggregate add \
  --partition_method metis --no_feat_norm \
  --runs 1 --data_dir $DATA_DIR"

if [[ "$DEVICE" == "cpu" ]]; then
    COMMON="$COMMON --cpu"
else
    COMMON="$COMMON --device $DEVICE"
fi

run_cora() {
    $PY visualization/analysis_plots.py --dataset cora \
        $COMMON --num_partitions 10 --plots $PLOTS \
        --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.4 --ours_dropout 0.2 \
        --graph_weight 0.8 --ours_weight_decay 0.001 \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --epochs 500
}

run_citeseer() {
    $PY visualization/analysis_plots.py --dataset citeseer \
        $COMMON --num_partitions 20 --plots $PLOTS \
        --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.01 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.7 --ours_weight_decay 0.01 \
        --rand_split_class --valid_num 500 --test_num 1000 \
        --seed 123 --epochs 500
}

run_chameleon() {
    $PY visualization/analysis_plots.py --dataset chameleon \
        $COMMON --num_partitions 10 --plots $PLOTS \
        --lr 0.01 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.001 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.8 --ours_weight_decay 0.01 \
        --seed 123 --epochs 500
}

run_squirrel() {
    $PY visualization/analysis_plots.py --dataset squirrel \
        $COMMON --num_partitions 10 --plots $PLOTS \
        --lr 0.01 --num_layers 4 --hidden_channels 64 \
        --weight_decay 5e-4 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.8 --ours_weight_decay 0.01 \
        --seed 123 --epochs 500
}

run_film() {
    $PY visualization/analysis_plots.py --dataset film \
        $COMMON --num_partitions 5 --plots $PLOTS \
        --lr 0.05 --num_layers 2 --hidden_channels 64 \
        --weight_decay 0.0005 --dropout 0.5 --ours_dropout 0.3 \
        --graph_weight 0.5 --ours_weight_decay 0.01 \
        --seed 123 --epochs 500
}

CMD="${1:-help}"

case "$CMD" in
    cora)      run_cora ;;
    citeseer)  run_citeseer ;;
    chameleon) run_chameleon ;;
    squirrel)  run_squirrel ;;
    film)      run_film ;;
    *)
        echo "PCGT Analysis Plots"
        echo ""
        echo "Usage: bash visualization/run_analysis.sh <dataset>"
        echo ""
        echo "Datasets: cora, citeseer, chameleon, squirrel, film"
        echo ""
        echo "Env vars:"
        echo "  PLOTS=tsne,alpha_beta,homophily,attention  (default: all)"
        echo "  DEVICE=0       GPU id (default: cpu)"
        echo ""
        echo "Output: visualization/outputs/<dataset>/analysis/"
        ;;
esac
