#!/bin/bash
# ============================================================================
# PCGT Large-Scale Experiments — All Datasets
# Install deps, download data, run SGFormer baseline + PCGT
#
# Usage:
#   bash run.sh setup          # Install deps + download data
#   bash run.sh sgformer       # Run all SGFormer baselines
#   bash run.sh pcgt           # Run all PCGT experiments
#   bash run.sh all            # Run everything (sgformer + pcgt)
#   bash run.sh check          # Check log progress
#   bash run.sh <dataset>      # Run one dataset (sgformer + pcgt)
#   bash run.sh night          # Run only MISSING experiments (safe overnight)
#
# Datasets: arxiv, proteins, amazon2m, pokec
# Environment: DEVICE=0 (GPU id)
# ============================================================================

set -uo pipefail  # NO set -e: one failure must NOT stop the rest

# Auto-detect python
if command -v /system/conda/miniconda3/envs/cloudspace/bin/python &>/dev/null; then
    PY="/system/conda/miniconda3/envs/cloudspace/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    PY="python"
fi
PYU="$PY -u"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LARGE_DIR="$SCRIPT_DIR"
LOG_DIR="$REPO_DIR/logs/large"
DATA_DIR="$REPO_DIR/data"
DEVICE="${DEVICE:-0}"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ============================================================================
# SETUP: Install dependencies + download data
# ============================================================================
do_setup() {
    log "=== SETUP: Installing dependencies ==="
    cd "$REPO_DIR"

    # Install PyTorch Geometric dependencies
    $PY -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-$($PY -c "import torch; print(torch.__version__.split('+')[0])")+cu$($PY -c "import torch; print(torch.version.cuda.replace('.',''))").html 2>/dev/null || \
    $PY -m pip install torch-scatter torch-sparse torch-cluster 2>/dev/null || true

    $PY -m pip install torch-geometric ogb pymetis 2>/dev/null || true

    log "=== SETUP: Downloading OGB datasets ==="
    cd "$LARGE_DIR"

    # Test imports
    $PY -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')
from torch_geometric.utils import to_undirected; print('PyG OK')
from ogb.nodeproppred import NodePropPredDataset; print('OGB OK')
import pymetis; print('pymetis OK')
"

    # Pre-download datasets (they auto-download on first use, but let's be explicit)
    $PY -c "
from ogb.nodeproppred import NodePropPredDataset
import os
data_dir = '$DATA_DIR'
os.makedirs(data_dir, exist_ok=True)
for name in ['ogbn-arxiv', 'ogbn-proteins', 'ogbn-products']:
    print(f'Downloading {name}...')
    try:
        NodePropPredDataset(name=name, root=os.path.join(data_dir, 'ogb'))
        print(f'  {name} OK')
    except Exception as e:
        print(f'  {name} error: {e}')
"

    log "=== SETUP COMPLETE ==="
}

# ============================================================================
# RUN EXPERIMENT: generic runner with logging (error-tolerant)
# ============================================================================
is_done() {
    local name="$1"
    local logfile="$LOG_DIR/${name}.log"
    [ -f "$logfile" ] && grep -q "Final Test:" "$logfile" 2>/dev/null
}

run_exp() {
    local name="$1"
    local script="$2"
    shift 2
    local logfile="$LOG_DIR/${name}.log"
    log "START $name -> $logfile"
    cd "$LARGE_DIR"
    if $PYU "$script" "$@" --data_dir "$DATA_DIR" --device $DEVICE 2>&1 | tee "$logfile"; then
        log "DONE  $name (success)"
    else
        log "DONE  $name (exit code $?)"
    fi
    echo ""
}

# Skip if already completed
run_exp_skip() {
    local name="$1"
    if is_done "$name"; then
        log "SKIP $name (already completed — has Final Test in log)"
        return 0
    fi
    run_exp "$@"
}

run_exp_bg() {
    local name="$1"
    local script="$2"
    shift 2
    local logfile="$LOG_DIR/${name}.log"
    log "LAUNCH (bg) $name -> $logfile"
    cd "$LARGE_DIR"
    nohup $PYU "$script" "$@" --data_dir "$DATA_DIR" --device $DEVICE > "$logfile" 2>&1 &
    echo "  PID: $!"
}

# ============================================================================
# OGBN-ARXIV (169K nodes, 1.2M edges) — uses main.py (full-batch)
# SGFormer paper: 72.63 ± 0.13
# ============================================================================
run_arxiv_sgformer() {
    run_exp "arxiv_sgformer" main.py \
        --method sgformer --dataset ogbn-arxiv --metric acc \
        --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
        --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --seed 123 --runs 5 --epochs 1000 --eval_step 9
}

run_arxiv_pcgt() {
    run_exp "arxiv_pcgt_k256" main.py \
        --method pcgt --dataset ogbn-arxiv --metric acc \
        --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
        --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --num_partitions 256 --partition_method metis \
        --seed 123 --runs 10 --epochs 1000 --eval_step 9
}

# ============================================================================
# OGBN-PROTEINS (132K nodes, 39.5M edges) — uses main-batch.py
# SGFormer paper: 79.53 ± 0.38 (ROC-AUC)
# ============================================================================
run_proteins_sgformer() {
    run_exp "proteins_sgformer" main-batch.py \
        --method sgformer --dataset ogbn-proteins --metric rocauc \
        --lr 0.01 --hidden_channels 64 \
        --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --batch_size 10000 --seed 123 --runs 5 --epochs 1000 --eval_step 9
}

run_proteins_pcgt() {
    run_exp "proteins_pcgt_k256" main-batch.py \
        --method pcgt --dataset ogbn-proteins --metric rocauc \
        --lr 0.01 --hidden_channels 64 \
        --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --num_partitions 256 --partition_method metis \
        --batch_size 10000 --seed 123 --runs 3 --epochs 1000 --eval_step 9
}

# ============================================================================
# AMAZON2M (2.4M nodes, 61.8M edges) — uses main-batch.py
# SGFormer paper: 89.09 ± 0.10
# ============================================================================
run_amazon2m_sgformer() {
    run_exp "amazon2m_sgformer" main-batch.py \
        --method sgformer --dataset amazon2m --metric acc \
        --lr 0.01 --hidden_channels 256 \
        --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9
}

run_amazon2m_pcgt() {
    run_exp "amazon2m_pcgt_k1000" main-batch.py \
        --method pcgt --dataset amazon2m --metric acc \
        --lr 0.01 --hidden_channels 256 \
        --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --num_partitions 1000 --partition_method metis \
        --batch_size 100000 --seed 123 --runs 3 --epochs 1000 --eval_step 9
}

# ============================================================================
# POKEC (1.6M nodes, 30.6M edges) — uses main-batch.py
# SGFormer paper: 73.76 ± 0.24
# ============================================================================
run_pokec_sgformer() {
    run_exp "pokec_sgformer" main-batch.py \
        --method sgformer --dataset pokec --rand_split --metric acc \
        --lr 0.01 --hidden_channels 64 \
        --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9
}

run_pokec_pcgt() {
    run_exp "pokec_pcgt_k500" main-batch.py \
        --method pcgt --dataset pokec --rand_split --metric acc \
        --lr 0.01 --hidden_channels 64 \
        --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
        --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
        --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
        --trans_use_residual --trans_use_weight --trans_use_bn \
        --use_graph --graph_weight 0.5 \
        --num_partitions 500 --partition_method metis \
        --batch_size 100000 --seed 123 --runs 3 --epochs 1000 --eval_step 9
}

# ============================================================================
# CHECK: Show progress of all experiments
# ============================================================================
do_check() {
    echo "======================================================================"
    echo "LARGE-SCALE EXPERIMENT STATUS — $(date)"
    echo "======================================================================"
    for logf in "$LOG_DIR"/*.log; do
        [ -f "$logf" ] || continue
        name=$(basename "$logf" .log)
        final=$(grep -o "Final Test: [0-9.]* ± [0-9.]*" "$logf" 2>/dev/null | tail -n 1)
        highest=$(grep -o "Highest Test: [0-9.]* ± [0-9.]*" "$logf" 2>/dev/null | tail -n 1)
        if [ -n "$final" ]; then
            printf "  %-30s DONE   %s  (%s)\n" "$name" "$final" "$highest"
        else
            lines=$(wc -l < "$logf" 2>/dev/null || echo 0)
            last=$(grep -o "Epoch: [0-9]*" "$logf" 2>/dev/null | tail -n 1 || echo "")
            running=$(ps aux | grep "$name" | grep -v grep | wc -l 2>/dev/null || echo 0)
            status="RUNNING"
            [ "$running" -eq 0 ] && status="STOPPED?"
            printf "  %-30s %s  (%s lines, %s)\n" "$name" "$status" "$lines" "$last"
        fi
    done
    echo "======================================================================"
}

# ============================================================================
# MAIN
# ============================================================================
CMD="${1:-help}"

case "$CMD" in
    setup)
        do_setup
        ;;
    sgformer)
        log "=== Running ALL SGFormer baselines ==="
        run_arxiv_sgformer
        run_proteins_sgformer
        run_amazon2m_sgformer
        run_pokec_sgformer
        log "=== ALL SGFormer baselines DONE ==="
        ;;
    pcgt)
        log "=== Running ALL PCGT experiments ==="
        run_arxiv_pcgt
        run_proteins_pcgt
        run_amazon2m_pcgt
        run_pokec_pcgt
        log "=== ALL PCGT experiments DONE ==="
        ;;
    all)
        log "=== Running EVERYTHING (SGFormer + PCGT, all datasets) ==="
        run_arxiv_sgformer
        run_arxiv_pcgt
        run_proteins_sgformer
        run_proteins_pcgt
        run_amazon2m_sgformer
        run_amazon2m_pcgt
        run_pokec_sgformer
        run_pokec_pcgt
        log "=== ALL DONE ==="
        ;;
    arxiv)
        run_arxiv_sgformer
        run_arxiv_pcgt
        ;;
    proteins)
        run_proteins_sgformer
        run_proteins_pcgt
        ;;
    amazon2m)
        run_amazon2m_sgformer
        run_amazon2m_pcgt
        ;;
    pokec)
        run_pokec_sgformer
        run_pokec_pcgt
        ;;
    check)
        do_check
        ;;
    night)
        log "=== 4-HOUR GPU RUN: PCGT only (use SGFormer published numbers) ==="
        log "=== Priority: proteins > pokec > amazon2m ==="
        log "=== Each experiment is independent — failures won't stop others ==="

        # 1. proteins_pcgt (~2h): 132K nodes, 3 runs
        run_exp_skip "proteins_pcgt_k256" main-batch.py \
            --method pcgt --dataset ogbn-proteins --metric rocauc \
            --lr 0.01 --hidden_channels 64 \
            --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
            --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
            --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
            --trans_use_residual --trans_use_weight --trans_use_bn \
            --use_graph --graph_weight 0.5 \
            --num_partitions 256 --partition_method metis \
            --batch_size 10000 --seed 123 --runs 3 --epochs 1000 --eval_step 9

        # 2. pokec_pcgt (~1h): 1.6M nodes, 3 runs
        run_exp_skip "pokec_pcgt_k500" main-batch.py \
            --method pcgt --dataset pokec --rand_split --metric acc \
            --lr 0.01 --hidden_channels 64 \
            --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
            --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
            --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
            --trans_use_residual --trans_use_weight --trans_use_bn \
            --use_graph --graph_weight 0.5 \
            --num_partitions 500 --partition_method metis \
            --batch_size 100000 --seed 123 --runs 3 --epochs 1000 --eval_step 9

        # 3. amazon2m_pcgt (~1-1.5h): 2.4M nodes, 1 run only to fit in time
        run_exp_skip "amazon2m_pcgt_k1000" main-batch.py \
            --method pcgt --dataset amazon2m --metric acc \
            --lr 0.01 --hidden_channels 256 \
            --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. \
            --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
            --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
            --trans_use_residual --trans_use_weight --trans_use_bn \
            --use_graph --graph_weight 0.5 \
            --num_partitions 1000 --partition_method metis \
            --batch_size 100000 --seed 123 --runs 1 --epochs 1000 --eval_step 9

        log "=== 4-HOUR GPU RUN COMPLETE ==="
        do_check
        ;;
    *)
        echo "Usage: bash run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup      Install deps + download OGB data"
        echo "  sgformer   Run SGFormer baselines on all 4 datasets"
        echo "  pcgt       Run PCGT on all 4 datasets"
        echo "  all        Run everything sequentially"
        echo "  night      PCGT-only for 4h GPU (proteins→pokec→amazon2m, skip SGFormer)"
        echo "  check      Show progress of all logs"
        echo ""
        echo "Per-dataset (runs both sgformer + pcgt):"
        echo "  arxiv      ogbn-arxiv (169K nodes)"
        echo "  proteins   ogbn-proteins (132K nodes)"
        echo "  amazon2m   Amazon2M (2.4M nodes)"
        echo "  pokec      Pokec (1.6M nodes)"
        echo ""
        echo "Environment: DEVICE=0 (GPU id)"
        echo ""
        echo "Recommended workflow on H100:"
        echo "  1. bash run.sh setup"
        echo "  2. nohup bash run.sh all > ../logs/large_all.log 2>&1 &"
        echo "  3. bash run.sh check"
        ;;
esac
