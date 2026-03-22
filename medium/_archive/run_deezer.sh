#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# DEEZER-EUROPE: 28,281 nodes, 31,241 features, 2 classes, binary
# Community-structured — METIS should carve natural communities
# SGFormer (ours) baseline: 66.16 (1-run, hidden=96, lr=0.01, L=2, gw=0.8)
# Uses --rand_split (no pre-computed splits), default train_prop=0.5 valid_prop=0.25
# ===========================================================================

DZ_BASE="--method pcgt --dataset deezer-europe --lr 0.01 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --ours_layers 1 --aggregate add --data_dir ../data/"

# DZ1: layers=2 K=20 hidden=64 (conservative start)
echo "=== DZ1: L=2 K=20 h=64 ==="
python -B main.py $DZ_BASE --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.3

# DZ2: layers=2 K=50
echo "=== DZ2: L=2 K=50 h=64 ==="
python -B main.py $DZ_BASE --hidden_channels 64 --num_layers 2 --num_partitions 50 --num_reps 4 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.3

# DZ3: layers=2 K=100 (more fine-grained communities)
echo "=== DZ3: L=2 K=100 h=64 ==="
python -B main.py $DZ_BASE --hidden_channels 64 --num_layers 2 --num_partitions 100 --num_reps 4 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.3

# DZ4: hidden=96 K=50 (match SGFormer capacity)
echo "=== DZ4: L=2 K=50 h=96 ==="
python -B main.py $DZ_BASE --hidden_channels 96 --num_layers 2 --num_partitions 50 --num_reps 4 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.4

echo "=== ALL DEEZER PROBES DONE ==="
