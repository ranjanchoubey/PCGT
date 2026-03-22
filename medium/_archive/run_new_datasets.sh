#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# NEW DATASETS: Squirrel (2,223 nodes) then Film (7,600 nodes)
# Run smallest first to avoid CPU crash while PubMed attack runs
# Both use pre-computed 10-fold splits (no --rand_split needed)
# ===========================================================================

# ===========================
# SQUIRREL (2,223 nodes, 5 classes, heterophilic)
# SGFormer paper Table 2: ~42-44 range (DifFormer got 44.02±1.72)
# Old PCGT (bad config): 39.50 with lr=0.001/layers=8/K=5
# ===========================

SQ_BASE="--method pcgt --dataset squirrel --lr 0.01 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --ours_layers 1 --aggregate add --data_dir ../data/"

# SQ1: layers=2 K=5 (like Chameleon winner)
echo "=== SQ1: L=2 K=5 gw=0.8 ==="
python -B main.py $SQ_BASE --hidden_channels 64 --num_layers 2 --num_partitions 5 --num_reps 4 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# SQ2: layers=2 K=10
echo "=== SQ2: L=2 K=10 gw=0.8 ==="
python -B main.py $SQ_BASE --hidden_channels 64 --num_layers 2 --num_partitions 10 --num_reps 4 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# SQ3: layers=2 K=15 (slightly more partitions)
echo "=== SQ3: L=2 K=15 gw=0.8 ==="
python -B main.py $SQ_BASE --hidden_channels 64 --num_layers 2 --num_partitions 15 --num_reps 4 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# SQ4: layers=4 K=10 (deeper)
echo "=== SQ4: L=4 K=10 gw=0.8 ==="
python -B main.py $SQ_BASE --hidden_channels 64 --num_layers 4 --num_partitions 10 --num_reps 4 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# SQ5: layers=2 K=10 gw=0.5 (less GCN — heterophilic)
echo "=== SQ5: L=2 K=10 gw=0.5 ==="
python -B main.py $SQ_BASE --hidden_channels 64 --num_layers 2 --num_partitions 10 --num_reps 4 --graph_weight 0.5 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# SQ6: layers=2 K=10 d=0.3 ours_d=0.2 (less dropout)
echo "=== SQ6: L=2 K=10 d=0.3 ours_d=0.2 ==="
python -B main.py $SQ_BASE --hidden_channels 64 --num_layers 2 --num_partitions 10 --num_reps 4 --graph_weight 0.8 --dropout 0.3 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.2

echo "=== ALL SQUIRREL PROBES DONE ==="

# ===========================
# FILM (7,600 nodes, 5 classes, heterophilic)
# SGFormer paper: film uses pre-computed 10-fold splits
# DifFormer baseline: 25.61±1.14 (very low — something wrong with their config)
# ===========================

FILM_BASE="--method pcgt --dataset film --lr 0.01 --partition_method metis --use_graph --use_residual --backbone gcn --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --ours_layers 1 --aggregate add --data_dir ../data/"

# F1: layers=2 K=10 (safe start)
echo "=== F1: L=2 K=10 gw=0.8 ==="
python -B main.py $FILM_BASE --hidden_channels 64 --num_layers 2 --num_partitions 10 --num_reps 4 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# F2: layers=2 K=20
echo "=== F2: L=2 K=20 gw=0.8 ==="
python -B main.py $FILM_BASE --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# F3: layers=2 K=50 (more partitions for 7.6K nodes)
echo "=== F3: L=2 K=50 gw=0.8 ==="
python -B main.py $FILM_BASE --hidden_channels 64 --num_layers 2 --num_partitions 50 --num_reps 4 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# F4: layers=4 K=20
echo "=== F4: L=4 K=20 gw=0.8 ==="
python -B main.py $FILM_BASE --hidden_channels 64 --num_layers 4 --num_partitions 20 --num_reps 4 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# F5: layers=2 K=20 gw=0.5 (less GCN for heterophilic)
echo "=== F5: L=2 K=20 gw=0.5 ==="
python -B main.py $FILM_BASE --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.5 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# F6: layers=2 K=20 d=0.6 (higher dropout like SGFormer film config)
echo "=== F6: L=2 K=20 d=0.6 ==="
python -B main.py $FILM_BASE --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.8 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

echo "=== ALL FILM PROBES DONE ==="
echo "=== ALL NEW DATASET PROBES DONE ==="
