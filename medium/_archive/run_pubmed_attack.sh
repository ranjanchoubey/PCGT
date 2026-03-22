#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# PUBMED ATTACK: Push past 80.3
# Current best 5-run: 80.12±0.59 (K=100, layers=4, gw=0.8, d=0.5, ours_wd=0.01, ours_d=0.3)
# ===========================================================================

BASE="--method pcgt --dataset pubmed --lr 0.01 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --ours_layers 1 --aggregate add"

# P1: layers=2 K=100 (less over-smoothing — saved Citeseer)
echo "=== PUB P1: layers=2 K=100 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 2 --num_partitions 100 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# P2: layers=2 K=50
echo "=== PUB P2: layers=2 K=50 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 2 --num_partitions 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# P3: layers=3 K=100
echo "=== PUB P3: layers=3 K=100 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 3 --num_partitions 100 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# P4: dropout=0.4 K=100 (helped Cora)
echo "=== PUB P4: d=0.4 K=100 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 100 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# P5: ours_d=0.2 K=100 (less attention dropout)
echo "=== PUB P5: ours_d=0.2 K=100 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 100 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.2

# P6: K=200 (more partitions, ~99 nodes each)
echo "=== PUB P6: K=200 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 200 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# P7: gw=0.9 K=100 (more GCN)
echo "=== PUB P7: gw=0.9 K=100 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 100 --graph_weight 0.9 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# P8: ours_wd=0.005 K=100 (less attention WD)
echo "=== PUB P8: ours_wd=0.005 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 100 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.005 --ours_dropout 0.3

# P9: d=0.4 ours_d=0.2 K=100 (double relaxation)
echo "=== PUB P9: d=0.4 ours_d=0.2 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 100 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.2

# P10: layers=2 d=0.4 K=100 (combine two winners)
echo "=== PUB P10: layers=2 d=0.4 K=100 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 2 --num_partitions 100 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# P11: K=50 ours_d=0.2 (earlier single-run hit 80.40)
echo "=== PUB P11: K=50 ours_d=0.2 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.2

# P12: wd=0.001 K=100 (less backbone WD)
echo "=== PUB P12: wd=0.001 K=100 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 100 --graph_weight 0.8 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3

echo "=== ALL PUBMED ATTACK DONE ==="
