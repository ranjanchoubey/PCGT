#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# CITESEER ATTACK: Aggressive HP search to beat 72.6
# Best so far: V5 71.30 (K=7, gw=0.7, wd=0.02, ours_d=0.3, d=0.5, decay=0.01)
# ===========================================================================

BASE="--method pcgt --dataset citeseer --lr 0.01 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 50 --ours_layers 1 --aggregate add"

# Probe 1: hidden=128 (more capacity)
echo "=== CITE P1: hidden=128 ==="
python -B main.py $BASE --hidden_channels 128 --num_layers 4 --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 2: num_layers=2 (less over-smoothing)
echo "=== CITE P2: layers=2 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 2 --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 3: num_layers=3
echo "=== CITE P3: layers=3 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 3 --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 4: M=8 (more attention reps)
echo "=== CITE P4: M=8 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --num_reps 8 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 5: dropout=0.4 (like Cora best)
echo "=== CITE P5: dropout=0.4 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --graph_weight 0.7 --dropout 0.4 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 6: dropout=0.3
echo "=== CITE P6: dropout=0.3 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --graph_weight 0.7 --dropout 0.3 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 7: backbone wd=0.005
echo "=== CITE P7: wd=0.005 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.005 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 8: gw=0.5 (less GCN, more attention)
echo "=== CITE P8: gw=0.5 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --graph_weight 0.5 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 9: gw=0.6
echo "=== CITE P9: gw=0.6 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --graph_weight 0.6 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 10: gw=0.9 (more GCN)
echo "=== CITE P10: gw=0.9 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --graph_weight 0.9 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 11: K=3 (fewer, larger partitions) 
echo "=== CITE P11: K=3 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 3 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 12: K=15 (more partitions)
echo "=== CITE P12: K=15 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 15 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 13: ours_dropout=0.5 (more regularization on attention)
echo "=== CITE P13: ours_d=0.5 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.5

# Probe 14: ours_dropout=0.1 ours_wd=0.03
echo "=== CITE P14: ours_d=0.1 wd=0.03 ==="
python -B main.py $BASE --hidden_channels 64 --num_layers 4 --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.03 --ours_dropout 0.1

# Probe 15: hidden=128 + dropout=0.4 + wd=0.005
echo "=== CITE P15: h=128 d=0.4 wd=0.005 ==="
python -B main.py $BASE --hidden_channels 128 --num_layers 4 --num_partitions 7 --graph_weight 0.7 --dropout 0.4 --weight_decay 0.005 --ours_weight_decay 0.02 --ours_dropout 0.3

# Probe 16: lr=0.005 (slower learning)
echo "=== CITE P16: lr=0.005 ==="
python -B main.py --method pcgt --dataset citeseer --lr 0.005 --hidden_channels 64 --num_layers 4 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 50 --ours_layers 1 --aggregate add --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# --- Also run V5 Chameleon with correct data_dir ---
echo "=== V5 CHAM K=10 (fixed) ==="
python -B main.py --method pcgt_v5 --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 200 --patience 100 --runs 1 --display_step 50 --graph_weight 0.8 --dropout 0.4 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3 --data_dir ../data/

echo "=== V5 CHAM K=5 (fixed) ==="
python -B main.py --method pcgt_v5 --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 200 --patience 100 --runs 1 --display_step 50 --graph_weight 0.8 --dropout 0.4 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3 --data_dir ../data/

echo "=== ALL CITESEER ATTACK + CHAMELEON V5 DONE ==="
