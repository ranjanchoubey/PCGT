#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# FINAL VALIDATION: 5-run for PubMed, 10-run for Film & Squirrel
# Then Deezer attack
# ===========================================================================

BASE_PLANETOID="--method pcgt --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --display_step 100 --ours_layers 1 --aggregate add"

# === 1. PUBMED 5-run ===
echo "=== PUBMED 5-RUN VALIDATION ==="
python -B main.py $BASE_PLANETOID --dataset pubmed --lr 0.01 --num_layers 2 --hidden_channels 64 --num_reps 4 --num_partitions 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3 --runs 5

# === 2. FILM 10-run (pre-computed splits) ===
echo "=== FILM 10-RUN VALIDATION ==="
python -B main.py --method pcgt --dataset film --lr 0.1 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --seed 123 --cpu --epochs 500 --patience 200 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 5 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3 --runs 10

# === 3. SQUIRREL 10-run (pre-computed splits) ===
echo "=== SQUIRREL 10-RUN VALIDATION ==="
python -B main.py --method pcgt --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3 --runs 10

# === 4. DEEZER ATTACK ===
# Problem: overfitting (train 98%, test 62%). SGFormer target: 67.1
# Try: more dropout, higher wd, fewer partitions, different gw
DZ="--method pcgt --dataset deezer-europe --partition_method metis --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu --epochs 500 --patience 200 --display_step 100 --ours_layers 1 --aggregate add --data_dir ../data/ --num_reps 4"

echo "=== DZ-A1: d=0.6 wd=5e-4 K=20 ==="
python -B main.py $DZ --lr 0.01 --hidden_channels 64 --num_layers 2 --num_partitions 20 --graph_weight 0.8 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.5 --runs 1

echo "=== DZ-A2: d=0.6 wd=0.001 K=10 h=96 ==="
python -B main.py $DZ --lr 0.01 --hidden_channels 96 --num_layers 2 --num_partitions 10 --graph_weight 0.8 --dropout 0.6 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.5 --runs 1

echo "=== DZ-A3: d=0.5 wd=0.001 K=20 gw=0.9 ==="
python -B main.py $DZ --lr 0.01 --hidden_channels 64 --num_layers 2 --num_partitions 20 --graph_weight 0.9 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.4 --runs 1

echo "=== DZ-A4: d=0.7 wd=0.005 K=20 h=96 ==="
python -B main.py $DZ --lr 0.01 --hidden_channels 96 --num_layers 2 --num_partitions 20 --graph_weight 0.8 --dropout 0.7 --weight_decay 0.005 --ours_weight_decay 0.01 --ours_dropout 0.5 --runs 1

echo "=== ALL FINAL RUNS DONE ==="
