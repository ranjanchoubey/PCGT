#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== FILM PROBE: lr=0.05, K=5, gw=0.5, d=0.5, runs=10 ==="
python -B main.py --method pcgt --dataset film --lr 0.05 --num_layers 2 \
  --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn --seed 123 --cpu --epochs 500 \
  --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ \
  --num_partitions 5 --graph_weight 0.5 --dropout 0.5 --weight_decay 5e-4 \
  --ours_weight_decay 0.01 --ours_dropout 0.3

echo "=== FILM PROBE: lr=0.01, K=5, gw=0.5, d=0.5, runs=10 ==="
python -B main.py --method pcgt --dataset film --lr 0.01 --num_layers 2 \
  --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn --seed 123 --cpu --epochs 500 \
  --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ \
  --num_partitions 5 --graph_weight 0.5 --dropout 0.5 --weight_decay 5e-4 \
  --ours_weight_decay 0.01 --ours_dropout 0.3

echo "=== SQUIRREL 10-RUN VALIDATION ==="
python -B main.py --method pcgt --dataset squirrel --lr 0.01 --num_layers 4 \
  --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 \
  --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

echo "=== DEEZER ATTACK ==="
echo "--- DZ-A1: h=96, d=0.6, wd=5e-5, ours_d=0.4 ---"
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 \
  --hidden_channels 96 --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 20 --graph_weight 0.8 --dropout 0.6 \
  --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.4

echo "--- DZ-A2: K=10, d=0.7, wd=1e-4 ---"
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 \
  --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.7 \
  --weight_decay 1e-4 --ours_weight_decay 0.02 --ours_dropout 0.5

echo "--- DZ-A3: gw=0.5, K=20, d=0.6, ours_d=0.5 ---"
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 \
  --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 20 --graph_weight 0.5 --dropout 0.6 \
  --weight_decay 5e-5 --ours_weight_decay 0.02 --ours_dropout 0.5

echo "=== ALL PROBES COMPLETE ==="
