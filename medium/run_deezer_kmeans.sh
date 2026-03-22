#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== DEEZER KMEANS ATTACK ==="

echo "--- DZK-1: kmeans K=100, gw=0.8, d=0.5 ---"
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 \
  --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method kmeans \
  --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 100 --graph_weight 0.8 --dropout 0.5 \
  --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.3

echo "--- DZK-2: kmeans K=200, gw=0.8, d=0.6 ---"
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 \
  --hidden_channels 64 --ours_layers 1 --num_reps 2 --partition_method kmeans \
  --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 200 --graph_weight 0.8 --dropout 0.6 \
  --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.4

echo "--- DZK-3: kmeans K=200, gw=0.9, d=0.6, h=96 ---"
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 \
  --hidden_channels 96 --ours_layers 1 --num_reps 2 --partition_method kmeans \
  --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 200 --graph_weight 0.9 --dropout 0.6 \
  --weight_decay 1e-4 --ours_weight_decay 0.02 --ours_dropout 0.5

echo "--- DZK-4: kmeans K=100, gw=0.95, d=0.5, reps=1 (minimal transformer) ---"
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 \
  --hidden_channels 64 --ours_layers 1 --num_reps 1 --partition_method kmeans \
  --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 100 --graph_weight 0.95 --dropout 0.5 \
  --weight_decay 1e-4 --ours_weight_decay 0.02 --ours_dropout 0.4

echo "--- DZK-5: kmeans K=50, gw=0.5 (heterophilic-style, feature clusters) ---"
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 \
  --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method kmeans \
  --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu \
  --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add \
  --data_dir ../data/ --num_partitions 50 --graph_weight 0.5 --dropout 0.6 \
  --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.4

echo "=== DEEZER KMEANS ATTACK COMPLETE ==="
