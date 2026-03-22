#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# V6 PROBES: Centroid-based global reps on all 4 datasets
# Using best-known HP configs from V4 experiments
# ===========================================================================

# --- CORA: best V4 config was K=7 d=0.4 → 84.56±0.52 ---
echo "=== V6 CORA K=7 d=0.4 ==="
python -B main.py --method pcgt_v6 --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2

echo "=== V6 CORA K=15 ==="
python -B main.py --method pcgt_v6 --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 15 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2

# --- CITESEER: best V4 was layers=2 K=7 → 73.02±1.03 ---
echo "=== V6 CITE layers=2 K=7 ==="
python -B main.py --method pcgt_v6 --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

echo "=== V6 CITE layers=2 K=5 ==="
python -B main.py --method pcgt_v6 --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 5 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# --- PUBMED: best V4 was K=100 → 80.12±0.59 ---
echo "=== V6 PUB K=100 ==="
python -B main.py --method pcgt_v6 --dataset pubmed --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 300 --patience 150 --runs 1 --display_step 100 --aggregate add --num_partitions 100 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

echo "=== V6 PUB K=50 ==="
python -B main.py --method pcgt_v6 --dataset pubmed --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 300 --patience 150 --runs 1 --display_step 100 --aggregate add --num_partitions 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# --- CHAMELEON: best V4 was K=10 → 48.59±3.47 ---
echo "=== V6 CHAM K=10 ==="
python -B main.py --method pcgt_v6 --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 200 --patience 100 --runs 1 --display_step 50 --aggregate add --num_partitions 10 --graph_weight 0.8 --dropout 0.4 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3 --data_dir ../data/

echo "=== V6 CHAM K=5 ==="
python -B main.py --method pcgt_v6 --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 200 --patience 100 --runs 1 --display_step 50 --aggregate add --num_partitions 5 --graph_weight 0.8 --dropout 0.4 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3 --data_dir ../data/

echo "=== ALL V6 PROBES DONE ==="
