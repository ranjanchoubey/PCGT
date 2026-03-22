#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# 5-RUN VALIDATION: Best configs for each dataset
# ===========================================================================

# --- CITESEER: layers=2 K=7 gw=0.7 wd=0.02 (single-run: 72.70) ---
echo "=== CITESEER 5-run: layers=2 ==="
python -B main.py --method pcgt --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

# --- Also try layers=2 with some variations ---
echo "=== CITESEER layers=2 gw=0.8 ==="
python -B main.py --method pcgt --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.8 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

echo "=== CITESEER layers=2 K=5 ==="
python -B main.py --method pcgt --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 5 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

echo "=== CITESEER layers=2 d=0.4 ==="
python -B main.py --method pcgt --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.7 --dropout 0.4 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

echo "=== CITESEER layers=2 wd=0.01 ==="
python -B main.py --method pcgt --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.01 --ours_dropout 0.3

# --- CORA 5-run: K=7 dropout=0.4 (single-run: 85.20) ---
echo "=== CORA 5-run: K=7 d=0.4 ==="
python -B main.py --method pcgt --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2

# --- PUBMED 5-run: K=100 (single-run: 80.90) ---
echo "=== PUBMED 5-run: K=100 ==="
python -B main.py --method pcgt --dataset pubmed --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 100 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# --- CHAMELEON 5-run: K=10 lr=0.01 (already have 49.15±2.65 from earlier) ---
echo "=== CHAMELEON 5-run: K=10 ==="
python -B main.py --method pcgt --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 200 --patience 100 --runs 5 --display_step 50 --aggregate add --num_partitions 10 --graph_weight 0.8 --dropout 0.4 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3 --data_dir ../data/

echo "=== ALL 5-RUN VALIDATIONS DONE ==="
