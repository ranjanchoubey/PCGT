#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# V5 PROBES: Boundary-Routed attention on all 4 datasets
# ===========================================================================

# --- CORA: Target 84.5 ---
CORA="--method pcgt_v5 --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2"

echo "=== V5 CORA K=7 ==="
python -B main.py $CORA --num_partitions 7

echo "=== V5 CORA K=10 ==="
python -B main.py $CORA --num_partitions 10

echo "=== V5 CORA K=5 ==="
python -B main.py $CORA --num_partitions 5

echo "=== V5 CORA K=15 ==="
python -B main.py $CORA --num_partitions 15

# --- CITESEER: Target 72.6 (hardest) ---
CITE="--method pcgt_v5 --dataset citeseer --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 50 --dropout 0.5 --weight_decay 0.01"

echo "=== V5 CITE K=7 gw=0.7 wd=0.02 ==="
python -B main.py $CITE --graph_weight 0.7 --ours_dropout 0.3 --ours_weight_decay 0.02 --num_partitions 7

echo "=== V5 CITE K=7 gw=0.7 wd=0.01 ==="
python -B main.py $CITE --graph_weight 0.7 --ours_dropout 0.3 --ours_weight_decay 0.01 --num_partitions 7

echo "=== V5 CITE K=7 gw=0.7 wd=0.015 ==="
python -B main.py $CITE --graph_weight 0.7 --ours_dropout 0.3 --ours_weight_decay 0.015 --num_partitions 7

echo "=== V5 CITE K=5 gw=0.7 wd=0.02 ==="
python -B main.py $CITE --graph_weight 0.7 --ours_dropout 0.3 --ours_weight_decay 0.02 --num_partitions 5

echo "=== V5 CITE K=10 gw=0.7 wd=0.02 ==="
python -B main.py $CITE --graph_weight 0.7 --ours_dropout 0.3 --ours_weight_decay 0.02 --num_partitions 10

echo "=== V5 CITE K=7 gw=0.8 wd=0.02 ==="
python -B main.py $CITE --graph_weight 0.8 --ours_dropout 0.3 --ours_weight_decay 0.02 --num_partitions 7

# --- PUBMED: Target 80.3 ---
PUB="--method pcgt_v5 --dataset pubmed --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 300 --patience 150 --runs 1 --display_step 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3"

echo "=== V5 PUB K=50 ==="
python -B main.py $PUB --num_partitions 50

echo "=== V5 PUB K=30 ==="
python -B main.py $PUB --num_partitions 30

echo "=== V5 PUB K=100 ==="
python -B main.py $PUB --num_partitions 100

# --- CHAMELEON: Target 44.9 (already beaten with V4) ---
CHAM="--method pcgt_v5 --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 200 --patience 100 --runs 1 --display_step 50 --graph_weight 0.8 --dropout 0.4 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3 --data_dir ../../data/"

echo "=== V5 CHAM K=10 ==="
python -B main.py $CHAM --num_partitions 10

echo "=== V5 CHAM K=5 ==="
python -B main.py $CHAM --num_partitions 5

echo "=== ALL V5 PROBES DONE ==="
