#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# CITESEER ROUND 3: Refine around wd=0.02 winner (71.20)
# Target: 72.6. Gap: 1.4 points.
# ===========================================================================
CITE="--method pcgt --dataset citeseer --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 50 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --num_partitions 7"

echo "=== CITE R3-1: wd=0.02 ours_d=0.1 (less dropout, more wd) ==="
python main.py $CITE --ours_dropout 0.1 --ours_weight_decay 0.02

echo "=== CITE R3-2: wd=0.02 M=2 (fewer reps) ==="
python main.py $CITE --ours_dropout 0.3 --ours_weight_decay 0.02 --num_reps 2

echo "=== CITE R3-3: wd=0.02 lr=0.02 ==="
python main.py $CITE --ours_dropout 0.3 --ours_weight_decay 0.02 --lr 0.02

echo "=== CITE R3-4: wd=0.02 use_weight (learned V) ==="
python main.py $CITE --ours_dropout 0.3 --ours_weight_decay 0.02 --ours_use_weight

echo "=== CITE R3-5: wd=0.025 K=6 ==="
python main.py $CITE --ours_dropout 0.3 --ours_weight_decay 0.025 --num_partitions 6

echo "=== CITE R3-6: wd=0.02 K=8 ==="
python main.py $CITE --ours_dropout 0.3 --ours_weight_decay 0.02 --num_partitions 8

echo "=== CITE R3-7: wd=0.02 gw=0.75 ==="
python main.py $CITE --ours_dropout 0.3 --ours_weight_decay 0.02 --graph_weight 0.75

echo "=== CITE R3-8: wd=0.02 dropout=0.4 ==="
python main.py $CITE --ours_dropout 0.3 --ours_weight_decay 0.02 --dropout 0.4

# ===========================================================================
# CORA: Proper probes with --rand_split_class + original wd/dropout
# Current best: 84.18±0.52 (K=7). Target: 84.5.
# ===========================================================================
CORA="--method pcgt --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2"

echo "=== CORA P1: K=10 (more partitions) ==="
python main.py $CORA --num_partitions 10

echo "=== CORA P2: K=5 ==="
python main.py $CORA --num_partitions 5

echo "=== CORA P3: K=15 ==="
python main.py $CORA --num_partitions 15

echo "=== CORA P4: K=7 dropout=0.4 ==="
python main.py $CORA --num_partitions 7 --dropout 0.4

echo "=== CORA P5: K=7 wd=0.005 ==="
python main.py $CORA --num_partitions 7 --ours_weight_decay 0.005

echo "=== CORA P6: K=7 gw=0.7 ==="
python main.py $CORA --num_partitions 7 --graph_weight 0.7

echo "=== CORA P7: K=7 ours_d=0.1 ==="
python main.py $CORA --num_partitions 7 --ours_dropout 0.1

echo "=== CORA P8: K=7 M=8 ==="
python main.py $CORA --num_partitions 7 --num_reps 8

# ===========================================================================
# PUBMED: Proper probes with --rand_split_class
# Current best: 80.10 (K=50, 1 run). Target: 80.3.
# ===========================================================================
PUB="--method pcgt --dataset pubmed --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 300 --patience 150 --runs 1 --display_step 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3"

echo "=== PUB P1: K=50 (reconf with correct flags) ==="
python main.py $PUB --num_partitions 50

echo "=== PUB P2: K=30 ==="
python main.py $PUB --num_partitions 30

echo "=== PUB P3: K=100 ==="
python main.py $PUB --num_partitions 100

echo "=== PUB P4: K=50 gw=0.7 ==="
python main.py $PUB --num_partitions 50 --graph_weight 0.7

echo "=== PUB P5: K=50 ours_d=0.2 ==="
python main.py $PUB --num_partitions 50 --ours_dropout 0.2

echo "=== PUB P6: K=50 wd=0.005 ==="
python main.py $PUB --num_partitions 50 --ours_weight_decay 0.005

echo "=== ALL PROBES DONE ==="
