#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# LEARNABLE GW TEST: Run best config per dataset with learnable graph_weight
# Key: ours_wd=0.0001 to avoid pulling gw toward 0.5
# ===========================================================================

# 1. CORA (best: L=4 K=7 d=0.4 gw_init=0.8)
echo "=== GW1: Cora ==="
python -B main.py --method pcgt --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.0001 --ours_dropout 0.2

# 2. CITESEER (best: L=2 K=7 gw_init=0.7)
echo "=== GW2: Citeseer ==="
python -B main.py --method pcgt --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.0001 --ours_dropout 0.3

# 3. PUBMED (best: L=2 K=50 gw_init=0.8)
echo "=== GW3: PubMed ==="
python -B main.py --method pcgt --dataset pubmed --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --num_partitions 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.0001 --ours_dropout 0.3

# 4. ACTOR/FILM (best: L=2 K=10 gw_init=0.5 lr=0.1)
echo "=== GW4: Film ==="
python -B main.py --method pcgt --dataset film --lr 0.1 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.0001 --ours_dropout 0.3

# 5. SQUIRREL (best: L=4 K=10 gw_init=0.8)
echo "=== GW5: Squirrel ==="
python -B main.py --method pcgt --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.0001 --ours_dropout 0.3

# 6. CHAMELEON (best: L=2 K=10 gw_init=0.8)
echo "=== GW6: Chameleon ==="
python -B main.py --method pcgt --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.0001 --ours_dropout 0.3

echo "=== ALL LEARNABLE GW TESTS DONE ==="
