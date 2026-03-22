#!/bin/bash
cd /Users/vn59a0h/thesis/PCGT/medium
source /Users/vn59a0h/thesis/PCGT/venv/bin/activate

# ===========================================================================
# FILM/ACTOR ATTACK: Beat SGFormer 37.9±1.1
# Round 1 learnings: gw=0.5 best (36.25), L=4 collapsed, lr=0.01 may be too low
# SGFormer used: lr=0.1, L=8, d=0.6, gw=0.5 for Film
# Film = sparse binary features (932 dims), 7600 nodes, 5 classes
# Uses pre-computed 10-fold splits (no --rand_split)
# ===========================================================================

FILM_BASE="--method pcgt --dataset film --partition_method metis --use_graph --use_residual --backbone gcn --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --ours_layers 1 --aggregate add --data_dir ../data/"

# FA1: lr=0.1 L=2 K=20 gw=0.5 d=0.6 (SGFormer-style LR + our best gw)
echo "=== FA1: lr=0.1 L=2 K=20 gw=0.5 d=0.6 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# FA2: lr=0.05 L=2 K=20 gw=0.5 d=0.6
echo "=== FA2: lr=0.05 L=2 K=20 gw=0.5 d=0.6 ==="
python -B main.py $FILM_BASE --lr 0.05 --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# FA3: lr=0.1 L=2 K=10 gw=0.5 d=0.6
echo "=== FA3: lr=0.1 L=2 K=10 gw=0.5 d=0.6 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 2 --num_partitions 10 --num_reps 4 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# FA4: lr=0.1 L=2 K=20 gw=0.3 d=0.6 (even less GCN)
echo "=== FA4: lr=0.1 L=2 K=20 gw=0.3 d=0.6 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.3 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# FA5: lr=0.1 L=2 K=20 gw=0.5 d=0.5 ours_d=0.5 (more balanced dropout)
echo "=== FA5: lr=0.1 L=2 K=20 gw=0.5 d=0.5 ours_d=0.5 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.5 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.5

# FA6: lr=0.1 L=2 K=20 gw=0.5 d=0.6 ours_d=0.6 (heavy attention dropout)
echo "=== FA6: lr=0.1 L=2 K=20 gw=0.5 d=0.6 ours_d=0.6 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.6

# FA7: lr=0.1 L=2 K=5 gw=0.5 d=0.6 (fewer partitions = bigger local groups)
echo "=== FA7: lr=0.1 L=2 K=5 gw=0.5 d=0.6 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 2 --num_partitions 5 --num_reps 4 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# FA8: lr=0.1 L=2 K=20 gw=0.5 d=0.6 wd=0.001 ours_wd=0.005
echo "=== FA8: lr=0.1 L=2 K=20 gw=0.5 d=0.6 wd=0.001 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.5 --dropout 0.6 --weight_decay 0.001 --ours_weight_decay 0.005 --ours_dropout 0.3

# FA9: lr=0.1 L=3 K=20 gw=0.5 d=0.6 (one more layer)
echo "=== FA9: lr=0.1 L=3 K=20 gw=0.5 d=0.6 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 3 --num_partitions 20 --num_reps 4 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

# FA10: lr=0.1 L=2 K=20 gw=0.0 d=0.6 (pure transformer, no GCN)
echo "=== FA10: lr=0.1 L=2 K=20 gw=0.0 d=0.6 ==="
python -B main.py $FILM_BASE --lr 0.1 --hidden_channels 64 --num_layers 2 --num_partitions 20 --num_reps 4 --graph_weight 0.0 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

echo "=== ALL FILM ATTACK DONE ==="
