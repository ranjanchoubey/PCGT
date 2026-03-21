#!/bin/bash
# Run PCGT v2 experiments sequentially
cd /Users/vn59a0h/thesis/PCGT/medium

echo "=== CITESEER ===" 
/Users/vn59a0h/thesis/PCGT/venv/bin/python main.py --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 --method pcgt --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --device 0 --runs 5 --num_partitions 5 --data_dir /Users/vn59a0h/thesis/PCGT/data/ 2>&1 | tail -5

echo "=== CHAMELEON ==="
/Users/vn59a0h/thesis/PCGT/venv/bin/python main.py --backbone gcn --dataset chameleon --lr 0.001 --num_layers 2 --hidden_channels 64 --ours_layers 1 --weight_decay 0.001 --dropout 0.6 --method pcgt --use_graph --num_heads 1 --ours_use_residual --alpha 0.5 --device 0 --runs 10 --epochs 200 --num_partitions 3 --data_dir /Users/vn59a0h/thesis/PCGT/data/ 2>&1 | tail -5

echo "ALL_DONE"
