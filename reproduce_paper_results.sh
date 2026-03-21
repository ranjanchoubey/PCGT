#!/bin/bash
# SGFormer paper reproduction - MANUAL command sheet
#
# This file is intentionally NOT an auto-run script.
# Open it and copy-paste commands one by one in your terminal.

# ============================================================
# 0) Environment
# ============================================================
# cd /Users/vn59a0h/thesis/PCGT
# source venv/bin/activate

# ============================================================
# 1) Download datasets used in the paper (11 datasets)
# ============================================================
# 1A) Planetoid (Cora, Citeseer, Pubmed)
# cd /Users/vn59a0h/thesis/PCGT
# python download_data.py --datasets cora citeseer pubmed

# 1B) Medium non-Planetoid datasets
# (Use the safe download-only script you already have)
# cd /Users/vn59a0h/thesis/PCGT
# bash download_all_paper_datasets_safe.sh

# Notes:
# - For paper medium experiments you need:
#   data/deezer/deezer-europe.mat
#   data/geom-gcn/film/out1_graph_edges.txt
#   data/geom-gcn/film/out1_node_feature_label.txt
#   data/geom-gcn/film/film_split_0.6_0.2_0..9.npz
#   data/wiki_new/chameleon/chameleon_filtered.npz
#   data/wiki_new/squirrel/squirrel_filtered.npz
# - For large experiments you need:
#   data/ogb/ogbn_arxiv/*
#   data/ogb/ogbn_proteins/*
#   data/ogb/ogbn_products/*
#   data/pokec/* (if available from Drive)

# ============================================================
# 2) Reproduce medium-size paper results (Table 1)
# ============================================================
# cd /Users/vn59a0h/thesis/PCGT/medium

# 2A) Cora
# python main.py --data_dir ../data/ --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
#     --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
#     --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
#     --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
#     --seed 123 --cpu --runs 5 --epochs 500

# 2B) Citeseer
# python main.py --data_dir ../data/ --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 \
#     --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
#     --method ours --ours_layers 1 --use_graph --graph_weight 0.7 \
#     --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
#     --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
#     --seed 123 --cpu --runs 5 --epochs 500

# 2C) Pubmed
# python main.py --data_dir ../data/ --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 \
#     --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
#     --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
#     --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
#     --seed 123 --cpu --runs 5 --epochs 500

# 2D) Film
# python main.py --data_dir ../data/ --backbone gcn --dataset film --lr 0.1 --num_layers 8 \
#     --hidden_channels 64 --weight_decay 0.0005 --dropout 0.6 \
#     --method difformer --use_graph --graph_weight 0.5 --num_heads 1 --ours_use_residual --ours_use_act \
#     --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 --cpu --runs 10 --epochs 500

# 2E) Deezer-Europe
# python main.py --data_dir ../data/ --backbone gcn --rand_split --dataset deezer-europe --lr 0.01 --num_layers 2 \
#     --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
#     --method ours --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
#     --alpha 0.5 --cpu --runs 5 --epochs 500

# 2F) Squirrel
# python main.py --data_dir ../data/ --backbone gcn --dataset squirrel --lr 0.001 --num_layers 8 \
#     --hidden_channels 64 --weight_decay 5e-4 --dropout 0.3 \
#     --method difformer --ours_layers 1 --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
#     --alpha 0.5 --cpu --runs 10 --epochs 200

# 2G) Chameleon
# python main.py --data_dir ../data/ --backbone gcn --dataset chameleon --lr 0.001 --num_layers 2 \
#     --hidden_channels 64 --ours_layers 1 --weight_decay 0.001 --dropout 0.6 \
#     --method ours --use_graph --num_heads 1 --ours_use_residual \
#     --alpha 0.5 --cpu --runs 10 --epochs 200

# ============================================================
# 3) Reproduce large-scale paper results (Table 2)
# ============================================================
# cd /Users/vn59a0h/thesis/PCGT/large

# 3A) ogbn-arxiv
# python main.py --method sgformer --dataset ogbn-arxiv --metric acc --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
#     --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. --trans_use_residual --trans_use_weight --trans_use_bn \
#     --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data/

# 3B) ogbn-proteins
# python main-batch.py --method sgformer --dataset ogbn-proteins --metric rocauc --lr 0.01 --hidden_channels 64 \
#     --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. --trans_use_residual --trans_use_weight --trans_use_bn \
#     --use_graph --graph_weight 0.5 --batch_size 10000 --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data/

# 3C) amazon2m
# python main-batch.py --method sgformer --dataset amazon2m --metric acc --lr 0.01 --hidden_channels 256 \
#     --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. --trans_use_residual --trans_use_weight --trans_use_bn \
#     --use_graph --graph_weight 0.5 --batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data/

# 3D) pokec
python main-batch.py --method sgformer --dataset pokec --rand_split --metric acc --lr 0.01 --hidden_channels 64 \
    --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. --trans_use_residual --trans_use_weight --trans_use_bn \
    --use_graph --graph_weight 0.5 --batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data/

# ============================================================
# End of manual command sheet
# ============================================================

echo "This is a manual copy-paste command sheet."
echo "Open this file and run commands one by one in terminal."
