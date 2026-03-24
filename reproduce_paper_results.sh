#!/bin/bash
# ============================================================
# PCGT Paper — Reproduction Command Sheet
# ============================================================
# This is a MANUAL reference file. Copy-paste commands to run.
# Each section shows SGFormer baseline + PCGT side by side.
#
# Usage: Open this file, copy the commands you need.
# DO NOT run this file directly (bash reproduce_paper_results.sh).
# ============================================================

echo "This is a manual reference file. Copy-paste commands to run."
echo "Do NOT run this file directly."
exit 0

# ============================================================
# 0) Setup
# ============================================================
# pip install -r requirements.txt
# bash download_data.sh

# ============================================================
# 1) MEDIUM-SCALE EXPERIMENTS (Table 1)
#    Run from: cd medium/
# ============================================================

# --- Cora ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset cora --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --rand_split_class --valid_num 500 --test_num 1000 \
#     --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset cora --backbone gcn \
#     --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --num_partitions 10 --partition_method metis \
#     --rand_split_class --valid_num 500 --test_num 1000 \
#     --seed 123 --runs 10 --epochs 500

# --- CiteSeer ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset citeseer --backbone gcn \
#     --lr 0.005 --num_layers 4 --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.3 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
#     --rand_split_class --valid_num 500 --test_num 1000 \
#     --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset citeseer --backbone gcn \
#     --lr 0.005 --num_layers 4 --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.3 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
#     --num_partitions 10 --partition_method metis \
#     --rand_split_class --valid_num 500 --test_num 1000 \
#     --seed 123 --runs 10 --epochs 500

# --- PubMed ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset pubmed --backbone gcn \
#     --lr 0.005 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
#     --rand_split_class --valid_num 500 --test_num 1000 \
#     --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset pubmed --backbone gcn \
#     --lr 0.005 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
#     --num_partitions 10 --partition_method metis \
#     --rand_split_class --valid_num 500 --test_num 1000 \
#     --seed 123 --runs 10 --epochs 500

# --- Chameleon ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset chameleon --backbone gcn \
#     --lr 0.001 --num_layers 2 --hidden_channels 64 --weight_decay 0.001 --dropout 0.6 \
#     --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual --alpha 0.5 \
#     --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset chameleon --backbone gcn \
#     --lr 0.001 --num_layers 2 --hidden_channels 64 --weight_decay 0.001 --dropout 0.6 \
#     --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual --alpha 0.5 \
#     --num_partitions 10 --partition_method metis \
#     --seed 123 --runs 10 --epochs 500

# --- Squirrel ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset squirrel --backbone gcn \
#     --lr 0.001 --num_layers 2 --hidden_channels 64 --weight_decay 0.001 --dropout 0.6 \
#     --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual --alpha 0.5 \
#     --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset squirrel --backbone gcn \
#     --lr 0.001 --num_layers 2 --hidden_channels 64 --weight_decay 0.001 --dropout 0.6 \
#     --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual --alpha 0.5 \
#     --num_partitions 10 --partition_method metis \
#     --seed 123 --runs 10 --epochs 500

# --- Deezer-Europe ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset deezer-europe --backbone gcn \
#     --lr 0.01 --num_layers 2 --hidden_channels 96 --weight_decay 5e-5 --dropout 0.4 \
#     --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual --alpha 0.5 \
#     --rand_split --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset deezer-europe --backbone gcn \
#     --lr 0.01 --num_layers 2 --hidden_channels 96 --weight_decay 5e-5 --dropout 0.4 \
#     --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual --alpha 0.5 \
#     --num_partitions 10 --partition_method metis \
#     --rand_split --seed 123 --runs 10 --epochs 500

# --- Film ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset film --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset film --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --num_partitions 10 --partition_method metis \
#     --seed 123 --runs 10 --epochs 500

# --- Coauthor-Physics ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset coauthor-physics --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --rand_split --seed 123 --runs 10 --epochs 500

# PCGT (K=20):
# python main.py --data_dir ../data --method pcgt --dataset coauthor-physics --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --num_partitions 20 --partition_method metis \
#     --rand_split --seed 123 --runs 10 --epochs 500

# --- Amazon-Computers ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset amazon-computers --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --rand_split --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset amazon-computers --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --num_partitions 10 --partition_method metis \
#     --rand_split --seed 123 --runs 10 --epochs 500

# --- Amazon-Photo ---
# SGFormer baseline:
# python main.py --data_dir ../data --method sgformer --dataset amazon-photo --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --rand_split --seed 123 --runs 10 --epochs 500

# PCGT (K=10):
# python main.py --data_dir ../data --method pcgt --dataset amazon-photo --backbone gcn \
#     --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --num_partitions 10 --partition_method metis \
#     --rand_split --seed 123 --runs 10 --epochs 500

# ============================================================
# 2) LARGE-SCALE EXPERIMENTS (Table 2)
#    Run from: cd large/
#    Requires GPU.
# ============================================================

# --- ogbn-arxiv (169K nodes) --- full-batch, main.py
# SGFormer baseline:
# python main.py --method sgformer --dataset ogbn-arxiv --metric acc \
#     --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
#     --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
#     --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
#     --trans_use_residual --trans_use_weight --trans_use_bn \
#     --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data

# PCGT (K=256):
# python main.py --method pcgt --dataset ogbn-arxiv --metric acc \
#     --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
#     --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. \
#     --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0. \
#     --trans_use_residual --trans_use_weight --trans_use_bn \
#     --num_partitions 256 --partition_method metis \
#     --seed 123 --runs 3 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data

# --- ogbn-proteins (132K nodes) --- mini-batch, main-batch.py
# SGFormer baseline:
# python main-batch.py --method sgformer --dataset ogbn-proteins --metric rocauc \
#     --lr 0.01 --hidden_channels 64 \
#     --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
#     --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
#     --trans_use_residual --trans_use_weight --trans_use_bn \
#     --use_graph --graph_weight 0.5 \
#     --batch_size 10000 --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data

# PCGT (K=256):
# python main-batch.py --method pcgt --dataset ogbn-proteins --metric rocauc \
#     --lr 0.01 --hidden_channels 64 \
#     --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
#     --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
#     --trans_use_residual --trans_use_weight --trans_use_bn \
#     --use_graph --graph_weight 0.5 \
#     --num_partitions 256 --partition_method metis \
#     --batch_size 10000 --seed 123 --runs 3 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data

# --- Pokec (1.6M nodes) --- mini-batch, main-batch.py
# SGFormer baseline:
# python main-batch.py --method sgformer --dataset pokec --rand_split --metric acc \
#     --lr 0.01 --hidden_channels 64 \
#     --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
#     --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
#     --trans_use_residual --trans_use_weight --trans_use_bn \
#     --use_graph --graph_weight 0.5 \
#     --batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data

# PCGT (K=500):
# python main-batch.py --method pcgt --dataset pokec --rand_split --metric acc \
#     --lr 0.01 --hidden_channels 64 \
#     --gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. \
#     --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
#     --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. \
#     --trans_use_residual --trans_use_weight --trans_use_bn \
#     --use_graph --graph_weight 0.5 \
#     --num_partitions 500 --partition_method metis \
#     --batch_size 100000 --seed 123 --runs 3 --epochs 1000 --eval_step 9 --device 0 --data_dir ../data

# ============================================================
# 3) ABLATION EXPERIMENTS
#    Run from: cd medium/
# ============================================================

# No PSE (disable partition structural encoding):
# python main.py --data_dir ../data --method pcgt --dataset cora --backbone gcn \
#     --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --num_partitions 10 --no_pse \
#     --seed 123 --runs 10 --epochs 500

# Local-only (intra-partition attention only):
# python main.py --data_dir ../data --method pcgt --dataset cora --backbone gcn \
#     --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --num_partitions 10 --local_only \
#     --seed 123 --runs 10 --epochs 500

# Global-only (cross-partition attention only):
# python main.py --data_dir ../data --method pcgt --dataset cora --backbone gcn \
#     --lr 0.01 --num_layers 3 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
#     --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 \
#     --use_residual --alpha 0.5 --ours_weight_decay 0.001 --no_feat_norm \
#     --num_partitions 10 --global_only \
#     --seed 123 --runs 10 --epochs 500
