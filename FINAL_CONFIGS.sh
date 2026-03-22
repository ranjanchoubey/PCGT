# PCGT Final Hyperparameter Configurations & Results
# Last updated: 22 March 2026
# ============================================================

## VERIFIED 5-RUN RESULTS
# These are our final paper numbers (5-run mean ± std)

### Cora (5-run validated)
# Result: 84.56 ± 0.52 | SGFormer: 84.5 ± 0.8 | ✅ BEATS
python -B main.py --method pcgt --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2

### CiteSeer (5-run validated)
# Result: 73.02 ± 1.03 | SGFormer: 72.6 ± 0.2 | ✅ BEATS
python -B main.py --method pcgt --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

### Chameleon (5-run validated)
# Result: 48.59 ± 3.47 | SGFormer: 44.9 ± 3.9 | ✅ BEATS (+3.69)
python -B main.py --method pcgt --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3


## SINGLE-RUN BEST — NEED 5-RUN VALIDATION

### PubMed (1-run: 81.20)
# SGFormer: 80.3 ± 0.6 | ✅ BEATS in 1-run
python -B main.py --method pcgt --dataset pubmed --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

### Actor/Film (1-run best: 38.82, FA7 K=5)
# SGFormer: 37.9 ± 1.1 | ✅ BEATS in 1-run
# Note: Film uses pre-computed 10-fold splits, use --runs 10
python -B main.py --method pcgt --dataset film --lr 0.1 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 5 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

### Squirrel (1-run best: 48.23, SQ4 L=4 K=10)
# SGFormer: 41.8 ± 2.2 | ✅ BEATS by +6.4
# Note: Squirrel uses pre-computed 10-fold splits, use --runs 10
python -B main.py --method pcgt --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3


## NEEDS MORE TUNING

### Deezer (best so far: 62.35, target: 67.1 ± 1.1) ❌
# Overfitting badly (train 98%, test 62%). Need: more dropout, fewer epochs, different config
# SGFormer used: hidden=96, L=2, gw=0.8, d=0.4, wd=5e-5, ours_d=0.4
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu --epochs 500 --patience 200 --runs 1 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 20 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.3


## SUMMARY TABLE (as of 22 March 2026)
# | Dataset   | SGFormer     | PCGT (1-run) | PCGT (multi-run) | Status       |
# |-----------|------------- |--------------|------------------|--------------|
# | Cora      | 84.5 ± 0.8  | 85.20        | 84.56 ± 0.52     | ✅ 5-run done |
# | CiteSeer  | 72.6 ± 0.2  | 72.70        | 73.02 ± 1.03     | ✅ 5-run done |
# | PubMed    | 80.3 ± 0.6  | 81.20        | —                | ⏳ need 5-run |
# | Actor     | 37.9 ± 1.1  | 38.82        | —                | ⏳ need 10-run|
# | Squirrel  | 41.8 ± 2.2  | 48.23        | —                | ⏳ need 10-run|
# | Chameleon | 44.9 ± 3.9  | 48.45        | 48.59 ± 3.47     | ✅ 5-run done |
# | Deezer    | 67.1 ± 1.1  | 62.35        | —                | ❌ need attack|

## ARCHITECTURE NOTES
# - V4 (pcgt.py) = main architecture with learned pool_seeds
# - V6 (pcgt_v6.py) = ablation with centroid-based reps (no pool_seeds)
# - V5 (pcgt_v5.py) = FAILED boundary-routing experiment
# - Learnable graph_weight = FAILED (hurts heterophilic datasets), reverted to fixed gw
# - All use: METIS partitioning, PSE (partition structural encoding), GCN branch
