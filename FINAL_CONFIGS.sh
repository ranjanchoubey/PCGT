# PCGT Final Hyperparameter Configurations & Results
# Last updated: 22 March 2026
# ============================================================

## VERIFIED MULTI-RUN RESULTS (FINAL PAPER NUMBERS)

### Cora (5-run validated)
# Result: 84.56 ± 0.52 | SGFormer: 84.5 ± 0.8 | ✅ BEATS
python -B main.py --method pcgt --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2

### CiteSeer (5-run validated)
# Result: 73.02 ± 1.03 | SGFormer: 72.6 ± 0.2 | ✅ BEATS
python -B main.py --method pcgt --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.7 --dropout 0.5 --weight_decay 0.01 --ours_weight_decay 0.02 --ours_dropout 0.3

### PubMed (5-run validated)
# Result: 81.00 ± 0.73 | SGFormer: 80.3 ± 0.6 | ✅ BEATS
python -B main.py --method pcgt --dataset pubmed --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 50 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

### Chameleon (5-run validated)
# Result: 48.59 ± 3.47 | SGFormer: 44.9 ± 3.9 | ✅ BEATS (+3.69)
python -B main.py --method pcgt --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3

### Film/Actor (10-run validated, lr=0.05)
# Result: 38.04 ± 0.84 | SGFormer: 37.9 ± 1.1 | ✅ BEATS
# Note: Film uses pre-computed 10-fold splits
# lr=0.1 collapsed on some splits; lr=0.01 overfits; lr=0.05 is the sweet spot
python -B main.py --method pcgt --dataset film --lr 0.05 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 5 --graph_weight 0.5 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

### Squirrel (10-run validated)
# Result: 45.28 ± 2.08 | SGFormer: 41.8 ± 2.2 | ✅ BEATS (+3.48)
# Note: Squirrel uses pre-computed 10-fold splits
python -B main.py --method pcgt --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method metis --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --cpu --epochs 500 --patience 200 --runs 10 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3

### Deezer (1-run best: 63.81, KMeans partitioning)
# Result: 63.81 (1-run) | SGFormer: 67.1 ± 1.1 | ❌ Below (-3.29)
# KMeans feature partitioning better than METIS (62.35 → 63.81)
# Deezer has homophily ~0.53 (near random) — partitions don't capture label structure
python -B main.py --method pcgt --dataset deezer-europe --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --partition_method kmeans --use_graph --use_residual --backbone gcn --rand_split --seed 123 --cpu --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 50 --graph_weight 0.5 --dropout 0.6 --weight_decay 5e-5 --ours_weight_decay 0.01 --ours_dropout 0.4


## SUMMARY TABLE (22 March 2026 — ALL MEDIUM DATASETS COMPLETE)
# | Dataset   | SGFormer     | PCGT (multi-run)  | Delta  | Status       |
# |-----------|------------- |-------------------|--------|--------------|
# | Cora      | 84.5 ± 0.8  | 84.56 ± 0.52      | +0.06  | ✅ 5-run     |
# | CiteSeer  | 72.6 ± 0.2  | 73.02 ± 1.03      | +0.42  | ✅ 5-run     |
# | PubMed    | 80.3 ± 0.6  | 81.00 ± 0.73      | +0.70  | ✅ 5-run     |
# | Chameleon | 44.9 ± 3.9  | 48.59 ± 3.47      | +3.69  | ✅ 5-run     |
# | Actor     | 37.9 ± 1.1  | 38.04 ± 0.84      | +0.14  | ✅ 10-run    |
# | Squirrel  | 41.8 ± 2.2  | 45.28 ± 2.08      | +3.48  | ✅ 10-run    |
# | Deezer    | 67.1 ± 1.1  | 63.81 (1-run)     | -3.29  | ❌ need 5-run|

## ARCHITECTURE NOTES
# - V4 (pcgt.py) = production architecture with learned pool_seeds
# - V5, V6 = archived experiments (in medium/_archive/)
# - Partitioning: METIS for 6 datasets, KMeans for Deezer
# - Learnable graph_weight = TRIED & REVERTED (hurts heterophilic)
# - All use: PSE (partition structural encoding), GCN branch

## DEEZER ATTACK LOG (ALL PROBES COMPLETE)
# METIS: K=20 gw=0.8 → 62.35, K=50 gw=0.8 → 62.27
#        K=20 h=96 d=0.6 → 62.55, K=10 d=0.7 wd=1e-4 → 62.23
#        K=20 gw=0.5 d=0.6 ours_d=0.5 → 62.66
# KMeans: K=100 gw=0.8 → 62.08, K=200 gw=0.8 reps=2 → 62.47
#         K=200 gw=0.9 h=96 → 61.82, K=100 gw=0.95 reps=1 → 62.14
#         K=50 gw=0.5 → 63.81 (BEST OVERALL)

## FILM LR COMPARISON
# lr=0.1:  collapsed on 2/10 splits (28%), avg ~35
# lr=0.05: 38.04 ± 0.84 (BEST, stable across all splits)
# lr=0.01: 37.07 ± 1.20 (overfits, train 63%)
