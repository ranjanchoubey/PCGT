# PubMed PCGT — 🔴 HIGH PRIORITY

## Problem
- **Paper claims**: 80.5 ± 0.6
- **Current run.sh config** (layers=4, lr=0.005, gw=0.9, K=10): **79.28 ± 0.70** (5 runs)
- **Old best config** (layers=2, lr=0.01, gw=0.8, K=50): **81.00 ± 0.73** (5 runs)
- **Gap**: run.sh is ~1.2% below paper; old config actually exceeds paper claim

## Evidence

### Best multi-run result (from result file):
```
81.00 ± 0.73 (5 runs)
Config: layers=2, lr=0.01, gw=0.8, K=50, wd=0.0005, dropout=0.5, ours_decay=0.01, ours_dropout=0.3
```

### Best single-run:
```
80.80 (1 run, same config as above)
```

### Sweep best (gw0.9_K10, 5 runs):
```
80.30 ± 0.23 (this matches run.sh K/gw but different lr/layers)
```

### Current run.sh config (validated 5 runs):
```
79.28 ± 0.70 (5 runs)
Config: layers=4, lr=0.005, gw=0.9, K=10, wd=0.0005, dropout=0.5, ours_decay=0.01, ours_dropout=0.3
```

## Root Cause
The run.sh config (layers=4, lr=0.005) underperforms the old config (layers=2, lr=0.01, K=50).
The old config was already validated with 5 runs at 81.00 ± 0.73.

## Action Items
1. **Revalidate old best config on GPU (10 runs)**:
```bash
python -u main.py --data_dir ../data --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 50 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --display_step 100
```

2. **Also try K=10 with layers=2 (the gw0.9 sweep was layers=4)**:
```bash
python -u main.py --data_dir ../data --dataset pubmed --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.9 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --display_step 100
```

## Resolution
- [ ] Validated with ≥5 runs
- [ ] Best config identified
- [ ] run.sh updated to layers=2 config if it holds
- [ ] Paper table updated
