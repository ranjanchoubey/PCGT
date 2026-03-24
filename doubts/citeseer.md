# CiteSeer PCGT — 🔴 HIGH PRIORITY

## Problem
- **Paper claims**: 73.4 ± 0.2
- **Current run.sh config** (layers=4, lr=0.005, gw=0.8, K=20): **70.56 ± 1.20** (5 runs)
- **Gap**: ~2.8%

## Evidence

### Best single-run result (from result file):
```
73.20 (1 run)
Config: layers=2, lr=0.01, gw=0.7, K=7, wd=0.01, dropout=0.5, ours_decay=0.01, ours_dropout=0.3
```

### Second best single-run:
```
73.00 (1 run)
Config: layers=2, lr=0.01, gw=0.7, K=7, wd=0.01, dropout=0.4, ours_decay=0.02, ours_dropout=0.3
```

### Current run.sh config (validated 5 runs):
```
70.56 ± 1.20 (5 runs)
Config: layers=4, lr=0.005, gw=0.8, K=20, wd=0.01, dropout=0.5, ours_decay=0.01, ours_dropout=0.3
```

### Sweep best (5 runs):
```
70.24 ± 0.82 (gw0.8_K20, 5 runs)
```

## Root Cause
The 73.20 result used a **completely different config** (layers=2, lr=0.01, gw=0.7, K=7)
vs run.sh (layers=4, lr=0.005, gw=0.8, K=20). The layers=2 config was only tested with
1 run so it could be a lucky seed.

## Action Items
1. **Run layers=2 config with 5 runs (seed 123)**:
```bash
python -u main.py --data_dir ../data --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 7 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --display_step 100
```

2. **Run layers=2 config with K=20 (5 runs)**:
```bash
python -u main.py --data_dir ../data --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 5 --display_step 100
```

3. **Run current run.sh config with 10 runs to get tighter estimate**:
```bash
python -u main.py --data_dir ../data --dataset citeseer --method pcgt \
    --backbone gcn --lr 0.005 --num_layers 4 --hidden_channels 64 \
    --weight_decay 0.01 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 20 --partition_method metis \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --runs 10 --display_step 100
```

## Resolution
- [ ] Validated with ≥5 runs
- [ ] Best config identified
- [ ] run.sh updated if needed
- [ ] Paper table updated
