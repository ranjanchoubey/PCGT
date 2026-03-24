# Film PCGT — 🔴 HIGH PRIORITY

## Problem
- **Paper claims**: 37.7 ± 1.0
- **Current run (8/10 done, Mac CPU)**: Individual runs: 29.54, 26.12, 35.26, 34.61, 31.64, 35.33, 31.51, 30.20 → avg ~31.8%
- **Gap**: ~6% below paper claim. This is the LARGEST discrepancy.

## Evidence

### Current run.sh config (running on Mac):
```
Run results so far (8/10):
  Run 01: 29.54  Run 02: 26.12  Run 03: 35.26  Run 04: 34.61
  Run 05: 31.64  Run 06: 35.33  Run 07: 31.51  Run 08: 30.20
Average: ~31.8%
Config: lr=0.1, layers=8, gw=0.6, K=5, ours_layers=2, h=64,
        wd=0.0005, dropout=0.6, ours_dropout=0.6, ours_wd=0.0005,
        ours_use_residual, ours_use_act, seed=42, runs=10
```

### Old exploratory single-run results (from film_pcgt_gcn.txt):
```
39.28 (1 run) — lr=0.1, gw=0.5, K=10, layers=2, ours_decay=0.0001
38.82 (1 run) — lr=0.1, gw=0.5, K=5, layers=2
38.62 (1 run) — lr=0.1, gw=0.5, K=10, layers=2
38.04 ± 0.84 (10 runs!) — lr=0.05, gw=0.5, K=5, layers=2, dropout=0.5
37.07 ± 1.20 (10 runs) — lr=0.01, gw=0.5, K=5, layers=2
```

### Sweep results (5 runs each, layers/gw grid):
```
Best: 33.43 ± 3.83 (gw0.6_K5) — matches run.sh gw but very different result
       33.32 ± 1.59 (gw0.6_K10)
       32.46 ± 4.04 (gw0.7_K10)
```

## Root Cause Analysis
1. The current run.sh config uses **layers=8** (deep GCN) which may be causing issues
2. The old best results all used **layers=2** with lr=0.1 or 0.05
3. The 38.04 ± 0.84 (10 runs, lr=0.05, layers=2) is the BEST validated multi-run result
4. The run.sh config with ours_layers=2 + layers=8 may be fundamentally different from optimal
5. Extreme variance (26% to 37% per run) suggests training instability

## Action Items

1. **Revalidate the best 10-run config (lr=0.05, layers=2, K=5)**:
```bash
python -u main.py --data_dir ../data --dataset film --method pcgt \
    --backbone gcn --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 5 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --display_step 100
```

2. **Try the single-run best config (lr=0.1, layers=2, K=10, ours_decay=0.0001) with 10 runs**:
```bash
python -u main.py --data_dir ../data --dataset film --method pcgt \
    --backbone gcn --lr 0.1 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.6 --ours_layers 1 \
    --use_graph --graph_weight 0.5 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.0001 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --display_step 100
```

3. **Try the current run.sh config but with layers=2 instead of 8**:
```bash
python -u main.py --data_dir ../data --dataset film --method pcgt \
    --backbone gcn --lr 0.1 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.0005 --dropout 0.6 --ours_layers 2 \
    --use_graph --graph_weight 0.6 --ours_dropout 0.6 \
    --ours_use_residual --ours_use_act \
    --alpha 0.5 --ours_weight_decay 0.0005 \
    --num_partitions 5 --partition_method metis \
    --seed 42 --runs 10 --epochs 500 --display_step 100
```

## Key Insight
The film_pcgt_gcn.txt result file shows **38.04 ± 0.84 (10 runs)** with a simpler config
(layers=2, lr=0.05, gw=0.5, K=5). This is close to paper claim of 37.7. The run.sh config
(layers=8) seems to be wrong — possibly copied from the SGFormer/DIFFormer Film config.

## Resolution
- [ ] Identified root cause (layers=8 vs layers=2)
- [ ] Validated best config on GPU
- [ ] run.sh updated
- [ ] Paper table updated
