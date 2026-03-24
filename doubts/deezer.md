# Deezer PCGT — 🟡 MEDIUM PRIORITY

## Problem
- **Paper claims**: 67.2 ± 0.5
- **Sweep best** (gw0.7_K20, 5 runs): **66.58 ± 0.61**
- **Gap**: ~0.6% — within random variance but worth confirming

## Evidence

### Sweep results (5 runs each):
```
gw0.7_K20: 66.58 ± 0.61 ← best
gw0.7_K15: 66.26 ± 0.53
gw0.8_K10: 66.31 ± 0.62
gw0.7_K10: 66.09 ± 0.49
gw0.7_K5:  66.04 ± 0.55
```

### Current run.sh config:
```
K=20, gw=0.7, h=96, layers=2, lr=0.01, wd=5e-5, dropout=0.4,
ours_layers=1, ours_dropout=0.4, ours_wd=5e-5, ours_use_residual,
seed=42, runs=5, epochs=500
```

### Currently running on Mac CPU (VERY slow, barely started):
```
Run 1, epoch ~100. Deezer has 28K nodes.
```

### Old SGFormer (ours) baseline:
```
66.16 (1 run) — SGFormer config
```

## Root Cause
- The sweep used seed=123 and slightly different base HPs from run.sh
- run.sh uses seed=42, h=96, ours_use_residual — the sweep may have used
  different base params
- The 0.6% gap could be seed variance or the h=96 vs h=64 difference

## Action Items
1. **Run current run.sh config on GPU (5 runs)**:
```bash
python -u main.py --data_dir ../data --dataset deezer-europe --method pcgt \
    --backbone gcn --rand_split --lr 0.01 --num_layers 2 \
    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
    --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.4 \
    --ours_use_residual --alpha 0.5 --ours_weight_decay 5e-05 \
    --num_partitions 20 --partition_method metis \
    --seed 42 --runs 5 --epochs 500 --display_step 100
```

2. **Also try with seed=123 for comparison**:
```bash
# Same command but --seed 123
```

## Resolution
- [ ] Validated on GPU with run.sh config
- [ ] Confirmed final number
- [ ] Paper table updated
