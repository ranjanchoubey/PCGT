# Chameleon & Squirrel PCGT — 🟡 MEDIUM PRIORITY

## Status
Both have 10-run validated results from result files. Just need GPU reconfirmation.

---

## Chameleon

### Best validated result:
```
49.06 ± 2.97 (10 runs)
Config: lr=0.01, layers=2, h=64, gw=0.8, K=10, wd=0.001, dropout=0.5,
        ours_decay=0.01, ours_dropout=0.3, no_feat_norm=False, epochs=500
```

### K-sweep results (all 10 runs, same base config):
```
K=5:  48.65 ± 3.43
K=10: 49.06 ± 2.97  ← default
K=15: 48.94 ± 2.98
K=20: 48.67 ± 3.27
K=30: 49.32 ± 2.54  ← best K
```

### Paper claims: 48.1 ± 2.4
Our result (49.06) actually **exceeds** the paper claim.

### run.sh config:
```
lr=0.01, layers=2, h=64, gw=0.8, K=10, wd=0.001, dropout=0.5,
ours_decay=0.01, ours_dropout=0.3, no_feat_norm (with --no_feat_norm)
```

### ⚠️ Minor flag:
run.sh has `--no_feat_norm` but the result file shows `use_feat_norm:False`.
These should be the same thing. Confirm.

### Action:
```bash
python -u main.py --data_dir ../data --dataset chameleon --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.001 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --display_step 100
```

---

## Squirrel

### Best validated result:
```
45.28 ± 2.08 (10 runs)
Config: lr=0.01, layers=4, h=64, gw=0.8, K=10, wd=0.0005, dropout=0.5,
        ours_decay=0.01, ours_dropout=0.3, no_feat_norm=False, epochs=500
```

### K-sweep results (all 10 runs):
```
K=5:  45.30 ± 2.01
K=10: 45.28 ± 2.08  ← default
K=15: 45.23 ± 2.00
K=20: 45.32 ± 1.81
K=30: 45.04 ± 1.96
```

### Paper claims: 45.1 ± 2.3
Our result (45.28) matches the paper claim.

### run.sh config:
```
lr=0.01, layers=4, h=64, gw=0.8, K=10, wd=0.0005, dropout=0.5,
ours_decay=0.01, ours_dropout=0.3
```

### Action:
```bash
python -u main.py --data_dir ../data --dataset squirrel --method pcgt \
    --backbone gcn --lr 0.01 --num_layers 4 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --ours_layers 1 \
    --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual \
    --alpha 0.5 --ours_weight_decay 0.01 --no_feat_norm \
    --num_partitions 10 --partition_method metis \
    --seed 123 --runs 10 --epochs 500 --display_step 100
```

---

## Resolution
- [ ] Chameleon revalidated on GPU
- [ ] Squirrel revalidated on GPU
- [ ] Paper numbers confirmed
