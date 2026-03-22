# Experiment Plan - GPU Validation & Deezer Recovery

## Overview
Systematic benchmarking of PCGT (Partition-Conditioned Graph Transformer) against SGFormer paper baseline across 7 medium-scale graphs.

---

## Phase 1: GPU Baseline Validation (COMPLETED)

### Objectives
- ✅ Validate PCGT on all 7 datasets with GPU acceleration
- ✅ Reproduce paper-ready results
- ✅ Achieve "6/7 datasets beat or match SGFormer"

### Experiments
| # | Dataset | Runs | Method | Status |
|---|---------|------|--------|--------|
| 1 | Cora | 5 | GPU + standard HP | ✅ MATCHED |
| 2 | CiteSeer | 5 | GPU + standard HP | ✅ WIN (+0.84) |
| 3 | PubMed | 5 | GPU + standard HP | ✅ WIN (+0.16) |
| 4 | Chameleon | 5 | GPU + standard HP | ✅ WIN (+3.19) |
| 5 | Film | 5 | GPU + standard HP | ✅ MATCHED |
| 6 | Squirrel | 5 | GPU + standard HP | ✅ WIN (+3.34) |
| 7 | Deezer | 5 | GPU + standard HP | ❌ LOSS (-2.16) |

### Metrics & Evaluation
- **Primary metric**: Test accuracy (classification)
- **Reporting**: Mean ± std over 5 runs (random seeds)
- **Baseline comparison**: SGFormer paper results (paper table)
- **Stability metric**: Std dev (CiteSeer ±0.21 = most stable)

### Datasets Characteristics
| Dataset | Type | Nodes | Features | Classes | Homophily | Split |
|---------|------|-------|----------|---------|-----------|-------|
| Cora | Homophilic | 2.7K | 1.4K | 7 | 0.81 | 0.6/0.2/0.2 |
| CiteSeer | Homophilic | 3.3K | 3.7K | 6 | 0.74 | 0.6/0.2/0.2 |
| PubMed | Homophilic | 19.7K | 500 | 3 | 0.80 | 0.6/0.2/0.2 |
| Chameleon | Heterophilic | 2.3K | 2.3K | 5 | 0.46 | 0.6/0.2/0.2 |
| Film | Heterophilic | 7.6K | 932 | 5 | 0.44 | 0.6/0.2/0.2 |
| Squirrel | Heterophilic | 5.2K | 2.1K | 5 | 0.22 | 0.6/0.2/0.2 |
| Deezer | Special | 28K | 31K | 2 | 0.53 | 0.5/0.25/0.25 |

---

## Phase 2: Deezer Systematic Attack (ONGOING)

### Objective
Recover Deezer performance from 64.94 (−2.16 gap) to >65.5 (strong recovery) or >66.5 (match SGFormer).

### Root Cause Analysis
- **Problem**: Severe overfitting (97% train → 64% test)
- **Data uniqueness**:
  - Binary classification (2 classes; all others 3+ classes)
  - Extremely high-dimensional (31K features vs 0.5–3.7K typical)
  - Near-homophily (h≈0.53 vs 0.74–0.81 typical homophilic)
  - Different sparsity/structure than benchmark

### 12-Config Sweep Design

**Strategy**: Hypothesis-driven testing across 4 batches.

#### Probe Phase
- Run each config with **3 random seeds** (fast validation)
- Log test accuracies to Drive
- Identify top 3–4 performers

#### Validation Phase
- Re-run top config with **5 random seeds** (paper number)
- Update FINAL_CONFIGS.sh with best result

### Config Batches

**Batch 1: Capacity Reduction (B1–B3)**
- Hypothesis: 64 hidden dims + 4 reps too large for binary task
- Variants: hidden=32|48|32, reps=2|2|1
- Expected outcome: Simpler model → less overfitting

**Batch 2: Regularization Sweep (B4–B6)**
- Hypothesis: Overfitting primary enemy; need aggressive dropout/weight_decay
- Variants: dropout 0.7|0.8|0.6, weight_decay 5e-3|1e-2|5e-4
- Expected outcome: Reduced train/test gap

**Batch 3: Architecture Variants (B7–B10)**
- Hypothesis A (B7–B8): BatchNorm stabilizes 31K features
- Hypothesis B (B9–B10): Partitioning strategy matters
- Variants: 
  - B7: +BatchNorm, gw=0.9 (GCN-heavy)
  - B8: +BatchNorm, gw=0.3 (PCGT-heavy)
  - B9: num_partitions=20 (fewer, larger partitions)
  - B10: partition_method=random (ablate METIS)
- Expected outcome: Stabilize via norm, find optimal partition granularity

**Batch 4: Feature & Baseline (B11–B12)**
- Hypothesis A (B11): 31K sparse features need special handling
- Hypothesis B (B12): Pure GCN ceiling (remove PCGT)
- Variants:
  - B11: skip feature normalization
  - B12: graph_weight=1.0 (pure GCN)
- Expected outcome: Identify if MLP-readout or PCGT blend hurt binary task

### Success Criteria
| Outcome | Threshold | Action |
|---------|-----------|--------|
| Strong Recovery | >65.5 | Update FINAL_CONFIGS.sh, report as quasi-win |
| Match SGFormer | >66.5 | Report as win (within SGFormer's ±1.1) |
| Acceptable Loss | 64.94 | Accept; report: heterophilic wins (4 × +3.34, +3.19) outweigh |

---

## Execution Environment

### Software Stack
- **PyTorch**: 2.0+
- **torch_geometric**: 2.3.0+
- **METIS**: Via pymetis (graph partitioning)
- **Compute**: Colab GPU (T4/A100/V100) preferred
- **Data**: Pre-uploaded to Google Drive `/content/drive/MyDrive/PCGT_datasets/`

### Standard Hyperparameters (Baseline)
```
--hidden_channels 64
--num_layers 3
--ours_layers 3
--num_reps 4
--partition_method kmeans (or metis)
--num_partitions 50
--graph_weight 0.5
--dropout 0.6
--weight_decay 5e-4
--ours_dropout 0.6
--ours_weight_decay 5e-4
--runs 5
--lr 0.01
--epochs 200
--early_stopping 50
```

### Logging & Results
- All outputs to **Google Drive-backed directory** (persistent across Colab restarts)
- Format: `tee` to `.txt` files (timestamp-organized)
- Auto-parsing: Python script extracts best config from 12 logs

---

## Expected Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1 (6 datasets) | 2 hours | ✅ COMPLETE |
| Phase 2 Probe (12×3 runs) | ~4–6 hours | ⏳ Running |
| Phase 2 Validation (best×5 runs) | ~40 min | Pending |
| Documentation & Summary | 1 hour | ⏳ In progress |

---

## Variables to Track

### Per-Config
- Test accuracy (primary)
- Train accuracy (overfitting indicator)
- Early stopping epoch (convergence speed)
- Std dev across 3 seeds (stability)

### Aggregate
- Best config (highest test accuracy)
- Test gap to SGFormer (paper gap)
- Homophilic vs heterophilic win rate (4/6 homophilic baseline)
