# Experiment Manifest — PCGT

All experiments run on **NVIDIA H100 80GB** (Lightning AI studio).
Hardware: AMD EPYC 7763, 256 GB RAM, CUDA 12.1, PyTorch 2.6.0, PyG 2.7.0, pymetis 2025.2.2.

---

## Directory Structure

```
experiments/
├── logs/
│   ├── h100_validation/      # Round 1: HP tuning + SGFormer baselines (10 runs)
│   ├── h100_round2/          # Round 2: Final 5-run PCGT + timing pilots
│   ├── h100_round3/          # Round 3: Timing (3 runs) + GCN/GAT baselines (10 runs)
│   ├── archive/              # Earlier sweeps (Mac + old H100 runs)
│   └── medium_new/           # Intermediate medium-scale runs
├── results/                  # Root-dir results/*.txt (round 3 main + phase2)
├── medium/results/           # medium-dir results/*.txt (fix script + phase2)
├── scripts/                  # All GPU experiment scripts
│   ├── run_h100_round3.sh
│   ├── run_h100_round3_fix.sh
│   ├── run_h100_round3_phase2.sh
│   └── compute_ttest.py
├── convergence_data.json     # Per-epoch data for convergence plots
└── convergence_plot.py       # Script that generated convergence_data.json
```

---

## Key Results (Paper Tables)

### Table 1 — Main Results (7 medium-scale datasets, 10 runs)

Source: `logs/h100_validation/` (SGFormer) + `logs/h100_round2/` (PCGT 5-run → extended to 10)

| Dataset    | SGFormer        | PCGT (ours)     |
|------------|-----------------|-----------------|
| Cora       | 84.5 ± 0.8     | 84.3 ± 0.4     |
| CiteSeer   | 72.6 ± 0.2     | 73.1 ± 0.4     |
| PubMed     | 80.3 ± 0.6     | 81.0 ± 0.6     |
| Film       | 37.9 ± 1.1     | 38.0 ± 0.9     |
| Squirrel   | 41.8 ± 2.2     | 45.5 ± 2.7     |
| Chameleon  | 44.9 ± 3.9     | 49.0 ± 2.8     |
| Deezer     | 67.1 ± 1.1     | 67.2 ± 0.7     |

### Table 3 — Additional Datasets (4 datasets, 10 runs)

Source: `results/` dir txt files

| Method     | Co-CS       | Co-Phys     | Am-Comp     | Am-Photo    |
|------------|-------------|-------------|-------------|-------------|
| GCN        | 91.54±0.30  | 95.72±0.17  | 82.83±1.75  | 90.87±1.18  |
| GAT        | 92.04±0.30  | 95.36±0.15  | 83.37±0.85  | 91.10±0.78  |
| SGFormer   | 94.9 ± 0.5  | 96.6 ± 0.2  | 87.5 ± 2.0  | 95.2 ± 1.2  |
| PCGT       | 95.1 ± 0.3  | 96.8 ± 0.2  | 88.8 ± 0.7  | 95.3 ± 0.4  |

GCN baselines: `results/coauthor-cs_gcn.txt`, etc. (10 runs, round3 main script)
GAT baselines: `medium/results/coauthor-cs_gat.txt`, etc. (10 runs, phase2 script)

### Table 4 — ogbn-arxiv (169K nodes)

Source: `logs/h100_round3_phase2.log` (lines 509–726)

| Method   | Accuracy | ms/epoch |
|----------|----------|----------|
| SGFormer | 72.60    | 56.76    |
| PCGT     | 72.17    | 430.20   |

### Runtime Table (ms/epoch, 3 runs avg)

Source: `results/*.txt` run_time field + `medium/results/*.txt` run_time field

| Dataset    | Nodes   | K   | GCN    | SGFormer | PCGT   | Ratio  |
|------------|---------|-----|--------|----------|--------|--------|
| Cora       |  2,708  | 10  |  3.49  |  5.48    | 15.93  | 2.9×   |
| CiteSeer   |  3,327  | 20  |  3.72  |  5.41    | 26.80  | 5.0×   |
| PubMed     | 19,717  | 50  |  3.61  |  5.43    | 61.07  | 11.2×  |
| Chameleon  |  2,277  | 10  |  2.36  |  4.63    | 15.06  | 3.3×   |
| Squirrel   |  5,201  | 10  |  3.33  |  8.04    | 16.87  | 2.1×   |
| Film       |  7,600  | 5   |  3.88  |  5.87    | 10.24  | 1.7×   |
| Deezer     | 28,281  | 20  |  7.59  | 14.32    | 34.12  | 2.4×   |
| Co-CS      | 18,333  | 15  |  3.96  |  6.44    | 22.28  | 3.5×   |
| Co-Phys    | 34,493  | 20  |  6.58  |  9.85    | 30.51  | 3.1×   |
| Am-Comp    | 13,752  | 10  |  4.16  |  6.54    | 17.88  | 2.7×   |
| Am-Photo   |  7,650  | 10  |  3.44  |  5.62    | 16.91  | 3.0×   |
| ogbn-arxiv |169,343  | 256 |  —     | 56.76    |430.20  | 7.6×   |

---

## Experiment Rounds

### Round 1: HP Validation (h100_validation/)
- **Date**: 2026-03-24
- **Goal**: Validate SGFormer baselines + early PCGT HP tuning
- Logs: `logs/h100_validation/`

### Round 2: Final PCGT + Timing Pilots (h100_round2/)
- **Date**: 2026-03-24
- **Goal**: PCGT final 5-run results on 7 datasets + K-sweep + timing pilots
- Logs: `logs/h100_round2/`

### Round 3: Timing + Baselines + Large-Scale (h100_round3/)
- **Date**: 2026-03-25
- **Goal**: Complete timing table + GCN/GAT baselines + ogbn-arxiv
- Scripts: `scripts/run_h100_round3.sh` (main), `_fix.sh` (chameleon/squirrel/film path fix), `_phase2.sh` (GAT + arxiv)
- Master logs: `logs/h100_round3_master.log`, `h100_round3_fix.log`, `h100_round3_phase2.log`
- Individual logs: `logs/h100_round3/timing_*.log`, `gcn_*.log`
- **Note**: chameleon/squirrel/film require running from `medium/` dir due to hardcoded `../data/` paths in `medium/data_utils.py` lines 151, 169, 182

---

## How to Verify Results

Each `results/*.txt` file contains the full output including per-run scores, means±std, and run_time in ms.

```bash
# Example: check Cora PCGT timing
cat experiments/results/cora_pcgt_gcn.txt
# → Highest Test: 84.53 ± 0.51 ... run_time: 15.93

# Example: check GAT baselines
cat experiments/medium/results/coauthor-cs_gat.txt
# → Highest Test: 92.04 ± 0.30 ... run_time: 13.45
```

---

## Known Issues
- **Data path bug**: `medium/data_utils.py` loads chameleon/squirrel/film/deezer via `../data/wiki_new/` — must run from `medium/` directory
- **arxiv tee bug**: Phase2 `run_large()` function's `tee` path is relative to `large/` dir, so individual arxiv logs weren't created; data is in `h100_round3_phase2.log` master log instead
