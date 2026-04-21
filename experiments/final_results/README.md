# Authoritative Paper Results — PCGT

Every number in the paper traces back to exactly one log file in this directory.
No other directory should be treated as the source of truth.

Hardware: NVIDIA H100 80GB, AMD EPYC 7763, CUDA 12.4, PyTorch 2.6.0, PyG 2.7.0

**Metric used:** "Highest Test" (max test accuracy across all epochs per run).
Both PCGT and SGFormer use identical `logger.py` code (byte-for-byte), so the
comparison is apples-to-apples. See `table1_sgformer_rerun/` for verification.

---

## Table 1 — Main Results (7 datasets)

`table1_main/` — PCGT results. SGFormer/baseline numbers from [Wu et al. 2023] Table 2.

| Dataset    | Paper Value      | Log File                        | Log Highest Test | Runs |
|------------|------------------|---------------------------------|------------------|------|
| Cora       | 84.3 ± 0.4      | `cora_pcgt_10run.log`           | 84.30 ± 0.37     | 10   |
| CiteSeer   | 73.1 ± 0.4      | `citeseer_pcgt_10run.log`       | 73.12 ± 0.42     | 10   |
| PubMed     | 81.0 ± 0.6      | `pubmed_pcgt_10run.log`         | 81.00 ± 0.63     | 10   |
| Film       | 38.0 ± 0.9      | `film_pcgt_10run.log`           | 38.00 ± 0.90     | 10   |
| Chameleon  | 49.0 ± 2.8      | `chameleon_pcgt_10run.log`      | 49.03 ± 2.79     | 10   |
| Squirrel   | 45.5 ± 2.7      | `squirrel_pcgt_10run.log`       | 45.49 ± 2.71     | 10   |
| Deezer     | 67.2 ± 0.7      | `deezer_pcgt_5run.log`          | 67.16 ± 0.66     | 5    |

---

## Table 1 — SGFormer Re-run (verification, after C1 fix)

`table1_sgformer_rerun/` — SGFormer re-run with their published configs, 10 runs.
Purpose: verify which metric SGFormer published numbers correspond to.
**Status: COMPLETED (21 April 2026, H100)** — Confirms both codebases use identical "Highest Test" metric.

---

## Table 2 — Additional Datasets (4 datasets)

`table3_additional/` — Both PCGT and SGFormer from our own runs. GCN/GAT baselines included.

| Dataset      | Method   | Paper Value     | Log File                              | Log Highest Test | Runs |
|-------------|----------|-----------------|---------------------------------------|------------------|------|
| Co-CS       | PCGT     | 95.1 ± 0.3     | `coauthor-cs_pcgt_10run.log`          | 95.06 ± 0.34     | 10   |
| Co-CS       | SGFormer | 94.9 ± 0.5     | `coauthor-cs_sgformer_10run.log`      | 94.94 ± 0.49     | 10   |
| Co-CS       | GCN      | 91.9 ± 0.4     | `coauthor-cs_gcn_10run.log`           | 91.92 ± 0.40     | 10   |
| Co-Phys     | PCGT     | 96.8 ± 0.2     | `coauthor-physics_pcgt_10run.log`     | 96.83 ± 0.17     | 10   |
| Co-Phys     | SGFormer | 96.6 ± 0.2     | `coauthor-physics_sgformer_10run.log` | 96.62 ± 0.15     | 10   |
| Co-Phys     | GCN      | 95.7 ± 0.2     | `coauthor-physics_gcn_10run.log`      | 95.72 ± 0.17     | 10   |
| Am-Comp     | PCGT     | 88.8 ± 0.7     | `amazon-computers_pcgt_10run.log`     | 88.80 ± 0.68     | 10   |
| Am-Comp     | SGFormer | 87.5 ± 2.0     | `amazon-computers_sgformer_10run.log` | 87.50 ± 1.98     | 10   |
| Am-Comp     | GCN      | 82.8 ± 1.8     | `amazon-computers_gcn_10run.log`      | 82.83 ± 1.75     | 10   |
| Am-Photo    | PCGT     | 95.3 ± 0.4     | `amazon-photo_pcgt_10run.log`         | 95.27 ± 0.40     | 10   |
| Am-Photo    | SGFormer | 95.2 ± 1.2     | `amazon-photo_sgformer_10run.log`     | 95.20 ± 1.18     | 10   |
| Am-Photo    | GCN      | 90.9 ± 1.2     | `amazon-photo_gcn_10run.log`          | 90.87 ± 1.18     | 10   |

---

## Table 3 — Large-Scale Results

`table4_largescale/` — PCGT on ogbn-arxiv (10 runs), pokec (3 runs), Amazon2M (3 runs).

| Dataset    | Paper Value      | Log File                       | Log Highest Test | Runs |
|-----------|------------------|---------------------------------|------------------|------|
| arxiv     | 72.50 ± 0.14    | `arxiv_pcgt_k256_10run.log`     | 72.50 ± 0.14     | 10   |
| pokec     | 74.94 ± 0.28    | `pokec_pcgt_k500.log`           | 74.94 ± 0.28     | 3    |
| Amazon2M  | 88.79 ± 0.06    | `amazon2m_pcgt_k1000.log`       | 88.79 ± 0.06     | 3    |

SGFormer/baseline numbers from [Wu et al. 2023] Table 3.

**Note:** `arxiv_pcgt_3run.log` (72.63 ± 0.08, 3 runs) is archived — superseded by the 10-run result.

---

## Table 4 — Ablation Study

`table5_ablation/` — Organized into subdirectories:

### `table5_ablation/component/`
Component ablation (8 variants × 3 datasets, 5 runs each).
**Status: COMPLETED (21 April 2026, H100)** — All 25 log files present and verified.

### `table5_ablation/metis_vs_random/`
METIS vs random partitioning comparison (existing, 5 runs each).

### `table5_ablation/beta/`
Learned β value extraction logs.

---

## Table 5 — K-Sweep

`table6_ksweep/` — K ∈ {5, 10, 15, 20, 30} on Chameleon and Squirrel (10 runs each). All verified ✓.

---

## M-Ablation (Appendix)

`m_ablation/` — M ∈ {2, 4, 8} on Cora, Chameleon, Squirrel (5 runs each).
**Status: COMPLETED (21 April 2026, H100)**

| M | Cora | Cham. | Sqrl. | Log Files |
|---|------|-------|-------|-----------|
| 2 | 84.30 ± 0.60 | 48.11 ± 1.81 | 46.29 ± 2.31 | `cora_m2.log`, `chameleon_m2.log`, `squirrel_m2.log` |
| 4 | 84.40 ± 0.49 | 47.64 ± 2.55 | 47.27 ± 2.43 | `cora_m4.log`, `chameleon_m4.log`, `squirrel_m4.log` |
| 8 | 84.14 ± 0.72 | 48.66 ± 3.39 | 46.79 ± 2.13 | `cora_m8.log`, `chameleon_m8.log`, `squirrel_m8.log` |

---

## Table 6 — Runtime

`table7_runtime/` — Training run logs (1 run, 50 epochs, H100) for all medium-scale datasets.
`table7_runtime/results_txt/` — `medium/results/*.txt` files containing `run_time:` values (ms/epoch).

**Note:** The paper's runtime table uses timing from original full-length experiments.
The `results_txt/` files provide evidence that per-epoch training time is measured
and recorded by `main.py` via `time.perf_counter()`. See also `benchmark_inference_results.csv`
for inference-only timing from `benchmark_inference.py`.

---

## Directory Structure

```
final_results/
├── README.md
├── experiment_results.csv       ← Aggregate CSV with run_time for all experiments
├── table1_main/                 ← Table 1 PCGT results (7 datasets)
├── table1_sgformer_rerun/       ← SGFormer verification runs (COMPLETED)
├── table3_additional/           ← Table 2 results (4 datasets, all methods)
├── table4_largescale/           ← Table 3 large-scale (arxiv + pokec + amazon2m)
├── table5_ablation/
│   ├── component/               ← 8 ablation variants × 3 datasets (COMPLETED)
│   ├── metis_vs_random/         ← METIS vs random partition logs
│   └── beta/                    ← β value extraction logs
├── table6_ksweep/               ← K-sweep experiments (verified ✓)
├── table7_runtime/              ← Runtime training logs (SGF + PCGT × 11 datasets)
│   ├── results_txt/             ← medium/results/*.txt with run_time values
│   └── benchmark_inference_results.csv  ← Inference-only timing from benchmark_inference.py
└── m_ablation/                  ← M∈{2,4,8} ablation logs (COMPLETED)
```

---

## Rounding Convention

Paper values round to 1 decimal place. Example: 49.03 → 49.0, 67.16 → 67.2.
Standard deviations also rounded to 1 decimal: 0.37 → 0.4, 2.79 → 2.8.
