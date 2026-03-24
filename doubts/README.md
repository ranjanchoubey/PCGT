# PCGT Results Validation Tracker

## Priority Legend
- 🔴 **HIGH**: Major discrepancy between paper claim and validated run, or only 1-run evidence
- 🟡 **MEDIUM**: Small gap or config uncertainty, needs fresh multi-run validation
- 🟢 **LOW/CONFIRMED**: Already validated with ≥5 runs, results match expectations

---

## Summary Table

| # | Dataset | Paper (old) | Best Validated | run.sh Config Match? | Priority | Status |
|---|---------|-------------|----------------|---------------------|----------|--------|
| 1 | Cora | 83.8±1.2 | **84.56±0.52** (5r) | ✅ K=7 gw=0.8 | 🟢 | Beats SGFormer 84.5 |
| 2 | CiteSeer | 73.4±0.2 | 70.56±1.20 (5r) | ❌ Different config | 🔴 | See [citeseer.md](citeseer.md) |
| 3 | PubMed | 80.5±0.6 | 79.28±0.70 (5r) | ❌ Different config | 🔴 | See [pubmed.md](pubmed.md) |
| 4 | Chameleon | 48.1±2.4 | **49.06±2.97** (10r) | ✅ K=10 gw=0.8 | 🟡 | Need GPU revalidation |
| 5 | Squirrel | 45.1±2.3 | **45.28±2.08** (10r) | ✅ K=10 gw=0.8 | 🟡 | Need GPU revalidation |
| 6 | Film | 37.7±1.0 | ~32% (8/10r running) | ❓ New config | 🔴 | See [film.md](film.md) |
| 7 | Deezer | 67.2±0.5 | 66.58±0.61 (5r sweep) | ~✅ K=20 gw=0.7 | 🟡 | See [deezer.md](deezer.md) |
| 8 | Coauthor-CS | — | **95.06±0.34** (10r) | ✅ | 🟢 | New dataset |
| 9 | Coauthor-Phys | — | **96.83±0.17** (10r) | ✅ | 🟢 | New dataset |
| 10 | Amazon-Comp | — | **88.80±0.68** (10r) | ✅ | 🟢 | New dataset |
| 11 | Amazon-Photo | — | **95.27±0.40** (10r) | ✅ | 🟢 | New dataset |

---

## H100 Validation Plan

**Phase 1 — HIGH priority (run first):**
1. CiteSeer: Try layers=2 config (5+ runs) → see [citeseer.md](citeseer.md)
2. PubMed: Try layers=2/K=50 config (5+ runs) → see [pubmed.md](pubmed.md)
3. Film: HP sweep needed → see [film.md](film.md)

**Phase 2 — MEDIUM priority (revalidate):**
4. Chameleon: Rerun run.sh config, 10 runs on GPU
5. Squirrel: Rerun run.sh config, 10 runs on GPU
6. Deezer: Rerun run.sh config, 5 runs on GPU

**Phase 3 — CONFIRM (quick revalidation):**
7. Cora: Rerun 10 runs on GPU (currently 5 runs)

**Phase 4 — Already done:**
8-11. Coauthor/Amazon: Already 10-run validated ✅

---

## SGFormer Baseline Note

For the original 7 datasets, the paper uses SGFormer paper (Table 2) numbers as baselines.
Our own SGFormer runs differ from their reported numbers:

| Dataset | SGFormer Paper | Our SGFormer Run | Gap |
|---------|---------------|-----------------|-----|
| Cora | 84.5±0.8 | 83.87±0.61 | -0.6 |
| CiteSeer | 72.6±0.2 | 69.10±0.24 | -3.5 |
| PubMed | 80.3±0.6 | 79.48±0.26 | -0.8 |
| Chameleon | 44.9±3.9 | 48.34±2.64 | +3.4 |
| Squirrel | 41.8±2.2 | 44.81±2.17 | +3.0 |
| Film | 37.9±1.1 | — | — |
| Deezer | 67.1±1.1 | 66.16 (1r) | -0.9 |

**Decision**: Use SGFormer paper numbers for SGFormer row (standard practice).
For PCGT row, use our own validated numbers.

For Coauthor/Amazon (no SGFormer paper numbers), use our own SGFormer runs.
