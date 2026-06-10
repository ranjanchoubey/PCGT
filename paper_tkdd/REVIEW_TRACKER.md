# PCGT TKDD Paper — Review Tracker

**Paper:** `paper_tkdd/main.tex`  
**Last updated:** 2026-06-08  
**Current score:** R1: 5.5→6.5 | R2: 6.5 | R3: 7.0 (Minor Revision)

---

## FIXED (Text-only — already applied)

| ID | Issue | Source | Resolution |
|----|-------|--------|------------|
| F1 | Proposition 1 tautological | R1-W1 | Demoted to Remark 1 |
| F2 | "11 of 14" overclaiming | R1-W2 | Replaced with "statistically significant gains on heterophilic graphs" |
| F3 | Highest test metric caveat | R1-W3 | Added caveat in §4.2 acknowledging limitation |
| F4 | Complexity vs O(N²) strawman | R1-W4, R2-m1 | Now leads with SGFormer comparison as "practically relevant" |
| F5 | Nyström section ornamental | R1-W5, R2-m4 | Condensed to 1 paragraph |
| F6 | Baselines discussed in text | R1-W6 | Added GOAT, Xing et al., Exphormer exclusion rationale |
| F7 | GCN dominance addressed | R1-W7, R2-M2 | Quantified in Discussion (attention = +4.8 on Squirrel, 0 on Cora) |
| F8 | Inductive extension discussed | R1-W8, R2-m8 | Concrete nearest-partition algorithm in §8 |
| F9 | Rhetorical question informal | R1-§8.1 | Rephrased to declarative |
| F10 | Balanced partitions assumed | R1-§8.2 | Added "approximately balanced" |
| F11 | Metric description unclear | R1-§8.3 | Clarified per-run recording |
| F12 | Table 1 self-serving caption | R1-§8.4, R2-m7 | Softened framing |
| F13 | Pearson n=11 caveat | R1-§8.6, R2-m5 | Added statistical power caveat |
| F14 | Broader impact generic | R1-§8.7 | Merged into Limitations point (6) |
| F15 | PSE vs PE terminology | R1-§8.9 | Distinction explained at first mention |
| F16 | Ablation caption confusing | R1-§7.1, R3-W2 | Explains shared config, why numbers differ |
| F17 | Co-CS rounding inconsistency | R1-§7.2, R3-W6 | Standardized to 1 decimal in significance table |
| F18 | GCN runtime "---" on arxiv | R1-§7.5, R2-m2 | †footnote: mini-batch not comparable |
| F19 | `learn_graph_weight` unused | R2-m6 | Mentioned in §4.4: "preliminary experiments showed no improvement" |
| F20 | GraphGPS Performer variant weak | R3-W7 | Noted: "selected for scalability; full-attention doesn't scale" |
| F21 | Lim et al. 2021 wrong authors | R3-bib | Fixed: Bhalerao, Ser Nam Lim |
| F22 | GOAT wrong year | Self-caught | Fixed: 2023 not 2024 |
| F23 | CoBFormer name fabricated | Self-caught | Removed; cite as "Xing et al." |
| F24 | Xing et al. wrong authors/venue | Self-caught | Fixed: correct authors, arXiv not ICML |
| F25 | Remark listed as contribution | R3-W3 | Softened to "empirical analysis + gradient characterization" |
| F26 | PubMed K=50 cost-benefit | R3-W5 | Suggests K=20, references K-sweep |
| F27 | Exphormer exclusion misleading | R3-N1 | "Does not report results on our datasets" |
| F28 | GOAT not explained why no experiment | R3-O2 | "Inductive sampling regime, different protocol" |
| F29 | GCN contribution not sharp enough | R3-O3 | "Entire margin attributable to attention branch" |
| F30 | Random partition gap numbers wrong | Cross-check | Fixed: -1.3 Squirrel, +0.07 Chameleon (was -0.68/-0.26) |
| F31 | p-value mismatch Discussion vs Table | Cross-check | Fixed: Welch's t-test p=0.010 (was paired t-test p=0.004) |

---

## OPEN — Needs Experiments

| ID | Issue | Source | Priority | Effort | Notes |
|----|-------|--------|----------|--------|-------|
| O2 | Simpler-branch ablation (MLP replacing attention) | R2-W1, R3-W1 | **MEDIUM** | 1-2 days | Shows if partition attention specifically needed or any regularizer works. Run on Cora/Chameleon/Squirrel. |
| O3 | METIS preprocessing time for pokec/Amazon2M | R3-W5 | **LOW** | 5 min | Run METIS once, record time. Not in codebase currently. |
| O4 | Small inductive experiment | R1-W8, R3-N3 | **LOW** | 1 day | Hide 20% of Cora nodes from METIS, assign post-hoc, report accuracy. |
| O5 | Attention visualization heatmap | R1-§6.3, R3-W3 | **LOW** | 1 day | Extract cross-attention weights, show which representatives nodes attend to. |
| O6 | Running example (10 nodes, 3 partitions) | R1-§6.1 | **LOW** | 1 day | Trace through pipeline with actual numbers from trained model. |

---

## OPEN — Acceptable as-is (won't fix unless reviewer insists)

| ID | Issue | Source | Why acceptable |
|----|-------|--------|----------------|
| A1 | No Exphormer experiment | R1-W6, R2-M3 | Different task (graph-level), different protocol. Explained in text. |
| A2 | No graph-level task evaluation | R1-§6.5 | Out of scope — paper is about node classification. |
| A3 | Thin theory for journal | R3-W10 | Empirical paper. Remark is appropriate. Would need real theorem = major research. |
| A4 | Table 3 baseline sourcing asymmetry | R1-§7.3 | Already noted in caption. Standard practice. |
| A5 | Table 5 missing GAT/DIFFormer on large-scale | R1-§7.4 | Baselines from SGFormer's published table — can't add what they didn't report. |
| A6 | No 2024+ experimental baseline | R2-M3 | GOAT uses different protocol (inductive/sampling). Explained in text. |
| A7 | Report Final Test alongside Highest Test | R1-W3, R2-M1, R3-O1 | Same metric as all baselines (identical logger.py). Caveat already in §4.2. Would make comparison unfair if only PCGT changes metric. |

---

## Workflow

1. Pick an item from OPEN
2. Do the work (experiment / text edit)
3. Move to FIXED with resolution notes
4. Recompile and verify
5. Update "Last updated" date
