# PCGT → ACM TKDD Submission Plan

- **Journal:** ACM Transactions on Knowledge Discovery from Data (TKDD)
- **Journal Link:** https://dl.acm.org/journal/TKDD
- **Author Guidelines:** https://dl.acm.org/journal/tkdd/author-guidelines
- **Submission System:** https://mc.manuscriptcentral.com/tkdd
- **Editor-in-Chief:** Jian Pei
- **Review Type:** Double-anonymous peer review
- **Current paper:** `paper_new_format/PCGT_ACM_Letters/main.tex` (ACM sigconf format, 11 pages)

---

## 1. KEY JOURNAL REQUIREMENTS (from Author Guidelines)

### 1.1 Formatting
- [ ] Must use **ACM authoring template** (`acmart.cls`)
- [ ] For **submission/review**: use `\documentclass[manuscript, screen, review]{acmart}`
  - Current paper uses `\documentclass[sigconf]{acmart}` — **MUST CHANGE** for submission
- [ ] `manuscript` mode = double-spaced, single-column (easier for reviewers)
- [ ] Discouraged: papers >50 double-spaced pages (including figures + refs)
- [ ] Short papers (even 5 pages) are welcome — focus is on contribution, not length

### 1.2 Double-Anonymous Review — CRITICAL
- [ ] **Remove ALL author names and affiliations** from the manuscript
- [ ] **Remove funding sources** from the manuscript
- [ ] **Remove acknowledgments** of collaborators/group members
- [ ] **Anonymize file names and metadata** (no author-identifying info)
- [ ] **Neutralize self-references**: change "As we showed in [our work]" → "As shown by [Author et al.]"
- [ ] Still include all relevant own prior work in references (just refer in 3rd person)
- [ ] **Do NOT** upload new versions to arXiv during review (existing ones are OK)

### 1.3 Prior Publication Policy (Conference Extension)
- TKDD **explicitly welcomes** "major value-added extensions" of conference papers
- Must have **≥30% new content material** (not just proofs or extra perf figures)
- New material should offer **substantial new insights** (new alternatives, future work items resolved)
- Must submit a **separate document** explaining differences from prior publication
- Must upload prior publication as supplementary file
- **Action needed:** Write a clear "differences document" highlighting novel TKDD contributions

### 1.4 Submission Requirements
- [ ] All submitting authors need **ORCID** (https://orcid.org)
- [ ] Submit via **ScholarOne**: https://mc.manuscriptcentral.com/tkdd
- [ ] Include: descriptive title, abstract, CCS content indicators, relevant citations
- [ ] Cover letter (no identifying info inside it)

### 1.5 Open Access (as of Jan 2026)
- ACM is now fully Open Access
- Article Processing Charge (APC) applies — check https://dl.acm.org/journal/tkdd/open-access for rates, waivers, discounts

---

## 2. WHAT REVIEWERS EVALUATE (from Referee Guidelines)

TKDD reviewers score on **four pillars**. A substandard level in ANY ONE = grounds for rejection:

| Pillar | What they look for | PCGT paper status |
|--------|-------------------|-------------------|
| **1. Technical Quality** | Correct proofs, sound methodology, reproducible experiments | ✅ Strong — algorithm well-defined, ablations present |
| **2. Relevance** | Falls within data science / knowledge discovery / data mining scope | ✅ Directly relevant — graph learning for node classification |
| **3. Novelty & Interest** | Significant improvement over state-of-the-art, new insights | ✅ Novel partition-conditioned attention, β analysis |
| **4. Presentation** | Readable, well-structured, examples, accessible to broad audience | ⚠️ Needs work — journal papers demand deeper exposition than conf papers |

### Additional reviewer expectations:
- **Theory ↔ Systems fusion**: Theory papers must discuss applications/implementation; systems papers must reference theory. PCGT must do both.
- **Readability for broad audience**: Include examples, expanded discussions. Not just for GT specialists.
- **Concise but complete**: No unnecessary digressions; every section should earn its place.

---

## 3. HOW TO WRITE SO IT WON'T GET REJECTED

### 3.1 Top Rejection Reasons at TKDD (and how to avoid them)

| Rejection Reason | Prevention Strategy |
|-----------------|---------------------|
| **Incremental contribution** | Frame PCGT as solving a fundamental tradeoff (O(N²) vs topology-blind). Show the partition-conditioned mechanism is a new paradigm, not just another GT variant. |
| **Insufficient novelty over conf version** | Add ≥30% new content: deeper theoretical analysis, new datasets, comprehensive ablations, scalability study, reproducibility artifacts |
| **Poor presentation / readability** | Add running examples, intuitive diagrams, expanded related work. Write for data mining audience, not just GT specialists. |
| **Missing baselines** | Compare against latest 2024-2025 methods (not just 2022-2023 baselines). Add recent graph transformers. |
| **No theoretical grounding** | Add formal complexity analysis, convergence properties, or approximation bounds for partition attention |
| **Anonymity violation** | Triple-check: no names, no "our prior work", no identifying metadata |
| **Overclaiming** | Be precise about wins/losses. "11/14 best" is strong — don't stretch to "universally superior" |

### 3.2 Content Strategy: What to Add for Journal Version

#### A. Expanded Introduction (2-3 pages)
- [ ] Open with the REAL problem: "Why do graph transformers still struggle at scale?"
- [ ] Motivating example with a concrete graph (e.g., citation network partition)
- [ ] Clear research questions (RQ1, RQ2, RQ3)
- [ ] Contribution list — explicitly map each to a section

#### B. Deeper Related Work (2-3 pages)
- [ ] Structured taxonomy: (1) Full-graph Transformers, (2) Sparse/Linear Transformers, (3) Partition-based methods, (4) Hybrid GNN+Transformer
- [ ] Position PCGT in the taxonomy — what gap does it fill?
- [ ] Discuss 2024-2025 methods: NAGphormer, Exphormer, GOAT, Graph-MLP, etc.
- [ ] Add comparison table of methods (complexity, topology-awareness, scalability)

#### C. Enhanced Method Section (4-5 pages)
- [ ] Running example through the entire pipeline (pick one small graph, trace step by step)
- [ ] Formal complexity analysis: time + space for each component
- [ ] Theoretical proposition: why partition attention preserves community structure
- [ ] Discussion of design choices with ablation pointers
- [ ] Pseudocode as Algorithm environment (already have this — refine)

#### D. Expanded Experiments (6-8 pages)
- [ ] **New experiments to add:**
  - [ ] Sensitivity analysis: effect of partition count K across all datasets
  - [ ] Partition method comparison: METIS vs spectral vs random
  - [ ] Convergence analysis: training curves comparing PCGT vs baselines
  - [ ] Visualization: attention patterns, partition quality, t-SNE embeddings
  - [ ] Statistical significance: std dev, confidence intervals, paired t-tests
  - [ ] Memory + runtime profiling table
  - [ ] New dataset(s) if available (e.g., ogbn-papers100M results, or domain-specific graphs)
- [ ] **Improve existing tables:**
  - [ ] Ensure consistent formatting (ACM style references)
  - [ ] Add "Avg. Rank" column
  - [ ] Clearly separate homophilic vs heterophilic results

#### E. New Sections
- [ ] **Discussion Section** (1-2 pages): When does PCGT help most? When does it fail? Connection between β and homophily — deeper analysis.
- [ ] **Reproducibility Section**: Code availability, hyperparameter sensitivity, hardware specs
- [ ] **Limitations & Future Work** (1 page): Honest about boundary conditions

### 3.3 Writing Style for TKDD

1. **Lead with "why", not "what"**: Don't start sections with "In this section we..." — start with the problem the section solves
2. **Concrete before abstract**: Show a partition attention example, THEN generalize the formula
3. **One idea per paragraph**: Data mining reviewers skim — make each paragraph self-contained
4. **Active voice**: "PCGT computes..." not "The computation is performed by..."
5. **Quantify everything**: "significantly better" → "+4.1% on Chameleon (p < 0.01)"
6. **Address the "so what"**: After every result, explain WHY it matters for practitioners
7. **Cross-reference aggressively**: "As shown in Table 3 (Section 5.2)" — help the reviewer navigate

---

## 4. DOCUMENT CLASS CHANGE REQUIRED

Current:
```latex
\documentclass[sigconf]{acmart}
```

For TKDD submission:
```latex
\documentclass[manuscript, screen, review]{acmart}
```

For final accepted version:
```latex
\documentclass[acmsmall]{acmart}  % TKDD uses acmsmall format
```

Also update metadata:
```latex
\acmJournal{TKDD}
\acmVolume{0}
\acmNumber{0}
\acmArticle{0}
\acmYear{2026}
```

Remove the conference-specific commands:
```latex
% REMOVE these:
\acmConference[Preprint]{...}
\acmBooktitle{...}
\acmISBN{}
```

---

## 5. SUBMISSION CHECKLIST

### Pre-Submission
- [ ] Change document class to `manuscript, screen, review`
- [ ] Switch metadata from conference to journal (`\acmJournal{TKDD}`)
- [ ] Anonymize: remove author block, acknowledgments, funding
- [ ] Neutralize all self-references to 3rd person
- [ ] Write cover letter (anonymous, no identifying info)
- [ ] Write "differences from prior work" document (if extending conference paper)
- [ ] Get ORCID for all authors
- [ ] Run spell check + grammar check
- [ ] Verify all figures are vector (PDF) not raster
- [ ] Check all references are complete (no "arXiv preprint" if published)

### Content Quality
- [ ] ≥30% new material over any prior version
- [ ] Ablation for every design choice
- [ ] Statistical significance reported (std dev, # runs)
- [ ] Comparison with latest baselines (2024+)
- [ ] Complexity analysis (time + space)
- [ ] Running example through the method
- [ ] Discussion of failure cases / limitations

### Formatting
- [ ] ACM Reference Format bibliography
- [ ] CCS classification codes included
- [ ] Abstract ≤250 words
- [ ] Figures have descriptive captions
- [ ] Tables use `booktabs` style
- [ ] No orphan/widow lines
- [ ] Page count reasonable (<50 double-spaced pages)

### Submission
- [ ] Submit at https://mc.manuscriptcentral.com/tkdd
- [ ] Upload main manuscript (anonymized PDF)
- [ ] Upload supplementary: prior publication, differences doc, code/data artifacts
- [ ] Declare any conflicts of interest
- [ ] Provide ORCID for all authors
- [ ] Select appropriate Associate Editor if prompted

---

## 6. TIMELINE ESTIMATE

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Reformat to TKDD + anonymize | 1-2 days |
| 2 | Expand intro, related work, method | 1-2 weeks |
| 3 | Run new experiments (K sweep, partitioning ablation, convergence) | 1-2 weeks |
| 4 | Write discussion + limitations | 3-5 days |
| 5 | Internal review + polish | 1 week |
| 6 | Submit | — |
| 7 | Review period (expect ~2-4 months) | — |
| 8 | Revision (if R&R, common for TKDD) | 2-4 weeks |

---

## 7. WHAT MAKES PCGT A STRONG TKDD FIT

- **Scope match**: TKDD covers "knowledge discovery from data" — graph node classification is core
- **Theory + Systems**: PCGT has both algorithmic novelty AND practical scaling — exactly what TKDD wants
- **Scalability story**: pokec (1.6M nodes) results + ogbn-arxiv show practical value
- **Novel insight**: The β parameter going negative on heterophilic graphs is a publishable finding on its own
- **Comprehensive evaluation**: 14 datasets across homophilic/heterophilic/large-scale — demonstrates broad applicability

---

## 8. QUICK REFERENCE LINKS

| Resource | URL |
|----------|-----|
| TKDD Homepage | https://dl.acm.org/journal/TKDD |
| Author Guidelines | https://dl.acm.org/journal/tkdd/author-guidelines |
| Reviewer Guidelines | https://dl.acm.org/journal/tkdd/reviewers |
| Submission System | https://mc.manuscriptcentral.com/tkdd |
| ACM Templates | https://www.acm.org/publications/authors/submissions |
| Overleaf ACM Template | https://www.overleaf.com/gallery/tagged/acm-official |
| ACM CCS Classifier | http://dl.acm.org/ccs.cfm |
| Open Access / APC Info | https://dl.acm.org/journal/tkdd/open-access |
| ORCID Registration | https://orcid.org |
| LaTeX Support | acmtexsupport@aptaracorp.com |
| Journal Admin | tkdd-admin@acm.org |