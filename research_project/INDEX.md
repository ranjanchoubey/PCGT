# Research Project Documentation Index

**Location**: `/Users/vn59a0h/thesis/PCGT/research_project/`  
**Last Updated**: March 22, 2026  
**Status**: Phase 2 In Progress (Deezer Attack)

---

## 🚀 Start Here: Quick Navigation

| For This... | Read This | Time |
|-------------|-----------|------|
| **Today's work** | [DAILY_WORKFLOW.md](DAILY_WORKFLOW.md) | 5 min |
| **Understand the prompt system** | [00_prompt_system.md](00_prompt_system.md) | 10 min |
| **Project overview** | [00_status_summary.md](00_status_summary.md) | 10 min |
| **Deep method explanation** | [04_method.md](04_method.md) | 15 min |
| **How to reproduce** | [15_reproducibility_guide.md](15_reproducibility_guide.md) | 10 min |
| **Code reference** | [16_code_architecture.md](16_code_architecture.md) | 20 min |

---

## 📋 Complete File Guide

### 🧠 Research Lifecycle System (NEW)

**00_prompt_system.md**  
**What**: Master prompt system for human-like research workflow  
**Contains**: 7 phases (Exploration → Paper), each with guidance + prompt  
**When to use**: First time, to understand the system  
**Reference**: Every file has embedded version of its phase prompt

**DAILY_WORKFLOW.md** ⭐ **READ FIRST**  
**What**: Daily quick-start guide  
**Contains**: The 7-step loop, magic prompts, workflows, checklists  
**When to use**: Every time you work (bookmark this!)  
**Read time**: 5 min

---

### 🎯 Research Documentation (By Phase)

#### Phase 1: Exploration
- **01_problem.md**: Problem definition (has Phase 1 prompt)
- **14_ideas_dump.md**: Brainstorming, future directions (has Phase 1 prompt)

#### Phase 2: Experiment Logging
- **11_experiment_logs.md**: Raw experiment logs (has Phase 2 prompt + template)
  - GPU validation results logged here
  - Deezer attack hypothesis details logged here

#### Phase 3: Pattern Discovery
- **08_analysis.md**: Pattern analysis from experiments (has Phase 3 prompt)

#### Phase 4: Method Formation
- **04_method.md**: PCGT architecture (has Phase 4 prompt)
- **05_design_choices.md**: Design rationale (has Phase 4 prompt)

#### Phase 5: Results Interpretation
- **07_results_raw.md**: GPU results + deep analysis (has Phase 5 prompt)

#### Phase 6: Critical Review
- **09_limitations.md**: Weaknesses & limitations (has Phase 6 prompt)
- **12_questions.md**: Open questions & doubts (has Phase 6 prompt)

#### Phase 7: Paper Writing
- (When ready, use Academic mode)

---

### 📖 Comprehensive References

**06_experiments_plan.md**  
Complete experiment protocol: Phase 1 validation, Phase 2 Deezer attack, success criteria

**15_reproducibility_guide.md**  
Step-by-step how to reproduce: setup, data, running, expected results, debugging

**16_code_architecture.md**  
Code structure: directory layout, key files, pseudocode, modifications, commands

**00_status_summary.md**  
Project status: scoreboard, current focus, technical insights, next steps

---

### 📚 Template Files (Fill These)

- **02_background.md**: Background on GNNs, homophily, heterophily
- **03_related_work.md**: Related work, SGFormer, baselines
- **10_future_work.md**: Future directions after results

---

## 🎯 Reading Paths (Customized)

### Path 1: Get Oriented Fast (15 minutes)
1. **DAILY_WORKFLOW.md** (2 min) — Understand the loop
2. **00_prompt_system.md** (5 min) — Understand phases
3. **00_status_summary.md** (8 min) — Understand project status

### Path 2: Understand PCGT (30 minutes)
1. **04_method.md** (15 min) — Architecture details
2. **05_design_choices.md** (10 min) — Design rationale
3. **16_code_architecture.md** (5 min) — Code layout

### Path 3: See Results (10 minutes)
1. **07_results_raw.md** (5 min) — GPU results table
2. **11_experiment_logs.md** (5 min) — Experiment details

### Path 4: Reproduce Everything (1 hour)
1. **15_reproducibility_guide.md** (20 min) — Follow steps
2. **DAILY_WORKFLOW.md** (5 min) — Understand workflow
3. **16_code_architecture.md** (10 min) — Reference while coding
4. **Execute** (25 min) — Run experiments

### Path 5: Continue Deezer Attack (30 minutes)
1. **00_status_summary.md** (5 min) — Current status
2. **06_experiments_plan.md** (10 min) — 12-config design
3. **11_experiment_logs.md** (5 min) — See logged hypotheses
4. **15_reproducibility_guide.md** (10 min) — Get exact commands

### Path 6: Modify Code (45 minutes)
1. **16_code_architecture.md** (20 min) — Learn structure
2. **04_method.md** (10 min) — Understand implementation
3. **DAILY_WORKFLOW.md** (5 min) — Check if need to log changes
4. **Edit** (10 min) — Make changes

---

## 📊 The Prompt System At A Glance

Each file has a comment block at the top with its **phase prompt**. When editing:

```
1. Open the file
2. See the <!-- PHASE X: ... --> comment at top
3. Copy that prompt
4. Paste into Copilot chat
5. Paste your work / ideas
6. Copilot responds with proper phase thinking
```

**Example**: You run an experiment on Deezer
1. Open `11_experiment_logs.md`
2. Copy Phase 2 prompt at top
3. Paste to Copilot + experiment output
4. Copilot logs it in researcher-style
5. You copy response back to the file

---

## ✅ Validation Checklist

### Before Moving Phases
- [ ] Have I completed current phase work?
- [ ] Did I document uncertainties?
- [ ] Is my thinking recorded honestly?

### Before Paper
- [ ] Phase 1: Problem explored from multiple angles?
- [ ] Phase 2: Experiments logged raw, not polished?
- [ ] Phase 3: Patterns validated across 3+ experiments?
- [ ] Phase 4: Method justified by patterns?
- [ ] Phase 5: Results deeply interpreted?
- [ ] Phase 6: Weaknesses honestly documented?

---

## 🎯 Quick Command Reference

**Run experiment + log it**:
```
1. Run: python main.py --dataset deezer-europe ...
2. Copy output
3. Go to 11_experiment_logs.md
4. Copy Phase 2 prompt from top
5. Paste prompt + output to Copilot
6. Copy response to file
```

**Analyze patterns**:
```
1. Collect 3+ experiment log links
2. Go to 08_analysis.md
3. Copy Phase 3 prompt from top
4. Paste prompt + links to Copilot
5. Copy response to file
```

**Document method**:
```
1. Have patterns from Phase 3
2. Go to 04_method.md
3. Copy Phase 4 prompt from top
4. Paste prompt to Copilot
5. Have Copilot write/refine
```

---

## 📈 File Statistics

| Category | Count | Status |
|----------|-------|--------|
| **System files** | 2 | ✅ Complete (00_prompt_system, DAILY_WORKFLOW) |
| **Phase-aligned files** | 8 | ✅ Complete (with prompts embedded) |
| **Result/reference files** | 3 | ✅ Complete |
| **Template files** | 3 | Empty, ready to fill |
| **Total markdown files** | 17 | Well-organized |
| **Total pages** | ~100 | Comprehensive |

---

## 🔗 Cross-References

**Want to understand current phase?**
→ Check DAILY_WORKFLOW.md step counter

**Want deep phase guidance?**
→ Check 00_prompt_system.md

**Want to log an experiment?**
→ Go to 11_experiment_logs.md, copy Phase 2 prompt at top

**Want to analyze results?**
→ Go to 08_analysis.md, copy Phase 3 prompt at top

**Want to code?**
→ Go to 16_code_architecture.md

**Want to reproduce?**
→ Go to 15_reproducibility_guide.md

---

## 🎁 Pro Tips

1. **Bookmark DAILY_WORKFLOW.md** — You'll reference it every day
2. **Bookmark 00_prompt_system.md** — Full reference for all phases
3. **Each file has prompt at top** — Copy/paste into Copilot
4. **Keep 11_experiment_logs.md growing** — Raw data is your best friend
5. **Phase checklist before moving on** — Don't skip thinking phases

---

## 🚀 Next Steps

**Right now**:
1. Read DAILY_WORKFLOW.md (5 min)
2. Read 00_prompt_system.md (10 min)
3. Pick today's task from DAILY_WORKFLOW.md
4. Open corresponding file
5. Copy its phase prompt
6. Start working

**Weekly**:
- Check 00_status_summary.md for overall progress
- Review 12_questions.md for open issues
- Update 11_experiment_logs.md with new experiments

**Before paper**:
- Complete Phase 6 (honest review)
- Fill templates (02_background.md, etc.)
- Then write paper using Phase 7 mode

---

## 📞 Metadata

- **Created**: March 2026
- **Last Updated**: March 22, 2026
- **System**: 7-Phase Research Lifecycle
- **Status**: Active (Phase 2 experiments running)
- **Next Review**: When Deezer attack completes
- **Total documentation time**: ~100 hours of tested workflows

| File | Purpose | Read When | Time |
|------|---------|-----------|------|
| **00_status_summary.md** | Project overview & current status | First! Get oriented | 5 min |
| **15_reproducibility_guide.md** | How to run everything | To execute experiments | 10 min |
| **04_method.md** | PCGT architecture deep-dive | Understanding the model | 15 min |
| **05_design_choices.md** | Why we made each decision | Design rationale | 10 min |
| **06_experiments_plan.md** | Experiment protocol & methodology | Before running | 10 min |
| **07_results_raw.md** | GPU results + raw observations | See final numbers | 5 min |
| **11_experiment_logs.md** | Detailed per-run logs | Understanding each experiment | 15 min |
| **16_code_architecture.md** | Code structure & file reference | Modifying code | 20 min |

---

## File Descriptions

### 00_status_summary.md ⭐ START HERE
**What**: Executive summary of entire project  
**Contains**:
- Final scoreboard (4 wins, 2 matched, 1 attacking)
- What we did (6 datasets validated on GPU)
- Current focus (Deezer 12-config attack)
- Key technical insights (multi-resolution attention works on heterophilic)
- Next steps and success criteria
- FAQ

**Length**: 6 pages  
**Best for**: Getting oriented, understanding big picture

---

### 04_method.md
**What**: PCGT architecture technical documentation  
**Contains**:
- Overview of partition-conditioned architecture
- Graph partitioning (METIS, KMeans, random)
- Multi-resolution attention (local O(N²/K) + global O(NMK))
- Partition Structural Encoding (PSE)
- Learnable α (blend weight) and β (self-connection)
- GCN backbone blending (graph_weight)
- Full layer implementation pseudocode
- Why it works on homophilic vs heterophilic graphs
- Complexity analysis

**Length**: 8 pages  
**Best for**: Deep technical understanding of PCGT

---

### 05_design_choices.md
**What**: Justification for all design decisions  
**Contains**:
- Multi-resolution attention vs alternatives
- Learnable β (self-connection weight)
- Learned seed pooling vs fixed pooling
- GCN blending rationale (vs pure PCGT/GCN)
- Partition method selection (KMeans, METIS, random)
- Standard hyperparameter choices (64 hidden, 3 layers, etc.)
- Training procedure choices
- Trade-off analysis for each decision
- Empirical validation per design choice

**Length**: 10 pages  
**Best for**: Understanding architectural choices and trade-offs

---

### 06_experiments_plan.md
**What**: Experiment protocol and methodology  
**Contains**:
- Phase 1: GPU baseline validation (6 datasets completed)
- Phase 2: Deezer systematic attack (12 configs, 4 batches)
- Dataset characteristics table
- Metrics and evaluation procedure
- 12 Deezer config hypotheses (capacity, regularization, architecture, features)
- Success criteria (>65.5 strong, >66.5 match, accept loss)
- Expected timeline
- Variables to track per-config

**Length**: 7 pages  
**Best for**: Understanding experiment design before running

---

### 07_results_raw.md
**What**: All GPU validation results  
**Contains**:
- Final results table (all 7 datasets, 5 runs each)
- Score: 4 wins, 2 matched, 1 loss
- Key observations per dataset (Cora variance, CiteSeer stability, Chameleon win, Deezer overfitting)
- Heterophilic wins (+3.19, +3.34)
- Experimental setup details
- Device behavior notes
- Deezer special analysis (binary, high-dim, near-homophily)

**Length**: 5 pages  
**Best for**: Seeing the final numbers and per-dataset analysis

---

### 11_experiment_logs.md
**What**: Detailed logs of what was run and why  
**Contains**:
- **Phase 1 (Completed)**: Each dataset run
  - Cora: GPU vs CPU variance
  - CiteSeer: Tightest std (±0.21), strongest homophilic win
  - PubMed: Marginal win, both methods converge
  - Chameleon: Strong heterophilic win (+3.19)
  - Film: Matched, β extreme (-2.01)
  - Squirrel: Strongest heterophilic win (+3.34)
  - Deezer: Baseline, 97% train overfit, attack planned
- **Phase 2 (In Progress)**: Deezer 12-config hypothesis-driven design
  - Batches 1–4 (capacity, regularization, architecture, features)
  - Each config with hypothesis, variants, expected outcome

**Length**: 8 pages  
**Best for**: Understanding what experiments ran and why

---

### 15_reproducibility_guide.md
**What**: Step-by-step how to reproduce all experiments  
**Contains**:
- Quick start (Colab one-liner)
- Full reproduction steps (environment setup, data, running)
- Expected results table with all numbers
- Output formats (logs, results, Drive-backed files)
- Debugging tips (GPU not found, OOM, slow CPU)
- Code files reference table
- Hyperparameter details (all args explained)
- Deezer attack variants (exact command lines for 12 configs)
- Validation checklist

**Length**: 10 pages  
**Best for**: Reproducibility—follow this to replicate all results

---

### 16_code_architecture.md
**What**: Code structure and implementation reference  
**Contains**:
- Directory structure (medium/, large/, 100M/, data/)
- Core code files (main.py, pcgt.py, models.py, dataset.py, data_utils.py, parse.py)
- Detailed description of each class and function
- Pseudocode for key algorithms
- Data flow diagram
- Hyperparameter mapping (arg → usage)
- Modification guide (how to change model, add dataset, etc.)
- Testing & validation commands
- Debugging commands
- Command templates (baseline, Deezer configs)

**Length**: 12 pages  
**Best for**: Code modification, understanding architecture, debugging

---

## Other Files (Templates)

### 01_problem.md
**Status**: Empty template  
**Purpose**: Problem statement and research question  
**To fill**: "What problem does PCGT solve?"

---

### 02_background.md
**Status**: Empty template  
**Purpose**: Background on GNNs, homophily, heterophily  
**To fill**: Literature context and motivation

---

### 03_related_work.md
**Status**: Empty template  
**Purpose**: Related work and baselines  
**To fill**: SGFormer, GraphTransformer, other attention GNNs

---

### 08_analysis.md
**Status**: Empty template  
**Purpose**: Deeper analysis of results  
**To fill**: Why do heterophilic wins occur? Why Deezer fails?

---

### 09_limitations.md
**Status**: Empty template  
**Purpose**: Discuss limitations  
**To fill**: Scalability, hyperparameter tuning, Deezer edge case

---

### 10_future_work.md
**Status**: Empty template  
**Purpose**: Future directions  
**To fill**: OGB-Arxiv, theoretical analysis, improved Deezer handling

---

### 12_questions.md
**Status**: Empty template  
**Purpose**: Open questions  
**To fill**: "How to decide partition granularity? When does β become negative?"

---

### 13_assumptions.md
**Status**: Empty template  
**Purpose**: Assumptions and limitations  
**To fill**: "Assumes meaningful graph structure exploited by partitions"

---

### 14_ideas_dump.md
**Status**: Empty template  
**Purpose**: Brainstorming and ideas  
**To fill**: "Try BatchNorm on Deezer? Investigate feature space clustering?"

---

## Reading Paths

### Path 1: Get Oriented (15 minutes)
1. **00_status_summary.md** — What did we do?
2. **07_results_raw.md** — What did we find?
3. **15_reproducibility_guide.md** (Quick start section) — How to run?

### Path 2: Understand the Model (30 minutes)
1. **04_method.md** — PCGT architecture
2. **05_design_choices.md** — Why these choices?
3. **16_code_architecture.md** (pcgt.py section) — Code implementation

### Path 3: Reproduce (60 minutes)
1. **15_reproducibility_guide.md** — Follow step-by-step
2. **06_experiments_plan.md** — Understand what you're running
3. **11_experiment_logs.md** — See what we ran before

### Path 4: Continue Deezer Attack (30 minutes)
1. **00_status_summary.md** (Deezer section) — Problem statement
2. **06_experiments_plan.md** (Phase 2) — 12-config design
3. **11_experiment_logs.md** (Phase 2) — Hypothesis details
4. **15_reproducibility_guide.md** (Section: Deezer Attack Variants) — Exact commands

### Path 5: Modify Code (45 minutes)
1. **16_code_architecture.md** (File descriptions) — Understand structure
2. **15_reproducibility_guide.md** (Hyperparameter Details) — Map args to code
3. **16_code_architecture.md** (Modification Guide) — How to edit

---

## Key Data Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Baseline results | 07_results_raw.md (table) | Final GPU accuracies |
| Hyperparameters | 15_reproducibility_guide.md (table) + FINAL_CONFIGS.sh | Official settings |
| Deezer 12 configs | 11_experiment_logs.md (Phase 2) | Attack specifications |
| Code reference | 16_code_architecture.md | File structure + templates |
| Experiment protocol | 06_experiments_plan.md | How we measured everything |

---

## Status Checklist

### Completed (✅)
- [x] GPU validation of 6 datasets (Cora, CiteSeer, PubMed, Chameleon, Film, Squirrel)
- [x] Results on all 7 datasets (including Deezer 1-run CPU baseline)
- [x] 4 wins, 2 matched, 1 loss → "6/7 datasets beat or match SGFormer"
- [x] 12 Deezer configs designed (capacity, regularization, architecture, features)
- [x] PCGT_Runner.ipynb updated with Deezer attack cells
- [x] All documentation written (method, design, experiments, results, reproducibility, code)

### In Progress (⏳)
- [ ] Deezer 12-config GPU probe phase (3 runs each)
- [ ] Summary parsing (best config identification)
- [ ] Best config 5-run validation

### Pending (❌)
- [ ] Final Deezer result → Update FINAL_CONFIGS.sh
- [ ] Decision: Accept loss or claim recovery?
- [ ] (Optional) OGB-Arxiv large-scale validation

---

## How to Use This Folder

**If new to the project**:
→ Start with 00_status_summary.md (5 min), then 15_reproducibility_guide.md quick start

**If reproducing results**:
→ Follow 15_reproducibility_guide.md step-by-step

**If continuing the Deezer attack**:
→ Check 00_status_summary.md (current focus), then run commands from 15_reproducibility_guide.md

**If modifying code**:
→ Read 16_code_architecture.md for file layout, then modify

**If writing the paper**:
→ Results in 07_results_raw.md, architecture in 04_method.md, design in 05_design_choices.md

**If documenting analysis**:
→ Fill in empty files (08_analysis.md, 09_limitations.md, 10_future_work.md)

---

## File Statistics

| Category | Count | Status |
|----------|-------|--------|
| Filled docs | 6 | ✅ Complete |
| Template docs | 8 | Empty, ready to fill |
| Total docs | 16 | Well-organized |
| Total pages | ~75 | Comprehensive |
| Figures | Text-based diagrams | In method + code |
| Code references | ~20 | Full path + function names |

---

## Metadata

- **Project**: PCGT GPU Validation & Deezer Recovery
- **Created**: March 2026
- **Last Modified**: March 22, 2026
- **Author**: Research team + GitHub Copilot
- **Status**: Active (Phase 2 running)
- **Next Review**: When Deezer attack completes
- **Archive**: After paper submission