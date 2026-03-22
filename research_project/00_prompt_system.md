# 🧠 Master Prompt System — Research Lifecycle Workflow

**Last Updated**: March 22, 2026  
**Purpose**: Guide both human researcher and Copilot through proper research thinking phases

---

## Overview

This system ensures that research moves from **messy exploration → insight → structure → paper**, rather than jumping to polished conclusions.

Each phase has:
- A **purpose** (what thinking should happen)
- A **prompt** (copy/paste to guide Copilot)
- A **corresponding file** (where to work)
- An **output** (what you should have at the end)

---

## 🧠 Core Research Principle

```
You are acting as a research assistant following how real researchers work.

Rules:
- Do NOT jump to conclusions
- Do NOT write like a polished paper unless asked
- Think step-by-step
- Prefer exploration over summarization
- Always include reasoning, assumptions, and uncertainties
- Highlight doubts and missing pieces

Write in a thinking + teaching style, not final academic style.
```

**Use this at the top of EVERY conversation** when working on research.

---

## 📋 The 7 Phases

### ⚡ Phase 1: Idea & Problem (Exploration Mode)

**File**: `01_problem.md` + `14_ideas_dump.md`  
**Duration**: Early stage (brainstorming)  
**Goal**: Expand messy ideas into structured questions

**Prompt to use**:
```
We are in early research phase for PCGT.

Expand the core problem:

- What exactly is the problem?
- Why does it matter?
- Where does it fail today?
- What are possible directions (multiple, not just one)?
- What are risks or unknowns?
- What are we assuming that might be wrong?

Do NOT finalize anything. Keep it exploratory.
Include: doubts, competing hypotheses, alternative framings.
```

**What you should produce**:
- Multiple angles on the problem (not one "correct" answer)
- Explicit assumptions listed
- Unknowns and risks documented
- Questions that need answering

---

### 🧪 Phase 2: Experiment Logging (THE MOST IMPORTANT PHASE)

**File**: `11_experiment_logs.md`  
**Duration**: Every single experiment  
**Goal**: Capture raw thinking, not smooth summaries

**Prompt to use** (use this EVERY TIME you run code):
```
Log this experiment like a researcher's lab notebook:

- What was attempted and why?
- What was your hypothesis or expectation?
- What actually happened? (include exact output)
- Key observations (what stood out?)
- Unexpected behaviors or surprises?
- Possible explanations or reasons?
- What this tells you about the problem?
- What should you try next?

Do NOT summarize. Do NOT minimize struggles.
Keep raw observations. Include failures and confusion.

Also include:
- What is unclear from this experiment?
- What needs verification?
- What might be wrong about my interpretation?
```

**What you should produce**:
- Detailed, messy experiment notes
- Raw outputs (don't clean them up)
- Honest observations including surprises
- Explicitly stated uncertainties
- Seeds for next experiments

**Example format**:
```
## Experiment: Deezer-GPU-Run-1
**Date**: March 22, 2026

**What was attempted**: 
Run standard PCGT baseline on Deezer with GPU...

**Expected**: 
Similar to CPU run (63.81), maybe 2-3% variance

**What happened**:
Test Acc: 63.94 ± 1.12 (first of 5 runs)

**Key observation**:
Still severe overfitting (train 97%→ test 64%)...

**Surprise**:
Early stopping triggered at epoch 11 vs epoch 50+ for other datasets

**Possible reasons**:
- Binary class imbalance makes learning different
- 31K dimensions in feature space cause different convergence...

**Unclear**:
- Is the partition structure actually useful on binary tasks?
- Does KMeans clustering in 31K-dim feature space even find meaningful partitions?

**Next step**:
Design systematic attack targeting overfitting...
```

---

### 🔍 Phase 3: Pattern Discovery (Analysis)

**File**: `08_analysis.md`  
**Duration**: After multiple experiments (3+ runs)  
**Goal**: Find **real patterns**, not just explain individual results

**Prompt to use**:
```
Analyze all experiments (collected from 11_experiment_logs.md):

- What patterns do you observe across experiments?
- What consistently works? What consistently fails?
- What assumptions seem incorrect?
- What new hypotheses emerge from the patterns?
- Where are the contradictions or surprises?
- What is counterintuitive?

Be critical. Do not try to make things look good.
Focus on: What is surprising? What breaks the model? Where does it fail?

Also include:
- What patterns are weak (need more data to confirm)?
- What assumptions need questioning?
- Where could you be wrong?
```

**What you should produce**:
- Patterns identified across 3+ experiments
- Contradictions highlighted
- New hypotheses formed
- Assumptions being tested listed
- Critical self-assessment

**Example**:
```
## Pattern Analysis: Heterophilic vs Homophilic

**Observation 1**: Chameleon (+3.19) and Squirrel (+3.34) show strong gains.
**Observation 2**: Cora and CiteSeer show modest/matched gains.

**Pattern**: Heterophilic datasets gain more than homophilic.

**New hypothesis**: Multi-resolution attention is specifically good for graphs where 
edges don't follow homophily assumption. Partition-based attention can capture 
non-neighbor relationships better than global softmax.

**But**: Why does Film (heterophilic, h=0.44) only match? 
- Different hypothesis needed: Not just homophily, but specific types of heterophily?
- Or: Feature dimensionality matters? Film has 932 dims vs Squirrel 2K?

**Critical doubt**: Maybe the gains are just from hyperparameter tuning variation?
Need to check: Are std deviations overlapping?
```

---

### 🧠 Phase 4: Method Formation (Formalization)

**File**: `04_method.md` + `05_design_choices.md`  
**Duration**: After patterns are clear  
**Goal**: Convert insights into structured, teachable method

**Prompt to use**:
```
Based on discovered patterns, define the method:

- What is the core insight?
- Step-by-step process (each step explains one finding)
- Why does each step exist? (connect to patterns you found)
- What problem does each step solve? (from failure analysis)
- Assumptions behind each step (list explicitly)
- Edge cases where it might fail

Explain like you're teaching someone who will build it from scratch.
Do NOT assume they know the motivation.

Also note:
- Which steps are most critical?
- Where are you least confident?
- What could break this method?
```

**What you should produce**:
- Clear step-by-step architecture description
- Rationale for each design choice
- Explicit assumptions per choice
- Edge cases and limitations

---

### 📊 Phase 5: Results Interpretation (Not Just Reporting)

**File**: `07_results_raw.md` (deep analysis section)  
**Duration**: After experiments complete  
**Goal**: Understand WHY results look like they do, not just report numbers

**Prompt to use**:
```
Interpret results deeply:

- Why did results look like this?
- Where does the method excel? Where does it fail?
- Are results consistent with your expectations? If not, why?
- What hidden factors might influence outcomes?
- What would someone misunderstand if you just showed them numbers?
- What do the error bars tell you?

Avoid surface-level explanation ("CiteSeer scored 73.44").
Go deeper ("CiteSeer's tightest std (±0.21) suggests stable learning on Planetoid splits").

Also include:
- What you're uncertain about in these results?
- What other interpretation could explain the data?
- What data would change your conclusion?
```

**What you should produce**:
- Deep analysis of each result
- Unexpected findings explained
- Connections to method design
- Honest uncertainty statements

---

### ⚠️ Phase 6: Critical Review (Simulate Harsh Reviewer)

**File**: `09_limitations.md` + `12_questions.md`  
**Duration**: Before writing paper  
**Goal**: Identify weaknesses BEFORE submission

**Prompt to use**:
```
Act like a harsh conference reviewer:

- What is unclear or poorly justified in this work?
- What assumptions are unjustified or questionable?
- What experiments are missing?
- What alternative explanations exist?
- Where is the evidence weak?
- What would make this paper rejectable?

Be brutally honest. Assume:
- You don't want to accept this paper
- You're looking for flaws
- You know similar work exists

Questions to answer:
- What are the 3 biggest weaknesses?
- What one experiment would change the verdict?
- Is the contribution actually novel?
- Are results reproducible with given details?
```

**What you should produce**:
- List of 3-5 major weaknesses
- Potential missing experiments
- Alternative interpretations of results
- Defense strategies (or acknowledgment of hard limits)

---

### ✍️ Phase 7: Paper Writing (ONLY AT THE END)

**File**: `Final paper document`  
**Duration**: After all phases complete  
**Goal**: Convert validated content into academic paper

**Prompt to use**:
```
Convert all validated content into research paper draft:

- Use academic tone
- Preserve all meaning but compress explanations
- Structure: Method → Experiments → Results → Discussion → Related Work → Intro → Abstract
- Each section should stand alone
- Assume reader has basic GNN knowledge but not your specific method

Order to write:
1. Method (most important, write first)
2. Experiments (what you tested and why)
3. Results (what you found)
4. Discussion (why you think it happened)
5. Related Work (how this fits in literature)
6. Introduction (motivation, written last with all context)
7. Abstract (summary, written absolute last)
```

**What you should produce**:
- 6-8 page research paper draft
- All claims connected to experiments
- Proper citations
- Professional technical writing

---

## 🔄 The Daily Workflow Loop

Don't memorize all 7 phases. Just follow this loop:

```
1. Plan experiment (Phase 1 thinking)
2. Run code
3. Paste output + observations into 11_experiment_logs.md
4. Use Phase 2 prompt above
5. (After 3-5 experiments) → Use Phase 3 prompt on 08_analysis.md
6. (After patterns clear) → Write 04_method.md with Phase 4 prompt
7. (Final stage) → Phase 5, 6, 7 in sequence
```

---

## 💡 Pro Tricks That Make Copilot Much Smarter

### Trick 1: End Every Output With Questions
At the end of ANY response, add:
```
Also tell me:
- What is still unclear?
- What needs verification?
- What might I be wrong about?
```

This prevents "confident nonsense."

### Trick 2: Use "But" and "However" Signals
When you spot something unexpected:
```
"We expected X, but got Y. Why would that be?"
```

This forces deeper thinking than just reporting numbers.

### Trick 3: Version Your Analysis
In 08_analysis.md, keep versions:
```
## Analysis v1 (after 6 experiments)
Pattern: Heterophilic wins

## Analysis v2 (after 10 experiments)  
Pattern revised: Not just heterophily, but also feature dimensionality...

## Analysis v3 (final)
Pattern: ...
```

This shows thinking evolution.

### Trick 4: The "Assume I'm Wrong" Prompt
When stuck:
```
"Assume everything I just wrote is wrong. What would that change?
What alternative explanation fits the data just as well?"
```

---

## 🎯 How to Use This System

### Option 1: Reference-Based
Keep this file open. When starting work in any file (01_problem.md, 11_experiment_logs.md, etc.), reference the corresponding phase above.

### Option 2: Embedded Prompts
I can add the phase prompt directly into each `.md` file as a comment block at the top:
```markdown
<!-- PHASE 2: EXPERIMENT LOGGING -->
<!-- Use this prompt when adding experiments: ... -->
```

### Option 3: VS Code Snippets
I can create snippets so you press `Ctrl+Shift+P` → "Research Phase 2" and get the full prompt injected.

---

## 📊 File → Phase Mapping

| File | Phase | Mode | When to Use |
|------|-------|------|-------------|
| 01_problem.md | 1 | Exploration | Start of project |
| 14_ideas_dump.md | 1 | Brainstorm | When stuck |
| 11_experiment_logs.md | 2 | Raw Logging | Every experiment |
| 08_analysis.md | 3 | Pattern Finding | After 3+ experiments |
| 04_method.md | 4 | Formalization | After patterns clear |
| 05_design_choices.md | 4 | Rationale | Explaining each design |
| 07_results_raw.md | 5 | Deep Analysis | Results complete |
| 09_limitations.md | 6 | Critical Review | Before paper |
| 12_questions.md | 6 | Doubts | Throughout |
| Final Paper | 7 | Academic | After all validation |

---

## ✅ Validation Checklist

Before moving to next phase:

- [ ] **Phase 1**: Have I listed 3+ possible directions (not just one)?
- [ ] **Phase 2**: Are my experiment logs raw and honest (including failures)?
- [ ] **Phase 3**: Do I have patterns from 3+ independent experiments?
- [ ] **Phase 4**: Can I explain method to someone unfamiliar?
- [ ] **Phase 5**: Did I consider alternative explanations?
- [ ] **Phase 6**: Did I list 3+ real weaknesses?
- [ ] **Phase 7**: Is every claim backed by experiments?

---

## 🔑 Key Insight

If you follow this system:

- Copilot stops being a "text generator"
- It becomes a **thinking partner**
- Your workflow becomes: Messy → Insight → Structure → Paper

Instead of: Prompt → Fancy text → Confusion

---

## Next: Implementation

I can:

1. ✅ **Add embedded prompts** to each file (comment block at top)
2. ✅ **Create VS Code snippets** for instant prompt injection
3. ✅ **Refactor existing files** to follow phases properly
4. ✅ **Create a checklist template** for tracking phase progress

Which would help most?