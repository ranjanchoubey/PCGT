# ⚡ Phase Prompts Quick Reference Card

**Keep this open. Copy prompts when needed.**

---

## 🧠 Phase 1: Exploration (Problem Definition)

**File**: `01_problem.md`, `14_ideas_dump.md`  
**When**: Early stage, brainstorming  
**Time**: 2-5 min

```
We are in early research phase for PCGT.

Expand this idea into:
- What exactly is the problem?
- Why does it matter?
- Where does it fail today?
- What are possible directions (multiple, not just one)?
- What are risks or unknowns?
- What assumptions might be wrong?

Do NOT finalize. Keep it exploratory.
Include: doubts, competing hypotheses, alternative framings.
```

---

## 🧪 Phase 2: Experiment Logging (RAW THINKING)

**File**: `11_experiment_logs.md`  
**When**: EVERY EXPERIMENT (most important!)  
**Time**: 5 min per experiment

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

---

## 🔍 Phase 3: Pattern Discovery (ANALYSIS)

**File**: `08_analysis.md`  
**When**: After 3+ experiments  
**Time**: 15 min

```
Analyze all previous experiments:

- What patterns do you observe across experiments?
- What consistently works?
- What consistently fails?
- What assumptions seem incorrect?
- What new hypotheses emerge?
- What are contradictions or surprises?
- What is counterintuitive?

Be critical. Do not try to make things look good.

Focus on:
- Where does it fail?
- What breaks the model?
- Where could you be wrong?
```

---

## 🧠 Phase 4: Method Formation (STRUCTURE)

**File**: `04_method.md`, `05_design_choices.md`  
**When**: After patterns are clear  
**Time**: 30-45 min

```
Based on discovered patterns, define the method:

- What is the core insight?
- Step-by-step process (each step explains one finding)
- Why does each step exist? (connect to patterns you found)
- What problem does each step solve? (from failure analysis)
- Assumptions behind each step (list explicitly)
- Edge cases where it might fail

Explain like you're teaching someone unfamiliar.
Without them knowing your paper, could they understand?

Ask yourself:
- Is each design choice justified by patterns?
- Where are you least confident?
- What could break this method?
```

---

## 📊 Phase 5: Results Interpretation (DEEP ANALYSIS)

**File**: `07_results_raw.md`  
**When**: Results are complete  
**Time**: 20 min

```
Interpret results deeply:

- Why did results look like this?
- Where does the method excel? Where does it fail?
- Are results consistent with your expectations? If not, why?
- What hidden factors might influence outcomes?
- What would someone misunderstand if you just reported numbers?
- What do error bars and std deviations tell you?

Avoid surface-level explanation.

BAD: "CiteSeer scored 73.44%"
GOOD: "CiteSeer's tightest std (±0.21) suggests stable learning. 
       This consistency indicates the method handles Planetoid splits well..."

Ask yourself:
- What are you uncertain about?
- What other interpretation fits the data equally well?
- What would change your conclusion?
```

---

## ⚠️ Phase 6: Critical Review (BRUTAL HONESTY)

**File**: `09_limitations.md`, `12_questions.md`  
**When**: Before paper writing  
**Time**: 30-60 min

```
Act like a harsh conference reviewer:

- What is unclear or poorly justified in this work?
- What assumptions are unjustified or questionable?
- What experiments are missing?
- What alternative explanations exist for the results?
- Where is the evidence weak?
- What would make this paper rejectable?

Be brutally honest. Assume:
- You don't want to accept this paper
- You are looking for flaws
- You know similar work exists

CRITICAL QUESTIONS:
- What are the 3 biggest weaknesses?
- What one experiment would change the verdict?
- Is the contribution actually novel?
- Are results reproducible with given details?
```

---

## ✍️ Phase 7: Paper Writing (FINAL POLISH)

**When**: All other phases complete  
**Time**: 2-3 hours

```
Convert all validated content into research paper draft:

- Use academic tone
- Keep logical flow
- Compress explanations but preserve meaning
- Assume reader has basic GNN knowledge but not your specific method

ORDER TO WRITE:
1. Method (most important, write first)
2. Experiments (what you tested and why)
3. Results (what you found)
4. Discussion (why you think it happened)
5. Related Work (how this fits in literature)
6. Introduction (motivation, written last with all context)
7. Abstract (summary, written absolute last)

CHECK:
- Is every claim connected to experiments?
- Are assumptions stated explicitly?
- Could a reviewer reproduce this?
```

---

## 🔄 Daily Loop (Copy This)

```
1. Open 14_ideas_dump.md → Plan experiment (2 min)
2. Run experiment
3. Open 11_experiment_logs.md → Log using Phase 2 prompt (5 min)
4. (After 3+ experiments) Open 08_analysis.md → Phase 3 prompt (15 min)
5. (Pattern clear) Open 04_method.md → Phase 4 prompt (30 min)
6. (Results in) Open 07_results_raw.md → Phase 5 prompt (20 min)
7. (Before paper) Open 09_limitations.md → Phase 6 prompt (30 min)
8. (Only when ready) Paper writing → Phase 7 mode
```

---

## 💡 How to Use This Card

### Scenario 1: You just ran an experiment
```
1. Copy Phase 2 prompt (above)
2. Paste to Copilot
3. Paste your experiment output
4. Copy response to 11_experiment_logs.md
```

### Scenario 2: You have 3+ experiments
```
1. Copy Phase 3 prompt (above)
2. List links to your experiment logs
3. Paste prompt + links to Copilot
4. Copy analysis to 08_analysis.md
```

### Scenario 3: You see patterns
```
1. Copy Phase 4 prompt (above)
2. Paste your patterns from Phase 3
3. Have Copilot write/refine method
4. Update 04_method.md + 05_design_choices.md
```

### Scenario 4: You're done, before paper
```
1. Copy Phase 6 prompt (above)
2. Paste entire work to Copilot
3. Have Copilot act as harsh reviewer
4. Document weaknesses in 09_limitations.md
5. Decide: ready for Phase 7 or need more work?
```

---

## ⚡ Pro Tips

### Tip 1: Always End With Questions
Add to every prompt:
```
Also tell me:
- What is still unclear?
- What needs verification?
- What might I be wrong about?
```

### Tip 2: Use "But" as Signal
If results surprise you:
```
"We expected X, but got Y. Why would that be?"
```

### Tip 3: Keep Versions
In analysis files, track evolution:
```
## Version 1 (exp 1-5)
Pattern: X

## Version 2 (exp 6-10)
Pattern: X + Y

## Version 3 (final)
Pattern: X + Y + Z
```

### Tip 4: When Stuck
```
"Assume everything I wrote is wrong.
What alternate explanation fits the data?"
```

---

## 📍 File Locations

| Phase | Files |
|-------|-------|
| 1 | `01_problem.md`, `14_ideas_dump.md` |
| 2 | `11_experiment_logs.md` |
| 3 | `08_analysis.md` |
| 4 | `04_method.md`, `05_design_choices.md` |
| 5 | `07_results_raw.md` |
| 6 | `09_limitations.md`, `12_questions.md` |
| 7 | Final paper document |

---

## ✅ Sanity Check Before Each Phase

- **Phase 1 done**: Can you describe 3+ different problem angles?
- **Phase 2 done**: Have you logged 3+ experiments raw/honest?
- **Phase 3 done**: Can you see clear patterns across experiments?
- **Phase 4 done**: Can you explain method without referring to paper?
- **Phase 5 done**: Do you understand WHY results look like they do?
- **Phase 6 done**: Have you listed 3+ real weaknesses?
- **Phase 7 ready**: Is everything else complete?

---

## 🚀 Remember

This system works because it **mimics real research thinking**:

```
Messy Exploration
        ↓
   Raw Logging
        ↓
   Pattern Discovery
        ↓
Method Formalization
        ↓
 Deep Interpretation
        ↓
  Honest Critique
        ↓
    Final Paper
```

NOT:

```
Prompt → Fancy Text → Confusion 😅
```

You are thinking like a researcher. Copilot is your thinking partner.