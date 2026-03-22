# 🚀 Daily Workflow Quick-Start

**Use this as your daily reference** while working on PCGT research.

---

## ⚡ The Loop (Copy/Paste Friendly)

Every day, you cycle through these steps:

### Step 1: Plan Experiment (2 min)
Open `14_ideas_dump.md`, jot down what you want to try today.
```
What am I testing? Why? What do I expect?
```

### Step 2: Run Experiment (variable)
Execute your experiment (code, GPU, analysis, etc.)

### Step 3: Log Raw Results (5 min) ⭐ CRITICAL
Open `11_experiment_logs.md` and add:

```
## Experiment: [NAME] ([DATE])

**What was attempted**: 
[Copy-paste, be detail-oriented]

**Hypothesis/Expected**: 
[What you thought would happen]

**What actually happened**:
[EXACT results and numbers]

**Key observations**:
[The "aha" moments]

**Surprises**:
[Anything unexpected?]

**Unclear/Needs verification**:
[What questions remain?]

**Next step**:
[What's next?]
```

### Step 4: After 3+ Experiments (15 min)
Open `08_analysis.md` → Use **Phase 3 prompt** (look at top of file)

Run the prompt. Write patterns you're seeing.

### Step 5: When Pattern is Clear (30 min)
Open `04_method.md` + `05_design_choices.md` → Use **Phase 4 prompts**

Formalize what you've learned into structured method.

### Step 6: Before Paper (1 hour)
Open `09_limitations.md` + `12_questions.md` → Use **Phase 6 prompts**

Be brutally honest about weaknesses.

### Step 7: Paper Time (only when ready)
Use Phase 7 mode → Academic polish

---

## 📁 File Reference

| File | Phase | Trigger | Time |
|------|-------|---------|------|
| `14_ideas_dump.md` | 1 | Daily, when stuck | 2 min |
| `11_experiment_logs.md` | 2 | After every experiment | 5 min |
| `08_analysis.md` | 3 | After 3+ experiments | 15 min |
| `04_method.md` | 4 | After patterns are clear | 30 min |
| `05_design_choices.md` | 4 | Explaining each choice | 20 min |
| `07_results_raw.md` | 5 | When all results are in | 20 min |
| `09_limitations.md` | 6 | Before paper writing | 30 min |
| `12_questions.md` | 6 | Throughout research | 10 min |

---

## 🧠 Magic Prompts (Copy These)

### When You Run An Experiment
```
Log this experiment like a researcher's lab notebook:
- What was attempted?
- Why?
- Expected?
- What actually happened?
- Observations?
- Surprises?
- What this tells us?
- What needs verification?
- What might be wrong?
```
👉 File: `11_experiment_logs.md`

### When You Have 3+ Experiments
```
Analyze all experiments:
- What patterns do you observe?
- What consistently works/fails?
- What new hypotheses emerge?
- Be critical. Don't make things look good.
- What could be wrong?
```
👉 File: `08_analysis.md`

### When Patterns Are Clear
```
Based on discovered patterns, define the method:
- What is the core insight?
- Step-by-step process
- Why each step exists?
- When might it fail?
- Explain like teaching someone from scratch.
```
👉 Files: `04_method.md`, `05_design_choices.md`

### Before Writing Paper
```
Act like a harsh conference reviewer:
- What is unclear?
- What are the 3 biggest weaknesses?
- What experiments are missing?
- Be brutally honest.
```
👉 Files: `09_limitations.md`, `12_questions.md`

---

## 💡 Pro Tips

### Tip 1: End Every Analysis With Questions
```
Also include:
- What is unclear?
- What needs verification?
- What might be wrong?
```
This prevents overconfidence.

### Tip 2: Use "But" as a Signal
```
"We expected X, but got Y. Why?"
```
This forces deeper thinking.

### Tip 3: Keep Versions
```
## Analysis v1 (after experiments 1-5)
Pattern: X

## Analysis v2 (after experiments 6-10)
Pattern: X + Y
```
Shows how thinking evolved.

### Tip 4: When Stuck, Use "Assume I'm Wrong"
```
"Assume everything I wrote is wrong.
What alternative explanation fits the data?"
```

---

## 🎯 Common Workflows

### Workflow: Run New Experiment
```
1. Jot idea in 14_ideas_dump.md (2 min)
2. Run experiment
3. Log in 11_experiment_logs.md using Phase 2 prompt (5 min)
4. Chat with Copilot: paste the Phase 2 prompt + your output
5. Done!
```
**Time**: 5-10 min logging (experiment time varies)

### Workflow: Found a Pattern
```
1. Open 08_analysis.md
2. Copy the Phase 3 prompt from top of file
3. Paste + "Here are my experiments [list links to logs]"
4. Copilot analyzes for patterns
5. Update 08_analysis.md with findings
6. Document in 12_questions.md what's still unclear
```
**Time**: 15-20 min

### Workflow: Design Method
```
1. Open 04_method.md + 05_design_choices.md
2. Copy Phase 4 prompt from top of files
3. Paste + "Here are the patterns [paste from analysis]"
4. Write explanation of each design choice
5. For each choice, answer:
   - Why this?
   - Alternatives?
   - Trade-offs?
   - Evidence?
```
**Time**: 30-45 min

### Workflow: Brutal Honesty Pass
```
1. Open 09_limitations.md + 12_questions.md
2. Copy Phase 6 prompt from top
3. Have Copilot act as harsh reviewer
4. Paste your entire method + results
5. Document weaknesses, missing experiments
6. Decide: Can this be defended? Or needs more work?
```
**Time**: 30-60 min

---

## ✅ Daily Checklist

- [ ] Opened `14_ideas_dump.md` with today's plan
- [ ] Logged experiments in `11_experiment_logs.md` (with Phase 2 prompt)
- [ ] (If 3+ experiments) Ran Phase 3 analysis
- [ ] (If patterns clear) Updated Phase 4 method
- [ ] (If final) Ran Phase 6 honest review
- [ ] Captured uncertain points in `12_questions.md`

---

## 📊 Phase Completion Checklist

Before moving to next phase, verify:

**Before Phase 2 → Phase 3**:
- [ ] Have 3+ independent experiments logged?
- [ ] Are logs raw and honest (not polished)?
- [ ] Do I see any patterns emerging?

**Before Phase 3 → Phase 4**:
- [ ] Have clear patterns across multiple experiments?
- [ ] Can I explain why results look this way?
- [ ] What is the core insight?

**Before Phase 4 → Phase 5**:
- [ ] Can someone understand the method from my description?
- [ ] Is each step justified by patterns?
- [ ] What are edge cases?

**Before Phase 5 → Phase 6**:
- [ ] All experiments completed?
- [ ] Have I explained unexpected results?
- [ ] Do results make sense with the method?

**Before Phase 6 → Phase 7**:
- [ ] Documented 3+ major weaknesses?
- [ ] Listed missing experiments?
- [ ] Honestly assessed novelty?

**Before Phase 7 (Paper)**:
- [ ] All phases complete?
- [ ] No major holes in logic?
- [ ] Ready to defend?

---

## 🚨 Red Flags (Stop & Rethink)

- "All results are great!" → You're not being critical enough
- "I'm not documenting, just running experiments" → Moving too fast
- "I jumped to conclusions" → Skipped exploration phase
- "I can't explain why this works" → Need Phase 3 analysis
- "The paper is polished but I don't understand the method" → Wrote Phase 7 too early

---

## 🎁 Templates Ready to Copy

### Template: Experiment Log Entry
```markdown
## Experiment: [NAME] ([DATE])

**What was attempted**: 

**Hypothesis/Expected**: 

**What actually happened**:

**Key observations**:

**Surprises**:

**Possible explanations**:

**Unclear/Needs verification**:

**Next step**:
```

### Template: Pattern Analysis
```markdown
## Pattern: [NAME]

**Observation**: 

**Consistency**: 

**New hypothesis**: 

**Evidence**: 

**Could be wrong if**: 
```

### Template: Weakness Analysis
```markdown
## Weakness: [NAME]

**Why it's a problem**: 

**Impact**: 

**Possible fixes**: 

**Or accept?**: 
```

---

## 🔗 Navigation

- **For deep guidance**: Read `00_prompt_system.md` (full theory)
- **For today's work**: Read this file (quick start)
- **For each phase**: Check embedded prompts at top of respective files
- **For project overview**: Read `00_status_summary.md`

---

## Final Thought

This system works because it **mimics real research thinking**:

> Messy → Insight → Structure → Paper
> (not)
> Prompt → Fancy text → Confusion

You're not writing for clients or bosses.

You're thinking like a researcher. Copilot is your thinking partner.

The documentation is just where your thinking **lives**.