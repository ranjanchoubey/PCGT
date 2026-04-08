# How to Use Claude Opus 4.6 to Push PCGT Beyond SOTA

## ⚠️ ISOLATION PROTOCOL — READ THIS FIRST

Your current clean paper lives on `version3` branch. The paper/ directory
is NOT tracked in git (only medium/*.py and experiments/ are tracked).
Here is the **exact** setup to run autonomous experiments without risking
anything:

### Step 1: Commit everything clean on version3

```bash
cd /Users/vn59a0h/thesis/PCGT
git add -A && git commit -m "clean state before experiments"
git push origin version3
```

### Step 2: Create an experiment branch

```bash
git checkout -b experiments/v4-architecture
```

All code changes from the autonomous agent happen on THIS branch.
Your `version3` branch stays untouched.

### Step 3: Create isolated log directory

```bash
mkdir -p logs/experiments_v4
```

All experiment logs go here (already gitignored via `/logs/`).

### Step 4: Protect the paper

Your `paper/` directory is NOT in git. Before starting experiments:

```bash
# One-time backup
cp -r paper/ paper_backup_$(date +%Y%m%d)/
```

Then tell Claude:

> **"Do NOT modify any files under paper/ or experiments/final_results/.
> Only modify files under medium/ and write logs to logs/experiments_v4/.
> You are on the experiments/v4-architecture branch."**

### Step 5: After experiments succeed

```bash
# On experiments/v4-architecture branch, review changes:
git diff version3 -- medium/pcgt.py

# If results are better, merge into version3:
git checkout version3
git merge experiments/v4-architecture

# If results are bad, just delete the branch:
git checkout version3
git branch -D experiments/v4-architecture
```

### Step 6: Add to every autonomous prompt

Add this preamble to EVERY conversation with Claude:

```
SAFETY RULES (non-negotiable):
1. You are on git branch: experiments/v4-architecture
2. NEVER modify files in: paper/, experiments/final_results/
3. ALL logs go to: logs/experiments_v4/
4. ONLY modify code in: medium/pcgt.py, medium/parse.py, medium/main.py
5. Before modifying pcgt.py, ALWAYS copy original:
   cp medium/pcgt.py medium/pcgt_backup.py
6. Use --data_dir ../data --cpu --seed 123
7. After each experiment, print the diff: git diff medium/pcgt.py
```

---

## Your Current Position

| Dataset | PCGT | Best Baseline | Gap |
|---------|------|---------------|-----|
| Cora | 84.3 | 84.5 (SGFormer) | -0.2 (tie) |
| CiteSeer | 73.1 | 72.6 (SGFormer) | +0.5 |
| PubMed | 81.0 | 80.3 (SGFormer) | +0.7 |
| Chameleon | 49.0 | 44.9 (SGFormer) | **+4.1** |
| Squirrel | 45.5 | 41.8 (SGFormer) | **+3.7** |
| Film | 38.0 | 37.9 (SGFormer) | +0.1 |
| Deezer | 67.2 | 67.1 (SGFormer) | +0.1 |
| GraphGPS | — | Loses to PCGT on all 7 | Already beaten |

**Hardware:** Mac with MPS (no CUDA). CPU-only for experiments.
**Constraint:** Medium-scale datasets (≤35K nodes) run in minutes on CPU.

---

## Strategy 1: Autonomous Architecture Search (Highest Impact)

### What to ask Claude

```
Read medium/pcgt.py completely. I want you to autonomously design, implement,
and test 5 architectural modifications to PCGTConvLayer that could improve
accuracy on heterophilic graphs (Chameleon, Squirrel). For each modification:

1. Implement it as a new variant in pcgt.py (with a flag to toggle)
2. Run 3-run experiments on Chameleon and Squirrel on CPU
3. Report results vs baseline (49.0 and 45.5)
4. Keep the variant if it improves, revert if it doesn't

Modifications to try (ranked by likelihood of helping):
- Multi-head attention (num_heads=4 instead of 1)
- Per-partition α instead of global α
- Separate seed vectors per partition (not shared)
- Add edge features to the attention computation
- Layer-wise partition re-encoding (refresh PSE each layer)

Run each experiment end-to-end. Do NOT ask me questions — just run them.
Use --cpu --runs 3 --epochs 500 --seed 123.
After all 5, summarize in a table which variants helped.
```

### Why this works
Claude can read your full codebase, modify `pcgt.py`, run experiments via
terminal, parse the output, and iterate—all in one conversation. A human
would spend 2-3 days on this. Claude can do it in one session.

---

## Strategy 2: Hyperparameter Sweep (Quick Wins)

### What to ask Claude

```
Run a systematic hyperparameter sweep for PCGT on Chameleon and Squirrel.
Current best configs are in paper/sections/09_appendix.tex Table 8.

Sweep these parameters one at a time, holding others at their current values:
- K ∈ {3, 5, 10, 15, 20, 30}
- M (num_reps) ∈ {2, 4, 8, 16}
- ours_layers ∈ {1, 2, 3}
- ours_dropout ∈ {0.1, 0.2, 0.3, 0.5}
- graph_weight ∈ {0.3, 0.5, 0.7, 0.8, 0.9}
- learn_graph_weight on vs off
- num_layers (GCN) ∈ {2, 3, 4, 6}

Use --cpu --runs 3 --epochs 500 --seed 123 for each.
Track results in a table. After the sweep, run the best config with 10 runs.
```

---

## Strategy 3: New Architectural Ideas (Medium Effort, High Reward)

Ask Claude to implement and test each of these:

### 3a. Attention with Edge Features
```
Modify PCGTConvLayer.forward() so that intra-partition attention uses
edge features: attn_ij = Q_i · K_j + edge_feat(i,j). This requires
passing edge_index into the attention layer. The edge feature can be
a learnable scalar per edge or derived from node degree difference.
Implement, test on Chameleon (3 runs), report accuracy.
```

### 3b. Hierarchical Partitions (2-level METIS)
```
Instead of flat K partitions, compute 2-level hierarchy:
- Level 1: K1=5 coarse partitions
- Level 2: K2=4 fine partitions within each coarse partition (total 20)
- Local attention at level 2 (fine), global attention at level 1 (coarse)
This exploits multi-resolution community structure more explicitly.
Implement in partition.py and pcgt.py. Test on Squirrel.
```

### 3c. Performer-style Global Attention
```
Replace the seed-based global attention with FAVOR+ (Performer) random
feature maps for O(N) global attention. Compare speed and accuracy
against the current seed approach on PubMed (19K nodes) and Deezer (28K).
```

### 3d. Learnable Partition Refinement
```
After initial METIS partition, add a learnable "soft reassignment" step:
each node gets a probability distribution over partitions (Gumbel-Softmax),
initialized from METIS but fine-tuned during training. This lets the model
fix partition mistakes. You already have diff_partition.py as a starting
point. Test on Cora and Chameleon.
```

---

## Strategy 4: Paper Strengthening (Theory + Writing)

### 4a. Formalize the Nyström bound
```
Read paper/sections/03_method.tex, specifically the Nystrom remark.
I want to upgrade it to a proposition with a proof sketch.

The key insight: PCGT's attention matrix is
  A = α·A_local + (1-α)·A_global
where A_local is block-diagonal (rank N) and A_global is rank KM.

Can you derive a bound on ||A_full - A_PCGT||_F in terms of the
between-partition block of the full attention matrix? Use the fact
that METIS minimizes edge cut, which relates to the off-diagonal
block magnitude. Write the proposition + proof in LaTeX.
```

### 4b. Add a theoretical analysis section
```
Read the full paper (all sections/*.tex). Add a short theoretical
analysis section (0.5 page) proving that PCGT's partition attention
is a better Nyström approximation than random landmarks when the graph
has community structure. Use spectral graph theory: the METIS partition
minimizes the Fiedler ratio, which bounds the between-cluster attention
energy. Write it as Section 3.6 before the Complexity subsection.
```

---

## Strategy 5: Autonomous Experiment Runner

The most powerful pattern: give Claude a **complete experimental plan**
and let it execute autonomously.

```
I want you to run the following experimental plan autonomously.
Do NOT ask me any questions. Execute each step, collect results,
and present a final summary.

PLAN:
1. Read medium/pcgt.py and understand the architecture
2. Implement multi-head attention (4 heads) as a flag --num_heads 4
3. Run on all 7 datasets with 3 runs each on CPU
4. Compare against current 1-head results
5. If any dataset improves by >0.3%, keep the change
6. Next: implement per-partition alpha (each partition gets its own α)
7. Run on the datasets where multi-head helped
8. Stack improvements: apply both changes together
9. Run final 10-run evaluation on the stacked improvements
10. Update the paper tables if results improve

Base configs per dataset are in paper/sections/09_appendix.tex Table 8.
Always use --cpu --seed 123 --data_dir ../data
```

---

## Practical Tips for Working with Claude

### DO:
- **Give full file paths.** Claude reads exact files, not guesses.
- **Say "run autonomously, don't ask questions."** Otherwise it will
  ask for confirmation at every step.
- **Provide the baseline numbers.** "Current best on Chameleon is 49.0.
  Beat it." Claude works better with a clear target.
- **Let it run experiments in background terminals.** It can launch
  long jobs with `isBackground=true` and check later.
- **Ask for incremental changes.** "Try one thing, test it, then
  decide the next step" works better than "redesign everything."

### DON'T:
- Don't ask "what should I try?" — instead say "try X, run it, report."
- Don't ask Claude to design experiments without running them — it will
  give generic advice. Force it to execute.
- Don't give vague goals like "improve the paper" — give specific
  targets like "beat 49.0 on Chameleon with 3-run mean."
- Don't run multiple heavy experiments in parallel on CPU — they'll
  compete for resources. Queue them sequentially.

### Session Management:
- Each conversation has a context limit. For long experiment sessions,
  save results to a log file and start a new conversation with:
  "Read logs/experiment_results.log and continue from where we left off."
- Use session memory (`/memories/session/`) to persist experiment state
  across conversation breaks.

---

## Recommended Order of Attack

| Priority | Action | Expected Gain | Time |
|----------|--------|---------------|------|
| 1 | Multi-head attention (4 heads) | +0.5-1.5% on hetero | 1 hour |
| 2 | HP sweep (K, M, dropout, gw) | +0.3-1.0% per dataset | 2-3 hours |
| 3 | Per-partition α | +0.2-0.5% on hetero | 1 hour |
| 4 | ours_layers=2 with proper tuning | +0.5-1.0% if stable | 1 hour |
| 5 | Edge-aware attention | +0.5-2.0% on hetero | 2 hours |
| 6 | Hierarchical partitions | unknown, exploratory | 3 hours |
| 7 | Strengthen Nyström theory | paper quality, not acc | 1 hour |

**Start with #1 and #2.** They're low-risk, high-signal, and will tell
you whether PCGT's architecture has headroom or is near its ceiling.

---

## Example: Full Autonomous Session Prompt

Copy-paste this into a new Claude conversation to start:

```
I'm working on PCGT, a graph transformer. Read these files to understand
the project:
- medium/pcgt.py (the PCGT architecture)
- medium/main.py (training loop)
- medium/parse.py (all CLI arguments)
- paper/sections/09_appendix.tex (hyperparameter table)

Current best results (10 runs, H100):
- Chameleon: 49.0 ± 2.8
- Squirrel: 45.5 ± 2.7
- Cora: 84.3 ± 0.4

Your mission: beat these numbers. You have CPU only (--cpu flag).
Use --runs 3 --seed 123 for testing, --runs 10 for final.

Step 1: Try num_heads=4 on Chameleon and Squirrel.
Step 2: Based on results, pick the next modification.
Step 3: Stack improvements and run 10-run final.

Run autonomously. Don't ask questions. Report results after each step.
```
