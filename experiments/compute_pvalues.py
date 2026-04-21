"""Compute Welch's t-test p-values: PCGT vs SGFormer for Table 3 datasets."""
import re, numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).parent / "final_results"

def extract_highest_test(logpath):
    """Extract per-run 'Highest Test' values from a log file."""
    vals = []
    with open(logpath) as f:
        for line in f:
            m = re.search(r"Run \d+:.*Highest Test:\s*([\d.]+)", line)
            if m:
                vals.append(float(m.group(1)))
    return np.array(vals)

# Table 3 datasets (both PCGT and SGFormer available, 10 runs each)
table3 = BASE / "table3_additional"
datasets = ["coauthor-cs", "coauthor-physics", "amazon-computers", "amazon-photo"]
nice_names = {"coauthor-cs": "Co-CS", "coauthor-physics": "Co-Physics",
              "amazon-computers": "Am-Computers", "amazon-photo": "Am-Photo"}

print("=" * 75)
print(f"{'Dataset':<16} {'PCGT':>14} {'SGFormer':>14} {'Diff':>7} {'t-stat':>8} {'p-value':>10} {'Sig?':>5}")
print("=" * 75)

for ds in datasets:
    pcgt = extract_highest_test(table3 / f"{ds}_pcgt_10run.log")
    sgf = extract_highest_test(table3 / f"{ds}_sgformer_10run.log")
    
    t_stat, p_val = stats.ttest_ind(pcgt, sgf, equal_var=False)  # Welch's
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    
    print(f"{nice_names[ds]:<16} {pcgt.mean():.2f}±{pcgt.std():.2f}  "
          f"{sgf.mean():.2f}±{sgf.std():.2f}  {pcgt.mean()-sgf.mean():+.2f}  "
          f"{t_stat:+.3f}  {p_val:.4f}    {sig}")

print("=" * 75)

# Also extract Table 1 PCGT per-run data (for reporting confidence intervals)
print("\n--- Table 1: PCGT per-run stats (no SGFormer per-run data available) ---")
table1 = BASE / "table1_main"
t1_datasets = ["cora", "citeseer", "pubmed", "chameleon", "squirrel", "film", "deezer"]
t1_nice = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed",
           "chameleon": "Chameleon", "squirrel": "Squirrel", "film": "Film", "deezer": "Deezer"}

for ds in t1_datasets:
    suffix = "5run" if ds == "deezer" else "10run"
    path = table1 / f"{ds}_pcgt_{suffix}.log"
    vals = extract_highest_test(path)
    ci95 = 1.96 * vals.std() / np.sqrt(len(vals))
    print(f"  {t1_nice[ds]:<12} {vals.mean():.2f} ± {vals.std():.2f}  "
          f"(n={len(vals)}, 95% CI: [{vals.mean()-ci95:.2f}, {vals.mean()+ci95:.2f}])")
