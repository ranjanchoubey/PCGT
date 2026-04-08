"""
Compute Pearson correlation between learned β and dataset homophily.
Reproduces the claim r=0.006, p=0.99 in Section 6 (Discussion).

Usage:
    cd experiments && python compute_beta_correlation.py

β values are extracted from 10-run training logs (multi-run mean).
Homophily values are from the respective dataset papers.
"""
from scipy import stats
import numpy as np

# β values from multi-run experiments (mean across runs)
# Source: experiments/final_results/table5_ablation/beta_extraction*.log
beta_values = {
    'cora':              1.19,
    'citeseer':          2.38,
    'pubmed':            0.31,
    'chameleon':         3.08,
    'squirrel':          3.53,
    'film':             -0.99,
    'deezer':           -0.57,
    'coauthor-cs':       2.08,
    'coauthor-physics':  2.09,
    'amazon-computers':  2.12,
    'amazon-photo':      1.75,
}

# Edge homophily h (fraction of edges connecting same-label nodes)
homophily = {
    'cora':              0.81,
    'citeseer':          0.74,
    'pubmed':            0.80,
    'chameleon':         0.24,
    'squirrel':          0.21,
    'film':              0.22,
    'deezer':            0.53,
    'coauthor-cs':       0.81,
    'coauthor-physics':  0.93,
    'amazon-computers':  0.78,
    'amazon-photo':      0.83,
}

datasets = sorted(beta_values.keys())
betas = [beta_values[d] for d in datasets]
homos = [homophily[d] for d in datasets]

r, p = stats.pearsonr(betas, homos)

print("Dataset              β        h")
print("-" * 42)
for d in datasets:
    print(f"{d:<22s} {beta_values[d]:>6.2f}    {homophily[d]:.2f}")
print("-" * 42)
print(f"\nPearson r = {r:.3f}, p = {p:.2f}")
print(f"n = {len(datasets)} datasets")
print(f"\nConclusion: {'No significant correlation' if p > 0.05 else 'Significant correlation'}")
