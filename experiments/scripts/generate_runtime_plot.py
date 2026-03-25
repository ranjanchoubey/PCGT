#!/usr/bin/env python3
"""
Generate runtime comparison bar chart: GCN vs SGFormer vs PCGT.
Uses the actual timing data from H100 experiments.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data from experiments/MANIFEST.md — H100 timing (ms/epoch)
datasets = [
    "Cora", "CiteSeer", "PubMed", "Chameleon", "Squirrel", 
    "Film", "Deezer", "Co-CS", "Co-Phys", "Am-Comp", "Am-Photo"
]
gcn_times  = [3.49, 3.72, 3.61, 2.36, 3.33, 3.88, 7.59, 3.96, 6.58, 4.16, 3.44]
sgf_times  = [5.48, 5.41, 5.43, 4.63, 8.04, 5.87, 14.32, 6.44, 9.85, 6.54, 5.62]
pcgt_times = [15.93, 26.80, 61.07, 15.06, 16.87, 10.24, 34.12, 22.28, 30.51, 17.88, 16.91]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

bars1 = ax.bar(x - width, gcn_times, width, label='GCN', color='#88CCEE', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, sgf_times, width, label='SGFormer', color='#4477AA', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, pcgt_times, width, label='PCGT (ours)', color='#CC3311', edgecolor='white', linewidth=0.5)

ax.set_ylabel('Time per Epoch (ms)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=8, rotation=30, ha='right')
ax.legend(fontsize=9, loc='upper left')
ax.set_yscale('log')
ax.set_ylim(1, 100)
ax.grid(True, axis='y', alpha=0.3)
ax.tick_params(labelsize=9)

# Add ratio labels on top of PCGT bars
for i, (p, s) in enumerate(zip(pcgt_times, sgf_times)):
    ratio = p / s
    ax.text(x[i] + width, p * 1.08, f'{ratio:.1f}×', 
            ha='center', va='bottom', fontsize=6.5, color='#CC3311', fontweight='bold')

plt.tight_layout()
outpath = os.path.join(BASE, "paper", "figures", "runtime_comparison.pdf")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, bbox_inches='tight')
print(f"Saved to {outpath}")
outpath_png = outpath.replace('.pdf', '.png')
plt.savefig(outpath_png, bbox_inches='tight')
print(f"Saved to {outpath_png}")
