#!/usr/bin/env python3
"""
Generate convergence comparison plot: PCGT vs SGFormer.
4 datasets in a 2×2 grid, test accuracy vs epoch.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = json.load(open(os.path.join(BASE, "experiments", "convergence_data_full.json")))

datasets = [
    ("cora", "Cora ($h$=0.81)"),
    ("pubmed", "PubMed ($h$=0.80)"),
    ("chameleon", "Chameleon ($h$=0.46)"),
    ("squirrel", "Squirrel ($h$=0.22)"),
]

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), dpi=150)
axes = axes.flatten()

colors = {"sgformer": "#4477AA", "pcgt": "#CC3311"}
labels = {"sgformer": "SGFormer", "pcgt": "PCGT (ours)"}

for idx, (ds_key, ds_title) in enumerate(datasets):
    ax = axes[idx]
    for method in ["sgformer", "pcgt"]:
        if ds_key in DATA and method in DATA[ds_key]:
            epochs_data = DATA[ds_key][method]
            if not epochs_data:
                continue
            epochs = [d["epoch"] for d in epochs_data]
            test_acc = [d["test"] for d in epochs_data]
            
            # Smooth with rolling average (window=10)
            if len(test_acc) > 10:
                kernel = np.ones(10) / 10
                test_smooth = np.convolve(test_acc, kernel, mode='valid')
                epochs_smooth = epochs[4:-5]  # center the window
            else:
                test_smooth = test_acc
                epochs_smooth = epochs
            
            ax.plot(epochs_smooth, test_smooth, 
                    color=colors[method], 
                    label=labels[method],
                    linewidth=1.5, alpha=0.9)
    
    ax.set_title(ds_title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

plt.tight_layout(pad=1.5)
outpath = os.path.join(BASE, "paper", "figures", "convergence.pdf")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, bbox_inches='tight')
print(f"Saved to {outpath}")

# Also save PNG for quick viewing
outpath_png = outpath.replace('.pdf', '.png')
plt.savefig(outpath_png, bbox_inches='tight')
print(f"Saved to {outpath_png}")
