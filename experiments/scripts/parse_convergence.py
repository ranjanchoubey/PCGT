#!/usr/bin/env python3
"""
Parse convergence logs and generate convergence_data_full.json
with per-epoch test accuracy for SGFormer vs PCGT.
"""
import re, json, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(BASE, "experiments", "logs")
CONV_JSON = os.path.join(BASE, "experiments", "convergence_data.json")

# Pattern: "Epoch: 42, Loss: 1.234, Train: 50.00%, Valid: 48.00%, Test: 47.00%"
PAT = re.compile(
    r"Epoch:\s+(\d+),.*Train:\s+([\d.]+)%.*Valid:\s+([\d.]+)%.*Test:\s+([\d.]+)%"
)

def parse_log(path):
    """Extract per-epoch data from a single log file."""
    data = []
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                data.append({
                    "epoch": int(m.group(1)),
                    "train": float(m.group(2)),
                    "valid": float(m.group(3)),
                    "test": float(m.group(4)),
                })
    return data

results = {}

# First, load existing convergence_data.json (cora, pubmed)
if os.path.exists(CONV_JSON):
    with open(CONV_JSON) as f:
        old = json.load(f)
    for ds, methods in old.items():
        for method, epochs in methods.items():
            if epochs:
                results.setdefault(ds, {})[method] = epochs

# Then parse individual convergence log files (chameleon, squirrel)
for ds in ["chameleon", "squirrel"]:
    for method in ["sgformer", "pcgt"]:
        logfile = os.path.join(LOG_DIR, f"convergence_{ds}_{method}.log")
        if os.path.exists(logfile):
            data = parse_log(logfile)
            if data:
                results.setdefault(ds, {})[method] = data
                print(f"  {ds}/{method}: {len(data)} epochs from log file")
            else:
                print(f"  {ds}/{method}: log exists but 0 epochs parsed")
        else:
            print(f"  {ds}/{method}: no log file")

# Save combined data
outpath = os.path.join(BASE, "experiments", "convergence_data_full.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {outpath}")
for ds in sorted(results):
    for m in sorted(results[ds]):
        n = len(results[ds][m])
        last = results[ds][m][-1]["test"] if n > 0 else "N/A"
        print(f"  {ds}/{m}: {n} epochs, final test={last}")
