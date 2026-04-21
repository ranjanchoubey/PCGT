
import subprocess, re, json, os

os.chdir("/teamspace/studios/this_studio/PCGT/medium")

datasets = {
    "chameleon": {"K": 10, "sgf_args": "", "pcgt_args": ""},
    "squirrel":  {"K": 10, "sgf_args": "", "pcgt_args": "--num_layers 4"},
    "cora":      {"K": 10, "sgf_args": "", "pcgt_args": ""},
    "pubmed":    {"K": 50, "sgf_args": "", "pcgt_args": ""},
}

results = {}

for ds, cfg in datasets.items():
    results[ds] = {}
    for method in ["sgformer", "pcgt"]:
        print(f"\n=== {ds} / {method} ===", flush=True)
        extra = cfg["pcgt_args"] if method == "pcgt" else cfg["sgf_args"]
        K_arg = f"--num_partitions {cfg[chr(75)]}" if method == "pcgt" else ""
        cmd = (
            f"source ../venv/bin/activate && "
            f"python main.py --dataset {ds} --method {method} "
            f"--hidden_channels 64 --lr 0.01 --weight_decay 5e-4 "
            f"--dropout 0.5 --epochs 500 --runs 1 --display_step 1 "
            f"--graph_weight 0.8 --aggregate add "
            f"{K_arg} {extra} --device 0"
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")

        epochs_data = []
        for line in proc.stdout.split("\n"):
            m = re.match(r"Epoch:\s+(\d+),.*Train:\s+([\d.]+)%.*Valid:\s+([\d.]+)%.*Test:\s+([\d.]+)%", line)
            if m:
                epochs_data.append({
                    "epoch": int(m.group(1)),
                    "train": float(m.group(2)),
                    "valid": float(m.group(3)),
                    "test": float(m.group(4)),
                })
        results[ds][method] = epochs_data
        print(f"  Collected {len(epochs_data)} epoch points", flush=True)

outpath = "/teamspace/studios/this_studio/PCGT/convergence_data.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outpath}")
