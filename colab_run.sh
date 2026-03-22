#!/bin/bash

# =============================================================
# PCGT — Colab GPU Runner
# Copy each section into a separate Colab cell, or run all at once.
# Assumes: Colab with GPU runtime (T4/A100/V100)
# =============================================================
set -e

# ── Cell 1: Clone repo & install dependencies ──────────────────
cd /content
if [ ! -d "PCGT" ]; then
  git clone https://github.com/ranjanchoubey/PCGT.git
else
  cd PCGT && git pull && cd /content
fi

pip install -q torch-geometric pyg-lib torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace('.',''))").html
pip install -q pymetis scikit-learn ogb gdown requests

# Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: No GPU detected! Enable GPU runtime.'
print(f'GPU OK: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}')
print(f'PyTorch {torch.__version__}')
import torch_geometric; print(f'PyG {torch_geometric.__version__}')
"

# ── Cell 2: Download all medium-scale datasets ─────────────────
cd /content/PCGT
DATA_DIR="data"
mkdir -p "$DATA_DIR/deezer" "$DATA_DIR/geom-gcn/film" \
         "$DATA_DIR/wiki_new/chameleon" "$DATA_DIR/wiki_new/squirrel" \
         "$DATA_DIR/Planetoid"

# 2a. Planetoid (cora/citeseer/pubmed) — auto-downloaded by PyG
python -c "
from torch_geometric.datasets import Planetoid
root = '$DATA_DIR/Planetoid'
for name in ['cora', 'citeseer', 'pubmed']:
    ds = Planetoid(root=root, name=name)
    d = ds[0]
    print(f'{name}: nodes={d.num_nodes} edges={d.num_edges} feats={d.num_node_features}')
"

# 2b. Deezer
if [ ! -f "$DATA_DIR/deezer/deezer-europe.mat" ]; then
  gdown "1P6w53eYamAPVuI_PVbbJVBidrxJfMl9f" -O "$DATA_DIR/deezer/deezer-europe.mat"
fi

# 2c. Film (edges + features + 10 split files)
FILM_DIR="$DATA_DIR/geom-gcn/film"
[ -f "$FILM_DIR/out1_graph_edges.txt" ] || \
  gdown "1szPPOymVXJibvI3SLCZkYMOFjHAAnhKK" -O "$FILM_DIR/out1_graph_edges.txt"
[ -f "$FILM_DIR/out1_node_feature_label.txt" ] || \
  gdown "1j8_2DsviL6W2cO4LCsNVo1r0c3htpbOS" -O "$FILM_DIR/out1_node_feature_label.txt"

FILM_SPLIT_IDS=(
  "1EehMEvc_HP4YKmWkra0jLGxAOKHZpu_i"
  "1XdHlGwgM8rjrnG-_dbhGjvvgE3J545qh"
  "1BjDcmtDTlRR0axSY-D_4i7L4_WfAOeIS"
  "1VPVDeLdQ8VJ9kAY-UC6v_RIL9Yd60-Li"
  "1ggi1VwfAy2IbAEpl6zezJF6uMxmfLjrS"
  "1f2Xo2bQFBlc-i5Hwg4tKdcixDcUfpPN_"
  "1AQj5U7R-StOWB7N6hXigcwZ8LyrCUBI4"
  "1qp4oALR17PkwzHPCjswKpM2q8ISHFuP8"
  "1Nm3i5zX0oN_Au9624EZU8iKYlw9zxq8o"
  "1BhWJEdr_b6vmdZFLLlKHDpCCHamURG_D"
)
for i in "${!FILM_SPLIT_IDS[@]}"; do
  target="$FILM_DIR/film_split_0.6_0.2_${i}.npz"
  [ -f "$target" ] || gdown "${FILM_SPLIT_IDS[$i]}" -O "$target"
done

# 2d. Chameleon & Squirrel (filtered, from yandex-research)
for name in chameleon squirrel; do
  target="$DATA_DIR/wiki_new/$name/${name}_filtered.npz"
  if [ ! -f "$target" ]; then
    python -c "
import requests
url = 'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/${name}_filtered.npz'
r = requests.get(url, timeout=120); r.raise_for_status()
with open('$target', 'wb') as f: f.write(r.content)
print('Downloaded ${name}_filtered.npz')
"
  fi
done

echo "=== All datasets ready ==="
ls -la "$DATA_DIR"/deezer/ "$DATA_DIR"/geom-gcn/film/ "$DATA_DIR"/wiki_new/*/

# ── Cell 3: Run ALL 7 medium-scale experiments ─────────────────
cd /content/PCGT/medium
RESULTS_DIR="results_gpu"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "========================================"
echo "  PCGT GPU Experiments — $TIMESTAMP"
echo "========================================"

# --- Cora (5-run) ---
echo ""
echo "=== [1/7] CORA 5-run ==="
python -B main.py --method pcgt --dataset cora \
  --lr 0.01 --num_layers 4 --hidden_channels 64 \
  --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn \
  --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 \
  --no_feat_norm --seed 123 --device 0 \
  --epochs 500 --patience 200 --runs 5 --display_step 100 \
  --aggregate add --data_dir ../data/ \
  --num_partitions 7 --graph_weight 0.8 \
  --dropout 0.4 --weight_decay 5e-4 \
  --ours_weight_decay 0.001 --ours_dropout 0.2 \
  2>&1 | tee "$RESULTS_DIR/cora_pcgt_gpu.txt"

# --- CiteSeer (5-run) ---
echo ""
echo "=== [2/7] CITESEER 5-run ==="
python -B main.py --method pcgt --dataset citeseer \
  --lr 0.01 --num_layers 2 --hidden_channels 64 \
  --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn \
  --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 \
  --no_feat_norm --seed 123 --device 0 \
  --epochs 500 --patience 200 --runs 5 --display_step 100 \
  --aggregate add --data_dir ../data/ \
  --num_partitions 7 --graph_weight 0.7 \
  --dropout 0.5 --weight_decay 0.01 \
  --ours_weight_decay 0.02 --ours_dropout 0.3 \
  2>&1 | tee "$RESULTS_DIR/citeseer_pcgt_gpu.txt"

# --- PubMed (5-run) ---
echo ""
echo "=== [3/7] PUBMED 5-run ==="
python -B main.py --method pcgt --dataset pubmed \
  --lr 0.01 --num_layers 2 --hidden_channels 64 \
  --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn \
  --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 \
  --no_feat_norm --seed 123 --device 0 \
  --epochs 500 --patience 200 --runs 5 --display_step 100 \
  --aggregate add --data_dir ../data/ \
  --num_partitions 50 --graph_weight 0.8 \
  --dropout 0.5 --weight_decay 5e-4 \
  --ours_weight_decay 0.01 --ours_dropout 0.3 \
  2>&1 | tee "$RESULTS_DIR/pubmed_pcgt_gpu.txt"

# --- Chameleon (5-run) ---
echo ""
echo "=== [4/7] CHAMELEON 5-run ==="
python -B main.py --method pcgt --dataset chameleon \
  --lr 0.01 --num_layers 2 --hidden_channels 64 \
  --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn \
  --no_feat_norm --seed 123 --device 0 \
  --epochs 500 --patience 200 --runs 5 --display_step 100 \
  --aggregate add --data_dir ../data/ \
  --num_partitions 10 --graph_weight 0.8 \
  --dropout 0.5 --weight_decay 0.001 \
  --ours_weight_decay 0.01 --ours_dropout 0.3 \
  2>&1 | tee "$RESULTS_DIR/chameleon_pcgt_gpu.txt"

# --- Film/Actor (10-run) ---
echo ""
echo "=== [5/7] FILM 10-run ==="
python -B main.py --method pcgt --dataset film \
  --lr 0.05 --num_layers 2 --hidden_channels 64 \
  --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn \
  --seed 123 --device 0 \
  --epochs 500 --patience 200 --runs 10 --display_step 100 \
  --aggregate add --data_dir ../data/ \
  --num_partitions 5 --graph_weight 0.5 \
  --dropout 0.5 --weight_decay 5e-4 \
  --ours_weight_decay 0.01 --ours_dropout 0.3 \
  2>&1 | tee "$RESULTS_DIR/film_pcgt_gpu.txt"

# --- Squirrel (10-run) ---
echo ""
echo "=== [6/7] SQUIRREL 10-run ==="
python -B main.py --method pcgt --dataset squirrel \
  --lr 0.01 --num_layers 4 --hidden_channels 64 \
  --ours_layers 1 --num_reps 4 --partition_method metis \
  --use_graph --use_residual --backbone gcn \
  --no_feat_norm --seed 123 --device 0 \
  --epochs 500 --patience 200 --runs 10 --display_step 100 \
  --aggregate add --data_dir ../data/ \
  --num_partitions 10 --graph_weight 0.8 \
  --dropout 0.5 --weight_decay 5e-4 \
  --ours_weight_decay 0.01 --ours_dropout 0.3 \
  2>&1 | tee "$RESULTS_DIR/squirrel_pcgt_gpu.txt"

# --- Deezer (5-run, KMeans partitioning) ---
echo ""
echo "=== [7/7] DEEZER 5-run (KMeans) ==="
python -B main.py --method pcgt --dataset deezer-europe \
  --lr 0.01 --num_layers 2 --hidden_channels 64 \
  --ours_layers 1 --num_reps 4 --partition_method kmeans \
  --use_graph --use_residual --backbone gcn \
  --rand_split --seed 123 --device 0 \
  --epochs 500 --patience 200 --runs 5 --display_step 100 \
  --aggregate add --data_dir ../data/ \
  --num_partitions 50 --graph_weight 0.5 \
  --dropout 0.6 --weight_decay 5e-5 \
  --ours_weight_decay 0.01 --ours_dropout 0.4 \
  2>&1 | tee "$RESULTS_DIR/deezer_pcgt_gpu.txt"

# ── Cell 4: Print summary ──────────────────────────────────────
echo ""
echo "========================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "========================================"
echo ""
echo "--- Results Summary ---"
for f in "$RESULTS_DIR"/*_gpu.txt; do
  dataset=$(basename "$f" | sed 's/_pcgt_gpu.txt//')
  result=$(grep -E "^[0-9]+ runs:" "$f" | tail -1)
  echo "  $dataset: $result"
done
echo ""
echo "Full logs in: /content/PCGT/medium/$RESULTS_DIR/"
