#!/bin/bash
# ============================================================
# COMBINED H100 SESSION — Resolves C1, C3, C4, M1
# ============================================================
# Run on Lightning.ai H100:
#   cd /teamspace/studios/this_studio/PCGT
#   bash experiments/h100_combined_session.sh
#
# Time budget: ~4 hours (parallelized to finish in ~1-1.5 hrs)
#   Phase 1: Ablation  — 8 variants, 3 datasets in parallel per variant
#   Phase 2: SGFormer  — 7 datasets fully parallel
#
# Output:
#   experiments/final_results/table5_ablation/component/
#   experiments/final_results/table1_sgformer_rerun/
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/medium"

# ─── ACTIVATE VENV ───
if [ -d "$HOME/venv" ]; then
    source "$HOME/venv/bin/activate"
elif [ -d "$REPO_ROOT/venv" ]; then
    source "$REPO_ROOT/venv/bin/activate"
fi

echo "============================================================"
echo "H100 COMBINED SESSION — $(date)"
echo "Repo root: $REPO_ROOT"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================================"

# ─── SETUP (skip if already installed) ───
python -c "import torch_geometric" 2>/dev/null || pip install torch-scatter torch-sparse torch-geometric pymetis -f https://data.pyg.org/whl/torch-2.6.0+cu124.html 2>/dev/null || true

# ============================================================
#  PHASE 1: ABLATION (C3, C4, M1)
#  8 variants × 3 datasets × 5 runs — 3 datasets run in PARALLEL
# ============================================================
ADIR="$SCRIPT_DIR/final_results/table5_ablation/component"
mkdir -p "$ADIR"
ALOG="$ADIR/ablation_run.log"

echo "" | tee "$ALOG"
echo "╔═══════════════════════════════════════════════════╗" | tee -a "$ALOG"
echo "║  PHASE 1: ABLATION (5 runs, 3 datasets parallel) ║" | tee -a "$ALOG"
echo "╚═══════════════════════════════════════════════════╝" | tee -a "$ALOG"
echo "Start: $(date)" | tee -a "$ALOG"

# ─── BASE CONFIGS ───
CORA="python -B main.py --method pcgt --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --num_partitions 7 --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2"
CHAM="python -B main.py --method pcgt --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --num_reps 4 --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3"
SQRL="python -B main.py --method pcgt --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --num_reps 4 --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --data_dir ../data/ --num_partitions 10 --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3"

# Helper: run 3 datasets in parallel for a given variant
run_variant() {
    local variant=$1
    local cora_cmd=$2
    local cham_cmd=$3
    local sqrl_cmd=$4

    echo "── $variant (3 datasets parallel) — $(date '+%H:%M:%S') ──" | tee -a "$ALOG"

    eval "$cora_cmd" > "$ADIR/cora_${variant}.log" 2>&1 &
    local pid1=$!
    eval "$cham_cmd" > "$ADIR/chameleon_${variant}.log" 2>&1 &
    local pid2=$!
    eval "$sqrl_cmd" > "$ADIR/squirrel_${variant}.log" 2>&1 &
    local pid3=$!

    wait $pid1 $pid2 $pid3

    for ds in cora chameleon squirrel; do
        local r=$(grep "Highest Test:" "$ADIR/${ds}_${variant}.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+')
        echo "    $ds: $r" | tee -a "$ALOG"
    done
}

# 1. Full PCGT
run_variant "full_pcgt" \
    "$CORA --partition_method metis" \
    "$CHAM --partition_method metis" \
    "$SQRL --partition_method metis"

# 2. w/o Global
run_variant "local_only" \
    "$CORA --partition_method metis --local_only" \
    "$CHAM --partition_method metis --local_only" \
    "$SQRL --partition_method metis --local_only"

# 3. w/o Local
run_variant "global_only" \
    "$CORA --partition_method metis --global_only" \
    "$CHAM --partition_method metis --global_only" \
    "$SQRL --partition_method metis --global_only"

# 4. w/o PSE
run_variant "no_pse" \
    "$CORA --partition_method metis --no_pse" \
    "$CHAM --partition_method metis --no_pse" \
    "$SQRL --partition_method metis --no_pse"

# 5. w/o GCN
CORA_NG=$(echo "$CORA" | sed 's/--use_graph//' | sed 's/--graph_weight 0.8/--graph_weight 0.0/')
CHAM_NG=$(echo "$CHAM" | sed 's/--use_graph//' | sed 's/--graph_weight 0.8/--graph_weight 0.0/')
SQRL_NG=$(echo "$SQRL" | sed 's/--use_graph//' | sed 's/--graph_weight 0.8/--graph_weight 0.0/')
run_variant "no_gcn" \
    "$CORA_NG --partition_method metis" \
    "$CHAM_NG --partition_method metis" \
    "$SQRL_NG --partition_method metis"

# 6. Random partitions
run_variant "random_part" \
    "$CORA --partition_method random" \
    "$CHAM --partition_method random" \
    "$SQRL --partition_method random"

# 7. Pure GCN
run_variant "pure_gcn" \
    "python -B main.py --method gcn --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --dropout 0.4 --weight_decay 5e-4" \
    "python -B main.py --method gcn --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --data_dir ../data/ --dropout 0.5 --weight_decay 0.001" \
    "python -B main.py --method gcn --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --data_dir ../data/ --dropout 0.5 --weight_decay 5e-4"

# 8. SGFormer (controlled)
run_variant "sgformer" \
    "python -B main.py --method sgformer --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --use_graph --use_residual --backbone gcn --rand_split_class --label_num_per_class 20 --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --graph_weight 0.8 --dropout 0.4 --weight_decay 5e-4 --ours_weight_decay 0.001 --ours_dropout 0.2" \
    "python -B main.py --method sgformer --dataset chameleon --lr 0.01 --num_layers 2 --hidden_channels 64 --ours_layers 1 --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --data_dir ../data/ --graph_weight 0.8 --dropout 0.5 --weight_decay 0.001 --ours_weight_decay 0.01 --ours_dropout 0.3" \
    "python -B main.py --method sgformer --dataset squirrel --lr 0.01 --num_layers 4 --hidden_channels 64 --ours_layers 1 --use_graph --use_residual --backbone gcn --no_feat_norm --seed 123 --epochs 500 --patience 200 --runs 5 --display_step 100 --aggregate add --data_dir ../data/ --graph_weight 0.8 --dropout 0.5 --weight_decay 5e-4 --ours_weight_decay 0.01 --ours_dropout 0.3"

echo "" | tee -a "$ALOG"
echo "Phase 1 complete: $(date)" | tee -a "$ALOG"

# ─── Phase 1 summary table ───
echo "────────────────────────────────────────" | tee -a "$ALOG"
printf "%-18s %-22s %-22s %-22s\n" "Variant" "Cora" "Chameleon" "Squirrel" | tee -a "$ALOG"
for v in full_pcgt local_only global_only no_pse no_gcn random_part pure_gcn sgformer; do
    c="—"; ch="—"; s="—"
    [ -f "$ADIR/cora_${v}.log" ]      && c=$(grep "Highest Test:" "$ADIR/cora_${v}.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+')
    [ -f "$ADIR/chameleon_${v}.log" ] && ch=$(grep "Highest Test:" "$ADIR/chameleon_${v}.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+')
    [ -f "$ADIR/squirrel_${v}.log" ]  && s=$(grep "Highest Test:" "$ADIR/squirrel_${v}.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+')
    printf "%-18s %-22s %-22s %-22s\n" "$v" "$c" "$ch" "$s" | tee -a "$ALOG"
done

# ============================================================
#  PHASE 2: SGFormer RE-RUN (C1)
#  7 datasets × 10 runs — ALL 7 in PARALLEL
# ============================================================
SDIR="$SCRIPT_DIR/final_results/table1_sgformer_rerun"
mkdir -p "$SDIR"
SLOG="$SDIR/sgformer_rerun.log"

echo "" | tee "$SLOG"
echo "╔═══════════════════════════════════════════════════╗" | tee -a "$SLOG"
echo "║  PHASE 2: SGFormer RE-RUN (7 datasets parallel)  ║" | tee -a "$SLOG"
echo "╚═══════════════════════════════════════════════════╝" | tee -a "$SLOG"
echo "Start: $(date)" | tee -a "$SLOG"

# Launch all 7 datasets in parallel
python -B main.py --backbone gcn --dataset cora --lr 0.01 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 --method sgformer --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.2 --use_residual --ours_weight_decay 0.001 --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 100 --data_dir ../data/ \
    > "$SDIR/cora_sgformer_10run.log" 2>&1 &
P1=$!

python -B main.py --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 --method sgformer --ours_layers 1 --use_graph --graph_weight 0.7 --ours_dropout 0.3 --use_residual --ours_weight_decay 0.01 --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 100 --data_dir ../data/ \
    > "$SDIR/citeseer_sgformer_10run.log" 2>&1 &
P2=$!

python -B main.py --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 --method sgformer --ours_layers 1 --use_graph --graph_weight 0.8 --ours_dropout 0.3 --use_residual --ours_weight_decay 0.01 --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 100 --data_dir ../data/ \
    > "$SDIR/pubmed_sgformer_10run.log" 2>&1 &
P3=$!

python -B main.py --backbone gcn --dataset film --lr 0.1 --num_layers 8 --hidden_channels 64 --weight_decay 0.0005 --dropout 0.6 --method sgformer --use_graph --graph_weight 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 --use_residual --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 100 --data_dir ../data/ \
    > "$SDIR/film_sgformer_10run.log" 2>&1 &
P4=$!

python -B main.py --backbone gcn --rand_split --dataset deezer-europe --lr 0.01 --num_layers 2 --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 --method sgformer --ours_layers 1 --use_graph --use_residual --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 100 --data_dir ../data/ \
    > "$SDIR/deezer_sgformer_10run.log" 2>&1 &
P5=$!

python -B main.py --backbone gcn --dataset squirrel --lr 0.001 --num_layers 8 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.3 --method sgformer --ours_layers 1 --use_graph --use_residual --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 100 --data_dir ../data/ \
    > "$SDIR/squirrel_sgformer_10run.log" 2>&1 &
P6=$!

python -B main.py --backbone gcn --dataset chameleon --lr 0.001 --num_layers 2 --hidden_channels 64 --ours_layers 1 --weight_decay 0.001 --dropout 0.6 --method sgformer --use_graph --use_residual --seed 123 --runs 10 --epochs 500 --patience 200 --display_step 100 --data_dir ../data/ \
    > "$SDIR/chameleon_sgformer_10run.log" 2>&1 &
P7=$!

echo "  Launched 7 SGFormer jobs: PIDs $P1 $P2 $P3 $P4 $P5 $P6 $P7" | tee -a "$SLOG"
echo "  Waiting for all to finish..." | tee -a "$SLOG"
wait $P1 $P2 $P3 $P4 $P5 $P6 $P7
echo "  All done: $(date)" | tee -a "$SLOG"

# ─── Phase 2 summary ───
echo "" | tee -a "$SLOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$SLOG"
echo "METRIC COMPARISON: SGFormer Highest Test vs Final Test" | tee -a "$SLOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$SLOG"
PDIR="$SCRIPT_DIR/final_results/table1_main"
printf "%-12s %-22s %-22s %-22s %-22s\n" "Dataset" "SGF Highest" "SGF Final" "PCGT Highest" "PCGT Final" | tee -a "$SLOG"
for ds in cora citeseer pubmed film squirrel chameleon deezer; do
    sh="—"; sf="—"; ph="—"; pf="—"
    sf_file="$SDIR/${ds}_sgformer_10run.log"
    if [ "$ds" = "deezer" ]; then pf_file="$PDIR/deezer_pcgt_5run.log"; else pf_file="$PDIR/${ds}_pcgt_10run.log"; fi
    [ -f "$sf_file" ] && sh=$(grep "Highest Test:" "$sf_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+')
    [ -f "$sf_file" ] && sf=$(grep "Final Test:" "$sf_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+')
    [ -f "$pf_file" ] && ph=$(grep "Highest Test:" "$pf_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+')
    [ -f "$pf_file" ] && pf=$(grep "Final Test:" "$pf_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+')
    printf "%-12s %-22s %-22s %-22s %-22s\n" "$ds" "$sh" "$sf" "$ph" "$pf" | tee -a "$SLOG"
done

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "============================================================"
echo ""
echo "OUTPUT:"
echo "  Ablation:  $ADIR/"
echo "  SGFormer:  $SDIR/"
echo ""
echo "DECISION after reviewing comparison table:"
echo "  If SGF Highest ≈ their published Table 2 → both use Highest → fair, fix text only"
echo "  If SGF Final ≈ published → PCGT inflated → must re-report with Final Test"
