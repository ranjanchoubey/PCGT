#!/bin/bash
# ============================================================================
# Medium-Scale PCGT HP Sweep Script
# Runs experiments with proper logging: timestamped dirs, per-experiment logs
# Usage: bash run_sweep.sh <phase> [dataset]
#   phase: A (quick sweep), B (cora+film fix), C (grid search), baseline
#   dataset: optional, run only specific dataset
# ============================================================================

set -euo pipefail

PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-../data/}"
DEVICE="${DEVICE:-0}"
BASE_LOG_DIR="${BASE_LOG_DIR:-../logs/medium}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Detect unbuffered python
PYU="$PY -u"

run_exp() {
    local name="$1"
    shift
    local log_dir="$BASE_LOG_DIR/$PHASE"
    mkdir -p "$log_dir"
    local logfile="$log_dir/${name}.log"
    echo "[$(date '+%H:%M:%S')] START  $name -> $logfile"
    $PYU main.py "$@" 2>&1 | tee "$logfile"
    echo "[$(date '+%H:%M:%S')] DONE   $name"
    echo ""
}

# ============================================================================
# BASELINES (SGFormer / ours with current configs)
# ============================================================================
run_baselines() {
    local ds="${1:-all}"

    if [[ "$ds" == "all" || "$ds" == "cora" ]]; then
        run_exp "baseline_cora_ours" \
            --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
            --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
            --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
            --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
            --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
            --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "citeseer" ]]; then
        run_exp "baseline_citeseer_ours" \
            --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 \
            --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
            --method ours --ours_layers 1 --use_graph --graph_weight 0.7 \
            --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
            --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
            --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "pubmed" ]]; then
        run_exp "baseline_pubmed_ours" \
            --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 \
            --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
            --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
            --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
            --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
            --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "film" ]]; then
        run_exp "baseline_film_difformer" \
            --backbone gcn --dataset film --lr 0.1 --num_layers 8 \
            --hidden_channels 64 --weight_decay 0.0005 --dropout 0.6 \
            --method difformer --use_graph --graph_weight 0.5 --num_heads 1 \
            --ours_use_residual --ours_use_act \
            --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 \
            --device $DEVICE --runs 10 --epochs 500 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "deezer" ]]; then
        run_exp "baseline_deezer_ours" \
            --backbone gcn --rand_split --dataset deezer-europe --lr 0.01 --num_layers 2 \
            --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
            --method ours --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
            --alpha 0.5 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "squirrel" ]]; then
        run_exp "baseline_squirrel_difformer" \
            --backbone gcn --dataset squirrel --lr 0.001 --num_layers 8 \
            --hidden_channels 64 --weight_decay 5e-4 --dropout 0.3 \
            --method difformer --ours_layers 1 --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
            --alpha 0.5 --device $DEVICE --runs 10 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "chameleon" ]]; then
        run_exp "baseline_chameleon_ours" \
            --backbone gcn --dataset chameleon --lr 0.001 --num_layers 2 \
            --hidden_channels 64 --ours_layers 1 --weight_decay 0.001 --dropout 0.6 \
            --method ours --use_graph --num_heads 1 --ours_use_residual \
            --alpha 0.5 --device $DEVICE --runs 10 --epochs 200 --data_dir "$DATA_DIR"
    fi
}

# ============================================================================
# PHASE A: Quick HP Sweep — try PCGT method on each dataset with K=10,15,20
# Also try graph_weight variations
# ============================================================================
run_phase_a() {
    local ds="${1:-all}"

    # --- CORA --- (currently -0.7% vs SGFormer, need to close gap)
    if [[ "$ds" == "all" || "$ds" == "cora" ]]; then
        for gw in 0.7 0.8 0.9; do
            for K in 5 10 15 20; do
                run_exp "cora_pcgt_gw${gw}_K${K}" \
                    --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
                    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
                    --method pcgt --ours_layers 1 --use_graph --graph_weight $gw \
                    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
                    --num_partitions $K --num_reps 4 \
                    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
                    --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
            done
        done
    fi

    # --- CITESEER --- (currently +0.84%, try to widen)
    if [[ "$ds" == "all" || "$ds" == "citeseer" ]]; then
        for gw in 0.6 0.7 0.8; do
            for K in 5 10 15 20; do
                run_exp "citeseer_pcgt_gw${gw}_K${K}" \
                    --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 \
                    --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
                    --method pcgt --ours_layers 1 --use_graph --graph_weight $gw \
                    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
                    --num_partitions $K --num_reps 4 \
                    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
                    --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
            done
        done
    fi

    # --- PUBMED --- (currently +0.16%, try to widen)
    if [[ "$ds" == "all" || "$ds" == "pubmed" ]]; then
        for gw in 0.7 0.8 0.9; do
            for K in 5 10 15 20; do
                run_exp "pubmed_pcgt_gw${gw}_K${K}" \
                    --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 \
                    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
                    --method pcgt --ours_layers 1 --use_graph --graph_weight $gw \
                    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
                    --num_partitions $K --num_reps 4 \
                    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
                    --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
            done
        done
    fi

    # --- FILM --- (currently -0.21%, need to close gap)
    if [[ "$ds" == "all" || "$ds" == "film" ]]; then
        for gw in 0.4 0.5 0.6 0.7; do
            for K in 5 10 15 20; do
                run_exp "film_pcgt_gw${gw}_K${K}" \
                    --backbone gcn --dataset film --lr 0.1 --num_layers 8 \
                    --hidden_channels 64 --weight_decay 0.0005 --dropout 0.6 \
                    --method pcgt --use_graph --graph_weight $gw --num_heads 1 \
                    --ours_use_residual --ours_use_act \
                    --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 \
                    --num_partitions $K --num_reps 4 \
                    --device $DEVICE --runs 5 --epochs 500 --data_dir "$DATA_DIR"
            done
        done
    fi

    # --- DEEZER --- (currently +0.14%, try to widen)
    if [[ "$ds" == "all" || "$ds" == "deezer" ]]; then
        for gw in 0.7 0.8 0.9; do
            for K in 5 10 15 20; do
                run_exp "deezer_pcgt_gw${gw}_K${K}" \
                    --backbone gcn --rand_split --dataset deezer-europe --lr 0.01 --num_layers 2 \
                    --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
                    --method pcgt --ours_layers 1 --use_graph --graph_weight $gw \
                    --num_heads 1 --ours_use_residual \
                    --alpha 0.5 --num_partitions $K --num_reps 4 \
                    --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
            done
        done
    fi

    # --- SQUIRREL --- (currently +3.34%, maintain/improve)
    if [[ "$ds" == "all" || "$ds" == "squirrel" ]]; then
        for gw in 0.4 0.5 0.6; do
            for K in 5 10 15 20; do
                run_exp "squirrel_pcgt_gw${gw}_K${K}" \
                    --backbone gcn --dataset squirrel --lr 0.001 --num_layers 8 \
                    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.3 \
                    --method pcgt --ours_layers 1 --use_graph --ours_use_act --ours_use_residual \
                    --num_heads 1 --graph_weight $gw \
                    --alpha 0.5 --num_partitions $K --num_reps 4 \
                    --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
            done
        done
    fi

    # --- CHAMELEON --- (currently +3.19%, maintain/improve)
    if [[ "$ds" == "all" || "$ds" == "chameleon" ]]; then
        for gw in 0.4 0.5 0.6 0.7; do
            for K in 5 10 15 20; do
                run_exp "chameleon_pcgt_gw${gw}_K${K}" \
                    --backbone gcn --dataset chameleon --lr 0.001 --num_layers 2 \
                    --hidden_channels 64 --ours_layers 1 --weight_decay 0.001 --dropout 0.6 \
                    --method pcgt --use_graph --num_heads 1 --ours_use_residual \
                    --alpha 0.5 --graph_weight $gw --num_partitions $K --num_reps 4 \
                    --device $DEVICE --runs 5 --epochs 200 --data_dir "$DATA_DIR"
            done
        done
    fi
}

# ============================================================================
# PHASE B: Focus on Cora + Film (the two losses)
# Try more aggressive tuning: lr, hidden, num_reps, alpha
# ============================================================================
run_phase_b() {
    local ds="${1:-all}"

    # --- CORA extra tuning ---
    if [[ "$ds" == "all" || "$ds" == "cora" ]]; then
        # Vary num_reps 
        for reps in 2 4 6 8; do
            run_exp "cora_pcgt_reps${reps}" \
                --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
                --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
                --method pcgt --ours_layers 1 --use_graph --graph_weight 0.8 \
                --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
                --num_partitions 10 --num_reps $reps \
                --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
                --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
        done

        # Vary alpha
        for alpha in 0.3 0.5 0.7 0.9; do
            run_exp "cora_pcgt_alpha${alpha}" \
                --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
                --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
                --method pcgt --ours_layers 1 --use_graph --graph_weight 0.8 \
                --ours_dropout 0.2 --use_residual --alpha $alpha --ours_weight_decay 0.001 \
                --num_partitions 10 --num_reps 4 \
                --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
                --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
        done

        # Try higher hidden_channels
        for hid in 64 128 256; do
            run_exp "cora_pcgt_hid${hid}" \
                --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
                --hidden_channels $hid --weight_decay 5e-4 --dropout 0.5 \
                --method pcgt --ours_layers 1 --use_graph --graph_weight 0.8 \
                --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
                --num_partitions 10 --num_reps 4 \
                --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
                --seed 123 --device $DEVICE --runs 5 --data_dir "$DATA_DIR"
        done
    fi

    # --- FILM extra tuning ---
    if [[ "$ds" == "all" || "$ds" == "film" ]]; then
        # Try pcgt method instead of difformer variant
        for gw in 0.3 0.4 0.5 0.6; do
            for K in 5 10 15; do
                run_exp "film_pcgt_v2_gw${gw}_K${K}" \
                    --backbone gcn --dataset film --lr 0.01 --num_layers 4 \
                    --hidden_channels 64 --weight_decay 0.0005 --dropout 0.5 \
                    --method pcgt --use_graph --graph_weight $gw --num_heads 1 \
                    --ours_use_residual --ours_use_act \
                    --alpha 0.5 --ours_dropout 0.5 --ours_weight_decay 0.0005 \
                    --num_partitions $K --num_reps 4 \
                    --device $DEVICE --runs 5 --epochs 500 --data_dir "$DATA_DIR"
            done
        done

        # Vary num_reps on film
        for reps in 2 4 6 8; do
            run_exp "film_pcgt_reps${reps}" \
                --backbone gcn --dataset film --lr 0.1 --num_layers 8 \
                --hidden_channels 64 --weight_decay 0.0005 --dropout 0.6 \
                --method pcgt --use_graph --graph_weight 0.5 --num_heads 1 \
                --ours_use_residual --ours_use_act \
                --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 \
                --num_partitions 10 --num_reps $reps \
                --device $DEVICE --runs 5 --epochs 500 --data_dir "$DATA_DIR"
        done

        # Vary lr on film
        for lr in 0.001 0.005 0.01 0.05 0.1; do
            run_exp "film_pcgt_lr${lr}" \
                --backbone gcn --dataset film --lr $lr --num_layers 8 \
                --hidden_channels 64 --weight_decay 0.0005 --dropout 0.6 \
                --method pcgt --use_graph --graph_weight 0.5 --num_heads 1 \
                --ours_use_residual --ours_use_act \
                --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 \
                --num_partitions 10 --num_reps 4 \
                --device $DEVICE --runs 5 --epochs 500 --data_dir "$DATA_DIR"
        done
    fi
}

# ============================================================================
# PHASE C: Systematic grid search on best Phase A/B configs
# Run with 10 runs for final numbers
# ============================================================================
run_phase_c() {
    local ds="${1:-all}"
    echo "Phase C: Run this after analyzing Phase A+B results."
    echo "Edit the configs below based on best findings."

    # Template — fill in best configs from A+B analysis
    if [[ "$ds" == "all" || "$ds" == "cora" ]]; then
        run_exp "cora_pcgt_final" \
            --backbone gcn --dataset cora --lr 0.01 --num_layers 4 \
            --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
            --method pcgt --ours_layers 1 --use_graph --graph_weight 0.8 \
            --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
            --num_partitions 10 --num_reps 4 \
            --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
            --seed 123 --device $DEVICE --runs 10 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "citeseer" ]]; then
        run_exp "citeseer_pcgt_final" \
            --backbone gcn --dataset citeseer --lr 0.005 --num_layers 4 \
            --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 \
            --method pcgt --ours_layers 1 --use_graph --graph_weight 0.7 \
            --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
            --num_partitions 10 --num_reps 4 \
            --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
            --seed 123 --device $DEVICE --runs 10 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "pubmed" ]]; then
        run_exp "pubmed_pcgt_final" \
            --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 \
            --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
            --method pcgt --ours_layers 1 --use_graph --graph_weight 0.8 \
            --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
            --num_partitions 10 --num_reps 4 \
            --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
            --seed 123 --device $DEVICE --runs 10 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "film" ]]; then
        run_exp "film_pcgt_final" \
            --backbone gcn --dataset film --lr 0.1 --num_layers 8 \
            --hidden_channels 64 --weight_decay 0.0005 --dropout 0.6 \
            --method pcgt --use_graph --graph_weight 0.5 --num_heads 1 \
            --ours_use_residual --ours_use_act \
            --alpha 0.5 --ours_dropout 0.6 --ours_weight_decay 0.0005 \
            --num_partitions 10 --num_reps 4 \
            --device $DEVICE --runs 10 --epochs 500 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "deezer" ]]; then
        run_exp "deezer_pcgt_final" \
            --backbone gcn --rand_split --dataset deezer-europe --lr 0.01 --num_layers 2 \
            --hidden_channels 96 --weight_decay 5e-05 --dropout 0.4 \
            --method pcgt --ours_layers 1 --use_graph --graph_weight 0.8 \
            --num_heads 1 --ours_use_residual \
            --alpha 0.5 --num_partitions 10 --num_reps 4 \
            --device $DEVICE --runs 10 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "squirrel" ]]; then
        run_exp "squirrel_pcgt_final" \
            --backbone gcn --dataset squirrel --lr 0.001 --num_layers 8 \
            --hidden_channels 64 --weight_decay 5e-4 --dropout 0.3 \
            --method pcgt --ours_layers 1 --use_graph --ours_use_act --ours_use_residual \
            --num_heads 1 --graph_weight 0.5 \
            --alpha 0.5 --num_partitions 10 --num_reps 4 \
            --device $DEVICE --runs 10 --data_dir "$DATA_DIR"
    fi

    if [[ "$ds" == "all" || "$ds" == "chameleon" ]]; then
        run_exp "chameleon_pcgt_final" \
            --backbone gcn --dataset chameleon --lr 0.001 --num_layers 2 \
            --hidden_channels 64 --ours_layers 1 --weight_decay 0.001 --dropout 0.6 \
            --method pcgt --use_graph --num_heads 1 --ours_use_residual \
            --alpha 0.5 --graph_weight 0.5 --num_partitions 10 --num_reps 4 \
            --device $DEVICE --runs 10 --epochs 200 --data_dir "$DATA_DIR"
    fi
}

# ============================================================================
# COLLECT: Parse all logs and summarize results
# ============================================================================
collect_results() {
    echo "======================================================================"
    echo "RESULTS SUMMARY — $(date)"
    echo "======================================================================"
    
    for logdir in "$BASE_LOG_DIR"/*/; do
        [[ -d "$logdir" ]] || continue
        phase=$(basename "$logdir")
        echo ""
        echo "--- Phase: $phase ---"
        printf "  %-45s  %s\n" "EXPERIMENT" "FINAL TEST"
        printf "  %-45s  %s\n" "---------" "----------"
        for logf in "$logdir"/*.log; do
            [[ -f "$logf" ]] || continue
            name=$(basename "$logf" .log)
            # Extract "Final Test: XX.XX ± YY.YY" or "Highest Test: XX.XX ± YY.YY"
            final=$(grep -oP 'Final Test:\s*[\d.]+\s*±\s*[\d.]+' "$logf" 2>/dev/null | tail -1)
            highest=$(grep -oP 'Highest Test:\s*[\d.]+\s*±\s*[\d.]+' "$logf" 2>/dev/null | tail -1)
            if [[ -n "$final" ]]; then
                printf "  %-45s  %s  (%s)\n" "$name" "$final" "$highest"
            else
                # Check if still running
                lines=$(wc -l < "$logf" 2>/dev/null || echo 0)
                last_epoch=$(grep -oP 'Epoch:\s*\d+' "$logf" 2>/dev/null | tail -1 || echo "")
                printf "  %-45s  RUNNING (%s lines, %s)\n" "$name" "$lines" "$last_epoch"
            fi
        done
    done
    echo ""
    echo "======================================================================"
    
    # Also check CSV if it exists
    CSV="results/experiment_results.csv"
    if [[ -f "$CSV" ]]; then
        echo ""
        echo "=== CSV Results (results/experiment_results.csv) ==="
        column -t -s',' "$CSV" 2>/dev/null || cat "$CSV"
    fi
}

# ============================================================================
# MAIN
# ============================================================================
PHASE="${1:-help}"
DATASET="${2:-all}"

case "$PHASE" in
    baseline)
        PHASE="baseline_${TIMESTAMP}"
        echo "=== Running baselines (${DATASET}) ==="
        echo "Logs: $BASE_LOG_DIR/$PHASE/"
        run_baselines "$DATASET"
        ;;
    A|a)
        PHASE="phaseA_${TIMESTAMP}"
        echo "=== Phase A: Quick HP Sweep (${DATASET}) ==="
        echo "Logs: $BASE_LOG_DIR/$PHASE/"
        run_phase_a "$DATASET"
        ;;
    B|b)
        PHASE="phaseB_${TIMESTAMP}"
        echo "=== Phase B: Cora + Film Focus (${DATASET}) ==="
        echo "Logs: $BASE_LOG_DIR/$PHASE/"
        run_phase_b "$DATASET"
        ;;
    C|c)
        PHASE="phaseC_${TIMESTAMP}"
        echo "=== Phase C: Final Grid Search (${DATASET}) ==="
        echo "Logs: $BASE_LOG_DIR/$PHASE/"
        run_phase_c "$DATASET"
        ;;
    collect)
        collect_results
        ;;
    *)
        echo "Usage: bash run_sweep.sh <phase> [dataset]"
        echo ""
        echo "Phases:"
        echo "  baseline  - Run current configs (SGFormer/ours) as reference"
        echo "  A         - Quick HP sweep: graph_weight x K grid per dataset"
        echo "  B         - Focus on Cora + Film: lr, hidden, num_reps, alpha"
        echo "  C         - Final runs with best configs (10 runs)"
        echo "  collect   - Parse all logs and show summary"
        echo ""
        echo "Examples:"
        echo "  bash run_sweep.sh A              # All datasets Phase A"
        echo "  bash run_sweep.sh A cora         # Only Cora Phase A"
        echo "  bash run_sweep.sh B film         # Only Film Phase B"
        echo "  bash run_sweep.sh collect        # Show all results"
        echo ""
        echo "Environment variables:"
        echo "  PY=python DEVICE=0 DATA_DIR=../data/ BASE_LOG_DIR=../logs/medium"
        ;;
esac
