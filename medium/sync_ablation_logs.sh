#!/bin/bash
# Sync logs from running ablation to experiments/logs/phase4_metis_ablation/
# Run this after run_metis_ablation.sh finishes
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_DIR/logs/metis_ablation"
DST="$REPO_DIR/experiments/logs/phase4_metis_ablation"
mkdir -p "$DST"
cp -v "$SRC"/*.log "$DST"/
echo ""
echo "=== Phase 4 Results Summary ==="
for f in "$DST"/*.log; do
    name=$(basename "$f" .log)
    final=$(grep "Highest Test:" "$f" | tail -1)
    intra=$(grep "Intra-partition" "$f" | head -1)
    if [ -n "$final" ]; then
        echo "  $name: $final  |  $intra"
    fi
done
