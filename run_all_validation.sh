#!/bin/bash
# Master validation script — runs ALL experiments sequentially on GPU
# Launched in background via screen; logs to ~/PCGT/logs/h100_validation/

set -uo pipefail
cd ~/PCGT

source venv/bin/activate
export DEVICE=0

echo "=========================================="
echo "PCGT FULL H100 VALIDATION"
echo "Started: $(date)"
echo "=========================================="

# Phase 1: PCGT experiments (high + medium + low priority)
echo "[$(date)] Starting Phase 1+2+3: All PCGT configs..."
bash doubts/validate_h100.sh all 2>&1 | tee logs/h100_validation/MASTER_pcgt.log

echo ""
echo "[$(date)] Starting SGFormer baselines..."
bash doubts/validate_h100.sh sgformer 2>&1 | tee logs/h100_validation/MASTER_sgformer.log

echo ""
echo "[$(date)] Collecting final summary..."
bash doubts/validate_h100.sh results

echo ""
echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
