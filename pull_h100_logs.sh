#!/bin/bash
# ============================================================================
# Pull H100 logs to local Mac
# Usage: bash pull_h100_logs.sh <h100_ssh_host>
# Example: bash pull_h100_logs.sh user@h100.lightning.ai
# ============================================================================

H100_HOST="${1:?Usage: bash pull_h100_logs.sh <ssh_host>}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_LOG_DIR="$REPO_DIR/logs/h100"

mkdir -p "$LOCAL_LOG_DIR"

echo "Pulling logs from $H100_HOST..."
scp "$H100_HOST:~/PCGT/logs/h100/*.log" "$LOCAL_LOG_DIR/" 2>/dev/null || \
scp "$H100_HOST:~/thesis/PCGT/logs/h100/*.log" "$LOCAL_LOG_DIR/" 2>/dev/null || \
echo "ERROR: Could not find logs. Check the remote path."

echo ""
echo "Local logs:"
for f in "$LOCAL_LOG_DIR"/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    result=$(grep -o "[0-9]* runs:.*" "$f" 2>/dev/null | tail -1)
    if [ -n "$result" ]; then
        printf "  %-25s  DONE  %s\n" "$name" "$result"
    else
        runs=$(grep -c "^Run [0-9]*:" "$f" 2>/dev/null || echo 0)
        printf "  %-25s  %s runs completed\n" "$name" "$runs"
    fi
done
