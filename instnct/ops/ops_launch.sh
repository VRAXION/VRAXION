#!/bin/bash
# Overnight: wait for prune to finish, then train
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "Waiting for prune to finish (checking for python processes)..."
while true; do
    # Check if greedy_prune is still running
    count=$(cmd.exe /c "tasklist" 2>/dev/null | grep -ci python)
    if [ "$count" -eq 0 ]; then
        echo "No Python running — prune finished!"
        break
    fi
    echo "  Still running ($count python processes)... waiting 60s"
    sleep 60
done

echo ""
echo "Starting training from pruned checkpoint..."
python recipes/train_english_1024n_18w.py
