#!/bin/bash
# Overnight: wait for prune to finish, then train
cd "S:/AI/work/VRAXION_DEV/v4.2"

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
python english_1024n_18w.py
