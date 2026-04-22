#!/bin/bash
# Run log-cosine fitness separately (uses eval_smooth_proj which now has log-space)
# We need a second binary because the main one uses temp annealing
# Quick hack: copy the binary and modify at runtime...
# Actually just wait for temp annealing to finish, then run log-cosine
echo "Waiting for temp annealing to finish, then running log-cosine..."
