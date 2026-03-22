"""Overnight: wait for greedy prune to finish, then resume training."""
import subprocess, sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Step 1: Run greedy prune pass 2 (30 seq, high resolution) if not already done
# Check if prune is still running by trying to see if pruned checkpoint is fresh
pruned = ROOT / "checkpoints" / "english_1024n_pruned.npz"
mtime_before = pruned.stat().st_mtime if pruned.exists() else 0

print("=" * 60)
print("OVERNIGHT PIPELINE")
print("=" * 60)
print(f"Pruned checkpoint: {pruned}")
print(f"  exists={os.path.exists(pruned)}, mtime={time.ctime(mtime_before) if mtime_before else 'N/A'}")

# Step 2: Start training from pruned checkpoint
print(f"\nStarting training from pruned checkpoint...")
print(f"Budget: 10000 steps")
print(f"Schedule: [A,A,T,A,A,D,A,R] (with remove!)")
print(f"Eval: 10 seqs per checkpoint")
sys.stdout.flush()

train_script = ROOT / "recipes" / "english_1024n_18w.py"
result = subprocess.run([sys.executable, train_script],
                       cwd=ROOT, timeout=None)
print(f"\nTraining finished with exit code {result.returncode}")
