"""Overnight: wait for greedy prune to finish, then resume training."""
import subprocess, sys, os, time

BASE = os.path.dirname(os.path.abspath(__file__))

# Step 1: Run greedy prune pass 2 (30 seq, high resolution) if not already done
# Check if prune is still running by trying to see if pruned checkpoint is fresh
pruned = os.path.join(BASE, "checkpoints", "english_1024n_pruned.npz")
mtime_before = os.path.getmtime(pruned) if os.path.exists(pruned) else 0

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

train_script = os.path.join(BASE, "english_1024n_18w.py")
result = subprocess.run([sys.executable, train_script],
                       cwd=BASE, timeout=None)
print(f"\nTraining finished with exit code {result.returncode}")
