"""Apply conservative prune using saved crystallize data.
Only removes edges where removal measurably HELPS (importance < -0.001).
Keeps all 0.00% edges — individually tiny but collectively important!

Usage: python apply_prune_1024n.py [threshold]
  threshold: minimum negative importance to remove (default -0.001)
  Example: python apply_prune_1024n.py -0.0005  (more aggressive)
"""
import sys, os, numpy as np, glob

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

IO = 256; H = IO * 4
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# Load the crystallize importance data (saved by bulk_prune)
importance_file = os.path.join(CKPT_DIR, "english_1024n_importance.npy")
if not os.path.exists(importance_file):
    print(f"ERROR: No importance data found at {importance_file}")
    print("Run bulk_prune_1024n.py first!")
    sys.exit(1)

importance = np.load(importance_file)

# Load checkpoint
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "english_1024n_step*.npz")),
               key=lambda x: int(x.split("step")[1].split(".")[0]))
ckpt = ckpts[-1]
print(f"Loading: {ckpt}")
d = np.load(ckpt)
rows = d['rows']; cols = d['cols']; vals = d['vals']
theta = d['theta']; decay = d['decay']
n_edges = len(rows)

if len(importance) != n_edges:
    print(f"ERROR: importance has {len(importance)} entries but checkpoint has {n_edges} edges!")
    print("The importance data doesn't match this checkpoint.")
    sys.exit(1)

# Threshold
threshold = float(sys.argv[1]) if len(sys.argv) > 1 else -0.001
print(f"Prune threshold: {threshold} (removing edges with importance < {threshold})")

# Apply prune
harmful_mask = importance < threshold
keep_idx = np.where(~harmful_mask)[0]
n_removed = np.sum(harmful_mask)

print(f"\nEdges: {n_edges}")
print(f"Harmful (removing): {n_removed}")
print(f"Keeping: {len(keep_idx)}")

# Distribution
print(f"\nImportance distribution:")
print(f"  < -0.01 (very harmful):  {np.sum(importance < -0.01)}")
print(f"  [-0.01, -0.001):         {np.sum((importance >= -0.01) & (importance < -0.001))}")
print(f"  [-0.001, +0.001] (zero): {np.sum((importance >= -0.001) & (importance <= 0.001))}")
print(f"  (+0.001, +0.01]:         {np.sum((importance > 0.001) & (importance <= 0.01))}")
print(f"  > +0.01 (critical):      {np.sum(importance > 0.01)}")

# Save
pruned_rows = rows[keep_idx]
pruned_cols = cols[keep_idx]
pruned_vals = vals[keep_idx]

save_path = os.path.join(CKPT_DIR, "english_1024n_pruned.npz")
np.savez(save_path,
         rows=pruned_rows, cols=pruned_cols, vals=pruned_vals,
         theta=theta, decay=decay)
print(f"\nSAVED: {save_path}")
print(f"  {len(pruned_rows)} edges (removed {n_removed})")

