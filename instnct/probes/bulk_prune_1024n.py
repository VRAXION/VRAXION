"""Bulk crystallize + prune: test every edge, remove all harmful ones, save clean checkpoint."""
import sys, os, numpy as np, glob, time

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

IO = 256; H = IO * 4
def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

bp = make_bp(IO)
pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "english_1024n_step*.npz")),
               key=lambda x: int(x.split("step")[1].split(".")[0]))
ckpt = ckpts[-1]
print(f"Loading: {ckpt}")
d = np.load(ckpt)
rows = d['rows']; cols = d['cols']; vals = d['vals']
theta = d['theta']; decay = d['decay']
n_edges = len(rows)
print(f"Edges: {n_edges}")

# Recreate input_projection/output_projection
np.random.seed(42)
net = SelfWiringGraph(IO, hidden_ratio=4)
input_projection = net.input_projection; output_projection = net.output_projection

# Load eval data
from lib.data import load_fineweb_bytes, resolve_fineweb_path
DATA = resolve_fineweb_path()
ALL_DATA = load_fineweb_bytes()

# Use more eval seqs for stable measurement
eval_rng = np.random.RandomState(9999)
eval_seqs = []
for _ in range(30):  # 30 seqs → ~5970 preds, 0.017% resolution
    off = eval_rng.randint(0, len(ALL_DATA) - 200)
    eval_seqs.append(ALL_DATA[off:off+200])

def eval_score(rs, cs, vs, theta, decay):
    ret = 1.0 - decay
    total_correct = 0; total_n = 0
    for text_bytes in eval_seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(6):
                if t == 0:
                    act = act + bp[text_bytes[i]] @ input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * vs)
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            if np.argmax(sims) == text_bytes[i+1]:
                total_correct += 1
            total_n += 1
    return total_correct / total_n if total_n else 0

# Baseline
print("Computing baseline...")
t0 = time.time()
baseline = eval_score(rows, cols, vals, theta, decay)
dt = time.time() - t0
print(f"Baseline: {baseline*100:.2f}% ({dt:.1f}s per eval)")
print(f"Estimated crystallize time: {dt * n_edges:.0f}s ({dt * n_edges / 60:.1f} min)")
sys.stdout.flush()

# Test each edge
print(f"\nCrystallizing {n_edges} edges...")
importance = np.zeros(n_edges, dtype=np.float32)
t0 = time.time()

for ei in range(n_edges):
    mask_rs = np.delete(rows, ei)
    mask_cs = np.delete(cols, ei)
    mask_vs = np.delete(vals, ei)
    score = eval_score(mask_rs, mask_cs, mask_vs, theta, decay)
    importance[ei] = baseline - score  # positive = edge was helpful

    if (ei + 1) % 50 == 0 or ei < 5:
        elapsed = time.time() - t0
        eta = elapsed / (ei + 1) * (n_edges - ei - 1)
        print(f"  [{ei+1}/{n_edges}] drop={importance[ei]*100:+.2f}% | {elapsed:.0f}s elapsed, ~{eta:.0f}s left")
        sys.stdout.flush()

elapsed = time.time() - t0
print(f"\nCrystallize done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Save raw importance data for later re-pruning with different thresholds
imp_path = os.path.join(CKPT_DIR, "english_1024n_importance.npy")
np.save(imp_path, importance)
print(f"Saved importance data: {imp_path}")

# Analysis
harmful = importance < -0.001  # removing HELPS
useless = (importance >= -0.001) & (importance <= 0.001)
helpful = importance > 0.001
critical = importance > 0.01

n_harmful = np.sum(harmful)
n_useless = np.sum(useless)
n_helpful = np.sum(helpful)
n_critical = np.sum(critical)

print(f"\n{'='*70}")
print(f"CRYSTALLIZE ANALYSIS — {n_edges} edges")
print(f"{'='*70}")
print(f"Baseline accuracy: {baseline*100:.2f}%")
print(f"  Harmful (removing HELPS):  {n_harmful} ({n_harmful/n_edges*100:.1f}%)")
print(f"  Useless (no effect):       {n_useless} ({n_useless/n_edges*100:.1f}%)")
print(f"  Helpful (removing HURTS):  {n_helpful} ({n_helpful/n_edges*100:.1f}%)")
print(f"  Critical (>1% drop):       {n_critical} ({n_critical/n_edges*100:.1f}%)")
sys.stdout.flush()

# PRUNE: ONLY remove clearly harmful edges
# Strategy: keep everything EXCEPT edges where removal measurably HELPS (< -0.001)
# The 0.00% edges are NOT removed — individually they look useless but collectively they matter!
harmful_mask = importance < -0.001  # these actively hurt the network
keep_idx = np.where(~harmful_mask)[0]
n_removed = np.sum(harmful_mask)

print(f"\n--- CONSERVATIVE BULK PRUNE ---")
print(f"  Removing {n_removed} HARMFUL edges only (importance < -0.001)")
print(f"  Keeping {len(keep_idx)} edges (including {np.sum(useless)} 'useless' ones — safety!)")

# Verify pruned score
pruned_rows = rows[keep_idx]
pruned_cols = cols[keep_idx]
pruned_vals = vals[keep_idx]

print("  Verifying pruned network...")
pruned_score = eval_score(pruned_rows, pruned_cols, pruned_vals, theta, decay)
print(f"  Pruned accuracy: {pruned_score*100:.2f}% (was {baseline*100:.2f}%, delta={pruned_score*100 - baseline*100:+.2f}%)")

# Also try keeping only clearly helpful (importance > 0.001)
strict_idx = np.where(importance > 0.001)[0]
if len(strict_idx) < len(keep_idx):
    strict_score = eval_score(rows[strict_idx], cols[strict_idx], vals[strict_idx], theta, decay)
    print(f"  Strict prune ({len(strict_idx)} edges): {strict_score*100:.2f}% (delta={strict_score*100 - baseline*100:+.2f}%)")

# Save pruned checkpoint
save_path = os.path.join(CKPT_DIR, "english_1024n_pruned.npz")
np.savez(save_path,
         rows=pruned_rows, cols=pruned_cols, vals=pruned_vals,
         theta=theta, decay=decay)
print(f"\n  SAVED: {save_path}")
print(f"  {len(pruned_rows)} edges (removed {n_removed})")

# Top 10 removed edges (most harmful)
order = np.argsort(importance)
print(f"\n--- TOP 10 REMOVED (most harmful) ---")
for idx in order[:10]:
    if importance[idx] <= 0:
        print(f"  {rows[idx]:>4} -> {cols[idx]:>4} (val={vals[idx]:+.1f}): "
              f"removing GAINS {-importance[idx]*100:+.2f}%")

print(f"\n--- TOP 10 KEPT (most critical) ---")
for idx in order[-10:][::-1]:
    print(f"  {rows[idx]:>4} -> {cols[idx]:>4} (val={vals[idx]:+.1f}): "
          f"removing drops {importance[idx]*100:+.2f}%")

print(f"\nDONE. Resume training from: {save_path}")


