"""Crystallize analysis: measure importance of each edge by removal impact."""
import sys, os, numpy as np, glob, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from graph import SelfWiringGraph

IO = 256; H = IO * 4
SelfWiringGraph.NV_RATIO = 4

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
net = SelfWiringGraph(IO)
input_projection = net.input_projection; output_projection = net.output_projection

# Load eval data
from lib.data import load_fineweb_bytes, resolve_fineweb_path
DATA = resolve_fineweb_path()
ALL_DATA = load_fineweb_bytes()
DATA_LEN = len(ALL_DATA)

eval_rng = np.random.RandomState(9999)
eval_seqs = []
for _ in range(30):  # 30 eval seqs → ~5970 preds, 0.017% resolution
    off = eval_rng.randint(0, DATA_LEN - 200)
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
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            if np.argmax(sims) == text_bytes[i+1]:
                total_correct += 1
            total_n += 1
    return total_correct / total_n if total_n else 0

# Baseline score with all edges
print("Computing baseline...")
t0 = time.time()
baseline = eval_score(rows, cols, vals, theta, decay)
dt = time.time() - t0
print(f"Baseline: {baseline*100:.2f}% ({dt:.1f}s)")
print(f"Estimated total time: {dt * n_edges:.0f}s ({dt * n_edges / 60:.1f} min)")

# Test each edge removal
print(f"\nTesting {n_edges} edge removals...")
sys.stdout.flush()

importance = np.zeros(n_edges, dtype=np.float32)
t0 = time.time()

for ei in range(n_edges):
    # Remove edge ei
    mask_rs = np.delete(rows, ei)
    mask_cs = np.delete(cols, ei)
    mask_vs = np.delete(vals, ei)

    score = eval_score(mask_rs, mask_cs, mask_vs, theta, decay)
    importance[ei] = baseline - score  # positive = edge was helpful

    if (ei + 1) % 100 == 0 or ei < 10:
        elapsed = time.time() - t0
        eta = elapsed / (ei + 1) * (n_edges - ei - 1)
        print(f"  [{ei+1}/{n_edges}] edge ({rows[ei]}->{cols[ei]} val={vals[ei]:.1f}): "
              f"drop={importance[ei]*100:+.2f}% | {elapsed:.0f}s elapsed, ~{eta:.0f}s left")
        sys.stdout.flush()

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Analysis
print(f"\n{'='*70}")
print(f"CRYSTALLIZE ANALYSIS — {n_edges} edges")
print(f"{'='*70}")
print(f"Baseline accuracy: {baseline*100:.2f}%")

# Sort by importance
order = np.argsort(importance)

print(f"\n--- IMPORTANCE DISTRIBUTION ---")
bins = [-0.05, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.05, 0.1]
for i in range(len(bins)-1):
    count = np.sum((importance >= bins[i]) & (importance < bins[i+1]))
    bar = '#' * min(count, 80)
    print(f"  [{bins[i]*100:+5.1f}% to {bins[i+1]*100:+5.1f}%): {count:>5} {bar}")

harmful = np.sum(importance < -0.001)
useless = np.sum((importance >= -0.001) & (importance <= 0.001))
helpful = np.sum(importance > 0.001)
critical = np.sum(importance > 0.01)

print(f"\n--- SUMMARY ---")
print(f"  Harmful (removing HELPS):  {harmful} ({harmful/n_edges*100:.1f}%)")
print(f"  Useless (no effect):       {useless} ({useless/n_edges*100:.1f}%)")
print(f"  Helpful (removing HURTS):  {helpful} ({helpful/n_edges*100:.1f}%)")
print(f"  Critical (>1% drop):       {critical} ({critical/n_edges*100:.1f}%)")

print(f"\n--- TOP 10 MOST CRITICAL EDGES ---")
for idx in order[-10:][::-1]:
    print(f"  {rows[idx]:>4} -> {cols[idx]:>4} (val={vals[idx]:+.1f}): "
          f"removing drops {importance[idx]*100:+.2f}% | "
          f"theta[src]={theta[rows[idx]]:.3f} theta[dst]={theta[cols[idx]]:.3f}")

print(f"\n--- TOP 10 MOST HARMFUL EDGES (should be removed!) ---")
for idx in order[:10]:
    print(f"  {rows[idx]:>4} -> {cols[idx]:>4} (val={vals[idx]:+.1f}): "
          f"removing GAINS {-importance[idx]*100:+.2f}% | "
          f"theta[src]={theta[rows[idx]]:.3f} theta[dst]={theta[cols[idx]]:.3f}")

# Can we remove bottom 50% edges?
print(f"\n--- PRUNING SIMULATION ---")
for pct in [10, 25, 50, 75]:
    n_remove = int(n_edges * pct / 100)
    keep = order[n_remove:]  # keep the most important
    pruned_score = eval_score(rows[keep], cols[keep], vals[keep], theta, decay)
    print(f"  Remove {pct}% ({n_remove} edges): {pruned_score*100:.2f}% "
          f"(delta={pruned_score*100 - baseline*100:+.2f}%)")
