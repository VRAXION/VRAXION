"""Greedy iterative prune: pick random edge, remove it, keep if not worse.
Each removal is tested against the CURRENT (already-pruned) network.
No compound effect risk — we always measure the real state."""
import sys, os, numpy as np, glob, time, random

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
# Prefer pruned checkpoint if exists, otherwise latest step
pruned = os.path.join(CKPT_DIR, "english_1024n_pruned.npz")
if os.path.exists(pruned):
    ckpt = pruned
else:
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "english_1024n_step*.npz")),
                   key=lambda x: int(x.split("step")[1].split(".")[0]))
    ckpt = ckpts[-1]
print(f"Loading: {ckpt}")
d = np.load(ckpt)
rows = list(d['rows']); cols = list(d['cols']); vals = list(d['vals'])
theta = d['theta']; decay = d['decay']
n_start = len(rows)
print(f"Starting edges: {n_start}")

# input_projection/output_projection
np.random.seed(42)
net = SelfWiringGraph(IO)
input_projection = net.input_projection; output_projection = net.output_projection

# Eval data — same seqs every time for consistency
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
with open(DATA, 'rb') as f:
    ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)

N_EVAL_SEQS = 30  # 30 seqs × 199 pred = ~5970 predictions → 0.017% resolution
eval_rng = np.random.RandomState(9999)
eval_seqs = []
for _ in range(N_EVAL_SEQS):
    off = eval_rng.randint(0, len(ALL_DATA) - 200)
    eval_seqs.append(ALL_DATA[off:off+200])

def eval_score(rs, cs, vs):
    rs_a = np.array(rs); cs_a = np.array(cs); vs_a = np.array(vs)
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
                if len(rs_a):
                    np.add.at(raw, cs_a, act[rs_a] * vs_a)
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

# Baseline
print("Computing baseline...")
baseline = eval_score(rows, cols, vals)
print(f"Baseline: {baseline*100:.2f}%")
sys.stdout.flush()

# Greedy prune loop
random.seed(42)
n_removed = 0
n_tested = 0
n_tried_total = 0
current_score = baseline
t0 = time.time()

# Shuffle order — try every edge once
indices = list(range(len(rows)))
random.shuffle(indices)

# Track which edges to remove (mark indices, apply at end of pass)
remove_set = set()

print(f"\nGreedy prune pass over {len(indices)} edges...")
for pass_i, ei in enumerate(indices):
    if ei in remove_set:
        continue  # already scheduled for removal

    # Build current edge lists WITHOUT already-removed AND without this candidate
    test_rows = []; test_cols = []; test_vals = []
    for j in range(len(rows)):
        if j in remove_set or j == ei:
            continue
        test_rows.append(rows[j])
        test_cols.append(cols[j])
        test_vals.append(vals[j])

    new_score = eval_score(test_rows, test_cols, test_vals)
    n_tested += 1

    if new_score >= current_score:
        # Remove it! Network is same or better without this edge
        remove_set.add(ei)
        delta = new_score - current_score
        current_score = new_score
        n_removed += 1
        tag = "BETTER" if delta > 0.0001 else "SAME"
        if n_removed <= 20 or n_removed % 10 == 0:
            print(f"  [{n_tested}/{len(indices)}] REMOVE edge {rows[ei]}->{cols[ei]} "
                  f"(val={vals[ei]:+.1f}): {tag} {current_score*100:.2f}% "
                  f"(removed {n_removed})")
            sys.stdout.flush()
    else:
        if n_tested <= 5 or n_tested % 100 == 0:
            print(f"  [{n_tested}/{len(indices)}] KEEP edge {rows[ei]}->{cols[ei]} "
                  f"(val={vals[ei]:+.1f}): drop to {new_score*100:.2f}%")
            sys.stdout.flush()

elapsed = time.time() - t0

# Build final edge lists
final_rows = []; final_cols = []; final_vals = []
for j in range(len(rows)):
    if j not in remove_set:
        final_rows.append(rows[j])
        final_cols.append(cols[j])
        final_vals.append(vals[j])

# Final verification
final_score = eval_score(final_rows, final_cols, final_vals)

print(f"\n{'='*70}")
print(f"GREEDY PRUNE COMPLETE")
print(f"{'='*70}")
print(f"  Started:  {n_start} edges, {baseline*100:.2f}%")
print(f"  Tested:   {n_tested} edges")
print(f"  Removed:  {n_removed} edges ({n_removed/n_start*100:.1f}%)")
print(f"  Remaining: {len(final_rows)} edges")
print(f"  Final:    {final_score*100:.2f}% (delta={final_score*100 - baseline*100:+.2f}%)")
print(f"  Time:     {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Save
save_path = os.path.join(CKPT_DIR, "english_1024n_pruned.npz")
np.savez(save_path,
         rows=np.array(final_rows), cols=np.array(final_cols),
         vals=np.array(final_vals, dtype=np.float32),
         theta=theta, decay=decay)
print(f"\n  SAVED: {save_path}")
print(f"  {len(final_rows)} edges (was {n_start})")
