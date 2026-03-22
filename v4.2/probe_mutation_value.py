"""Probe: which mutation type gives the most value right now?
Runs 100 proposals of each type, measures accept rate + avg delta."""
import sys, os, numpy as np, glob, time, random
from multiprocessing import Pool

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

# Load data
from lib.data import load_fineweb_bytes, resolve_fineweb_path
DATA = resolve_fineweb_path()
ALL_DATA = load_fineweb_bytes()

# Load latest checkpoint
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "english_1024n_step*.npz")),
               key=lambda x: int(x.split("step")[1].split(".")[0]))
ckpt = ckpts[-1]
print(f"Loading: {ckpt}")
d = np.load(ckpt)
rows = d['rows']; cols = d['cols']; vals = d['vals']
theta = d['theta']; decay = d['decay']
mask = np.zeros((H, H), dtype=np.float32)
mask[rows, cols] = vals
n_edges = len(rows)

np.random.seed(42)
net = SelfWiringGraph(IO)
input_projection = net.input_projection; output_projection = net.output_projection

_bp = bp
_all_data = ALL_DATA
_seq_len = 200
_n_train = 5

pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)

def _eval_on_seqs(mask, theta, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        correct = 0; prob_sum = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(6):
                if t == 0:
                    act = act + bp[text_bytes[i]] @ input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.clip(charge, -1.0, 1.0)
            state = act.copy()
            out = charge @ output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            probs = e / e.sum()
            target = text_bytes[i+1]
            if np.argmax(probs) == target: correct += 1
            prob_sum += probs[target]; n += 1
        acc = correct/n if n else 0
        avg_p = prob_sum/n if n else 0
        total += 0.5*acc + 0.5*avg_p
    return total / len(seqs)

alive_set = set(zip(rows.tolist(), cols.tolist()))

N_TRIALS = 200  # per type
print(f"\nEdges: {n_edges}, theta mean={theta.mean():.3f}, decay mean={decay.mean():.3f}")
print(f"Running {N_TRIALS} trials per mutation type...\n")
sys.stdout.flush()

results = {}

for ptype in ['add', 'theta', 'decay', 'remove']:
    deltas = []
    accepts = 0
    t0 = time.time()

    for trial in range(N_TRIALS):
        rng = random.Random(7777 + trial)
        np_rng = np.random.RandomState(7777 + trial)

        # Sample sequences
        seqs = []
        for _ in range(5):
            off = np_rng.randint(0, len(ALL_DATA) - 200)
            seqs.append(ALL_DATA[off:off+200])

        old_score = _eval_on_seqs(mask, theta, decay, seqs)

        if ptype == 'add':
            r = rng.randint(0, H-1)
            c = rng.randint(0, H-1)
            if r == c or (r, c) in alive_set:
                continue
            val = 0.6 if rng.random() < 0.5 else -0.6
            new_mask = mask.copy()
            new_mask[r, c] = val
            new_score = _eval_on_seqs(new_mask, theta, decay, seqs)
            delta = new_score - old_score
            deltas.append(delta)
            if delta > 0: accepts += 1

        elif ptype == 'theta':
            idx = rng.randint(0, H-1)
            new_theta = theta.copy()
            new_theta[idx] = rng.random()
            new_score = _eval_on_seqs(mask, new_theta, decay, seqs)
            delta = new_score - old_score
            deltas.append(delta)
            if delta > 0: accepts += 1

        elif ptype == 'decay':
            idx = rng.randint(0, H-1)
            new_decay = decay.copy()
            new_decay[idx] = rng.uniform(0.01, 0.5)
            new_score = _eval_on_seqs(mask, theta, new_decay, seqs)
            delta = new_score - old_score
            deltas.append(delta)
            if delta > 0: accepts += 1

        elif ptype == 'remove':
            if n_edges == 0: continue
            edge_idx = rng.randint(0, n_edges - 1)
            r, c = rows[edge_idx], cols[edge_idx]
            new_mask = mask.copy()
            new_mask[r, c] = 0.0
            new_score = _eval_on_seqs(new_mask, theta, decay, seqs)
            delta = new_score - old_score
            deltas.append(delta)
            if delta > 0: accepts += 1

        if (trial + 1) % 50 == 0:
            print(f"  {ptype}: {trial+1}/{N_TRIALS}...")
            sys.stdout.flush()

    elapsed = time.time() - t0
    deltas = np.array(deltas)
    pos_deltas = deltas[deltas > 0]

    results[ptype] = {
        'n_trials': len(deltas),
        'accepts': accepts,
        'accept_rate': accepts / len(deltas) * 100 if len(deltas) else 0,
        'mean_delta': deltas.mean() if len(deltas) else 0,
        'mean_pos_delta': pos_deltas.mean() if len(pos_deltas) else 0,
        'max_delta': deltas.max() if len(deltas) else 0,
        'time': elapsed
    }

    print(f"  {ptype}: {accepts}/{len(deltas)} accepted ({results[ptype]['accept_rate']:.1f}%), "
          f"avg_delta={results[ptype]['mean_delta']*100:.3f}%, "
          f"avg_pos={results[ptype]['mean_pos_delta']*100:.3f}%, "
          f"max={results[ptype]['max_delta']*100:.3f}% "
          f"({elapsed:.0f}s)")
    sys.stdout.flush()

# Summary
print(f"\n{'='*70}")
print(f"MUTATION VALUE ANALYSIS @ {n_edges} edges")
print(f"{'='*70}")
print(f"{'Type':>8} {'Trials':>7} {'Accept%':>9} {'AvgDelta':>10} {'AvgPos':>10} {'MaxDelta':>10} {'Value/s':>10}")
for ptype in ['add', 'theta', 'decay', 'remove']:
    r = results[ptype]
    # Value per second = accept_rate * avg_positive_delta / time_per_trial
    val_per_s = r['accept_rate']/100 * r['mean_pos_delta'] * r['n_trials'] / r['time'] if r['time'] > 0 else 0
    print(f"{ptype:>8} {r['n_trials']:>7} {r['accept_rate']:>8.1f}% {r['mean_delta']*100:>+9.4f}% "
          f"{r['mean_pos_delta']*100:>9.4f}% {r['max_delta']*100:>9.4f}% {val_per_s*100:>9.5f}%")

print(f"\n--- INTERPRETATION ---")
best = max(['add', 'theta', 'decay', 'remove'],
           key=lambda t: results[t]['accept_rate'] * results[t]['mean_pos_delta'])
print(f"  Best mutation type: {best}")
print(f"  Remove accept rate shows how many current edges are harmful")
if results['remove']['accept_rate'] > 30:
    print(f"  WARNING: {results['remove']['accept_rate']:.0f}% of edges are harmful! Need pruning.")
