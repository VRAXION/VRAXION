"""Quick inference: load checkpoint, generate text byte-by-byte."""
import sys, os, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

IO = 256
H = IO * 4  # 1024
bp = make_bp(IO)
pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)

# Load latest checkpoint
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
# Find latest 1024n checkpoint
import glob
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "english_1024n_step*.npz")),
               key=lambda x: int(x.split("step")[1].split(".")[0]))
if not ckpts:
    print("No checkpoint found!"); sys.exit(1)
ckpt = ckpts[-1]  # latest by step number
print(f"Loading: {ckpt}")

d = np.load(ckpt, allow_pickle=True)
rows = d['rows']; cols = d['cols']; vals = d['vals']
theta = d['theta']; decay = d['decay']
mask = np.zeros((H, H), dtype=np.float32)
mask[rows, cols] = vals
n_edges = len(rows)
print(f"Edges: {n_edges}, theta mean={theta.mean():.3f}, decay mean={decay.mean():.3f}")

# Recreate W_in, W_out with same seed as training
from graph import SelfWiringGraph
SelfWiringGraph.NV_RATIO = 4
np.random.seed(42)
net = SelfWiringGraph(IO)
W_in = net.W_in
W_out = net.W_out

rs, cs = np.where(mask != 0)
sp_vals = mask[rs, cs]
ret = 1.0 - decay

def generate(prompt_bytes, n_generate=200, temperature=1.0):
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    # Feed prompt
    for i in range(len(prompt_bytes) - 1):
        act = state.copy()
        for t in range(6):
            if t == 0:
                act = act + bp[prompt_bytes[i]] @ W_in
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()

    # Generate
    output = list(prompt_bytes)
    current_byte = prompt_bytes[-1]

    for _ in range(n_generate):
        act = state.copy()
        for t in range(6):
            if t == 0:
                act = act + bp[current_byte] @ W_in
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()

        out = charge @ W_out
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T

        # Temperature sampling
        if temperature < 0.01:
            current_byte = np.argmax(sims)
        else:
            sims_t = sims / temperature
            e = np.exp(sims_t - sims_t.max())
            probs = e / e.sum()
            current_byte = np.random.choice(256, p=probs)
        output.append(current_byte)

    return bytes(output)

# Test prompts
prompts = [
    b"The ",
    b"In the beginning ",
    b"Science is ",
    b"Once upon a time ",
    b"Hello world",
]

print("\n" + "=" * 70)
print("GREEDY (temperature=0)")
print("=" * 70)
for p in prompts:
    result = generate(list(p), n_generate=100, temperature=0.0)
    text = result.decode('ascii', errors='backslashreplace')
    print(f"\n>>> {p.decode()}")
    print(f"    {text}")

print("\n" + "=" * 70)
print("SAMPLED (temperature=0.8)")
print("=" * 70)
for p in prompts:
    result = generate(list(p), n_generate=100, temperature=0.8)
    text = result.decode('ascii', errors='backslashreplace')
    print(f"\n>>> {p.decode()}")
    print(f"    {text}")

# Top predictions for common bytes
print("\n" + "=" * 70)
print("TOP-5 PREDICTIONS after single byte")
print("=" * 70)
for ch in [ord(' '), ord('e'), ord('t'), ord('\n'), ord('.')]:
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    act = state.copy()
    for t in range(6):
        if t == 0:
            act = act + bp[ch] @ W_in
        raw = np.zeros(H, dtype=np.float32)
        if len(rs):
            np.add.at(raw, cs, act[rs] * sp_vals)
        charge += raw; charge *= ret
        act = np.maximum(charge - theta, 0.0)
        charge = np.clip(charge, -1.0, 1.0)
    out = charge @ W_out
    out_n = out / (np.linalg.norm(out) + 1e-8)
    sims = out_n @ pat_norm.T
    e = np.exp(sims - sims.max())
    probs = e / e.sum()
    top5 = np.argsort(probs)[-5:][::-1]
    ch_str = repr(chr(ch))
    preds = ", ".join([f"{chr(b) if 32<=b<127 else f'x{b:02x}'}({probs[b]*100:.1f}%)" for b in top5])
    print(f"  After {ch_str:>5}: {preds}")
