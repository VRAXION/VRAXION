"""Quick inference: load checkpoint, generate text byte-by-byte."""
import sys, os, random
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "english_768n_step2500.npz")

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def load_net_with_projections(ckpt_path):
    """Load checkpoint into a net with correct input_projection/output_projection (same seed as training)."""
    random.seed(42); np.random.seed(42)
    d = np.load(ckpt_path)
    V = int(d['V'])
    net = SelfWiringGraph(V)  # rebuilds input_projection/output_projection with same RNG state
    # Overwrite mask/theta/decay from checkpoint
    net.mask[:] = 0
    rows, cols, vals = d['rows'], d['cols'], d['vals']
    net.mask[rows, cols] = vals
    net.alive = list(zip(rows.tolist(), cols.tolist()))
    net.alive_set = set(net.alive)
    net._sync_sparse_idx()
    if 'theta' in d:
        net.theta = np.array(d['theta'], dtype=np.float32)
    if 'decay' in d:
        net.decay = np.array(d['decay'], dtype=np.float32)
    net.state *= 0; net.charge *= 0
    return net

def generate(net, bp, prompt_bytes, gen_len=200, temperature=1.0, top_k=0):
    """Generate text autoregressively."""
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    H = net.H
    input_projection, output_projection = net.input_projection, net.output_projection
    theta, decay = net.theta, net.decay
    ret = 1.0 - decay

    # Sparse indices
    rs, cs = np.where(net.mask != 0)
    sp_vals = net.mask[rs, cs]

    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)

    output = list(prompt_bytes)

    # Feed prompt (teacher forcing)
    for i, b in enumerate(prompt_bytes):
        act = state.copy()
        for t in range(6):
            if t == 0:
                act = act + bp[b] @ input_projection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.clip(charge, -1.0, 1.0)
        state = act.copy()

    # Generate
    last_byte = prompt_bytes[-1] if prompt_bytes else ord(' ')
    for step in range(gen_len):
        act = state.copy()
        for t in range(6):
            if t == 0:
                act = act + bp[last_byte] @ input_projection
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

        # Temperature + optional top-k
        logits = sims / max(temperature, 1e-8)
        if top_k > 0:
            idx_sort = np.argsort(logits)[::-1]
            logits[idx_sort[top_k:]] = -1e9
        e = np.exp(logits - logits.max())
        probs = e / e.sum()

        chosen = np.random.choice(256, p=probs)
        output.append(chosen)
        last_byte = chosen

    return bytes(output)

if __name__ == "__main__":
    print(f"Loading {CKPT}...")
    net = load_net_with_projections(CKPT)
    bp = make_bp(net.V)
    edges = net.count_connections()
    print(f"Loaded: {net.H} neurons, {edges} edges, theta_mean={net.theta.mean():.3f}, decay_mean={net.decay.mean():.3f}")

    prompts = [
        b"The ",
        b"In the beginning ",
        b"Once upon a time ",
        b"Science is ",
        b"The cat sat on ",
    ]

    for temp in [0.5, 1.0]:
        print(f"\n{'='*60}")
        print(f"Temperature = {temp}")
        print(f"{'='*60}")
        for p in prompts:
            result = generate(net, bp, list(p), gen_len=150, temperature=temp, top_k=10)
            # Show printable ASCII only
            text = ''.join(chr(b) if 32 <= b < 127 or b == 10 else '.' for b in result)
            print(f"\nPrompt: {p.decode()!r}")
            print(f"Output: {text}")
