"""
Structured Loop Init Smoke Test
================================
Instead of random edges, initialize the graph with structured paths:
  - Each path: input neuron → chain of 1-10 hidden neurons → output neuron
  - Loops: some paths loop back (last neuron feeds into an earlier one)
  - Random signs on edges (±0.6)
  - Various path lengths to test which lengths survive crystallize

Hypothesis: structured paths already contain multi-hop routes. Crystallize
should keep the useful ones and cut the noise.

Compare:
  A) Random dense 2% init → crystallize (baseline from previous test)
  B) Structured loop init (same edge budget) → crystallize
  C) Structured loop init → NO crystallize (raw performance)
"""
import sys, os, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

IO = 256
H = 256
TICKS = 8
INJ_TICKS = 2
SEED = 42
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "loop_init"


def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


def compute_bigram_from_bytes(data_bytes):
    bigram = np.zeros((256, 256), dtype=np.float64)
    for i in range(len(data_bytes) - 1):
        bigram[data_bytes[i], data_bytes[i + 1]] += 1
    row_sums = bigram.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram /= row_sums
    return bigram.astype(np.float32)


def build_loop_graph(H, n_paths, rng, input_proj, output_proj):
    """
    Build structured paths through the hidden graph.

    Each path:
      1. Pick a "source" neuron (good input receiver = high input_proj norm)
      2. Chain through random hidden neurons (length 1-10)
      3. End at a "sink" neuron (high output_proj norm)
      4. Optionally loop: connect last → earlier node in the chain

    Returns: mask (H, H) with the structured edges
    """
    mask = np.zeros((H, H), dtype=np.float32)

    # Score neurons by input/output sensitivity
    # Input sensitivity: how much does each neuron receive from input_proj?
    input_sensitivity = np.linalg.norm(input_proj, axis=0)  # (H,)
    # Output sensitivity: how much does each neuron contribute to output?
    output_sensitivity = np.linalg.norm(output_proj, axis=1)  # (H,)

    # Pool of available neurons (shuffled)
    available = list(range(H))
    used_in_paths = set()
    path_info = []

    for path_id in range(n_paths):
        # Path length: 1-10 neurons in the chain
        path_len = rng.randint(1, 10)

        # Pick source: prefer high input sensitivity
        # Weighted random pick from top 50% input-sensitive neurons
        candidates = [n for n in range(H) if n not in used_in_paths]
        if len(candidates) < path_len + 2:
            candidates = list(range(H))  # reuse if running low

        # Sort by input sensitivity, pick from top half with some randomness
        candidates_scored = [(n, input_sensitivity[n]) for n in candidates]
        candidates_scored.sort(key=lambda x: -x[1])
        top_half = [n for n, _ in candidates_scored[:max(len(candidates_scored)//2, path_len+2)]]
        rng.shuffle(top_half)

        # Build chain
        chain = top_half[:path_len + 1]  # +1 for at least 2 nodes
        if len(chain) < 2:
            continue

        # Add forward edges along the chain
        for i in range(len(chain) - 1):
            src, dst = chain[i], chain[i + 1]
            if src != dst:
                sign = 0.6 if rng.random() < 0.5 else -0.6
                mask[src, dst] = sign

        # Loop: connect last → some earlier node (50% chance)
        if len(chain) >= 3 and rng.random() < 0.5:
            loop_target = rng.randint(0, len(chain) - 2)  # not the last node
            src, dst = chain[-1], chain[loop_target]
            if src != dst and mask[src, dst] == 0:
                sign = 0.6 if rng.random() < 0.5 else -0.6
                mask[src, dst] = sign

        # Branch: sometimes add a side branch from middle of chain (30% chance)
        if len(chain) >= 4 and rng.random() < 0.3:
            branch_from = chain[rng.randint(1, len(chain) - 2)]
            branch_to = rng.randint(0, H - 1)
            if branch_from != branch_to and mask[branch_from, branch_to] == 0:
                sign = 0.6 if rng.random() < 0.5 else -0.6
                mask[branch_from, branch_to] = sign

        for n in chain:
            used_in_paths.add(n)

        has_loop = mask[chain[-1], chain[0]] != 0 or any(
            mask[chain[-1], chain[j]] != 0 for j in range(len(chain)-1))
        path_info.append({
            'length': len(chain),
            'has_loop': has_loop,
            'source_input_sens': float(input_sensitivity[chain[0]]),
        })

    np.fill_diagonal(mask, 0)
    return mask, path_info


def network_stats(mask):
    edges = int(np.count_nonzero(mask))
    in_deg = np.count_nonzero(mask, axis=0)
    out_deg = np.count_nonzero(mask, axis=1)
    connected = (in_deg + out_deg) > 0
    has_in = in_deg > 0
    has_out = out_deg > 0
    return {
        'edges': edges,
        'connected': int(connected.sum()),
        'bidirectional': int((has_in & has_out).sum()),
        'max_deg': int(max(in_deg.max(), out_deg.max())) if edges > 0 else 0,
    }


def eval_accuracy(mask, theta, decay, text_bytes, bp, input_proj, output_proj):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        for t in range(TICKS):
            if t < INJ_TICKS:
                act = act + bp[text_bytes[i]] @ input_proj
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            np.clip(charge, -10.0, 10.0, out=charge)
            act = np.maximum(charge, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_proj
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i + 1]:
            correct += 1
        total += 1
    return correct / total if total else 0


def eval_loglik(mask, theta, decay, seqs, bp, input_proj, output_proj):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        ll = 0.0; n = 0
        for i in range(len(text_bytes) - 1):
            act = state.copy()
            for t in range(TICKS):
                if t < INJ_TICKS:
                    act = act + bp[text_bytes[i]] @ input_proj
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                np.clip(charge, -10.0, 10.0, out=charge)
                act = np.maximum(charge, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ output_proj
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            probs = e / e.sum()
            ll += np.log(probs[text_bytes[i + 1]] + 1e-10)
            n += 1
        total += ll / n if n else 0
    return total / len(seqs)


def crystallize_with_loglik(net, eval_seqs, bp, input_proj, output_proj, verbose=True):
    """Crystallize using log-likelihood."""
    def score_fn():
        return eval_loglik(net.mask, net.theta, net.decay, eval_seqs, bp, input_proj, output_proj)

    score = score_fn()
    total_removed = 0
    pass_num = 0
    while True:
        alive_snapshot = list(net.alive)
        random.shuffle(alive_snapshot)
        removed_this_pass = 0
        for r, c in alive_snapshot:
            if net.mask[r, c] == 0:
                continue
            old_val = net.mask[r, c]
            net.mask[r, c] = 0.0
            net.alive_set.discard((r, c))
            new_score = score_fn()
            if new_score >= score:
                score = new_score
                removed_this_pass += 1
                total_removed += 1
            else:
                net.mask[r, c] = old_val
                net.alive_set.add((r, c))
        net.resync_alive()
        pass_num += 1
        stats = network_stats(net.mask)
        if verbose:
            print(f"    pass {pass_num}: removed {removed_this_pass}, "
                  f"edges={stats['edges']} bidir={stats['bidirectional']} "
                  f"score={score:.4f}")
        if removed_this_pass == 0:
            break
    return total_removed


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading alice.txt...")
    with open(DATA_DIR / "alice.txt", "rb") as f:
        all_data = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"  {len(all_data)} bytes")

    bigram = compute_bigram_from_bytes(all_data)
    bp = make_bp(IO)

    random.seed(SEED); np.random.seed(SEED)
    ref = SelfWiringGraph(IO, hidden_ratio=1, projection_scale=1.0, seed=SEED)
    input_proj = ref.input_projection
    output_proj = ref.output_projection

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[off:off + 200]
                 for off in [eval_rng.randint(0, len(all_data) - 200) for _ in range(5)]]

    # ── Build loop graphs with different path counts ────────────────
    for n_paths, label in [(50, "50 paths"), (100, "100 paths"), (200, "200 paths")]:
        print(f"\n{'='*65}")
        print(f"  STRUCTURED LOOP INIT: {label}")
        print(f"{'='*65}")

        # Build structured graph
        build_rng = random.Random(SEED + n_paths)
        loop_mask, path_info = build_loop_graph(H, n_paths, build_rng, input_proj, output_proj)

        # Path stats
        lengths = [p['length'] for p in path_info]
        loops = sum(1 for p in path_info if p['has_loop'])
        stats = network_stats(loop_mask)
        print(f"  Built: {len(path_info)} paths, {loops} with loops")
        print(f"  Path lengths: min={min(lengths)} max={max(lengths)} avg={np.mean(lengths):.1f}")
        print(f"  Graph: {stats['edges']} edges, {stats['connected']} connected, "
              f"{stats['bidirectional']} bidirectional")

        # Create network with this structured mask
        random.seed(SEED); np.random.seed(SEED)
        net = SelfWiringGraph(IO, hidden_ratio=1, projection_scale=1.0, seed=SEED)
        net.theta[:] = 0.0
        decay_rng = np.random.RandomState(99)
        net.decay[:] = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)
        net.input_projection = input_proj
        net.output_projection = output_proj
        net.mask[:] = loop_mask
        net.resync_alive()

        # Raw performance (no crystallize)
        raw_acc = np.mean([eval_accuracy(net.mask, net.theta, net.decay, s, bp,
                           input_proj, output_proj) for s in eval_seqs])
        raw_ll = eval_loglik(net.mask, net.theta, net.decay, eval_seqs, bp, input_proj, output_proj)
        print(f"  RAW (no prune): acc={raw_acc*100:.2f}% ll={raw_ll:.3f}")

        # Save raw checkpoint
        net.save(str(CKPT_DIR / f"loop_{n_paths}_raw.npz"))

        # Crystallize
        print(f"  Crystallizing (loglik)...")
        t0 = time.time()
        removed = crystallize_with_loglik(net, eval_seqs[:3], bp, input_proj, output_proj)
        elapsed = time.time() - t0

        stats = network_stats(net.mask)
        final_acc = np.mean([eval_accuracy(net.mask, net.theta, net.decay, s, bp,
                             input_proj, output_proj) for s in eval_seqs])
        final_ll = eval_loglik(net.mask, net.theta, net.decay, eval_seqs, bp, input_proj, output_proj)

        print(f"  CRYSTAL: acc={final_acc*100:.2f}% ll={final_ll:.3f} "
              f"edges={stats['edges']} bidir={stats['bidirectional']} "
              f"removed={removed} {elapsed:.1f}s")

        net.save(str(CKPT_DIR / f"loop_{n_paths}_crystal.npz"))

        # Length analysis: which path lengths survived?
        # Check which edges from original paths are still alive
        alive_set = set(zip(*np.where(net.mask != 0))) if stats['edges'] > 0 else set()
        orig_edges = set(zip(*np.where(loop_mask != 0)))
        surviving = alive_set & orig_edges
        print(f"  Surviving edges: {len(surviving)}/{len(orig_edges)} "
              f"({len(surviving)/max(len(orig_edges),1)*100:.0f}%)")

    # ── Comparison with random dense (same edge count) ──────────────
    print(f"\n{'='*65}")
    print(f"  COMPARISON: Random dense (matched edge count)")
    print(f"{'='*65}")

    # Match the 100-path edge count
    random.seed(SEED); np.random.seed(SEED)
    _, path_info_100 = build_loop_graph(H, 100, random.Random(SEED + 100), input_proj, output_proj)

    # Use ~same edge count with random dense
    target_edges = network_stats(build_loop_graph(H, 100, random.Random(SEED + 100),
                                                   input_proj, output_proj)[0])['edges']
    target_density = target_edges / (H * H) * 100
    # density param: need >1.0 for percentage interpretation
    density_param = max(1.01, target_density)

    random.seed(SEED); np.random.seed(SEED)
    net_rand = SelfWiringGraph(IO, hidden_ratio=1, density=density_param,
                                projection_scale=1.0, seed=SEED)
    net_rand.theta[:] = 0.0
    decay_rng = np.random.RandomState(99)
    net_rand.decay[:] = decay_rng.uniform(0.08, 0.24, H).astype(np.float32)
    net_rand.input_projection = input_proj
    net_rand.output_projection = output_proj

    stats_rand = network_stats(net_rand.mask)
    raw_acc_rand = np.mean([eval_accuracy(net_rand.mask, net_rand.theta, net_rand.decay, s, bp,
                            input_proj, output_proj) for s in eval_seqs])
    raw_ll_rand = eval_loglik(net_rand.mask, net_rand.theta, net_rand.decay, eval_seqs, bp,
                               input_proj, output_proj)
    print(f"  Random dense: {stats_rand['edges']} edges, "
          f"acc={raw_acc_rand*100:.2f}% ll={raw_ll_rand:.3f}")

    print(f"  Crystallizing random dense...")
    t0 = time.time()
    removed_rand = crystallize_with_loglik(net_rand, eval_seqs[:3], bp, input_proj, output_proj)
    elapsed_rand = time.time() - t0

    stats_rand = network_stats(net_rand.mask)
    final_acc_rand = np.mean([eval_accuracy(net_rand.mask, net_rand.theta, net_rand.decay, s, bp,
                              input_proj, output_proj) for s in eval_seqs])
    final_ll_rand = eval_loglik(net_rand.mask, net_rand.theta, net_rand.decay, eval_seqs, bp,
                                 input_proj, output_proj)
    print(f"  CRYSTAL: acc={final_acc_rand*100:.2f}% ll={final_ll_rand:.3f} "
          f"edges={stats_rand['edges']} bidir={stats_rand['bidirectional']} "
          f"removed={removed_rand} {elapsed_rand:.1f}s")

    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"{'='*65}")
