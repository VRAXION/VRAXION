"""
INSTNCT — Loop Burst on English Text: the real test
=====================================================
Real byte-level English text (not synthetic patterns).
Burst-add loops of varying length, then train with full mutate().
Sweep tick rates: 8, 12, 16.

Task: given byte A, predict byte B (bigram prediction on proverbs).
This is the closest to the mainline English recipe at mini scale.
"""
import sys, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256
INPUT_DURATION = 2
PLATEAU_WINDOW = 400

# Real English text — proverbs + additional for variety
TEXT = (
    "a stitch in time saves nine. the early bird catches the worm. "
    "all that glitters is not gold. actions speak louder than words. "
    "fortune favors the bold. knowledge is power. practice makes perfect. "
    "the pen is mightier than the sword. where there is a will there is a way. "
    "birds of a feather flock together. every cloud has a silver lining. "
    "honesty is the best policy. look before you leap. better late than never. "
    "two wrongs do not make a right. when in rome do as the romans do. "
    "curiosity killed the cat. do not put all your eggs in one basket. "
    "the apple does not fall far from the tree. "
    "people who live in glass houses should not throw stones. "
    "you can lead a horse to water but you cannot make it drink. "
    "if you want something done right do it yourself. "
)

TEXT_BYTES = np.frombuffer(TEXT.encode('ascii'), dtype=np.uint8)

# Build bigram distribution from text
BIGRAM = np.zeros((256, 256), dtype=np.float32)
for i in range(len(TEXT_BYTES) - 1):
    BIGRAM[TEXT_BYTES[i], TEXT_BYTES[i + 1]] += 1
# Normalize rows
row_sums = BIGRAM.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
BIGRAM /= row_sums


def make_sequences(rng, n_seqs=4, seq_len=50):
    """Random slices from the text corpus."""
    seqs = []
    for _ in range(n_seqs):
        start = rng.randint(0, max(1, len(TEXT_BYTES) - seq_len))
        seqs.append(TEXT_BYTES[start:start + seq_len].tolist())
    return seqs


def eval_bigram_acc(net, seqs, ticks):
    """Byte-level next-character accuracy on real English text."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    correct = 0; total = 0

    for seq in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        for i in range(len(seq) - 1):
            injected = net.input_projection[seq[i]]
            state, charge = SelfWiringGraph.rollout_token(
                injected, mask=net.mask, theta=net._theta_f32,
                decay=net.decay, ticks=ticks, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )
            logits = charge @ net.output_projection
            pred = int(np.argmax(logits))
            if pred == seq[i + 1]:
                correct += 1
            total += 1

    return correct / total if total else 0.0


def eval_bigram_cosine(net, seqs, ticks):
    """Bigram cosine similarity — same metric as mainline recipe."""
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    bp = np.eye(256, dtype=np.float32)  # one-hot patterns for cosine
    bp_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    total = 0.0; n = 0

    for seq in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        for i in range(len(seq) - 1):
            injected = net.input_projection[seq[i]]
            state, charge = SelfWiringGraph.rollout_token(
                injected, mask=net.mask, theta=net._theta_f32,
                decay=net.decay, ticks=ticks, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel,
            )
            out = charge @ net.output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ bp_norm.T
            e = np.exp(sims - sims.max())
            pred_dist = e / e.sum()
            target_dist = BIGRAM[seq[i]]
            cos = np.dot(pred_dist, target_dist) / (
                np.linalg.norm(pred_dist) * np.linalg.norm(target_dist) + 1e-8)
            total += cos; n += 1

    return total / n if n else 0.0


def burst_loops(net, n_loops, loop_len):
    H = net.H
    added = 0
    for _ in range(n_loops):
        if H < loop_len:
            continue
        nodes = random.sample(range(H), loop_len)
        for i in range(len(nodes)):
            r, c = nodes[i], nodes[(i + 1) % len(nodes)]
            if r != c and not net.mask[r, c]:
                net.mask[r, c] = True; added += 1
    net.resync_alive()
    return added


def run_config(H, ticks, loop_len, n_burst, train_steps, seed, eval_seqs):
    """Full run: init → plateau → burst loops → train → eval."""
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)

    def avg_acc():
        return np.mean([eval_bigram_acc(net, [s], ticks) for s in eval_seqs])

    # Phase 1: train until plateau
    best_acc = avg_acc()
    accepts = 0; stale = 0

    for step in range(1, train_steps + 1):
        if stale >= PLATEAU_WINDOW:
            # Phase 2: burst inject loops
            burst_loops(net, n_burst, loop_len)
            stale = 0
            new_acc = avg_acc()
            if new_acc > best_acc:
                best_acc = new_acc; accepts += 1

        state_snap = net.save_state()
        undo = net.mutate()
        new_acc = avg_acc()
        if new_acc > best_acc:
            best_acc = new_acc; accepts += 1; stale = 0
        else:
            net.restore_state(state_snap); stale += 1

    # Final eval
    cos = np.mean([eval_bigram_cosine(net, [s], ticks) for s in eval_seqs])
    return best_acc, cos, accepts, len(net.alive)


def main():
    master_rng = np.random.RandomState(99)
    eval_seqs = make_sequences(master_rng, n_seqs=6, seq_len=60)

    H = 64   # match previous tests for speed
    TRAIN_STEPS = 3000
    N_BURST = 15  # loops per burst injection
    SEED = 42

    print("=" * 90)
    print("  Loop Burst on English Text — Ticks × Loop Length Sweep")
    print(f"  H={H} | Train: {TRAIN_STEPS} steps | Burst: {N_BURST} loops | Plateau: {PLATEAU_WINDOW}")
    print(f"  Task: byte-level English bigram prediction (proverbs)")
    print("=" * 90)

    tick_options = [8, 12, 16]
    loop_options = [0, 3, 4, 5, 7]  # 0 = no loops (baseline)

    results = []
    for ticks in tick_options:
        for loop_len in loop_options:
            label = f"T={ticks:2d} L={loop_len}"
            if loop_len == 0:
                label = f"T={ticks:2d} none"
            print(f"  {label}...", end="", flush=True)
            t0 = time.time()
            acc, cos, accepts, edges = run_config(
                H, ticks, loop_len if loop_len > 0 else 3,
                N_BURST if loop_len > 0 else 0,
                TRAIN_STEPS, SEED, eval_seqs,
            )
            elapsed = time.time() - t0
            print(f" acc={acc:.3f} cos={cos:.3f} edges={edges} ({elapsed:.0f}s)")
            results.append((ticks, loop_len, acc, cos, accepts, edges))

    # Summary table
    print(f"\n{'='*90}")
    print(f"  RESULTS — English Bigram, H={H}")
    print(f"{'='*90}")

    # Table: rows = loop length, cols = ticks
    print(f"\n  Accuracy:")
    print(f"  {'':>8}", end="")
    for t in tick_options:
        print(f" |  T={t:>2} ", end="")
    print()
    print(f"  {'':>8}", end="")
    for _ in tick_options:
        print(f" +-------", end="")
    print()
    for loop_len in loop_options:
        lbl = "none" if loop_len == 0 else f"L={loop_len}"
        print(f"  {lbl:>8}", end="")
        for ticks in tick_options:
            r = [x for x in results if x[0] == ticks and x[1] == loop_len][0]
            best_at_tick = max(x[2] for x in results if x[0] == ticks)
            marker = " *" if r[2] == best_at_tick and r[2] > 0 else "  "
            print(f" | {r[2]:.3f}{marker}", end="")
        print()

    print(f"\n  Cosine:")
    print(f"  {'':>8}", end="")
    for t in tick_options:
        print(f" |  T={t:>2} ", end="")
    print()
    print(f"  {'':>8}", end="")
    for _ in tick_options:
        print(f" +-------", end="")
    print()
    for loop_len in loop_options:
        lbl = "none" if loop_len == 0 else f"L={loop_len}"
        print(f"  {lbl:>8}", end="")
        for ticks in tick_options:
            r = [x for x in results if x[0] == ticks and x[1] == loop_len][0]
            print(f" | {r[3]:.3f}  ", end="")
        print()

    # Best overall
    best = max(results, key=lambda x: x[2])
    print(f"\n  Best: T={best[0]} L={best[1]} → acc={best[2]:.3f} cos={best[3]:.3f}")


if __name__ == "__main__":
    main()
