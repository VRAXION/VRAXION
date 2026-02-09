"""Probe 7 extended — streams results to CSV for live dashboard.

Run from: S:/AI/work/VRAXION_DEV/Golden Draft/
"""

import csv
import os
import random
import sys
import time

_golden_code = r"S:\AI\Golden Code"
_golden_draft = r"S:\AI\work\VRAXION_DEV\Golden Draft"
for p in [_golden_code, _golden_draft, os.path.join(_golden_draft, "tools")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("VRX_SYNTH", "1")
os.environ.setdefault("VRX_SYNTH_MODE", "assoc_clean")
os.environ.setdefault("VRX_SYNTH_LEN", "16")
os.environ.setdefault("VRX_ASSOC_KEYS", "4")
os.environ.setdefault("VRX_ASSOC_PAIRS", "3")
os.environ.setdefault("VRX_MAX_SAMPLES", "256")
os.environ.setdefault("VRX_BATCH_SIZE", "16")
os.environ["VRX_SENSORY_RING"] = "0"
os.environ["VRX_VAULT"] = "0"
os.environ["VRX_PRISMION"] = "0"
os.environ["VRX_THINK_RING"] = "1"
os.environ["VRX_THINK_RING_DUAL"] = "0"
os.environ["VRX_THINK_RING_BRAINSTEM"] = "0"

import torch
import torch.nn.functional as F

MODEL_KW = dict(input_dim=1, num_classes=2, ring_len=64, slot_dim=32)

CSV_PATH = os.path.join(os.path.dirname(__file__), "probe7_live.csv")
STATUS_PATH = os.path.join(os.path.dirname(__file__), "probe7_status.txt")

STEPS = 1500
LR = 1e-3
BATCH_SIZE = 16
SEQ_LEN = 24
N_KEYS = 26
N_PAIRS = 4


def _make_model():
    from vraxion.platinum.hallway import AbsoluteHallway
    return AbsoluteHallway(**MODEL_KW)


def _generate_batch(step_seed):
    rng = random.Random(step_seed)
    x = torch.zeros((BATCH_SIZE, SEQ_LEN, 1), dtype=torch.float32)
    y = torch.zeros((BATCH_SIZE,), dtype=torch.long)

    for b in range(BATCH_SIZE):
        keys = rng.sample(range(N_KEYS), N_PAIRS)
        vals = [rng.randint(0, 1) for _ in range(N_PAIRS)]

        available = list(range(0, SEQ_LEN - 1))
        rng.shuffle(available)
        positions = []
        used = set()
        for pos in available:
            if pos in used or (pos + 1) in used or (pos + 1) == SEQ_LEN - 1:
                continue
            used.add(pos)
            used.add(pos + 1)
            positions.append(pos)
            if len(positions) >= N_PAIRS:
                break

        for i, pos in enumerate(positions):
            x[b, pos, 0] = float(2 + keys[i])
            x[b, pos + 1, 0] = -1.0 if vals[i] == 0 else -2.0

        q_idx = rng.randint(0, N_PAIRS - 1)
        x[b, -1, 0] = float(2 + keys[q_idx])
        y[b] = vals[q_idx]

    return x, y


def _write_status(msg):
    with open(STATUS_PATH, "w") as f:
        f.write(msg)


def main():
    # Initialize CSV with header.
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "model", "loss", "acc", "avg_loss_20", "avg_acc_20"])

    _write_status("starting")
    print(f"[probe7_live] CSV -> {CSV_PATH}")
    print(f"[probe7_live] {STEPS} steps, {N_KEYS} keys, {N_PAIRS} pairs, "
          f"seq_len={SEQ_LEN}, streaming")

    models = {}
    optimizers = {}
    histories = {}

    for label, scale in [("A_scale1.0", 1.0), ("B_scale0.01", 0.01)]:
        torch.manual_seed(42)
        random.seed(42)
        m = _make_model()
        m.update_scale = scale
        m.train()
        models[label] = m
        optimizers[label] = torch.optim.Adam(m.parameters(), lr=LR)
        histories[label] = {"losses": [], "accs": []}

    t0 = time.time()

    for step in range(1, STEPS + 1):
        x_batch, y_batch = _generate_batch(step_seed=step * 1000 + 7)

        rows = []
        for label in ["A_scale1.0", "B_scale0.01"]:
            model = models[label]
            opt = optimizers[label]
            hist = histories[label]

            opt.zero_grad()
            out = model(x_batch)
            logits = out[0]
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y_batch).float().mean().item()

            hist["losses"].append(loss.item())
            hist["accs"].append(acc)

            # Rolling 20-step average.
            w = 20
            avg_loss = sum(hist["losses"][-w:]) / len(hist["losses"][-w:])
            avg_acc = sum(hist["accs"][-w:]) / len(hist["accs"][-w:])

            rows.append([step, label, f"{loss.item():.6f}", f"{acc:.4f}",
                         f"{avg_loss:.6f}", f"{avg_acc:.4f}"])

        # Append both rows to CSV.
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)

        if step % 50 == 0:
            ha = histories["A_scale1.0"]
            hb = histories["B_scale0.01"]
            avg_a = sum(ha["accs"][-50:]) / 50
            avg_b = sum(hb["accs"][-50:]) / 50
            elapsed = time.time() - t0
            eta = elapsed / step * (STEPS - step)
            _write_status(f"step {step}/{STEPS}  A={avg_a:.3f}  B={avg_b:.3f}  "
                          f"delta={avg_a-avg_b:+.3f}  ETA={eta:.0f}s")
            print(f"  step {step:4d}/{STEPS}  "
                  f"A acc={avg_a:.3f}  B acc={avg_b:.3f}  "
                  f"delta={avg_a-avg_b:+.3f}  [{elapsed:.0f}s elapsed, ~{eta:.0f}s left]")

    # Final summary.
    ha = histories["A_scale1.0"]
    hb = histories["B_scale0.01"]
    final_n = 100
    avg_a = sum(ha["accs"][-final_n:]) / final_n
    avg_b = sum(hb["accs"][-final_n:]) / final_n
    delta = avg_a - avg_b
    dt = time.time() - t0

    print(f"\n{'='*60}")
    print(f"FINAL ({final_n}-step avg)")
    print(f"  A (scale=1.0):  acc={avg_a:.3f}  (+{(avg_a-0.5)*100:.1f}pp above chance)")
    print(f"  B (scale=0.01): acc={avg_b:.3f}  (+{(avg_b-0.5)*100:.1f}pp above chance)")
    print(f"  delta={delta:+.3f} ({delta*100:+.1f}pp)")
    print(f"  time={dt:.1f}s")
    print(f"{'='*60}")

    if delta > 0.05:
        print("  VERDICT: A >> B — full ring writes NEEDED for generalization")
    elif delta < -0.05:
        print("  VERDICT: B >> A — dampening helps on streaming")
    else:
        print("  VERDICT: A ~ B — ring scale not the bottleneck")

    _write_status("done")


if __name__ == "__main__":
    main()
