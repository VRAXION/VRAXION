"""Probe 9 — Full GPU Prismion Swarm: Does scale matter at real size?

Prismion bank topology, N=4 beings, GPU-accelerated.
Same infinite-assoc streaming task (26 keys, 4 pairs/seq, un-memorizable).

Compare:
  A: scale=1.0 (all beings) — platinum default
  B: scale=0.01 (all beings) — old AGC-bugged value

1500 steps, batch_size=32, GPU accelerated.
Writes probe9_live.csv for live dashboard on :8502.

If bank shows signal, run Fibonacci swarm variant (step 3 of the plan).

Run:  python tools/_scratch/probe9_swarm.py
      python tools/_scratch/probe9_swarm.py --fib   (Fibonacci swarm)
From: S:/AI/work/VRAXION_DEV/Golden Draft/
"""

import csv
import os
import random as _random
import sys
import time
import traceback

# ── Path setup ──────────────────────────────────────────────────────────
_golden_code = r"S:\AI\Golden Code"
_golden_draft = r"S:\AI\work\VRAXION_DEV\Golden Draft"
for p in [_golden_code, _golden_draft, os.path.join(_golden_draft, "tools")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Env: enable Prismion swarm ──────────────────────────────────────────
os.environ.setdefault("VRX_SYNTH", "1")
os.environ.setdefault("VRX_SYNTH_MODE", "assoc_clean")
os.environ.setdefault("VRX_SYNTH_LEN", "16")
os.environ.setdefault("VRX_ASSOC_KEYS", "4")
os.environ.setdefault("VRX_ASSOC_PAIRS", "3")
os.environ.setdefault("VRX_MAX_SAMPLES", "256")
os.environ.setdefault("VRX_BATCH_SIZE", "32")
# Core subsystems.
os.environ["VRX_SENSORY_RING"] = "0"
os.environ["VRX_VAULT"] = "0"
os.environ["VRX_THINK_RING"] = "1"
os.environ["VRX_THINK_RING_DUAL"] = "0"
os.environ["VRX_THINK_RING_BRAINSTEM"] = "0"

import torch
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────────────
STEPS = 1500
LR = 1e-3
LOG_EVERY = 50
BATCH_SIZE = 32
SEQ_LEN = 24
N_KEYS = 26
N_PAIRS = 4
NUM_CLASSES = 2
MODEL_KW = dict(input_dim=1, num_classes=NUM_CLASSES, ring_len=64, slot_dim=32)

SCRATCH_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRATCH_DIR, "probe9_live.csv")
STATUS_PATH = os.path.join(SCRATCH_DIR, "probe9_status.txt")


def _write_status(msg: str):
    with open(STATUS_PATH, "w") as f:
        f.write(msg)


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_swarm_model(scale: float, fibonacci: bool = False):
    """Build a Platinum model with Prismion swarm enabled."""
    os.environ["VRX_PRISMION"] = "1"
    os.environ["VRX_PRISMION_N"] = "4"
    os.environ["VRX_PRISMION_LEN"] = "64"
    os.environ["VRX_PRISMION_TOPOLOGY"] = "bank"
    if fibonacci:
        os.environ["VRX_PRISMION_FIBONACCI"] = "1"
        os.environ["VRX_PRISMION_FIB_MAX_ANTS"] = "8"
    else:
        os.environ["VRX_PRISMION_FIBONACCI"] = "0"

    from vraxion.platinum.hallway import AbsoluteHallway as PlatinumHallway
    model = PlatinumHallway(**MODEL_KW)
    model.update_scale = float(scale)

    # Restore env defaults.
    os.environ["VRX_PRISMION"] = "0"
    os.environ["VRX_PRISMION_FIBONACCI"] = "0"

    return model


def _generate_batch(step_seed: int, device: torch.device):
    """Generate one batch of fresh assoc data, deterministic per seed."""
    rng = _random.Random(step_seed)
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
            key_token = float(2 + keys[i])
            val_token = -1.0 if vals[i] == 0 else -2.0
            x[b, pos, 0] = key_token
            x[b, pos + 1, 0] = val_token

        q_idx = rng.randint(0, N_PAIRS - 1)
        x[b, -1, 0] = float(2 + keys[q_idx])
        y[b] = vals[q_idx]

    return x.to(device), y.to(device)


def _train_run(scale: float, label: str, device: torch.device,
               seed: int = 42, fibonacci: bool = False,
               csv_writer=None, csv_rows: list = None):
    """Train a swarm model on streaming data. Returns (losses, accs)."""
    torch.manual_seed(seed)
    _random.seed(seed)

    model = _make_swarm_model(scale, fibonacci=fibonacci)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    prism_info = ""
    if model.prismion_core is not None:
        prism_info = f"  prismion_n={model.prismion_n_used}"
    fib_info = ""
    if getattr(model, "prismion_fib_active", False):
        n_ants = len(model.prismion_swarm_configs) if model.prismion_swarm_configs else 0
        fib_info = f"  fib_ants={n_ants}"
    print(f"    [{label}] params={n_params:,}  device={device}"
          f"{prism_info}{fib_info}  scale={scale}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    accs = []
    t0 = time.time()

    for step in range(1, STEPS + 1):
        x_batch, y_batch = _generate_batch(step_seed=step * 1000 + 7, device=device)

        optimizer.zero_grad()
        out = model(x_batch)
        logits = out[0]
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y_batch).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        # Rolling averages for CSV.
        w = min(20, len(losses))
        avg_loss_20 = sum(losses[-w:]) / w
        avg_acc_20 = sum(accs[-w:]) / w

        if csv_rows is not None:
            csv_rows.append({
                "step": step,
                "model": label,
                "loss": f"{loss.item():.6f}",
                "acc": f"{acc:.4f}",
                "avg_loss_20": f"{avg_loss_20:.6f}",
                "avg_acc_20": f"{avg_acc_20:.4f}",
            })

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / max(elapsed, 0.01)
            wl = losses[-LOG_EVERY:]
            wa = accs[-LOG_EVERY:]
            print(f"    [{label}] step {step:4d}  "
                  f"loss={sum(wl)/len(wl):.4f}  acc={sum(wa)/len(wa):.3f}  "
                  f"({steps_per_sec:.1f} step/s)")

    return losses, accs


def run_probe9(fibonacci: bool = False):
    """Main probe 9 entry point."""
    mode = "Fibonacci swarm" if fibonacci else "Prismion bank N=4"
    print(f"\n{'='*70}")
    print(f"Probe 9 — GPU Swarm: {mode}")
    print(f"{'='*70}")

    device = _detect_device()
    print(f"  Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name} ({props.total_mem / 1e9:.1f} GB)")
    print(f"  Steps: {STEPS}  Batch: {BATCH_SIZE}  Keys: {N_KEYS}  Pairs: {N_PAIRS}")

    _write_status(f"starting {mode}")

    # ── Reproducibility check ────────────────────────────────────────────
    print("\n  === Reproducibility check (scale=1.0, seed=42, x2) ===")
    print("  Run R1...")
    losses_r1, _ = _train_run(1.0, "repro-R1", device, seed=42, fibonacci=fibonacci)
    print("  Run R2...")
    losses_r2, _ = _train_run(1.0, "repro-R2", device, seed=42, fibonacci=fibonacci)

    repro_ok = abs(losses_r1[-1] - losses_r2[-1]) < 1e-5
    print(f"  R1 final loss={losses_r1[-1]:.6f}  R2={losses_r2[-1]:.6f}  match={repro_ok}")
    if not repro_ok:
        _write_status("FAIL: reproducibility")
        print("  FAIL: reproducibility check failed!")
        return

    # ── Prepare CSV ──────────────────────────────────────────────────────
    csv_rows_a: list = []
    csv_rows_b: list = []

    # ── Main comparison ──────────────────────────────────────────────────
    _write_status("training A (scale=1.0)")
    print(f"\n  === Model A: scale=1.0 ({mode}) ===")
    losses_a, accs_a = _train_run(
        1.0, "A_scale1.0", device, seed=42, fibonacci=fibonacci,
        csv_rows=csv_rows_a)

    _write_status("training B (scale=0.01)")
    print(f"\n  === Model B: scale=0.01 ({mode}) ===")
    losses_b, accs_b = _train_run(
        0.01, "B_scale0.01", device, seed=42, fibonacci=fibonacci,
        csv_rows=csv_rows_b)

    # ── Write CSV ────────────────────────────────────────────────────────
    all_rows = csv_rows_a + csv_rows_b
    fieldnames = ["step", "model", "loss", "acc", "avg_loss_20", "avg_acc_20"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  CSV written: {CSV_PATH}")

    # ── Curve comparison ─────────────────────────────────────────────────
    print(f"\n  === Step-by-step comparison ({LOG_EVERY}-step avg) ===")
    print(f"  {'step':>5s}  {'A_loss':>7s}  {'A_acc':>6s}  {'B_loss':>7s}  {'B_acc':>6s}  {'delta':>7s}")
    for s in range(LOG_EVERY, STEPS + 1, LOG_EVERY):
        wa = accs_a[s-LOG_EVERY:s]
        wb = accs_b[s-LOG_EVERY:s]
        la = losses_a[s-LOG_EVERY:s]
        lb = losses_b[s-LOG_EVERY:s]
        d = sum(wa)/len(wa) - sum(wb)/len(wb)
        print(f"  {s:5d}  {sum(la)/len(la):7.4f}  {sum(wa)/len(wa):6.3f}  "
              f"{sum(lb)/len(lb):7.4f}  {sum(wb)/len(wb):6.3f}  {d:+7.3f}")

    # ── Verdict ──────────────────────────────────────────────────────────
    final_n = 100
    avg_acc_a = sum(accs_a[-final_n:]) / final_n
    avg_acc_b = sum(accs_b[-final_n:]) / final_n
    avg_loss_a = sum(losses_a[-final_n:]) / final_n
    avg_loss_b = sum(losses_b[-final_n:]) / final_n
    delta = avg_acc_a - avg_acc_b

    above_chance_a = avg_acc_a - 0.5
    above_chance_b = avg_acc_b - 0.5

    if delta > 0.05:
        verdict = f"1.0 >> 0.01 by {delta*100:.1f}pp: AGC bug was harmful at swarm scale"
    elif delta < -0.05:
        verdict = f"0.01 >> 1.0 by {-delta*100:.1f}pp: dampening = regularization benefit"
    else:
        verdict = "1.0 ~ 0.01 (within 5pp): ring scale irrelevant even in swarm"

    print(f"\n{'='*70}")
    print(f"VERDICT ({mode})")
    print(f"{'='*70}")
    print(f"  {verdict}")
    print(f"  A (scale=1.0):  loss={avg_loss_a:.4f}  acc={avg_acc_a:.3f}  "
          f"(+{above_chance_a*100:.1f}pp above chance)")
    print(f"  B (scale=0.01): loss={avg_loss_b:.4f}  acc={avg_acc_b:.3f}  "
          f"(+{above_chance_b*100:.1f}pp above chance)")
    print(f"  delta_acc={delta:+.3f} ({delta*100:+.1f}pp)")
    print(f"  repro_check=PASS  device={device}  steps={STEPS}")
    print(f"{'='*70}")

    _write_status("done")
    return verdict


if __name__ == "__main__":
    fibonacci = "--fib" in sys.argv
    try:
        run_probe9(fibonacci=fibonacci)
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        _write_status(f"FAIL: {e}")
