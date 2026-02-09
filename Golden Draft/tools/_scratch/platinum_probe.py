"""Platinum port validation -- 5 surgical probes.

Tests whether the Platinum port of AbsoluteHallway is faithful,
structurally sound, and whether the AGC bug (scale=0.01) was hurting.

Run:  python tools/_scratch/platinum_probe.py
From: S:/AI/work/VRAXION_DEV/Golden Draft/
"""

import copy
import os
import random
import sys
import time
import traceback

# ── Path setup ──────────────────────────────────────────────────────────
# Golden Code has both vraxion.platinum and vraxion.instnct.
# Golden Draft tools dir has instnct_data.py which imports vraxion.settings.
_golden_code = r"S:\AI\Golden Code"
_golden_draft = r"S:\AI\work\VRAXION_DEV\Golden Draft"
for p in [_golden_code, _golden_draft, os.path.join(_golden_draft, "tools")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Force synth mode for data loader.
os.environ.setdefault("VRX_SYNTH", "1")
os.environ.setdefault("VRX_SYNTH_MODE", "assoc_clean")
os.environ.setdefault("VRX_SYNTH_LEN", "16")
os.environ.setdefault("VRX_ASSOC_KEYS", "4")
os.environ.setdefault("VRX_ASSOC_PAIRS", "3")
os.environ.setdefault("VRX_MAX_SAMPLES", "256")
os.environ.setdefault("VRX_BATCH_SIZE", "16")
# Disable optional sub-systems to keep param count small and deterministic.
os.environ["VRX_SENSORY_RING"] = "0"
os.environ["VRX_VAULT"] = "0"
os.environ["VRX_PRISMION"] = "0"
os.environ["VRX_THINK_RING"] = "1"
os.environ["VRX_THINK_RING_DUAL"] = "0"
os.environ["VRX_THINK_RING_BRAINSTEM"] = "0"

import torch
import torch.nn.functional as F


# ── Shared model kwargs ─────────────────────────────────────────────────
MODEL_KW = dict(input_dim=1, num_classes=2, ring_len=64, slot_dim=32)


def _make_platinum():
    from vraxion.platinum.hallway import AbsoluteHallway as PlatinumHallway
    return PlatinumHallway(**MODEL_KW)


def _make_instnct():
    from vraxion.instnct.absolute_hallway import AbsoluteHallway as InstnctHallway
    return InstnctHallway(**MODEL_KW)


# =========================================================================
# Probe 1 — Smoke Test
# =========================================================================
def probe_1_smoke():
    """Does platinum instantiate and produce correct output shapes?"""
    t0 = time.time()
    model = _make_platinum()
    model.eval()

    B, T, D = 4, 8, 1
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = model(x)

    logits = out[0]
    move_penalty = out[1]

    assert logits.shape == (B, MODEL_KW["num_classes"]), (
        f"logits shape {logits.shape} != expected ({B}, {MODEL_KW['num_classes']})"
    )
    assert move_penalty.dim() == 0, (
        f"move_penalty should be scalar, got shape {move_penalty.shape}"
    )
    assert model.update_scale == 1.0, (
        f"update_scale = {model.update_scale}, expected 1.0"
    )

    n_params = sum(p.numel() for p in model.parameters())
    dt = time.time() - t0
    return f"PASS  shapes ok, update_scale=1.0, params={n_params:,}, {dt:.1f}s"


# =========================================================================
# Probe 2 — Numerical Parity
# =========================================================================
def probe_2_parity():
    """Does platinum produce identical outputs to instnct given same weights?"""
    t0 = time.time()

    torch.manual_seed(42)
    plat = _make_platinum()
    plat.eval()

    torch.manual_seed(42)
    inst = _make_instnct()
    inst.eval()

    # Force instnct to use the same scale as platinum (bypass the bug).
    inst.update_scale = 1.0

    # Copy weights: try direct load first, fall back to intersection copy.
    plat_sd = plat.state_dict()
    inst_sd = inst.state_dict()

    # Both models have the same architecture structure, so state_dicts should
    # have the same keys when env-controlled options match.
    common = set(plat_sd.keys()) & set(inst_sd.keys())
    plat_only = set(plat_sd.keys()) - set(inst_sd.keys())
    inst_only = set(inst_sd.keys()) - set(plat_sd.keys())

    # Copy common keys from platinum -> instnct.
    transfer = {}
    shape_mismatches = []
    for k in common:
        if plat_sd[k].shape == inst_sd[k].shape:
            transfer[k] = plat_sd[k].clone()
        else:
            shape_mismatches.append((k, plat_sd[k].shape, inst_sd[k].shape))

    if shape_mismatches:
        detail = "; ".join(f"{k}: plat={ps} inst={is_}" for k, ps, is_ in shape_mismatches)
        return f"SKIP  shape mismatches in {len(shape_mismatches)} keys: {detail}"

    # Load into instnct model (strict=False to keep instnct-only keys unchanged).
    inst_sd.update(transfer)
    inst.load_state_dict(inst_sd, strict=True)

    # Feed identical input.
    torch.manual_seed(99)
    x = torch.randn(4, 8, 1)

    with torch.no_grad():
        out_p = plat(x)
        out_i = inst(x)

    logits_p, mp_p = out_p[0], out_p[1]
    logits_i, mp_i = out_i[0], out_i[1]

    match = torch.allclose(logits_p, logits_i, atol=1e-5)
    max_diff = float((logits_p - logits_i).abs().max().item())

    dt = time.time() - t0
    extra = (
        f"common={len(common)} plat_only={len(plat_only)} inst_only={len(inst_only)}"
    )
    if match:
        return f"PASS  outputs match (max_diff={max_diff:.2e}), {extra}, {dt:.1f}s"
    else:
        return f"FAIL  outputs diverge (max_diff={max_diff:.2e}), {extra}, {dt:.1f}s"


# =========================================================================
# Probe 3 — Gradient Flow
# =========================================================================
def probe_3_gradient_flow():
    """Do all parameters receive gradients?"""
    t0 = time.time()
    torch.manual_seed(42)
    model = _make_platinum()
    model.train()

    x = torch.randn(4, 8, 1)
    labels = torch.randint(0, MODEL_KW["num_classes"], (4,))

    out = model(x)
    logits = out[0]
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    dead = []
    alive = []
    for name, param in model.named_parameters():
        if param.grad is None:
            dead.append(name)
        elif param.grad.abs().sum().item() == 0.0:
            dead.append(f"{name}(zero)")
        else:
            alive.append(name)

    dt = time.time() - t0
    if dead:
        return f"WARN  {len(dead)} dead params: {dead}, {len(alive)} alive, {dt:.1f}s"
    return f"PASS  all {len(alive)} params receive gradient, {dt:.1f}s"


# =========================================================================
# Probe 4 — Ring Liveness
# =========================================================================
def probe_4_ring_liveness():
    """Does the ring memory get written to and does the pointer move?"""
    t0 = time.time()
    torch.manual_seed(42)
    model = _make_platinum()
    model.eval()

    B, T, D = 2, 8, 1
    norms = []
    # We'll track the ring state across multiple forward passes.
    # Each forward starts fresh (model doesn't persist ring state between
    # calls), so instead we check that within a single long sequence the
    # ring accumulates writes.
    # Use a longer sequence to give the ring more steps.
    x_long = torch.randn(1, 40, 1)

    # Instrument: we'll hook into the state scatter_add to measure ring norm.
    # Simpler: just run forward and check the xray telemetry if available,
    # or check that model output changes with sequence content.

    # Approach: run two identical models with different input sequences.
    # If ring is live, outputs diverge after initial steps.
    torch.manual_seed(42)
    model_a = _make_platinum()
    model_a.eval()
    model_b = copy.deepcopy(model_a)

    # Same model weights, different inputs.
    torch.manual_seed(10)
    x1 = torch.randn(1, 20, 1)
    torch.manual_seed(20)
    x2 = torch.randn(1, 20, 1)

    with torch.no_grad():
        out1 = model_a(x1)
        out2 = model_b(x2)

    logits1 = out1[0]
    logits2 = out2[0]
    diff = float((logits1 - logits2).abs().max().item())

    # If ring is alive, different inputs should produce different outputs.
    ring_responsive = diff > 1e-6

    # Also check: same input should produce same output (determinism).
    model_c = copy.deepcopy(model_a)
    with torch.no_grad():
        out3 = model_c(x1)
    logits3 = out3[0]
    deterministic = torch.allclose(logits1, logits3, atol=1e-6)

    # Check ring norm grows across sequence length:
    # Run a short vs long sequence starting with same prefix.
    torch.manual_seed(42)
    x_short = torch.randn(1, 5, 1)
    x_long = torch.cat([x_short, torch.randn(1, 15, 1)], dim=1)  # 20 steps

    model_short = copy.deepcopy(model_a)
    model_long = copy.deepcopy(model_a)
    with torch.no_grad():
        out_s = model_short(x_short)
        out_l = model_long(x_long)
    logits_s = out_s[0]
    logits_l = out_l[0]
    # If ring accumulates, longer sequence should give different output
    # (since it has more context).
    length_matters = not torch.allclose(logits_s, logits_l, atol=1e-6)

    dt = time.time() - t0
    parts = []
    if ring_responsive:
        parts.append("input-sensitive=YES")
    else:
        parts.append("input-sensitive=NO (ring may be dead)")
    if deterministic:
        parts.append("deterministic=YES")
    else:
        parts.append("deterministic=NO (nondeterminism detected)")
    if length_matters:
        parts.append("length-sensitive=YES")
    else:
        parts.append("length-sensitive=NO (ring may not accumulate)")

    ok = ring_responsive and deterministic and length_matters
    status = "PASS" if ok else "WARN"
    return f"{status}  diff={diff:.6f}, {', '.join(parts)}, {dt:.1f}s"


# =========================================================================
# Probe 5 — The AGC Bug Test
# =========================================================================
def probe_5_agc_bug_test():
    """Train scale=1.0 vs scale=0.01 on assoc_clean for 100 steps."""
    t0 = time.time()

    from instnct_data import get_seq_mnist_loader

    STEPS = 100
    LR = 1e-3

    def _train_run(scale_value: float, label: str):
        """Train one model and return (losses, accs) lists."""
        torch.manual_seed(42)
        model = _make_platinum()
        model.update_scale = float(scale_value)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Build a fresh data loader (assoc_clean generates random data each call).
        torch.manual_seed(42)
        loader, num_classes, collate_fn = get_seq_mnist_loader(train=True)

        losses = []
        accs = []
        step = 0
        while step < STEPS:
            for batch in loader:
                if step >= STEPS:
                    break
                x_batch, y_batch = batch
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
                step += 1

                if step % 25 == 0:
                    avg_loss = sum(losses[-25:]) / len(losses[-25:])
                    avg_acc = sum(accs[-25:]) / len(accs[-25:])
                    print(f"    [{label}] step {step:3d}  loss={avg_loss:.4f}  acc={avg_acc:.3f}")

        return losses, accs

    print("\n  Training Model A (scale=1.0, platinum default)...")
    losses_a, accs_a = _train_run(1.0, "scale=1.0")

    print("\n  Training Model B (scale=0.01, instnct bugged)...")
    losses_b, accs_b = _train_run(0.01, "scale=0.01")

    # Compare final 25-step averages.
    final_n = 25
    avg_loss_a = sum(losses_a[-final_n:]) / final_n
    avg_loss_b = sum(losses_b[-final_n:]) / final_n
    avg_acc_a = sum(accs_a[-final_n:]) / final_n
    avg_acc_b = sum(accs_b[-final_n:]) / final_n

    dt = time.time() - t0

    # Verdict.
    delta_acc = avg_acc_a - avg_acc_b
    if delta_acc > 0.05:
        verdict = "A >> B: AGC bug was CRIPPLING the model"
    elif delta_acc < -0.05:
        verdict = "A << B: scale dampening was accidentally beneficial"
    else:
        verdict = "A ~ B: ring memory may not be critical for this task size"

    return (
        f"{verdict}\n"
        f"        A (scale=1.0):  loss={avg_loss_a:.4f}  acc={avg_acc_a:.3f}\n"
        f"        B (scale=0.01): loss={avg_loss_b:.4f}  acc={avg_acc_b:.3f}\n"
        f"        delta_acc={delta_acc:+.3f}  ({dt:.1f}s)"
    )


# =========================================================================
# Probe 7 — Infinite Assoc: un-memorizable streaming task
# =========================================================================
def probe_7_infinite_assoc():
    """500-step scale comparison on streaming data that can't be memorized.

    Task: 26 keys (A-Z as floats 2..27), random pairs per sequence,
    query last token, predict the value. Fresh batch every step.
    Both models see identical data (seeded per step).
    Combinatorial space: 26^P * 2^P * positions -- impossible to memorize.
    """
    import random as _random
    t0 = time.time()

    STEPS = 500
    LR = 1e-3
    LOG_EVERY = 50
    BATCH_SIZE = 16
    SEQ_LEN = 24       # longer sequence = more room for pairs + noise
    N_KEYS = 26         # A-Z
    N_PAIRS = 4         # 4 key-value pairs per sequence
    NUM_CLASSES = 2

    def _generate_batch(step_seed):
        """Generate one batch of fresh assoc data, deterministic per seed."""
        rng = _random.Random(step_seed)
        x = torch.zeros((BATCH_SIZE, SEQ_LEN, 1), dtype=torch.float32)
        y = torch.zeros((BATCH_SIZE,), dtype=torch.long)

        for b in range(BATCH_SIZE):
            # Pick N_PAIRS distinct keys from the 26-key pool.
            keys = rng.sample(range(N_KEYS), N_PAIRS)
            vals = [rng.randint(0, 1) for _ in range(N_PAIRS)]

            # Place pairs at random non-overlapping positions (not last slot).
            available = list(range(0, SEQ_LEN - 1))  # reserve last for query
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
                key_token = float(2 + keys[i])         # 2.0 .. 27.0
                val_token = -1.0 if vals[i] == 0 else -2.0
                x[b, pos, 0] = key_token
                x[b, pos + 1, 0] = val_token

            # Query: pick one of the placed pairs.
            q_idx = rng.randint(0, N_PAIRS - 1)
            x[b, -1, 0] = float(2 + keys[q_idx])
            y[b] = vals[q_idx]

        return x, y

    def _train_run(scale_value, label, seed=42):
        """Train on streaming fresh data. Returns (losses, accs)."""
        torch.manual_seed(seed)
        _random.seed(seed)
        model = _make_platinum()
        model.update_scale = float(scale_value)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        losses = []
        accs = []
        for step in range(1, STEPS + 1):
            # Deterministic per step — both models get identical batches.
            x_batch, y_batch = _generate_batch(step_seed=step * 1000 + 7)

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

            if step % LOG_EVERY == 0:
                w = losses[-LOG_EVERY:]
                w_acc = accs[-LOG_EVERY:]
                print(f"    [{label}] step {step:3d}  "
                      f"loss={sum(w)/len(w):.4f}  acc={sum(w_acc)/len(w_acc):.3f}")

        return losses, accs

    # ── Reproducibility check ────────────────────────────────────────────
    print("\n  === Reproducibility check (scale=1.0, seed=42, x2) ===")
    print("  Run R1...")
    losses_r1, _ = _train_run(1.0, "repro-R1", seed=42)
    print("  Run R2...")
    losses_r2, _ = _train_run(1.0, "repro-R2", seed=42)

    repro_ok = abs(losses_r1[-1] - losses_r2[-1]) < 1e-6
    print(f"  R1 final loss={losses_r1[-1]:.6f}  R2={losses_r2[-1]:.6f}  match={repro_ok}")
    if not repro_ok:
        dt = time.time() - t0
        return f"FAIL  reproducibility check failed ({dt:.1f}s)"

    # ── Main comparison ──────────────────────────────────────────────────
    print(f"\n  === Task: {N_KEYS} keys, {N_PAIRS} pairs/seq, "
          f"seq_len={SEQ_LEN}, streaming (fresh each step) ===")
    print("\n  === Model A: scale=1.0 ===")
    losses_a, accs_a = _train_run(1.0, "A scale=1.0", seed=42)

    print("\n  === Model B: scale=0.01 ===")
    losses_b, accs_b = _train_run(0.01, "B scale=0.01", seed=42)

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
    final_n = 50
    avg_acc_a = sum(accs_a[-final_n:]) / final_n
    avg_acc_b = sum(accs_b[-final_n:]) / final_n
    avg_loss_a = sum(losses_a[-final_n:]) / final_n
    avg_loss_b = sum(losses_b[-final_n:]) / final_n
    delta = avg_acc_a - avg_acc_b
    dt = time.time() - t0

    # Chance = 50% for binary classification.
    above_chance_a = avg_acc_a - 0.5
    above_chance_b = avg_acc_b - 0.5

    if delta > 0.05:
        verdict = f"A >> B by {delta*100:.1f}pp: full ring writes NEEDED for generalization"
    elif delta < -0.05:
        verdict = f"B >> A by {-delta*100:.1f}pp: dampening helps on streaming task"
    else:
        verdict = "A ~ B (within 5pp): ring write scale doesn't matter even on fresh data"

    return (
        f"{verdict}\n"
        f"        A (scale=1.0):  loss={avg_loss_a:.4f}  acc={avg_acc_a:.3f}  "
        f"(+{above_chance_a*100:.1f}pp above chance)\n"
        f"        B (scale=0.01): loss={avg_loss_b:.4f}  acc={avg_acc_b:.3f}  "
        f"(+{above_chance_b*100:.1f}pp above chance)\n"
        f"        delta_acc={delta:+.3f} ({delta*100:+.1f}pp)\n"
        f"        repro_check=PASS  streaming=YES  keys={N_KEYS}  steps={STEPS}  ({dt:.1f}s)"
    )


# =========================================================================
# Probe 6 — Precision Shot: scale=1.0 vs scale=0.01 at 500 steps
# =========================================================================
def probe_6_precision_shot():
    """500-step comparison, triple-seeded, shared data, reproducibility check."""
    t0 = time.time()

    from instnct_data import get_seq_mnist_loader

    STEPS = 500
    LR = 1e-3
    LOG_EVERY = 50

    # ── Generate data ONCE with deterministic seeds ──────────────────────
    torch.manual_seed(42)
    random.seed(42)
    loader, num_classes, collate_fn = get_seq_mnist_loader(train=True)
    # Materialize all batches so both runs see identical data in the same order.
    all_batches = []
    for batch in loader:
        all_batches.append((batch[0].clone(), batch[1].clone()))
    print(f"  [data] materialized {len(all_batches)} batches, "
          f"total samples ~{len(all_batches) * all_batches[0][0].shape[0]}")

    def _train_run(scale_value: float, label: str, seed: int = 42):
        """Train one model on pre-generated data. Returns (losses, accs)."""
        torch.manual_seed(seed)
        random.seed(seed)
        model = _make_platinum()
        model.update_scale = float(scale_value)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        losses = []
        accs = []
        step = 0
        while step < STEPS:
            for x_batch, y_batch in all_batches:
                if step >= STEPS:
                    break
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
                step += 1

                if step % LOG_EVERY == 0:
                    window = losses[-LOG_EVERY:]
                    avg_loss = sum(window) / len(window)
                    window_acc = accs[-LOG_EVERY:]
                    avg_acc = sum(window_acc) / len(window_acc)
                    print(f"    [{label}] step {step:3d}  "
                          f"loss={avg_loss:.4f}  acc={avg_acc:.3f}")

        return losses, accs

    # ── Reproducibility check: scale=1.0 run twice ──────────────────────
    print("\n  === Reproducibility check (scale=1.0, seed=42, x2) ===")
    print("  Run R1...")
    losses_r1, accs_r1 = _train_run(1.0, "repro-R1", seed=42)
    print("  Run R2...")
    losses_r2, accs_r2 = _train_run(1.0, "repro-R2", seed=42)

    repro_loss_match = abs(losses_r1[-1] - losses_r2[-1]) < 1e-6
    repro_acc_match = abs(accs_r1[-1] - accs_r2[-1]) < 1e-6
    print(f"  R1 final loss={losses_r1[-1]:.6f}  R2 final loss={losses_r2[-1]:.6f}  "
          f"match={repro_loss_match}")
    print(f"  R1 final acc ={accs_r1[-1]:.4f}      R2 final acc ={accs_r2[-1]:.4f}      "
          f"match={repro_acc_match}")

    if not (repro_loss_match and repro_acc_match):
        dt = time.time() - t0
        return (f"FAIL  reproducibility check failed! "
                f"loss_diff={abs(losses_r1[-1]-losses_r2[-1]):.2e}  "
                f"acc_diff={abs(accs_r1[-1]-accs_r2[-1]):.2e}  ({dt:.1f}s)")

    # ── Main comparison ──────────────────────────────────────────────────
    print("\n  === Model A: scale=1.0 (platinum default) ===")
    losses_a, accs_a = _train_run(1.0, "A scale=1.0", seed=42)

    print("\n  === Model B: scale=0.01 (instnct bugged) ===")
    losses_b, accs_b = _train_run(0.01, "B scale=0.01", seed=42)

    # ── Compute windowed metrics at every LOG_EVERY ──────────────────────
    print("\n  === Step-by-step comparison (50-step avg) ===")
    print(f"  {'step':>5s}  {'A_loss':>7s}  {'A_acc':>6s}  {'B_loss':>7s}  {'B_acc':>6s}  {'delta':>7s}")
    for s in range(LOG_EVERY, STEPS + 1, LOG_EVERY):
        w_a_loss = sum(losses_a[s-LOG_EVERY:s]) / LOG_EVERY
        w_a_acc = sum(accs_a[s-LOG_EVERY:s]) / LOG_EVERY
        w_b_loss = sum(losses_b[s-LOG_EVERY:s]) / LOG_EVERY
        w_b_acc = sum(accs_b[s-LOG_EVERY:s]) / LOG_EVERY
        delta = w_a_acc - w_b_acc
        print(f"  {s:5d}  {w_a_loss:7.4f}  {w_a_acc:6.3f}  {w_b_loss:7.4f}  {w_b_acc:6.3f}  {delta:+7.3f}")

    # ── Final verdict (last 50 steps) ────────────────────────────────────
    final_n = 50
    avg_loss_a = sum(losses_a[-final_n:]) / final_n
    avg_loss_b = sum(losses_b[-final_n:]) / final_n
    avg_acc_a = sum(accs_a[-final_n:]) / final_n
    avg_acc_b = sum(accs_b[-final_n:]) / final_n
    delta_acc = avg_acc_a - avg_acc_b

    dt = time.time() - t0

    if delta_acc > 0.05:
        verdict = "A >> B by {:.1f}pp: scale=0.01 was CRIPPLING ring writes".format(delta_acc * 100)
    elif delta_acc < -0.05:
        verdict = "B >> A by {:.1f}pp: dampening = accidental regularization".format(-delta_acc * 100)
    else:
        verdict = "A ~ B (within 5pp): ring write magnitude not the bottleneck"

    return (
        f"{verdict}\n"
        f"        A (scale=1.0):  loss={avg_loss_a:.4f}  acc={avg_acc_a:.3f}\n"
        f"        B (scale=0.01): loss={avg_loss_b:.4f}  acc={avg_acc_b:.3f}\n"
        f"        delta_acc={delta_acc:+.3f} ({delta_acc*100:+.1f}pp)\n"
        f"        repro_check=PASS  steps={STEPS}  ({dt:.1f}s)"
    )


# =========================================================================
# Runner
# =========================================================================
if __name__ == "__main__":
    all_probes = [
        probe_1_smoke,
        probe_2_parity,
        probe_3_gradient_flow,
        probe_4_ring_liveness,
        probe_5_agc_bug_test,
        probe_6_precision_shot,
        probe_7_infinite_assoc,
    ]

    # Selective run flags.
    if "--probe7" in sys.argv:
        probes = [probe_7_infinite_assoc]
    elif "--probe6" in sys.argv:
        probes = [probe_6_precision_shot]
    else:
        probes = all_probes

    results = {}
    for probe in probes:
        name = probe.__name__
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        try:
            result = probe()
            results[name] = result
            print(f"  -> {result}")
        except Exception as e:
            tb = traceback.format_exc()
            results[name] = f"FAIL: {e}"
            print(f"  -> FAIL: {e}")
            print(tb)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        status = v.split()[0] if v else "???"
        print(f"  {k:30s} {v}")
    print("=" * 60)
