"""Probe 10 — Harmonic-Weighted Fibonacci Swarm Full Run.

Modes:
  (default)   2500-step harmonic-weighted training + strong eval
  --compare   50-step comparison: harmonic vs stride vs uniform
  --quick     250-step quick training run (for testing)

Volume weighting: all ants run every step, but contribution is scaled
  by ring size (volume). Bigger ring = bigger voice.
  ant[0] (biggest ring, slow anchor)    weight = ring_len  (strongest)
  ant[N] (smallest ring, fast refiner)  weight = ring_len  (weakest)
Weights are normalized (divide by sum).

Run:  python tools/_scratch/probe10_fib_smoke.py
From: S:/AI/work/VRAXION_DEV/Golden Draft/
"""

import argparse
import math
import os
import random as _random
import sys
import time
from collections import deque

# ── Dashboard log path (shared with live_dashboard.py) ──────────────────
# All probes MUST write dashboard-compatible log lines so the Streamlit
# dashboard at :8501 can visualize training in real time.
_PROBE_LOG_DIR = os.path.join(r"S:\AI\work\VRAXION_DEV\Golden Draft", "logs", "probe")
_PROBE_LOG_DEFAULT = os.path.join(_PROBE_LOG_DIR, "probe_live.log")

# ── Path setup ──────────────────────────────────────────────────────────
_golden_code = r"S:\AI\Golden Code"
_golden_draft = r"S:\AI\work\VRAXION_DEV\Golden Draft"
for p in [_golden_code, _golden_draft, os.path.join(_golden_draft, "tools")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("VRX_SYNTH", "1")
os.environ.setdefault("VRX_SYNTH_MODE", "assoc_clean")
os.environ["VRX_SENSORY_RING"] = "0"
os.environ["VRX_VAULT"] = "0"
os.environ["VRX_THINK_RING"] = "1"
os.environ["VRX_THINK_RING_DUAL"] = "0"
os.environ["VRX_THINK_RING_BRAINSTEM"] = "0"
# Enable Fibonacci swarm.
os.environ["VRX_PRISMION"] = "1"
os.environ["VRX_PRISMION_N"] = "4"
os.environ["VRX_PRISMION_LEN"] = "64"
os.environ["VRX_PRISMION_TOPOLOGY"] = "bank"
os.environ["VRX_PRISMION_FIBONACCI"] = "1"
os.environ["VRX_PRISMION_FIB_BUDGET_MB"] = "1.2"  # 300K param budget, 6 ants (safe on 16GB)
os.environ["VRX_PRISMION_FIB_MAX_ANTS"] = "16"

import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_KW = dict(input_dim=1, num_classes=2, ring_len=64, slot_dim=32)
LR = 1e-3
BATCH_SIZE = 16

# Harmonic task config.
HARMONIC_SEQ_LEN = 64
F_SLOW = 1.0   # 1 cycle per sequence (the "regime" signal)
F_FAST = 7.0   # 7 cycles per sequence (the "detail" signal)
A_SLOW = 1.0
A_FAST = 0.5


# ── Dashboard log writer ────────────────────────────────────────────────

import json as _json

_log_fh = None       # dashboard step log (text)
_telem_fh = None     # per-ant telemetry (JSONL)
_telem_path = None


def _open_log(log_path):
    """Open (truncate) the dashboard log + ant telemetry JSONL. Called once."""
    global _log_fh, _telem_fh, _telem_path
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log_fh = open(log_path, "w", encoding="utf-8")
    _telem_path = log_path.replace(".log", "_ant_telemetry.jsonl")
    _telem_fh = open(_telem_path, "w", encoding="utf-8")
    print(f"  Dashboard log:    {log_path}")
    print(f"  Ant telemetry:    {_telem_path}")
    return log_path


def _log_step(step, loss, acc, s_per_step, label="", ant_data=None, rolling=None):
    """Write one dashboard-compatible line + one JSONL telemetry line.

    Dashboard format: step N | loss X.XXXXXX | acc=X.XXXX RD:X.XXXX traction=X.XXXX shard=0/0
    Telemetry JSONL: {"step":N, "loss":X, "acc":X, "ants":[...], "rolling":{...}}
    """
    if _log_fh is not None:
        line = (f"step {step} | loss {loss:.6f} | "
                f"acc={acc:.4f} RD:{s_per_step:.4f} traction={acc:.4f} shard=0/0")
        if label:
            line += f" [{label}]"
        _log_fh.write(line + "\n")
        _log_fh.flush()

    if _telem_fh is not None:
        row = {"step": step, "loss": round(loss, 6), "acc": round(acc, 4),
               "s_per_step": round(s_per_step, 3)}
        if ant_data:
            row["ants"] = ant_data
        if rolling:
            row["rolling"] = rolling
        _telem_fh.write(_json.dumps(row) + "\n")
        _telem_fh.flush()


def _close_log():
    global _log_fh, _telem_fh
    for fh in (_log_fh, _telem_fh):
        if fh is not None:
            fh.close()
    _log_fh = None
    _telem_fh = None


# ── Data generation ─────────────────────────────────────────────────────

def _generate_harmonic_batch(step_seed, return_components=False):
    """Harmonic XOR: x[t] = slow_sine + fast_sine, label = XOR(sign_slow, sign_fast).

    Both frequency components are needed to solve the task.
    One component alone -> 50% accuracy (chance).

    If return_components=True, returns (x, y_xor, y_slow, y_fast).
    """
    rng = _random.Random(step_seed)
    sl = HARMONIC_SEQ_LEN
    x = torch.zeros((BATCH_SIZE, sl, 1), dtype=torch.float32)
    y_xor = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    y_slow = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    y_fast = torch.zeros((BATCH_SIZE,), dtype=torch.long)

    for b in range(BATCH_SIZE):
        phase_slow = rng.uniform(0, 2 * math.pi)
        phase_fast = rng.uniform(0, 2 * math.pi)

        for t in range(sl):
            frac = t / sl
            slow = A_SLOW * math.sin(2 * math.pi * F_SLOW * frac + phase_slow)
            fast = A_FAST * math.sin(2 * math.pi * F_FAST * frac + phase_fast)
            x[b, t, 0] = slow + fast

        # Label: XOR of component signs at the last timestep.
        frac_end = (sl - 1) / sl
        slow_end = math.sin(2 * math.pi * F_SLOW * frac_end + phase_slow)
        fast_end = math.sin(2 * math.pi * F_FAST * frac_end + phase_fast)
        slow_sign = 1 if slow_end >= 0 else 0
        fast_sign = 1 if fast_end >= 0 else 0
        y_xor[b] = slow_sign ^ fast_sign
        y_slow[b] = slow_sign
        y_fast[b] = fast_sign

    x = x.to(DEVICE)
    y_xor = y_xor.to(DEVICE)
    y_slow = y_slow.to(DEVICE)
    y_fast = y_fast.to(DEVICE)
    if return_components:
        return x, y_xor, y_slow, y_fast
    return x, y_xor


# ── Model creation ──────────────────────────────────────────────────────

def _create_model():
    from vraxion.platinum.hallway import AbsoluteHallway as PlatinumHallway
    model = PlatinumHallway(**MODEL_KW)
    model.update_scale = 1.0
    model = model.to(DEVICE)
    return model


def _verify_swarm(model):
    """Verify Fibonacci swarm is active. Returns (n_ants, configs) or raises."""
    if not getattr(model, "prismion_fib_active", False):
        raise RuntimeError("Fibonacci swarm not active. Check env vars.")
    if model.prismion_swarm is None:
        raise RuntimeError("prismion_swarm is None.")
    n_ants = len(model.prismion_swarm)
    configs = model.prismion_swarm_configs
    return n_ants, configs


def _print_model_info(model, n_ants, configs, steps, label=""):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} total params")
    print(f"  Fibonacci ants: {n_ants}")
    for j, spec in enumerate(configs):
        ant_params = sum(p.numel() for p in model.prismion_swarm[j].parameters())
        head_params = sum(p.numel() for p in model.prismion_swarm_heads[j].parameters())
        print(f"    ant[{j}]: ring_len={spec['ring_len']:>4d}  "
              f"slot_dim={spec['slot_dim']:>4d}  "
              f"frac={spec['fraction']:.4f}  "
              f"params={ant_params + head_params:,}")
    if label:
        print(f"  Mode: {label}")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM: {vram_total:.1f} GB")
    print(f"  Task: Harmonic XOR, seq_len={HARMONIC_SEQ_LEN}, {steps} steps")


# ── Monkey-patches ──────────────────────────────────────────────────────

def _patch_harmonic_weighted(model, n_ants, per_step_ant_data, prev_msgs,
                             active_set=None):
    """Monkey-patch: all ants every step, volume-weighted contributions.

    weight[i] = ring_len[i]  →  bigger ring = bigger voice.
    ant[0] (biggest ring, slow anchor)   = strongest weight.
    ant[N] (smallest ring, fast refiner) = weakest weight.
    Normalized so weights sum to 1.

    If active_set is provided, inactive ants get weight=0 (no vote, no gradient).
    Also stores last per-ant logits in model._last_ant_logits for telemetry.
    """
    raw_weights = [float(model.prismion_swarm_configs[i]["ring_len"]) for i in range(n_ants)]
    # Zero out inactive ants.
    if active_set is not None:
        for i in range(n_ants):
            if i not in active_set:
                raw_weights[i] = 0.0
    w_sum = sum(raw_weights)
    if w_sum == 0:
        w_sum = 1.0  # safety: avoid div-by-zero if no ants active
    norm_weights = [w / w_sum for w in raw_weights]
    # Shared storage: training loop reads these after forward pass.
    model._last_ant_logits = [None] * n_ants
    model._last_ant_ring_vars = [0.0] * n_ants

    def harmonic_fib_apply(chrom, fib_prism_states, active_mask):
        assert model.prismion_swarm is not None
        assert model.prismion_swarm_heads is not None

        N = len(model.prismion_swarm)
        B = int(chrom.size(0))
        swarm_logits = torch.zeros(B, int(model.num_classes),
                                   device=chrom.device, dtype=chrom.dtype)
        feedback_msgs = []
        new_states = list(fib_prism_states)
        ant_metrics = []

        for i in range(N):
            ant = model.prismion_swarm[i]
            head = model.prismion_swarm_heads[i]
            msg_i, st_i = ant.step(chrom, fib_prism_states[i], active_mask)
            new_states[i] = st_i

            # Harmonic weighting: scale both feedback and logit contributions.
            w = norm_weights[i]
            feedback_msgs.append(msg_i * w)
            logit_i = head(msg_i.to(head.weight.dtype))
            swarm_logits = swarm_logits + (logit_i.to(swarm_logits.dtype) * w)

            # Store for telemetry (overwritten each timestep; last one is used).
            model._last_ant_logits[i] = logit_i.detach()
            model._last_ant_ring_vars[i] = float(st_i.ring.detach().float().var().item())

            # Instrumentation.
            msg_norm = float(msg_i.detach().float().norm().item())
            logit_norm = float(logit_i.detach().float().norm().item())
            logit_contrib = float((logit_i * w).detach().float().norm().item())
            msg_delta = 0.0
            if prev_msgs[i] is not None:
                msg_delta = float((msg_i.detach().float() - prev_msgs[i]).norm().item())
            prev_msgs[i] = msg_i.detach().float()

            ant_metrics.append({
                "msg_norm": msg_norm,
                "msg_delta": msg_delta,
                "logit_norm": logit_norm,
                "logit_contrib": logit_contrib,
                "weight": w,
            })

        per_step_ant_data.append(ant_metrics)

        # Feedback: weighted sum (already weighted), stack and sum.
        feedback = torch.stack(feedback_msgs, dim=0).sum(dim=0)
        return feedback, new_states, swarm_logits

    model._prismion_apply_fibonacci = harmonic_fib_apply
    return norm_weights


def _patch_uniform(model, n_ants):
    """Monkey-patch: all ants every step, equal weight (original behavior)."""
    # Original: each ant contributes equally, feedback = mean.
    # No instrumentation needed for comparison mode.
    def uniform_fib_apply(chrom, fib_prism_states, active_mask):
        assert model.prismion_swarm is not None
        assert model.prismion_swarm_heads is not None

        N = len(model.prismion_swarm)
        B = int(chrom.size(0))
        swarm_logits = torch.zeros(B, int(model.num_classes),
                                   device=chrom.device, dtype=chrom.dtype)
        feedback_msgs = []
        new_states = list(fib_prism_states)

        for i in range(N):
            ant = model.prismion_swarm[i]
            head = model.prismion_swarm_heads[i]
            msg_i, st_i = ant.step(chrom, fib_prism_states[i], active_mask)
            new_states[i] = st_i
            feedback_msgs.append(msg_i)
            swarm_logits = swarm_logits + head(msg_i.to(head.weight.dtype)).to(swarm_logits.dtype)

        feedback = torch.stack(feedback_msgs, dim=0).mean(dim=0)
        return feedback, new_states, swarm_logits

    model._prismion_apply_fibonacci = uniform_fib_apply


def _patch_fibonacci_stride(model, n_ants):
    """Monkey-patch: tiered Fibonacci strides — skip big ants on most steps.

    Stride pattern (fib-like): ant[0]=13, ant[1]=8, ant[2]=5, ant[3]=3
    Ant only runs when global_step % stride == 0.
    """
    strides = [13, 8, 5, 3][:n_ants]
    # Pad or trim to match n_ants.
    while len(strides) < n_ants:
        strides.append(1)

    call_counter = [0]  # mutable counter across calls

    def stride_fib_apply(chrom, fib_prism_states, active_mask):
        assert model.prismion_swarm is not None
        assert model.prismion_swarm_heads is not None

        N = len(model.prismion_swarm)
        B = int(chrom.size(0))
        swarm_logits = torch.zeros(B, int(model.num_classes),
                                   device=chrom.device, dtype=chrom.dtype)
        feedback_msgs = []
        new_states = list(fib_prism_states)
        step_idx = call_counter[0]
        call_counter[0] += 1

        for i in range(N):
            ant = model.prismion_swarm[i]
            head = model.prismion_swarm_heads[i]

            if step_idx % strides[i] == 0:
                msg_i, st_i = ant.step(chrom, fib_prism_states[i], active_mask)
                new_states[i] = st_i
                feedback_msgs.append(msg_i)
                swarm_logits = swarm_logits + head(msg_i.to(head.weight.dtype)).to(swarm_logits.dtype)
            else:
                # Re-use last state's msg_ema as stale message.
                new_states[i] = fib_prism_states[i]
                stale_msg = fib_prism_states[i].msg_ema
                feedback_msgs.append(stale_msg)
                swarm_logits = swarm_logits + head(stale_msg.to(head.weight.dtype)).to(swarm_logits.dtype)

        feedback = torch.stack(feedback_msgs, dim=0).mean(dim=0)
        return feedback, new_states, swarm_logits

    model._prismion_apply_fibonacci = stride_fib_apply


# ── Training loop ───────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, step, ckpt_dir, active_set=None):
    """Save model + optimizer state to checkpoint file."""
    path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "active_set": sorted(active_set) if active_set else None,
    }, path)
    # Also save as 'latest' for easy resuming.
    latest = os.path.join(ckpt_dir, "ckpt_latest.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "active_set": sorted(active_set) if active_set else None,
    }, latest)
    return path


def _train(model, steps, log_interval=50, label="", ckpt_dir=None, save_every=100,
           active_set=None):
    """Train on harmonic XOR. Returns (losses, accs, elapsed)."""
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    losses = []
    accs = []

    has_swarm = (getattr(model, "prismion_swarm", None) is not None
                 and getattr(model, "prismion_swarm_heads", None) is not None)
    n_ants = len(model.prismion_swarm) if has_swarm else 0

    # Rolling accumulation buffers: window=100 steps * batch=16 = 1600 samples.
    # Stores per-step prediction tensors; concat on demand for clean accuracy.
    ROLLING_WINDOW = 100
    _roll_y_xor = deque(maxlen=ROLLING_WINDOW)
    _roll_y_slow = deque(maxlen=ROLLING_WINDOW)
    _roll_y_fast = deque(maxlen=ROLLING_WINDOW)
    _roll_combined = deque(maxlen=ROLLING_WINDOW)  # combined model predictions
    _roll_ant_preds = [deque(maxlen=ROLLING_WINDOW) for _ in range(n_ants)]

    t0 = time.time()
    for step in range(1, steps + 1):
        x_batch, y_batch, y_slow, y_fast = _generate_harmonic_batch(
            step_seed=step * 1000 + 7, return_components=True)
        optimizer.zero_grad()
        out = model(x_batch)
        logits = out[0]
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        # ── Per-ant telemetry (after backward, before optimizer.step) ────
        ant_data = None
        last_logits = getattr(model, "_last_ant_logits", None)
        last_rvars = getattr(model, "_last_ant_ring_vars", None)
        if has_swarm and n_ants > 0 and last_logits is not None:
            ant_data = []
            for i in range(n_ants):
                ant = model.prismion_swarm[i]
                head = model.prismion_swarm_heads[i]
                # Gradient norm across ant + head params.
                gnorm = 0.0
                for p in list(ant.parameters()) + list(head.parameters()):
                    if p.grad is not None:
                        gnorm += float(p.grad.detach().float().norm().item()) ** 2
                gnorm = gnorm ** 0.5

                # Per-ant vote + accuracy from stored last-timestep logits.
                ant_logit = last_logits[i]
                if ant_logit is not None:
                    preds_i = ant_logit.argmax(dim=-1)
                    vote = int(preds_i.float().mean().item() + 0.5)
                    ant_acc_xor = float((preds_i == y_batch).float().mean().item())
                    ant_acc_slow = float((preds_i == y_slow).float().mean().item())
                    ant_acc_fast = float((preds_i == y_fast).float().mean().item())
                else:
                    vote, ant_acc_xor, ant_acc_slow, ant_acc_fast = -1, 0.0, 0.0, 0.0

                ring_var = last_rvars[i] if last_rvars else 0.0

                ant_data.append({
                    "ring": int(model.prismion_swarm_configs[i]["ring_len"]),
                    "gnorm": round(gnorm, 4),
                    "vote": vote,
                    "xor": round(ant_acc_xor, 3),
                    "slow": round(ant_acc_slow, 3),
                    "fast": round(ant_acc_fast, 3),
                    "rvar": round(ring_var, 4),
                })

        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y_batch).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

        # ── Rolling accumulation (1600-sample window for clean accuracy) ──
        _roll_y_xor.append(y_batch.detach())
        _roll_y_slow.append(y_slow.detach())
        _roll_y_fast.append(y_fast.detach())
        _roll_combined.append(preds.detach())
        if has_swarm and last_logits is not None:
            for i in range(n_ants):
                if last_logits[i] is not None:
                    _roll_ant_preds[i].append(last_logits[i].argmax(dim=-1))

        rolling_data = None
        n_buf = len(_roll_y_xor)
        if n_buf >= 10:  # need 160+ samples for meaningful signal
            cat_y_xor = torch.cat(list(_roll_y_xor))
            cat_y_slow = torch.cat(list(_roll_y_slow))
            cat_y_fast = torch.cat(list(_roll_y_fast))
            cat_comb = torch.cat(list(_roll_combined))
            r_comb = float((cat_comb == cat_y_xor).float().mean())

            rolling_data = {"n": n_buf, "combined_xor": round(r_comb, 4)}

            if has_swarm and ant_data is not None:
                r_ants = []
                for i in range(n_ants):
                    if len(_roll_ant_preds[i]) == n_buf:
                        cat_ant = torch.cat(list(_roll_ant_preds[i]))
                        r_xor_i = float((cat_ant == cat_y_xor).float().mean())
                        r_slow_i = float((cat_ant == cat_y_slow).float().mean())
                        r_fast_i = float((cat_ant == cat_y_fast).float().mean())
                        ant_data[i]["r_xor"] = round(r_xor_i, 4)
                        ant_data[i]["r_slow"] = round(r_slow_i, 4)
                        ant_data[i]["r_fast"] = round(r_fast_i, 4)
                        r_ants.append({"r_xor": round(r_xor_i, 4),
                                       "r_slow": round(r_slow_i, 4),
                                       "r_fast": round(r_fast_i, 4)})
                rolling_data["ants"] = r_ants

        # Write EVERY step to dashboard log + JSONL telemetry.
        elapsed = time.time() - t0
        s_per_step = elapsed / step
        _log_step(step, loss.item(), acc, s_per_step, label=label,
                  ant_data=ant_data, rolling=rolling_data)

        if step % log_interval == 0:
            w = min(log_interval, len(losses))
            avg_loss = sum(losses[-w:]) / w
            avg_acc = sum(accs[-w:]) / w
            eta = s_per_step * (steps - step)
            prefix = f"  [{label}]" if label else "   "
            print(f"{prefix} step {step:>5d}/{steps}  "
                  f"loss={avg_loss:.4f}  acc={avg_acc:.3f}  "
                  f"({s_per_step:.2f} s/step, ETA {eta/60:.1f}m)")

        # ── Periodic checkpoint ──
        if ckpt_dir and save_every and step % save_every == 0:
            ckpt_path = _save_checkpoint(model, optimizer, step, ckpt_dir, active_set)
            print(f"  >> Checkpoint saved: {ckpt_path}")

    # Final checkpoint.
    if ckpt_dir:
        ckpt_path = _save_checkpoint(model, optimizer, steps, ckpt_dir, active_set)
        print(f"  >> Final checkpoint: {ckpt_path}")

    elapsed = time.time() - t0
    return losses, accs, elapsed


# ── Evaluation ──────────────────────────────────────────────────────────

def _strong_eval(model, n_ants, configs, norm_weights):
    """Strong evaluation: 100 fresh batches, per-ant solo + component accuracy."""
    N_EVAL_BATCHES = 100
    SEED_BASE = 100000

    model.eval()
    print(f"\n{'='*70}")
    print(f"EVAL ({N_EVAL_BATCHES} batches, {N_EVAL_BATCHES * BATCH_SIZE} samples)")
    print(f"{'='*70}")

    # Collect all eval data.
    all_x, all_y_xor, all_y_slow, all_y_fast = [], [], [], []
    for i in range(N_EVAL_BATCHES):
        x, y_xor, y_slow, y_fast = _generate_harmonic_batch(
            step_seed=SEED_BASE + i, return_components=True)
        all_x.append(x)
        all_y_xor.append(y_xor)
        all_y_slow.append(y_slow)
        all_y_fast.append(y_fast)

    all_x = torch.cat(all_x, dim=0)
    all_y_xor = torch.cat(all_y_xor, dim=0)
    all_y_slow = torch.cat(all_y_slow, dim=0)
    all_y_fast = torch.cat(all_y_fast, dim=0)
    N_samples = all_x.size(0)

    # ── Combined accuracy (full model) ──────────────────────────────────
    combined_correct = 0
    with torch.no_grad():
        for i in range(N_EVAL_BATCHES):
            s = i * BATCH_SIZE
            e = s + BATCH_SIZE
            out = model(all_x[s:e])
            preds = out[0].argmax(dim=1)
            combined_correct += (preds == all_y_xor[s:e]).sum().item()
    combined_acc = combined_correct / N_samples * 100

    # ── Main-only accuracy (zero out fib_swarm_logits) ──────────────────
    # Monkey-patch to make swarm return zeros.
    saved_fn = model._prismion_apply_fibonacci

    def zero_swarm(chrom, fib_prism_states, active_mask):
        feedback, new_states, swarm_logits = saved_fn(chrom, fib_prism_states, active_mask)
        return feedback, new_states, torch.zeros_like(swarm_logits)

    model._prismion_apply_fibonacci = zero_swarm
    main_correct = 0
    with torch.no_grad():
        for i in range(N_EVAL_BATCHES):
            s = i * BATCH_SIZE
            e = s + BATCH_SIZE
            out = model(all_x[s:e])
            preds = out[0].argmax(dim=1)
            main_correct += (preds == all_y_xor[s:e]).sum().item()
    main_acc = main_correct / N_samples * 100
    model._prismion_apply_fibonacci = saved_fn

    print(f"\n  Combined acc:     {combined_acc:5.1f}%")
    print(f"  Main-only acc:    {main_acc:5.1f}%  (fib logits zeroed)")

    # ── Per-ant solo accuracy ───────────────────────────────────────────
    # For each ant: zero out all OTHER ants, test solo.
    print(f"\n  Per-ant solo accuracy:")
    print(f"  {'ant':>6s}  {'ring':>6s}  {'XOR':>8s}  {'slow':>8s}  {'fast':>8s}  {'note':>20s}")

    for a in range(n_ants):
        # Patch: only ant[a] contributes to swarm_logits.
        def make_solo_fn(ant_idx):
            def solo_fib_apply(chrom, fib_prism_states, active_mask):
                assert model.prismion_swarm is not None
                assert model.prismion_swarm_heads is not None
                N = len(model.prismion_swarm)
                B = int(chrom.size(0))
                swarm_logits = torch.zeros(B, int(model.num_classes),
                                           device=chrom.device, dtype=chrom.dtype)
                feedback_msgs = []
                new_states = list(fib_prism_states)
                for i in range(N):
                    ant = model.prismion_swarm[i]
                    head = model.prismion_swarm_heads[i]
                    msg_i, st_i = ant.step(chrom, fib_prism_states[i], active_mask)
                    new_states[i] = st_i
                    w = norm_weights[i]
                    feedback_msgs.append(msg_i * w)
                    if i == ant_idx:
                        swarm_logits = swarm_logits + (head(msg_i.to(head.weight.dtype)).to(swarm_logits.dtype) * w)
                feedback = torch.stack(feedback_msgs, dim=0).sum(dim=0)
                return feedback, new_states, swarm_logits
            return solo_fib_apply

        model._prismion_apply_fibonacci = make_solo_fn(a)

        xor_correct = 0
        slow_correct = 0
        fast_correct = 0
        with torch.no_grad():
            for i in range(N_EVAL_BATCHES):
                s = i * BATCH_SIZE
                e = s + BATCH_SIZE
                out = model(all_x[s:e])
                preds = out[0].argmax(dim=1)
                xor_correct += (preds == all_y_xor[s:e]).sum().item()
                slow_correct += (preds == all_y_slow[s:e]).sum().item()
                fast_correct += (preds == all_y_fast[s:e]).sum().item()

        xor_acc = xor_correct / N_samples * 100
        slow_acc = slow_correct / N_samples * 100
        fast_acc = fast_correct / N_samples * 100

        ring_len = configs[a]["ring_len"]
        if a == 0:
            note = "<- should track slow"
        elif a == n_ants - 1:
            note = "<- should track fast"
        else:
            note = ""
        print(f"    ant[{a}] ring={ring_len:>4d}  "
              f"XOR={xor_acc:5.1f}%  slow={slow_acc:5.1f}%  fast={fast_acc:5.1f}%  {note}")

    model._prismion_apply_fibonacci = saved_fn
    model.train()

    return combined_acc, main_acc


# ── Analysis ────────────────────────────────────────────────────────────

def _analyze_instrumentation(per_step_ant_data, n_ants, configs, seq_len, n_steps):
    """Analyze per-ant frequency behavior from instrumentation data."""
    print(f"\n{'='*70}")
    print("ANALYSIS: Per-ant frequency separation")
    print(f"{'='*70}")

    calls_per_step = seq_len
    total_calls = len(per_step_ant_data)
    actual_steps = total_calls // calls_per_step if calls_per_step > 0 else 0

    print(f"\n  Total fib_apply calls: {total_calls}")
    print(f"  Expected calls/step: {calls_per_step}")
    print(f"  Actual training steps captured: {actual_steps}")

    if total_calls < calls_per_step * 2:
        print("  Not enough data for analysis.")
        return

    # Aggregate per training step.
    step_ant_deltas = []
    step_ant_logit_contribs = []
    for s in range(min(actual_steps, n_steps)):
        start = s * calls_per_step
        end = start + calls_per_step
        if end > total_calls:
            break
        chunk = per_step_ant_data[start:end]
        ant_mean_delta = []
        ant_mean_contrib = []
        for a in range(n_ants):
            deltas = [c[a]["msg_delta"] for c in chunk]
            contribs = [c[a]["logit_contrib"] for c in chunk]
            ant_mean_delta.append(sum(deltas) / len(deltas))
            ant_mean_contrib.append(sum(contribs) / len(contribs))
        step_ant_deltas.append(ant_mean_delta)
        step_ant_logit_contribs.append(ant_mean_contrib)

    n_captured = len(step_ant_deltas)
    if n_captured < 10:
        print(f"  Only {n_captured} steps captured, need at least 10.")
        return

    # ── Message volatility ──────────────────────────────────────────────
    print(f"\n  --- Message volatility (mean |delta_msg| per step) ---")
    print(f"  {'ant':>4s}  {'ring_len':>8s}  {'weight':>8s}  {'volatility':>10s}")
    volatilities = []
    for a in range(n_ants):
        vol = sum(step_ant_deltas[s][a] for s in range(n_captured)) / n_captured
        volatilities.append(vol)
        spec = configs[a]
        w_raw = a + 1
        print(f"  {a:>4d}  {spec['ring_len']:>8d}  {w_raw:>8d}  {vol:>10.4f}")

    # ── Logit contribution (weighted) ──────────────────────────────────
    print(f"\n  --- Weighted logit contribution (mean per step) ---")
    print(f"  {'ant':>4s}  {'ring_len':>8s}  {'mean_contrib':>12s}  {'std_contrib':>12s}")
    for a in range(n_ants):
        vals = [step_ant_logit_contribs[s][a] for s in range(n_captured)]
        mean_c = sum(vals) / len(vals)
        var_c = sum((v - mean_c)**2 for v in vals) / len(vals)
        std_c = var_c ** 0.5
        print(f"  {a:>4d}  {configs[a]['ring_len']:>8d}  {mean_c:>12.4f}  {std_c:>12.4f}")

    # ── Autocorrelation (lag-1 of msg_delta) ────────────────────────────
    print(f"\n  --- Message autocorrelation (lag-1) ---")
    print(f"  {'ant':>4s}  {'ring_len':>8s}  {'autocorr':>10s}  {'band':>15s}")
    autocorrs = []
    for a in range(n_ants):
        series = [step_ant_deltas[s][a] for s in range(n_captured)]
        mean_s = sum(series) / len(series)
        var_s = sum((v - mean_s)**2 for v in series) / len(series)
        if var_s < 1e-12:
            ac = 0.0
        else:
            cov = sum((series[i] - mean_s) * (series[i+1] - mean_s)
                      for i in range(len(series) - 1)) / (len(series) - 1)
            ac = cov / var_s
        autocorrs.append(ac)
        if ac > 0.5:
            band = "LOW-FREQ (slow)"
        elif ac > 0.2:
            band = "MID-FREQ"
        elif ac > -0.1:
            band = "HIGH-FREQ (fast)"
        else:
            band = "ANTI-CORR"
        print(f"  {a:>4d}  {configs[a]['ring_len']:>8d}  {ac:>10.4f}  {band:>15s}")

    # ── Kendall tau (ring_len vs autocorr) ──────────────────────────────
    ring_lens = [configs[a]["ring_len"] for a in range(n_ants)]
    concordant = 0
    discordant = 0
    for i in range(n_ants):
        for j in range(i + 1, n_ants):
            ring_order = (ring_lens[i] > ring_lens[j]) - (ring_lens[i] < ring_lens[j])
            ac_order = (autocorrs[i] > autocorrs[j]) - (autocorrs[i] < autocorrs[j])
            if ring_order * ac_order > 0:
                concordant += 1
            elif ring_order * ac_order < 0:
                discordant += 1
    pairs = concordant + discordant
    tau = (concordant - discordant) / max(pairs, 1)
    print(f"\n  Kendall tau (ring_len vs autocorr): {tau:+.3f}")


# ── Main run: harmonic-weighted full training ───────────────────────────

def run_full(steps, log_path=_PROBE_LOG_DEFAULT, active_set=None,
             resume_path=None, save_every=100):
    print(f"\n{'='*70}")
    print(f"Probe 10 — Harmonic-Weighted Fibonacci Swarm ({steps} steps)")
    if active_set is not None:
        print(f"  Staged mode: active ants = {sorted(active_set)}")
    if resume_path:
        print(f"  Resuming from: {resume_path}")
    print(f"{'='*70}")
    _open_log(log_path)

    torch.manual_seed(42)
    _random.seed(42)

    model = _create_model()
    n_ants, configs = _verify_swarm(model)

    # Load checkpoint if resuming.
    if resume_path:
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint from {resume_path} (was step {ckpt.get('step', '?')})")

    _print_model_info(model, n_ants, configs, steps, label="Harmonic-weighted")

    # Volume weights info (ring_len = voice, zeroed for inactive).
    raw_weights = [float(configs[i]["ring_len"]) for i in range(n_ants)]
    if active_set is not None:
        for i in range(n_ants):
            if i not in active_set:
                raw_weights[i] = 0.0
    w_sum = sum(raw_weights) or 1.0
    print(f"\n  Volume weights{' (staged)' if active_set else ''}:")
    for i in range(n_ants):
        nw = raw_weights[i] / w_sum
        status = ""
        if active_set is not None:
            status = " ACTIVE" if i in active_set else " frozen"
        elif i == 0:
            status = " (anchor)"
        elif i == n_ants - 1:
            status = " (refiner)"
        print(f"    ant[{i}] ring={configs[i]['ring_len']:>4d}  "
              f"raw_w={raw_weights[i]:.0f}  norm_w={nw:.3f}{status}")

    # Freeze inactive ants.
    if active_set is not None:
        for i in range(n_ants):
            if i not in active_set:
                for p in model.prismion_swarm[i].parameters():
                    p.requires_grad = False
                for p in model.prismion_swarm_heads[i].parameters():
                    p.requires_grad = False
        n_frozen = sum(1 for i in range(n_ants) if i not in active_set)
        print(f"\n  Frozen {n_frozen}/{n_ants} ants (only {sorted(active_set)} get gradient)")

    # Patch model.
    per_step_ant_data = []
    prev_msgs = [None] * n_ants
    norm_weights = _patch_harmonic_weighted(model, n_ants, per_step_ant_data, prev_msgs,
                                            active_set=active_set)

    # Speed check: first 5 steps.
    model.train()
    print(f"\n  Speed check (5 steps)...")
    t_speed = time.time()
    opt_tmp = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    for s in range(1, 6):
        x, y = _generate_harmonic_batch(step_seed=s * 999)
        opt_tmp.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out[0], y)
        loss.backward()
        opt_tmp.step()
    dt_speed = time.time() - t_speed
    s_per_step = dt_speed / 5
    eta_min = s_per_step * steps / 60
    print(f"  => {s_per_step:.2f} s/step, ETA for {steps} steps: {eta_min:.1f} min")
    if DEVICE.type == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  VRAM after warmup: {vram_used:.2f} GB allocated, {vram_reserved:.2f} GB reserved")

    # Re-create model for clean training.
    torch.manual_seed(42)
    _random.seed(42)
    model = _create_model()
    n_ants, configs = _verify_swarm(model)

    # Re-load checkpoint after re-creation.
    if resume_path:
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    # Re-freeze inactive ants.
    if active_set is not None:
        for i in range(n_ants):
            if i not in active_set:
                for p in model.prismion_swarm[i].parameters():
                    p.requires_grad = False
                for p in model.prismion_swarm_heads[i].parameters():
                    p.requires_grad = False

    per_step_ant_data = []
    prev_msgs = [None] * n_ants
    norm_weights = _patch_harmonic_weighted(model, n_ants, per_step_ant_data, prev_msgs,
                                            active_set=active_set)
    model.train()

    # Checkpoint dir.
    ckpt_dir = os.path.join(os.path.dirname(log_path), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Train.
    log_interval = 50
    print(f"\n  Training {steps} steps (logging every {log_interval})...")
    print(f"  Checkpoints: {ckpt_dir} (every {save_every} steps)")
    losses, accs, elapsed = _train(model, steps, log_interval=log_interval, label="harmonic",
                                   ckpt_dir=ckpt_dir, save_every=save_every,
                                   active_set=active_set)

    print(f"\n  Training complete: {elapsed:.1f}s ({elapsed/steps:.2f} s/step)")
    print(f"  Final avg loss (last 50): {sum(losses[-50:])/50:.4f}")
    print(f"  Final avg acc  (last 50): {sum(accs[-50:])/50:.3f}")

    # Analysis.
    _analyze_instrumentation(per_step_ant_data, n_ants, configs,
                             HARMONIC_SEQ_LEN, steps)

    # Strong eval.
    combined_acc, main_acc = _strong_eval(model, n_ants, configs, norm_weights)

    # Final summary.
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Steps:          {steps}")
    print(f"  Time:           {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  s/step:         {elapsed/steps:.2f}")
    print(f"  Final loss:     {sum(losses[-50:])/50:.4f}")
    print(f"  Final acc:      {sum(accs[-50:])/50:.3f}")
    print(f"  Eval combined:  {combined_acc:.1f}%")
    print(f"  Eval main-only: {main_acc:.1f}%")
    print(f"{'='*70}")


# ── Comparison mode ─────────────────────────────────────────────────────

def run_compare(steps=50, log_path=_PROBE_LOG_DEFAULT):
    print(f"\n{'='*70}")
    print(f"Probe 10 — Comparison: Harmonic vs Stride vs Uniform ({steps} steps)")
    print(f"{'='*70}")
    _open_log(log_path)

    results = {}

    for mode_name, patch_fn_name in [
        ("A: Harmonic-weighted", "harmonic"),
        ("B: Fibonacci-stride",  "stride"),
        ("C: Uniform",           "uniform"),
    ]:
        torch.manual_seed(42)
        _random.seed(42)

        model = _create_model()
        n_ants, configs = _verify_swarm(model)

        if patch_fn_name == "harmonic":
            per_step_ant_data = []
            prev_msgs = [None] * n_ants
            _patch_harmonic_weighted(model, n_ants, per_step_ant_data, prev_msgs)
        elif patch_fn_name == "stride":
            _patch_fibonacci_stride(model, n_ants)
        elif patch_fn_name == "uniform":
            _patch_uniform(model, n_ants)

        model.train()
        losses, accs, elapsed = _train(model, steps, log_interval=10, label=mode_name)

        final_loss = sum(losses[-10:]) / min(10, len(losses))
        final_acc = sum(accs[-10:]) / min(10, len(accs))
        s_per_step = elapsed / steps

        results[mode_name] = {
            "loss": final_loss,
            "acc": final_acc,
            "s_per_step": s_per_step,
            "elapsed": elapsed,
        }

    # Print comparison table.
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"  {'Mode':<30s}  {'Loss':>8s}  {'Acc':>8s}  {'s/step':>8s}  {'Time':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, r in results.items():
        print(f"  {name:<30s}  {r['loss']:>8.4f}  {r['acc']:>7.1f}%  "
              f"{r['s_per_step']:>7.2f}s  {r['elapsed']:>7.1f}s")
    print(f"{'='*70}")


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe 10: Harmonic-Weighted Fibonacci Swarm")
    parser.add_argument("--compare", action="store_true",
                        help="Run 50-step comparison (harmonic vs stride vs uniform)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick 250-step run (for testing)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override step count")
    parser.add_argument("--log", type=str, default=_PROBE_LOG_DEFAULT,
                        help="Dashboard log path (default: logs/probe/probe_live.log)")
    parser.add_argument("--active-ants", type=str, default=None,
                        help="Comma-separated ant indices to train (e.g. '0' or '0,1,2'). "
                             "Inactive ants are frozen with zero vote weight.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save checkpoint every N steps (default: 25)")
    args = parser.parse_args()

    try:
        if args.compare:
            steps = args.steps or 50
            run_compare(steps=steps, log_path=args.log)
        else:
            if args.steps:
                steps = args.steps
            elif args.quick:
                steps = 250
            else:
                steps = 2500
            # Parse active ants set.
            active_set = None
            if args.active_ants is not None:
                active_set = set(int(x.strip()) for x in args.active_ants.split(","))
            run_full(steps, log_path=args.log, active_set=active_set,
                     resume_path=args.resume, save_every=args.save_every)
    finally:
        _close_log()
        # Restore env.
        os.environ["VRX_PRISMION"] = "0"
        os.environ["VRX_PRISMION_FIBONACCI"] = "0"
