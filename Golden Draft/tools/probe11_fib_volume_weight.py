"""Probe 11 — Fibonacci Swarm with Volume Weighting.

Harmonic XOR task: x[t] = sin(t, F=1) + sin(t, F=7), label = XOR(sign_slow, sign_fast).
Tests whether volume-weighted ants (bigger ring → louder voice) produce
frequency separation: big ants learn the slow signal, small ants learn the fast.

Previous probe (probe10) used harmonic weighting (weight = i+1) which caused
17x gradient imbalance — smallest ant dominated, biggest starved to 0.2 accuracy.
Volume weighting is now baked into _prismion_apply_fibonacci().

Usage:
    python "S:/AI/Golden Draft/tools/probe11_fib_volume_weight.py"
    python "S:/AI/Golden Draft/tools/probe11_fib_volume_weight.py" --steps 5000 --device cuda
    python "S:/AI/Golden Draft/tools/probe11_fib_volume_weight.py" --no-dashboard
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import webbrowser
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Golden Code"))

# Set env BEFORE importing the model (settings are env-driven, loaded once).
os.environ.update({
    "VRX_PRISMION": "1",
    "VRX_PRISMION_FIBONACCI": "1",
    "VRX_PRISMION_FIB_BUDGET_MB": "2",
    "VRX_PRISMION_FIB_MIN_RING": "4",
    "VRX_PRISMION_FIB_MAX_ANTS": "8",
    "VRX_THINK_RING": "1",
    "VRX_THINK_RING_MODE": "replace",
    "VRX_THINK_RING_DUAL": "1",
    "VRX_THINK_RING_BRAINSTEM": "0",
    "VRX_THINK_RING_LEN": "8",
    "VRX_AUXDIM": "16",
    "VRX_SENSORY_RING": "0",
    "VRX_VAULT": "0",
    "VRX_NAN_GUARD": "",
    "VRX_MAIN_LOGIT_WEIGHT": "1.0",
    "VRX_PRISMION_TOPOLOGY": "bank",
    "VRX_PRISMION_N": "2",
    "VRX_PRISMION_LEN": "4",
    "VRX_PRISMION_ALPHA": "1.0",
})

from vraxion.platinum.hallway import AbsoluteHallway  # noqa: E402


# ---------------------------------------------------------------------------
# Dashboard auto-launcher
# ---------------------------------------------------------------------------
_DASHBOARD_PORT_START = 8511


def _is_port_open(port: int) -> bool:
    """Check if a TCP port is already listening on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _find_free_port(start: int = _DASHBOARD_PORT_START, scan: int = 20) -> int:
    """Return *start* if free, otherwise scan upward. Raises if all taken."""
    for offset in range(scan):
        if not _is_port_open(start + offset):
            return start + offset
    raise RuntimeError(f"No free port in {start}..{start + scan - 1}")


def _launch_dashboard(telemetry_path: str, port: int | None = None) -> int | None:
    """Launch probe11_dashboard.py in a detached Streamlit process.

    Returns the port number on success, or None on failure (non-fatal).
    """
    try:
        import importlib
        importlib.import_module("streamlit")
    except ImportError:
        print("[probe11] WARNING: streamlit not installed -- skipping dashboard")
        return None

    dashboard_script = Path(__file__).resolve().parent / "probe11_dashboard.py"
    if not dashboard_script.exists():
        print(f"[probe11] WARNING: dashboard not found at {dashboard_script}")
        return None

    if port is None:
        try:
            port = _find_free_port()
        except RuntimeError as exc:
            print(f"[probe11] WARNING: {exc} -- skipping dashboard")
            return None

    if _is_port_open(port):
        url = f"http://localhost:{port}"
        print(f"[probe11] Dashboard already running on {url}")
        webbrowser.open(url)
        return port

    # Log file for the dashboard process.
    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / "bench_vault" / "_tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"probe11_dash_{port}_{ts}.log"

    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(dashboard_script),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    env = os.environ.copy()
    env["PROBE11_TELEMETRY"] = str(Path(telemetry_path).resolve())

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            int(getattr(subprocess, "DETACHED_PROCESS", 0))
            | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        )

    try:
        with log_path.open("w", encoding="utf-8") as lf:
            subprocess.Popen(
                cmd,
                env=env,
                cwd=str(repo_root),
                stdout=lf,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
    except Exception as exc:
        print(f"[probe11] WARNING: failed to launch dashboard: {exc}")
        return None

    url = f"http://localhost:{port}"
    print(f"[probe11] Dashboard launching on {url}")
    print(f"[probe11] Dashboard log: {log_path}")

    time.sleep(1.5)
    webbrowser.open(url)
    return port


# ---------------------------------------------------------------------------
# Harmonic XOR data generator
# ---------------------------------------------------------------------------
def make_harmonic_xor_batch(
    batch_size: int,
    seq_len: int,
    f_slow: float = 1.0,
    f_fast: float = 7.0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of harmonic XOR sequences.

    x[t] = sin(2pi * f_slow * t/seq_len) + sin(2pi * f_fast * t/seq_len)
    label = XOR(sign(slow_component), sign(fast_component))

    Returns: (x, labels, slow_labels, fast_labels)
      x: [B, seq_len, input_dim]  (input_dim=4: sin_slow, cos_slow, sin_fast, cos_fast)
      labels: [B] binary XOR label (from last timestep)
      slow_labels: [B] binary slow component sign
      fast_labels: [B] binary fast component sign
    """
    # Random phase offsets per sample for variety.
    phase_slow = torch.rand(batch_size, 1, device=device) * 2 * math.pi
    phase_fast = torch.rand(batch_size, 1, device=device) * 2 * math.pi

    t = torch.linspace(0, 1, seq_len, device=device).unsqueeze(0)  # [1, T]

    slow_sin = torch.sin(2 * math.pi * f_slow * t + phase_slow)  # [B, T]
    slow_cos = torch.cos(2 * math.pi * f_slow * t + phase_slow)
    fast_sin = torch.sin(2 * math.pi * f_fast * t + phase_fast)
    fast_cos = torch.cos(2 * math.pi * f_fast * t + phase_fast)

    x = torch.stack([slow_sin, slow_cos, fast_sin, fast_cos], dim=2)  # [B, T, 4]

    # Labels from last timestep.
    slow_sign = (slow_sin[:, -1] > 0).long()  # [B]
    fast_sign = (fast_sin[:, -1] > 0).long()
    labels = (slow_sign ^ fast_sign).long()     # XOR

    return x, labels, slow_sign, fast_sign


# ---------------------------------------------------------------------------
# Per-ant gradient norm tracker
# ---------------------------------------------------------------------------
def compute_per_ant_gnorms(model: AbsoluteHallway) -> list[float]:
    """Compute gradient L2 norm for each ant's parameters."""
    if model.prismion_swarm is None:
        return []
    gnorms = []
    for ant in model.prismion_swarm:
        total = 0.0
        for p in ant.parameters():
            if p.grad is not None:
                total += p.grad.float().norm().item() ** 2
        gnorms.append(math.sqrt(total))
    # Also include head gradients.
    if model.prismion_swarm_heads is not None:
        for i, head in enumerate(model.prismion_swarm_heads):
            head_gnorm_sq = 0.0
            for p in head.parameters():
                if p.grad is not None:
                    head_gnorm_sq += p.grad.float().norm().item() ** 2
            if i < len(gnorms):
                gnorms[i] = math.sqrt(gnorms[i] ** 2 + head_gnorm_sq)
    return gnorms


# ---------------------------------------------------------------------------
# Per-ant vote and accuracy
# ---------------------------------------------------------------------------
def compute_per_ant_votes(
    model: AbsoluteHallway,
    chrom: torch.Tensor,
    fib_prism_states: list,
    active_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Run each ant independently and return per-ant logits."""
    if model.prismion_swarm is None:
        return []
    votes = []
    with torch.no_grad():
        for i in range(len(model.prismion_swarm)):
            ant = model.prismion_swarm[i]
            head = model.prismion_swarm_heads[i]
            msg_i, _ = ant.step(chrom, fib_prism_states[i], active_mask)
            logits_i = head(msg_i.to(head.weight.dtype))
            votes.append(logits_i)
    return votes


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Probe 11: Fibonacci Volume Weight")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--telemetry", type=str, default="probe11_telemetry.jsonl",
                        help="JSONL telemetry output path")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable automatic Streamlit dashboard launch")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[probe11] CUDA not available, falling back to CPU")
        device = "cpu"

    # Build model.
    input_dim = 4  # sin_slow, cos_slow, sin_fast, cos_fast
    num_classes = 2  # binary XOR
    model = AbsoluteHallway(
        input_dim=input_dim,
        num_classes=num_classes,
        ring_len=64,
        slot_dim=32,
        ptr_stride=1,
        gauss_k=1,
        gauss_tau=2.0,
    ).to(device)

    # Report swarm config.
    n_ants = len(model.prismion_swarm) if model.prismion_swarm else 0
    print(f"[probe11] Fibonacci swarm active: {model.prismion_fib_active}, ants: {n_ants}")
    if model.prismion_swarm:
        ring_lens = [int(ant.ring_len) for ant in model.prismion_swarm]
        total_rl = sum(ring_lens)
        weights = [rl / total_rl for rl in ring_lens]
        for i, (ant, rl, w) in enumerate(zip(model.prismion_swarm, ring_lens, weights)):
            params = sum(p.numel() for p in ant.parameters())
            print(f"  ant[{i}]: ring_len={rl}, weight={w:.4f}, params={params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Rolling accuracy tracker (window=100).
    acc_window = deque(maxlen=100)
    slow_acc_windows = [deque(maxlen=100) for _ in range(n_ants)]
    fast_acc_windows = [deque(maxlen=100) for _ in range(n_ants)]

    telemetry_path = Path(args.telemetry)
    telemetry_fh = open(telemetry_path, "w", encoding="utf-8")

    # Auto-launch dashboard.
    if not args.no_dashboard:
        dash_port = _launch_dashboard(str(telemetry_path))
        if dash_port:
            print(f"[probe11] Live dashboard: http://localhost:{dash_port}")
    else:
        print("[probe11] Dashboard disabled (--no-dashboard)")

    print(f"[probe11] Starting {args.steps} steps | batch={args.batch_size} | "
          f"seq_len={args.seq_len} | lr={args.lr} | device={device}")
    print(f"[probe11] Telemetry -> {telemetry_path.resolve()}")
    print("-" * 100)

    t0 = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        x, labels, slow_labels, fast_labels = make_harmonic_xor_batch(
            args.batch_size, args.seq_len, device=device,
        )

        optimizer.zero_grad()
        logits, move_penalty = model(x)
        loss = criterion(logits, labels) + 0.01 * move_penalty
        loss.backward()

        # Per-ant gradient norms (after backward, before optimizer step).
        gnorms = compute_per_ant_gnorms(model)

        optimizer.step()

        # Overall accuracy.
        pred = logits.argmax(dim=1)
        correct = (pred == labels).float().mean().item()
        acc_window.append(correct)

        # Per-ant slow/fast accuracy.
        per_ant_slow_acc = []
        per_ant_fast_acc = []
        if n_ants > 0 and hasattr(model, '_last_fib_prism_states'):
            # We can't easily get per-ant votes post-step without re-running.
            # Use the swarm's telemetry instead. For now, track overall only.
            pass

        # Build telemetry record.
        record = {
            "step": step,
            "loss": round(loss.item(), 6),
            "acc": round(correct, 4),
            "acc_ma100": round(sum(acc_window) / len(acc_window), 4),
            "gnorms": [round(g, 4) for g in gnorms],
        }

        # Add swarm telemetry from model.
        if hasattr(model, "fib_swarm_weights") and model.fib_swarm_weights is not None:
            record["weights"] = [round(w, 4) for w in model.fib_swarm_weights]
        if hasattr(model, "fib_swarm_ring_lens") and model.fib_swarm_ring_lens is not None:
            record["ring_lens"] = model.fib_swarm_ring_lens
        if hasattr(model, "fib_swarm_logit_norm") and model.fib_swarm_logit_norm is not None:
            record["swarm_logit_norm"] = round(model.fib_swarm_logit_norm, 4)
        if hasattr(model, "fib_swarm_msg_norms") and model.fib_swarm_msg_norms is not None:
            record["msg_norms"] = [round(n, 4) for n in model.fib_swarm_msg_norms]

        # Gnorm ratio (biggest / smallest).
        if len(gnorms) >= 2 and gnorms[-1] > 1e-12:
            record["gnorm_ratio"] = round(gnorms[0] / max(gnorms[-1], 1e-12), 2)

        telemetry_fh.write(json.dumps(record) + "\n")

        # Dashboard-compatible log line.
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            steps_per_sec = step / max(elapsed, 1e-6)
            gnorm_str = " | ".join(f"ant{i}_gn={g:.2f}" for i, g in enumerate(gnorms))
            ratio_str = f"ratio={record.get('gnorm_ratio', 0):.1f}x" if "gnorm_ratio" in record else ""
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"acc {correct:.2f} (ma={record['acc_ma100']:.3f}) | "
                f"{gnorm_str} | {ratio_str} | "
                f"{steps_per_sec:.1f} step/s"
            )

        telemetry_fh.flush()

    telemetry_fh.close()
    elapsed = time.time() - t0
    print("-" * 100)
    print(f"[probe11] Done. {args.steps} steps in {elapsed:.1f}s ({args.steps / elapsed:.1f} step/s)")
    print(f"[probe11] Final loss: {loss.item():.4f} | Final acc (MA100): {record['acc_ma100']:.3f}")
    if gnorms:
        print(f"[probe11] Final gnorms: {['%.3f' % g for g in gnorms]}")
        if len(gnorms) >= 2:
            print(f"[probe11] Final gnorm ratio (ant[0]/ant[-1]): {gnorms[0] / max(gnorms[-1], 1e-12):.2f}x")
    print(f"[probe11] Telemetry saved to {telemetry_path.resolve()}")


if __name__ == "__main__":
    main()
