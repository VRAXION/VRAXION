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
    "VRX_PRISMION_FIB_BUDGET_MB": "1.2",
    "VRX_PRISMION_FIB_MIN_RING": "4",
    "VRX_PRISMION_FIB_MAX_ANTS": "16",
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
from _checkpoint_io import atomic_torch_save, safe_torch_load  # noqa: E402


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
def compute_per_ant_gnorms(model: AbsoluteHallway, active_ants: int | None = None) -> list[float]:
    """Compute gradient L2 norm for each active ant's parameters."""
    if model.prismion_swarm is None:
        return []
    n = len(model.prismion_swarm)
    if active_ants is not None and active_ants >= 0:
        n = min(active_ants, n)
    gnorms = []
    for i in range(n):
        ant = model.prismion_swarm[i]
        total = 0.0
        for p in ant.parameters():
            if p.grad is not None:
                total += p.grad.float().norm().item() ** 2
        gnorms.append(math.sqrt(total))
    # Also include head gradients.
    if model.prismion_swarm_heads is not None:
        for i in range(min(n, len(model.prismion_swarm_heads))):
            head = model.prismion_swarm_heads[i]
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
    active_ants: int | None = None,
) -> list[torch.Tensor]:
    """Run each active ant independently and return per-ant logits."""
    if model.prismion_swarm is None:
        return []
    n = len(model.prismion_swarm)
    if active_ants is not None and active_ants >= 0:
        n = min(active_ants, n)
    votes = []
    with torch.no_grad():
        for i in range(n):
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
    parser.add_argument("--active-ants", type=int, default=1,
                        help="Number of active ants (0=none, -1=all, default=1)")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N steps (default=10)")
    parser.add_argument("--checkpoint-dir", type=str, default="logs/probe/checkpoints",
                        help="Directory for checkpoint files")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-sync", action="store_true",
                        help="Disable auto git sync to nightly branch")
    parser.add_argument("--freeze-ants", type=str, default=None,
                        help="Comma-separated ant indices to freeze (still vote, no gradients). "
                             "E.g. --freeze-ants 0 freezes ant[0], --freeze-ants 0,1 freezes both.")
    args = parser.parse_args()

    # Set active ants env var BEFORE model construction.
    os.environ["VRX_PRISMION_FIB_ACTIVE_ANTS"] = str(args.active_ants)

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
    n_ants_total = len(model.prismion_swarm) if model.prismion_swarm else 0
    active = int(model.prismion_fib_active_ants)
    if active < 0:
        active = n_ants_total
    else:
        active = min(active, n_ants_total)
    n_ants = active  # only track active ants
    print(f"[probe11] Fibonacci swarm: {active}/{n_ants_total} ants active")
    if model.prismion_swarm:
        ring_lens = [int(ant.ring_len) for ant in model.prismion_swarm]
        total_rl = sum(ring_lens[:active]) if active > 0 else 1
        for i, ant in enumerate(model.prismion_swarm):
            rl = ring_lens[i]
            w = rl / total_rl if i < active else 0.0
            params = sum(p.numel() for p in ant.parameters())
            frozen = "FROZEN" if i >= active else "ACTIVE"
            print(f"  ant[{i}]: ring_len={rl}, weight={w:.4f}, params={params:,} [{frozen}]")

    # Freeze specific ants (still participate in forward pass, no gradients).
    frozen_ant_indices = set()
    if args.freeze_ants:
        frozen_ant_indices = {int(x.strip()) for x in args.freeze_ants.split(",")}
        if model.prismion_swarm:
            for idx in frozen_ant_indices:
                if idx < len(model.prismion_swarm):
                    for p in model.prismion_swarm[idx].parameters():
                        p.requires_grad = False
                    # Also freeze the corresponding head.
                    if model.prismion_swarm_heads and idx < len(model.prismion_swarm_heads):
                        for p in model.prismion_swarm_heads[idx].parameters():
                            p.requires_grad = False
                    print(f"[probe11] ant[{idx}] FROZEN (votes but no gradients)")
                else:
                    print(f"[probe11] WARNING: ant[{idx}] does not exist, skipping freeze")
        frozen_params = sum(
            sum(p.numel() for p in model.prismion_swarm[i].parameters())
            for i in frozen_ant_indices if i < len(model.prismion_swarm)
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[probe11] Frozen params: {frozen_params:,} | Trainable params: {trainable_params:,}")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    # Checkpoint directory.
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    start_step = 1

    # Resume from checkpoint.
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = safe_torch_load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            # Skip optimizer restore if freeze config changed (param count mismatch).
            if frozen_ant_indices:
                print(f"[probe11] Freeze config active — using fresh optimizer for unfrozen params")
            else:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except (ValueError, KeyError, RuntimeError) as exc:
                    print(f"[probe11] WARNING: optimizer state mismatch: {exc}")
                    print(f"[probe11] Continuing with fresh optimizer")
            start_step = ckpt.get("step", 0) + 1
            best_acc = ckpt.get("acc_ma100", 0.0)
            print(f"[probe11] Resumed from {ckpt_path} at step {start_step}, best_acc={best_acc:.4f}")
        else:
            print(f"[probe11] WARNING: checkpoint not found at {ckpt_path}, starting fresh")

    # Auto-sync setup.
    _git_sync = None
    if not args.no_sync:
        try:
            from _git_sync import auto_sync_nightly
            _git_sync = auto_sync_nightly
        except ImportError:
            print("[probe11] WARNING: _git_sync not available, auto-sync disabled")

    # Rolling accuracy trackers.
    acc_window = deque(maxlen=100)
    acc_window_50 = deque(maxlen=50)
    acc_window_10 = deque(maxlen=10)
    slow_acc_windows = [deque(maxlen=100) for _ in range(active)]
    fast_acc_windows = [deque(maxlen=100) for _ in range(active)]

    telemetry_path = Path(args.telemetry)
    telemetry_mode = "a" if args.resume else "w"
    telemetry_fh = open(telemetry_path, telemetry_mode, encoding="utf-8")

    # Auto-launch dashboard.
    if not args.no_dashboard:
        dash_port = _launch_dashboard(str(telemetry_path))
        if dash_port:
            print(f"[probe11] Live dashboard: http://localhost:{dash_port}")
    else:
        print("[probe11] Dashboard disabled (--no-dashboard)")

    print(f"[probe11] Starting {args.steps} steps (from {start_step}) | batch={args.batch_size} | "
          f"seq_len={args.seq_len} | lr={args.lr} | device={device}")
    print(f"[probe11] Telemetry -> {telemetry_path.resolve()}")
    print(f"[probe11] Checkpoints -> {ckpt_dir.resolve()} (every {args.checkpoint_every} steps)")
    print("-" * 100)

    t0 = time.time()
    model.train()

    for step in range(start_step, args.steps + 1):
        x, labels, slow_labels, fast_labels = make_harmonic_xor_batch(
            args.batch_size, args.seq_len, device=device,
        )

        optimizer.zero_grad()
        logits, move_penalty = model(x)
        loss = criterion(logits, labels) + 0.01 * move_penalty
        loss.backward()

        # Per-ant gradient norms (after backward, before optimizer step).
        gnorms = compute_per_ant_gnorms(model, active_ants=active)

        optimizer.step()

        # Overall accuracy.
        pred = logits.argmax(dim=1)
        correct = (pred == labels).float().mean().item()
        acc_window.append(correct)
        acc_window_50.append(correct)
        acc_window_10.append(correct)

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
            "acc_ma50": round(sum(acc_window_50) / len(acc_window_50), 4),
            "acc_ma10": round(sum(acc_window_10) / len(acc_window_10), 4),
            "active_ants": active,
            "total_ants": n_ants_total,
            "gnorms": [round(g, 4) for g in gnorms],
        }

        # Add swarm telemetry from model (sliced to active only).
        if hasattr(model, "fib_swarm_weights") and model.fib_swarm_weights is not None:
            record["weights"] = [round(w, 4) for w in model.fib_swarm_weights[:active]]
        if hasattr(model, "fib_swarm_ring_lens") and model.fib_swarm_ring_lens is not None:
            record["ring_lens"] = model.fib_swarm_ring_lens[:active]
        if hasattr(model, "fib_swarm_logit_norm") and model.fib_swarm_logit_norm is not None:
            record["swarm_logit_norm"] = round(model.fib_swarm_logit_norm, 4)
        if hasattr(model, "fib_swarm_msg_norms") and model.fib_swarm_msg_norms is not None:
            record["msg_norms"] = [round(n, 4) for n in model.fib_swarm_msg_norms[:active]]

        # Gnorm ratio (biggest / smallest) — only meaningful with 2+ active ants.
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

        # Checkpoint saving.
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            ckpt_payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "loss": record["loss"],
                "acc_ma100": record["acc_ma100"],
                "acc_ma50": record["acc_ma50"],
                "acc_ma10": record["acc_ma10"],
            }
            step_path = ckpt_dir / f"probe11_step_{step:05d}.pt"
            atomic_torch_save(ckpt_payload, step_path)

            # Best model tracking.
            if record["acc_ma100"] > best_acc:
                best_acc = record["acc_ma100"]
                atomic_torch_save(ckpt_payload, ckpt_dir / "best.pt")
                print(f"[probe11] New best MA100={best_acc:.4f} saved at step {step}")

            # Latest checkpoint (overwritten each time).
            atomic_torch_save(ckpt_payload, ckpt_dir / "latest.pt")

            # Auto-sync to nightly.
            if _git_sync is not None:
                try:
                    _git_sync(
                        message=f"[auto] probe11 step {step}: MA100={record['acc_ma100']:.4f}",
                        paths=[str(telemetry_path)],
                    )
                except Exception as exc:
                    print(f"[probe11] WARNING: auto-sync failed: {exc}")

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
    print(f"[probe11] Checkpoints in {ckpt_dir.resolve()} (best MA100={best_acc:.4f})")


if __name__ == "__main__":
    main()
