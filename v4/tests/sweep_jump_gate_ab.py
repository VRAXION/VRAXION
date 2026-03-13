"""A/B sweep: sequential pointer with vs without learned jump gate.

Tests whether the learned φ-jump gate improves memory tasks.
The gate learns when to walk (+1) vs jump to golden-ratio destination.

Usage:
    python sweep_jump_gate_ab.py                  # 1000 steps, CPU
    python sweep_jump_gate_ab.py --steps 5000     # longer run
    python sweep_jump_gate_ab.py --device cuda    # GPU
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for subdir in ("model", "training"):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from instnct import INSTNCT  # type: ignore[import-not-found]
from train import ByteDataset, func_discover_dat, func_maskloss_ce  # type: ignore[import-not-found]

torch.set_num_threads(16)


def _set_determinism(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _discover_dataset(seq, seed):
    data_dir = ROOT / "training_data"
    if not data_dir.exists():
        fallback = Path(r"S:\AI\work\VRAXION_DEV\v4\training_data")
        if fallback.exists():
            data_dir = fallback
    files = func_discover_dat(str(data_dir))
    return ByteDataset(files, seq, embed_mode=True, seed=seed)


CONFIGS = {
    "baseline": {
        "jump_gate": False,
        "c19_mode": "dualphi",
        "pointer_interp_mode": "linear",
        "pointer_seam_mode": "shortest_arc",
    },
    "jump_gate": {
        "jump_gate": True,
        "c19_mode": "dualphi",
        "pointer_interp_mode": "linear",
        "pointer_seam_mode": "shortest_arc",
    },
}


def run_one(label, config, steps, batch, seq, hidden_dim, M, slot_dim, R, device, seed):
    _set_determinism(seed)
    dataset = _discover_dataset(seq, seed)
    dataset.init_sequential(batch)

    model = INSTNCT(
        M=M, hidden_dim=hidden_dim, slot_dim=slot_dim, N=1, R=R,
        embed_mode=True, embed_encoding='learned', output_encoding='lowrank_c19',
        kernel_mode='vshape', pointer_mode='sequential',
        write_mode='replace', expert_weighting=False,
        checkpoint_chunks=0, bb_enabled=False,
        jump_gate=config["jump_gate"],
        c19_mode=config["c19_mode"],
        pointer_interp_mode=config["pointer_interp_mode"],
        pointer_seam_mode=config["pointer_seam_mode"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses, accs = [], []
    state = None
    max_grad = 0.0
    t0 = time.time()

    for step in range(1, steps + 1):
        model._diag_enabled = (step % 100 == 0 or step == 1)
        xb, yb, mask = dataset.sample_batch_sequential(batch, device)
        logits, new_state = model(xb, S='dotprod', state=state)
        if new_state is not None:
            state = {k: v.detach() for k, v in new_state.items()}
        _, masked_loss = func_maskloss_ce(logits, yb, mask)
        opt.zero_grad()
        masked_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
        opt.step()

        losses.append(masked_loss.item())
        max_grad = max(max_grad, gn)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            accs.append((correct.sum() / mask.sum().clamp(min=1)).item())

        if step % 100 == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            elapsed = time.time() - t0
            gate_info = ""
            if config["jump_gate"] and model._diag_enabled:
                gate_mean = model._diag.get('jump_gate_mean_0', 0)
                gate_max = model._diag.get('jump_gate_max_0', 0)
                gate_info = f"  gate={gate_mean:.4f}/{gate_max:.4f}"
            print(
                f"  [{label}] step {step:5d}/{steps}  "
                f"loss={avg_loss:.4f}  bpc={avg_loss*1.4427:.3f}  "
                f"acc={avg_acc:.3f}  gnorm={gn:.1f}  "
                f"{elapsed:.0f}s{gate_info}"
            )

    elapsed = time.time() - t0
    return {
        "label": label,
        "jump_gate": config["jump_gate"],
        "n_params": n_params,
        "final_loss": sum(losses[-100:]) / min(100, len(losses)),
        "final_bpc": sum(losses[-100:]) / min(100, len(losses)) * 1.4427,
        "final_acc": sum(accs[-100:]) / min(100, len(accs)),
        "best_acc": max(accs),
        "time_s": elapsed,
        "s_per_step": elapsed / steps,
        "max_grad": max_grad,
    }


def main():
    parser = argparse.ArgumentParser(description="A/B: jump gate vs baseline")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--slot-dim", type=int, default=8)
    parser.add_argument("--R", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "sweep_results" / f"jump_gate_ab_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, config in CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"  Running: {label}")
        print(f"{'='*80}")
        result = run_one(
            label, config, args.steps, args.batch, args.seq,
            args.hidden_dim, args.M, args.slot_dim, args.R,
            args.device, args.seed,
        )
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("JUMP GATE A/B COMPARISON")
    print(f"{'='*80}")
    print(f"{'Label':<15} {'Params':>8} {'Loss':>8} {'BPC':>8} {'Acc':>8} {'Best':>8} {'Time':>8}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['label']:<15} {r['n_params']:>8} {r['final_loss']:>8.4f} "
            f"{r['final_bpc']:>8.4f} {r['final_acc']:>8.4f} "
            f"{r['best_acc']:>8.4f} {r['time_s']:>7.1f}s"
        )

    if len(results) == 2:
        b, jg = results[0], results[1]
        print(f"\nDelta (jump_gate - baseline):")
        print(f"  loss: {jg['final_loss'] - b['final_loss']:+.4f}")
        print(f"  acc:  {(jg['final_acc'] - b['final_acc'])*100:+.2f}pp")
        print(f"  params: +{jg['n_params'] - b['n_params']}")
        print(f"  speed: {(b['time_s'] - jg['time_s'])/b['time_s']*100:+.1f}%")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"timestamp": stamp, "results": results}, f, indent=2)
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
