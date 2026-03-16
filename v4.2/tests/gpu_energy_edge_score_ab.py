"""GPU microprobe: energy-guided existing-edge scoring A/B.

This isolates one question:
  - for existing-edge mutations, is edge score better as product(src,dst)
    or max(src,dst)?

We keep everything else fixed:
  - same V64_N192 config
  - same two-bit controller skeleton
  - same budget per run
  - same energy-guided add rule: src weighted by energy, dst random
  - energy = accumulated abs activation across all ticks
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.log import live_log, log_msg
from tests.gpu_int_mood_ab import (
    CHARGE_RATE,
    CLIP_BOUND,
    CONFIGS,
    GAIN,
    SELF_CONN,
    THRESHOLD,
    TICKS,
    gpu_init,
)


SCORING_MODES = ("product", "max")
DEFAULT_CONFIG = "V64_N192"
DEFAULT_SEEDS = "42,77,123"
DEFAULT_ATTEMPTS = 16000
DEFAULT_ENERGY_PCT = 0.60
CHECKPOINT_EVERY = 2000
LOSS_DRIFT = 0.2
PATIENCE = 0.35


@dataclass
class EnergyEvalBuffers:
    eye: torch.Tensor
    charges: torch.Tensor
    acts: torch.Tensor
    weff: torch.Tensor
    row_idx: torch.Tensor
    energy: torch.Tensor


@dataclass
class GuidedChanges:
    cells: list[tuple[int, int, int]]
    effective: int = 0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    ap.add_argument("--energy-pct", type=float, default=DEFAULT_ENERGY_PCT)
    ap.add_argument("--modes", default="product,max")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--log-name", default="gpu_energy_edge_score_ab")
    return ap.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def retention_from_loss(loss_pct_t: torch.Tensor) -> torch.Tensor:
    return 1.0 - loss_pct_t.to(torch.float32) * 0.01


def mask_density(mask: torch.Tensor) -> float:
    n = mask.shape[0]
    total = n * n - n
    return float((mask != 0).sum().item()) / float(total)


def make_energy_eval_buffers(vocab: int, neurons: int, device: torch.device) -> EnergyEvalBuffers:
    return EnergyEvalBuffers(
        eye=torch.eye(vocab, dtype=torch.float32, device=device),
        charges=torch.empty((vocab, neurons), dtype=torch.float32, device=device),
        acts=torch.empty((vocab, neurons), dtype=torch.float32, device=device),
        weff=torch.empty((neurons, neurons), dtype=torch.float32, device=device),
        row_idx=torch.arange(vocab, device=device, dtype=torch.long),
        energy=torch.empty(neurons, dtype=torch.float32, device=device),
    )


def gpu_eval_with_energy(
    mask: torch.Tensor,
    leak: torch.Tensor,
    targets: torch.Tensor,
    out_start: int,
    buffers: EnergyEvalBuffers,
):
    eye = buffers.eye
    charges = buffers.charges
    acts = buffers.acts
    weff = buffers.weff
    row_idx = buffers.row_idx
    energy = buffers.energy

    charges.zero_()
    acts.zero_()
    energy.zero_()
    weff.copy_(mask)
    weff.mul_(GAIN)

    for t in range(TICKS):
        if t == 0:
            acts[:, : eye.shape[0]] = eye
        raw = acts @ weff + acts * SELF_CONN
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw * CHARGE_RATE
        charges *= leak
        acts = torch.clamp(charges - THRESHOLD, min=0.0)
        charges = torch.clamp(charges, -CLIP_BOUND, CLIP_BOUND)
        energy += acts.abs().sum(dim=0)

    logits = charges[:, out_start : out_start + eye.shape[0]]
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = (preds == targets).to(torch.float32).mean()
    tp = probs[row_idx, targets].mean()
    score = 0.5 * acc + 0.5 * tp
    return score, acc, energy.clone()


def controller_state() -> dict[str, int]:
    return {"signal": 0, "grow": 1, "intensity": 7}


def maybe_flip_strategy_gpu(controller: dict[str, int], gen: torch.Generator, device: torch.device):
    if float(torch.rand((), generator=gen, device=device).item()) < PATIENCE:
        controller["signal"] = 1 - controller["signal"]
    if float(torch.rand((), generator=gen, device=device).item()) < PATIENCE:
        controller["grow"] = 1 - controller["grow"]


def pick_alive_index(
    mask: torch.Tensor,
    energy: torch.Tensor,
    scoring_mode: str,
    energy_pct: float,
    gen: torch.Generator,
):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return None, None
    use_energy = float(torch.rand((), generator=gen, device=mask.device).item()) < energy_pct
    if use_energy:
        src_e = energy.index_select(0, alive[:, 0]).clamp_min(0)
        dst_e = energy.index_select(0, alive[:, 1]).clamp_min(0)
        if scoring_mode == "product":
            weights = src_e * dst_e
        elif scoring_mode == "max":
            weights = torch.maximum(src_e, dst_e)
        else:
            raise ValueError(f"unknown scoring mode: {scoring_mode}")
        if bool((weights > 0).any().item()):
            idx = int(torch.multinomial(weights, 1, replacement=True, generator=gen).item())
            return alive, idx
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    return alive, idx


def pick_src_index(neurons: int, energy: torch.Tensor, energy_pct: float, gen: torch.Generator, device: torch.device) -> int:
    use_energy = float(torch.rand((), generator=gen, device=device).item()) < energy_pct
    if use_energy:
        weights = energy.clamp_min(0)
        if bool((weights > 0).any().item()):
            return int(torch.multinomial(weights, 1, replacement=True, generator=gen).item())
    return int(torch.randint(neurons, (1,), generator=gen, device=device).item())


def energy_guided_add(
    mask: torch.Tensor,
    energy: torch.Tensor,
    energy_pct: float,
    gen: torch.Generator,
    changes: GuidedChanges,
) -> bool:
    n = mask.shape[0]
    for _ in range(64):
        src = pick_src_index(n, energy, energy_pct, gen, mask.device)
        dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
        if dst == src or int(mask[src, dst].item()) != 0:
            continue
        new = 1 if float(torch.rand((), generator=gen, device=mask.device).item()) > 0.5 else -1
        changes.cells.append((src, dst, 0))
        changes.effective += 1
        mask[src, dst] = new
        return True
    return False


def energy_guided_flip(
    mask: torch.Tensor,
    energy: torch.Tensor,
    scoring_mode: str,
    energy_pct: float,
    gen: torch.Generator,
    changes: GuidedChanges,
) -> bool:
    alive, idx = pick_alive_index(mask, energy, scoring_mode, energy_pct, gen)
    if alive is None:
        return False
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    old = int(mask[row, col].item())
    changes.cells.append((row, col, old))
    changes.effective += 1
    mask[row, col] = -old
    return True


def energy_guided_remove(
    mask: torch.Tensor,
    energy: torch.Tensor,
    scoring_mode: str,
    energy_pct: float,
    gen: torch.Generator,
    changes: GuidedChanges,
) -> bool:
    alive, idx = pick_alive_index(mask, energy, scoring_mode, energy_pct, gen)
    if alive is None:
        return False
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    old = int(mask[row, col].item())
    changes.cells.append((row, col, old))
    changes.effective += 1
    mask[row, col] = 0
    return True


def energy_guided_rewire(
    mask: torch.Tensor,
    energy: torch.Tensor,
    scoring_mode: str,
    energy_pct: float,
    gen: torch.Generator,
    changes: GuidedChanges,
) -> bool:
    alive, idx = pick_alive_index(mask, energy, scoring_mode, energy_pct, gen)
    if alive is None:
        return False
    rc = alive[idx]
    src = int(rc[0].item())
    dst = int(rc[1].item())
    old = int(mask[src, dst].item())
    for _ in range(64):
        new_dst = int(torch.randint(mask.shape[0], (1,), generator=gen, device=mask.device).item())
        if new_dst == src or new_dst == dst or int(mask[src, new_dst].item()) != 0:
            continue
        changes.cells.append((src, dst, old))
        changes.cells.append((src, new_dst, 0))
        changes.effective += 1
        mask[src, dst] = 0
        mask[src, new_dst] = old
        return True
    return False


def rollback_current(mask: torch.Tensor, loss_pct: torch.Tensor, prev_loss: int, changes: GuidedChanges) -> None:
    for row, col, old in reversed(changes.cells):
        mask[row, col] = old
    loss_pct.fill_(prev_loss)


def mutate_energy_guided(
    mask: torch.Tensor,
    loss_pct: torch.Tensor,
    controller: dict[str, int],
    gen: torch.Generator,
    energy: torch.Tensor,
    scoring_mode: str,
    energy_pct: float,
):
    changes = GuidedChanges(cells=[])
    prev_loss = int(loss_pct.item())
    if float(torch.rand((), generator=gen, device=mask.device).item()) < PATIENCE:
        controller["intensity"] = max(1, min(15, controller["intensity"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < LOSS_DRIFT:
        loss_pct.fill_(max(1, min(50, int(loss_pct.item()) + random.randint(-3, 3))))

    for _ in range(controller["intensity"]):
        if controller["signal"]:
            energy_guided_flip(mask, energy, scoring_mode, energy_pct, gen, changes)
        else:
            if controller["grow"]:
                energy_guided_add(mask, energy, energy_pct, gen, changes)
            else:
                if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.7:
                    energy_guided_remove(mask, energy, scoring_mode, energy_pct, gen, changes)
                else:
                    energy_guided_rewire(mask, energy, scoring_mode, energy_pct, gen, changes)
    return prev_loss, changes


def run_one(config_name: str, scoring_mode: str, seed: int, attempts: int, energy_pct: float, checkpoint_every: int, log_q=None):
    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, _leak, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    loss_pct = torch.tensor(15, device=device, dtype=torch.int16)
    controller = controller_state()
    buffers = make_energy_eval_buffers(vocab, neurons, device)

    score, acc, energy = gpu_eval_with_energy(mask, retention_from_loss(loss_pct), targets, out_start, buffers)
    best_score = score.clone()
    best_acc = acc.clone()
    accepted = 0
    total_effective = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(1, attempts + 1):
        prev_loss, changes = mutate_energy_guided(mask, loss_pct, controller, gen, energy, scoring_mode, energy_pct)
        new_score, new_acc, new_energy = gpu_eval_with_energy(mask, retention_from_loss(loss_pct), targets, out_start, buffers)
        total_effective += changes.effective
        if bool((new_score > score).item()):
            score = new_score
            energy = new_energy
            accepted += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback_current(mask, loss_pct, prev_loss, changes)
            maybe_flip_strategy_gpu(controller, gen, device)

        if att % checkpoint_every == 0:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            aps = att / dt if dt > 0 else float("inf")
            log_msg(
                log_q,
                f"{config_name:9s} {scoring_mode:7s} seed={seed:3d} att={att:5d} "
                f"best_acc={float(best_acc.item())*100:5.1f}% score={float(best_score.item()):.4f} "
                f"density={mask_density(mask):.4f} aps={aps:.1f}",
            )

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    row = {
        "config": config_name,
        "scoring_mode": scoring_mode,
        "seed": seed,
        "energy_pct": energy_pct,
        "attempts": attempts,
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "attempts_per_sec": attempts / dt if dt > 0 else float("inf"),
        "final_density": mask_density(mask),
        "accepted": accepted,
        "mean_effective_changes_per_attempt": total_effective / attempts,
        "final_loss_pct": int(loss_pct.item()),
        "final_signal": int(controller["signal"]),
        "final_grow": int(controller["grow"]),
        "final_intensity": int(controller["intensity"]),
    }
    log_msg(log_q, "RESULT_JSON " + json.dumps(row, sort_keys=True))
    return row


def summarize(rows: list[dict], log_q) -> None:
    log_msg(log_q, "")
    log_msg(log_q, "SUMMARY")
    if not rows:
        return
    config = rows[0]["config"]
    means = {}
    for mode in SCORING_MODES:
        mode_rows = [r for r in rows if r["scoring_mode"] == mode]
        means[mode] = {
            "mean_acc": float(np.mean([r["best_acc"] for r in mode_rows])),
            "std_acc": float(np.std([r["best_acc"] for r in mode_rows])),
            "mean_score": float(np.mean([r["best_score"] for r in mode_rows])),
            "mean_aps": float(np.mean([r["attempts_per_sec"] for r in mode_rows])),
            "wins": int(sum(1 for a, b in zip(
                [r for r in mode_rows],
                [r for r in rows if r["scoring_mode"] == ("max" if mode == "product" else "product")]
            ) if a["best_acc"] > b["best_acc"])),
        }
        payload = {"config": config, "scoring_mode": mode, **means[mode]}
        log_msg(log_q, "SUMMARY_JSON " + json.dumps(payload, sort_keys=True))
    log_msg(
        log_q,
        f"{config} product={means['product']['mean_acc']*100:5.1f}% "
        f"max={means['max']['mean_acc']*100:5.1f}% "
        f"delta={(means['product']['mean_acc'] - means['max']['mean_acc'])*100:+.1f}pp",
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    args = parse_args()
    if args.config not in CONFIGS:
        raise SystemExit(f"Unknown config: {args.config}")
    seeds = parse_csv_ints(args.seeds)
    modes = parse_csv(args.modes)
    for mode in modes:
        if mode not in SCORING_MODES:
            raise SystemExit(f"Unknown scoring mode: {mode}")

    rows = []
    with live_log(args.log_name) as (log_q, log_path):
        log_msg(
            log_q,
            f"GPU ENERGY EDGE SCORE AB config={args.config} modes={modes} seeds={seeds} "
            f"attempts={args.attempts} energy_pct={args.energy_pct:.2f}",
        )
        log_msg(log_q, "=" * 120)
        for mode in modes:
            for seed in seeds:
                rows.append(
                    run_one(
                        args.config,
                        mode,
                        seed,
                        args.attempts,
                        args.energy_pct,
                        args.checkpoint_every,
                        log_q=log_q,
                    )
                )
        summarize(rows, log_q)
        log_msg(log_q, f"LOG_PATH {log_path}")


if __name__ == "__main__":
    main()
