"""GPU microprobe: existing-edge energy guidance via shortlist instead of weights.

Question:
  Can a thresholded hot-edge shortlist keep most of the quality benefit of
  weighted max-scoring while avoiding some weighted-sampling overhead?

This is still a correctness-first reference path. It is not yet a fully cached
GPU-native alive-edge backend.
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
from tests.gpu_energy_edge_score_ab import (
    CHECKPOINT_EVERY,
    DEFAULT_ATTEMPTS,
    DEFAULT_CONFIG,
    DEFAULT_ENERGY_PCT,
    DEFAULT_SEEDS,
    GuidedChanges,
    controller_state,
    gpu_eval_with_energy,
    make_energy_eval_buffers,
    mask_density,
    maybe_flip_strategy_gpu,
    parse_csv,
    parse_csv_ints,
    retention_from_loss,
)
from tests.gpu_int_mood_ab import CONFIGS, gpu_init


MODES = ("random", "weighted_max", "hot_threshold")
DEFAULT_THRESHOLD = 0.05


@dataclass
class PickStats:
    selection_calls: int = 0
    guided_calls: int = 0
    random_calls: int = 0
    hot_hits: int = 0
    hot_fallbacks: int = 0
    total_alive_seen: int = 0
    total_hot_seen: int = 0
    selection_seconds: float = 0.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    ap.add_argument("--energy-pct", type=float, default=DEFAULT_ENERGY_PCT)
    ap.add_argument("--modes", default="random,weighted_max,hot_threshold")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--log-name", default="gpu_energy_hotlist_ab")
    return ap.parse_args()


def pick_existing_edge(
    mask: torch.Tensor,
    energy: torch.Tensor,
    mode: str,
    energy_pct: float,
    threshold: float,
    gen: torch.Generator,
    stats: PickStats,
):
    t0 = time.perf_counter()
    alive = torch.nonzero(mask != 0, as_tuple=False)
    stats.selection_calls += 1
    stats.total_alive_seen += int(alive.shape[0])
    if alive.numel() == 0:
        stats.selection_seconds += time.perf_counter() - t0
        return None, None

    if mode == "random":
        idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
        stats.random_calls += 1
        stats.selection_seconds += time.perf_counter() - t0
        return alive, idx

    use_guided = float(torch.rand((), generator=gen, device=mask.device).item()) < energy_pct
    if not use_guided:
        idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
        stats.random_calls += 1
        stats.selection_seconds += time.perf_counter() - t0
        return alive, idx

    stats.guided_calls += 1
    src_e = energy.index_select(0, alive[:, 0]).clamp_min(0)
    dst_e = energy.index_select(0, alive[:, 1]).clamp_min(0)

    if mode == "weighted_max":
        weights = torch.maximum(src_e, dst_e)
        if bool((weights > 0).any().item()):
            idx = int(torch.multinomial(weights, 1, replacement=True, generator=gen).item())
            stats.selection_seconds += time.perf_counter() - t0
            return alive, idx
        idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
        stats.random_calls += 1
        stats.selection_seconds += time.perf_counter() - t0
        return alive, idx

    if mode == "hot_threshold":
        hot_mask = (src_e > threshold) & (dst_e > threshold)
        hot = alive[hot_mask]
        hot_count = int(hot.shape[0])
        stats.total_hot_seen += hot_count
        if hot_count > 0:
            idx = int(torch.randint(hot_count, (1,), generator=gen, device=mask.device).item())
            stats.hot_hits += 1
            stats.selection_seconds += time.perf_counter() - t0
            return hot, idx
        idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
        stats.hot_fallbacks += 1
        stats.random_calls += 1
        stats.selection_seconds += time.perf_counter() - t0
        return alive, idx

    stats.selection_seconds += time.perf_counter() - t0
    raise ValueError(f"unknown mode: {mode}")


def pick_src_index(
    neurons: int,
    energy: torch.Tensor,
    energy_pct: float,
    gen: torch.Generator,
    device: torch.device,
) -> int:
    use_energy = float(torch.rand((), generator=gen, device=device).item()) < energy_pct
    if use_energy:
        weights = energy.clamp_min(0)
        if bool((weights > 0).any().item()):
            return int(torch.multinomial(weights, 1, replacement=True, generator=gen).item())
    return int(torch.randint(neurons, (1,), generator=gen, device=device).item())


def guided_add(
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


def guided_flip(
    mask: torch.Tensor,
    energy: torch.Tensor,
    mode: str,
    energy_pct: float,
    threshold: float,
    gen: torch.Generator,
    changes: GuidedChanges,
    stats: PickStats,
) -> bool:
    alive, idx = pick_existing_edge(mask, energy, mode, energy_pct, threshold, gen, stats)
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


def guided_remove(
    mask: torch.Tensor,
    energy: torch.Tensor,
    mode: str,
    energy_pct: float,
    threshold: float,
    gen: torch.Generator,
    changes: GuidedChanges,
    stats: PickStats,
) -> bool:
    alive, idx = pick_existing_edge(mask, energy, mode, energy_pct, threshold, gen, stats)
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


def guided_rewire(
    mask: torch.Tensor,
    energy: torch.Tensor,
    mode: str,
    energy_pct: float,
    threshold: float,
    gen: torch.Generator,
    changes: GuidedChanges,
    stats: PickStats,
) -> bool:
    alive, idx = pick_existing_edge(mask, energy, mode, energy_pct, threshold, gen, stats)
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


def mutate_guided(
    mask: torch.Tensor,
    loss_pct: torch.Tensor,
    controller: dict[str, int],
    gen: torch.Generator,
    energy: torch.Tensor,
    mode: str,
    energy_pct: float,
    threshold: float,
    stats: PickStats,
) -> tuple[int, GuidedChanges]:
    changes = GuidedChanges(cells=[])
    prev_loss = int(loss_pct.item())
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["intensity"] = max(1, min(15, controller["intensity"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        loss_pct.fill_(max(1, min(50, int(loss_pct.item()) + random.randint(-3, 3))))

    for _ in range(controller["intensity"]):
        if controller["signal"]:
            guided_flip(mask, energy, mode, energy_pct, threshold, gen, changes, stats)
        else:
            if controller["grow"]:
                guided_add(mask, energy, energy_pct, gen, changes)
            else:
                if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.7:
                    guided_remove(mask, energy, mode, energy_pct, threshold, gen, changes, stats)
                else:
                    guided_rewire(mask, energy, mode, energy_pct, threshold, gen, changes, stats)
    return prev_loss, changes


def run_one(
    config_name: str,
    mode: str,
    seed: int,
    attempts: int,
    energy_pct: float,
    threshold: float,
    checkpoint_every: int,
    log_q=None,
):
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
    stats = PickStats()

    score, acc, energy = gpu_eval_with_energy(mask, retention_from_loss(loss_pct), targets, out_start, buffers)
    best_score = score.clone()
    best_acc = acc.clone()
    accepted = 0
    total_effective = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(1, attempts + 1):
        prev_loss, changes = mutate_guided(mask, loss_pct, controller, gen, energy, mode, energy_pct, threshold, stats)
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
                f"{config_name:9s} {mode:12s} seed={seed:3d} att={att:5d} "
                f"best_acc={float(best_acc.item())*100:5.1f}% score={float(best_score.item()):.4f} "
                f"density={mask_density(mask):.4f} aps={aps:.1f}",
            )

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    row = {
        "config": config_name,
        "mode": mode,
        "seed": seed,
        "threshold": threshold,
        "energy_pct": energy_pct,
        "attempts": attempts,
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "attempts_per_sec": attempts / dt if dt > 0 else float("inf"),
        "final_density": mask_density(mask),
        "accepted": accepted,
        "mean_effective_changes_per_attempt": total_effective / attempts,
        "selection_ms_per_attempt": (stats.selection_seconds * 1000.0) / max(1, attempts),
        "guided_pick_rate": stats.guided_calls / max(1, stats.selection_calls),
        "hot_hit_rate": stats.hot_hits / max(1, stats.guided_calls),
        "hot_fallback_rate": stats.hot_fallbacks / max(1, stats.guided_calls),
        "mean_alive_edges_seen": stats.total_alive_seen / max(1, stats.selection_calls),
        "mean_hot_edges_seen": stats.total_hot_seen / max(1, stats.guided_calls),
    }
    log_msg(log_q, "RESULT_JSON " + json.dumps(row, sort_keys=True))
    return row


def summarize(rows: list[dict], log_q) -> None:
    log_msg(log_q, "")
    log_msg(log_q, "SUMMARY")
    if not rows:
        return
    config = rows[0]["config"]
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            continue
        payload = {
            "config": config,
            "mode": mode,
            "mean_acc": float(np.mean([r["best_acc"] for r in mode_rows])),
            "std_acc": float(np.std([r["best_acc"] for r in mode_rows])),
            "mean_score": float(np.mean([r["best_score"] for r in mode_rows])),
            "mean_aps": float(np.mean([r["attempts_per_sec"] for r in mode_rows])),
            "mean_selection_ms": float(np.mean([r["selection_ms_per_attempt"] for r in mode_rows])),
            "mean_hot_hit_rate": float(np.mean([r["hot_hit_rate"] for r in mode_rows])),
            "mean_hot_fallback_rate": float(np.mean([r["hot_fallback_rate"] for r in mode_rows])),
        }
        log_msg(log_q, "SUMMARY_JSON " + json.dumps(payload, sort_keys=True))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    args = parse_args()
    if args.config not in CONFIGS:
        raise SystemExit(f"Unknown config: {args.config}")
    seeds = parse_csv_ints(args.seeds)
    modes = parse_csv(args.modes)
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f"Unknown mode: {mode}")

    rows = []
    with live_log(args.log_name) as (log_q, log_path):
        log_msg(
            log_q,
            f"GPU ENERGY HOTLIST AB config={args.config} modes={modes} seeds={seeds} "
            f"attempts={args.attempts} energy_pct={args.energy_pct:.2f} threshold={args.threshold:.4f}",
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
                        args.threshold,
                        args.checkpoint_every,
                        log_q=log_q,
                    )
                )
        summarize(rows, log_q)
        log_msg(log_q, f"LOG_PATH {log_path}")


if __name__ == "__main__":
    main()
