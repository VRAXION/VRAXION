"""Deterministic GPU A/B for periodic ranked pruning as a regularizer.

Goal:
  - compare `no_prune` vs `periodic bottom-N% prune`
  - accept/add edges on TRAIN split only
  - measure both TRAIN and HOLDOUT after the same add-only proposal stream
  - keep CPU core untouched

Definition:
  - start from an empty mask
  - every attempt samples one dead edge with random sign
  - accept only if TRAIN score improves
  - prune policy periodically ranks alive edges by TRAIN delta
  - remove the bottom-N% (least harmful / most helpful to remove) in one shot
  - HOLDOUT is never used for acceptance or pruning, only for measurement
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.graph import SelfWiringGraph
from tests.gpu_experimental.gpu_full_evo_prototype import (
    CONFIGS,
    TICKS,
    BenchConfig,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")


@dataclass
class SplitEvalBuffers:
    projected_inputs: torch.Tensor
    charges: torch.Tensor
    acts: torch.Tensor
    row_idx: torch.Tensor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="V64_N192", choices=sorted(CONFIGS))
    ap.add_argument("--seeds", default="42", help="Comma-separated integer seeds")
    ap.add_argument("--attempts", type=int, default=1024)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--prune-interval", type=int, default=256)
    ap.add_argument("--prune-frac", type=float, default=0.10)
    ap.add_argument("--log-every", type=int, default=128)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--output", default="", help="Optional explicit JSON output path")
    return ap.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def split_indices(vocab: int, seed: int, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 17_003)
    order = rng.permutation(vocab)
    train_n = max(1, min(vocab - 1, int(round(vocab * train_frac))))
    train_ids = np.sort(order[:train_n])
    holdout_ids = np.sort(order[train_n:])
    return train_ids.astype(np.int64), holdout_ids.astype(np.int64)


def make_reference_init(cfg: BenchConfig, seed: int, device: torch.device):
    np.random.seed(seed)
    random.seed(seed)
    cpu_net = SelfWiringGraph(cfg.neurons, cfg.vocab)
    targets = np.random.permutation(cfg.vocab).astype(np.int64)
    mask = torch.from_numpy(cpu_net.mask.copy()).to(device=device, dtype=torch.float32)
    retain = torch.tensor(float(cpu_net.retention), device=device, dtype=torch.float32)
    w_in = torch.from_numpy(cpu_net.W_in.copy()).to(device=device, dtype=torch.float32)
    w_out = torch.from_numpy(cpu_net.W_out.copy()).to(device=device, dtype=torch.float32)
    targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.long)
    return mask, retain, w_in, w_out, targets_t, cpu_net.out_start


def make_subset_buffers(
    cfg: BenchConfig,
    input_ids: np.ndarray,
    w_in: torch.Tensor,
    device: torch.device,
) -> SplitEvalBuffers:
    batch = len(input_ids)
    inputs = torch.zeros((batch, cfg.vocab), dtype=torch.float32, device=device)
    if batch:
        rows = torch.arange(batch, device=device, dtype=torch.long)
        cols = torch.from_numpy(input_ids).to(device=device, dtype=torch.long)
        inputs[rows, cols] = 1.0
    projected_inputs = inputs @ w_in
    return SplitEvalBuffers(
        projected_inputs=projected_inputs,
        charges=torch.empty((batch, cfg.neurons), dtype=torch.float32, device=device),
        acts=torch.empty((batch, cfg.neurons), dtype=torch.float32, device=device),
        row_idx=torch.arange(batch, device=device, dtype=torch.long),
    )


def gpu_eval_subset(
    mask: torch.Tensor,
    retain: torch.Tensor,
    targets: torch.Tensor,
    out_start: int,
    buffers: SplitEvalBuffers,
    w_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    charges = buffers.charges
    acts = buffers.acts
    projected_inputs = buffers.projected_inputs
    row_idx = buffers.row_idx

    charges.zero_()
    acts.zero_()

    for t in range(TICKS):
        if t == 0:
            acts += projected_inputs
        raw = acts @ mask
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges = charges + raw
        charges = charges * retain
        acts = torch.clamp(charges - SelfWiringGraph.THRESHOLD, min=0.0)
        charges = torch.clamp(charges, -1.0, 1.0)

    logits = charges @ w_out
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    subset_targets = targets
    acc = (preds == subset_targets).to(torch.float32).mean()
    tp = probs[row_idx, subset_targets].mean()
    score = 0.5 * acc + 0.5 * tp
    return logits, score, acc


def make_split_eval_runner(
    cfg: BenchConfig,
    targets: torch.Tensor,
    out_start: int,
    train_ids: np.ndarray,
    holdout_ids: np.ndarray,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    device: torch.device,
):
    train_buf = make_subset_buffers(cfg, train_ids, w_in, device)
    hold_buf = make_subset_buffers(cfg, holdout_ids, w_in, device)
    train_targets = targets[torch.from_numpy(train_ids).to(device=device, dtype=torch.long)]
    hold_targets = targets[torch.from_numpy(holdout_ids).to(device=device, dtype=torch.long)]

    def train_eval(mask: torch.Tensor, retain: torch.Tensor):
        return gpu_eval_subset(mask, retain, train_targets, out_start, train_buf, w_out)

    def holdout_eval(mask: torch.Tensor, retain: torch.Tensor):
        return gpu_eval_subset(mask, retain, hold_targets, out_start, hold_buf, w_out)

    return train_eval, holdout_eval


def sample_dead_edge(rng: random.Random, n: int, alive_set: set[tuple[int, int]]) -> tuple[int, int] | None:
    for _ in range(256):
        row = rng.randrange(n)
        col = rng.randrange(n - 1)
        if col >= row:
            col += 1
        if (row, col) not in alive_set:
            return row, col
    for row in range(n):
        for col in range(n):
            if row != col and (row, col) not in alive_set:
                return row, col
    return None


def current_mask_hash(mask: torch.Tensor) -> str:
    return hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def edge_delta_ranking(
    mask: torch.Tensor,
    retain: torch.Tensor,
    alive: list[tuple[int, int]],
    train_eval,
) -> tuple[list[tuple[float, int, int]], float]:
    _, base_score_t, _ = train_eval(mask, retain)
    base_score = float(base_score_t.item())
    ranking: list[tuple[float, int, int]] = []
    for row, col in alive:
        old_val = int(mask[row, col].item())
        mask[row, col] = 0
        _, score_after_t, _ = train_eval(mask, retain)
        delta = float(score_after_t.item()) - base_score
        ranking.append((delta, row, col))
        mask[row, col] = old_val
    ranking.sort(reverse=True)
    return ranking, base_score


def apply_rank_prune(
    *,
    mask: torch.Tensor,
    retain: torch.Tensor,
    alive: list[tuple[int, int]],
    alive_set: set[tuple[int, int]],
    train_eval,
    holdout_eval,
    prune_frac: float,
    event_step: int,
) -> dict[str, object]:
    edges_before = len(alive)
    if edges_before == 0:
        _, train_score_t, train_acc_t = train_eval(mask, retain)
        _, hold_score_t, hold_acc_t = holdout_eval(mask, retain)
        return {
            "step": event_step,
            "edges_before": 0,
            "edges_after": 0,
            "removed": 0,
            "train_score_before": float(train_score_t.item()),
            "train_score_after": float(train_score_t.item()),
            "train_acc_after": float(train_acc_t.item()),
            "holdout_score_after": float(hold_score_t.item()),
            "holdout_acc_after": float(hold_acc_t.item()),
            "best_delta": 0.0,
            "worst_delta": 0.0,
        }

    ranking, base_train_score = edge_delta_ranking(mask, retain, alive, train_eval)
    k = max(1, int(math.floor(edges_before * prune_frac)))
    doomed = ranking[:k]
    for _, row, col in doomed:
        mask[row, col] = 0
        alive_set.remove((row, col))
    alive[:] = [edge for edge in alive if edge in alive_set]

    _, train_score_after_t, train_acc_after_t = train_eval(mask, retain)
    _, holdout_score_after_t, holdout_acc_after_t = holdout_eval(mask, retain)
    return {
        "step": event_step,
        "edges_before": edges_before,
        "edges_after": len(alive),
        "removed": k,
        "train_score_before": base_train_score,
        "train_score_after": float(train_score_after_t.item()),
        "train_acc_after": float(train_acc_after_t.item()),
        "holdout_score_after": float(holdout_score_after_t.item()),
        "holdout_acc_after": float(holdout_acc_after_t.item()),
        "best_delta": float(doomed[0][0]),
        "worst_delta": float(doomed[-1][0]),
    }


def evaluate_both(mask: torch.Tensor, retain: torch.Tensor, train_eval, holdout_eval) -> dict[str, float]:
    _, train_score_t, train_acc_t = train_eval(mask, retain)
    _, hold_score_t, hold_acc_t = holdout_eval(mask, retain)
    return {
        "train_score": float(train_score_t.item()),
        "train_acc": float(train_acc_t.item()),
        "holdout_score": float(hold_score_t.item()),
        "holdout_acc": float(hold_acc_t.item()),
    }


def run_once(
    *,
    cfg: BenchConfig,
    seed: int,
    attempts: int,
    train_frac: float,
    prune_interval: int,
    prune_frac: float,
    log_every: int,
    eps: float,
    policy: str,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    rng = random.Random(seed + 810_001 + cfg.vocab * 1000 + cfg.neurons)

    mask, retain, w_in, w_out, targets, out_start = make_reference_init(cfg, seed, device)
    mask.zero_()
    train_ids, holdout_ids = split_indices(cfg.vocab, seed, train_frac)
    train_eval, holdout_eval = make_split_eval_runner(
        cfg, targets, out_start, train_ids, holdout_ids, w_in, w_out, device
    )

    alive: list[tuple[int, int]] = []
    alive_set: set[tuple[int, int]] = set()
    metrics = evaluate_both(mask, retain, train_eval, holdout_eval)
    promotions = 0
    prune_events: list[dict[str, object]] = []
    history: list[dict[str, object]] = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, attempts + 1):
        edge = sample_dead_edge(rng, cfg.neurons, alive_set)
        if edge is None:
            break
        row, col = edge
        sign = SelfWiringGraph.DRIVE if rng.random() > 0.5 else -SelfWiringGraph.DRIVE
        old_train_score = metrics["train_score"]
        mask[row, col] = sign
        trial = evaluate_both(mask, retain, train_eval, holdout_eval)
        if trial["train_score"] > old_train_score + eps:
            alive.append((row, col))
            alive_set.add((row, col))
            promotions += 1
            metrics = trial
        else:
            mask[row, col] = 0

        if policy == "periodic_prune" and prune_interval > 0 and step % prune_interval == 0 and alive:
            event = apply_rank_prune(
                mask=mask,
                retain=retain,
                alive=alive,
                alive_set=alive_set,
                train_eval=train_eval,
                holdout_eval=holdout_eval,
                prune_frac=prune_frac,
                event_step=step,
            )
            prune_events.append(event)
            metrics = {
                "train_score": float(event["train_score_after"]),
                "train_acc": float(event["train_acc_after"]),
                "holdout_score": float(event["holdout_score_after"]),
                "holdout_acc": float(event["holdout_acc_after"]),
            }

        if log_every > 0 and (step % log_every == 0 or step == attempts):
            history.append(
                {
                    "step": step,
                    "train_score": metrics["train_score"],
                    "train_acc": metrics["train_acc"],
                    "holdout_score": metrics["holdout_score"],
                    "holdout_acc": metrics["holdout_acc"],
                    "edges": len(alive),
                    "promotions": promotions,
                }
            )

    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "config": cfg.name,
        "seed": seed,
        "policy": policy,
        "attempts": attempts,
        "train_frac": train_frac,
        "train_count": int(len(train_ids)),
        "holdout_count": int(len(holdout_ids)),
        "prune_interval": prune_interval,
        "prune_frac": prune_frac,
        "promotions": promotions,
        "candidate_evals": attempts,
        "wall_ms": wall_ms,
        "train_score_final": metrics["train_score"],
        "train_acc_final": metrics["train_acc"],
        "holdout_score_final": metrics["holdout_score"],
        "holdout_acc_final": metrics["holdout_acc"],
        "final_edges": len(alive),
        "prune_events": prune_events,
        "history": history,
        "mask_hash": current_mask_hash(mask),
    }


def run_case(**kwargs) -> dict[str, object]:
    first = run_once(**kwargs)
    second = run_once(**kwargs)
    deterministic = (
        abs(float(first["train_score_final"]) - float(second["train_score_final"])) == 0.0
        and abs(float(first["holdout_score_final"]) - float(second["holdout_score_final"])) == 0.0
        and int(first["final_edges"]) == int(second["final_edges"])
        and int(first["promotions"]) == int(second["promotions"])
        and first["mask_hash"] == second["mask_hash"]
        and len(first["prune_events"]) == len(second["prune_events"])
    )
    return {**first, "deterministic": deterministic}


def summarize(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for policy in sorted({str(r["policy"]) for r in results}):
        rows = [r for r in results if str(r["policy"]) == policy]
        out[policy] = {
            "n_cases": float(len(rows)),
            "train_score_median": float(np.median([float(r["train_score_final"]) for r in rows])),
            "holdout_score_median": float(np.median([float(r["holdout_score_final"]) for r in rows])),
            "train_acc_median": float(np.median([float(r["train_acc_final"]) for r in rows])),
            "holdout_acc_median": float(np.median([float(r["holdout_acc_final"]) for r in rows])),
            "final_edges_median": float(np.median([int(r["final_edges"]) for r in rows])),
            "wall_ms_median": float(np.median([float(r["wall_ms"]) for r in rows])),
            "promotions_median": float(np.median([int(r["promotions"]) for r in rows])),
            "deterministic_all": float(all(bool(r["deterministic"]) for r in rows)),
        }
    return out


def verdict(summary: dict[str, dict[str, float]]) -> dict[str, object]:
    base = summary["no_prune"]
    prune = summary["periodic_prune"]
    hold_gain = prune["holdout_score_median"] - base["holdout_score_median"]
    train_delta = prune["train_score_median"] - base["train_score_median"]
    edge_ratio = prune["final_edges_median"] / max(base["final_edges_median"], 1.0)
    positive = (
        prune["deterministic_all"] == 1.0
        and hold_gain >= 0.003
        and train_delta >= -0.010
    ) or (
        prune["deterministic_all"] == 1.0
        and hold_gain >= -0.003
        and edge_ratio <= 0.90
    )
    return {
        "holdout_gain": hold_gain,
        "train_delta": train_delta,
        "edge_ratio_vs_no_prune": edge_ratio,
        "positive": bool(positive),
    }


def main() -> int:
    args = parse_args()
    cfg = CONFIGS[args.config]
    seeds = parse_csv_ints(args.seeds)

    results: list[dict[str, object]] = []
    for policy in ("no_prune", "periodic_prune"):
        for seed in seeds:
            row = run_case(
                cfg=cfg,
                seed=seed,
                attempts=args.attempts,
                train_frac=args.train_frac,
                prune_interval=args.prune_interval,
                prune_frac=args.prune_frac,
                log_every=args.log_every,
                eps=args.eps,
                policy=policy,
            )
            results.append(row)
            print(
                f"{policy:14s} seed={seed:3d} "
                f"train={row['train_score_final']:.4f} hold={row['holdout_score_final']:.4f} "
                f"edges={row['final_edges']:4d} prom={row['promotions']:4d} "
                f"prunes={len(row['prune_events']):2d} det={row['deterministic']}"
            )

    summary = summarize(results)
    out = {
        "device": "cuda",
        "branch_intent": "gpu deterministic rank-prune regularization A/B",
        "args": vars(args),
        "results": results,
        "summary": summary,
        "verdict": verdict(summary),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        logs_dir = Path("S:/AI/work/VRAXION_DEV/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        output_path = logs_dir / f"gpu_rank_prune_regularization_ab_{stamp}.json"
    output_path.write_text(json.dumps(out, indent=2))
    print(f"WROTE {output_path}")
    print(json.dumps(out["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
