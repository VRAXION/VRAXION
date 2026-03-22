"""GPU A/B harness for remaining hyperparameter candidates.

Focused scope:
  - same current v4.2 GPU reference semantics
  - compare a tiny candidate set on V128_N384
  - first as a short microprobe
  - then as a plateau run for the winner vs baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lib.log import live_log, log_msg
from tests.gpu_experimental.gpu_full_evo_prototype import (
    BenchConfig,
    ReferenceMutationDelta,
    add_connection_gpu_reference,
    flip_connection_gpu_reference,
    gpu_init_from_cpu,
    rand_uniform,
    randn_scaled,
    remove_connection_gpu_reference,
    rewire_connection_gpu_reference,
    rollback_gpu_reference,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    threshold: float
    ticks: int
    clip_factor: float
    mood_step: float
    mood_prob: float


CANDIDATES = {
    "baseline": Candidate("baseline", 0.5, 8, 2.0, 0.15, 0.20),
    "mprob35": Candidate("mprob35", 0.5, 8, 2.0, 0.15, 0.35),
    "mood_both": Candidate("mood_both", 0.5, 8, 2.0, 0.10, 0.35),
    "ticks6_both": Candidate("ticks6_both", 0.5, 6, 2.0, 0.10, 0.35),
}

GPU_GAIN = 2.0
GPU_CHARGE_RATE = 0.3
GPU_SELF_CONN = 0.1

PLATEAU_WINDOW = 4
PLATEAU_PATIENCE = 3
DENSITY_EPS = 0.0015
SCORE_EPS = 0.0010
ACCEPT_EPS = 48
CHECKPOINT_EVERY = 1000
MIN_ATTEMPTS = 4000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="V128_N384")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="microprobe", choices=["microprobe", "plateau"])
    ap.add_argument("--attempts", type=int, default=20000)
    ap.add_argument("--safety-cap", type=int, default=200000)
    ap.add_argument("--candidates", default="baseline,mprob35,mood_both")
    ap.add_argument("--log-name", default="remaining_params_gpu")
    return ap.parse_args()


def mask_density(mask: torch.Tensor) -> float:
    n = mask.shape[0]
    total = n * n - n
    return float((mask != 0).sum().item()) / float(total)


def window_frozen(history) -> bool:
    if len(history) < PLATEAU_WINDOW:
        return False
    chunk = history[-PLATEAU_WINDOW:]
    density_delta = abs(chunk[-1]["density"] - chunk[0]["density"])
    score_delta = abs(chunk[-1]["best_score"] - chunk[0]["best_score"])
    accept_delta = chunk[-1]["accepted"] - chunk[0]["accepted"]
    return density_delta <= DENSITY_EPS and score_delta <= SCORE_EPS and accept_delta <= ACCEPT_EPS


def gpu_eval_custom(
    mask: torch.Tensor,
    leak: torch.Tensor,
    targets: torch.Tensor,
    out_start: int,
    candidate: Candidate,
    eye: torch.Tensor,
    charges: torch.Tensor,
    acts: torch.Tensor,
    weff: torch.Tensor,
    row_idx: torch.Tensor,
):
    vocab = eye.shape[0]
    charges.zero_()
    acts.zero_()
    weff.copy_(mask)
    weff.mul_(GPU_GAIN)

    clip_bound = candidate.threshold * candidate.clip_factor

    for t in range(candidate.ticks):
        if t == 0:
            acts[:, :vocab] = eye
        raw = acts @ weff + acts * GPU_SELF_CONN
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges.add_(raw * GPU_CHARGE_RATE)
        charges.mul_(leak)
        acts.copy_(torch.clamp(charges - candidate.threshold, min=0.0))
        charges.clamp_(-clip_bound, clip_bound)

    logits = charges[:, out_start : out_start + vocab]
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = (preds == targets).to(torch.float32).mean()
    tp = probs[row_idx, targets].mean()
    score = 0.5 * acc + 0.5 * tp
    return score, acc


def mutate_custom_gpu_reference(
    mask: torch.Tensor,
    mood_x: torch.Tensor,
    mood_z: torch.Tensor,
    leak: torch.Tensor,
    gen: torch.Generator,
    diag_mask: torch.Tensor,
    candidate: Candidate,
):
    device = mask.device
    delta = ReferenceMutationDelta(
        prev_mood_x=float(mood_x.item()),
        prev_mood_z=float(mood_z.item()),
        prev_leak=float(leak.item()),
        changes=[],
    )

    if rand_uniform(gen, device) < candidate.mood_prob:
        mood_x.add_(randn_scaled(gen, device, candidate.mood_step)).clamp_(0.0, 1.0)
    if rand_uniform(gen, device) < candidate.mood_prob:
        mood_z.add_(randn_scaled(gen, device, candidate.mood_step)).clamp_(0.0, 1.0)
    if rand_uniform(gen, device) < 0.2:
        leak.add_(randn_scaled(gen, device, 0.03)).clamp_(0.5, 0.99)

    n_changes = max(1, int(1 + float(mood_z.item()) * 14))
    for _ in range(n_changes):
        mx = float(mood_x.item())
        if mx < 0.25:
            if rand_uniform(gen, device) < 0.7:
                add_connection_gpu_reference(mask, gen, diag_mask, delta.changes)
            else:
                flip_connection_gpu_reference(mask, gen, delta.changes)
        elif mx < 0.50:
            r = rand_uniform(gen, device)
            if r < 0.6:
                rewire_connection_gpu_reference(mask, gen, delta.changes)
            elif r < 0.8:
                flip_connection_gpu_reference(mask, gen, delta.changes)
            else:
                add_connection_gpu_reference(mask, gen, diag_mask, delta.changes)
        elif mx < 0.75:
            if rand_uniform(gen, device) < 0.8:
                flip_connection_gpu_reference(mask, gen, delta.changes)
            else:
                rewire_connection_gpu_reference(mask, gen, delta.changes)
        else:
            r = rand_uniform(gen, device)
            if r < 0.7:
                remove_connection_gpu_reference(mask, gen, delta.changes)
            elif r < 0.9:
                flip_connection_gpu_reference(mask, gen, delta.changes)
            else:
                rewire_connection_gpu_reference(mask, gen, delta.changes)
    return delta


def run_candidate(cfg: BenchConfig, candidate: Candidate, seed: int, mode: str, attempts: int, safety_cap: int, log_q=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, mood_x, mood_z, leak, targets, out_start = gpu_init_from_cpu(cfg, seed, device)
    diag_mask = ~torch.eye(cfg.neurons, dtype=torch.bool, device=device)
    eye = torch.eye(cfg.vocab, dtype=torch.float32, device=device)
    charges = torch.empty((cfg.vocab, cfg.neurons), dtype=torch.float32, device=device)
    acts = torch.empty((cfg.vocab, cfg.neurons), dtype=torch.float32, device=device)
    weff = torch.empty((cfg.neurons, cfg.neurons), dtype=torch.float32, device=device)
    row_idx = torch.arange(cfg.vocab, dtype=torch.long, device=device)

    score, acc = gpu_eval_custom(mask, leak, targets, out_start, candidate, eye, charges, acts, weff, row_idx)
    best_score = score.clone()
    best_acc = acc.clone()
    accepted = 0
    init_density = mask_density(mask)
    history = [{"att": 0, "density": init_density, "best_score": float(best_score.item()), "accepted": 0}]
    frozen = 0

    total_attempts = attempts if mode == "microprobe" else safety_cap
    stop_reason = "budget" if mode == "microprobe" else "safety_cap"
    stop_att = total_attempts

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(1, total_attempts + 1):
        delta = mutate_custom_gpu_reference(mask, mood_x, mood_z, leak, gen, diag_mask, candidate)
        new_score, new_acc = gpu_eval_custom(mask, leak, targets, out_start, candidate, eye, charges, acts, weff, row_idx)

        if bool((new_score > score).item()):
            score = new_score
            accepted += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback_gpu_reference(mask, mood_x, mood_z, leak, delta)

        if att % CHECKPOINT_EVERY == 0:
            density_now = mask_density(mask)
            history.append(
                {
                    "att": att,
                    "density": density_now,
                    "best_score": float(best_score.item()),
                    "accepted": accepted,
                }
            )
            frozen = frozen + 1 if window_frozen(history) else 0
            log_msg(
                log_q,
                f"{cfg.name:10s} {candidate.name:10s} seed={seed:3d} att={att:6d} "
                f"best_acc={float(best_acc.item())*100:5.1f}% score={float(best_score.item()):.4f} "
                f"density={density_now:0.4f} leak={float(leak.item()):0.3f} accepted={accepted:5d} frozen={frozen:2d}",
            )
            if mode == "plateau" and att >= MIN_ATTEMPTS and frozen >= PLATEAU_PATIENCE:
                stop_reason = "plateau"
                stop_att = att
                break

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    result = {
        "config": cfg.name,
        "candidate": candidate.name,
        "seed": seed,
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "final_density": mask_density(mask),
        "final_leak": float(leak.item()),
        "accepted": accepted,
        "attempts": stop_att,
        "attempts_per_sec": stop_att / dt if dt > 0 else float("inf"),
        "stop_reason": stop_reason,
    }
    log_msg(
        log_q,
        f"{cfg.name:10s} {candidate.name:10s} seed={seed:3d} FINAL "
        f"acc={result['best_acc']*100:5.1f}% score={result['best_score']:.4f} "
        f"density={init_density:0.4f}->{result['final_density']:0.4f} leak={result['final_leak']:0.3f} "
        f"accepted={accepted:5d} stop={stop_reason}@{stop_att} aps={result['attempts_per_sec']:.1f}",
    )
    return result


def main():
    args = parse_args()
    if args.config not in {"V64_N192", "V128_N384", "V128_dense", "V256_N768"}:
        raise ValueError(f"unsupported config: {args.config}")
    cfg_map = {
        "V64_N192": BenchConfig("V64_N192", 64, 192, 0.06),
        "V128_N384": BenchConfig("V128_N384", 128, 384, 0.06),
        "V128_dense": BenchConfig("V128_dense", 128, 384, 0.15),
        "V256_N768": BenchConfig("V256_N768", 256, 768, 0.06),
    }
    cfg = cfg_map[args.config]
    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    candidates = [CANDIDATES[n] for n in names]

    print(f"GPU PARAM A/B mode={args.mode} config={cfg.name} seed={args.seed} candidates={','.join(names)}", flush=True)
    print(f"attempts={args.attempts} safety_cap={args.safety_cap}", flush=True)
    print("=" * 100, flush=True)

    with live_log(args.log_name, log_dir=os.path.join(os.path.dirname(__file__), "..", "logs")) as (log_q, _):
        for cand in candidates:
            run_candidate(cfg, cand, args.seed, args.mode, args.attempts, args.safety_cap, log_q=log_q)


if __name__ == "__main__":
    raise SystemExit(main())
