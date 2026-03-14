"""Sanity + adversarial test harness for gpu_full_evo_prototype.py.

Ordered checks:
  1. Initial CPU/GPU evaluation parity on the same seeded net snapshot
  2. Determinism of the full GPU evolutionary loop
  3. Boundedness / semantic invariants after short runs
  4. Adversarial config sweep over dense/normal large configs
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gpu_full_evo_prototype import (
    CONFIGS,
    BenchConfig,
    TICKS,
    cpu_train,
    determinism_check,
    gpu_eval,
    gpu_init_from_cpu,
    gpu_train,
    make_cpu_reference,
)


PARITY_CONFIGS = ["V64_N192", "V64_dense", "V128_N384", "V128_dense"]
ADVERSARIAL_CONFIGS = ["V64_dense", "V128_N384", "V128_dense"]
SEEDS = [42, 77]
DET_ATTEMPTS = 120
SHORT_ATTEMPTS = 120


def initial_eval_parity(cfg: BenchConfig, seed: int) -> tuple[float, float]:
    cpu_net, targets = make_cpu_reference(cfg, seed)
    logits_cpu = cpu_net.forward_batch(TICKS)
    e = np.exp(logits_cpu - logits_cpu.max(axis=1, keepdims=True))
    probs_cpu = e / e.sum(axis=1, keepdims=True)
    preds_cpu = np.argmax(probs_cpu, axis=1)

    device = torch.device("cuda")
    mask, mood_x, mood_z, leak, targets_t, out_start = gpu_init_from_cpu(cfg, seed, device)
    eye = torch.eye(cfg.vocab, dtype=torch.float32, device=device)
    logits_gpu_t, score_gpu, acc_gpu = gpu_eval(mask, leak, targets_t, out_start, eye)
    logits_gpu = logits_gpu_t.detach().cpu().numpy()
    preds_gpu = np.argmax(logits_gpu, axis=1)

    max_abs = float(np.max(np.abs(logits_cpu - logits_gpu)))
    pred_agree = float((preds_cpu == preds_gpu).mean())
    return max_abs, pred_agree


def bounded_and_semantic(cfg: BenchConfig, seed: int, attempts: int):
    res = gpu_train(cfg, attempts, seed, verbose_every=0)
    ok = True
    reasons = []
    if not math.isfinite(res["best_acc"]):
        ok = False
        reasons.append("best_acc not finite")
    if not math.isfinite(res["best_score"]):
        ok = False
        reasons.append("best_score not finite")
    if not math.isfinite(res["final_leak"]):
        ok = False
        reasons.append("final_leak not finite")
    if not (0.5 <= res["final_leak"] <= 0.99 + 1e-6):
        ok = False
        reasons.append("final_leak out of bounds")
    if res["kept"] < 0 or res["kept"] > attempts:
        ok = False
        reasons.append("kept out of bounds")
    if res["attempts_per_sec"] <= 0:
        ok = False
        reasons.append("attempts_per_sec <= 0")
    return ok, reasons, res


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    print("GPU FULL EVO TEST SUITE")
    print("=" * 80)

    # 1) Initial parity
    parity_fail = False
    print("1) INITIAL CPU/GPU PARITY")
    for name in PARITY_CONFIGS:
        cfg = CONFIGS[name]
        for seed in SEEDS:
            max_abs, pred_agree = initial_eval_parity(cfg, seed)
            print(
                f"  {name:10s} seed={seed:3d} max_abs={max_abs:.3e} pred_agree={pred_agree:.3f}"
            )
            if max_abs > 1e-5 or pred_agree < 1.0:
                parity_fail = True

    # 2) Determinism
    det_fail = False
    print("\n2) DETERMINISM")
    for name in ["V64_N192", "V128_N384"]:
        cfg = CONFIGS[name]
        rc = determinism_check(cfg, DET_ATTEMPTS, 42)
        print(f"  {name:10s} deterministic_rc={rc}")
        if rc != 0:
            det_fail = True

    # 3) Boundedness / semantic invariants
    bound_fail = False
    print("\n3) BOUNDEDNESS / SEMANTICS")
    for name in ADVERSARIAL_CONFIGS:
        cfg = CONFIGS[name]
        for seed in SEEDS:
            ok, reasons, res = bounded_and_semantic(cfg, seed, SHORT_ATTEMPTS)
            print(
                f"  {name:10s} seed={seed:3d} ok={ok} "
                f"acc={res['best_acc']*100:5.1f}% aps={res['attempts_per_sec']:6.1f} "
                f"leak={res['final_leak']:.4f} kept={res['kept']:3d}"
            )
            if not ok:
                bound_fail = True
                print(f"    reasons={reasons}")

    failed = parity_fail or det_fail or bound_fail
    print("\nSUMMARY")
    print(
        {
            "parity_fail": parity_fail,
            "determinism_fail": det_fail,
            "boundedness_fail": bound_fail,
            "failed": failed,
        }
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
