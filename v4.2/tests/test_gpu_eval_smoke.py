"""Deterministic smoke for the v4.2 GPU eval path."""

from __future__ import annotations

import numpy as np
import torch

from gpu_eval_bench_torch import BenchConfig, SEED, generate_candidates, torch_eval_candidate


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    cfg = BenchConfig("V128_N384", 128, 384, 0.06)
    _, dense_net, _, _, targets, candidates = generate_candidates(cfg)
    mask, leak = candidates[0]
    device = torch.device("cuda")
    mask_i8 = torch.from_numpy(mask).to(device=device, dtype=torch.int8)
    targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.long)

    logits1, score1 = torch_eval_candidate(
        mask_i8, leak, dense_net.V, dense_net.N, dense_net.threshold, dense_net.clip_factor,
        dense_net.self_conn, dense_net.charge_rate, dense_net.gain, dense_net.out_start, targets_t
    )
    logits2, score2 = torch_eval_candidate(
        mask_i8, leak, dense_net.V, dense_net.N, dense_net.threshold, dense_net.clip_factor,
        dense_net.self_conn, dense_net.charge_rate, dense_net.gain, dense_net.out_start, targets_t
    )
    torch.cuda.synchronize()

    arr1 = logits1.detach().cpu().numpy()
    arr2 = logits2.detach().cpu().numpy()
    preds1 = np.argmax(arr1, axis=1)
    preds2 = np.argmax(arr2, axis=1)

    logits_equal = np.array_equal(arr1.view(np.uint32), arr2.view(np.uint32))
    preds_equal = np.array_equal(preds1, preds2)
    score_equal = score1 == score2

    print(
        {
            "logits_bitwise_equal": logits_equal,
            "predictions_equal": preds_equal,
            "score_equal": score_equal,
            "score": score1,
        }
    )
    return 0 if logits_equal and preds_equal and score_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
