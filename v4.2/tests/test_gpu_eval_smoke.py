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
    candidate = candidates[0]
    device = torch.device("cuda")
    mask_t = torch.from_numpy(candidate["mask"]).to(device=device, dtype=torch.float32)
    input_projection_t = torch.from_numpy(dense_net.input_projection).to(device=device, dtype=torch.float32)
    output_projection_t = torch.from_numpy(dense_net.output_projection).to(device=device, dtype=torch.float32)
    theta_t = torch.from_numpy(candidate["theta"]).to(device=device, dtype=torch.float32)
    decay_t = torch.from_numpy(candidate["decay"]).to(device=device, dtype=torch.float32)
    targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.long)

    logits1, score1 = torch_eval_candidate(
        mask_t, input_projection_t, output_projection_t, theta_t, decay_t, targets_t
    )
    logits2, score2 = torch_eval_candidate(
        mask_t, input_projection_t, output_projection_t, theta_t, decay_t, targets_t
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
