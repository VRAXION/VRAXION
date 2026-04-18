"""Run the exact-first H=81 merger quant pipeline end to end.

Stages:
  1. Exact H=81 pure-float trainer.
  2. Exact lookup-codebook freeze.
  3. Strict staged int8 freeze.

Plastic/meta compression is intentionally not part of this driver yet. The
exact-first path should stall naturally first; only then should local scales
or structured bucket metadata be introduced.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).parent
PYTHON = sys.executable


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--float-init-source", type=str, default="")
    parser.add_argument("--float-seed", type=int, default=42)
    parser.add_argument("--float-restarts", type=int, default=1)
    parser.add_argument("--float-adam-epochs", type=int, default=2500)
    parser.add_argument("--float-lbfgs-outer", type=int, default=250)
    parser.add_argument("--lookup-max-iters", type=int, default=24)
    parser.add_argument("--lookup-retrain-outer", type=int, default=60)
    parser.add_argument("--int8-top-k", type=int, default=16)
    parser.add_argument("--int8-candidate-width", type=int, default=2)
    parser.add_argument("--int8-max-accepts", type=int, default=20)
    parser.add_argument("--int8-short-retrain-iters", type=int, default=10)
    parser.add_argument("--int8-full-retrain-iters", type=int, default=40)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    float_out = out_root / "01_exact_float"
    lookup_out = out_root / "02_lookup_exact"
    int8_out = out_root / "03_strict_int8"
    t0 = time.time()

    float_cmd = [
        PYTHON,
        str(THIS_DIR / "diag_byte_pair_merger_exact_h81_float.py"),
        "--out",
        str(float_out),
        "--seed",
        str(args.float_seed),
        "--restarts",
        str(args.float_restarts),
        "--adam-epochs",
        str(args.float_adam_epochs),
        "--lbfgs-outer",
        str(args.float_lbfgs_outer),
    ]
    if args.float_init_source:
        float_cmd.extend(["--init-source", args.float_init_source])
    run_step("STAGE 1: EXACT FLOAT", float_cmd)

    lookup_cmd = [
        PYTHON,
        str(THIS_DIR / "diag_byte_pair_merger_lookup_codebook_exact.py"),
        "--source",
        str(float_out / "final_model.json"),
        "--out",
        str(lookup_out),
        "--max-iters",
        str(args.lookup_max_iters),
        "--retrain-outer",
        str(args.lookup_retrain_outer),
    ]
    run_step("STAGE 2: EXACT LOOKUP", lookup_cmd)

    int8_cmd = [
        PYTHON,
        str(THIS_DIR / "diag_byte_pair_merger_strict_staged_int8.py"),
        "--source",
        str(lookup_out / "final_model.json"),
        "--out",
        str(int8_out),
        "--top-k",
        str(args.int8_top_k),
        "--candidate-width",
        str(args.int8_candidate_width),
        "--max-accepts",
        str(args.int8_max_accepts),
        "--short-retrain-iters",
        str(args.int8_short_retrain_iters),
        "--full-retrain-iters",
        str(args.int8_full_retrain_iters),
    ]
    run_step("STAGE 3: STRICT STAGED INT8", int8_cmd)

    summary = {
        "time_s": time.time() - t0,
        "float_final": str(float_out / "final_model.json"),
        "lookup_final": str(lookup_out / "final_model.json"),
        "int8_final": str(int8_out / "final_model.json"),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PIPELINE DONE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
