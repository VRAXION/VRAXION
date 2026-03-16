import importlib.util
import argparse
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"S:\AI\work\VRAXION_DEV")
SRC = ROOT / "v4.2" / "model" / "graph_v3.c"
PY_SRC = ROOT / "v4.2" / "model" / "graph.py"
OUT_DIR = ROOT / "v4.2" / "model" / "_probe_bins"
OUT_DIR.mkdir(exist_ok=True)

FINAL_RE = re.compile(r"Final:\s+([0-9.]+)%")


def compile_variant(name: str, extra_flags: list[str]) -> Path:
    exe = OUT_DIR / f"{name}.exe"
    cmd = ["gcc", "-O3", "-std=c11", "-Wall", "-Wextra", "-o", str(exe), str(SRC), "-lm", "-DVERBOSE_DEFAULT=0", *extra_flags]
    subprocess.run(cmd, check=True)
    return exe


def run_variant(exe: Path, seeds: list[int], vocab: int = 64, budget: int = 16000):
    scores = []
    for seed in seeds:
        proc = subprocess.run([str(exe), str(vocab), str(seed), str(budget)], capture_output=True, text=True, check=True)
        m = FINAL_RE.search(proc.stdout)
        if not m:
            raise RuntimeError(f"could not parse output for {exe.name} seed={seed}\n{proc.stdout}\n{proc.stderr}")
        scores.append(float(m.group(1)))
    return scores


def summarize(name: str, scores: list[float]):
    mean = statistics.mean(scores)
    std = statistics.pstdev(scores)
    print(f"{name:16s} mean={mean:5.2f}% std={std:4.2f} min={min(scores):5.1f} max={max(scores):5.1f}")
    print("  seeds:", " ".join(f"{v:.1f}" for v in scores))


def load_python_graph():
    spec = importlib.util.spec_from_file_location("graph_py_probe", PY_SRC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PY_SRC}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["graph_py_probe"] = module
    spec.loader.exec_module(module)
    return module


def run_python(seeds: list[int], vocab: int = 64, budget: int = 16000):
    graph = load_python_graph()
    scores = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        net = graph.SelfWiringGraph(vocab)
        targets = np.arange(vocab)
        np.random.shuffle(targets)
        best = graph.train(net, targets, vocab, max_attempts=budget, verbose=False)
        scores.append(float(best * 100.0))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--budget", type=int, default=16000)
    parser.add_argument("--seeds", default="0,1,2,10,11,13,15,17,18,19")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    summarize("python_canon", run_python(seeds, vocab=args.vocab, budget=args.budget))
    variants = [
        ("loss1_seedraw", ["-DINIT_LOSS_PCT=1", "-DRNG_SEED_MODE=0"]),
        ("loss15_seedraw", ["-DINIT_LOSS_PCT=15", "-DRNG_SEED_MODE=0"]),
        ("loss1_seedmix", ["-DINIT_LOSS_PCT=1", "-DRNG_SEED_MODE=1"]),
        ("loss15_seedmix", ["-DINIT_LOSS_PCT=15", "-DRNG_SEED_MODE=1"]),
    ]

    for name, flags in variants:
        exe = compile_variant(name, flags)
        scores = run_variant(exe, seeds, vocab=args.vocab, budget=args.budget)
        summarize(name, scores)


if __name__ == "__main__":
    main()
