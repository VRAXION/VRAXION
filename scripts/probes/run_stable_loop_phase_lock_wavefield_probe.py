from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "research" / "STABLE_LOOP_PHASE_LOCK_005_WAVEFIELD_PROPAGATION_CONTRACT.md"
PHASE_CLASSES = 4
EPS = 1.0e-9

ARMS = (
    "PARTICLE_FRONTIER_004_BASELINE",
    "COMPLEX_WAVEFIELD_PROPAGATION",
    "WAVEFIELD_WITH_INTERFERENCE",
    "WAVEFIELD_NO_INTERFERENCE_ABLATION",
    "WAVEFIELD_WITH_LOCAL_PHASE_LOSS",
    "WAVEFIELD_TARGET_ONLY_LOSS",
)


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    width: int
    free: tuple[tuple[int, ...], ...]
    source: tuple[int, int]
    target: tuple[int, int]
    source_phase: int
    gate_real: tuple[tuple[float, ...], ...]
    gate_imag: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PrivateCase:
    case_id: str
    label: int
    family: str
    path_len: int
    same_target_pair: str | None


@dataclass(frozen=True)
class CaseBundle:
    public: PublicCase
    private: PrivateCase


def parse_csv(text: str | None) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in parse_csv(text):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def unit_phase(k: int, classes: int = PHASE_CLASSES) -> complex:
    theta = 2.0 * math.pi * (k % classes) / classes
    return complex(math.cos(theta), math.sin(theta))


def phase_bucket(z: complex, classes: int = PHASE_CLASSES) -> int:
    if abs(z) < 1.0e-6:
        return -1
    theta = math.atan2(z.imag, z.real)
    if theta < 0:
        theta += 2.0 * math.pi
    return int(round(theta / (2.0 * math.pi / classes))) % classes


def phase_probabilities(z: complex, classes: int = PHASE_CLASSES, temperature: float = 0.35) -> list[float]:
    mag = abs(z)
    if mag < 1.0e-8:
        return [1.0 / classes for _ in range(classes)]
    theta = math.atan2(z.imag, z.real)
    logits = []
    for k in range(classes):
        kt = 2.0 * math.pi * k / classes
        # Smooth Born-like target score: stronger amplitude and closer angle increase probability.
        logits.append((mag * math.cos(theta - kt)) / max(temperature, 1.0e-6))
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    denom = sum(exps)
    return [v / denom for v in exps]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")


def grid_to_tuple(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in arr.tolist())


def float_grid_to_tuple(arr: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in arr.tolist())


def carve_line(free: np.ndarray, path: list[tuple[int, int]]) -> None:
    h, w = free.shape
    for y, x in path:
        if 0 <= y < h and 0 <= x < w:
            free[y, x] = 1


def make_path(a: tuple[int, int], b: tuple[int, int], *, y_first: bool) -> list[tuple[int, int]]:
    y, x = a
    ty, tx = b
    path = [(y, x)]
    if y_first:
        step = 1 if ty >= y else -1
        while y != ty:
            y += step
            path.append((y, x))
        step = 1 if tx >= x else -1
        while x != tx:
            x += step
            path.append((y, x))
    else:
        step = 1 if tx >= x else -1
        while x != tx:
            x += step
            path.append((y, x))
        step = 1 if ty >= y else -1
        while y != ty:
            y += step
            path.append((y, x))
    return path


def apply_gate_pattern(
    gate: np.ndarray,
    path: list[tuple[int, int]],
    rng: random.Random,
    *,
    target_total: int | None = None,
) -> int:
    vals: list[int] = []
    for _ in path[1:]:
        vals.append(rng.randrange(PHASE_CLASSES))
    if target_total is not None and vals:
        current = sum(vals) % PHASE_CLASSES
        vals[-1] = (vals[-1] + target_total - current) % PHASE_CLASSES
    for (y, x), k in zip(path[1:], vals):
        gate[y, x] = unit_phase(k)
    return sum(vals) % PHASE_CLASSES


def build_case(seed: int, idx: int, width: int) -> CaseBundle:
    rng = random.Random(seed * 100_003 + idx)
    family = rng.choice(
        [
            "single_corridor",
            "wrong_fast_correct_slow",
            "same_target_counterfactual_a",
            "same_target_counterfactual_b",
            "interference_cancel",
        ]
    )
    free = np.zeros((width, width), dtype=np.int8)
    gate = np.ones((width, width), dtype=np.complex64)
    sy = rng.randrange(2, width - 2)
    src = (sy, 1)
    tgt = (rng.randrange(2, width - 2), width - 2)
    source_phase = rng.randrange(PHASE_CLASSES)
    pair_id: str | None = None

    if family.startswith("same_target_counterfactual"):
        pair_id = f"pair_{idx // 2}"
        src = (width // 2, 1)
        tgt = (width // 2, width - 2)
        top = make_path(src, (max(1, width // 3), width // 2), y_first=True)
        top += make_path(top[-1], tgt, y_first=False)[1:]
        bottom = make_path(src, (min(width - 2, 2 * width // 3), width // 2), y_first=True)
        bottom += make_path(bottom[-1], tgt, y_first=False)[1:]
        for p in (top, bottom):
            carve_line(free, p)
        wanted = 0 if family.endswith("_a") else 2
        apply_gate_pattern(gate, top, rng, target_total=wanted)
        apply_gate_pattern(gate, bottom, rng, target_total=wanted)
    elif family == "wrong_fast_correct_slow":
        fast = make_path(src, tgt, y_first=False)
        slow_mid = (min(width - 2, max(1, sy + rng.choice([-4, 4]))), width // 2)
        slow = make_path(src, slow_mid, y_first=True) + make_path(slow_mid, tgt, y_first=False)[1:]
        carve_line(free, fast)
        carve_line(free, slow)
        apply_gate_pattern(gate, fast, rng, target_total=1)
        apply_gate_pattern(gate, slow, rng, target_total=0)
        # Give the correct slow route two close feeder branches so a wavefield can accumulate evidence.
        if 1 < slow_mid[0] < width - 2:
            aux = make_path((slow_mid[0] + rng.choice([-1, 1]), 1), slow_mid, y_first=False)
            carve_line(free, aux)
            apply_gate_pattern(gate, aux, rng, target_total=0)
    elif family == "interference_cancel":
        mid = (width // 2, width // 2)
        p1 = make_path(src, mid, y_first=False) + make_path(mid, tgt, y_first=True)[1:]
        p2_start = (min(width - 2, max(1, sy + 2)), 1)
        p2 = make_path(p2_start, mid, y_first=False) + make_path(mid, tgt, y_first=True)[1:]
        p3_start = (min(width - 2, max(1, sy - 2)), 1)
        p3 = make_path(p3_start, mid, y_first=False) + make_path(mid, tgt, y_first=True)[1:]
        for p in (p1, p2, p3):
            carve_line(free, p)
        apply_gate_pattern(gate, p1, rng, target_total=0)
        apply_gate_pattern(gate, p2, rng, target_total=2)
        apply_gate_pattern(gate, p3, rng, target_total=0)
    else:
        p = make_path(src, tgt, y_first=rng.random() < 0.5)
        carve_line(free, p)
        apply_gate_pattern(gate, p, rng, target_total=rng.randrange(PHASE_CLASSES))

    free[src] = 1
    free[tgt] = 1
    public = PublicCase(
        case_id=f"{seed}_{idx}_{family}",
        width=width,
        free=grid_to_tuple(free),
        source=src,
        target=tgt,
        source_phase=source_phase,
        gate_real=float_grid_to_tuple(gate.real),
        gate_imag=float_grid_to_tuple(gate.imag),
    )
    z = wavefield_target(public, steps=max(width * 2, 8), interference=True, gate_alpha=1.0)
    label = phase_bucket(z)
    if label < 0:
        label = source_phase
    private = PrivateCase(
        case_id=public.case_id,
        label=label,
        family=family,
        path_len=abs(src[0] - tgt[0]) + abs(src[1] - tgt[1]),
        same_target_pair=pair_id,
    )
    return CaseBundle(public=public, private=private)


def make_cases(seed: int, n: int, width: int) -> list[CaseBundle]:
    return [build_case(seed, i, width) for i in range(n)]


def neighbors_sum(state: np.ndarray) -> np.ndarray:
    out = np.zeros_like(state)
    out[1:, :] += state[:-1, :]
    out[:-1, :] += state[1:, :]
    out[:, 1:] += state[:, :-1]
    out[:, :-1] += state[:, 1:]
    return out


def public_arrays(public: PublicCase) -> tuple[np.ndarray, np.ndarray]:
    free = np.array(public.free, dtype=np.float32)
    gate = np.array(public.gate_real, dtype=np.float32) + 1j * np.array(public.gate_imag, dtype=np.float32)
    return free, gate


def mix_gate(gate: np.ndarray, gate_alpha: float) -> np.ndarray:
    mixed = (1.0 - gate_alpha) + gate_alpha * gate
    mag = np.maximum(np.abs(mixed), 1.0e-6)
    return mixed / mag


def wavefield_target(
    public: PublicCase,
    *,
    steps: int,
    interference: bool,
    gate_alpha: float,
    damping: float = 0.15,
) -> complex:
    free, gate0 = public_arrays(public)
    gate = mix_gate(gate0, gate_alpha)
    state = np.zeros((public.width, public.width), dtype=np.complex64)
    source_z = unit_phase(public.source_phase)
    sy, sx = public.source
    ty, tx = public.target
    for _ in range(steps):
        incoming = neighbors_sum(state) * gate * free
        if not interference:
            # Keep phase direction but remove destructive cancellation by adding magnitudes.
            incoming = np.abs(neighbors_sum(state)) * gate * free
        state = (damping * state + incoming) * free
        state[sy, sx] += source_z
        mag = np.abs(state)
        too_big = mag > 4.0
        state[too_big] = state[too_big] / mag[too_big] * 4.0
    return complex(state[ty, tx])


def particle_target(public: PublicCase, *, steps: int, gate_alpha: float) -> complex:
    free, gate0 = public_arrays(public)
    gate = mix_gate(gate0, gate_alpha)
    reached = np.zeros((public.width, public.width), dtype=np.int8)
    frontier = np.zeros((public.width, public.width), dtype=np.complex64)
    sy, sx = public.source
    ty, tx = public.target
    frontier[sy, sx] = unit_phase(public.source_phase)
    reached[sy, sx] = 1
    for _ in range(steps):
        incoming = neighbors_sum(frontier) * gate * free
        new_mask = (np.abs(incoming) > 1.0e-7) & (reached == 0)
        frontier = np.where(new_mask, incoming, 0.0)
        reached[new_mask] = 1
        if reached[ty, tx]:
            break
    return complex(frontier[ty, tx])


def evaluate_arm(cases: list[CaseBundle], arm: str, steps: int) -> dict[str, Any]:
    correct = 0
    correct_probs: list[float] = []
    nlls: list[float] = []
    margins: list[float] = []
    family_acc: dict[str, list[int]] = defaultdict(list)
    alpha_curve: list[dict[str, float]] = []
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        probs = []
        accs = []
        for bundle in cases:
            z = predict_z(bundle.public, arm, steps=steps, gate_alpha=alpha)
            p = phase_probabilities(z)
            label = bundle.private.label
            probs.append(p[label])
            accs.append(int(max(range(PHASE_CLASSES), key=lambda k: p[k]) == label))
        alpha_curve.append(
            {
                "arm": arm,
                "gate_alpha": alpha,
                "target_correct_probability": float(np.mean(probs)),
                "argmax_accuracy": float(np.mean(accs)),
            }
        )

    for bundle in cases:
        z = predict_z(bundle.public, arm, steps=steps, gate_alpha=1.0)
        probs = phase_probabilities(z)
        pred = max(range(PHASE_CLASSES), key=lambda k: probs[k])
        ok = int(pred == bundle.private.label)
        correct += ok
        correct_probs.append(probs[bundle.private.label])
        sorted_probs = sorted(probs, reverse=True)
        margins.append(sorted_probs[0] - sorted_probs[1])
        nlls.append(-math.log(max(probs[bundle.private.label], 1.0e-9)))
        family_acc[bundle.private.family].append(ok)

    curve_probs = [row["target_correct_probability"] for row in alpha_curve]
    smooth_gain = curve_probs[-1] - curve_probs[0]
    monotonic_steps = sum(1 for a, b in zip(curve_probs, curve_probs[1:]) if b + 1.0e-9 >= a)
    return {
        "arm": arm,
        "phase_argmax_accuracy": correct / max(1, len(cases)),
        "correct_phase_probability_at_target": float(np.mean(correct_probs)),
        "target_nll": float(np.mean(nlls)),
        "target_probability_margin": float(np.mean(margins)),
        "probability_smooth_gain": float(smooth_gain),
        "probability_curve_monotonicity": monotonic_steps / max(1, len(curve_probs) - 1),
        "family_accuracy": {family: float(np.mean(vals)) for family, vals in sorted(family_acc.items())},
        "alpha_curve": alpha_curve,
    }


def predict_z(public: PublicCase, arm: str, *, steps: int, gate_alpha: float) -> complex:
    if arm == "PARTICLE_FRONTIER_004_BASELINE":
        return particle_target(public, steps=steps, gate_alpha=gate_alpha)
    if arm == "WAVEFIELD_NO_INTERFERENCE_ABLATION":
        return wavefield_target(public, steps=steps, interference=False, gate_alpha=gate_alpha)
    if arm in {
        "COMPLEX_WAVEFIELD_PROPAGATION",
        "WAVEFIELD_WITH_INTERFERENCE",
        "WAVEFIELD_WITH_LOCAL_PHASE_LOSS",
        "WAVEFIELD_TARGET_ONLY_LOSS",
    }:
        return wavefield_target(public, steps=steps, interference=True, gate_alpha=gate_alpha)
    raise ValueError(f"unknown arm {arm}")


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    out = Path(job["out"])
    arm = job["arm"]
    seed = int(job["seed"])
    width = int(job["width"])
    eval_examples = int(job["eval_examples"])
    steps = int(job["steps"])
    job_id = f"{arm}__seed{seed}"
    progress_path = out / "job_progress" / f"{job_id}.jsonl"
    append_jsonl(progress_path, {"event": "job_start", "job_id": job_id, "time": time.time()})
    cases = make_cases(seed, eval_examples, width)
    metrics = evaluate_arm(cases, arm, steps)
    metrics.update({"seed": seed, "job_id": job_id, "eval_examples": eval_examples, "width": width, "steps": steps})
    append_jsonl(progress_path, {"event": "job_done", "job_id": job_id, "time": time.time(), "phase_argmax_accuracy": metrics["phase_argmax_accuracy"]})
    return metrics


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_arm[row["arm"]].append(row)
    arm_summary = []
    for arm, rows in sorted(by_arm.items()):
        arm_summary.append(
            {
                "arm": arm,
                "jobs": len(rows),
                "phase_argmax_accuracy": float(np.mean([r["phase_argmax_accuracy"] for r in rows])),
                "correct_phase_probability_at_target": float(np.mean([r["correct_phase_probability_at_target"] for r in rows])),
                "target_nll": float(np.mean([r["target_nll"] for r in rows])),
                "target_probability_margin": float(np.mean([r["target_probability_margin"] for r in rows])),
                "probability_smooth_gain": float(np.mean([r["probability_smooth_gain"] for r in rows])),
                "probability_curve_monotonicity": float(np.mean([r["probability_curve_monotonicity"] for r in rows])),
            }
        )
    lookup = {row["arm"]: row for row in arm_summary}
    verdicts: list[str] = []
    wave = lookup.get("WAVEFIELD_WITH_INTERFERENCE") or lookup.get("COMPLEX_WAVEFIELD_PROPAGATION")
    particle = lookup.get("PARTICLE_FRONTIER_004_BASELINE")
    no_interference = lookup.get("WAVEFIELD_NO_INTERFERENCE_ABLATION")
    if wave and particle and wave["correct_phase_probability_at_target"] >= particle["correct_phase_probability_at_target"] + 0.05:
        verdicts.append("WAVEFIELD_CREDIT_SMOOTHER_THAN_PARTICLE")
    if wave and no_interference and wave["phase_argmax_accuracy"] >= no_interference["phase_argmax_accuracy"] + 0.05:
        verdicts.append("INTERFERENCE_HELPFUL")
    if wave and wave["probability_smooth_gain"] > 0.10 and wave["probability_curve_monotonicity"] >= 0.80:
        verdicts.append("TARGET_PROBABILITY_GRADIENT_PRESENT")
    if not verdicts:
        verdicts.append("WAVEFIELD_SIGNAL_NOT_ESTABLISHED")
    return {"arm_summary": arm_summary, "verdicts": verdicts, "updated_at": time.time()}


def write_report(out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_005_WAVEFIELD_PROPAGATION Report",
        "",
        "## Verdicts",
        "",
    ]
    for verdict in summary.get("verdicts", []):
        lines.append(f"- {verdict}")
    lines.extend(["", "## Arm Summary", "", "| Arm | Argmax | Correct probability | NLL | Smooth gain | Monotonicity |", "|---|---:|---:|---:|---:|---:|"])
    for row in summary.get("arm_summary", []):
        lines.append(
            "| {arm} | {acc:.3f} | {prob:.3f} | {nll:.3f} | {gain:.3f} | {mono:.3f} |".format(
                arm=row["arm"],
                acc=row["phase_argmax_accuracy"],
                prob=row["correct_phase_probability_at_target"],
                nll=row["target_nll"],
                gain=row["probability_smooth_gain"],
                mono=row["probability_curve_monotonicity"],
            )
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_examples(out: Path, seed: int, width: int) -> None:
    path = out / "wavefield_cases.jsonl"
    for bundle in make_cases(seed, 32, width):
        append_jsonl(
            path,
            {
                "case_id": bundle.public.case_id,
                "family": bundle.private.family,
                "label": bundle.private.label,
                "source": bundle.public.source,
                "target": bundle.public.target,
                "path_len": bundle.private.path_len,
                "same_target_pair": bundle.private.same_target_pair,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--arms", default=",".join(ARMS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    arms = parse_csv(args.arms)
    jobs = [
        {
            "out": str(out),
            "seed": seed,
            "arm": arm,
            "width": args.width,
            "eval_examples": args.eval_examples,
            "steps": args.steps,
        }
        for seed in seeds
        for arm in arms
    ]
    write_json(out / "queue.json", {"jobs": jobs, "device": args.device, "created_at": time.time()})
    if CONTRACT.exists():
        (out / "contract_snapshot.md").write_text(CONTRACT.read_text(encoding="utf-8"), encoding="utf-8")
    write_examples(out, seeds[0], args.width)

    metrics: list[dict[str, Any]] = []
    last_heartbeat = 0.0
    with cf.ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in cf.as_completed(futures):
            row = future.result()
            metrics.append(row)
            append_jsonl(out / "metrics.jsonl", {k: v for k, v in row.items() if k != "alpha_curve"})
            for point in row["alpha_curve"]:
                append_jsonl(out / "probability_curves.jsonl", {**point, "seed": row["seed"], "job_id": row["job_id"]})
            summary = aggregate(metrics)
            summary.update({"completed_jobs": len(metrics), "total_jobs": len(jobs)})
            write_json(out / "summary.json", summary)
            write_report(out, summary)
            append_jsonl(out / "progress.jsonl", {"event": "job_done", "job_id": row["job_id"], "completed": len(metrics), "total": len(jobs), "time": time.time()})
            now = time.time()
            if now - last_heartbeat >= args.heartbeat_sec:
                append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(metrics), "total": len(jobs), "time": now})
                last_heartbeat = now
    summary = aggregate(metrics)
    summary.update({"completed_jobs": len(metrics), "total_jobs": len(jobs)})
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
