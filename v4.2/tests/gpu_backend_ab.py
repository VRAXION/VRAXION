"""GPU backend A/B for two-bit controller: current guided vs edge-list backend.

This isolates mutation backend cost under identical controller semantics and the
same compiled buffered GPU eval path. The controller remains the current
two-bit decoupled policy; only mutation production changes.
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
from tests.gpu_edge_list_backend import (
    EdgeListState,
    edge_patch_add_connection,
    edge_patch_flip_connection,
    edge_patch_remove_connection,
    edge_patch_rewire_connection,
    validate_mask_matches_state,
)
from tests.gpu_int_mood_ab import CONFIGS, gpu_init, make_eval_runner


BACKEND_ALIASES = {
    "edge_list_backend": "edge_patch_backend",
}
BACKENDS = ("current_guided_backend", "edge_patch_backend")
MAX_INTENSITY = 15
MAX_PATCHES = MAX_INTENSITY * 2


@dataclass
class GuidedChanges:
    cells: list[tuple[int, int, int]]
    effective: int = 0


@dataclass
class MaskPatchBuffers:
    host_rows: np.ndarray
    host_cols: np.ndarray
    host_vals: np.ndarray
    rows: torch.Tensor
    cols: torch.Tensor
    vals: torch.Tensor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="V128_N384,V256_N768", help="Comma-separated config names")
    ap.add_argument("--attempts", type=int, default=16000)
    ap.add_argument("--seeds", default="42,77,123")
    ap.add_argument("--backends", default="current_guided_backend,edge_list_backend")
    ap.add_argument("--log-name", default="gpu_backend_ab")
    ap.add_argument("--smoke-check", action="store_true", help="Run invariant checks and a 200-attempt smoke.")
    ap.add_argument("--skip-ab", action="store_true", help="Only run smoke/invariants, skip fixed-budget A/B.")
    return ap.parse_args()


def parse_csv(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def normalize_backend_name(name: str) -> str:
    return BACKEND_ALIASES.get(name, name)


def retention_from_loss(loss_pct_t: torch.Tensor) -> torch.Tensor:
    return 1.0 - loss_pct_t.to(torch.float32) * 0.01


def mask_density(mask: torch.Tensor) -> float:
    n = mask.shape[0]
    total = n * n - n
    return float((mask != 0).sum().item()) / float(total)


def make_patch_buffers(device: torch.device, max_patches: int = MAX_PATCHES) -> MaskPatchBuffers:
    return MaskPatchBuffers(
        host_rows=np.empty(max_patches, dtype=np.int64),
        host_cols=np.empty(max_patches, dtype=np.int64),
        host_vals=np.empty(max_patches, dtype=np.int8),
        rows=torch.empty(max_patches, dtype=torch.long, device=device),
        cols=torch.empty(max_patches, dtype=torch.long, device=device),
        vals=torch.empty(max_patches, dtype=torch.int8, device=device),
    )


def materialize_candidate_mask(
    accepted_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    patches: list[tuple[int, int, int]],
    buffers: MaskPatchBuffers,
) -> None:
    candidate_mask.copy_(accepted_mask)
    # Later writes in the same attempt should win. Deduplicate before a single
    # indexed GPU write so the dense candidate matches the CPU sparse state.
    latest: dict[tuple[int, int], int] = {}
    for row, col, value in patches:
        latest[(row, col)] = value
    count = len(latest)
    if count == 0:
        return
    if count > buffers.host_rows.shape[0]:
        raise AssertionError(f"patch overflow: {count} > {buffers.host_rows.shape[0]}")
    for idx, ((row, col), value) in enumerate(latest.items()):
        buffers.host_rows[idx] = row
        buffers.host_cols[idx] = col
        buffers.host_vals[idx] = value
    rows_cpu = torch.from_numpy(buffers.host_rows[:count])
    cols_cpu = torch.from_numpy(buffers.host_cols[:count])
    vals_cpu = torch.from_numpy(buffers.host_vals[:count])
    buffers.rows[:count].copy_(rows_cpu)
    buffers.cols[:count].copy_(cols_cpu)
    buffers.vals[:count].copy_(vals_cpu)
    candidate_mask[buffers.rows[:count], buffers.cols[:count]] = buffers.vals[:count]


def current_add_connection(mask, gen, diag_mask, changes: GuidedChanges) -> bool:
    dead = torch.nonzero((mask == 0) & diag_mask, as_tuple=False)
    if dead.numel() == 0:
        return False
    idx = int(torch.randint(dead.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = dead[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    new = 1 if float(torch.rand((), generator=gen, device=mask.device).item()) > 0.5 else -1
    changes.cells.append((row, col, 0))
    changes.effective += 1
    mask[row, col] = new
    return True


def current_flip_connection(mask, gen, changes: GuidedChanges) -> bool:
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return False
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    old = int(mask[row, col].item())
    changes.cells.append((row, col, old))
    changes.effective += 1
    mask[row, col] = -old
    return True


def current_remove_connection(mask, gen, changes: GuidedChanges) -> bool:
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return False
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    old = int(mask[row, col].item())
    changes.cells.append((row, col, old))
    changes.effective += 1
    mask[row, col] = 0
    return True


def current_rewire_connection(mask, gen, diag_mask, changes: GuidedChanges) -> bool:
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return False
    probes = min(32, alive.shape[0])
    for _ in range(probes):
        idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
        rc = alive[idx]
        src = int(rc[0].item())
        dst = int(rc[1].item())
        dead_cols = torch.nonzero((mask[src] == 0) & diag_mask[src], as_tuple=False).flatten()
        if dead_cols.numel() == 0:
            continue
        new_idx = int(torch.randint(dead_cols.shape[0], (1,), generator=gen, device=mask.device).item())
        new_dst = int(dead_cols[new_idx].item())
        old = int(mask[src, dst].item())
        changes.cells.append((src, dst, old))
        changes.cells.append((src, new_dst, 0))
        changes.effective += 1
        mask[src, dst] = 0
        mask[src, new_dst] = old
        return True
    return False


def rollback_current(mask, loss_pct, prev_loss: int, changes: GuidedChanges) -> None:
    for row, col, old in reversed(changes.cells):
        mask[row, col] = old
    loss_pct.fill_(prev_loss)


def mutate_current_backend(mask, loss_pct, controller, gen, diag_mask):
    changes = GuidedChanges(cells=[])
    prev_loss = int(loss_pct.item())
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["intensity"] = max(1, min(15, controller["intensity"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        loss_pct.fill_(max(1, min(50, int(loss_pct.item()) + random.randint(-3, 3))))

    for _ in range(controller["intensity"]):
        if controller["signal"]:
            current_flip_connection(mask, gen, changes)
        else:
            if controller["grow"]:
                current_add_connection(mask, gen, diag_mask, changes)
            else:
                if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.7:
                    current_remove_connection(mask, gen, changes)
                else:
                    current_rewire_connection(mask, gen, diag_mask, changes)
    return prev_loss, changes


def mutate_edge_patch_backend(state: EdgeListState, loss_pct, controller, rng: random.Random):
    candidate_state = state.clone()
    patches: list[tuple[int, int, int]] = []
    prev_loss = int(loss_pct.item())
    if rng.random() < 0.35:
        controller["intensity"] = max(1, min(15, controller["intensity"] + rng.choice([-1, 1])))
    if rng.random() < 0.2:
        loss_pct.fill_(max(1, min(50, int(loss_pct.item()) + rng.randint(-3, 3))))

    effective = 0
    for _ in range(controller["intensity"]):
        if controller["signal"]:
            effective += 1 if edge_patch_flip_connection(candidate_state, rng, patches) else 0
        else:
            if controller["grow"]:
                effective += 1 if edge_patch_add_connection(candidate_state, rng, patches) else 0
            else:
                if rng.random() < 0.7:
                    effective += 1 if edge_patch_remove_connection(candidate_state, rng, patches) else 0
                else:
                    effective += 1 if edge_patch_rewire_connection(candidate_state, rng, patches) else 0
    return prev_loss, candidate_state, patches, effective


def maybe_flip_strategy_gpu(controller: dict, gen: torch.Generator, device: torch.device):
    if float(torch.rand((), generator=gen, device=device).item()) < 0.35:
        controller["signal"] = 1 - controller["signal"]
    if float(torch.rand((), generator=gen, device=device).item()) < 0.35:
        controller["grow"] = 1 - controller["grow"]


def maybe_flip_strategy_cpu(controller: dict, rng: random.Random):
    if rng.random() < 0.35:
        controller["signal"] = 1 - controller["signal"]
    if rng.random() < 0.35:
        controller["grow"] = 1 - controller["grow"]


def validate_edge_backend(mask: torch.Tensor, state: EdgeListState) -> None:
    state.validate()
    validate_mask_matches_state(mask, state)


def run_one(config_name: str, seed: int, attempts: int, backend_name: str, log_q=None, debug_validate: bool = False):
    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")
    backend_name = normalize_backend_name(backend_name)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    host_rng = random.Random(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    accepted_mask, _leak, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    candidate_mask = torch.empty_like(accepted_mask)
    diag_mask = ~torch.eye(neurons, dtype=torch.bool, device=device)
    eval_runner = make_eval_runner(vocab, neurons, targets, out_start, device)
    loss_pct = torch.tensor(15, device=device, dtype=torch.int16)
    controller = {"signal": 0, "grow": 1, "intensity": 7}
    accepted_state = None
    patch_buffers = None
    if backend_name == "edge_patch_backend":
        accepted_state = EdgeListState.from_mask(accepted_mask)
        patch_buffers = make_patch_buffers(device)
        validate_edge_backend(accepted_mask, accepted_state)

    score, acc = eval_runner(accepted_mask, retention_from_loss(loss_pct))
    best_score = score.clone()
    best_acc = acc.clone()
    accepted = 0
    total_effective = 0
    total_eval_ms = 0.0
    total_mask_materialize_ms = 0.0
    total_mutation_ms = 0.0

    torch.cuda.synchronize()
    total_t0 = time.perf_counter()
    for att in range(1, attempts + 1):
        attempt_t0 = time.perf_counter()
        if backend_name == "current_guided_backend":
            prev_loss, changes = mutate_current_backend(accepted_mask, loss_pct, controller, gen, diag_mask)
            effective = changes.effective
            new_score = None
            new_acc = None
            materialize_ms = 0.0
        else:
            prev_loss, candidate_state, patches, effective = mutate_edge_patch_backend(
                accepted_state,
                loss_pct,
                controller,
                host_rng,
            )
            materialize_start = torch.cuda.Event(enable_timing=True)
            materialize_end = torch.cuda.Event(enable_timing=True)
            materialize_start.record()
            materialize_candidate_mask(accepted_mask, candidate_mask, patches, patch_buffers)
            materialize_end.record()
            if debug_validate:
                validate_edge_backend(candidate_mask, candidate_state)

        eval_start = torch.cuda.Event(enable_timing=True)
        eval_end = torch.cuda.Event(enable_timing=True)
        if backend_name == "current_guided_backend":
            eval_start.record()
            new_score, new_acc = eval_runner(accepted_mask, retention_from_loss(loss_pct))
            eval_end.record()
        else:
            eval_start.record()
            new_score, new_acc = eval_runner(candidate_mask, retention_from_loss(loss_pct))
            eval_end.record()
        torch.cuda.synchronize()
        if backend_name == "current_guided_backend":
            materialize_ms = 0.0
        else:
            materialize_ms = materialize_start.elapsed_time(materialize_end)
        eval_ms = eval_start.elapsed_time(eval_end)
        total_eval_ms += eval_ms
        total_mask_materialize_ms += materialize_ms

        if bool((new_score > score).item()):
            score = new_score
            accepted += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
            if backend_name == "edge_patch_backend":
                accepted_state = candidate_state
                accepted_mask, candidate_mask = candidate_mask, accepted_mask
        else:
            if backend_name == "current_guided_backend":
                rollback_current(accepted_mask, loss_pct, prev_loss, changes)
                maybe_flip_strategy_gpu(controller, gen, accepted_mask.device)
            else:
                loss_pct.fill_(prev_loss)
                maybe_flip_strategy_cpu(controller, host_rng)

        total_effective += effective
        attempt_total_ms = (time.perf_counter() - attempt_t0) * 1000.0
        total_mutation_ms += max(0.0, attempt_total_ms - materialize_ms - eval_ms)

        if att % 4000 == 0:
            mode = "SIGNAL" if controller["signal"] else ("GROW" if controller["grow"] else "SHRINK")
            density_now = accepted_state.density() if accepted_state is not None else mask_density(accepted_mask)
            log_msg(
                log_q,
                f"{config_name:10s} {backend_name:23s} seed={seed:3d} att={att:5d} "
                f"best_acc={best_acc.item()*100:5.1f}% score={best_score.item():.4f} "
                f"density={density_now:.4f} accepted={accepted:5d} loss={int(loss_pct.item()):2d}% "
                f"{mode:6s} int={controller['intensity']:2d} eff/att={total_effective/att:.2f} "
                f"mut_ms={total_mutation_ms/att:.3f} mat_ms={total_mask_materialize_ms/att:.3f} eval_ms={total_eval_ms/att:.3f}",
            )
            if backend_name == "edge_patch_backend":
                validate_edge_backend(accepted_mask, accepted_state)

    torch.cuda.synchronize()
    total_dt = time.perf_counter() - total_t0
    mean_eval_ms = total_eval_ms / attempts if attempts else 0.0
    mean_materialize_ms = total_mask_materialize_ms / attempts if attempts else 0.0
    mean_mutation_ms = total_mutation_ms / attempts if attempts else 0.0
    final_density = accepted_state.density() if accepted_state is not None else mask_density(accepted_mask)
    result = {
        "config": config_name,
        "seed": seed,
        "backend": backend_name,
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "attempts_per_sec": attempts / total_dt if total_dt > 0 else float("inf"),
        "mutation_ms_per_attempt": mean_mutation_ms,
        "mask_materialize_ms_per_attempt": mean_materialize_ms,
        "eval_ms_per_attempt": mean_eval_ms,
        "final_density": final_density,
        "accepted": accepted,
        "mean_effective_changes_per_attempt": total_effective / attempts if attempts else 0.0,
        "final_loss_pct": int(loss_pct.item()),
        "final_signal": int(controller["signal"]),
        "final_grow": int(controller["grow"]),
        "final_intensity": int(controller["intensity"]),
    }
    return result


def print_result(log_q, row):
    mode = "SIGNAL" if row["final_signal"] else ("GROW" if row["final_grow"] else "SHRINK")
    log_msg(
        log_q,
        f"{row['config']:10s} {row['backend']:23s} seed={row['seed']:3d} "
        f"acc={row['best_acc']*100:5.1f}% score={row['best_score']:.4f} aps={row['attempts_per_sec']:.1f} "
        f"mut_ms={row['mutation_ms_per_attempt']:.3f} mat_ms={row['mask_materialize_ms_per_attempt']:.3f} "
        f"eval_ms={row['eval_ms_per_attempt']:.3f} "
        f"density={row['final_density']:.4f} accepted={row['accepted']:5d} "
        f"eff={row['mean_effective_changes_per_attempt']:.2f} "
        f"loss={row['final_loss_pct']:2d}% {mode:6s} int={row['final_intensity']:2d}",
    )


def summarize(results, configs, log_q):
    log_msg(log_q, "")
    log_msg(log_q, "SUMMARY")
    decision_rows = []
    for config in configs:
        current_rows = [r for r in results if r["config"] == config and r["backend"] == "current_guided_backend"]
        edge_rows = [r for r in results if r["config"] == config and r["backend"] == "edge_patch_backend"]
        cur_acc = np.array([r["best_acc"] for r in current_rows], dtype=np.float64)
        new_acc = np.array([r["best_acc"] for r in edge_rows], dtype=np.float64)
        cur_score = np.array([r["best_score"] for r in current_rows], dtype=np.float64)
        new_score = np.array([r["best_score"] for r in edge_rows], dtype=np.float64)
        cur_aps = np.array([r["attempts_per_sec"] for r in current_rows], dtype=np.float64)
        new_aps = np.array([r["attempts_per_sec"] for r in edge_rows], dtype=np.float64)
        cur_mut = np.array([r["mutation_ms_per_attempt"] for r in current_rows], dtype=np.float64)
        new_mut = np.array([r["mutation_ms_per_attempt"] for r in edge_rows], dtype=np.float64)
        cur_mat = np.array([r["mask_materialize_ms_per_attempt"] for r in current_rows], dtype=np.float64)
        new_mat = np.array([r["mask_materialize_ms_per_attempt"] for r in edge_rows], dtype=np.float64)
        cur_eval = np.array([r["eval_ms_per_attempt"] for r in current_rows], dtype=np.float64)
        new_eval = np.array([r["eval_ms_per_attempt"] for r in edge_rows], dtype=np.float64)
        cur_eff = np.array([r["mean_effective_changes_per_attempt"] for r in current_rows], dtype=np.float64)
        new_eff = np.array([r["mean_effective_changes_per_attempt"] for r in edge_rows], dtype=np.float64)
        row = {
            "config": config,
            "current_mean_acc": float(cur_acc.mean()),
            "edge_mean_acc": float(new_acc.mean()),
            "current_mean_score": float(cur_score.mean()),
            "edge_mean_score": float(new_score.mean()),
            "current_mean_aps": float(cur_aps.mean()),
            "edge_mean_aps": float(new_aps.mean()),
            "current_mean_mut_ms": float(cur_mut.mean()),
            "edge_mean_mut_ms": float(new_mut.mean()),
            "current_mean_mat_ms": float(cur_mat.mean()),
            "edge_mean_mat_ms": float(new_mat.mean()),
            "current_mean_eval_ms": float(cur_eval.mean()),
            "edge_mean_eval_ms": float(new_eval.mean()),
            "current_mean_eff": float(cur_eff.mean()),
            "edge_mean_eff": float(new_eff.mean()),
            "edge_p10_acc": float(np.percentile(new_acc, 10)),
            "current_p10_acc": float(np.percentile(cur_acc, 10)),
        }
        quality_ok = (
            (row["edge_mean_acc"] + 0.005 >= row["current_mean_acc"])
            and (row["edge_mean_score"] + 0.002 >= row["current_mean_score"])
        )
        speed_ok = row["edge_mean_aps"] >= 1.25 * row["current_mean_aps"]
        row["phase1_pass"] = bool(quality_ok and speed_ok)
        decision_rows.append(row)
        log_msg(
            log_q,
            f"{config:10s} cur_acc={row['current_mean_acc']*100:5.1f}% edge_acc={row['edge_mean_acc']*100:5.1f}% "
            f"cur_score={row['current_mean_score']:.4f} edge_score={row['edge_mean_score']:.4f} "
            f"cur_aps={row['current_mean_aps']:.1f} edge_aps={row['edge_mean_aps']:.1f} "
            f"mut_ms={row['current_mean_mut_ms']:.3f}->{row['edge_mean_mut_ms']:.3f} "
            f"mat_ms={row['current_mean_mat_ms']:.3f}->{row['edge_mean_mat_ms']:.3f} "
            f"eval_ms={row['current_mean_eval_ms']:.3f}->{row['edge_mean_eval_ms']:.3f} "
            f"eff={row['current_mean_eff']:.2f}->{row['edge_mean_eff']:.2f} "
            f"phase1_pass={row['phase1_pass']}",
        )
    all_pass = all(r["phase1_pass"] for r in decision_rows) if decision_rows else False
    log_msg(log_q, f"PHASE1_DECISION edge_patch_backend={'PASS' if all_pass else 'KEEP_CURRENT'}")
    log_msg(log_q, "SUMMARY_JSON " + json.dumps({"rows": decision_rows, "all_pass": all_pass}, sort_keys=True))


def run_smoke(log_q):
    log_msg(log_q, "SMOKE start config=V128_N384 seed=42 attempts=200")
    smoke_rows = []
    for backend in BACKENDS:
        row = run_one("V128_N384", 42, 200, backend, log_q=log_q, debug_validate=True)
        smoke_rows.append(row)
        print_result(log_q, row)
        log_msg(log_q, "SMOKE_RESULT " + json.dumps(row, sort_keys=True))
    return smoke_rows


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    args = parse_args()
    configs = parse_csv(args.configs)
    seeds = parse_csv_ints(args.seeds)
    backends = [normalize_backend_name(x) for x in parse_csv(args.backends)]
    for config in configs:
        if config not in CONFIGS:
            raise SystemExit(f"Unknown config: {config}")
    for backend in backends:
        if backend not in BACKENDS:
            raise SystemExit(f"Unknown backend: {backend}")

    with live_log(args.log_name) as (log_q, log_path):
        log_msg(
            log_q,
            f"GPU BACKEND A/B attempts={args.attempts} configs={configs} seeds={seeds} backends={backends}",
        )
        log_msg(log_q, "=" * 120)
        if args.smoke_check:
            run_smoke(log_q)
        if not args.skip_ab:
            results = []
            for config in configs:
                for backend in backends:
                    for seed in seeds:
                        row = run_one(config, seed, args.attempts, backend, log_q=log_q)
                        results.append(row)
                        print_result(log_q, row)
                        log_msg(log_q, "RESULT_JSON " + json.dumps(row, sort_keys=True))
            summarize(results, configs, log_q)
        log_msg(log_q, f"LOG_PATH {log_path}")


if __name__ == "__main__":
    main()
