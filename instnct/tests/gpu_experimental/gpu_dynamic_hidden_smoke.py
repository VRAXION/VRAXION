"""Deterministic GPU feasibility smoke for fixed-I/O dynamic hidden capacity.

This is an isolated prototype:
  - input slots are fixed
  - output slots are fixed
  - only the hidden pool grows/shrinks
  - resize is scripted, not learned

The goal is not to replace graph.py. The goal is to answer:
  1. Does dynamic hidden capacity preserve architecture invariants?
  2. Is the run deterministic on GPU?
  3. Does scripted growth help over zero-hidden split I/O?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lib.log import live_log, log_msg
from tests.gpu_backend_ab import (
    mask_density,
    maybe_flip_strategy_gpu,
    mutate_current_backend,
    retention_from_loss,
)
from tests.gpu_int_mood_ab import make_eval_runner


CONFIGS = {
    "V32_H32": {"vocab": 32, "hidden_cap": 32, "attempts": 8000},
    "V64_H64": {"vocab": 64, "hidden_cap": 64, "attempts": 16000},
}

VARIANTS = ("static_hidden0", "dynamic_grow_schedule", "static_hidden_target")
CHECKPOINT_EVERY = 2000
GROW_EVERY = 2000
SEED = 42
DENSITY = 0.06


@dataclass
class RunResult:
    config: str
    variant: str
    seed: int
    best_acc: float
    best_score: float
    attempts_per_sec: float
    final_density: float
    final_loss_pct: int
    final_signal: int
    final_grow: int
    final_intensity: int
    mask_hash: str
    active_hidden_trace: list[int]
    checkpoint_history: list[dict]


class DynamicHiddenPrototype:
    def __init__(
        self,
        vocab: int,
        hidden_cap: int,
        active_hidden: int,
        density: float,
        seed: int,
        device: torch.device,
    ):
        self.V = vocab
        self.hidden_cap = hidden_cap
        self.active_hidden = active_hidden
        self.N_max = 2 * vocab + hidden_cap
        self.input_ids = list(range(vocab))
        self.hidden_order = list(range(vocab, vocab + hidden_cap))
        self.output_ids = list(range(vocab + hidden_cap, vocab + hidden_cap + vocab))
        self.mask = torch.zeros((self.N_max, self.N_max), dtype=torch.int8, device=device)
        self.device = device

        if active_hidden > 0 or vocab > 0:
            self._randomize_active_subgraph(density, seed)

    def _randomize_active_subgraph(self, density: float, seed: int) -> None:
        idx = self.active_indices_cpu()
        n = len(idx)
        rng = np.random.default_rng(seed)
        block = np.zeros((n, n), dtype=np.int8)
        r = rng.random((n, n))
        block[r < density / 2] = -1
        block[r > 1 - density / 2] = 1
        np.fill_diagonal(block, 0)
        idx_t = torch.tensor(idx, dtype=torch.long, device=self.device)
        block_t = torch.from_numpy(block).to(device=self.device, dtype=torch.int8)
        self.mask.index_put_((idx_t[:, None], idx_t[None, :]), block_t)

    def active_indices_cpu(self) -> list[int]:
        return self.input_ids + self.hidden_order[: self.active_hidden] + self.output_ids

    def active_indices(self) -> torch.Tensor:
        return torch.tensor(self.active_indices_cpu(), dtype=torch.long, device=self.device)

    def active_total(self) -> int:
        return 2 * self.V + self.active_hidden

    def submask(self) -> torch.Tensor:
        idx = self.active_indices()
        return self.mask.index_select(0, idx).index_select(1, idx).clone()

    def density(self) -> float:
        return mask_density(self.submask())

    def mask_hash(self) -> str:
        arr = self.mask.detach().cpu().numpy()
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    def grow_hidden(self) -> int | None:
        if self.active_hidden >= self.hidden_cap:
            return None
        new_slot = self.hidden_order[self.active_hidden]
        row_sum = int(torch.count_nonzero(self.mask[new_slot]).item())
        col_sum = int(torch.count_nonzero(self.mask[:, new_slot]).item())
        if row_sum != 0 or col_sum != 0:
            raise AssertionError("inactive hidden slot is not zero before grow")
        self.active_hidden += 1
        return new_slot

    def prune_dead_hidden(self) -> int | None:
        if self.active_hidden <= 0:
            return None
        for pos in range(self.active_hidden - 1, -1, -1):
            slot = self.hidden_order[pos]
            row_dead = int(torch.count_nonzero(self.mask[slot]).item()) == 0
            col_dead = int(torch.count_nonzero(self.mask[:, slot]).item()) == 0
            if row_dead and col_dead:
                last_pos = self.active_hidden - 1
                self.hidden_order[pos], self.hidden_order[last_pos] = (
                    self.hidden_order[last_pos],
                    self.hidden_order[pos],
                )
                self.active_hidden -= 1
                return slot
        return None

    def apply_changes(self, active_idx: torch.Tensor, submask: torch.Tensor, changes) -> None:
        cells = changes.cells if hasattr(changes, "cells") else changes
        if not cells:
            return
        rows = torch.tensor([c[0] for c in cells], dtype=torch.long, device=self.device)
        cols = torch.tensor([c[1] for c in cells], dtype=torch.long, device=self.device)
        global_rows = active_idx.index_select(0, rows)
        global_cols = active_idx.index_select(0, cols)
        self.mask[global_rows, global_cols] = submask[rows, cols]

    def assert_grow_preserves_old_active(self) -> None:
        old_idx = self.active_indices()
        before = self.mask.index_select(0, old_idx).index_select(1, old_idx).clone()
        grown = self.grow_hidden()
        if grown is None:
            return
        after = self.mask.index_select(0, old_idx).index_select(1, old_idx)
        if not torch.equal(before, after):
            raise AssertionError("grow changed previous active weights")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="V32_H32,V64_H64")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--grow-every", type=int, default=GROW_EVERY)
    ap.add_argument("--density", type=float, default=DENSITY)
    ap.add_argument("--log-name", default="gpu_dynamic_hidden_smoke")
    return ap.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def targets_for_seed(vocab: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.permutation(vocab).astype(np.int64)).to(device=device, dtype=torch.long)


def initial_active_hidden(variant: str, hidden_cap: int) -> int:
    if variant == "static_hidden0":
        return 0
    if variant == "dynamic_grow_schedule":
        return 0
    return hidden_cap


def active_hidden_target(variant: str, hidden_cap: int) -> int:
    return hidden_cap if variant != "static_hidden0" else 0


def get_eval_runner(cache: dict[int, callable], vocab: int, active_total: int, targets: torch.Tensor, device: torch.device):
    if active_total not in cache:
        out_start = active_total - vocab
        cache[active_total] = make_eval_runner(vocab, active_total, targets, out_start, device)
    return cache[active_total]


def controller_state() -> dict[str, int]:
    return {"signal": 0, "grow": 1, "intensity": 7}


def assert_logits_shape(vocab: int, active_total: int, device: torch.device) -> None:
    out_start = active_total - vocab
    charges = torch.empty((vocab, active_total), dtype=torch.float32, device=device)
    logits = charges[:, out_start : out_start + vocab]
    if logits.shape != (vocab, vocab):
        raise AssertionError(f"logits shape mismatch: {tuple(logits.shape)}")


def run_variant(
    config_name: str,
    variant: str,
    seed: int,
    checkpoint_every: int,
    grow_every: int,
    density: float,
    log_q=None,
) -> RunResult:
    cfg = CONFIGS[config_name]
    vocab = cfg["vocab"]
    hidden_cap = cfg["hidden_cap"]
    attempts = cfg["attempts"]
    device = torch.device("cuda")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    proto = DynamicHiddenPrototype(
        vocab=vocab,
        hidden_cap=hidden_cap,
        active_hidden=initial_active_hidden(variant, hidden_cap),
        density=density,
        seed=seed,
        device=device,
    )
    targets = targets_for_seed(vocab, seed, device)
    loss_pct = torch.tensor(15, device=device, dtype=torch.int16)
    controller = controller_state()
    runner_cache: dict[int, callable] = {}

    active_idx = proto.active_indices()
    score, acc = get_eval_runner(runner_cache, vocab, proto.active_total(), targets, device)(
        proto.submask(), retention_from_loss(loss_pct)
    )
    best_score = score.clone()
    best_acc = acc.clone()
    active_hidden_trace = [proto.active_hidden]
    checkpoint_history = [
        {
            "att": 0,
            "best_acc": float(best_acc.item()),
            "best_score": float(best_score.item()),
            "density": proto.density(),
            "active_hidden": proto.active_hidden,
            "aps": 0.0,
        }
    ]

    torch.cuda.synchronize()
    total_t0 = time.perf_counter()
    for att in range(1, attempts + 1):
        if variant == "dynamic_grow_schedule" and att % grow_every == 0 and proto.active_hidden < hidden_cap:
            proto.grow_hidden()
            active_hidden_trace.append(proto.active_hidden)

        active_idx = proto.active_indices()
        submask = proto.submask()
        diag_mask = ~torch.eye(proto.active_total(), dtype=torch.bool, device=device)
        prev_loss, changes = mutate_current_backend(submask, loss_pct, controller, gen, diag_mask)
        runner = get_eval_runner(runner_cache, vocab, proto.active_total(), targets, device)
        new_score, new_acc = runner(submask, retention_from_loss(loss_pct))

        if bool((new_score > score).item()):
            proto.apply_changes(active_idx, submask, changes)
            score = new_score
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            loss_pct.fill_(prev_loss)
            maybe_flip_strategy_gpu(controller, gen, device)

        if att % checkpoint_every == 0:
            torch.cuda.synchronize()
            dt = time.perf_counter() - total_t0
            aps = att / dt if dt > 0 else float("inf")
            row = {
                "att": att,
                "best_acc": float(best_acc.item()),
                "best_score": float(best_score.item()),
                "density": proto.density(),
                "active_hidden": proto.active_hidden,
                "aps": aps,
            }
            checkpoint_history.append(row)
            log_msg(
                log_q,
                f"{config_name:8s} {variant:20s} seed={seed:3d} att={att:5d} "
                f"best_acc={row['best_acc']*100:5.1f}% score={row['best_score']:.4f} "
                f"density={row['density']:.4f} hidden={row['active_hidden']:3d} aps={row['aps']:.1f}",
            )

    torch.cuda.synchronize()
    dt = time.perf_counter() - total_t0
    return RunResult(
        config=config_name,
        variant=variant,
        seed=seed,
        best_acc=float(best_acc.item()),
        best_score=float(best_score.item()),
        attempts_per_sec=attempts / dt if dt > 0 else float("inf"),
        final_density=proto.density(),
        final_loss_pct=int(loss_pct.item()),
        final_signal=int(controller["signal"]),
        final_grow=int(controller["grow"]),
        final_intensity=int(controller["intensity"]),
        mask_hash=proto.mask_hash(),
        active_hidden_trace=active_hidden_trace,
        checkpoint_history=checkpoint_history,
    )


def structural_invariants(config_name: str, density: float) -> dict:
    cfg = CONFIGS[config_name]
    device = torch.device("cuda")
    proto = DynamicHiddenPrototype(
        vocab=cfg["vocab"],
        hidden_cap=cfg["hidden_cap"],
        active_hidden=0,
        density=density,
        seed=123,
        device=device,
    )

    proto.assert_grow_preserves_old_active()
    get_eval_runner({}, proto.V, proto.active_total(), targets_for_seed(proto.V, 123, device), device)(
        proto.submask(),
        retention_from_loss(torch.tensor(15, device=device, dtype=torch.int16)),
    )
    assert_logits_shape(proto.V, proto.active_total(), device)

    while proto.active_hidden < min(4, proto.hidden_cap):
        proto.grow_hidden()
    out_before = proto.mask[proto.output_ids, :].clone(), proto.mask[:, proto.output_ids].clone()
    dead_slot = proto.hidden_order[proto.active_hidden - 1]
    proto.mask[dead_slot].zero_()
    proto.mask[:, dead_slot].zero_()
    pruned = proto.prune_dead_hidden()
    if pruned != dead_slot:
        raise AssertionError("prune_dead_hidden did not prune the globally dead hidden slot")
    out_after = proto.mask[proto.output_ids, :], proto.mask[:, proto.output_ids]
    if not torch.equal(out_before[0], out_after[0]) or not torch.equal(out_before[1], out_after[1]):
        raise AssertionError("output rows/cols changed during dead-hidden prune")
    get_eval_runner({}, proto.V, proto.active_total(), targets_for_seed(proto.V, 123, device), device)(
        proto.submask(),
        retention_from_loss(torch.tensor(15, device=device, dtype=torch.int16)),
    )
    assert_logits_shape(proto.V, proto.active_total(), device)

    return {
        "config": config_name,
        "grow_preserves_active": True,
        "prune_fixed_output": True,
        "logits_shape_ok": True,
    }


def determinism_check(config_name: str, variant: str, seed: int, checkpoint_every: int, grow_every: int, density: float, log_q=None) -> dict:
    a = run_variant(config_name, variant, seed, checkpoint_every, grow_every, density, log_q=None)
    b = run_variant(config_name, variant, seed, checkpoint_every, grow_every, density, log_q=None)
    ok = (
        a.best_acc == b.best_acc
        and a.best_score == b.best_score
        and a.mask_hash == b.mask_hash
        and a.active_hidden_trace == b.active_hidden_trace
    )
    payload = {
        "config": config_name,
        "variant": variant,
        "seed": seed,
        "deterministic": ok,
        "best_acc": a.best_acc,
        "best_score": a.best_score,
        "mask_hash": a.mask_hash,
        "active_hidden_trace": a.active_hidden_trace,
    }
    log_msg(log_q, "DETERMINISM " + json.dumps(payload, sort_keys=True))
    return payload


def summarize(results: list[RunResult], det_rows: list[dict], inv_rows: list[dict], log_q) -> None:
    log_msg(log_q, "")
    log_msg(log_q, "SUMMARY")

    for config_name in sorted({r.config for r in results}):
        rows = [r for r in results if r.config == config_name]
        by_variant = {variant: [r for r in rows if r.variant == variant] for variant in VARIANTS}
        inv_for_config = [row for row in inv_rows if row["config"] == config_name]
        det_for_config = [row for row in det_rows if row["config"] == config_name]
        inv_ok = all(
            all(value for key, value in row.items() if key != "config")
            for row in inv_for_config
        )
        det_ok = all(row["deterministic"] for row in det_for_config)
        base0 = by_variant["static_hidden0"][0]
        dyn = by_variant["dynamic_grow_schedule"][0]
        tgt = by_variant["static_hidden_target"][0]
        dynamic_beats_zero = dyn.best_acc > base0.best_acc
        close_to_target = dyn.best_acc + 0.10 >= tgt.best_acc
        utility_ok = dynamic_beats_zero and close_to_target
        verdict = (
            "mechanics_pass / utility_pass"
            if inv_ok and det_ok and utility_ok
            else "mechanics_pass / utility_fail"
            if inv_ok and det_ok
            else "mechanics_fail"
        )
        summary_row = {
            "config": config_name,
            "static_hidden0_acc": base0.best_acc,
            "dynamic_acc": dyn.best_acc,
            "static_target_acc": tgt.best_acc,
            "dynamic_beats_zero": dynamic_beats_zero,
            "within_10pp_of_target": close_to_target,
            "deterministic": det_ok,
            "invariants": inv_ok,
            "verdict": verdict,
        }
        log_msg(log_q, "SUMMARY_JSON " + json.dumps(summary_row, sort_keys=True))
        log_msg(
            log_q,
            f"{config_name:8s} zero={base0.best_acc*100:5.1f}% dyn={dyn.best_acc*100:5.1f}% "
            f"target={tgt.best_acc*100:5.1f}% deterministic={det_ok} invariants={inv_ok} verdict={verdict}",
        )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    args = parse_args()
    configs = parse_csv(args.configs)
    for config_name in configs:
        if config_name not in CONFIGS:
            raise SystemExit(f"Unknown config: {config_name}")

    with live_log(args.log_name) as (log_q, log_path):
        log_msg(
            log_q,
            f"DYNAMIC HIDDEN GPU SMOKE configs={configs} seed={args.seed} "
            f"checkpoint={args.checkpoint_every} grow_every={args.grow_every}",
        )
        log_msg(log_q, "=" * 120)

        inv_rows = []
        for config_name in configs:
            inv = structural_invariants(config_name, args.density)
            inv_rows.append(inv)
            log_msg(log_q, "INVARIANTS " + json.dumps(inv, sort_keys=True))

        results = []
        det_rows = []
        for config_name in configs:
            for variant in VARIANTS:
                row = run_variant(
                    config_name,
                    variant,
                    args.seed,
                    args.checkpoint_every,
                    args.grow_every,
                    args.density,
                    log_q=log_q,
                )
                results.append(row)
                log_msg(log_q, "RESULT_JSON " + json.dumps(row.__dict__, sort_keys=True))
                det_rows.append(
                    determinism_check(
                        config_name,
                        variant,
                        args.seed,
                        args.checkpoint_every,
                        args.grow_every,
                        args.density,
                        log_q=log_q,
                    )
                )

        summarize(results, det_rows, inv_rows, log_q)
        log_msg(log_q, f"LOG_PATH {log_path}")


if __name__ == "__main__":
    main()
