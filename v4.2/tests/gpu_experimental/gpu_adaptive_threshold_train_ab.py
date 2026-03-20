"""Deterministic GPU add-only A/B for adaptive top-k threshold training."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.gpu_experimental.gpu_english_common import (
    DEFAULT_DATA_PATH,
    N_EVAL_SEQS,
    N_TRAIN_SEQS,
    SEQ_LEN,
    ThresholdPolicy,
    eval_sequence_batch,
    gather_seq_batch,
    load_all_data,
    make_empty_init,
    make_fixed_eval_sequences,
    make_fixed_train_report_sequences,
    make_output_path,
    mask_hash,
    parse_threshold_policies_csv,
    resolve_data_path,
    seq_batch_to_device,
)


@dataclass
class Proposal:
    row: int
    col: int
    sign: float
    offsets: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--seeds", default="42,77")
    ap.add_argument("--attempts-list", default="1024,2048")
    ap.add_argument("--policies", default="fixed:0.5,topk:2")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--n-train-seqs", type=int, default=N_TRAIN_SEQS)
    ap.add_argument("--n-eval-seqs", type=int, default=N_EVAL_SEQS)
    ap.add_argument("--eval-seed", type=int, default=9999)
    ap.add_argument("--train-report-seed", type=int, default=424242)
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=256)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--output", default="")
    return ap.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def proposals_for_seed(
    *,
    seed: int,
    attempts: int,
    neurons: int,
    data_len: int,
    seq_len: int,
    n_train_seqs: int,
) -> list[Proposal]:
    rng = random.Random(seed + 91_117)
    np_rng = np.random.RandomState(seed + 17_771)
    proposals: list[Proposal] = []
    high = data_len - seq_len
    for _ in range(attempts):
        row = rng.randrange(neurons)
        col = rng.randrange(neurons - 1)
        if col >= row:
            col += 1
        sign = 0.6 if rng.random() < 0.5 else -0.6
        offsets = np_rng.randint(0, high, size=n_train_seqs).astype(np.int64)
        proposals.append(Proposal(row=row, col=col, sign=sign, offsets=offsets))
    return proposals


def case_equal(a: dict[str, object], b: dict[str, object]) -> bool:
    return (
        abs(float(a["train_score_final"]) - float(b["train_score_final"])) <= 1e-9
        and abs(float(a["eval_acc_final"]) - float(b["eval_acc_final"])) <= 1e-9
        and abs(float(a["best_eval_acc"]) - float(b["best_eval_acc"])) <= 1e-9
        and int(a["best_eval_step"]) == int(b["best_eval_step"])
        and int(a["final_edges"]) == int(b["final_edges"])
        and int(a["promotions"]) == int(b["promotions"])
        and str(a["mask_hash"]) == str(b["mask_hash"])
    )


def summarize_budget(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    labels = sorted({str(r["policy"]) for r in rows})
    summary: dict[str, dict[str, float]] = {}
    for label in labels:
        group = [r for r in rows if str(r["policy"]) == label]
        summary[label] = {
            "train_score_final_median": float(np.median([float(r["train_score_final"]) for r in group])),
            "train_acc_final_median": float(np.median([float(r["train_acc_final"]) for r in group])),
            "eval_acc_final_median": float(np.median([float(r["eval_acc_final"]) for r in group])),
            "eval_score_final_median": float(np.median([float(r["eval_score_final"]) for r in group])),
            "best_eval_acc_median": float(np.median([float(r["best_eval_acc"]) for r in group])),
            "final_edges_median": float(np.median([int(r["final_edges"]) for r in group])),
            "wall_ms_median": float(np.median([float(r["wall_ms"]) for r in group])),
            "deterministic_all": float(all(bool(r["deterministic"]) for r in group)),
        }
    return summary


def budget_verdict(summary: dict[str, dict[str, float]], baseline_label: str) -> dict[str, dict[str, object]]:
    baseline = summary[baseline_label]
    verdict: dict[str, dict[str, object]] = {}
    for key, row in summary.items():
        if key == baseline_label:
            continue
        eval_gain = row["eval_acc_final_median"] - baseline["eval_acc_final_median"]
        edge_ok = row["final_edges_median"] <= 1.5 * baseline["final_edges_median"]
        positive = row["deterministic_all"] == 1.0 and eval_gain >= 0.01 and edge_ok
        research_only = row["deterministic_all"] == 1.0 and not positive and eval_gain >= 0.005
        verdict[key] = {
            "eval_gain": eval_gain,
            "edge_ratio_vs_baseline": (
                row["final_edges_median"] / baseline["final_edges_median"]
                if baseline["final_edges_median"]
                else None
            ),
            "positive": positive,
            "research_only": research_only,
        }
    return verdict


def overall_verdict(budgets: list[dict[str, object]]) -> dict[str, object]:
    positive_counts: dict[str, int] = {}
    research_counts: dict[str, int] = {}
    for bucket in budgets:
        for key, row in bucket["verdict"].items():
            if row["positive"]:
                positive_counts[key] = positive_counts.get(key, 0) + 1
            if row["research_only"]:
                research_counts[key] = research_counts.get(key, 0) + 1
    winner = None
    research_winner = None
    for key, count in positive_counts.items():
        if count >= 2:
            winner = key
            break
    if winner is None:
        for key, count in research_counts.items():
            if count >= 1:
                research_winner = key
                break
    return {
        "positive_budget_counts": positive_counts,
        "research_only_budget_counts": research_counts,
        "winner": winner,
        "research_only_winner": research_winner,
        "overall_positive": winner is not None,
    }


def run_once(
    *,
    seed: int,
    attempts: int,
    policy: ThresholdPolicy,
    ticks: int,
    proposals: list[Proposal],
    all_data: np.ndarray,
    eval_batch: torch.Tensor,
    train_report_batch: torch.Tensor,
    device: torch.device,
    seq_len: int,
    log_every: int,
    eps: float,
) -> dict[str, object]:
    init = make_empty_init(device=device, io_dim=256, net_seed=42)
    mask = init.mask.clone()
    alive_set: set[tuple[int, int]] = set()
    promotions = 0
    best_eval_acc = -1.0
    best_eval_step = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for step in range(1, attempts + 1):
        proposal = proposals[step - 1]
        if proposal.row == proposal.col or (proposal.row, proposal.col) in alive_set:
            pass
        else:
            train_batch_np = gather_seq_batch(all_data, proposal.offsets, seq_len)
            train_batch = seq_batch_to_device(train_batch_np, device)
            base = eval_sequence_batch(
                mask=mask,
                w_in=init.w_in,
                w_out=init.w_out,
                bp=init.bp,
                bp_norm=init.bp_norm,
                seq_batch=train_batch,
                retention=init.retention,
                threshold_policy=policy,
                ticks=ticks,
            )
            old_val = float(mask[proposal.row, proposal.col].item())
            mask[proposal.row, proposal.col] = proposal.sign
            candidate = eval_sequence_batch(
                mask=mask,
                w_in=init.w_in,
                w_out=init.w_out,
                bp=init.bp,
                bp_norm=init.bp_norm,
                seq_batch=train_batch,
                retention=init.retention,
                threshold_policy=policy,
                ticks=ticks,
            )
            if candidate["score"] > base["score"] + eps:
                alive_set.add((proposal.row, proposal.col))
                promotions += 1
            else:
                mask[proposal.row, proposal.col] = old_val

        if step % log_every == 0 or step == attempts:
            eval_metrics = eval_sequence_batch(
                mask=mask,
                w_in=init.w_in,
                w_out=init.w_out,
                bp=init.bp,
                bp_norm=init.bp_norm,
                seq_batch=eval_batch,
                retention=init.retention,
                threshold_policy=policy,
                ticks=ticks,
            )
            if eval_metrics["acc"] > best_eval_acc:
                best_eval_acc = eval_metrics["acc"]
                best_eval_step = step
            print(
                f"seed={seed} policy={policy.label()} attempts={attempts} step={step} "
                f"eval={eval_metrics['acc']*100:.2f}% edges={len(alive_set)} promotions={promotions}"
            )

    train_report = eval_sequence_batch(
        mask=mask,
        w_in=init.w_in,
        w_out=init.w_out,
        bp=init.bp,
        bp_norm=init.bp_norm,
        seq_batch=train_report_batch,
        retention=init.retention,
        threshold_policy=policy,
        ticks=ticks,
    )
    eval_final = eval_sequence_batch(
        mask=mask,
        w_in=init.w_in,
        w_out=init.w_out,
        bp=init.bp,
        bp_norm=init.bp_norm,
        seq_batch=eval_batch,
        retention=init.retention,
        threshold_policy=policy,
        ticks=ticks,
    )
    if eval_final["acc"] > best_eval_acc:
        best_eval_acc = eval_final["acc"]
        best_eval_step = attempts

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "seed": seed,
        "attempt_budget": attempts,
        "policy": policy.label(),
        "policy_mode": policy.mode,
        "fixed_value": policy.fixed_value,
        "k_active": policy.k_active,
        "ticks": ticks,
        "train_score_final": train_report["score"],
        "train_acc_final": train_report["acc"],
        "eval_acc_final": eval_final["acc"],
        "eval_score_final": eval_final["score"],
        "best_eval_acc": best_eval_acc,
        "best_eval_step": best_eval_step,
        "final_edges": len(alive_set),
        "promotions": promotions,
        "mask_hash": mask_hash(mask),
        "wall_ms": wall_ms,
    }


def run_case(
    *,
    seed: int,
    attempts: int,
    policy: ThresholdPolicy,
    ticks: int,
    proposals: list[Proposal],
    all_data: np.ndarray,
    eval_batch: torch.Tensor,
    train_report_batch: torch.Tensor,
    device: torch.device,
    seq_len: int,
    log_every: int,
    eps: float,
) -> dict[str, object]:
    first = run_once(
        seed=seed,
        attempts=attempts,
        policy=policy,
        ticks=ticks,
        proposals=proposals,
        all_data=all_data,
        eval_batch=eval_batch,
        train_report_batch=train_report_batch,
        device=device,
        seq_len=seq_len,
        log_every=log_every,
        eps=eps,
    )
    second = run_once(
        seed=seed,
        attempts=attempts,
        policy=policy,
        ticks=ticks,
        proposals=proposals,
        all_data=all_data,
        eval_batch=eval_batch,
        train_report_batch=train_report_batch,
        device=device,
        seq_len=seq_len,
        log_every=log_every,
        eps=eps,
    )
    row = dict(first)
    row["deterministic"] = case_equal(first, second)
    return row


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = resolve_data_path(args.data_path)
    all_data = load_all_data(data_path)
    policies = parse_threshold_policies_csv(args.policies)
    baseline_label = next((p.label() for p in policies if p.mode == "fixed" and p.fixed_value == 0.5), None)
    if baseline_label is None:
        raise ValueError("Policies must include baseline fixed:0.5")
    seeds = parse_csv_ints(args.seeds)
    attempts_list = parse_csv_ints(args.attempts_list)
    max_attempts = max(attempts_list)

    eval_batch_np = make_fixed_eval_sequences(all_data, seed=args.eval_seed, n_eval=args.n_eval_seqs, seq_len=args.seq_len)
    eval_batch = seq_batch_to_device(eval_batch_np, device)
    train_report_np = make_fixed_train_report_sequences(
        all_data,
        seed=args.train_report_seed,
        n_train=args.n_train_seqs,
        seq_len=args.seq_len,
    )
    train_report_batch = seq_batch_to_device(train_report_np, device)

    proposal_bank = {
        seed: proposals_for_seed(
            seed=seed,
            attempts=max_attempts,
            neurons=256 * 3,
            data_len=len(all_data),
            seq_len=args.seq_len,
            n_train_seqs=args.n_train_seqs,
        )
        for seed in seeds
    }

    results: list[dict[str, object]] = []
    budgets_summary: list[dict[str, object]] = []
    for attempts in attempts_list:
        budget_rows: list[dict[str, object]] = []
        for seed in seeds:
            proposals = proposal_bank[seed][:attempts]
            for policy in policies:
                row = run_case(
                    seed=seed,
                    attempts=attempts,
                    policy=policy,
                    ticks=args.ticks,
                    proposals=proposals,
                    all_data=all_data,
                    eval_batch=eval_batch,
                    train_report_batch=train_report_batch,
                    device=device,
                    seq_len=args.seq_len,
                    log_every=args.log_every,
                    eps=args.eps,
                )
                results.append(row)
                budget_rows.append(row)

        summary = summarize_budget(budget_rows)
        verdict = budget_verdict(summary, baseline_label)
        budgets_summary.append(
            {
                "attempt_budget": attempts,
                "summary": summary,
                "verdict": verdict,
            }
        )

    overall = overall_verdict(budgets_summary)

    output = {
        "branch_intent": "adaptive_threshold_train_ab",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "device": str(device),
        "args": {
            "data_path": str(data_path),
            "seeds": seeds,
            "attempts_list": attempts_list,
            "policies": [p.to_dict() for p in policies],
            "ticks": args.ticks,
            "seq_len": args.seq_len,
            "n_train_seqs": args.n_train_seqs,
            "n_eval_seqs": args.n_eval_seqs,
            "eval_seed": args.eval_seed,
            "train_report_seed": args.train_report_seed,
            "eps": args.eps,
        },
        "results": results,
        "budgets": budgets_summary,
        "overall_verdict": overall,
    }

    output_path = make_output_path(args.output, "gpu_adaptive_threshold_train_ab.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
