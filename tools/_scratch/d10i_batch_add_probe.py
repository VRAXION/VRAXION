#!/usr/bin/env python3
"""D10i batch-add / edge-graft microprobe.

Scratch/prototype only. Uses D10g's Torch evaluator and D10h's checkpoint
helpers to test whether small edge batches can safely improve beta.8.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import d10g_gpu_eval_probe as gpu_eval
import d10h_dense_crystallize_probe as d10h


@dataclass
class Proposal:
    arm: str
    policy: str
    batch_size: int
    proposal_idx: int
    edges: list[tuple[int, int]]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def candidate_class(delta: d10h.EvalResult, args) -> str:
    safety = (
        delta.accuracy >= args.min_accuracy_delta
        and abs(delta.echo) <= args.max_abs_echo_delta
        and delta.unigram >= args.min_unigram_delta
    )
    if safety and delta.smooth >= args.strong_smooth_delta and d10h.mo_score(delta) >= args.strong_mo_delta:
        return "ADD_STRONG"
    if safety and delta.smooth > 0.0 and d10h.mo_score(delta) > 0.0:
        return "ADD_WEAK"
    near = (
        delta.smooth > 0.0
        and abs(delta.echo) <= args.max_abs_echo_delta
        and delta.accuracy >= args.min_accuracy_delta - 0.001
        and delta.unigram >= args.min_unigram_delta - 0.001
    )
    if near:
        return "ADD_NEEDS_THRESHOLD_POLISH"
    return "ADD_REJECT"


def add_edges(net: gpu_eval.NetworkArrays, edges: list[tuple[int, int]]) -> gpu_eval.NetworkArrays:
    if not edges:
        return d10h.clone_network(net)
    return gpu_eval.NetworkArrays(
        h=net.h,
        sources=np.concatenate([net.sources, np.asarray([s for s, _ in edges], dtype=np.int64)]),
        targets=np.concatenate([net.targets, np.asarray([t for _, t in edges], dtype=np.int64)]),
        threshold=net.threshold.copy(),
        channel=net.channel.copy(),
        polarity=net.polarity.copy(),
    )


def existing_neighbors(net: gpu_eval.NetworkArrays) -> dict[int, set[int]]:
    h = int(net.h)
    out = {idx: set() for idx in range(h)}
    for s, t in zip(net.sources.astype(int), net.targets.astype(int)):
        out[int(s)].add(int(t))
        out[int(t)].add(int(s))
    return out


def sample_missing_edge(
    h: int,
    existing: set[tuple[int, int]],
    rng: random.Random,
    policy: str,
    base_edges: list[tuple[int, int]],
    neighbors: dict[int, set[int]],
) -> tuple[int, int] | None:
    for _ in range(2000):
        if policy == "global_random":
            s = rng.randrange(h)
            t = rng.randrange(h)
        elif policy == "local_existing":
            s0, t0 = rng.choice(base_edges)
            if rng.random() < 0.5:
                s = s0
                pool = list(neighbors.get(t0, ())) or [rng.randrange(h)]
                t = rng.choice(pool)
            else:
                pool = list(neighbors.get(s0, ())) or [rng.randrange(h)]
                s = rng.choice(pool)
                t = t0
        elif policy == "motif_closure":
            s0, t0 = rng.choice(base_edges)
            mode = rng.randrange(3)
            if mode == 0:
                s, t = t0, s0
            elif mode == 1:
                mid = t0
                outs = list(neighbors.get(mid, ()))
                s, t = s0, rng.choice(outs) if outs else rng.randrange(h)
            else:
                mid = s0
                ins = list(neighbors.get(mid, ()))
                s, t = rng.choice(ins) if ins else rng.randrange(h), t0
        else:
            raise ValueError(f"unknown policy: {policy}")
        if s == t:
            continue
        key = (int(s), int(t))
        if key not in existing:
            return key
    return None


def generate_proposal(
    current: gpu_eval.NetworkArrays,
    reference_edges: list[tuple[int, int]],
    neighbors: dict[int, set[int]],
    policy: str,
    batch_size: int,
    rng: random.Random,
    arm: str,
    proposal_idx: int,
) -> Proposal:
    existing = d10h.edge_keys(current)
    edges: list[tuple[int, int]] = []
    for _ in range(batch_size):
        edge = sample_missing_edge(current.h, existing, rng, policy, reference_edges, neighbors)
        if edge is None:
            break
        existing.add(edge)
        edges.append(edge)
    return Proposal(arm=arm, policy=policy, batch_size=batch_size, proposal_idx=proposal_idx, edges=edges)


def load_inputs(args):
    target = gpu_eval.load_checkpoint(Path(args.target))
    table = gpu_eval.load_vcbp(Path(args.packed))
    pair_ids, hot_to_idx, n_classes = gpu_eval.build_corpus_pairs(Path(args.corpus), table, gpu_eval.MAX_CLASSES)
    bigram, unigram = gpu_eval.build_bigram_unigram(pair_ids, hot_to_idx, n_classes)
    if target.projection.output_classes != n_classes:
        raise ValueError(f"target projection classes {target.projection.output_classes} != corpus classes {n_classes}")
    return target, table, pair_ids, hot_to_idx, bigram, unigram


def eval_checkpoints(checkpoints, table, pair_ids, hot_to_idx, bigram, unigram, args, device):
    return d10h.evaluate_batch(
        checkpoints,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        args.eval_len,
        parse_int_csv(args.eval_seeds),
        device,
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(proposal: Proposal, score: d10h.EvalResult, delta: d10h.EvalResult, klass: str, accepted: bool) -> dict:
    return {
        "arm": proposal.arm,
        "policy": proposal.policy,
        "batch_size": proposal.batch_size,
        "proposal_idx": proposal.proposal_idx,
        "added_edges": len(proposal.edges),
        "class": klass,
        "accepted": accepted,
        "smooth_score": score.smooth,
        "accuracy_score": score.accuracy,
        "echo_score": score.echo,
        "unigram_score": score.unigram,
        "smooth_delta": delta.smooth,
        "accuracy_delta": delta.accuracy,
        "echo_delta": delta.echo,
        "unigram_delta": delta.unigram,
        "mo_score": d10h.mo_score(delta),
        "edge_list": ";".join(f"{s}->{t}" for s, t in proposal.edges),
    }


def rank_key(row: dict) -> tuple[int, float, float]:
    class_rank = {
        "ADD_STRONG": 4,
        "ADD_WEAK": 3,
        "ADD_NEEDS_THRESHOLD_POLISH": 2,
        "ADD_REJECT": 1,
    }.get(row["class"], 0)
    return (class_rank, float(row["mo_score"]), float(row["smooth_delta"]))


def choose_verdict(candidate_rows: list[dict], confirmed: bool = False) -> str:
    classes = [row["class"] for row in candidate_rows]
    if "ADD_STRONG" in classes:
        return "D10I_ADD_SIGNAL_FOUND" if confirmed else "D10I_ADD_SIGNAL_SCOUT"
    if "ADD_WEAK" in classes:
        return "D10I_WEAK_ADD_SIGNAL"
    if "ADD_NEEDS_THRESHOLD_POLISH" in classes:
        return "D10I_ADD_NEEDS_THRESHOLD_POLISH"
    positives = [row for row in candidate_rows if float(row["mo_score"]) > 0.0]
    if positives:
        return "D10I_ADD_TOO_CLIFFY"
    return "D10I_ADD_ONLY_NO_SIGNAL"


def run_probe(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    target, table, pair_ids, hot_to_idx, bigram, unigram = load_inputs(args)
    reference_edges = list(d10h.edge_keys(target.network))
    neighbors = existing_neighbors(target.network)
    reference_scores, _ = eval_checkpoints([target], table, pair_ids, hot_to_idx, bigram, unigram, args, device)
    reference_score = reference_scores[0]
    no_op_scores, _ = eval_checkpoints(
        [target, d10h.clone_checkpoint(target, d10h.clone_network(target.network), "noop_copy")],
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        args,
        device,
    )
    no_op_delta = d10h.delta(no_op_scores[1], no_op_scores[0])
    if not d10h.nearly_zero_delta(no_op_delta, eps=1e-8):
        raise RuntimeError(f"no-op control failed: {no_op_delta}")

    path_rows: list[dict] = []
    candidate_rows: list[dict] = []
    arm_rows: list[dict] = []
    exported = []
    started = time.perf_counter()
    policies = parse_csv(args.policies)
    batch_sizes = parse_int_csv(args.batch_sizes)
    for policy in policies:
        for batch_size in batch_sizes:
            arm = f"{policy}_b{batch_size}"
            rng = random.Random(args.seed + batch_size * 1009 + sum(ord(c) for c in policy))
            current_net = d10h.clone_network(target.network)
            current_score = reference_score
            accepted_count = 0
            best_arm_row: dict | None = None
            proposal_idx = 0
            while proposal_idx < args.proposals_per_arm:
                batch_props = []
                batch_ckpts = [target]
                for _ in range(min(args.eval_batch_size, args.proposals_per_arm - proposal_idx)):
                    proposal = generate_proposal(
                        current_net,
                        reference_edges,
                        neighbors,
                        policy,
                        batch_size,
                        rng,
                        arm,
                        proposal_idx,
                    )
                    proposal_idx += 1
                    if not proposal.edges:
                        continue
                    batch_props.append(proposal)
                    proposal_net = add_edges(current_net, proposal.edges)
                    batch_ckpts.append(d10h.clone_checkpoint(target, proposal_net, f"{arm}_{proposal.proposal_idx}"))
                if not batch_props:
                    break
                scores, elapsed = eval_checkpoints(batch_ckpts, table, pair_ids, hot_to_idx, bigram, unigram, args, device)
                evaluated = []
                for idx, proposal in enumerate(batch_props, start=1):
                    d = d10h.delta(scores[idx], no_op_scores[0])
                    klass = candidate_class(d, args)
                    evaluated.append((proposal, scores[idx], d, klass))
                evaluated.sort(key=lambda item: (
                    {"ADD_STRONG": 4, "ADD_WEAK": 3, "ADD_NEEDS_THRESHOLD_POLISH": 2, "ADD_REJECT": 1}[item[3]],
                    d10h.mo_score(item[2]),
                    item[2].smooth,
                ), reverse=True)
                accepted = False
                for proposal, score, d, klass in evaluated:
                    is_accept = (not accepted) and klass in ("ADD_STRONG", "ADD_WEAK")
                    row = metric_row(proposal, score, d, klass, is_accept)
                    row["elapsed_s"] = elapsed
                    path_rows.append(row)
                    if best_arm_row is None or rank_key(row) > rank_key(best_arm_row):
                        best_arm_row = row
                    if klass != "ADD_REJECT":
                        candidate_rows.append(row.copy())
                    if is_accept:
                        current_net = add_edges(current_net, proposal.edges)
                        current_score = score
                        accepted_count += 1
                        accepted = True
                if accepted_count >= args.max_accepts_per_arm:
                    break
            arm_rows.append({
                "arm": arm,
                "policy": policy,
                "batch_size": batch_size,
                "proposals": proposal_idx,
                "accepted_count": accepted_count,
                "best_class": best_arm_row["class"] if best_arm_row else "NONE",
                "best_smooth_delta": best_arm_row["smooth_delta"] if best_arm_row else 0.0,
                "best_accuracy_delta": best_arm_row["accuracy_delta"] if best_arm_row else 0.0,
                "best_echo_delta": best_arm_row["echo_delta"] if best_arm_row else 0.0,
                "best_unigram_delta": best_arm_row["unigram_delta"] if best_arm_row else 0.0,
                "best_mo_score": best_arm_row["mo_score"] if best_arm_row else 0.0,
            })
            print(
                f"D10i arm={arm} proposals={proposal_idx} accepted={accepted_count} "
                f"best={arm_rows[-1]['best_class']} smooth={float(arm_rows[-1]['best_smooth_delta']):+.6f} "
                f"mo={float(arm_rows[-1]['best_mo_score']):+.6f}",
                flush=True,
            )

    candidate_rows.sort(key=rank_key, reverse=True)
    for idx, row in enumerate(candidate_rows[: args.export_top], start=1):
        edges = []
        for item in row["edge_list"].split(";"):
            if not item:
                continue
            s, t = item.split("->")
            edges.append((int(s), int(t)))
        ckpt = d10h.clone_checkpoint(target, add_edges(target.network, edges), f"d10i_top_{idx}")
        ckpt_path = out / "candidates" / f"top_{idx:02d}.ckpt"
        d10h.write_checkpoint(ckpt_path, ckpt, f"d10i_top_{idx}")
        row["rank"] = idx
        row["checkpoint"] = str(ckpt_path)
        exported.append(row)
    for idx, row in enumerate(candidate_rows[args.export_top :], start=args.export_top + 1):
        row["rank"] = idx
        row["checkpoint"] = ""

    path_fields = [
        "arm",
        "policy",
        "batch_size",
        "proposal_idx",
        "added_edges",
        "class",
        "accepted",
        "smooth_score",
        "accuracy_score",
        "echo_score",
        "unigram_score",
        "smooth_delta",
        "accuracy_delta",
        "echo_delta",
        "unigram_delta",
        "mo_score",
        "elapsed_s",
        "edge_list",
    ]
    candidate_fields = ["rank", "checkpoint"] + path_fields
    write_csv(out / "add_paths.csv", path_rows, path_fields)
    write_csv(out / "add_candidates.csv", candidate_rows, candidate_fields)
    write_csv(out / "arm_summary.csv", arm_rows, list(arm_rows[0].keys()) if arm_rows else ["arm"])

    verdict = choose_verdict(candidate_rows, confirmed=False)
    summary = {
        "verdict": verdict,
        "device": args.device,
        "eval_len": args.eval_len,
        "eval_seeds": parse_int_csv(args.eval_seeds),
        "proposals_total": len(path_rows),
        "candidates_total": len(candidate_rows),
        "exported_total": len(exported),
        "elapsed_s": time.perf_counter() - started,
        "no_op_delta": no_op_delta.as_dict(),
        "best_candidate": candidate_rows[0] if candidate_rows else None,
        "note": "GPU/Torch scout only; CPU/Rust confirm required before promotion.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out, summary, arm_rows)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_confirm(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    target, table, pair_ids, hot_to_idx, bigram, unigram = load_inputs(args)
    rows = list(csv.DictReader(Path(args.confirm_candidates).open(encoding="utf-8")))
    rows = [row for row in rows if row.get("checkpoint")]
    rows.sort(key=lambda row: int(row.get("rank") or 999999))
    rows = rows[: args.export_top]
    if not rows:
        raise RuntimeError(f"no checkpoint rows found in {args.confirm_candidates}")

    checkpoints = [target] + [gpu_eval.load_checkpoint(Path(row["checkpoint"])) for row in rows]
    scores, elapsed = d10h.evaluate_batch(
        checkpoints,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        args.eval_len,
        parse_int_csv(args.eval_seeds),
        device,
    )
    confirm_rows = []
    for idx, source_row in enumerate(rows, start=1):
        d = d10h.delta(scores[idx], scores[0])
        klass = candidate_class(d, args)
        confirm_rows.append({
            "rank": source_row.get("rank", idx),
            "checkpoint": source_row["checkpoint"],
            "source_class": source_row.get("class", ""),
            "confirm_class": klass,
            "policy": source_row.get("policy", ""),
            "batch_size": source_row.get("batch_size", ""),
            "added_edges": source_row.get("added_edges", ""),
            "smooth_score": scores[idx].smooth,
            "accuracy_score": scores[idx].accuracy,
            "echo_score": scores[idx].echo,
            "unigram_score": scores[idx].unigram,
            "smooth_delta": d.smooth,
            "accuracy_delta": d.accuracy,
            "echo_delta": d.echo,
            "unigram_delta": d.unigram,
            "mo_score": d10h.mo_score(d),
            "edge_list": source_row.get("edge_list", ""),
        })
    confirm_rows.sort(key=lambda row: (
        {"ADD_STRONG": 4, "ADD_WEAK": 3, "ADD_NEEDS_THRESHOLD_POLISH": 2, "ADD_REJECT": 1}.get(row["confirm_class"], 0),
        float(row["mo_score"]),
        float(row["smooth_delta"]),
    ), reverse=True)
    fields = [
        "rank",
        "checkpoint",
        "source_class",
        "confirm_class",
        "policy",
        "batch_size",
        "added_edges",
        "smooth_score",
        "accuracy_score",
        "echo_score",
        "unigram_score",
        "smooth_delta",
        "accuracy_delta",
        "echo_delta",
        "unigram_delta",
        "mo_score",
        "edge_list",
    ]
    write_csv(out / "confirm_candidates.csv", confirm_rows, fields)
    verdict = choose_verdict(
        [
            {
                "class": row["confirm_class"],
                "mo_score": row["mo_score"],
                "smooth_delta": row["smooth_delta"],
            }
            for row in confirm_rows
        ],
        confirmed=True,
    )
    summary = {
        "verdict": verdict,
        "device": args.device,
        "eval_len": args.eval_len,
        "eval_seeds": parse_int_csv(args.eval_seeds),
        "candidate_count": len(confirm_rows),
        "elapsed_s": elapsed,
        "best_candidate": confirm_rows[0] if confirm_rows else None,
        "note": "Confirm is still Torch scout; CPU/Rust confirm required before promotion.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_confirm_report(out, summary, confirm_rows)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def write_confirm_report(out: Path, summary: dict, rows: list[dict]) -> None:
    lines = [
        "# D10i Batch Add Confirm Report",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| rank | confirm class | policy | batch | smooth | accuracy | echo | unigram | MO |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | `{row['confirm_class']}` | {row['policy']} | {row['batch_size']} | "
            f"{float(row['smooth_delta']):.6f} | {float(row['accuracy_delta']):.6f} | "
            f"{float(row['echo_delta']):.6f} | {float(row['unigram_delta']):.6f} | {float(row['mo_score']):.6f} |"
        )
    (out / "D10I_BATCH_ADD_CONFIRM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(out: Path, summary: dict, arm_rows: list[dict]) -> None:
    lines = [
        "# D10i Batch Add Report",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "## Summary",
        "",
        f"- Eval length: `{summary['eval_len']}`",
        f"- Eval seeds: `{summary['eval_seeds']}`",
        f"- Proposals: `{summary['proposals_total']}`",
        f"- Candidates: `{summary['candidates_total']}`",
        f"- Runtime seconds: `{summary['elapsed_s']:.1f}`",
        "",
        "## Arm Summary",
        "",
        "| arm | proposals | accepted | best class | best smooth | best accuracy | best echo | best unigram | best MO |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in arm_rows:
        lines.append(
            f"| {row['arm']} | {row['proposals']} | {row['accepted_count']} | `{row['best_class']}` | "
            f"{float(row['best_smooth_delta']):.6f} | {float(row['best_accuracy_delta']):.6f} | "
            f"{float(row['best_echo_delta']):.6f} | {float(row['best_unigram_delta']):.6f} | {float(row['best_mo_score']):.6f} |"
        )
    lines += [
        "",
        "## Progress Map",
        "",
        "```text",
        "[1] beta.8 generalist: DONE",
        "[2] causal mechanism: DONE",
        "[3] seed replication: RUNNING D10b",
        "[4] dense fill/prune: global dense too cliffy",
        "[4.1] batch-add graft: D10i current",
        "[5] H512/H1024: blocked until repeatable H384 signal",
        "```",
    ]
    (out / "D10I_BATCH_ADD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    p.add_argument("--confirm-candidates", default="")
    p.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    p.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    p.add_argument("--out", default="output/phase_d10i_batch_add_probe_20260429/smoke")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--eval-len", type=int, default=128)
    p.add_argument("--eval-seeds", default="982001,982002")
    p.add_argument("--policies", default="global_random,local_existing")
    p.add_argument("--batch-sizes", default="1,4")
    p.add_argument("--proposals-per-arm", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--export-top", type=int, default=2)
    p.add_argument("--max-accepts-per-arm", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--min-accuracy-delta", type=float, default=-0.0005)
    p.add_argument("--max-abs-echo-delta", type=float, default=0.0015)
    p.add_argument("--min-unigram-delta", type=float, default=-0.0010)
    p.add_argument("--strong-smooth-delta", type=float, default=0.0005)
    p.add_argument("--strong-mo-delta", type=float, default=0.0005)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.confirm_candidates:
        run_confirm(args)
    else:
        run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
