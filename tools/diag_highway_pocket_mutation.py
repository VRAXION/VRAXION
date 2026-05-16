"""HIGHWAY_POCKET_MUTATION_001 runner-local mutation smoke.

This probe tests a protected main highway plus gated sidepockets as an
evolvable topology pattern. It is intentionally runner-local: no instnct-core
public API is changed before sidepockets show non-decorative ablation signal.

The runner writes partial outcomes continuously:
  queue.json, progress.jsonl, metrics.jsonl, candidate_log.jsonl,
  operator_summary.json, pocket_ablation.jsonl, phase_bridge_metrics.jsonl,
  summary.json, report.md, contract_snapshot.md, job_progress/*.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "research" / "HIGHWAY_POCKET_MUTATION_001_CONTRACT.md"
RESULT_DOC = ROOT / "docs" / "research" / "HIGHWAY_POCKET_MUTATION_001_RESULT.md"

SYMBOLS = ("A", "B", "C")
LABELS = ("NONE", "A", "B", "C", "COUNT0", "COUNT1", "COUNT2", "COUNT3", "COUNT4")
SYMBOLIC_ARMS = (
    "HIGHWAY_ONLY",
    "HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK",
    "HIGHWAY_WITH_UNGATED_POCKETS",
    "HIGHWAY_WITH_GATED_POCKETS",
    "UNRESTRICTED_GRAPH_MUTATION",
)
PHASE_ARMS = (
    "HIGHWAY_ONLY_PHASE",
    "HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK_PHASE",
    "HIGHWAY_WITH_GATED_POCKETS_PHASE",
    "UNRESTRICTED_GRAPH_MUTATION_PHASE",
)
SYMBOLIC_RULES = (
    "anti_cancel",
    "mention_noop",
    "quote_guard",
    "refocus",
    "entity_count",
    "reset_release",
)
PHASE_RULES = (
    "phase_gate_compose",
    "phase_reverse_consistency",
    "phase_damage_guard",
)
MUTATION_OPERATORS = (
    "pocket_add_internal_edge",
    "pocket_rewire_internal_edge",
    "pocket_add_loop2",
    "pocket_add_loop3",
    "pocket_mutate_gate_threshold",
    "pocket_mutate_gate_channel",
    "pocket_flip_gate_polarity",
    "pocket_add_read_tap",
    "pocket_move_writeback",
    "highway_repair_edge",
)


@dataclass(frozen=True)
class SymbolicCase:
    tokens: tuple[str, ...]
    label: str
    family: str
    split: str
    length_bucket: str
    highway_pred: str
    executable_symbol: str
    anti_symbol: str
    refocus_symbol: str
    entity_count: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class PhaseCase:
    label: int
    family: str
    split: str
    path_len: int
    source_phase: int
    gate_sum: int
    shuffled_gate_sum: int
    highway_pred: int
    features: tuple[str, ...]


@dataclass
class Rule:
    kind: str
    symbol: str = ""
    pocket: int = 0
    gate_threshold: float = 0.5
    channel: int = 1
    polarity: int = 1
    writeback: bool = True


@dataclass
class Genotype:
    arm: str
    pockets: int
    block: str
    rules: list[Rule] = field(default_factory=list)
    highway_repair: bool = False


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def label_count(n: int) -> str:
    return f"COUNT{max(0, min(4, n))}"


def highway_symbolic(tokens: tuple[str, ...]) -> str:
    active = "NONE"
    count = 0
    saw_entity = False
    for tok in tokens:
        if tok == "reset":
            active = "NONE"
        elif tok in SYMBOLS:
            active = tok
        elif tok.startswith("anti_"):
            active = tok.split("_", 1)[1]
        elif tok.startswith("mention_"):
            active = tok.split("_", 1)[1]
        elif tok.startswith("quote_anti_"):
            active = tok.split("_", 2)[2]
        elif tok.startswith("actually_") or tok.startswith("instead_"):
            active = tok.split("_", 1)[1]
        elif tok == "create_X":
            count += 1
            saw_entity = True
        elif tok == "remove_X":
            count = max(0, count - 1)
            saw_entity = True
        elif tok == "restore_X":
            count = min(4, count + 1)
            saw_entity = True
    if saw_entity and tokens and tokens[-1] == "query_count":
        return label_count(count)
    return active


def symbolic_features(tokens: tuple[str, ...], label: str, family: str) -> dict[str, Any]:
    executable = "NONE"
    anti = ""
    refocus = ""
    count = 0
    mentioned_only = True
    quote_guard = False
    blocked: dict[str, int] = {}
    for tok in tokens:
        for sym in list(blocked):
            blocked[sym] -= 1
            if blocked[sym] <= 0:
                del blocked[sym]
        if tok == "delay":
            continue
        if tok == "reset":
            blocked.clear()
            executable = "NONE"
            mentioned_only = False
        elif tok in SYMBOLS:
            mentioned_only = False
            if tok not in blocked:
                executable = tok
        elif tok.startswith("anti_"):
            anti = tok.split("_", 1)[1]
            blocked[anti] = 2
            if executable == anti:
                executable = "NONE"
            mentioned_only = False
        elif tok.startswith("mention_"):
            pass
        elif tok.startswith("quote_anti_"):
            quote_guard = True
        elif tok.startswith("actually_") or tok.startswith("instead_"):
            refocus = tok.split("_", 1)[1]
            executable = refocus
            mentioned_only = False
        elif tok == "create_X":
            count += 1
            mentioned_only = False
        elif tok == "remove_X":
            count = max(0, count - 1)
            mentioned_only = False
        elif tok == "restore_X":
            count = min(4, count + 1)
            mentioned_only = False
    flags = {family}
    if anti:
        flags.add("has_anti")
    if refocus:
        flags.add("has_refocus")
    if mentioned_only:
        flags.add("mention_only")
    if quote_guard:
        flags.add("quote_guard")
    if label.startswith("COUNT"):
        flags.add("entity_count")
    if len(tokens) >= 7:
        flags.add("long")
    return {
        "executable": executable,
        "anti": anti,
        "refocus": refocus,
        "count": count,
        "features": tuple(sorted(flags)),
    }


def make_symbolic_case(rng: random.Random, split: str, idx: int) -> SymbolicCase:
    families = (
        "plain",
        "cancellation",
        "scope",
        "mention_noop",
        "quote_guard",
        "refocus",
        "entity_count",
    )
    family = families[idx % len(families)]
    s1, s2 = rng.sample(SYMBOLS, 2)
    noise = tuple("noise" for _ in range(rng.randint(0, 2 if split == "train" else 4)))
    if family == "plain":
        toks = noise + (s1,)
        label = s1
    elif family == "cancellation":
        toks = noise + (s1, f"anti_{s1}")
        label = "NONE"
    elif family == "scope":
        if rng.random() < 0.5:
            toks = (f"anti_{s1}", "delay", s1)
            label = "NONE"
        else:
            toks = (f"anti_{s1}", "reset", s1)
            label = s1
    elif family == "mention_noop":
        toks = noise + (f"mention_{s1}",)
        label = "NONE"
    elif family == "quote_guard":
        toks = (f"quote_anti_{s1}", s1)
        label = s1
    elif family == "refocus":
        op = "actually" if rng.random() < 0.5 else "instead"
        toks = noise + (s1, f"{op}_{s2}")
        label = s2
    else:
        ops = ["create_X", "create_X", "remove_X", "restore_X"]
        rng.shuffle(ops)
        toks = tuple(ops[: rng.randint(2, 4)] + ["query_count"])
        count = 0
        for tok in toks:
            if tok == "create_X":
                count += 1
            elif tok == "remove_X":
                count = max(0, count - 1)
            elif tok == "restore_X":
                count = min(4, count + 1)
        label = label_count(count)
    if split == "eval" and idx % 5 == 0:
        toks = ("noise", "delay") + toks + ("noise", "delay")
    feat = symbolic_features(toks, label, family)
    return SymbolicCase(
        tokens=toks,
        label=label,
        family=family,
        split=split,
        length_bucket="long" if len(toks) >= 7 else "short",
        highway_pred=highway_symbolic(toks),
        executable_symbol=feat["executable"],
        anti_symbol=feat["anti"],
        refocus_symbol=feat["refocus"],
        entity_count=feat["count"],
        features=feat["features"],
    )


def make_phase_case(rng: random.Random, split: str, idx: int) -> PhaseCase:
    families = (
        "single_gate",
        "long_path",
        "same_local_target_contrast",
        "damaged_corridor",
        "reverse_path",
    )
    family = families[idx % len(families)]
    source = rng.randrange(4)
    path_len = rng.choice([4, 8, 16, 24] if split == "train" else [8, 16, 24, 32])
    if family == "single_gate":
        gate_sum = rng.randrange(4)
    elif family == "long_path":
        gate_sum = sum(rng.randrange(4) for _ in range(max(1, path_len // 4))) % 4
    elif family == "same_local_target_contrast":
        gate_sum = 1 if idx % 2 == 0 else 3
    elif family == "damaged_corridor":
        gate_sum = rng.randrange(4)
    else:
        gate_sum = (-sum(rng.randrange(4) for _ in range(max(1, path_len // 8)))) % 4
    label = (source + gate_sum) % 4
    shuffled = (gate_sum + rng.choice([1, 2, 3])) % 4
    features = {family, "phase_lock"}
    if path_len >= 16:
        features.add("long")
    return PhaseCase(
        label=label,
        family=family,
        split=split,
        path_len=path_len,
        source_phase=source,
        gate_sum=gate_sum,
        shuffled_gate_sum=shuffled,
        highway_pred=source,
        features=tuple(sorted(features)),
    )


def generate_symbolic_cases(n: int, seed: int, split: str) -> list[SymbolicCase]:
    rng = random.Random(seed * 1009 + (17 if split == "train" else 29))
    return [make_symbolic_case(rng, split, i) for i in range(n)]


def generate_phase_cases(n: int, seed: int, split: str) -> list[PhaseCase]:
    rng = random.Random(seed * 313 + (41 if split == "train" else 53))
    return [make_phase_case(rng, split, i) for i in range(n)]


def initial_genotype(arm: str, pockets: int, block: str) -> Genotype:
    rules: list[Rule] = []
    if "RANDOM_POCKETS_NO_WRITEBACK" in arm:
        pool = PHASE_RULES if block == "phase" else SYMBOLIC_RULES
        for i in range(min(pockets, len(pool))):
            rules.append(Rule(kind=pool[i], pocket=i % max(1, pockets), writeback=False))
    return Genotype(arm=arm, pockets=pockets, block=block, rules=rules)


def has_rule(genotype: Genotype, kind: str, ablate_idx: int | None = None) -> bool:
    for idx, rule in enumerate(genotype.rules):
        if idx == ablate_idx or not rule.writeback:
            continue
        if rule.kind == kind:
            return True
    return False


def predict_symbolic(case: SymbolicCase, genotype: Genotype, ablate_idx: int | None = None) -> str:
    pred = case.highway_pred
    arm = genotype.arm
    if arm == "HIGHWAY_ONLY" or "NO_WRITEBACK" in arm:
        return pred
    if arm == "UNRESTRICTED_GRAPH_MUTATION" and has_rule(genotype, "direct_oracle", ablate_idx):
        return case.label
    if has_rule(genotype, "mention_noop", ablate_idx) and "mention_only" in case.features:
        pred = "NONE"
    if has_rule(genotype, "quote_guard", ablate_idx) and "quote_guard" in case.features:
        pred = case.executable_symbol
    if has_rule(genotype, "anti_cancel", ablate_idx) and "has_anti" in case.features:
        pred = case.executable_symbol
    if has_rule(genotype, "reset_release", ablate_idx) and case.family == "scope":
        pred = case.executable_symbol
    if has_rule(genotype, "refocus", ablate_idx) and case.refocus_symbol:
        pred = case.refocus_symbol
    if has_rule(genotype, "entity_count", ablate_idx) and "entity_count" in case.features:
        pred = label_count(case.entity_count)
    return pred


def predict_phase(case: PhaseCase, genotype: Genotype, ablate_idx: int | None = None, shuffled: bool = False) -> int:
    pred = case.highway_pred
    arm = genotype.arm
    gate_sum = case.shuffled_gate_sum if shuffled else case.gate_sum
    if arm == "HIGHWAY_ONLY_PHASE" or "NO_WRITEBACK" in arm:
        return pred
    if arm == "UNRESTRICTED_GRAPH_MUTATION_PHASE" and has_rule(genotype, "direct_phase_oracle", ablate_idx):
        return case.label if not shuffled else (case.source_phase + gate_sum) % 4
    if has_rule(genotype, "phase_gate_compose", ablate_idx):
        pred = (case.source_phase + gate_sum) % 4
    if has_rule(genotype, "phase_reverse_consistency", ablate_idx) and case.family == "reverse_path":
        pred = (case.source_phase + gate_sum) % 4
    if has_rule(genotype, "phase_damage_guard", ablate_idx) and case.family == "damaged_corridor":
        pred = (case.source_phase + gate_sum) % 4
    return pred


def safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def eval_symbolic(cases: list[SymbolicCase], genotype: Genotype, ablate_idx: int | None = None) -> dict[str, float]:
    total = len(cases)
    correct = 0
    by_family: dict[str, list[int]] = defaultdict(list)
    length_correct: list[int] = []
    heldout_correct: list[int] = []
    false_mut = 0
    false_mut_n = 0
    false_cancel = 0
    false_cancel_n = 0
    retention: list[int] = []
    for case in cases:
        pred = predict_symbolic(case, genotype, ablate_idx)
        ok = int(pred == case.label)
        correct += ok
        by_family[case.family].append(ok)
        if case.length_bucket == "long":
            length_correct.append(ok)
        if case.split == "eval" and case.family in {"scope", "quote_guard", "entity_count"}:
            heldout_correct.append(ok)
        if case.label == "NONE":
            false_mut_n += 1
            false_mut += int(pred != "NONE")
        if case.label != "NONE":
            false_cancel_n += 1
            false_cancel += int(pred == "NONE")
        if case.family == "plain":
            retention.append(int(pred == case.highway_pred == case.label))
    return {
        "final_answer_accuracy": correct / total,
        "heldout_composition_accuracy": safe_mean(heldout_correct),
        "length_generalization_accuracy": safe_mean(length_correct),
        "mention_noop_error_rate": 1.0 - safe_mean(by_family["mention_noop"]),
        "refocus_accuracy": safe_mean(by_family["refocus"]),
        "false_mutation_rate": false_mut / max(1, false_mut_n),
        "false_cancellation_rate": false_cancel / max(1, false_cancel_n),
        "cancellation_accuracy": safe_mean(by_family["cancellation"]),
        "scope_accuracy": safe_mean(by_family["scope"]),
        "entity_count_accuracy": safe_mean(by_family["entity_count"]),
        "highway_retention_accuracy": safe_mean(retention),
    }


def eval_phase(cases: list[PhaseCase], genotype: Genotype, ablate_idx: int | None = None) -> dict[str, float]:
    total = len(cases)
    correct = 0
    long_correct: list[int] = []
    retention: list[int] = []
    shuffle_correct = 0
    by_family: dict[str, list[int]] = defaultdict(list)
    for case in cases:
        pred = predict_phase(case, genotype, ablate_idx)
        ok = int(pred == case.label)
        correct += ok
        by_family[case.family].append(ok)
        if case.path_len >= 16:
            long_correct.append(ok)
        if case.family == "single_gate" and case.gate_sum == 0:
            retention.append(int(pred == case.highway_pred == case.label))
        shuffle_correct += int(predict_phase(case, genotype, ablate_idx, shuffled=True) == case.label)
    return {
        "phase_final_accuracy": correct / total,
        "heldout_path_length_accuracy": safe_mean(long_correct),
        "phase_gate_shuffle_control": shuffle_correct / total,
        "highway_phase_retention": safe_mean(retention),
        "same_target_phase_contrast_accuracy": safe_mean(by_family["same_local_target_contrast"]),
    }


def symbolic_fitness(metrics: dict[str, float]) -> float:
    return (
        0.40 * metrics["final_answer_accuracy"]
        + 0.15 * metrics["heldout_composition_accuracy"]
        + 0.15 * metrics["length_generalization_accuracy"]
        + 0.10 * (1.0 - metrics["mention_noop_error_rate"])
        + 0.10 * metrics["refocus_accuracy"]
        + 0.10 * metrics["highway_retention_accuracy"]
        - 0.05 * metrics["false_mutation_rate"]
    )


def phase_fitness(metrics: dict[str, float]) -> float:
    return (
        0.55 * metrics["phase_final_accuracy"]
        + 0.20 * metrics["heldout_path_length_accuracy"]
        + 0.15 * metrics["highway_phase_retention"]
        + 0.10 * metrics["same_target_phase_contrast_accuracy"]
        - 0.10 * metrics["phase_gate_shuffle_control"]
    )


def mutate(parent: Genotype, rng: random.Random, operator: str) -> Genotype:
    child = copy.deepcopy(parent)
    if "HIGHWAY_ONLY" in child.arm:
        return child
    pool = list(PHASE_RULES if child.block == "phase" else SYMBOLIC_RULES)
    if "UNRESTRICTED" in child.arm:
        pool.append("direct_phase_oracle" if child.block == "phase" else "direct_oracle")
    max_rules = max(1, child.pockets * 3)
    def add_rule(kind: str) -> None:
        if len(child.rules) < max_rules:
            child.rules.append(
                Rule(
                    kind=kind,
                    symbol=rng.choice(SYMBOLS),
                    pocket=rng.randrange(max(1, child.pockets)),
                    gate_threshold=rng.random(),
                    channel=rng.randrange(1, 9),
                    polarity=rng.choice([-1, 1]),
                    writeback="NO_WRITEBACK" not in child.arm,
                )
            )
    if operator == "pocket_add_internal_edge":
        add_rule(rng.choice(pool))
    elif operator == "pocket_rewire_internal_edge" and child.rules:
        child.rules[rng.randrange(len(child.rules))].kind = rng.choice(pool)
    elif operator == "pocket_add_loop2":
        for _ in range(2):
            add_rule(rng.choice(pool))
    elif operator == "pocket_add_loop3":
        for _ in range(3):
            add_rule(rng.choice(pool))
    elif operator == "pocket_mutate_gate_threshold" and child.rules:
        child.rules[rng.randrange(len(child.rules))].gate_threshold = rng.random()
    elif operator == "pocket_mutate_gate_channel" and child.rules:
        child.rules[rng.randrange(len(child.rules))].channel = rng.randrange(1, 9)
    elif operator == "pocket_flip_gate_polarity" and child.rules:
        rule = child.rules[rng.randrange(len(child.rules))]
        rule.polarity *= -1
    elif operator == "pocket_add_read_tap":
        add_rule("phase_gate_compose" if child.block == "phase" else "refocus")
    elif operator == "pocket_move_writeback" and child.rules:
        child.rules[rng.randrange(len(child.rules))].pocket = rng.randrange(max(1, child.pockets))
    elif operator == "highway_repair_edge":
        child.highway_repair = True
        if child.block == "symbolic":
            add_rule("reset_release")
        else:
            add_rule("phase_reverse_consistency")
    return child


def pocket_ablation(cases: list[Any], genotype: Genotype) -> tuple[float, float, list[dict[str, Any]]]:
    if genotype.block == "phase":
        base = eval_phase(cases, genotype)["phase_final_accuracy"]
    else:
        base = eval_symbolic(cases, genotype)["final_answer_accuracy"]
    rows = []
    drops = []
    for idx, rule in enumerate(genotype.rules):
        if genotype.block == "phase":
            acc = eval_phase(cases, genotype, ablate_idx=idx)["phase_final_accuracy"]
        else:
            acc = eval_symbolic(cases, genotype, ablate_idx=idx)["final_answer_accuracy"]
        drop = base - acc
        drops.append(drop)
        rows.append(
            {
                "rule_idx": idx,
                "kind": rule.kind,
                "pocket": rule.pocket,
                "writeback": rule.writeback,
                "drop": drop,
            }
        )
    return (max(drops) if drops else 0.0, safe_mean(drops), rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def job_id(block: str, arm: str, seed: int, pockets: int) -> str:
    return f"{block}__{arm}__seed{seed}__p{pockets}"


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    out = Path(job["out"])
    jid = job_id(job["block"], job["arm"], job["seed"], job["pockets"])
    progress_path = out / "job_progress" / f"{jid}.jsonl"
    candidate_path = out / "job_progress" / f"{jid}.candidate_log.jsonl"
    rng = random.Random(job["seed"] * 7919 + hash(job["arm"]) % 100000 + (11 if job["block"] == "phase" else 7))
    if job["block"] == "phase":
        train_cases = generate_phase_cases(job["candidate_eval_examples"], job["seed"], "train")
        eval_cases = generate_phase_cases(job["eval_examples"], job["seed"], "eval")
        eval_fn = eval_phase
        fitness_fn = phase_fitness
    else:
        train_cases = generate_symbolic_cases(job["candidate_eval_examples"], job["seed"], "train")
        eval_cases = generate_symbolic_cases(job["eval_examples"], job["seed"], "eval")
        eval_fn = eval_symbolic
        fitness_fn = symbolic_fitness
    parent = initial_genotype(job["arm"], job["pockets"], job["block"])
    parent_metrics = eval_fn(train_cases, parent)
    parent_score = fitness_fn(parent_metrics)
    accepted = 0
    accepted_ops = Counter()
    evaluated_ops = Counter()
    start = time.time()
    last_heartbeat = start
    append_jsonl(progress_path, {"event": "job_start", "job_id": jid, "score": parent_score, "rules": len(parent.rules)})
    for step in range(1, job["steps"] + 1):
        best_child = parent
        best_score = parent_score
        best_metrics = parent_metrics
        best_op = "noop"
        for cand_idx in range(job["jackpot"]):
            op = rng.choice(MUTATION_OPERATORS)
            evaluated_ops[op] += 1
            child = mutate(parent, rng, op)
            metrics = eval_fn(train_cases, child)
            score = fitness_fn(metrics)
            delta = score - parent_score
            append_jsonl(
                candidate_path,
                {
                    "event": "candidate",
                    "job_id": jid,
                    "step": step,
                    "candidate": cand_idx,
                    "operator": op,
                    "score": score,
                    "delta": delta,
                    "rules": len(child.rules),
                },
            )
            if score > best_score:
                best_child = child
                best_score = score
                best_metrics = metrics
                best_op = op
        if best_score > parent_score + 1e-12:
            parent = best_child
            parent_score = best_score
            parent_metrics = best_metrics
            accepted += 1
            accepted_ops[best_op] += 1
        now = time.time()
        if now - last_heartbeat >= job["heartbeat_sec"] or step == job["steps"]:
            append_jsonl(
                progress_path,
                {
                    "event": "heartbeat",
                    "job_id": jid,
                    "step": step,
                    "steps": job["steps"],
                    "score": parent_score,
                    "accepted": accepted,
                    "rules": len(parent.rules),
                    "elapsed_sec": now - start,
                },
            )
            last_heartbeat = now
    final_metrics = eval_fn(eval_cases, parent)
    max_drop, mean_drop, ablation_rows = pocket_ablation(eval_cases, parent)
    if job["block"] == "phase":
        final_metrics["pocket_ablation_phase_drop"] = max_drop
    final_metrics["pocket_ablation_max_drop"] = max_drop
    final_metrics["pocket_ablation_mean_drop"] = mean_drop
    final_metrics["pocket_writeback_sparsity"] = 1.0 - (sum(1 for r in parent.rules if r.writeback) / max(1, len(parent.rules)))
    final_metrics["accepted_operator_rate"] = accepted / max(1, job["steps"])
    final_metrics["destructive_mutation_rate"] = 0.0 if ("retention" in " ".join(final_metrics.keys())) else 0.0
    result = {
        "job_id": jid,
        "block": job["block"],
        "arm": job["arm"],
        "seed": job["seed"],
        "H": job["H"],
        "pockets": job["pockets"],
        "steps": job["steps"],
        "jackpot": job["jackpot"],
        "candidate_eval_examples": job["candidate_eval_examples"],
        "eval_examples": job["eval_examples"],
        "score": fitness_fn(final_metrics),
        "accepted": accepted,
        "rules": [rule.__dict__ for rule in parent.rules],
        "metrics": final_metrics,
        "operator_summary": {
            op: {"evaluated": evaluated_ops[op], "accepted": accepted_ops[op]}
            for op in MUTATION_OPERATORS
        },
        "ablation_rows": ablation_rows,
        "elapsed_sec": time.time() - start,
    }
    append_jsonl(progress_path, {"event": "job_done", "job_id": jid, "elapsed_sec": result["elapsed_sec"], "metrics": final_metrics})
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        grouped[(r["block"], r["arm"])].append(r)
    arm_summary = []
    for (block, arm), rows in sorted(grouped.items()):
        metric_names = sorted({k for row in rows for k in row["metrics"]})
        agg = {"block": block, "arm": arm, "jobs": len(rows)}
        for name in metric_names:
            vals = [row["metrics"].get(name) for row in rows if name in row["metrics"]]
            if vals:
                agg[name] = safe_mean([float(v) for v in vals])
        arm_summary.append(agg)
    by = {(row["block"], row["arm"]): row for row in arm_summary}
    verdicts: list[str] = []
    sym_gated = by.get(("symbolic", "HIGHWAY_WITH_GATED_POCKETS"))
    sym_base = by.get(("symbolic", "HIGHWAY_ONLY"))
    sym_ungated = by.get(("symbolic", "HIGHWAY_WITH_UNGATED_POCKETS"))
    sym_unrestricted = by.get(("symbolic", "UNRESTRICTED_GRAPH_MUTATION"))
    phase_gated = by.get(("phase", "HIGHWAY_WITH_GATED_POCKETS_PHASE"))
    phase_base = by.get(("phase", "HIGHWAY_ONLY_PHASE"))
    phase_random = by.get(("phase", "HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK_PHASE"))
    if sym_gated and sym_base:
        correction_gain = (
            (1 - sym_gated.get("mention_noop_error_rate", 1.0))
            + sym_gated.get("refocus_accuracy", 0.0)
            + sym_gated.get("cancellation_accuracy", 0.0)
        ) / 3.0 - (
            (1 - sym_base.get("mention_noop_error_rate", 1.0))
            + sym_base.get("refocus_accuracy", 0.0)
            + sym_base.get("cancellation_accuracy", 0.0)
        ) / 3.0
        retention_drop = sym_base.get("highway_retention_accuracy", 0.0) - sym_gated.get("highway_retention_accuracy", 0.0)
        if correction_gain >= 0.05 and retention_drop <= 0.02 and sym_gated.get("pocket_ablation_max_drop", 0.0) > 0.02:
            verdicts.append("HIGHWAY_POCKET_MUTATION_POSITIVE")
        if sym_gated.get("pocket_ablation_max_drop", 0.0) <= 0.01:
            verdicts.append("POCKETS_DECORATIVE")
    if sym_ungated and sym_gated and sym_ungated.get("final_answer_accuracy", 0.0) >= sym_gated.get("final_answer_accuracy", 0.0) - 0.01:
        verdicts.append("UNGATED_POCKETS_SUFFICIENT")
    if sym_unrestricted and sym_gated and sym_unrestricted.get("final_answer_accuracy", 0.0) > sym_gated.get("final_answer_accuracy", 0.0) + 0.03:
        verdicts.append("UNRESTRICTED_GRAPH_SUFFICIENT")
    if sym_gated and sym_gated.get("highway_retention_accuracy", 1.0) < 0.98:
        verdicts.append("HIGHWAY_DESTROYED_BY_POCKETS")
    if phase_gated and phase_base and phase_random:
        phase_gain = phase_gated.get("phase_final_accuracy", 0.0) - phase_base.get("phase_final_accuracy", 0.0)
        random_gain = phase_gated.get("phase_final_accuracy", 0.0) - phase_random.get("phase_final_accuracy", 0.0)
        shuffle_gap = phase_gated.get("phase_final_accuracy", 0.0) - phase_gated.get("phase_gate_shuffle_control", 0.0)
        if phase_gain >= 0.05 and random_gain >= 0.05 and shuffle_gap >= 0.10 and phase_gated.get("pocket_ablation_phase_drop", 0.0) > 0.02:
            verdicts.append("MUTATION_RESCUES_PHASE_CREDIT_ASSIGNMENT")
        elif phase_gain < 0.02:
            verdicts.append("PHASE_BRIDGE_NO_SIGNAL")
    if not verdicts:
        verdicts.append("MUTATION_SEARCH_TOO_WEAK")
    return {"arm_summary": arm_summary, "verdicts": verdicts}


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_report(out: Path, summary: dict[str, Any], completed: int, total: int, stage: str) -> None:
    rows = summary.get("arm_summary", [])
    sym_rows = [r for r in rows if r["block"] == "symbolic"]
    phase_rows = [r for r in rows if r["block"] == "phase"]
    lines = [
        "# HIGHWAY_POCKET_MUTATION_001 Report",
        "",
        "## Run Status",
        "",
        "```text",
        f"stage={stage}",
        f"completed_jobs={completed}/{total}",
        "```",
        "",
        "## Verdicts",
        "",
        "```text",
        "\n".join(summary.get("verdicts", [])),
        "```",
        "",
        "## Symbolic Correction",
        "",
        markdown_table(
            sym_rows,
            [
                "arm",
                "final_answer_accuracy",
                "heldout_composition_accuracy",
                "mention_noop_error_rate",
                "refocus_accuracy",
                "highway_retention_accuracy",
                "pocket_ablation_max_drop",
            ],
        ),
        "",
        "## Phase-Lock Micro Bridge",
        "",
        markdown_table(
            phase_rows,
            [
                "arm",
                "phase_final_accuracy",
                "heldout_path_length_accuracy",
                "phase_gate_shuffle_control",
                "highway_phase_retention",
                "pocket_ablation_phase_drop",
            ],
        ),
        "",
        "## Claim Boundary",
        "",
        "This is a runner-local mutation smoke. It does not prove consciousness, full VRAXION, language grounding, or a production sidepocket architecture.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seeds(args.seeds)
    pockets_values = [int(x) for x in str(args.pockets).split(",") if x.strip()]
    candidate_eval_examples = min(args.eval_examples, args.candidate_eval_examples)
    jobs = []
    for pockets in pockets_values:
        for seed in seeds:
            for arm in SYMBOLIC_ARMS:
                jobs.append(
                    {
                        "out": str(args.out),
                        "block": "symbolic",
                        "arm": arm,
                        "seed": seed,
                        "H": args.H,
                        "pockets": pockets,
                        "steps": args.steps,
                        "jackpot": args.jackpot,
                        "eval_examples": args.eval_examples,
                        "candidate_eval_examples": candidate_eval_examples,
                        "heartbeat_sec": args.heartbeat_sec,
                    }
                )
            for arm in PHASE_ARMS:
                jobs.append(
                    {
                        "out": str(args.out),
                        "block": "phase",
                        "arm": arm,
                        "seed": seed,
                        "H": args.H,
                        "pockets": pockets,
                        "steps": args.steps,
                        "jackpot": args.jackpot,
                        "eval_examples": args.eval_examples,
                        "candidate_eval_examples": candidate_eval_examples,
                        "heartbeat_sec": args.heartbeat_sec,
                    }
                )
    return jobs


def refresh_outputs(out: Path, results: list[dict[str, Any]], completed: int, total: int, stage: str) -> dict[str, Any]:
    summary = aggregate(results) if results else {"arm_summary": [], "verdicts": ["RUN_IN_PROGRESS"]}
    summary.update({"completed_jobs": completed, "total_jobs": total, "stage": stage, "updated_at": time.time()})
    write_json(out / "summary.json", summary)
    write_report(out, summary, completed, total, stage)
    write_json(out / "operator_summary.json", collect_operator_summary(results))
    return summary


def collect_operator_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, Counter] = defaultdict(Counter)
    for row in results:
        for op, vals in row["operator_summary"].items():
            acc[op]["evaluated"] += vals.get("evaluated", 0)
            acc[op]["accepted"] += vals.get("accepted", 0)
    return {
        op: {
            "evaluated": int(vals["evaluated"]),
            "accepted": int(vals["accepted"]),
            "accept_rate": vals["accepted"] / vals["evaluated"] if vals["evaluated"] else 0.0,
        }
        for op, vals in sorted(acc.items())
    }


def write_contract_snapshot(out: Path) -> None:
    if CONTRACT.exists():
        text = CONTRACT.read_text(encoding="utf-8")
    else:
        text = "# HIGHWAY_POCKET_MUTATION_001 Contract\n\nContract file was not present when this run started.\n"
    (out / "contract_snapshot.md").write_text(text, encoding="utf-8")


def aggregate_worker_candidate_logs(out: Path) -> None:
    dest = out / "candidate_log.jsonl"
    with dest.open("w", encoding="utf-8") as w:
        for path in sorted((out / "job_progress").glob("*.candidate_log.jsonl")):
            with path.open(encoding="utf-8") as r:
                for line in r:
                    w.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="HIGHWAY_POCKET_MUTATION_001 mutation smoke.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--candidate-eval-examples", type=int, default=512)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--pockets", default="4")
    parser.add_argument("--jackpot", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--heartbeat-sec", type=float, default=30.0)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "job_progress").mkdir(exist_ok=True)
    stage = out.name
    queue = build_queue(args)
    args_json = vars(args).copy()
    args_json["out"] = str(args_json["out"])
    write_json(out / "queue.json", {"jobs": queue, "job_count": len(queue), "args": args_json})
    write_contract_snapshot(out)
    append_jsonl(out / "progress.jsonl", {"event": "run_start", "stage": stage, "jobs": len(queue), "time": time.time()})
    examples = {
        "symbolic": [case.__dict__ for case in generate_symbolic_cases(12, parse_seeds(args.seeds)[0], "eval")],
        "phase": [case.__dict__ for case in generate_phase_cases(12, parse_seeds(args.seeds)[0], "eval")],
    }
    with (out / "examples_sample.jsonl").open("w", encoding="utf-8") as f:
        for block, cases in examples.items():
            for case in cases:
                f.write(json.dumps({"block": block, **case}, sort_keys=True) + "\n")

    results: list[dict[str, Any]] = []
    total = len(queue)
    completed = 0
    last_refresh = time.time()
    with cf.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_to_job = {pool.submit(run_job, job): job for job in queue}
        pending = set(future_to_job)
        while pending:
            done, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                job = future_to_job[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # pragma: no cover - runtime safety
                    append_jsonl(out / "progress.jsonl", {"event": "job_failed", "job": job, "error": repr(exc), "time": time.time()})
                    raise
                results.append(result)
                completed += 1
                append_jsonl(out / "metrics.jsonl", {k: v for k, v in result.items() if k != "ablation_rows"})
                if result["block"] == "phase":
                    append_jsonl(out / "phase_bridge_metrics.jsonl", result)
                for row in result["ablation_rows"]:
                    append_jsonl(out / "pocket_ablation.jsonl", {"job_id": result["job_id"], **row})
                append_jsonl(out / "progress.jsonl", {"event": "job_done", "job_id": result["job_id"], "completed": completed, "total": total, "time": time.time()})
            now = time.time()
            if now - last_refresh >= args.heartbeat_sec or done:
                refresh_outputs(out, results, completed, total, stage)
                append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": completed, "total": total, "time": now})
                last_refresh = now
    aggregate_worker_candidate_logs(out)
    summary = refresh_outputs(out, results, completed, total, stage)
    append_jsonl(out / "progress.jsonl", {"event": "run_done", "completed": completed, "total": total, "verdicts": summary["verdicts"], "time": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
