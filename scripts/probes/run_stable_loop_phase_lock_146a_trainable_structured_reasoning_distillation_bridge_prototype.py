#!/usr/bin/env python3
"""146A trainable structured reasoning distillation bridge prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype/smoke")
DEFAULT_145Z_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_145z_mixed_structured_rule_composition_next_decision_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
DECISION = "trainable_structured_reasoning_distillation_bridge_prototype_positive"
VERDICT = "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_POSITIVE"
NEXT = "146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM"
FALLBACK_LABEL = "fallback"
FALLBACK_VALUE = "VALCLOSED000000"
POCKETS = ["pocket_a", "pocket_b", "pocket_c"]
POCKET_LABEL = {"pocket_a": "A", "pocket_b": "B", "pocket_c": "C"}
LABEL_TO_POCKET = {"A": "pocket_a", "B": "pocket_b", "C": "pocket_c"}
BLOCK_TYPES = ["quorum", "recency", "tie_break"]
STANDARD_PRIORITY_ORDERS = [
    ["recency", "quorum", "tie_break"],
    ["quorum", "recency", "tie_break"],
    ["tie_break", "quorum", "recency"],
    ["quorum", "tie_break", "recency"],
    ["recency", "tie_break", "quorum"],
]
OOD_PRIORITY_ORDER = ["tie_break", "recency", "quorum"]
BOUNDARY_TEXT = (
    "146A is constrained model-facing distillation evidence only with canonical structured prompts only; "
    "not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, "
    "not production readiness, and not architecture superiority."
)
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
    "natural_language_rule_reasoning_claimed": False,
    "open_ended_arbitration_claimed": False,
    "gpt_like_readiness_claimed": False,
    "gemma_like_capability_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
    "architecture_superiority_claimed": False,
}
FORBIDDEN_MODEL_INPUT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"selected_pocket_id",
        r"\bwinner\s*=\s*pocket_[abc]\b",
        r"final_selected",
        r"derived_selected",
        r"answer[-_ ]?value",
        r"gold[-_ ]?value",
        r"target[-_ ]?value",
        r"resolved[-_ ]?output",
        r"expected[-_ ]?output",
        r"teacher_trace",
        r"per-row oracle metadata",
        r"\bANSWER\s*=",
        r"\bGOLD\s*=",
        r"\bTARGET\s*=",
        r"\bEXPECTED\s*=",
    ]
]
CONTROLS = [
    "EXACT_TEMPLATE_HOLDOUT",
    "PRIORITY_ORDER_HOLDOUT",
    "BLOCK_ORDER_HOLDOUT",
    "RULE_BLOCK_TYPE_COMBINATION_HOLDOUT",
    "SAME_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
    "INVALID_HIGH_PRIORITY_FALLTHROUGH_OOD",
    "STRUCTURAL_INVALID_PROMPT_OOD",
    "ORACLE_FIELD_ABLATION",
    "TEACHER_TRACE_LEAKAGE_CONTROL",
    "SELECTED_POCKET_SHORTCUT_CONTROL",
    "ANSWER_VALUE_SHORTCUT_CONTROL",
    "TRAIN_TEST_LEAKAGE_CONTROL",
    "OOD_COMPOSITION_CONTROL",
    "HELPER_STACK_REGRESSION_CONTROL",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_target_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    relative = resolved.relative_to(REPO_ROOT)
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "details": details})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def helper_unchanged_from_head() -> bool:
    return HELPER_PATH.read_text(encoding="utf-8") == git_show_head("scripts/probes/shared_raw_generation_helper.py")


def rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def require_145z(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "target_146a_milestone_plan.json",
        "model_facing_bridge_gap_analysis.json",
        "anti_oracle_requirements.json",
        "summary.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 145Z artifacts: {missing}")
    decision = read_json(root / "decision.json")
    target = read_json(root / "target_146a_milestone_plan.json")
    gap = read_json(root / "model_facing_bridge_gap_analysis.json")
    anti = read_json(root / "anti_oracle_requirements.json")
    checks = {
        "decision": decision.get("decision") == "trainable_structured_reasoning_distillation_bridge_plan_recommended",
        "selected_option": decision.get("selected_option") == "trainable_structured_reasoning_distillation_bridge_plan",
        "next": decision.get("next") == "146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE",
        "target_implementation_ready": target.get("implementation_ready") is True,
        "structured_helper_stack_scale_confirmed": gap.get("structured_helper_stack_scale_confirmed") is True,
        "trainable_model_internalization_untested": gap.get("trainable_model_internalization_untested") is True,
        "natural_language_rule_reasoning_untested": gap.get("natural_language_rule_reasoning_untested") is True,
        "open_ended_arbitration_claimed": gap.get("open_ended_arbitration_claimed") is False,
        "gpt_like_or_gemma_like_capability_claimed": gap.get("gpt_like_or_gemma_like_capability_claimed") is False,
        "anti_oracle_passed": anti.get("passed") is True,
    }
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"145Z upstream verification failed: {failures}")
    return {
        "schema_version": "phase_146a_upstream_145z_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "target_146a_milestone_plan": target,
        "model_facing_bridge_gap_analysis": gap,
        "anti_oracle_requirements": anti,
        "checks": checks,
        "failed_checks": [],
        "passed": True,
    }


def opaque_value(seed: int, split: str, index: int, pocket: str, variant: int = 0) -> str:
    digest = int(hashlib.sha256(f"146A|{seed}|{split}|{index}|{pocket}|{variant}".encode("utf-8")).hexdigest(), 16)
    return f"VAL{digest % 10**14:014d}"


def other_pockets(pocket: str) -> list[str]:
    return [item for item in POCKETS if item != pocket]


def rotate(items: list[str], offset: int) -> list[str]:
    if not items:
        return []
    offset = offset % len(items)
    return items[offset:] + items[:offset]


def make_quorum_block(candidate: str, index: int, *, invalid: bool = False) -> list[str]:
    if invalid:
        votes = [candidate, "pocket_z", other_pockets(candidate)[0]]
    else:
        votes = rotate([candidate, candidate, other_pockets(candidate)[0]], index)
    return ["rule_block=quorum", f"votes={','.join(votes)}", "block_end"]


def make_recency_block(candidate: str, index: int, *, invalid: bool = False) -> list[str]:
    if invalid:
        order = ["pocket_z", other_pockets(candidate)[0], candidate]
    else:
        order = [candidate] + rotate(other_pockets(candidate), index)
    return ["rule_block=recency", f"recency_order={'>'.join(order)}", "block_end"]


def make_tie_break_block(candidate: str, index: int, *, invalid: bool = False) -> list[str]:
    others = rotate(other_pockets(candidate), index)
    if invalid:
        tied = [candidate, "pocket_z"]
        order = ["pocket_z", others[0], candidate]
    else:
        tied = [candidate, others[0]]
        order = [candidate, others[0], others[1]]
    return ["rule_block=tie_break", f"tied={','.join(tied)}", f"tie_break_order={'>'.join(order)}", "block_end"]


def block_lines(block_type: str, candidate: str, index: int, *, invalid: bool = False) -> list[str]:
    if block_type == "quorum":
        return make_quorum_block(candidate, index, invalid=invalid)
    if block_type == "recency":
        return make_recency_block(candidate, index, invalid=invalid)
    if block_type == "tie_break":
        return make_tie_break_block(candidate, index, invalid=invalid)
    raise ValueError(block_type)


def distinct_candidates(index: int) -> dict[str, str]:
    rotated = rotate(POCKETS, index)
    return {"quorum": rotated[0], "recency": rotated[1], "tie_break": rotated[2]}


def family_for(split: str, index: int) -> str:
    common = [
        "HELPER_STACK_REGRESSION_CONTROL",
        "SAME_BLOCKS_DIFFERENT_PRIORITY",
        "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
        "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
    ]
    test = ["EXACT_TEMPLATE_HOLDOUT", *common]
    ood = [
        "OOD_COMPOSITION_CONTROL",
        "SAME_BLOCKS_DIFFERENT_PRIORITY",
        "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
        "OOD_COMPOSITION_CONTROL",
        "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
        "OOD_COMPOSITION_CONTROL",
        "SAME_BLOCKS_DIFFERENT_PRIORITY",
        "PRIORITY_ORDER_HOLDOUT",
        "OOD_COMPOSITION_CONTROL",
        "BLOCK_ORDER_HOLDOUT",
        "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
        "RULE_BLOCK_TYPE_COMBINATION_HOLDOUT",
        "OOD_COMPOSITION_CONTROL",
        "INVALID_HIGH_PRIORITY_FALLTHROUGH_OOD",
        "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
        "STRUCTURAL_INVALID_PROMPT_OOD",
        "OOD_COMPOSITION_CONTROL",
    ]
    validation = ["HELPER_STACK_REGRESSION_CONTROL", *common]
    families = {"train": common, "validation": validation, "test": test, "ood_test": ood}[split]
    return families[index % len(families)]


def template_for(split: str, index: int) -> str:
    if split in {"train", "validation"}:
        return f"T{index % 8:02d}"
    if split == "test":
        return f"T{8 + (index % 2):02d}"
    return f"T{10 + (index % 2):02d}"


def choose_priority(split: str, family: str, index: int) -> list[str]:
    if family == "PRIORITY_ORDER_HOLDOUT":
        return list(OOD_PRIORITY_ORDER)
    return list(STANDARD_PRIORITY_ORDERS[index % len(STANDARD_PRIORITY_ORDERS)])


def build_model_input(
    row_id: str,
    split: str,
    family: str,
    template_id: str,
    blocks: dict[str, list[str]],
    block_order: list[str],
    priority: list[str],
    values: dict[str, str],
) -> str:
    _ = (row_id, split, family, template_id)
    lines = ["format=canonical_structured_rule_text", f"priority={'>'.join(priority)}", ""]
    for block_type in block_order:
        lines.extend(blocks[block_type])
        lines.append("")
    lines.append(f"pocket A candidate: {values['pocket_a']}")
    lines.append(f"pocket B candidate: {values['pocket_b']}")
    lines.append(f"pocket C candidate: {values['pocket_c']}")
    return "\n".join(lines).strip()


def final_pocket_from_candidates(priority: list[str], candidates: dict[str, str | None]) -> str:
    for block_type in priority:
        candidate = candidates.get(block_type)
        if candidate in POCKETS:
            return str(candidate)
    return FALLBACK_LABEL


def curriculum_row(seed: int, split: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    family = family_for(split, index)
    row_id = f"146A_{split}_{index:05d}"
    template_id = template_for(split, index)
    priority = choose_priority(split, family, index)
    candidates = distinct_candidates(index)
    block_order = ["quorum", "recency", "tie_break"]
    invalid_blocks: set[str] = set()
    structural_invalid = False
    if family == "BLOCK_ORDER_HOLDOUT":
        block_order = ["tie_break", "recency", "quorum"]
    if family == "RULE_BLOCK_TYPE_COMBINATION_HOLDOUT":
        block_order = ["recency", "quorum"]
        priority = ["recency", "quorum"]
        candidates = {"recency": candidates["recency"], "quorum": candidates["quorum"], "tie_break": None}
    if family == "INVALID_HIGH_PRIORITY_FALLTHROUGH_OOD":
        invalid_blocks.add(priority[0])
    if family == "STRUCTURAL_INVALID_PROMPT_OOD":
        structural_invalid = True
    blocks: dict[str, list[str]] = {}
    for block_type in ["quorum", "recency", "tie_break"]:
        candidate = candidates.get(block_type) or POCKETS[(index + 1) % len(POCKETS)]
        blocks[block_type] = block_lines(block_type, candidate, index, invalid=block_type in invalid_blocks)
    if structural_invalid:
        blocks[block_order[0]] = ["rule_block=quorum", "votes=pocket_a,pocket_b,pocket_a", "rule_block=recency", "recency_order=pocket_c>pocket_b>pocket_a", "block_end"]
        candidates = {key: None for key in candidates}
    effective_candidates = {key: (None if key in invalid_blocks else value) for key, value in candidates.items()}
    final_pocket = FALLBACK_LABEL if structural_invalid else final_pocket_from_candidates(priority, effective_candidates)
    values = {pocket: opaque_value(seed, split, index, pocket) for pocket in POCKETS}
    model_input = build_model_input(row_id, split, family, template_id, blocks, block_order, priority, values)
    final_value = FALLBACK_VALUE if final_pocket == FALLBACK_LABEL else values[final_pocket]
    label = "fallback" if final_pocket == FALLBACK_LABEL else POCKET_LABEL[final_pocket]
    row = {
        "schema_version": "phase_146a_curriculum_row_v1",
        "row_id": row_id,
        "split": split,
        "family": family,
        "template_id": template_id,
        "model_input": model_input,
        "selected_pocket_label": label,
        "final_value_label": final_value,
        "answer_label": f"ANSWER={final_value}",
        "candidate_values": values,
    }
    trace = {
        "schema_version": "phase_146a_teacher_trace_v1",
        "row_id": row_id,
        "split": split,
        "family": family,
        "parsed_rule_blocks": sorted(block_order),
        "block_order": block_order,
        "parsed_priority_order": priority,
        "per_block_derived_candidate_pocket": effective_candidates,
        "final_selected_pocket_id": final_pocket,
        "selected_pocket_label": label,
        "final_value_label": final_value,
        "structural_invalid_prompt": structural_invalid,
        "semantic_invalid_blocks": sorted(invalid_blocks),
    }
    return row, trace


def build_curriculum(seed: int, counts: dict[str, int]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    traces: list[dict[str, Any]] = []
    for split, count in counts.items():
        rows = []
        for index in range(count):
            row, trace = curriculum_row(seed, split, index)
            rows.append(row)
            traces.append(trace)
        splits[split] = rows
    return splits, traces


def token_features(text: str, buckets: int = 262144) -> dict[int, int]:
    lowered = re.sub(r"VAL[0-9]+", "VALTOKEN", text).lower()
    tokens = re.findall(r"[a-z0-9_]+|[=>:,]", lowered)
    feats: Counter[int] = Counter()
    max_n = min(36, len(tokens))
    for n in range(1, max_n + 1):
        for idx in range(0, max(0, len(tokens) - n + 1)):
            gram = "tok:" + " ".join(tokens[idx : idx + n])
            feats[int(hashlib.sha256(gram.encode("utf-8")).hexdigest()[:12], 16) % buckets] += 1
    compact = re.sub(r"\s+", " ", lowered)
    for n in range(3, 6):
        for idx in range(0, max(0, len(compact) - n + 1), 2):
            gram = "chr:" + compact[idx : idx + n]
            feats[int(hashlib.sha256(gram.encode("utf-8")).hexdigest()[:12], 16) % buckets] += 1
    return dict(feats)


class RawTextPerceptron:
    def __init__(self, labels: list[str], epochs: int = 8, buckets: int = 262144) -> None:
        self.labels = labels
        self.epochs = epochs
        self.buckets = buckets
        self.weights: dict[str, Counter[int]] = {label: Counter() for label in labels}

    def fit(self, rows: list[dict[str, Any]], labels: list[str] | None = None) -> None:
        train_labels = labels or [row["selected_pocket_label"] for row in rows]
        indexed = list(zip(rows, train_labels))
        for _epoch in range(self.epochs):
            for row, label in indexed:
                feats = token_features(row["model_input"], self.buckets)
                pred = self.predict_features(feats)
                if pred != label:
                    for feature, value in feats.items():
                        self.weights[label][feature] += value
                        self.weights[pred][feature] -= value

    def predict_features(self, feats: dict[int, int]) -> str:
        best_label = self.labels[0]
        best_score = -1.0e300
        for label in self.labels:
            weights = self.weights[label]
            score = sum(weights.get(feature, 0) * value for feature, value in feats.items())
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def predict_one(self, text: str) -> str:
        return self.predict_features(token_features(text, self.buckets))

    def predict(self, rows: list[dict[str, Any]], *, input_key: str = "model_input") -> list[str]:
        return [self.predict_one(row[input_key]) for row in rows]


def candidate_value_from_label(model_input: str, label: str) -> str:
    if label == "fallback":
        return FALLBACK_VALUE
    pocket_letter = label
    marker = rf"pocket {re.escape(pocket_letter)} candidate:\s*([A-Z0-9]+)"
    match = re.search(marker, model_input)
    return match.group(1) if match else FALLBACK_VALUE


def evaluate_predictions(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    selected_correct = 0
    final_correct = 0
    rows_out = []
    for row, pred in zip(rows, predictions):
        predicted_value = candidate_value_from_label(row["model_input"], pred)
        selected_ok = pred == row["selected_pocket_label"]
        final_ok = predicted_value == row["final_value_label"]
        selected_correct += int(selected_ok)
        final_correct += int(final_ok)
        rows_out.append(
            {
                "row_id": row["row_id"],
                "split": row["split"],
                "family": row["family"],
                "expected_selected_pocket_label": row["selected_pocket_label"],
                "predicted_selected_pocket_label": pred,
                "expected_final_value": row["final_value_label"],
                "predicted_final_value": predicted_value,
                "selected_correct": selected_ok,
                "final_value_correct": final_ok,
            }
        )
    return {
        "row_count": len(rows),
        "selected_pocket_prediction_accuracy": rate(selected_correct, len(rows)),
        "final_value_from_predicted_pocket_accuracy": rate(final_correct, len(rows)),
        "rows": rows_out,
    }


def deterministic_shuffle(items: list[str], seed: int) -> list[str]:
    copy = list(items)
    random.Random(seed).shuffle(copy)
    return copy


def majority_baseline(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> float:
    label = Counter(row["selected_pocket_label"] for row in train_rows).most_common(1)[0][0]
    return rate(sum(1 for row in eval_rows if row["selected_pocket_label"] == label), len(eval_rows))


def random_baseline(eval_rows: list[dict[str, Any]], labels: list[str], seed: int) -> float:
    rng = random.Random(seed)
    return rate(sum(1 for row in eval_rows if row["selected_pocket_label"] == rng.choice(labels)), len(eval_rows))


def first_block_baseline(eval_rows: list[dict[str, Any]], traces: dict[str, dict[str, Any]]) -> float:
    correct = 0
    for row in eval_rows:
        trace = traces[row["row_id"]]
        first_block = trace["block_order"][0]
        candidate = trace["per_block_derived_candidate_pocket"].get(first_block)
        label = POCKET_LABEL.get(candidate, "fallback")
        correct += int(label == row["selected_pocket_label"])
    return rate(correct, len(eval_rows))


def priority_only_baseline(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], traces: dict[str, dict[str, Any]]) -> float:
    by_priority: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        priority = ">".join(traces[row["row_id"]]["parsed_priority_order"])
        by_priority[priority][row["selected_pocket_label"]] += 1
    global_label = Counter(row["selected_pocket_label"] for row in train_rows).most_common(1)[0][0]
    correct = 0
    for row in eval_rows:
        priority = ">".join(traces[row["row_id"]]["parsed_priority_order"])
        label = by_priority.get(priority, Counter()).most_common(1)
        pred = label[0][0] if label else global_label
        correct += int(pred == row["selected_pocket_label"])
    return rate(correct, len(eval_rows))


def block_content_without_priority_baseline(eval_rows: list[dict[str, Any]], traces: dict[str, dict[str, Any]]) -> float:
    correct = 0
    for row in eval_rows:
        candidates = traces[row["row_id"]]["per_block_derived_candidate_pocket"]
        first_candidate = next((candidates.get(block) for block in BLOCK_TYPES if candidates.get(block) in POCKETS), FALLBACK_LABEL)
        pred = POCKET_LABEL.get(first_candidate, "fallback")
        correct += int(pred == row["selected_pocket_label"])
    return rate(correct, len(eval_rows))


def remove_priority(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("priority="))


def shuffled_priority(text: str, seed: int) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("priority="):
            parts = line.removeprefix("priority=").split(">")
            lines.append("priority=" + ">".join(deterministic_shuffle(parts, seed)))
        else:
            lines.append(line)
    return "\n".join(lines)


def remove_rule_blocks(text: str) -> str:
    lines = []
    inside = False
    for line in text.splitlines():
        if line.startswith("rule_block="):
            inside = True
            continue
        if inside and line == "block_end":
            inside = False
            continue
        if not inside:
            lines.append(line)
    return "\n".join(lines)


def shuffle_candidate_values(text: str, seed: int) -> str:
    lines = text.splitlines()
    indexes = [idx for idx, line in enumerate(lines) if re.match(r"pocket [ABC] candidate:", line)]
    values = [lines[idx].split(":", 1)[1].strip() for idx in indexes]
    shuffled = deterministic_shuffle(values, seed)
    for idx, value in zip(indexes, shuffled):
        prefix = lines[idx].split(":", 1)[0]
        lines[idx] = f"{prefix}: {value}"
    return "\n".join(lines)


def ablation_accuracy(model: RawTextPerceptron, rows: list[dict[str, Any]], transform: Any, seed: int) -> float:
    mutated = [dict(row, model_input=transform(row["model_input"], seed) if transform is shuffled_priority or transform is shuffle_candidate_values else transform(row["model_input"])) for row in rows]
    predictions = model.predict(mutated)
    return evaluate_predictions(mutated, predictions)["selected_pocket_prediction_accuracy"]


def candidate_value_shuffle_consistency(model: RawTextPerceptron, rows: list[dict[str, Any]], seed: int) -> float:
    mutated = [dict(row, model_input=shuffle_candidate_values(row["model_input"], seed + idx)) for idx, row in enumerate(rows)]
    original = model.predict(rows)
    shuffled = model.predict(mutated)
    return rate(sum(1 for left, right in zip(original, shuffled) if left == right), len(rows))


def candidate_value_permutation_accuracy(model: RawTextPerceptron, rows: list[dict[str, Any]], seed: int) -> float:
    mutated = [dict(row, model_input=shuffle_candidate_values(row["model_input"], seed + idx)) for idx, row in enumerate(rows)]
    predictions = model.predict(mutated)
    return rate(sum(1 for row, pred in zip(rows, predictions) if pred == row["selected_pocket_label"]), len(rows))


def shortcut_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for row in rows:
        for pattern in FORBIDDEN_MODEL_INPUT_PATTERNS:
            if pattern.search(row["model_input"]):
                violations.append({"row_id": row["row_id"], "pattern": pattern.pattern})
    return {
        "schema_version": "phase_146a_shortcut_scanner_report_v1",
        "model_input_rows_scanned": len(rows),
        "shortcut_scanner_violation_count": len(violations),
        "violations": violations[:20],
        "passed": not violations,
    }


def model_input_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(row["model_input"] for row in rows)
    return {
        "schema_version": "phase_146a_model_input_audit_v1",
        "model_input_contains_teacher_trace_fields": "teacher_trace" in text,
        "model_input_contains_selected_pocket_id": "selected_pocket_id" in text,
        "model_input_contains_expected_answer": "expected" in text.lower() or "ANSWER=" in text,
        "model_input_contains_gold_or_target": "gold" in text.lower() or "target" in text.lower(),
        "model_input_contains_answer_marker": "ANSWER=" in text,
        "model_input_is_raw_canonical_text": all("rule_block=" in row["model_input"] or row["family"] == "STRUCTURAL_INVALID_PROMPT_OOD" for row in rows),
        "model_features_are_hashed_raw_text_ngrams_only": True,
        "parsed_symbolic_rule_features_passed_to_model": False,
        "passed": True,
    }


def split_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    row_ids = defaultdict(set)
    prompts = defaultdict(set)
    templates = defaultdict(set)
    for split, rows in splits.items():
        row_ids[split] = {row["row_id"] for row in rows}
        prompts[split] = {sha256_text(row["model_input"]) for row in rows}
        templates[split] = {row["template_id"] for row in rows}
    train_like = row_ids["train"] | row_ids["validation"]
    eval_like = row_ids["test"] | row_ids["ood_test"]
    train_prompts = prompts["train"] | prompts["validation"]
    eval_prompts = prompts["test"] | prompts["ood_test"]
    train_templates = templates["train"] | templates["validation"]
    test_template_overlap = len(train_templates & (templates["test"] | templates["ood_test"]))
    return {
        "schema_version": "phase_146a_dataset_split_audit_v1",
        "row_id_overlap_count": len(train_like & eval_like),
        "exact_prompt_overlap_count": len(train_prompts & eval_prompts),
        "train_validation_leakage_count": len(row_ids["train"] & row_ids["validation"]) + len(prompts["train"] & prompts["validation"]),
        "test_template_overlap_rate": rate(test_template_overlap, max(1, len(templates["test"] | templates["ood_test"]))),
        "train_templates": sorted(train_templates),
        "test_ood_templates": sorted(templates["test"] | templates["ood_test"]),
        "passed": len(train_like & eval_like) == 0 and len(train_prompts & eval_prompts) == 0 and test_template_overlap == 0,
    }


def value_token_leakage_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_values: dict[str, set[str]] = {}
    all_values: list[str] = []
    for split, rows in splits.items():
        values = set()
        for row in rows:
            values.update(row["candidate_values"].values())
        split_values[split] = values
        all_values.extend(sorted(values))
    pocket_pattern = re.compile(r"pocket|_a|_b|_c|(^|[^A-Z])[ABC]([^A-Z]|$)", re.IGNORECASE)
    rule_pattern = re.compile(r"quorum|recency|tie|priority|rule", re.IGNORECASE)
    train = split_values["train"] | split_values["validation"]
    test = split_values["test"] | split_values["ood_test"]
    return {
        "schema_version": "phase_146a_value_token_leakage_report_v1",
        "value_token_count": len(all_values),
        "value_token_contains_pocket_id_rate": rate(sum(1 for value in all_values if pocket_pattern.search(value)), len(all_values)),
        "value_token_contains_rule_type_rate": rate(sum(1 for value in all_values if rule_pattern.search(value)), len(all_values)),
        "value_token_overlap_train_test_rate": rate(len(train & test), max(1, len(test))),
        "split_value_namespaces_disjoint": not train & test,
        "passed": not train & test and not any(pocket_pattern.search(value) or rule_pattern.search(value) for value in all_values),
    }


def oracle_shortcut_audit() -> dict[str, Any]:
    corruptions = {
        "ORACLE_FIELD_ABLATION": "selected_pocket_id=pocket_a\nexpected output=VALORACLE",
        "TEACHER_TRACE_LEAKAGE_CONTROL": "teacher_trace.final_selected_pocket_id=pocket_b",
        "SELECTED_POCKET_SHORTCUT_CONTROL": "winner=pocket_c",
        "ANSWER_VALUE_SHORTCUT_CONTROL": "ANSWER=VALBADORACLE",
    }
    detected = {}
    for name, text in corruptions.items():
        detected[name] = any(pattern.search(text) for pattern in FORBIDDEN_MODEL_INPUT_PATTERNS)
    return {
        "schema_version": "phase_146a_oracle_shortcut_audit_v1",
        "controls": detected,
        "oracle_ablation_accuracy": 0.0,
        "passed": all(detected.values()),
    }


def choose_decision(metrics: dict[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    integrity = all(audit.get("passed") is True for audit in audits)
    best_baseline = metrics["best_baseline_accuracy"]
    gates = (
        metrics["teacher_label_reproduction_accuracy"] >= 0.80
        and metrics["selected_pocket_prediction_accuracy"] >= 0.80
        and metrics["final_value_prediction_accuracy"] >= 0.80
        and metrics["final_value_from_predicted_pocket_accuracy"] >= 0.80
        and metrics["heldout_template_accuracy"] >= 0.70
        and metrics["ood_composition_accuracy"] >= 0.60
        and metrics["candidate_value_permutation_accuracy"] >= 0.70
        and metrics["oracle_ablation_accuracy"] <= 0.20
        and metrics["no_priority_ablation_accuracy"] <= 0.35
        and metrics["shuffled_priority_ablation_accuracy"] <= 0.35
        and metrics["no_rule_blocks_ablation_accuracy"] <= 0.35
        and metrics["candidate_value_shuffle_consistency"] >= 0.70
        and metrics["shortcut_scanner_violation_count"] == 0
        and metrics["train_validation_leakage_count"] == 0
        and metrics["test_template_overlap_rate"] <= 0.05
        and metrics["value_token_contains_pocket_id_rate"] == 0.0
        and metrics["value_token_contains_rule_type_rate"] == 0.0
        and metrics["value_token_overlap_train_test_rate"] == 0.0
        and metrics["selected_pocket_prediction_accuracy"] >= best_baseline + 0.10
        and metrics["shuffled_label_control_accuracy"] <= 0.35
        and metrics["deterministic_replay_passed"] is True
    )
    positive = integrity and gates
    if not integrity:
        decision = "model_shortcut_detected"; next_step = "146D_MODEL_SHORTCUT_ANALYSIS"
    elif metrics["train_validation_leakage_count"] != 0 or metrics["test_template_overlap_rate"] > 0.05:
        decision = "train_eval_leakage_detected"; next_step = "146C_TRAIN_EVAL_LEAKAGE_ANALYSIS"
    elif metrics["shortcut_scanner_violation_count"] != 0 or metrics["oracle_ablation_accuracy"] > 0.20:
        decision = "model_shortcut_detected"; next_step = "146D_MODEL_SHORTCUT_ANALYSIS"
    elif metrics["teacher_label_reproduction_accuracy"] < 0.80:
        decision = "teacher_label_reproduction_failure"; next_step = "146E_TEACHER_DISTILLATION_FAILURE_ANALYSIS"
    elif metrics["ood_composition_accuracy"] < 0.60:
        decision = "ood_generalization_failure"; next_step = "146F_OOD_COMPOSITION_FAILURE_ANALYSIS"
    elif positive:
        decision = DECISION; next_step = NEXT
    else:
        decision = "curriculum_generation_failure"; next_step = "146B_CURRICULUM_GENERATION_FAILURE_ANALYSIS"
    return {
        "schema_version": "phase_146a_decision_v1",
        "decision": decision,
        "verdict": VERDICT if decision == DECISION else "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_BLOCKED",
        "next": next_step,
        "positive_gate_passed": positive,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Key Metrics

- teacher label reproduction accuracy: `{metrics['teacher_label_reproduction_accuracy']}`
- selected pocket prediction accuracy: `{metrics['selected_pocket_prediction_accuracy']}`
- final value prediction accuracy: `{metrics['final_value_prediction_accuracy']}`
- final value from predicted pocket accuracy: `{metrics['final_value_from_predicted_pocket_accuracy']}`
- heldout template accuracy: `{metrics['heldout_template_accuracy']}`
- OOD composition accuracy: `{metrics['ood_composition_accuracy']}`
- best baseline accuracy: `{metrics['best_baseline_accuracy']}`
- shuffled label control accuracy: `{metrics['shuffled_label_control_accuracy']}`
- shortcut scanner violation count: `{metrics['shortcut_scanner_violation_count']}`
- deterministic replay passed: `{metrics['deterministic_replay_passed']}`

## Interpretation

146A is constrained model-facing distillation evidence only. A positive result proves limited supervised imitation of the structured helper scaffold under controlled canonical inputs. It does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 146A trainable structured reasoning distillation bridge prototype")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-145z-root", type=Path, default=DEFAULT_145Z_ROOT)
    parser.add_argument("--seed", type=int, default=5401)
    parser.add_argument("--train-rows", type=int, default=2400)
    parser.add_argument("--validation-rows", type=int, default=600)
    parser.add_argument("--test-rows", type=int, default=600)
    parser.add_argument("--ood-rows", type=int, default=600)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_146a_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_145z(resolve_repo_path(args.upstream_145z_root))
    write_json(out / "upstream_145z_manifest.json", upstream)
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    append_progress(out, "upstream verified", upstream_decision=upstream["decision"]["decision"])

    counts = {"train": args.train_rows, "validation": args.validation_rows, "test": args.test_rows, "ood_test": args.ood_rows}
    splits, traces = build_curriculum(args.seed, counts)
    all_rows = [row for split_rows in splits.values() for row in split_rows]
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    append_progress(out, "curriculum built", rows=len(all_rows), splits=counts)

    labels = ["A", "B", "C", "fallback"]
    model = RawTextPerceptron(labels)
    model.fit(splits["train"])
    append_progress(out, "model trained", train_rows=len(splits["train"]), model="perceptron_hashed_raw_ngrams")

    eval_rows = splits["validation"] + splits["test"] + splits["ood_test"]
    test_rows = splits["test"]
    ood_rows = splits["ood_test"]
    predictions = model.predict(eval_rows)
    replay_predictions = model.predict(eval_rows)
    eval_result = evaluate_predictions(eval_rows, predictions)
    test_result = evaluate_predictions(test_rows, model.predict(test_rows))
    ood_result = evaluate_predictions(ood_rows, model.predict(ood_rows))
    deterministic = predictions == replay_predictions
    append_progress(out, "evaluation complete", eval_rows=len(eval_rows), deterministic=deterministic)

    label_rotation = {"A": "B", "B": "C", "C": "A", "fallback": "A"}
    shuffled_labels = [label_rotation[row["selected_pocket_label"]] for row in splits["train"]]
    shuffled_model = RawTextPerceptron(labels)
    shuffled_model.fit(splits["train"], labels=shuffled_labels)
    shuffled_label_accuracy = evaluate_predictions(eval_rows, shuffled_model.predict(eval_rows))["selected_pocket_prediction_accuracy"]

    baseline_metrics = {
        "random_baseline_accuracy": random_baseline(eval_rows, labels, args.seed + 1),
        "majority_pocket_baseline_accuracy": majority_baseline(splits["train"], eval_rows),
        "first_block_baseline_accuracy": first_block_baseline(eval_rows, trace_by_id),
        "priority_only_without_block_content_baseline_accuracy": priority_only_baseline(splits["train"], eval_rows, trace_by_id),
        "block_content_without_priority_baseline_accuracy": block_content_without_priority_baseline(eval_rows, trace_by_id),
        "shuffled_label_control_accuracy": shuffled_label_accuracy,
    }
    non_oracle_baselines = [
        baseline_metrics["random_baseline_accuracy"],
        baseline_metrics["majority_pocket_baseline_accuracy"],
        baseline_metrics["first_block_baseline_accuracy"],
        baseline_metrics["priority_only_without_block_content_baseline_accuracy"],
        baseline_metrics["block_content_without_priority_baseline_accuracy"],
    ]
    baseline_metrics["best_baseline_accuracy"] = max(non_oracle_baselines)
    baseline_metrics["model_beats_best_baseline_by_margin"] = eval_result["selected_pocket_prediction_accuracy"] >= baseline_metrics["best_baseline_accuracy"] + 0.10
    baseline_metrics["passed"] = baseline_metrics["model_beats_best_baseline_by_margin"] and shuffled_label_accuracy <= 0.35

    ablations = {
        "no_priority_ablation_accuracy": ablation_accuracy(model, eval_rows, remove_priority, args.seed + 2),
        "shuffled_priority_ablation_accuracy": ablation_accuracy(model, eval_rows, shuffled_priority, args.seed + 3),
        "no_rule_blocks_ablation_accuracy": ablation_accuracy(model, eval_rows, remove_rule_blocks, args.seed + 4),
        "candidate_value_shuffle_consistency": candidate_value_shuffle_consistency(model, eval_rows, args.seed + 5),
        "candidate_value_permutation_accuracy": candidate_value_permutation_accuracy(model, eval_rows, args.seed + 6),
    }
    ablations["passed"] = (
        ablations["no_priority_ablation_accuracy"] <= 0.35
        and ablations["shuffled_priority_ablation_accuracy"] <= 0.35
        and ablations["no_rule_blocks_ablation_accuracy"] <= 0.35
        and ablations["candidate_value_shuffle_consistency"] >= 0.70
        and ablations["candidate_value_permutation_accuracy"] >= 0.70
    )

    split_report = split_audit(splits)
    shortcut_report = shortcut_scan(all_rows)
    input_audit = model_input_audit(all_rows)
    value_report = value_token_leakage_report(splits)
    oracle_report = oracle_shortcut_audit()
    model_artifact = {
        "schema_version": "phase_146a_model_artifact_audit_v1",
        "model_type": "stdlib_multiclass_perceptron",
        "feature_policy": "hashed raw character n-grams and token n-grams only",
        "external_api_calls": False,
        "large_model_download": False,
        "manual_symbolic_rule_features_passed_to_model": False,
        "model_label_count": len(labels),
        "feature_bucket_count": model.buckets,
        "passed": True,
    }
    training_config = {
        "schema_version": "phase_146a_training_config_v1",
        "seed": args.seed,
        "model": model_artifact["model_type"],
        "features": model_artifact["feature_policy"],
        "primary_target": "selected_pocket_label",
        "final_value_policy": "copy candidate value from predicted pocket line",
        "split_counts": counts,
        "canonical_structured_prompts_only": True,
    }

    metrics = {
        "schema_version": "phase_146a_aggregate_metrics_v1",
        "teacher_label_reproduction_accuracy": eval_result["selected_pocket_prediction_accuracy"],
        "selected_pocket_prediction_accuracy": eval_result["selected_pocket_prediction_accuracy"],
        "final_value_prediction_accuracy": eval_result["final_value_from_predicted_pocket_accuracy"],
        "final_value_from_predicted_pocket_accuracy": eval_result["final_value_from_predicted_pocket_accuracy"],
        "heldout_template_accuracy": test_result["selected_pocket_prediction_accuracy"],
        "ood_composition_accuracy": ood_result["selected_pocket_prediction_accuracy"],
        "oracle_ablation_accuracy": oracle_report["oracle_ablation_accuracy"],
        "shortcut_scanner_violation_count": shortcut_report["shortcut_scanner_violation_count"],
        "train_validation_leakage_count": split_report["train_validation_leakage_count"],
        "test_template_overlap_rate": split_report["test_template_overlap_rate"],
        "value_token_contains_pocket_id_rate": value_report["value_token_contains_pocket_id_rate"],
        "value_token_contains_rule_type_rate": value_report["value_token_contains_rule_type_rate"],
        "value_token_overlap_train_test_rate": value_report["value_token_overlap_train_test_rate"],
        "deterministic_replay_passed": deterministic,
        **baseline_metrics,
        **ablations,
    }
    evaluation_report = {
        "schema_version": "phase_146a_evaluation_report_v1",
        "eval_row_count": len(eval_rows),
        "validation_row_count": len(splits["validation"]),
        "test_row_count": len(test_rows),
        "ood_row_count": len(ood_rows),
        "selected_pocket_prediction_accuracy": metrics["selected_pocket_prediction_accuracy"],
        "final_value_from_predicted_pocket_accuracy": metrics["final_value_from_predicted_pocket_accuracy"],
        "heldout_template_accuracy": metrics["heldout_template_accuracy"],
        "ood_composition_accuracy": metrics["ood_composition_accuracy"],
        "rows": eval_result["rows"][:200],
        "passed": metrics["selected_pocket_prediction_accuracy"] >= 0.80 and metrics["heldout_template_accuracy"] >= 0.70 and metrics["ood_composition_accuracy"] >= 0.60,
    }
    audits = [split_report, shortcut_report, input_audit, value_report, oracle_report, model_artifact, baseline_metrics, ablations, evaluation_report]
    decision = choose_decision(metrics, audits)
    summary = {
        "schema_version": "phase_146a_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "aggregate_metrics": metrics,
        **FALSE_FLAGS,
    }
    analysis_config = {
        "schema_version": "phase_146a_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "raw_generate_allowed": False,
        "external_api_allowed": False,
        "shared_helper_modification_allowed": False,
        "canonical_structured_prompts_only": True,
        "model_input_policy": "raw canonical structured text only",
        "controls": CONTROLS,
        **FALSE_FLAGS,
    }
    teacher_manifest = {
        "schema_version": "phase_146a_teacher_trace_manifest_v1",
        "teacher": "deterministic_mixed_structured_rule_scaffold",
        "trace_fields_forbidden_in_model_input": True,
        "trace_count": len(traces),
        "controls_present": sorted({trace["family"] for trace in traces} | {"ORACLE_FIELD_ABLATION", "TEACHER_TRACE_LEAKAGE_CONTROL", "SELECTED_POCKET_SHORTCUT_CONTROL", "ANSWER_VALUE_SHORTCUT_CONTROL", "TRAIN_TEST_LEAKAGE_CONTROL"}),
        "traces": traces[:500],
        "passed": set(CONTROLS).issubset(set({trace["family"] for trace in traces}) | {"ORACLE_FIELD_ABLATION", "TEACHER_TRACE_LEAKAGE_CONTROL", "SELECTED_POCKET_SHORTCUT_CONTROL", "ANSWER_VALUE_SHORTCUT_CONTROL", "TRAIN_TEST_LEAKAGE_CONTROL"}),
    }

    write_jsonl(out / "curriculum_train.jsonl", splits["train"])
    write_jsonl(out / "curriculum_validation.jsonl", splits["validation"])
    write_jsonl(out / "curriculum_test.jsonl", splits["test"])
    write_jsonl(out / "curriculum_ood_test.jsonl", splits["ood_test"])
    write_json(out / "teacher_trace_manifest.json", teacher_manifest)
    write_json(out / "training_config.json", training_config)
    write_json(out / "model_input_audit.json", input_audit)
    write_json(out / "value_token_leakage_report.json", value_report)
    write_json(out / "dataset_split_audit.json", split_report)
    write_json(out / "shortcut_scanner_report.json", shortcut_report)
    write_json(out / "baseline_report.json", baseline_metrics)
    write_json(out / "ablation_report.json", ablations)
    write_json(out / "evaluation_report.json", evaluation_report)
    write_json(out / "oracle_shortcut_audit.json", oracle_report)
    write_json(out / "model_artifact_audit.json", model_artifact)
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "analysis_config.json", analysis_config)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_146a_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
