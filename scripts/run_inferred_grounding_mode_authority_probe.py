#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "inferred-grounding-mode-authority"
REPORT_PATH = ROOT / "docs" / "research" / "INFERRED_GROUNDING_MODE_AUTHORITY_PROBE.md"
SUMMARY_NAME = "inferred_grounding_mode_authority_summary.json"

ACTORS = ["dog", "cat", "snake"]
ACTIONS = ["bit", "chased", "scared"]
PATIENTS = ["me", "john", "child"]
MODES = ["reality", "tv", "game", "dream", "memory"]

CUE_FEATURES: dict[str, list[str]] = {
    "reality": ["now", "present", "physical", "in_room"],
    "tv": ["screen", "watched", "fictional", "episode"],
    "game": ["avatar", "level", "respawn", "controller"],
    "dream": ["asleep", "unreal", "nightmare", "remembered_after"],
    "memory": ["past", "remembered", "autobiographical", "yesterday"],
}
STRICT_UNSEEN_CUES: dict[str, list[str]] = {
    "reality": ["live_body", "immediate_world"],
    "tv": ["broadcast_channel", "subtitles"],
    "game": ["health_bar", "quest_log"],
    "dream": ["sleep_image", "dream_logic"],
    "memory": ["recollection", "old_event"],
}
CUE_ATOMS = [atom for mode in MODES for atom in CUE_FEATURES[mode]]
STRICT_CUE_ATOMS = [atom for mode in MODES for atom in STRICT_UNSEEN_CUES[mode]]


@dataclass(frozen=True)
class Example:
    actor: str
    action: str
    patient: str
    mode: str
    cue: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer grounding mode from compositional cue features.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden = 32
        args.epochs = 30
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cue_pairs(mode: str) -> list[tuple[str, str]]:
    return list(combinations(CUE_FEATURES[mode], 2))


def event_tuples() -> list[tuple[str, str, str]]:
    return [(a, act, p) for a in ACTORS for act in ACTIONS for p in PATIENTS]


def make_examples_for_bundles(bundle_by_mode: dict[str, list[tuple[str, ...]]]) -> list[Example]:
    return [
        Example(actor, action, patient, mode, tuple(bundle))
        for actor, action, patient in event_tuples()
        for mode in MODES
        for bundle in bundle_by_mode[mode]
    ]


def make_compositional_split() -> dict[str, list[Example]]:
    train: dict[str, list[tuple[str, ...]]] = {}
    validation: dict[str, list[tuple[str, ...]]] = {}
    final_test: dict[str, list[tuple[str, ...]]] = {}
    for mode in MODES:
        pairs = cue_pairs(mode)
        train[mode] = [tuple(item) for item in pairs[:4]]
        validation[mode] = [tuple(pairs[4])]
        final_test[mode] = [tuple(pairs[5])]
    return {
        "train": make_examples_for_bundles(train),
        "validation": make_examples_for_bundles(validation),
        "final_test": make_examples_for_bundles(final_test),
    }


def stable_bucket(parts: tuple[Any, ...], mod: int) -> int:
    text = "|".join(map(str, parts)).encode()
    return int(hashlib.sha256(text).hexdigest(), 16) % mod


def make_seen_combo_split() -> dict[str, list[Example]]:
    all_examples = make_examples_for_bundles({mode: [tuple(pair) for pair in cue_pairs(mode)] for mode in MODES})
    split = {"train": [], "validation": [], "final_test": []}
    for ex in all_examples:
        bucket = stable_bucket((ex.actor, ex.action, ex.patient, ex.mode, ex.cue), 7)
        if bucket == 0:
            split["final_test"].append(ex)
        elif bucket == 1:
            split["validation"].append(ex)
        else:
            split["train"].append(ex)
    return split


def make_strict_unseen_examples() -> list[Example]:
    bundle_by_mode = {
        mode: [tuple(STRICT_UNSEEN_CUES[mode])]
        for mode in MODES
    }
    return make_examples_for_bundles(bundle_by_mode)


def input_dim() -> int:
    return len(ACTORS) + len(ACTIONS) + len(PATIENTS) + len(CUE_ATOMS) + len(STRICT_CUE_ATOMS) + len(MODES)


def encode(
    examples: list[Example],
    *,
    input_kind: str,
    mode_override: str | None = None,
    ablate_context: bool = False,
    shuffled_cues: list[tuple[str, ...]] | None = None,
) -> torch.Tensor:
    rows = torch.zeros(len(examples), input_dim())
    actor_offset = 0
    action_offset = actor_offset + len(ACTORS)
    patient_offset = action_offset + len(ACTIONS)
    cue_offset = patient_offset + len(PATIENTS)
    strict_cue_offset = cue_offset + len(CUE_ATOMS)
    mode_offset = strict_cue_offset + len(STRICT_CUE_ATOMS)
    for i, ex in enumerate(examples):
        rows[i, actor_offset + ACTORS.index(ex.actor)] = 1.0
        rows[i, action_offset + ACTIONS.index(ex.action)] = 1.0
        rows[i, patient_offset + PATIENTS.index(ex.patient)] = 1.0
        mode = mode_override or ex.mode
        if input_kind == "explicit_mode":
            rows[i, mode_offset + MODES.index(mode)] = 1.0
        elif input_kind == "cue" and not ablate_context:
            if mode_override is not None:
                cue = tuple(cue_pairs(mode_override)[0])
            else:
                cue = shuffled_cues[i] if shuffled_cues is not None else ex.cue
            for atom in cue:
                if atom in CUE_ATOMS:
                    rows[i, cue_offset + CUE_ATOMS.index(atom)] = 1.0
                elif atom in STRICT_CUE_ATOMS:
                    rows[i, strict_cue_offset + STRICT_CUE_ATOMS.index(atom)] = 1.0
        elif input_kind == "no_context":
            pass
        else:
            if input_kind not in {"cue", "no_context"}:
                raise ValueError(input_kind)
    return rows


def targets(examples: list[Example]) -> dict[str, torch.Tensor]:
    semantic = []
    mode = []
    authority = []
    for ex in examples:
        threat = ex.action in {"bit", "scared"}
        semantic.append([float(ex.action == "bit"), float(threat)])
        mode.append(MODES.index(ex.mode))
        real_action = float(ex.mode == "reality" and threat and ex.patient == "me")
        story = float(ex.mode == "tv")
        game = float(ex.mode == "game")
        memory = float(ex.mode in {"dream", "memory"})
        self_relevance = float(ex.mode == "reality" and ex.patient == "me")
        help_other = float(ex.mode == "reality" and threat and ex.patient != "me")
        authority.append([real_action, story, game, memory, self_relevance, help_other])
    return {
        "semantic": torch.tensor(semantic, dtype=torch.float32),
        "mode": torch.tensor(mode, dtype=torch.long),
        "authority": torch.tensor(authority, dtype=torch.float32),
    }


class GroundingModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim(), hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.semantic = nn.Linear(hidden, 2)
        self.mode = nn.Linear(hidden, len(MODES))
        self.authority = nn.Linear(hidden, 6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {"semantic": self.semantic(h), "mode": self.mode(h), "authority": self.authority(h)}


def loss_for(model: GroundingModel, examples: list[Example], *, input_kind: str, arm: str) -> torch.Tensor:
    pred = model(encode(examples, input_kind=input_kind))
    tgt = targets(examples)
    sem = F.binary_cross_entropy_with_logits(pred["semantic"], tgt["semantic"])
    if arm == "semantic_only_baseline":
        return sem
    mode = F.cross_entropy(pred["mode"], tgt["mode"])
    auth = F.binary_cross_entropy_with_logits(pred["authority"], tgt["authority"])
    return sem + mode + auth


def train_model(seed: int, split: dict[str, list[Example]], *, hidden: int, epochs: int, lr: float, input_kind: str, arm: str) -> GroundingModel:
    set_seed(seed)
    model = GroundingModel(hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_for(model, split["train"], input_kind=input_kind, arm=arm)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def binary_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(((torch.sigmoid(logits) >= 0.5) == (y >= 0.5)).float().mean().item())


@torch.no_grad()
def evaluate_model(
    model: GroundingModel,
    examples: list[Example],
    *,
    input_kind: str,
    mode_override: str | None = None,
    ablate_context: bool = False,
    shuffled_cues: list[tuple[str, ...]] | None = None,
) -> dict[str, float]:
    pred = model(encode(examples, input_kind=input_kind, mode_override=mode_override, ablate_context=ablate_context, shuffled_cues=shuffled_cues))
    tgt = targets(examples)
    semantic_prob = torch.sigmoid(pred["semantic"])
    authority_prob = torch.sigmoid(pred["authority"])
    mode_pred = pred["mode"].argmax(dim=-1)
    semantic_acc = binary_acc(pred["semantic"], tgt["semantic"])
    mode_acc = float((mode_pred == tgt["mode"]).float().mean().item())
    authority_acc = binary_acc(pred["authority"], tgt["authority"])
    real_self = [i for i, ex in enumerate(examples) if ex.mode == "reality" and ex.patient == "me" and ex.action in {"bit", "scared"}]
    nonreal_self = [i for i, ex in enumerate(examples) if ex.mode != "reality" and ex.patient == "me" and ex.action in {"bit", "scared"}]
    real_me_bit = [i for i, ex in enumerate(examples) if ex.mode == "reality" and ex.patient == "me" and ex.action == "bit"]
    real_john_bit = [i for i, ex in enumerate(examples) if ex.mode == "reality" and ex.patient == "john" and ex.action == "bit"]
    bit_me = [i for i, ex in enumerate(examples) if ex.patient == "me" and ex.action == "bit"]
    semantic_by_mode = []
    for mode in MODES:
        idx = [i for i in bit_me if examples[i].mode == mode]
        if idx:
            semantic_by_mode.append(float(semantic_prob[idx, 0].mean().item()))
    reality_action = float(authority_prob[real_self, 0].mean().item()) if real_self else 0.0
    nonreality_leakage = float(authority_prob[nonreal_self, 0].mean().item()) if nonreal_self else 0.0
    self_anchor_gain = 0.0
    if real_me_bit and real_john_bit:
        self_anchor_gain = float(authority_prob[real_me_bit, 0].mean().item() - authority_prob[real_john_bit, 0].mean().item())
    return {
        "overall_accuracy": float(np.mean([semantic_acc, mode_acc, authority_acc])),
        "semantic_accuracy": semantic_acc,
        "grounding_mode_accuracy": mode_acc,
        "action_authority_accuracy": authority_acc,
        "reality_action_authority": reality_action,
        "nonreality_action_leakage": nonreality_leakage,
        "grounding_authority_margin": reality_action - nonreality_leakage,
        "self_anchor_gain": self_anchor_gain,
        "semantic_consistency_range": max(semantic_by_mode) - min(semantic_by_mode) if semantic_by_mode else 0.0,
    }


@torch.no_grad()
def control_metrics(model: GroundingModel, examples: list[Example], *, input_kind: str, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    cue_pool = [ex.cue for ex in examples]
    shuffled = cue_pool[:]
    rng.shuffle(shuffled)
    base = evaluate_model(model, examples, input_kind=input_kind)
    no_context = evaluate_model(model, examples, input_kind=input_kind, ablate_context=True)
    shuffled_metrics = evaluate_model(model, examples, input_kind=input_kind, shuffled_cues=shuffled)
    forced_tv = evaluate_model(model, examples, input_kind=input_kind, mode_override="tv")
    forced_reality = evaluate_model(model, examples, input_kind=input_kind, mode_override="reality")
    forced_tv_sem = forced_tv["semantic_accuracy"]
    forced_reality_sem = forced_reality["semantic_accuracy"]
    real_self_examples = [ex for ex in examples if ex.mode == "reality" and ex.patient == "me" and ex.action in {"bit", "scared"}]
    nonreal_self_examples = [ex for ex in examples if ex.mode != "reality" and ex.patient == "me" and ex.action in {"bit", "scared"}]
    base_real = evaluate_model(model, real_self_examples, input_kind=input_kind)["reality_action_authority"] if real_self_examples else 0.0
    forced_tv_real = evaluate_model(model, real_self_examples, input_kind=input_kind, mode_override="tv")["reality_action_authority"] if real_self_examples else 0.0
    base_nonreal_leak = evaluate_model(model, nonreal_self_examples, input_kind=input_kind)["nonreality_action_leakage"] if nonreal_self_examples else 0.0
    forced_reality_nonreal = evaluate_model(model, nonreal_self_examples, input_kind=input_kind, mode_override="reality")["nonreality_action_leakage"] if nonreal_self_examples else 0.0
    return {
        "mode_ablation_drop": base["overall_accuracy"] - no_context["overall_accuracy"],
        "mode_shuffle_drop": base["overall_accuracy"] - shuffled_metrics["overall_accuracy"],
        "wrong_forced_mode_drop": base_real - forced_tv_real,
        "wrong_forced_reality_rise": forced_reality_nonreal - base_nonreal_leak,
        "wrong_forced_tv_semantic_accuracy": forced_tv_sem,
        "wrong_forced_reality_semantic_accuracy": forced_reality_sem,
    }


def run_split(args: argparse.Namespace, split_name: str, split: dict[str, list[Example]], seed: int) -> list[dict[str, Any]]:
    arms = [
        ("explicit_mode_upper_bound", "explicit_mode"),
        ("inferred_context_mode", "cue"),
        ("no_context", "no_context"),
        ("semantic_only_baseline", "cue"),
    ]
    rows: list[dict[str, Any]] = []
    for arm_i, (arm, input_kind) in enumerate(arms):
        model = train_model(seed + 10_000 * arm_i, split, hidden=args.hidden, epochs=args.epochs, lr=args.learning_rate, input_kind=input_kind, arm=arm)
        final = evaluate_model(model, split["final_test"], input_kind=input_kind)
        row = {"split": split_name, "arm": arm, "seed": seed, "final_test": final}
        if arm == "inferred_context_mode":
            row["controls"] = control_metrics(model, split["final_test"], input_kind=input_kind, seed=seed + 991)
            strict = make_strict_unseen_examples()
            row["strict_unseen_cue_token_diagnostic"] = evaluate_model(model, strict, input_kind=input_kind)
        rows.append(row)
    return rows


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in sorted({row["split"] for row in records}):
        out[split] = {}
        for arm in sorted({row["arm"] for row in records if row["split"] == split}):
            rows = [row for row in records if row["split"] == split and row["arm"] == arm]
            metrics = sorted({key for row in rows for key in row["final_test"]})
            item = {key: numeric_summary([float(row["final_test"][key]) for row in rows]) for key in metrics}
            control_keys = sorted({key for row in rows for key in row.get("controls", {})})
            if control_keys:
                item["controls"] = {key: numeric_summary([float(row["controls"][key]) for row in rows if "controls" in row]) for key in control_keys}
            strict_keys = sorted({key for row in rows for key in row.get("strict_unseen_cue_token_diagnostic", {})})
            if strict_keys:
                item["strict_unseen_cue_token_diagnostic"] = {
                    key: numeric_summary([float(row["strict_unseen_cue_token_diagnostic"][key]) for row in rows if "strict_unseen_cue_token_diagnostic" in row])
                    for key in strict_keys
                }
            out[split][arm] = item
    return out


def mean_at(agg: dict[str, Any], split: str, arm: str, key: str) -> float:
    value = agg.get(split, {}).get(arm, {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def control_mean(agg: dict[str, Any], split: str, arm: str, key: str) -> float:
    value = agg.get(split, {}).get(arm, {}).get("controls", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    split = "compositional_cue_split"
    arm = "inferred_context_mode"
    semantic = mean_at(agg, split, arm, "semantic_accuracy")
    mode = mean_at(agg, split, arm, "grounding_mode_accuracy")
    margin = mean_at(agg, split, arm, "grounding_authority_margin")
    leakage = mean_at(agg, split, arm, "nonreality_action_leakage")
    self_gain = mean_at(agg, split, arm, "self_anchor_gain")
    wrong_drop = control_mean(agg, split, arm, "wrong_forced_mode_drop")
    no_context_gap = mean_at(agg, split, arm, "overall_accuracy") - mean_at(agg, split, "no_context", "overall_accuracy")
    shuffled_drop = control_mean(agg, split, arm, "mode_shuffle_drop")
    return {
        "supports_inferred_grounding_mode_authority": semantic >= 0.98 and mode >= 0.95 and margin >= 0.70 and leakage <= 0.15,
        "supports_semantic_grounding_split": semantic >= 0.98,
        "supports_grounding_mode_prediction": mode >= 0.95,
        "supports_self_anchor_authority": self_gain >= 0.40,
        "supports_wrong_forced_mode_control": wrong_drop >= 0.60,
        "no_context_control_hurts": no_context_gap >= 0.10,
        "shuffled_context_control_hurts": shuffled_drop >= 0.05,
        "mode_leakage_low": leakage <= 0.15,
    }


def fmt_metric(item: dict[str, Any], key: str) -> str:
    value = item.get(key, {}).get("mean")
    return "null" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Inferred Grounding Mode Authority Probe",
        "",
        "## Goal",
        "",
        "Test whether grounding mode can be inferred from compositional cue-feature bundles while semantic event recognition remains stable and action authority changes by grounding/self relevance.",
        "",
        "The main split holds out cue-feature combinations, not all cue atoms. Strict unseen cue tokens are diagnostic only.",
        "",
        "## Final-Test Results",
    ]
    for split in ["compositional_cue_split", "seen_cue_heldout_combinations"]:
        lines.extend([
            "",
            f"### {split}",
            "",
            "| Arm | Overall | Semantic | Mode | Authority | Reality Action | Nonreality Leakage | Margin | Self Gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for arm in ["explicit_mode_upper_bound", "inferred_context_mode", "no_context", "semantic_only_baseline"]:
            item = agg[split][arm]
            lines.append(
                f"| `{arm}` | `{fmt_metric(item, 'overall_accuracy')}` | `{fmt_metric(item, 'semantic_accuracy')}` "
                f"| `{fmt_metric(item, 'grounding_mode_accuracy')}` | `{fmt_metric(item, 'action_authority_accuracy')}` "
                f"| `{fmt_metric(item, 'reality_action_authority')}` | `{fmt_metric(item, 'nonreality_action_leakage')}` "
                f"| `{fmt_metric(item, 'grounding_authority_margin')}` | `{fmt_metric(item, 'self_anchor_gain')}` |"
            )
    controls = agg["compositional_cue_split"]["inferred_context_mode"].get("controls", {})
    strict = agg["compositional_cue_split"]["inferred_context_mode"].get("strict_unseen_cue_token_diagnostic", {})
    lines.extend([
        "",
        "## Controls",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ])
    for key, item in controls.items():
        lines.append(f"| `{key}` | `{item['mean']:.6f}` | `{item['std']:.6f}` |")
    lines.extend([
        "",
        "## Strict Unseen Cue Token Diagnostic",
        "",
        "This is not the main result. It is expected to fail or weaken because strict unseen cue atoms have no pretrained/compositional semantics.",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ])
    for key, item in strict.items():
        lines.append(f"| `{key}` | `{item['mean']:.6f}` | `{item['std']:.6f}` |")
    lines.extend([
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. No consciousness, biology, quantum, production, or natural-language-understanding claim.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    splits = {
        "compositional_cue_split": make_compositional_split(),
        "seen_cue_heldout_combinations": make_seen_combo_split(),
    }
    for seed in range(args.seeds):
        print(f"[inferred-grounding] seed={seed}", flush=True)
        for split_name, split in splits.items():
            records.extend(run_split(args, split_name, split, seed))
    summary = {
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "records": records,
        "aggregate": aggregate(records),
    }
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
