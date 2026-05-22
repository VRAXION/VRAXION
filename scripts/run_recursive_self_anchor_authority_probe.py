#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "recursive-self-anchor-authority"
REPORT_PATH = ROOT / "docs" / "research" / "RECURSIVE_SELF_ANCHOR_AUTHORITY_PROBE.md"
SUMMARY_NAME = "recursive_self_anchor_authority_summary.json"

ACTORS = ["dog", "cat", "snake", "self"]
ACTIONS = ["bit", "chased", "scared", "barked", "status_check"]
PATIENTS = ["me", "john", "child", "none"]
MODES = ["reality", "tv", "game", "dream", "memory"]
SELF_STATES = ["safe", "injured", "busy", "uncertain", "alert"]
INITIAL_SELF_STATES = ["safe", "busy", "uncertain"]
THREAT_ACTIONS = {"bit", "chased", "scared"}

CUE_FEATURES: dict[str, list[str]] = {
    "reality": ["now", "present", "physical", "in_room"],
    "tv": ["screen", "watched", "fictional", "episode"],
    "game": ["avatar", "level", "respawn", "controller"],
    "dream": ["asleep", "unreal", "nightmare", "remembered_after"],
    "memory": ["past", "remembered", "autobiographical", "yesterday"],
}
CUE_ATOMS = [atom for mode in MODES for atom in CUE_FEATURES[mode]]


@dataclass(frozen=True)
class SequenceExample:
    actor: str
    action: str
    patient: str
    mode: str
    cue: tuple[str, ...]
    initial_self: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive self-anchor authority toy probe.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--train-limit", type=int, default=900)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden = 32
        args.epochs = 40
        args.train_limit = 240
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cue_pairs(mode: str) -> list[tuple[str, str]]:
    return list(combinations(CUE_FEATURES[mode], 2))


def split_bundles() -> dict[str, dict[str, list[tuple[str, ...]]]]:
    split: dict[str, dict[str, list[tuple[str, ...]]]] = {"train": {}, "validation": {}, "final_test": {}}
    for mode in MODES:
        pairs = cue_pairs(mode)
        split["train"][mode] = [tuple(item) for item in pairs[:4]]
        split["validation"][mode] = [tuple(pairs[4])]
        split["final_test"][mode] = [tuple(pairs[5])]
    return split


def make_split() -> dict[str, list[SequenceExample]]:
    bundles = split_bundles()
    out: dict[str, list[SequenceExample]] = {}
    for split_name, by_mode in bundles.items():
        rows = []
        for actor in ["dog", "cat", "snake"]:
            for action in ["bit", "chased", "scared", "barked"]:
                for patient in ["me", "john", "child"]:
                    for mode in MODES:
                        for cue in by_mode[mode]:
                            for initial_self in INITIAL_SELF_STATES:
                                rows.append(SequenceExample(actor, action, patient, mode, cue, initial_self))
        out[split_name] = rows
    return out


def capped_train_examples(examples: list[SequenceExample], limit: int, seed: int) -> list[SequenceExample]:
    if limit <= 0 or len(examples) <= limit:
        return examples
    update = [ex for ex in examples if is_update_case(ex)]
    nonupdate = [ex for ex in examples if not is_update_case(ex)]
    rng = random.Random(seed)
    rng.shuffle(nonupdate)
    keep_nonupdate = max(0, limit - len(update))
    return update + nonupdate[:keep_nonupdate]


def next_self_state(ex: SequenceExample) -> str:
    if ex.mode == "reality" and ex.patient == "me":
        if ex.action == "bit":
            return "injured"
        if ex.action in {"chased", "scared"}:
            return "alert"
    return ex.initial_self


def is_update_case(ex: SequenceExample) -> bool:
    return next_self_state(ex) != ex.initial_self


def input_dim() -> int:
    return len(ACTORS) + len(ACTIONS) + len(PATIENTS) + len(CUE_ATOMS) + len(MODES) + len(SELF_STATES)


def encode_step(
    examples: list[SequenceExample],
    *,
    step: int,
    self_state_probs: torch.Tensor | None = None,
    input_kind: str = "cue",
    ablate_grounding: bool = False,
    ablate_self_anchor: bool = False,
    mode_override: str | None = None,
    shuffled_self: list[str] | None = None,
) -> torch.Tensor:
    rows = torch.zeros(len(examples), input_dim())
    actor_offset = 0
    action_offset = actor_offset + len(ACTORS)
    patient_offset = action_offset + len(ACTIONS)
    cue_offset = patient_offset + len(PATIENTS)
    mode_offset = cue_offset + len(CUE_ATOMS)
    self_offset = mode_offset + len(MODES)
    for i, ex in enumerate(examples):
        if step == 1:
            actor = ex.actor
            action = ex.action
            patient = ex.patient if not ablate_self_anchor else "john"
        else:
            actor = "self"
            action = "status_check"
            patient = "none"
        rows[i, actor_offset + ACTORS.index(actor)] = 1.0
        rows[i, action_offset + ACTIONS.index(action)] = 1.0
        rows[i, patient_offset + PATIENTS.index(patient)] = 1.0
        if not ablate_grounding:
            mode = mode_override or ex.mode
            if input_kind == "explicit_mode":
                rows[i, mode_offset + MODES.index(mode)] = 1.0
            else:
                cue = tuple(cue_pairs(mode)[0]) if mode_override is not None else ex.cue
                for atom in cue:
                    rows[i, cue_offset + CUE_ATOMS.index(atom)] = 1.0
        if self_state_probs is None:
            state = shuffled_self[i] if shuffled_self is not None else ex.initial_self
            rows[i, self_offset + SELF_STATES.index(state)] = 1.0
        else:
            rows[i, self_offset:self_offset + len(SELF_STATES)] = self_state_probs[i]
    return rows


def labels(examples: list[SequenceExample]) -> dict[str, torch.Tensor]:
    semantic = []
    mode = []
    authority = []
    next_state = []
    second_action = []
    for ex in examples:
        threat = ex.action in THREAT_ACTIONS
        semantic.append([float(ex.action == "bit"), float(threat)])
        mode.append(MODES.index(ex.mode))
        real_action = float(ex.mode == "reality" and threat and ex.patient == "me")
        story = float(ex.mode == "tv")
        game = float(ex.mode == "game")
        memory = float(ex.mode in {"dream", "memory"})
        self_relevance = float(ex.mode == "reality" and ex.patient == "me")
        help_other = float(ex.mode == "reality" and threat and ex.patient != "me")
        authority.append([real_action, story, game, memory, self_relevance, help_other])
        ns = next_self_state(ex)
        next_state.append(SELF_STATES.index(ns))
        second_action.append(float(ns in {"injured", "alert"}))
    return {
        "semantic": torch.tensor(semantic, dtype=torch.float32),
        "mode": torch.tensor(mode, dtype=torch.long),
        "authority": torch.tensor(authority, dtype=torch.float32),
        "next_state": torch.tensor(next_state, dtype=torch.long),
        "second_action": torch.tensor(second_action, dtype=torch.float32),
    }


class RecursiveSelfAnchorModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.trunk1 = nn.Sequential(nn.Linear(input_dim(), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.semantic = nn.Linear(hidden, 2)
        self.mode = nn.Linear(hidden, len(MODES))
        self.authority = nn.Linear(hidden, 6)
        self.next_state = nn.Linear(hidden, len(SELF_STATES))
        self.trunk2 = nn.Sequential(nn.Linear(input_dim(), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.second_action = nn.Linear(hidden, 1)

    def forward(
        self,
        examples: list[SequenceExample],
        *,
        input_kind: str,
        carry_mode: str,
        ablate_grounding: bool = False,
        ablate_self_anchor: bool = False,
        mode_override: str | None = None,
        shuffled_self: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        x1 = encode_step(
            examples,
            step=1,
            input_kind=input_kind,
            ablate_grounding=ablate_grounding,
            ablate_self_anchor=ablate_self_anchor,
            mode_override=mode_override,
            shuffled_self=shuffled_self,
        )
        h1 = self.trunk1(x1)
        next_logits = self.next_state(h1)
        if carry_mode == "recursive":
            state_for_step2 = F.softmax(next_logits, dim=-1)
        elif carry_mode == "no_update":
            state_for_step2 = None
        elif carry_mode == "oracle":
            state_for_step2 = F.one_hot(labels(examples)["next_state"], num_classes=len(SELF_STATES)).float()
        else:
            raise ValueError(carry_mode)
        x2 = encode_step(
            examples,
            step=2,
            self_state_probs=state_for_step2,
            input_kind=input_kind,
            ablate_grounding=ablate_grounding,
            ablate_self_anchor=ablate_self_anchor,
            mode_override=mode_override,
            shuffled_self=shuffled_self,
        )
        h2 = self.trunk2(x2)
        return {
            "semantic": self.semantic(h1),
            "mode": self.mode(h1),
            "authority": self.authority(h1),
            "next_state": next_logits,
            "second_action": self.second_action(h2).squeeze(-1),
        }


def second_action_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positives = target.sum().clamp(min=1.0)
    negatives = (target.numel() - target.sum()).clamp(min=1.0)
    pos_weight = negatives / positives
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def train_loss(model: RecursiveSelfAnchorModel, examples: list[SequenceExample], *, input_kind: str, carry_mode: str, arm: str) -> torch.Tensor:
    pred = model(
        examples,
        input_kind=input_kind,
        carry_mode=carry_mode,
        ablate_grounding=arm == "no_grounding_mode",
        ablate_self_anchor=arm == "no_self_anchor",
    )
    tgt = labels(examples)
    sem = F.binary_cross_entropy_with_logits(pred["semantic"], tgt["semantic"])
    mode = F.cross_entropy(pred["mode"], tgt["mode"])
    auth = F.binary_cross_entropy_with_logits(pred["authority"], tgt["authority"])
    ns = F.cross_entropy(pred["next_state"], tgt["next_state"])
    second = second_action_loss(pred["second_action"], tgt["second_action"])
    return sem + mode + auth + ns + second


def train_model(seed: int, split: dict[str, list[SequenceExample]], *, hidden: int, epochs: int, lr: float, input_kind: str, carry_mode: str, arm: str) -> RecursiveSelfAnchorModel:
    set_seed(seed)
    model = RecursiveSelfAnchorModel(hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = train_loss(model, split["train"], input_kind=input_kind, carry_mode=carry_mode, arm=arm)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def binary_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(((torch.sigmoid(logits) >= 0.5) == (y >= 0.5)).float().mean().item())


@torch.no_grad()
def evaluate(
    model: RecursiveSelfAnchorModel,
    examples: list[SequenceExample],
    *,
    input_kind: str,
    carry_mode: str,
    ablate_grounding: bool = False,
    ablate_self_anchor: bool = False,
    mode_override: str | None = None,
    shuffled_self: list[str] | None = None,
) -> dict[str, float]:
    pred = model(
        examples,
        input_kind=input_kind,
        carry_mode=carry_mode,
        ablate_grounding=ablate_grounding,
        ablate_self_anchor=ablate_self_anchor,
        mode_override=mode_override,
        shuffled_self=shuffled_self,
    )
    tgt = labels(examples)
    sem = binary_acc(pred["semantic"], tgt["semantic"])
    mode = float((pred["mode"].argmax(dim=-1) == tgt["mode"]).float().mean().item())
    auth = binary_acc(pred["authority"], tgt["authority"])
    ns = float((pred["next_state"].argmax(dim=-1) == tgt["next_state"]).float().mean().item())
    second = binary_acc(pred["second_action"], tgt["second_action"])
    update_idx = [i for i, ex in enumerate(examples) if is_update_case(ex)]
    nonupdate_idx = [i for i, ex in enumerate(examples) if not is_update_case(ex)]
    second_probs = torch.sigmoid(pred["second_action"])
    second_update_acc = binary_acc(pred["second_action"][update_idx], tgt["second_action"][update_idx]) if update_idx else 0.0
    second_nonupdate_acc = binary_acc(pred["second_action"][nonupdate_idx], tgt["second_action"][nonupdate_idx]) if nonupdate_idx else 0.0
    auth_probs = torch.sigmoid(pred["authority"])
    real_self = [i for i, ex in enumerate(examples) if ex.mode == "reality" and ex.patient == "me" and ex.action in THREAT_ACTIONS]
    nonreal_self = [i for i, ex in enumerate(examples) if ex.mode != "reality" and ex.patient == "me" and ex.action in THREAT_ACTIONS]
    me_bit = [i for i, ex in enumerate(examples) if ex.mode == "reality" and ex.patient == "me" and ex.action == "bit"]
    john_bit = [i for i, ex in enumerate(examples) if ex.mode == "reality" and ex.patient == "john" and ex.action == "bit"]
    real_action = float(auth_probs[real_self, 0].mean().item()) if real_self else 0.0
    leakage = float(auth_probs[nonreal_self, 0].mean().item()) if nonreal_self else 0.0
    self_gain = float(auth_probs[me_bit, 0].mean().item() - auth_probs[john_bit, 0].mean().item()) if me_bit and john_bit else 0.0
    return {
        "semantic_accuracy": sem,
        "grounding_mode_accuracy": mode,
        "authority_accuracy": auth,
        "next_self_state_accuracy": ns,
        "second_step_consistency": second,
        "second_step_update_case_accuracy": second_update_acc,
        "second_step_nonupdate_case_accuracy": second_nonupdate_acc,
        "update_case_need_help_prob": float(second_probs[update_idx].mean().item()) if update_idx else 0.0,
        "nonupdate_case_need_help_prob": float(second_probs[nonupdate_idx].mean().item()) if nonupdate_idx else 0.0,
        "reality_action_authority": real_action,
        "action_authority_leakage": leakage,
        "self_anchor_gain": self_gain,
        "overall_accuracy": float(np.mean([sem, mode, auth, ns, second])),
    }


@torch.no_grad()
def control_metrics(model: RecursiveSelfAnchorModel, examples: list[SequenceExample], *, input_kind: str, carry_mode: str, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    shuffled_states = [ex.initial_self for ex in examples]
    rng.shuffle(shuffled_states)
    base = evaluate(model, examples, input_kind=input_kind, carry_mode=carry_mode)
    forced_tv = evaluate(model, examples, input_kind=input_kind, carry_mode=carry_mode, mode_override="tv")
    shuffled = evaluate(model, examples, input_kind=input_kind, carry_mode=carry_mode, shuffled_self=shuffled_states)
    return {
        "wrong_mode_drop": base["reality_action_authority"] - forced_tv["reality_action_authority"],
        "wrong_mode_semantic_accuracy": forced_tv["semantic_accuracy"],
        "wrong_self_state_drop": base["second_step_update_case_accuracy"] - shuffled["second_step_update_case_accuracy"],
    }


def run_arm(args: argparse.Namespace, split: dict[str, list[SequenceExample]], seed: int, arm: str, input_kind: str, carry_mode: str) -> dict[str, Any]:
    model = train_model(seed, split, hidden=args.hidden, epochs=args.epochs, lr=args.learning_rate, input_kind=input_kind, carry_mode=carry_mode, arm=arm)
    final = evaluate(
        model,
        split["final_test"],
        input_kind=input_kind,
        carry_mode=carry_mode,
        ablate_grounding=arm == "no_grounding_mode",
        ablate_self_anchor=arm == "no_self_anchor",
    )
    row = {"arm": arm, "seed": seed, "final_test": final}
    if arm == "recursive_state_model":
        row["controls"] = control_metrics(model, split["final_test"], input_kind=input_kind, carry_mode=carry_mode, seed=seed + 991)
    return row


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in records}):
        rows = [row for row in records if row["arm"] == arm]
        keys = sorted({key for row in rows for key in row["final_test"]})
        item = {key: numeric_summary([float(row["final_test"][key]) for row in rows]) for key in keys}
        control_keys = sorted({key for row in rows for key in row.get("controls", {})})
        if control_keys:
            item["controls"] = {key: numeric_summary([float(row["controls"][key]) for row in rows if "controls" in row]) for key in control_keys}
        out[arm] = item
    return out


def mean_at(agg: dict[str, Any], arm: str, key: str) -> float:
    value = agg.get(arm, {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def control_mean(agg: dict[str, Any], arm: str, key: str) -> float:
    value = agg.get(arm, {}).get("controls", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    rec_update = mean_at(agg, "recursive_state_model", "second_step_update_case_accuracy")
    no_update = mean_at(agg, "no_recursive_update_baseline", "second_step_update_case_accuracy")
    static_update = mean_at(agg, "static_baseline_without_next_state", "second_step_update_case_accuracy")
    return {
        "supports_recursive_self_anchor_authority": (
            mean_at(agg, "recursive_state_model", "semantic_accuracy") >= 0.98
            and mean_at(agg, "recursive_state_model", "next_self_state_accuracy") >= 0.90
            and mean_at(agg, "recursive_state_model", "second_step_consistency") >= 0.85
            and rec_update - no_update >= 0.15
        ),
        "supports_next_self_state_update": mean_at(agg, "recursive_state_model", "next_self_state_accuracy") >= 0.90,
        "supports_second_step_recursive_use": rec_update - no_update >= 0.15 and rec_update - static_update >= 0.15,
        "supports_self_anchor_gain": mean_at(agg, "recursive_state_model", "self_anchor_gain") >= 0.40,
        "supports_wrong_mode_control": control_mean(agg, "recursive_state_model", "wrong_mode_drop") >= 0.60,
        "wrong_self_state_control_hurts": control_mean(agg, "recursive_state_model", "wrong_self_state_drop") >= 0.10,
    }


def fmt(item: dict[str, Any], key: str) -> str:
    value = item.get(key, {}).get("mean")
    return "null" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Recursive Self-Anchor Authority Probe",
        "",
        "## Goal",
        "",
        "Test whether grounded authority can update self-state, and whether the updated self-state affects the next decision.",
        "",
        "This run uses inferred compositional grounding cues because the inferred grounding phase passed.",
        "",
        "## Results",
        "",
        "| Arm | Overall | Semantic | Mode | Authority | Next Self | Step2 | Step2 Update Cases | Self Gain | Leakage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in [
        "recursive_state_model",
        "no_recursive_update_baseline",
        "static_baseline_without_next_state",
        "no_self_anchor",
        "no_grounding_mode",
        "oracle_next_state_baseline",
    ]:
        item = agg[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item, 'overall_accuracy')}` | `{fmt(item, 'semantic_accuracy')}` "
            f"| `{fmt(item, 'grounding_mode_accuracy')}` | `{fmt(item, 'authority_accuracy')}` "
            f"| `{fmt(item, 'next_self_state_accuracy')}` | `{fmt(item, 'second_step_consistency')}` "
            f"| `{fmt(item, 'second_step_update_case_accuracy')}` | `{fmt(item, 'self_anchor_gain')}` "
            f"| `{fmt(item, 'action_authority_leakage')}` |"
        )
    controls = agg["recursive_state_model"].get("controls", {})
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
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "This is a partial/negative recursive result. The model can infer grounding, action authority, and next self-state cleanly. However, the `no_recursive_update_baseline` and `static_baseline_without_next_state` also solve the update-case second-step metric. That means this task did not isolate recursive use of the updated self-state strongly enough.",
        "",
        "Safe readout:",
        "",
        "- next self-state update works in this toy",
        "- self-anchor and wrong-grounding controls work",
        "- recursive second-step authority is not proven",
        "- the next recursive probe needs harder counterfactual second-step pairs where the same second-step input requires different action depending only on the committed updated self-state",
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
    split = make_split()
    records: list[dict[str, Any]] = []
    arms = [
        ("recursive_state_model", "cue", "recursive"),
        ("no_recursive_update_baseline", "cue", "no_update"),
        ("static_baseline_without_next_state", "cue", "no_update"),
        ("no_self_anchor", "cue", "recursive"),
        ("no_grounding_mode", "cue", "recursive"),
        ("oracle_next_state_baseline", "cue", "oracle"),
    ]
    for seed in range(args.seeds):
        print(f"[recursive-self-anchor] seed={seed}", flush=True)
        seed_split = {
            "train": capped_train_examples(split["train"], args.train_limit, seed + 313),
            "validation": split["validation"],
            "final_test": split["final_test"],
        }
        for i, (arm, input_kind, carry_mode) in enumerate(arms):
            records.append(run_arm(args, seed_split, seed + 10_000 * i, arm, input_kind, carry_mode))
    summary = {
        "config": vars(args) | {"out_dir": str(args.out_dir), "grounding": "inferred_compositional_cues"},
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
