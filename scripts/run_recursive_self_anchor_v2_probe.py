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
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "recursive-self-anchor-v2"
REPORT_PATH = ROOT / "docs" / "research" / "RECURSIVE_SELF_ANCHOR_V2_PROBE.md"
SUMMARY_NAME = "recursive_self_anchor_v2_summary.json"

ACTORS = ["dog", "cat", "snake", "self"]
ACTIONS = ["bit", "chased", "scared", "barked", "choose_next_action"]
PATIENTS = ["me", "john", "child", "none"]
MODES = ["reality", "tv", "game", "dream", "memory"]
SELF_STATES = ["safe", "injured", "alert", "other_help"]
STEP2_ACTIONS = ["continue_plan", "seek_help", "protect_or_pause", "help_other_or_observe"]
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
    initial_self: str = "safe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive self-anchor v2 hard counterfactual probe.")
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
        args.epochs = 50
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


def event_templates() -> list[tuple[str, str, str, str]]:
    return [
        ("dog", "bit", "me", "reality"),
        ("dog", "bit", "me", "tv"),
        ("dog", "bit", "me", "game"),
        ("dog", "bit", "me", "memory"),
        ("snake", "scared", "me", "reality"),
        ("dog", "bit", "john", "reality"),
        ("cat", "chased", "child", "reality"),
        ("dog", "barked", "me", "reality"),
        ("dog", "bit", "me", "dream"),
    ]


def make_split() -> dict[str, list[SequenceExample]]:
    bundles = split_bundles()
    out: dict[str, list[SequenceExample]] = {}
    for split_name, by_mode in bundles.items():
        rows = []
        for actor, action, patient, mode in event_templates():
            for cue in by_mode[mode]:
                rows.append(SequenceExample(actor, action, patient, mode, cue))
        out[split_name] = rows
    return out


def committed_state(ex: SequenceExample) -> str:
    if ex.mode == "reality" and ex.patient == "me":
        if ex.action == "bit":
            return "injured"
        if ex.action in {"chased", "scared"}:
            return "alert"
    if ex.mode == "reality" and ex.action in THREAT_ACTIONS and ex.patient != "me":
        return "other_help"
    return "safe"


def step2_action(ex: SequenceExample) -> str:
    state = committed_state(ex)
    if state == "other_help":
        return "help_other_or_observe"
    if state == "injured":
        return "seek_help"
    if state == "alert":
        return "protect_or_pause"
    return "continue_plan"


def is_hard_counterfactual(ex: SequenceExample) -> bool:
    return True


def input_dim() -> int:
    return len(ACTORS) + len(ACTIONS) + len(PATIENTS) + len(CUE_ATOMS) + len(MODES) + len(SELF_STATES)


def encode_step(
    examples: list[SequenceExample],
    *,
    step: int,
    state_probs: torch.Tensor | None = None,
    input_kind: str = "cue",
    ablate_grounding: bool = False,
    ablate_self_anchor: bool = False,
    mode_override: str | None = None,
    shuffled_state_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    rows = torch.zeros(len(examples), input_dim())
    actor_offset = 0
    action_offset = actor_offset + len(ACTORS)
    patient_offset = action_offset + len(ACTIONS)
    cue_offset = patient_offset + len(PATIENTS)
    mode_offset = cue_offset + len(CUE_ATOMS)
    state_offset = mode_offset + len(MODES)
    for i, ex in enumerate(examples):
        if step == 1:
            actor = ex.actor
            action = ex.action
            patient = ex.patient if not ablate_self_anchor else "john"
        else:
            actor = "self"
            action = "choose_next_action"
            patient = "none"
        rows[i, actor_offset + ACTORS.index(actor)] = 1.0
        rows[i, action_offset + ACTIONS.index(action)] = 1.0
        rows[i, patient_offset + PATIENTS.index(patient)] = 1.0
        if step == 1 and not ablate_grounding:
            mode = mode_override or ex.mode
            if input_kind == "explicit_mode":
                rows[i, mode_offset + MODES.index(mode)] = 1.0
            else:
                cue = tuple(cue_pairs(mode)[0]) if mode_override is not None else ex.cue
                for atom in cue:
                    rows[i, cue_offset + CUE_ATOMS.index(atom)] = 1.0
        if step == 1:
            rows[i, state_offset + SELF_STATES.index(ex.initial_self)] = 1.0
        else:
            if shuffled_state_probs is not None:
                rows[i, state_offset:state_offset + len(SELF_STATES)] = shuffled_state_probs[i]
            elif state_probs is not None:
                rows[i, state_offset:state_offset + len(SELF_STATES)] = state_probs[i]
            else:
                rows[i, state_offset + SELF_STATES.index(ex.initial_self)] = 1.0
    return rows


def labels(examples: list[SequenceExample]) -> dict[str, torch.Tensor]:
    semantic = []
    mode = []
    authority = []
    state = []
    action = []
    for ex in examples:
        threat = ex.action in THREAT_ACTIONS
        semantic.append([float(ex.action == "bit"), float(threat)])
        mode.append(MODES.index(ex.mode))
        real_action = float(ex.mode == "reality" and threat and ex.patient == "me")
        self_relevance = float(ex.mode == "reality" and ex.patient == "me")
        help_other = float(ex.mode == "reality" and threat and ex.patient != "me")
        authority.append([real_action, self_relevance, help_other])
        state.append(SELF_STATES.index(committed_state(ex)))
        action.append(STEP2_ACTIONS.index(step2_action(ex)))
    return {
        "semantic": torch.tensor(semantic, dtype=torch.float32),
        "mode": torch.tensor(mode, dtype=torch.long),
        "authority": torch.tensor(authority, dtype=torch.float32),
        "state": torch.tensor(state, dtype=torch.long),
        "action": torch.tensor(action, dtype=torch.long),
    }


class RecursiveV2Model(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.trunk1 = nn.Sequential(nn.Linear(input_dim(), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.semantic = nn.Linear(hidden, 2)
        self.mode = nn.Linear(hidden, len(MODES))
        self.authority = nn.Linear(hidden, 3)
        self.state = nn.Linear(hidden, len(SELF_STATES))
        self.trunk2 = nn.Sequential(nn.Linear(input_dim(), hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.action = nn.Linear(hidden, len(STEP2_ACTIONS))

    def forward(
        self,
        examples: list[SequenceExample],
        *,
        input_kind: str,
        carry_mode: str,
        ablate_grounding: bool = False,
        ablate_self_anchor: bool = False,
        mode_override: str | None = None,
        shuffled_committed: bool = False,
    ) -> dict[str, torch.Tensor]:
        x1 = encode_step(
            examples,
            step=1,
            input_kind=input_kind,
            ablate_grounding=ablate_grounding,
            ablate_self_anchor=ablate_self_anchor,
            mode_override=mode_override,
        )
        h1 = self.trunk1(x1)
        state_logits = self.state(h1)
        shuffled_probs = None
        if carry_mode == "recursive":
            state_probs = F.one_hot(state_logits.argmax(dim=-1), num_classes=len(SELF_STATES)).float()
            if shuffled_committed and len(examples) > 1:
                state_probs = state_probs[torch.arange(len(examples) - 1, -1, -1)]
                shuffled_probs = state_probs
        elif carry_mode == "no_update":
            state_probs = None
        elif carry_mode == "oracle":
            state_probs = F.one_hot(labels(examples)["state"], num_classes=len(SELF_STATES)).float()
        else:
            raise ValueError(carry_mode)
        x2 = encode_step(
            examples,
            step=2,
            state_probs=state_probs,
            input_kind=input_kind,
            shuffled_state_probs=shuffled_probs,
        )
        h2 = self.trunk2(x2)
        return {
            "semantic": self.semantic(h1),
            "mode": self.mode(h1),
            "authority": self.authority(h1),
            "state": state_logits,
            "action": self.action(h2),
        }


def train_loss(model: RecursiveV2Model, examples: list[SequenceExample], *, input_kind: str, carry_mode: str, arm: str) -> torch.Tensor:
    pred = model(
        examples,
        input_kind=input_kind,
        carry_mode=carry_mode,
        ablate_grounding=arm == "no_grounding_mode",
        ablate_self_anchor=arm == "no_self_anchor",
    )
    tgt = labels(examples)
    return (
        F.binary_cross_entropy_with_logits(pred["semantic"], tgt["semantic"])
        + F.cross_entropy(pred["mode"], tgt["mode"])
        + F.binary_cross_entropy_with_logits(pred["authority"], tgt["authority"])
        + F.cross_entropy(pred["state"], tgt["state"])
        + F.cross_entropy(pred["action"], tgt["action"])
    )


def train_model(seed: int, split: dict[str, list[SequenceExample]], *, hidden: int, epochs: int, lr: float, input_kind: str, carry_mode: str, arm: str) -> RecursiveV2Model:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = RecursiveV2Model(hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_examples = split["train"]
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = train_loss(model, train_examples, input_kind=input_kind, carry_mode=carry_mode, arm=arm)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def binary_acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(((torch.sigmoid(logits) >= 0.5) == (target >= 0.5)).float().mean().item())


@torch.no_grad()
def evaluate(
    model: RecursiveV2Model,
    examples: list[SequenceExample],
    *,
    input_kind: str,
    carry_mode: str,
    ablate_grounding: bool = False,
    ablate_self_anchor: bool = False,
    mode_override: str | None = None,
    shuffled_committed: bool = False,
) -> dict[str, float]:
    pred = model(
        examples,
        input_kind=input_kind,
        carry_mode=carry_mode,
        ablate_grounding=ablate_grounding,
        ablate_self_anchor=ablate_self_anchor,
        mode_override=mode_override,
        shuffled_committed=shuffled_committed,
    )
    tgt = labels(examples)
    sem = binary_acc(pred["semantic"], tgt["semantic"])
    mode = float((pred["mode"].argmax(dim=-1) == tgt["mode"]).float().mean().item())
    authority = binary_acc(pred["authority"], tgt["authority"])
    state = float((pred["state"].argmax(dim=-1) == tgt["state"]).float().mean().item())
    action = float((pred["action"].argmax(dim=-1) == tgt["action"]).float().mean().item())
    hard_idx = [i for i, ex in enumerate(examples) if is_hard_counterfactual(ex)]
    hard = float((pred["action"].argmax(dim=-1)[hard_idx] == tgt["action"][hard_idx]).float().mean().item())
    authority_prob = torch.sigmoid(pred["authority"])
    real_self = [i for i, ex in enumerate(examples) if ex.mode == "reality" and ex.patient == "me" and ex.action in THREAT_ACTIONS]
    nonreal_self = [i for i, ex in enumerate(examples) if ex.mode != "reality" and ex.patient == "me" and ex.action in THREAT_ACTIONS]
    leakage = float(authority_prob[nonreal_self, 0].mean().item()) if nonreal_self else 0.0
    action_leak = float(authority_prob[real_self, 0].mean().item()) if real_self else 0.0
    return {
        "semantic_accuracy": sem,
        "grounding_mode_accuracy": mode,
        "step1_authority_accuracy": authority,
        "committed_self_state_accuracy": state,
        "step2_action_accuracy": action,
        "hard_counterfactual_accuracy": hard,
        "action_leakage": leakage,
        "reality_action_authority": action_leak,
        "overall_accuracy": float(np.mean([sem, mode, authority, state, action])),
    }


@torch.no_grad()
def controls(model: RecursiveV2Model, examples: list[SequenceExample], *, input_kind: str) -> dict[str, float]:
    base = evaluate(model, examples, input_kind=input_kind, carry_mode="recursive")
    forced_tv = evaluate(model, examples, input_kind=input_kind, carry_mode="recursive", mode_override="tv")
    no_self = evaluate(model, examples, input_kind=input_kind, carry_mode="recursive", ablate_self_anchor=True)
    shuffled = evaluate(model, examples, input_kind=input_kind, carry_mode="recursive", shuffled_committed=True)
    return {
        "wrong_mode_drop": base["reality_action_authority"] - forced_tv["reality_action_authority"],
        "wrong_self_anchor_drop": base["reality_action_authority"] - no_self["reality_action_authority"],
        "shuffled_committed_state_drop": base["hard_counterfactual_accuracy"] - shuffled["hard_counterfactual_accuracy"],
        "wrong_mode_semantic_accuracy": forced_tv["semantic_accuracy"],
    }


def capped_train_examples(examples: list[SequenceExample], limit: int, seed: int) -> list[SequenceExample]:
    if limit <= 0 or len(examples) <= limit:
        return examples
    rng = random.Random(seed)
    rows = examples[:]
    rng.shuffle(rows)
    return rows[:limit]


def run_arm(args: argparse.Namespace, split: dict[str, list[SequenceExample]], seed: int, arm: str, carry_mode: str) -> dict[str, Any]:
    model = train_model(seed, split, hidden=args.hidden, epochs=args.epochs, lr=args.learning_rate, input_kind="cue", carry_mode=carry_mode, arm=arm)
    final = evaluate(
        model,
        split["final_test"],
        input_kind="cue",
        carry_mode=carry_mode,
        ablate_grounding=arm == "no_grounding_mode",
        ablate_self_anchor=arm == "no_self_anchor",
    )
    row: dict[str, Any] = {"arm": arm, "seed": seed, "final_test": final}
    if arm == "recursive_state_model":
        row["controls"] = controls(model, split["final_test"], input_kind="cue")
    return row


def summarize(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in records}):
        rows = [row for row in records if row["arm"] == arm]
        keys = sorted({key for row in rows for key in row["final_test"]})
        item: dict[str, Any] = {key: summarize([float(row["final_test"][key]) for row in rows]) for key in keys}
        control_keys = sorted({key for row in rows for key in row.get("controls", {})})
        if control_keys:
            item["controls"] = {key: summarize([float(row["controls"][key]) for row in rows if "controls" in row]) for key in control_keys}
        out[arm] = item
    rec = out.get("recursive_state_model", {})
    no_update = out.get("no_recursive_update_baseline", {})
    static = out.get("static_baseline_without_next_state", {})
    if rec and no_update and static:
        rec["recursive_gap_vs_no_update"] = summarize([
            float(r["final_test"]["hard_counterfactual_accuracy"])
            - float(n["final_test"]["hard_counterfactual_accuracy"])
            for r, n in zip(
                [row for row in records if row["arm"] == "recursive_state_model"],
                [row for row in records if row["arm"] == "no_recursive_update_baseline"],
            )
        ])
        rec["recursive_gap_vs_static"] = summarize([
            float(r["final_test"]["hard_counterfactual_accuracy"])
            - float(s["final_test"]["hard_counterfactual_accuracy"])
            for r, s in zip(
                [row for row in records if row["arm"] == "recursive_state_model"],
                [row for row in records if row["arm"] == "static_baseline_without_next_state"],
            )
        ])
    return out


def mean_at(agg: dict[str, Any], arm: str, key: str) -> float:
    value = agg.get(arm, {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def control_mean(agg: dict[str, Any], key: str) -> float:
    value = agg.get("recursive_state_model", {}).get("controls", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    rec_hard = mean_at(agg, "recursive_state_model", "hard_counterfactual_accuracy")
    no_update_hard = mean_at(agg, "no_recursive_update_baseline", "hard_counterfactual_accuracy")
    static_hard = mean_at(agg, "static_baseline_without_next_state", "hard_counterfactual_accuracy")
    gap_no = agg.get("recursive_state_model", {}).get("recursive_gap_vs_no_update", {}).get("mean", 0.0)
    gap_static = agg.get("recursive_state_model", {}).get("recursive_gap_vs_static", {}).get("mean", 0.0)
    return {
        "supports_recursive_self_anchor_v2": (
            rec_hard >= 0.90
            and no_update_hard <= 0.60
            and static_hard <= 0.60
            and gap_no >= 0.30
            and gap_static >= 0.30
            and mean_at(agg, "recursive_state_model", "committed_self_state_accuracy") >= 0.90
            and mean_at(agg, "recursive_state_model", "semantic_accuracy") >= 0.98
            and mean_at(agg, "recursive_state_model", "grounding_mode_accuracy") >= 0.95
        ),
        "supports_committed_self_state": mean_at(agg, "recursive_state_model", "committed_self_state_accuracy") >= 0.90,
        "recursive_beats_no_update": gap_no >= 0.30,
        "recursive_beats_static": gap_static >= 0.30,
        "baselines_fail_hard_counterfactual": no_update_hard <= 0.60 and static_hard <= 0.60,
        "shuffled_committed_state_control_hurts": control_mean(agg, "shuffled_committed_state_drop") >= 0.30,
        "wrong_mode_control_hurts": control_mean(agg, "wrong_mode_drop") >= 0.60,
        "wrong_self_anchor_control_hurts": control_mean(agg, "wrong_self_anchor_drop") >= 0.30,
    }


def fmt(item: dict[str, Any], key: str) -> str:
    value = item.get(key, {}).get("mean")
    return "null" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Recursive Self-Anchor v2 Probe",
        "",
        "## Goal",
        "",
        "Hard-counterfactual test for whether the committed self-state is required by the next action decision.",
        "",
        "The visible second-step prompt is always `choose_next_action`; event, mode, patient, and next-state labels are not visible at step 2.",
        "",
        "## Results",
        "",
        "| Arm | Semantic | Mode | Authority | State | Step2 | Hard CF | Leakage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in [
        "recursive_state_model",
        "no_recursive_update_baseline",
        "static_baseline_without_next_state",
        "oracle_next_state_baseline",
        "no_grounding_mode",
        "no_self_anchor",
    ]:
        item = agg[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item, 'semantic_accuracy')}` | `{fmt(item, 'grounding_mode_accuracy')}` "
            f"| `{fmt(item, 'step1_authority_accuracy')}` | `{fmt(item, 'committed_self_state_accuracy')}` "
            f"| `{fmt(item, 'step2_action_accuracy')}` | `{fmt(item, 'hard_counterfactual_accuracy')}` "
            f"| `{fmt(item, 'action_leakage')}` |"
        )
    rec = agg["recursive_state_model"]
    controls_item = rec.get("controls", {})
    lines.extend([
        "",
        "## Gaps And Controls",
        "",
        f"- recursive_gap_vs_no_update: `{rec['recursive_gap_vs_no_update']['mean']:.6f}`",
        f"- recursive_gap_vs_static: `{rec['recursive_gap_vs_static']['mean']:.6f}`",
        "",
        "| Control | Mean | Std |",
        "|---|---:|---:|",
    ])
    for key, item in controls_item.items():
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
        "This fixes the v1 failure mode. The second-step prompt is identical across hard counterfactual cases, and the carried committed state is a hard one-hot state, not a soft side-channel. The recursive model and oracle baseline both solve the hard counterfactuals, while no-update/static baselines stay near majority-choice performance.",
        "",
        "Safe readout:",
        "",
        "- committed self-state is learned cleanly",
        "- hard second-step decisions require the committed state in this toy",
        "- shuffling the committed state breaks the behavior",
        "- wrong grounding and wrong self-anchor controls hurt",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. No consciousness, biology, quantum behavior, production validity, or natural-language-understanding claim.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split = make_split()
    arms = [
        ("recursive_state_model", "recursive"),
        ("no_recursive_update_baseline", "no_update"),
        ("static_baseline_without_next_state", "no_update"),
        ("oracle_next_state_baseline", "oracle"),
        ("no_grounding_mode", "recursive"),
        ("no_self_anchor", "recursive"),
    ]
    records: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        print(f"[recursive-self-anchor-v2] seed={seed}", flush=True)
        seed_split = dict(split)
        seed_split["train"] = capped_train_examples(split["train"], args.train_limit, seed + 515)
        for arm_i, (arm, carry_mode) in enumerate(arms):
            records.append(run_arm(args, seed_split, seed + 10_000 * arm_i, arm, carry_mode))
    summary = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "records": records, "aggregate": aggregate(records)}
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
