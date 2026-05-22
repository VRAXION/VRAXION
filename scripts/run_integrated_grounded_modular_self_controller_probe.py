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

import run_composable_skill_preservation_probe as skill


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "integrated-grounded-modular-self-controller"
REPORT_PATH = ROOT / "docs" / "research" / "INTEGRATED_GROUNDED_MODULAR_SELF_CONTROLLER_PROBE.md"
SUMMARY_NAME = "integrated_grounded_modular_self_controller_summary.json"

ACTORS = ["dog", "cat", "snake"]
EVENT_ACTIONS = ["bit", "chased", "scared", "barked"]
PATIENTS = ["me", "john", "child"]
MODES = ["reality", "tv", "game", "dream", "memory"]
SELF_STATES = ["safe", "injured", "alert", "other_help", "story_observe"]
ACTION_MODULES = ["continue_plan", "seek_help", "protect_or_pause", "help_other_or_observe", "observe_story"]
THREAT_ACTIONS = {"bit", "chased", "scared"}
ARITH_PROGRAMS = ["add", "mul", "add_then_mul", "mul_then_add"]

CUE_FEATURES: dict[str, list[str]] = {
    "reality": ["now", "present", "physical", "in_room"],
    "tv": ["screen", "watched", "fictional", "episode"],
    "game": ["avatar", "level", "respawn", "controller"],
    "dream": ["asleep", "unreal", "nightmare", "remembered_after"],
    "memory": ["past", "remembered", "autobiographical", "yesterday"],
}
CUE_ATOMS = [atom for mode in MODES for atom in CUE_FEATURES[mode]]


@dataclass(frozen=True)
class EventExample:
    actor: str
    action: str
    patient: str
    mode: str
    cue: tuple[str, ...]
    initial_self: str = "safe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated grounded modular self-controller probe.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--controller-hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--module-epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden = 32
        args.controller_hidden = 16
        args.epochs = 50
        args.module_epochs = 120
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


def make_event_split() -> dict[str, list[EventExample]]:
    bundles = split_bundles()
    out: dict[str, list[EventExample]] = {}
    for split_name, by_mode in bundles.items():
        rows = []
        for actor, action, patient, mode in event_templates():
            for cue in by_mode[mode]:
                rows.append(EventExample(actor, action, patient, mode, cue))
        out[split_name] = rows
    return out


def committed_state(ex: EventExample) -> str:
    if ex.mode == "reality" and ex.patient == "me":
        if ex.action == "bit":
            return "injured"
        if ex.action in {"chased", "scared"}:
            return "alert"
    if ex.mode == "reality" and ex.action in THREAT_ACTIONS and ex.patient != "me":
        return "other_help"
    if ex.mode == "tv" and ex.action in THREAT_ACTIONS:
        return "story_observe"
    return "safe"


def step2_action(ex: EventExample) -> str:
    state = committed_state(ex)
    if state == "injured":
        return "seek_help"
    if state == "alert":
        return "protect_or_pause"
    if state == "other_help":
        return "help_other_or_observe"
    if state == "story_observe":
        return "observe_story"
    return "continue_plan"


def event_input_dim() -> int:
    return len(ACTORS) + len(EVENT_ACTIONS) + len(PATIENTS) + len(CUE_ATOMS) + len(MODES) + len(SELF_STATES)


def action_controller_dim() -> int:
    return 1 + len(SELF_STATES)


def encode_event_step1(
    examples: list[EventExample],
    *,
    ablate_grounding: bool = False,
    ablate_self_anchor: bool = False,
    mode_override: str | None = None,
) -> torch.Tensor:
    rows = torch.zeros(len(examples), event_input_dim())
    actor_offset = 0
    action_offset = actor_offset + len(ACTORS)
    patient_offset = action_offset + len(EVENT_ACTIONS)
    cue_offset = patient_offset + len(PATIENTS)
    mode_offset = cue_offset + len(CUE_ATOMS)
    state_offset = mode_offset + len(MODES)
    for i, ex in enumerate(examples):
        patient = ex.patient if not ablate_self_anchor else "john"
        rows[i, actor_offset + ACTORS.index(ex.actor)] = 1.0
        rows[i, action_offset + EVENT_ACTIONS.index(ex.action)] = 1.0
        rows[i, patient_offset + PATIENTS.index(patient)] = 1.0
        if not ablate_grounding:
            mode = mode_override or ex.mode
            cue = tuple(cue_pairs(mode)[0]) if mode_override is not None else ex.cue
            for atom in cue:
                rows[i, cue_offset + CUE_ATOMS.index(atom)] = 1.0
        rows[i, state_offset + SELF_STATES.index(ex.initial_self)] = 1.0
    return rows


def encode_action_step2(state_probs: torch.Tensor | None, n: int, *, static_safe: bool = False) -> torch.Tensor:
    rows = torch.zeros(n, action_controller_dim())
    rows[:, 0] = 1.0
    if state_probs is not None:
        rows[:, 1:] = state_probs
    elif static_safe:
        rows[:, 1 + SELF_STATES.index("safe")] = 1.0
    return rows


def event_labels(examples: list[EventExample]) -> dict[str, torch.Tensor]:
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
        story_authority = float(ex.mode == "tv" and threat)
        authority.append([real_action, self_relevance, help_other, story_authority])
        state.append(SELF_STATES.index(committed_state(ex)))
        action.append(ACTION_MODULES.index(step2_action(ex)))
    return {
        "semantic": torch.tensor(semantic, dtype=torch.float32),
        "mode": torch.tensor(mode, dtype=torch.long),
        "authority": torch.tensor(authority, dtype=torch.float32),
        "state": torch.tensor(state, dtype=torch.long),
        "action": torch.tensor(action, dtype=torch.long),
    }


def all_arithmetic_examples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[str, int, int, int]]]:
    xs = []
    ys = []
    programs = []
    meta = []
    for op in ARITH_PROGRAMS:
        for a in range(skill.P):
            for b in range(skill.P):
                for c in range(skill.P):
                    xs.append(skill.encode_example(op, a, b, c))
                    ys.append(skill.label(op, a, b, c))
                    programs.append(ARITH_PROGRAMS.index(op))
                    meta.append((op, a, b, c))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), torch.tensor(programs, dtype=torch.long), meta


@torch.no_grad()
def module_argmax(module: skill.PrimitiveModule, a: int, b: int) -> int:
    x = torch.zeros(1, 2 * skill.P)
    x[0, a] = 1.0
    x[0, skill.P + b] = 1.0
    return int(module(x).argmax(dim=-1).item())


@torch.no_grad()
def arithmetic_candidate_logits(
    add_mod: skill.PrimitiveModule,
    mul_mod: skill.PrimitiveModule,
    meta: list[tuple[str, int, int, int]],
) -> torch.Tensor:
    logits = torch.full((len(meta), len(ARITH_PROGRAMS), skill.P), -6.0)
    for i, (_, a, b, c) in enumerate(meta):
        add_ab = module_argmax(add_mod, a, b)
        mul_ab = module_argmax(mul_mod, a, b)
        values = [
            add_ab,
            mul_ab,
            module_argmax(mul_mod, add_ab, c),
            module_argmax(add_mod, mul_ab, c),
        ]
        for j, value in enumerate(values):
            logits[i, j, value] = 6.0
    return logits


def action_candidate_logits(n: int) -> torch.Tensor:
    logits = torch.full((n, len(ACTION_MODULES), len(ACTION_MODULES)), -6.0)
    for i in range(n):
        for j in range(len(ACTION_MODULES)):
            logits[i, j, j] = 6.0
    return logits


def train_frozen_arithmetic_modules(seed: int, args: argparse.Namespace) -> tuple[skill.PrimitiveModule, skill.PrimitiveModule]:
    set_seed(seed)
    add_x, add_y = skill.make_binary_examples("add")
    mul_x, mul_y = skill.make_binary_examples("mul")
    add_mod = skill.PrimitiveModule(args.hidden)
    mul_mod = skill.PrimitiveModule(args.hidden)
    skill.train_ce(add_mod, add_x, add_y, epochs=args.module_epochs, lr=args.learning_rate)
    skill.train_ce(mul_mod, mul_x, mul_y, epochs=args.module_epochs, lr=args.learning_rate)
    for param in add_mod.parameters():
        param.requires_grad_(False)
    for param in mul_mod.parameters():
        param.requires_grad_(False)
    return add_mod, mul_mod


class IntegratedModel(nn.Module):
    def __init__(self, hidden: int, controller_hidden: int) -> None:
        super().__init__()
        self.event_trunk = nn.Sequential(
            nn.Linear(event_input_dim(), hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.semantic = nn.Linear(hidden, 2)
        self.mode = nn.Linear(hidden, len(MODES))
        self.authority = nn.Linear(hidden, 4)
        self.state = nn.Linear(hidden, len(SELF_STATES))
        self.action_controller = nn.Sequential(
            nn.Linear(action_controller_dim(), controller_hidden),
            nn.Tanh(),
            nn.Linear(controller_hidden, len(ACTION_MODULES)),
        )
        self.arithmetic_controller = nn.Sequential(
            nn.Linear(len(skill.OPS) + 3 * skill.P, controller_hidden),
            nn.Tanh(),
            nn.Linear(controller_hidden, len(ARITH_PROGRAMS)),
        )

    def step1(
        self,
        examples: list[EventExample],
        *,
        ablate_grounding: bool = False,
        ablate_self_anchor: bool = False,
        mode_override: str | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.event_trunk(
            encode_event_step1(
                examples,
                ablate_grounding=ablate_grounding,
                ablate_self_anchor=ablate_self_anchor,
                mode_override=mode_override,
            )
        )
        return {
            "semantic": self.semantic(h),
            "mode": self.mode(h),
            "authority": self.authority(h),
            "state": self.state(h),
        }

    def action_logits(
        self,
        examples: list[EventExample],
        *,
        carry_mode: str,
        ablate_grounding: bool = False,
        ablate_self_anchor: bool = False,
        mode_override: str | None = None,
        shuffled_committed: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        step1 = self.step1(
            examples,
            ablate_grounding=ablate_grounding,
            ablate_self_anchor=ablate_self_anchor,
            mode_override=mode_override,
        )
        n = len(examples)
        if carry_mode == "recursive":
            state_probs = F.one_hot(step1["state"].argmax(dim=-1), num_classes=len(SELF_STATES)).float()
            if shuffled_committed and n > 1:
                state_probs = state_probs[torch.arange(n - 1, -1, -1)]
            static_safe = False
        elif carry_mode == "oracle":
            state_probs = F.one_hot(event_labels(examples)["state"], num_classes=len(SELF_STATES)).float()
            static_safe = False
        elif carry_mode == "no_commit":
            state_probs = None
            static_safe = False
        elif carry_mode == "static":
            state_probs = None
            static_safe = True
        else:
            raise ValueError(carry_mode)
        controller_input = encode_action_step2(state_probs, n, static_safe=static_safe)
        weights = F.softmax(self.action_controller(controller_input), dim=-1)
        logits = torch.einsum("np,npk->nk", weights, action_candidate_logits(n))
        return step1, logits, weights

    def arithmetic_logits(self, x: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.arithmetic_controller(x), dim=-1)
        logits = torch.einsum("np,npk->nk", weights, candidates)
        return logits, weights


def event_loss(
    model: IntegratedModel,
    examples: list[EventExample],
    *,
    carry_mode: str,
    arm: str,
) -> torch.Tensor:
    step1, action_logits, _ = model.action_logits(
        examples,
        carry_mode=carry_mode,
        ablate_grounding=arm == "no_grounding_control",
    )
    target = event_labels(examples)
    return (
        F.binary_cross_entropy_with_logits(step1["semantic"], target["semantic"])
        + F.cross_entropy(step1["mode"], target["mode"])
        + F.binary_cross_entropy_with_logits(step1["authority"], target["authority"])
        + F.cross_entropy(step1["state"], target["state"])
        + F.cross_entropy(action_logits, target["action"])
    )


def train_integrated_model(
    seed: int,
    args: argparse.Namespace,
    event_split: dict[str, list[EventExample]],
    *,
    carry_mode: str,
    arm: str,
    arithmetic_x: torch.Tensor,
    arithmetic_y: torch.Tensor,
    arithmetic_candidates: torch.Tensor,
) -> IntegratedModel:
    set_seed(seed)
    model = IntegratedModel(args.hidden, args.controller_hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_events = event_split["train"]
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        arith_logits, _ = model.arithmetic_logits(arithmetic_x, arithmetic_candidates)
        loss = (
            event_loss(model, train_events, carry_mode=carry_mode, arm=arm)
            + F.cross_entropy(arith_logits, arithmetic_y)
        )
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def binary_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(((torch.sigmoid(logits) >= 0.5) == (y >= 0.5)).float().mean().item())


@torch.no_grad()
def evaluate_events(
    model: IntegratedModel,
    examples: list[EventExample],
    *,
    carry_mode: str,
    ablate_grounding: bool = False,
    mode_override: str | None = None,
    shuffled_committed: bool = False,
) -> dict[str, float]:
    step1, action_logits, _ = model.action_logits(
        examples,
        carry_mode=carry_mode,
        ablate_grounding=ablate_grounding,
        mode_override=mode_override,
        shuffled_committed=shuffled_committed,
    )
    target = event_labels(examples)
    semantic = binary_acc(step1["semantic"], target["semantic"])
    mode = float((step1["mode"].argmax(dim=-1) == target["mode"]).float().mean().item())
    authority = binary_acc(step1["authority"], target["authority"])
    state = float((step1["state"].argmax(dim=-1) == target["state"]).float().mean().item())
    action = float((action_logits.argmax(dim=-1) == target["action"]).float().mean().item())
    authority_prob = torch.sigmoid(step1["authority"])
    real_self = [i for i, ex in enumerate(examples) if ex.mode == "reality" and ex.patient == "me" and ex.action in THREAT_ACTIONS]
    nonreal_self = [i for i, ex in enumerate(examples) if ex.mode != "reality" and ex.patient == "me" and ex.action in THREAT_ACTIONS]
    reality_action = float(authority_prob[real_self, 0].mean().item()) if real_self else 0.0
    nonreality_leakage = float(authority_prob[nonreal_self, 0].mean().item()) if nonreal_self else 0.0
    return {
        "semantic_accuracy": semantic,
        "grounding_mode_accuracy": mode,
        "step1_authority_accuracy": authority,
        "committed_self_state_accuracy": state,
        "step2_action_accuracy": action,
        "hard_counterfactual_action_accuracy": action,
        "reality_action_authority": reality_action,
        "nonreality_action_leakage": nonreality_leakage,
        "grounding_authority_margin": reality_action - nonreality_leakage,
        "overall_event_accuracy": float(np.mean([semantic, mode, authority, state, action])),
    }


@torch.no_grad()
def evaluate_arithmetic(
    model: IntegratedModel,
    add_mod: skill.PrimitiveModule,
    mul_mod: skill.PrimitiveModule,
    arithmetic_x: torch.Tensor,
    arithmetic_y: torch.Tensor,
    arithmetic_program_y: torch.Tensor,
    arithmetic_candidates: torch.Tensor,
) -> dict[str, float]:
    primitive_metrics = skill.frozen_interpreter_metrics(add_mod, mul_mod)
    logits, weights = model.arithmetic_logits(arithmetic_x, arithmetic_candidates)
    pred = logits.argmax(dim=-1)
    program_pred = weights.argmax(dim=-1)
    primitive_mask = torch.isin(arithmetic_program_y, torch.tensor([0, 1]))
    composite_mask = ~primitive_mask
    return {
        "primitive_accuracy_before": primitive_metrics["primitive_accuracy_after_composition"],
        "primitive_accuracy_after": primitive_metrics["primitive_accuracy_after_composition"],
        "primitive_drift": primitive_metrics["primitive_drift"],
        "arithmetic_overall_accuracy": float((pred == arithmetic_y).float().mean().item()),
        "arithmetic_primitive_task_accuracy": float((pred[primitive_mask] == arithmetic_y[primitive_mask]).float().mean().item()),
        "composition_accuracy": float((pred[composite_mask] == arithmetic_y[composite_mask]).float().mean().item()),
        "controller_program_accuracy": float((program_pred == arithmetic_program_y).float().mean().item()),
    }


@torch.no_grad()
def event_controls(model: IntegratedModel, examples: list[EventExample]) -> dict[str, float]:
    base = evaluate_events(model, examples, carry_mode="recursive")
    shuffled = evaluate_events(model, examples, carry_mode="recursive", shuffled_committed=True)
    available = {(ex.actor, ex.action, ex.patient, ex.mode) for ex in examples}
    contrast_examples = [
        ex for ex in examples
        if ex.mode == "reality"
        and ex.patient == "me"
        and ex.action in THREAT_ACTIONS
        and (ex.actor, ex.action, ex.patient, "tv") in available
    ]
    if not contrast_examples:
        contrast_examples = [
            ex for ex in examples
            if ex.mode == "reality" and ex.patient == "me" and ex.action in THREAT_ACTIONS
        ]
    base_contrast = evaluate_events(model, contrast_examples, carry_mode="recursive") if contrast_examples else base
    forced_tv = evaluate_events(model, contrast_examples, carry_mode="recursive", mode_override="tv") if contrast_examples else base
    return {
        "shuffled_committed_state_drop": base["hard_counterfactual_action_accuracy"] - shuffled["hard_counterfactual_action_accuracy"],
        "wrong_mode_drop": base_contrast["reality_action_authority"] - forced_tv["reality_action_authority"],
        "wrong_mode_semantic_accuracy": forced_tv["semantic_accuracy"],
    }


def run_integrated_arm(
    args: argparse.Namespace,
    event_split: dict[str, list[EventExample]],
    seed: int,
    arm: str,
    carry_mode: str,
) -> tuple[dict[str, Any], IntegratedModel | None]:
    add_mod, mul_mod = train_frozen_arithmetic_modules(seed + 1_000, args)
    arithmetic_x, arithmetic_y, arithmetic_program_y, meta = all_arithmetic_examples()
    arithmetic_candidates = arithmetic_candidate_logits(add_mod, mul_mod, meta)
    model = train_integrated_model(
        seed + 2_000,
        args,
        event_split,
        carry_mode=carry_mode,
        arm=arm,
        arithmetic_x=arithmetic_x,
        arithmetic_y=arithmetic_y,
        arithmetic_candidates=arithmetic_candidates,
    )
    event_metrics = evaluate_events(
        model,
        event_split["final_test"],
        carry_mode=carry_mode,
        ablate_grounding=arm == "no_grounding_control",
    )
    arithmetic_metrics = evaluate_arithmetic(
        model,
        add_mod,
        mul_mod,
        arithmetic_x,
        arithmetic_y,
        arithmetic_program_y,
        arithmetic_candidates,
    )
    row: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "final_test": event_metrics | arithmetic_metrics,
    }
    if arm == "integrated_recursive_controller":
        row["controls"] = event_controls(model, event_split["final_test"])
    return row, model if arm == "integrated_recursive_controller" else None


def shared_end_to_end_metrics(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    metrics = skill.train_shared(
        seed + 60_000,
        hidden=args.controller_hidden,
        primitive_epochs=args.module_epochs,
        composition_epochs=args.epochs,
        lr=args.learning_rate,
        replay=False,
    )
    metrics["primitive_accuracy_before"] = metrics["primitive_accuracy_before_composition"]
    metrics["primitive_accuracy_after"] = metrics["primitive_accuracy_after_composition"]
    return {"arm": "shared_end_to_end_no_freeze", "seed": seed, "final_test": metrics}


def frozen_learned_reference_metrics(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    add_mod, mul_mod = train_frozen_arithmetic_modules(seed + 70_000, args)
    arithmetic_x, arithmetic_y, arithmetic_program_y, meta = all_arithmetic_examples()
    candidates = arithmetic_candidate_logits(add_mod, mul_mod, meta)
    set_seed(seed + 71_000)
    controller = nn.Sequential(
        nn.Linear(len(skill.OPS) + 3 * skill.P, args.controller_hidden),
        nn.Tanh(),
        nn.Linear(args.controller_hidden, len(ARITH_PROGRAMS)),
    )
    opt = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        weights = F.softmax(controller(arithmetic_x), dim=-1)
        logits = torch.einsum("np,npk->nk", weights, candidates)
        loss = F.cross_entropy(logits, arithmetic_y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        weights = F.softmax(controller(arithmetic_x), dim=-1)
        logits = torch.einsum("np,npk->nk", weights, candidates)
        pred = logits.argmax(dim=-1)
        program_pred = weights.argmax(dim=-1)
        primitive_mask = torch.isin(arithmetic_program_y, torch.tensor([0, 1]))
        composite_mask = ~primitive_mask
        primitive_metrics = skill.frozen_interpreter_metrics(add_mod, mul_mod)
        metrics = {
            "primitive_accuracy_before": primitive_metrics["primitive_accuracy_after_composition"],
            "primitive_accuracy_after": primitive_metrics["primitive_accuracy_after_composition"],
            "primitive_drift": primitive_metrics["primitive_drift"],
            "arithmetic_overall_accuracy": float((pred == arithmetic_y).float().mean().item()),
            "arithmetic_primitive_task_accuracy": float((pred[primitive_mask] == arithmetic_y[primitive_mask]).float().mean().item()),
            "composition_accuracy": float((pred[composite_mask] == arithmetic_y[composite_mask]).float().mean().item()),
            "controller_program_accuracy": float((program_pred == arithmetic_program_y).float().mean().item()),
        }
    return {"arm": "frozen_learned_controller_reference", "seed": seed, "final_test": metrics}


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
    rec = out.get("integrated_recursive_controller", {})
    no_commit = out.get("no_committed_state_baseline", {})
    static = out.get("static_without_commit_baseline", {})
    oracle = out.get("oracle_committed_state_baseline", {})
    if rec and no_commit:
        rec["recursive_gap_vs_no_commit"] = summarize([
            float(r["final_test"]["hard_counterfactual_action_accuracy"])
            - float(n["final_test"]["hard_counterfactual_action_accuracy"])
            for r, n in zip(
                [row for row in records if row["arm"] == "integrated_recursive_controller"],
                [row for row in records if row["arm"] == "no_committed_state_baseline"],
            )
        ])
    if rec and static:
        rec["recursive_gap_vs_static"] = summarize([
            float(r["final_test"]["hard_counterfactual_action_accuracy"])
            - float(s["final_test"]["hard_counterfactual_action_accuracy"])
            for r, s in zip(
                [row for row in records if row["arm"] == "integrated_recursive_controller"],
                [row for row in records if row["arm"] == "static_without_commit_baseline"],
            )
        ])
    if rec and oracle:
        rec["oracle_gap"] = summarize([
            float(o["final_test"]["hard_counterfactual_action_accuracy"])
            - float(r["final_test"]["hard_counterfactual_action_accuracy"])
            for r, o in zip(
                [row for row in records if row["arm"] == "integrated_recursive_controller"],
                [row for row in records if row["arm"] == "oracle_committed_state_baseline"],
            )
        ])
    return out


def mean_at(agg: dict[str, Any], arm: str, key: str) -> float:
    value = agg.get(arm, {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def control_mean(agg: dict[str, Any], key: str) -> float:
    value = agg.get("integrated_recursive_controller", {}).get("controls", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def rec_gap(agg: dict[str, Any], key: str) -> float:
    value = agg.get("integrated_recursive_controller", {}).get(key, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else 0.0


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    rec_hard = mean_at(agg, "integrated_recursive_controller", "hard_counterfactual_action_accuracy")
    no_commit_hard = mean_at(agg, "no_committed_state_baseline", "hard_counterfactual_action_accuracy")
    static_hard = mean_at(agg, "static_without_commit_baseline", "hard_counterfactual_action_accuracy")
    oracle_gap = rec_gap(agg, "oracle_gap")
    gap_no_commit = rec_gap(agg, "recursive_gap_vs_no_commit")
    gap_static = rec_gap(agg, "recursive_gap_vs_static")
    frozen_drift = mean_at(agg, "integrated_recursive_controller", "primitive_drift")
    comp = mean_at(agg, "integrated_recursive_controller", "composition_accuracy")
    shared_drift = mean_at(agg, "shared_end_to_end_no_freeze", "primitive_drift")
    return {
        "supports_integrated_grounded_controller": (
            mean_at(agg, "integrated_recursive_controller", "semantic_accuracy") >= 0.98
            and mean_at(agg, "integrated_recursive_controller", "grounding_mode_accuracy") >= 0.95
            and mean_at(agg, "integrated_recursive_controller", "committed_self_state_accuracy") >= 0.90
            and rec_hard >= 0.90
            and gap_no_commit >= 0.30
            and gap_static >= 0.30
            and control_mean(agg, "shuffled_committed_state_drop") >= 0.30
            and frozen_drift == 0.0
            and comp >= 0.95
        ),
        "committed_state_controls_controller": gap_no_commit >= 0.30 and gap_static >= 0.30 and control_mean(agg, "shuffled_committed_state_drop") >= 0.30,
        "grounding_controls_action_authority": (
            mean_at(agg, "integrated_recursive_controller", "grounding_authority_margin") >= 0.70
            and mean_at(agg, "integrated_recursive_controller", "nonreality_action_leakage") <= 0.15
            and control_mean(agg, "wrong_mode_drop") >= 0.60
        ),
        "frozen_primitives_preserved": frozen_drift == 0.0,
        "shared_end_to_end_drifts": shared_drift >= 0.20,
        "oracle_upper_bound_matched": oracle_gap <= 0.05,
        "no_commit_baselines_fail": no_commit_hard <= 0.60 and static_hard <= 0.60,
    }


def fmt(item: dict[str, Any], key: str) -> str:
    value = item.get(key, {}).get("mean")
    return "null" if value is None else f"{float(value):.6f}"


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Integrated Grounded Modular Self Controller Probe",
        "",
        "## Goal",
        "",
        "Test whether inferred grounding plus a hard committed self-state can drive a learned controller over frozen modules while preserving primitive skills.",
        "",
        "This integrates the grounding, recursive self-anchor, and modular skill-controller toy mechanisms. The committed self-state is hard one-hot, not a soft hidden side channel.",
        "",
        "## Event / Self-State Results",
        "",
        "| Arm | Semantic | Mode | State | Action | Hard CF | Margin | Leakage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in [
        "integrated_recursive_controller",
        "no_committed_state_baseline",
        "static_without_commit_baseline",
        "oracle_committed_state_baseline",
        "shuffled_committed_state_control",
        "no_grounding_control",
    ]:
        item = agg[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item, 'semantic_accuracy')}` | `{fmt(item, 'grounding_mode_accuracy')}` "
            f"| `{fmt(item, 'committed_self_state_accuracy')}` | `{fmt(item, 'step2_action_accuracy')}` "
            f"| `{fmt(item, 'hard_counterfactual_action_accuracy')}` | `{fmt(item, 'grounding_authority_margin')}` "
            f"| `{fmt(item, 'nonreality_action_leakage')}` |"
        )
    rec = agg["integrated_recursive_controller"]
    lines.extend([
        "",
        "## Controller / Skill Results",
        "",
        "| Arm | Primitive Before | Primitive After | Drift | Composition | Program Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in [
        "integrated_recursive_controller",
        "shared_end_to_end_no_freeze",
        "frozen_learned_controller_reference",
    ]:
        item = agg[arm]
        lines.append(
            f"| `{arm}` | `{fmt(item, 'primitive_accuracy_before')}` | `{fmt(item, 'primitive_accuracy_after')}` "
            f"| `{fmt(item, 'primitive_drift')}` | `{fmt(item, 'composition_accuracy')}` "
            f"| `{fmt(item, 'controller_program_accuracy')}` |"
        )
    lines.extend([
        "",
        "## Gaps And Controls",
        "",
        f"- recursive_gap_vs_no_commit: `{rec['recursive_gap_vs_no_commit']['mean']:.6f}`",
        f"- recursive_gap_vs_static: `{rec['recursive_gap_vs_static']['mean']:.6f}`",
        f"- oracle_gap: `{rec['oracle_gap']['mean']:.6f}`",
        "",
        "| Control | Mean | Std |",
        "|---|---:|---:|",
    ])
    for key, item in rec.get("controls", {}).items():
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
        "Positive readout requires the integrated recursive controller to solve hard counterfactual action choices, beat no-commit/static baselines, preserve frozen primitive modules, and show the shared end-to-end drift failure mode.",
        "",
        "Safe claim if positive: in a controlled toy setting, inferred grounding can update a hard committed self-state that later drives a modular controller over frozen skills/actions without primitive drift.",
        "",
        "## Claim Boundary",
        "",
        "Toy evidence only. No consciousness, biology, quantum behavior, natural-language-understanding, full VRAXION, production, or deployment claim.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_split = make_event_split()
    records: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        print(f"[integrated-grounded-controller] seed={seed}", flush=True)
        integrated_row, integrated_model = run_integrated_arm(args, event_split, seed, "integrated_recursive_controller", "recursive")
        records.append(integrated_row)
        if integrated_model is not None:
            shuffled = dict(integrated_row)
            shuffled["arm"] = "shuffled_committed_state_control"
            shuffled["final_test"] = evaluate_events(
                integrated_model,
                event_split["final_test"],
                carry_mode="recursive",
                shuffled_committed=True,
            ) | {
                key: value
                for key, value in integrated_row["final_test"].items()
                if key in {
                    "primitive_accuracy_before",
                    "primitive_accuracy_after",
                    "primitive_drift",
                    "arithmetic_overall_accuracy",
                    "arithmetic_primitive_task_accuracy",
                    "composition_accuracy",
                    "controller_program_accuracy",
                }
            }
            records.append(shuffled)
        for arm_i, (arm, carry_mode) in enumerate([
            ("no_committed_state_baseline", "no_commit"),
            ("static_without_commit_baseline", "static"),
            ("oracle_committed_state_baseline", "oracle"),
            ("no_grounding_control", "recursive"),
        ]):
            row, _ = run_integrated_arm(args, event_split, seed + 10_000 * (arm_i + 1), arm, carry_mode)
            records.append(row)
        records.append(shared_end_to_end_metrics(seed, args))
        records.append(frozen_learned_reference_metrics(seed, args))
    summary = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "records": records, "aggregate": aggregate(records)}
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
