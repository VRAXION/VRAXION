#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_composable_skill_preservation_probe as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "target" / "context-cancellation-probe" / "modular-skill-controller"
REPORT_PATH = ROOT / "docs" / "research" / "MODULAR_SKILL_CONTROLLER_PROBE.md"
SUMMARY_NAME = "modular_skill_controller_summary.json"

PROGRAMS = ["add", "mul", "add_then_mul", "mul_then_add"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen primitive modules plus learned program controller sanity probe.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--controller-hidden", type=int, default=24)
    parser.add_argument("--primitive-epochs", type=int, default=350)
    parser.add_argument("--composition-epochs", type=int, default=350)
    parser.add_argument("--module-epochs", type=int, default=700)
    parser.add_argument("--controller-epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = 1
        args.hidden = 16
        args.controller_hidden = 16
        args.primitive_epochs = 40
        args.composition_epochs = 40
        args.module_epochs = 80
        args.controller_epochs = 40
    return args


class ProgramController(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(base.OPS) + 3 * base.P, hidden),
            nn.Tanh(),
            nn.Linear(hidden, len(PROGRAMS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def all_examples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[str, int, int, int]]]:
    xs = []
    ys = []
    programs = []
    meta = []
    for op in base.OPS:
        for a in range(base.P):
            for b in range(base.P):
                for c in range(base.P):
                    xs.append(base.encode_example(op, a, b, c))
                    ys.append(base.label(op, a, b, c))
                    programs.append(PROGRAMS.index(op))
                    meta.append((op, a, b, c))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), torch.tensor(programs, dtype=torch.long), meta


@torch.no_grad()
def module_argmax(module: base.PrimitiveModule, a: int, b: int) -> int:
    x = torch.zeros(1, 2 * base.P)
    x[0, a] = 1.0
    x[0, base.P + b] = 1.0
    return int(module(x).argmax(dim=-1).item())


@torch.no_grad()
def candidate_logits(add_mod: base.PrimitiveModule, mul_mod: base.PrimitiveModule, meta: list[tuple[str, int, int, int]]) -> torch.Tensor:
    logits = torch.full((len(meta), len(PROGRAMS), base.P), -6.0)
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


def train_modules(seed: int, *, hidden: int, epochs: int, lr: float) -> tuple[base.PrimitiveModule, base.PrimitiveModule]:
    base.set_seed(seed)
    add_x, add_y = base.make_binary_examples("add")
    mul_x, mul_y = base.make_binary_examples("mul")
    add_mod = base.PrimitiveModule(hidden)
    mul_mod = base.PrimitiveModule(hidden)
    base.train_ce(add_mod, add_x, add_y, epochs=epochs, lr=lr)
    base.train_ce(mul_mod, mul_x, mul_y, epochs=epochs, lr=lr)
    for param in add_mod.parameters():
        param.requires_grad_(False)
    for param in mul_mod.parameters():
        param.requires_grad_(False)
    return add_mod, mul_mod


def train_controller(seed: int, args: argparse.Namespace) -> dict[str, float]:
    base.set_seed(seed)
    add_mod, mul_mod = train_modules(seed + 1_000, hidden=args.hidden, epochs=args.module_epochs, lr=args.learning_rate)
    x, y, program_y, meta = all_examples()
    cand = candidate_logits(add_mod, mul_mod, meta)
    controller = ProgramController(args.controller_hidden)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.controller_epochs):
        opt.zero_grad(set_to_none=True)
        weights = F.softmax(controller(x), dim=-1)
        final_logits = torch.einsum("np,npk->nk", weights, cand)
        loss = F.cross_entropy(final_logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        weights = F.softmax(controller(x), dim=-1)
        final_logits = torch.einsum("np,npk->nk", weights, cand)
        pred = final_logits.argmax(dim=-1)
        program_pred = weights.argmax(dim=-1)
    primitive_metrics = base.frozen_interpreter_metrics(add_mod, mul_mod)
    primitive_mask = torch.isin(program_y, torch.tensor([0, 1]))
    composite_mask = ~primitive_mask
    return {
        "primitive_accuracy_before_composition": primitive_metrics["primitive_accuracy_after_composition"],
        "primitive_accuracy_after_composition": primitive_metrics["primitive_accuracy_after_composition"],
        "primitive_drift": primitive_metrics["primitive_drift"],
        "overall_accuracy": float((pred == y).float().mean().item()),
        "primitive_task_accuracy": float((pred[primitive_mask] == y[primitive_mask]).float().mean().item()),
        "composition_accuracy": float((pred[composite_mask] == y[composite_mask]).float().mean().item()),
        "controller_program_accuracy": float((program_pred == program_y).float().mean().item()),
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row["metrics"] for row in rows if row["arm"] == arm]
        keys = sorted({key for row in arm_rows for key in row})
        out[arm] = {key: summarize([float(row[key]) for row in arm_rows]) for key in keys}
    return out


def verdict(agg: dict[str, Any]) -> dict[str, bool]:
    return {
        "learned_controller_preserves_primitives": agg["frozen_learned_controller"]["primitive_drift"]["mean"] == 0.0,
        "learned_controller_composes_successfully": agg["frozen_learned_controller"]["composition_accuracy"]["mean"] >= 0.95,
        "learned_controller_program_selection_successful": agg["frozen_learned_controller"]["controller_program_accuracy"]["mean"] >= 0.95,
        "shared_no_replay_still_forgets": agg["shared_no_replay"]["primitive_drift"]["mean"] >= 0.20,
        "modular_controller_hypothesis_supported": agg["frozen_learned_controller"]["composition_accuracy"]["mean"] >= 0.95 and agg["shared_no_replay"]["primitive_drift"]["mean"] >= 0.20,
    }


def write_report(summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Modular Skill Controller Probe",
        "",
        "## Goal",
        "",
        "Test whether frozen primitive modules can be composed by a learned controller without primitive skill drift.",
        "",
        "## Results",
        "",
        "| Arm | Primitive Before | Primitive After | Drift | Composition | Program Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ["shared_no_replay", "shared_with_replay", "frozen_hand_composition", "frozen_learned_controller"]:
        item = agg[arm]
        program = item.get("controller_program_accuracy", {}).get("mean")
        lines.append(
            f"| `{arm}` | `{item['primitive_accuracy_before_composition']['mean']:.6f}` "
            f"| `{item['primitive_accuracy_after_composition']['mean']:.6f}` "
            f"| `{item['primitive_drift']['mean']:.6f}` "
            f"| `{item['composition_accuracy']['mean']:.6f}` "
            f"| `{program:.6f}` |" if program is not None else
            f"| `{arm}` | `{item['primitive_accuracy_before_composition']['mean']:.6f}` "
            f"| `{item['primitive_accuracy_after_composition']['mean']:.6f}` "
            f"| `{item['primitive_drift']['mean']:.6f}` "
            f"| `{item['composition_accuracy']['mean']:.6f}` | `null` |"
        )
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
        "This is a controller sanity check. It uses a small fixed candidate-program set, so it does not prove open-ended program discovery. It does test whether composition can be learned above frozen primitives without overwriting them.",
        "",
        "Claim boundary: toy evidence only; no consciousness, biology, or production claim.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        print(f"[modular-controller] seed={seed}", flush=True)
        rows.append({"arm": "shared_no_replay", "seed": seed, "metrics": base.train_shared(seed, hidden=args.hidden, primitive_epochs=args.primitive_epochs, composition_epochs=args.composition_epochs, lr=args.learning_rate, replay=False)})
        rows.append({"arm": "shared_with_replay", "seed": seed, "metrics": base.train_shared(seed + 10_000, hidden=args.hidden, primitive_epochs=args.primitive_epochs, composition_epochs=args.composition_epochs, lr=args.learning_rate, replay=True)})
        rows.append({"arm": "frozen_hand_composition", "seed": seed, "metrics": base.train_frozen_modules(seed + 20_000, hidden=args.hidden, module_epochs=args.module_epochs, lr=args.learning_rate)})
        rows.append({"arm": "frozen_learned_controller", "seed": seed, "metrics": train_controller(seed + 30_000, args)})
    summary = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "records": rows, "aggregate": aggregate(rows)}
    summary["verdict"] = verdict(summary["aggregate"])
    out_json = args.out_dir / SUMMARY_NAME
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary)
    print(json.dumps({"verdict": summary["verdict"], "json": str(out_json), "report": str(REPORT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
