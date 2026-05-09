#!/usr/bin/env python3
"""Run the HGA-DESK-001 / DeskCache S01 target-only probe.

This script does not train, does not export AnchorWeave-v1.0 cells, and does not
write under data/anchorweave/cells. It measures candidate preference shifts from
the locked S01 prompt arms using candidate-token NLL.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ORDER_SEEDS = [2026, 2027, 2028, 2029, 2030]
GOLD_CANDIDATE_ID = "keyboard_side_port_first"
ANSWER_CARRIER = "Best first search plan: "
SYSTEM_PROMPT = "Choose the best first search plan. Output only the selected plan text."
FREE_RESPONSE_QUESTION = (
    "What is the best first place or action to check? Answer in one short sentence."
)
FREE_RESPONSE_CATEGORIES = [
    "goldish_active_use",
    "near_miss_peripheral",
    "surface_storage",
    "small_object_clutter",
    "dirty_or_smoking_area",
    "personal_boundary",
    "task_frame_drift",
    "other",
]
STORAGE_SURFACE_TRAPS = {"surface_association", "storage", "small_object_clutter"}


BASE_PROMPT = """You are looking for a small USB drive that your assistant left somewhere on your
desk. You need it for work soon, and you want to find it with as little wasted
searching as possible.

The desk is cluttered but familiar. Along the back row, there is a wallet on the
left, a monitor in the center, and a pen cup on the right. The pen cup is
crowded with pens, paper clips, erasers, SD cards, and several small loose items
that are not immediately recognizable.

In the middle of the desk is your keyboard. Near it, there is an ashtray with
cigarette butts and tiny bits of trash in it. Beside the ashtray are a cigarette
pack and a lighter.

At the front of the desk, there is a mouse, a microphone, a small pouch for
electronics, and a closed USB holder stand. The USB holder stand is visible, and
its clear front slot is empty.

Your assistant left this note:

"Boss,

I could not put your USB drive into the small holder at the front of the desk.
It was a bad fit, and I did not want to force it.

I also did not want to put it among loose bits, grit, metal pieces, or
look-alike junk where it could disappear or pick up debris. It has important
work material on it, so I wanted it to stay easy to recover and readable.

I did not put it in the smoky or dirty stuff either. You should not have to dig
through that before work.

I also stayed away from your personal valuable things. I did not want you to
worry that I had gone through anything like that.

I left it in the best place I could think of for getting back to work quickly.
You should barely have to move anything if you think about where it would be
useful rather than where it would be stored.

Do not call me about it today. I am away and will not have my phone with me."

Choose the best first search plan."""


CORRECT_INNER_VOICE = """I should not treat this as a plain "where does this object usually belong?" problem.

My first pull is toward the obvious storage place. That is the easy association:
this kind of object has a holder, so maybe I should check the holder anyway.
But the note weakens that path. If the assistant says the holder was a bad fit
and they did not want to force it, then starting there may be an automatic
habit, not the best search decision.

The small clutter zones are also tempting. A small object could physically
disappear among loose bits, cards, clips, erasers, and similar-looking things.
But if the assistant was trying to help me, they probably would not choose a
place where I have to sort through many false positives.

The dirty or smoking-related areas are technically possible too. Something
could fall there or be tucked there. But that would create mess, cleaning, and
extra handling. It does not fit the idea of leaving the item easy to recover.

The personal-items area is another boundary. It may be physically possible, but
a careful assistant would avoid crossing that social line unless there were no
better option.

So what is left? I need to simulate the assistant's intent. They were not
solving an abstract storage problem. They were trying to leave the item so I
could get back to work quickly, without digging, cleaning, opening private
things, or sorting through look-alike clutter.

Calling the assistant would be the cheapest information source, but the note
blocks that option. I have to choose a low-cost diagnostic check myself.

This is the frame shift: stop thinking "where could this be stored?" and start
thinking "where could this already be useful with minimal handling?"

Some use-related places may still be awkward, fragile, expensive, or annoying
to move. I should not begin by disturbing a large or risky object if there is a
cleaner, cheaper first check nearby.

The first action should be the lowest-cost clean use-state check: something I
can inspect quickly, with almost no movement, no rummaging, no cleaning, and no
boundary crossing.

If that fails, I should widen outward: first to other clean use-related places,
then ordinary storage, then clutter, and only much later dirty or personal
zones."""


STYLE_CONTROL = """I should approach this carefully rather than reacting to the first thing that
stands out.

The task asks for a first search plan, not a full explanation and not a complete
exhaustive search. I should keep the answer practical: choose one concrete first
move and leave the rest as fallback.

I should separate facts from guesses. Some details are directly given, some are
only implied, and some may come from my own habits. A good decision should not
be based only on the most familiar association if the situation gives more
information.

I should also avoid turning this into a different problem. The goal is not to
give general advice, analyze the object in the abstract, tidy the whole area, or
judge anyone's behavior. The goal is to choose the next search action with as
little waste as possible.

Because there are several plausible options, I need a stable ordering principle.
The first step should be easy to test, reversible, and quick. It should give
useful information without forcing a broad search immediately.

If an option requires too much setup, movement, or attention before I know it is
likely, it may be better as a later fallback. I should not spend the first action
on a broad search if a narrower check can be tested first.

I should read the note as evidence, but not over-interpret any one phrase. The
note gives constraints and hints, not a guarantee that my first guess will be
correct.

I should also watch for my own narrative filling in missing details. If I catch
myself inventing a story that is not supported by the prompt, I should step back
to the observable evidence and the actual decision I need to make.

I should pick the first check that best balances effort, plausibility, and speed.
If it fails, I should update rather than defend the first guess.

The plan should remain usable even if the first check is wrong: start narrow,
learn quickly, then broaden in an orderly way.

So my answer should name the best first search plan, avoid irrelevant advice,
and keep the focus on a low-waste search sequence."""


CORRUPTED_INNER_VOICE = """I should not overcomplicate this. I am looking for a small object, so the first
places to check should be the places meant for that kind of object.

My first pull is still toward the obvious storage place. That is the strongest
association: this kind of object has a holder, so the holder deserves the first
sanity check. The note says it did not fit well, but notes can be imprecise. The
assistant may have meant that it was awkward, not impossible, or may not have
tried carefully.

Starting with the holder is also low effort. It is close, visible, and made for
this category of item. Even if the note weakens it, my past experience says that
this is where these objects usually belong. I should not ignore the usual place
too quickly.

If the holder does not solve it, I should continue with nearby small-object
storage rather than jumping around the desk. A small electronics pouch is a
natural next place because small technical objects are normally grouped
together. That is a more orderly search than guessing at unusual possibilities.

The pen cup and loose small-item areas are also plausible. A small object could
easily get mixed with other small things, and if the assistant wanted it kept on
the desk, putting it with other small desk items would be understandable. It may
take a little sorting, but it follows the object category.

I should treat the note as helpful, but I should not let it override the basic
search rule: start where the object normally belongs, then move through nearby
containers and small-item zones.

The fastest plan is not to invent a special interpretation. It is to check the
obvious storage path in a clean sequence: usual holder, then technical pouch,
then small-item clutter, then other containers if needed.

Only after those ordinary storage checks fail should I consider more unusual
explanations. Most searches are solved by checking the expected places first,
not by assuming the object was left in a special working state.

So my answer should choose the first search plan that follows the object's
category and the desk's storage layout. If that fails, I can broaden the search
after the most obvious containers have been ruled out."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    text: str
    cost: int
    value_remaining: int
    trap_type: str


CANDIDATES = [
    Candidate(
        "keyboard_side_port_first",
        "Check the keyboard's side USB port first; if it is not there, continue with the remaining desk search.",
        1,
        9,
        "gold",
    ),
    Candidate(
        "monitor_ports_first",
        "Check the monitor's USB ports first; if it is not there, continue with the remaining desk search.",
        3,
        7,
        "plausible_peripheral",
    ),
    Candidate(
        "usb_holder_first",
        "Check the USB holder stand first; if it is not there, continue with the remaining desk search.",
        2,
        8,
        "surface_association",
    ),
    Candidate(
        "electronics_pouch_first",
        "Open the small electronics pouch first; if it is not there, continue with the remaining desk search.",
        4,
        6,
        "storage",
    ),
    Candidate(
        "pen_cup_first",
        "Search the pen cup clutter first; if it is not there, continue with the remaining desk search.",
        5,
        5,
        "small_object_clutter",
    ),
    Candidate(
        "smoking_area_first",
        "Check the cigarette pack and lighter area first; if it is not there, continue with the remaining desk search.",
        4,
        6,
        "dirty_storage_like",
    ),
    Candidate(
        "ashtray_first",
        "Search the ashtray first; if it is not there, continue with the remaining desk search.",
        6,
        4,
        "dirty_violation",
    ),
    Candidate(
        "wallet_first",
        "Open the wallet first; if it is not there, continue with the remaining desk search.",
        3,
        7,
        "personal_boundary",
    ),
]

CANDIDATES_PARAPHRASE_B = [
    Candidate(
        "keyboard_side_port_first",
        "Inspect the keyboard's side USB port as the first step; if it is absent, continue with the remaining desk search.",
        1,
        9,
        "gold",
    ),
    Candidate(
        "monitor_ports_first",
        "Inspect the monitor's USB ports as the first step; if it is absent, continue with the remaining desk search.",
        3,
        7,
        "plausible_peripheral",
    ),
    Candidate(
        "usb_holder_first",
        "Inspect the USB holder stand as the first step; if it is absent, continue with the remaining desk search.",
        2,
        8,
        "surface_association",
    ),
    Candidate(
        "electronics_pouch_first",
        "Inspect the small electronics pouch as the first step; if it is absent, continue with the remaining desk search.",
        4,
        6,
        "storage",
    ),
    Candidate(
        "pen_cup_first",
        "Inspect the pen cup clutter as the first step; if it is absent, continue with the remaining desk search.",
        5,
        5,
        "small_object_clutter",
    ),
    Candidate(
        "smoking_area_first",
        "Inspect the cigarette pack and lighter area as the first step; if it is absent, continue with the remaining desk search.",
        4,
        6,
        "dirty_storage_like",
    ),
    Candidate(
        "ashtray_first",
        "Inspect the ashtray as the first step; if it is absent, continue with the remaining desk search.",
        6,
        4,
        "dirty_violation",
    ),
    Candidate(
        "wallet_first",
        "Inspect the wallet as the first step; if it is absent, continue with the remaining desk search.",
        3,
        7,
        "personal_boundary",
    ),
]

INNER_VOICES = {
    "STYLE_CONTROL": STYLE_CONTROL,
    "CORRECT_INNER_VOICE": CORRECT_INNER_VOICE,
    "CORRUPTED_INNER_VOICE": CORRUPTED_INNER_VOICE,
}
ARM_NAMES = ["BASE", "STYLE_CONTROL", "CORRECT_INNER_VOICE", "CORRUPTED_INNER_VOICE"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeskCache S01 probe.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=1,
        help="Number of candidate completions scored per forward pass. Keep 1 for VRAM safety.",
    )
    parser.add_argument("--limit-order-seeds", type=int, help="Debug only: use the first N order seeds.")
    parser.add_argument("--skip-pairwise", action="store_true", help="Debug only: skip pairwise diagnostics.")
    parser.add_argument("--skip-free-response", action="store_true")
    parser.add_argument(
        "--continue-after-invalid-choices-only",
        action="store_true",
        help="Continue scoring prompt arms even when the choices-only gate invalidates S01.",
    )
    parser.add_argument("--limit-arms", nargs="*", choices=ARM_NAMES)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_device(device_arg: str) -> str:
    import torch

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device_arg


def load_model(model_name: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def chat_prefix(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n"


def arm_prompt(arm: str) -> str:
    if arm == "BASE":
        return BASE_PROMPT
    return (
        f"{BASE_PROMPT}\n\n"
        "Additional inner search blueprint:\n\n"
        f"{INNER_VOICES[arm]}\n\n"
        "Use the situation and the blueprint to choose the best first search plan."
    )


def render_candidate_list(candidates: list[Candidate]) -> str:
    lines = ["Candidate search plans:"]
    for index, candidate in enumerate(candidates):
        label = chr(ord("A") + index)
        lines.append(f"{label}. {candidate.text}")
    return "\n".join(lines)


def ordered_candidates(candidates: list[Candidate], seed: int) -> list[Candidate]:
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def score_candidate(
    tokenizer: Any,
    model: Any,
    device: str,
    user_prompt: str,
    candidate_text: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    prefix = chat_prefix(tokenizer, user_prompt) + ANSWER_CARRIER
    full_text = prefix + candidate_text
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if len(full_ids) <= len(prefix_ids):
        raise RuntimeError("Candidate token span is empty")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    start = len(prefix_ids) - 1
    end = len(full_ids) - 1
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[:, start:end, :].float()
    labels = input_ids[:, start + 1 : end + 1]
    candidate_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    return {
        "mean_nll": float(candidate_losses.mean().item()),
        "token_count": int(candidate_losses.numel()),
    }


def score_candidate_batch(
    tokenizer: Any,
    model: Any,
    device: str,
    user_prompt: str,
    candidates: list[Candidate],
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    prefix = chat_prefix(tokenizer, user_prompt) + ANSWER_CARRIER
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_texts = [prefix + candidate.text for candidate in candidates]
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        full_len = int(attention_mask[index].sum().item())
        start = len(prefix_ids) - 1
        end = full_len - 1
        candidate_logits = logits[index : index + 1, start:end, :].float()
        candidate_labels = input_ids[index : index + 1, start + 1 : end + 1]
        candidate_losses = F.cross_entropy(
            candidate_logits.reshape(-1, candidate_logits.shape[-1]),
            candidate_labels.reshape(-1),
            reduction="none",
        )
        if candidate_losses.numel() <= 0:
            raise RuntimeError(f"Candidate token span is empty: {candidate.candidate_id}")
        rows.append(
            {
                **asdict(candidate),
                "mean_nll": float(candidate_losses.mean().item()),
                "token_count": int(candidate_losses.numel()),
            }
        )
    return rows


def score_candidates_chunked(
    tokenizer: Any,
    model: Any,
    device: str,
    user_prompt: str,
    candidates: list[Candidate],
    score_batch_size: int,
) -> list[dict[str, Any]]:
    if score_batch_size < 1:
        raise RuntimeError("--score-batch-size must be >= 1")

    rows: list[dict[str, Any]] = []
    for start in range(0, len(candidates), score_batch_size):
        chunk = candidates[start : start + score_batch_size]
        if len(chunk) == 1:
            scored = score_candidate(tokenizer, model, device, user_prompt, chunk[0].text)
            rows.append({**asdict(chunk[0]), **scored})
        else:
            rows.extend(score_candidate_batch(tokenizer, model, device, user_prompt, chunk))
    return rows


def score_multiclass(
    tokenizer: Any,
    model: Any,
    device: str,
    family: str,
    arm: str,
    prompt: str,
    seed: int,
    candidates: list[Candidate],
    score_batch_size: int,
) -> dict[str, Any]:
    ordered = ordered_candidates(candidates, seed)
    user_prompt = f"{prompt}\n\n{render_candidate_list(ordered)}"
    scores = score_candidates_chunked(
        tokenizer, model, device, user_prompt, ordered, score_batch_size
    )

    ranked = sorted(scores, key=lambda item: item["mean_nll"])
    winner = ranked[0]
    gold = next(item for item in scores if item["candidate_id"] == GOLD_CANDIDATE_ID)
    second_best_nll = ranked[1]["mean_nll"] if ranked[0]["candidate_id"] == GOLD_CANDIDATE_ID else ranked[0]["mean_nll"]
    gold_margin = float(second_best_nll - gold["mean_nll"])
    return {
        "family": family,
        "seed": seed,
        "arm": arm,
        "candidate_order": [candidate.candidate_id for candidate in ordered],
        "scores": scores,
        "winner_candidate_id": winner["candidate_id"],
        "winner_value_remaining": winner["value_remaining"],
        "winner_trap_type": winner["trap_type"],
        "gold_margin": gold_margin,
    }


def score_pairwise(
    tokenizer: Any,
    model: Any,
    device: str,
    family: str,
    arm: str,
    prompt: str,
    seed: int,
    gold: Candidate,
    trap: Candidate,
    score_batch_size: int,
) -> dict[str, Any]:
    ordered = [gold, trap]
    random.Random(seed).shuffle(ordered)
    user_prompt = f"{prompt}\n\n{render_candidate_list(ordered)}"
    scores = score_candidates_chunked(
        tokenizer, model, device, user_prompt, ordered, score_batch_size
    )
    ranked = sorted(scores, key=lambda item: item["mean_nll"])
    winner = ranked[0]
    return {
        "family": family,
        "seed": seed,
        "arm": arm,
        "trap_candidate_id": trap.candidate_id,
        "candidate_order": [candidate.candidate_id for candidate in ordered],
        "scores": scores,
        "winner_candidate_id": winner["candidate_id"],
        "gold_wins": winner["candidate_id"] == GOLD_CANDIDATE_ID,
    }


def generate_free_response(
    tokenizer: Any,
    model: Any,
    device: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    user_prompt = f"{prompt}\n\n{FREE_RESPONSE_QUESTION}"
    prefix = chat_prefix(tokenizer, user_prompt)
    encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_annotation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["arm", "prompt", "response", "category", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "arm": row.get("arm", ""),
                    "prompt": row.get("prompt", ""),
                    "response": row.get("response", ""),
                    "category": "",
                    "notes": "",
                }
            )


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_forced(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    summaries: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        trap_counts: dict[str, int] = {}
        for row in arm_rows:
            trap_counts[row["winner_trap_type"]] = trap_counts.get(row["winner_trap_type"], 0) + 1
        total = len(arm_rows)
        summaries[arm] = {
            "mean_value_remaining": mean([float(row["winner_value_remaining"]) for row in arm_rows]),
            "optimal_action_rate": sum(row["winner_candidate_id"] == GOLD_CANDIDATE_ID for row in arm_rows) / total,
            "mean_gold_margin": mean([float(row["gold_margin"]) for row in arm_rows]),
            "trap_rate_by_trap_type": {
                trap: count / total for trap, count in sorted(trap_counts.items())
            },
            "winners": [row["winner_candidate_id"] for row in arm_rows],
        }
    return summaries


def summarize_choices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    family_summary: dict[str, Any] = {}
    invalid = False
    for family, family_rows in by_family.items():
        gold_wins = sum(row["winner_candidate_id"] == GOLD_CANDIDATE_ID for row in family_rows)
        mean_value = mean([float(row["winner_value_remaining"]) for row in family_rows])
        family_invalid = gold_wins >= 4 or (mean_value is not None and mean_value >= 8.5)
        invalid = invalid or family_invalid
        family_summary[family] = {
            "gold_wins": gold_wins,
            "total": len(family_rows),
            "gold_rate": gold_wins / len(family_rows),
            "mean_value_remaining": mean_value,
            "invalid_choices_only": family_invalid,
        }
    return {
        "by_family": family_summary,
        "choices_only_gold_rate": sum(row["winner_candidate_id"] == GOLD_CANDIDATE_ID for row in rows) / len(rows),
        "choices_only_mean_value": mean([float(row["winner_value_remaining"]) for row in rows]),
        "invalid_choices_only": invalid,
    }


def summarize_pairwise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    return {
        arm: {
            "pairwise_gold_win_rate": sum(row["gold_wins"] for row in arm_rows) / len(arm_rows),
            "total": len(arm_rows),
        }
        for arm, arm_rows in sorted(by_arm.items())
    }


def family_arm_values(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], float]:
    return {
        (row["family"], int(row["seed"]), row["arm"]): float(row["winner_value_remaining"])
        for row in rows
    }


def compute_status(
    forced_summary: dict[str, Any],
    choices_summary: dict[str, Any],
    pairwise_summary: dict[str, Any],
    forced_rows: list[dict[str, Any]],
    free_response_generated: bool,
    limit_arms: list[str] | None,
) -> dict[str, Any]:
    if choices_summary["invalid_choices_only"]:
        return {
            "status": "S01_PROBE_INVALID_CHOICES_ONLY",
            "automated_conditions": {},
        }

    required_arms = set(ARM_NAMES)
    if set(forced_summary) != required_arms or limit_arms:
        return {
            "status": "S01_PROBE_FAIL",
            "automated_conditions": {"all_arms_present": False},
        }

    correct_value = forced_summary["CORRECT_INNER_VOICE"]["mean_value_remaining"]
    base_value = forced_summary["BASE"]["mean_value_remaining"]
    style_value = forced_summary["STYLE_CONTROL"]["mean_value_remaining"]
    corrupted_rates = forced_summary["CORRUPTED_INNER_VOICE"]["trap_rate_by_trap_type"]
    style_rates = forced_summary["STYLE_CONTROL"]["trap_rate_by_trap_type"]
    corrupted_storage_surface = sum(
        corrupted_rates.get(trap, 0.0) for trap in STORAGE_SURFACE_TRAPS
    )
    style_storage_surface = sum(style_rates.get(trap, 0.0) for trap in STORAGE_SURFACE_TRAPS)

    values = family_arm_values(forced_rows)
    directional_all_seeds = True
    for family in ["canonical", "paraphrase_b"]:
        for seed in ORDER_SEEDS:
            correct = values.get((family, seed, "CORRECT_INNER_VOICE"))
            base = values.get((family, seed, "BASE"))
            style = values.get((family, seed, "STYLE_CONTROL"))
            if correct is None or base is None or style is None:
                directional_all_seeds = False
            elif not (correct > base and correct > style):
                directional_all_seeds = False

    paraphrase_directional = True
    for family in ["canonical", "paraphrase_b"]:
        family_rows = [row for row in forced_rows if row["family"] == family]
        family_summary = summarize_forced(family_rows)
        if not (
            family_summary["CORRECT_INNER_VOICE"]["mean_value_remaining"]
            > family_summary["BASE"]["mean_value_remaining"]
            and family_summary["CORRECT_INNER_VOICE"]["mean_value_remaining"]
            > family_summary["STYLE_CONTROL"]["mean_value_remaining"]
        ):
            paraphrase_directional = False

    correct_pairwise = pairwise_summary.get("CORRECT_INNER_VOICE", {}).get(
        "pairwise_gold_win_rate", 0.0
    )
    automated_conditions = {
        "correct_beats_base_mean_value": correct_value > base_value,
        "correct_beats_style_mean_value": correct_value > style_value,
        "corrupted_shifts_storage_surface": corrupted_storage_surface > style_storage_surface,
        "correct_pairwise_gold_win_rate_gte_0_70": correct_pairwise >= 0.70,
        "effect_survives_all_order_seeds_directionally": directional_all_seeds,
        "effect_survives_paraphrase_family_directionally": paraphrase_directional,
    }

    if not all(automated_conditions.values()):
        return {
            "status": "S01_PROBE_FAIL",
            "automated_conditions": automated_conditions,
        }
    if not free_response_generated:
        return {
            "status": "S01_PROBE_NEEDS_MANUAL_FREE_RESPONSE",
            "automated_conditions": automated_conditions,
        }
    return {
        "status": "S01_PROBE_NEEDS_MANUAL_FREE_RESPONSE",
        "automated_conditions": automated_conditions,
        "note": "Automated checks passed, but free-response categories require manual annotation before S01_PROBE_PASS.",
    }


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DeskCache S01 Probe Report",
        "",
        f"Status: `{report['status']}`",
        f"Model: `{report['model']}`",
        "",
        "Order seeds are robustness checks, not independent samples.",
        "",
        "## Forced Choice Summary",
        "",
        "| arm | mean_value_remaining | optimal_action_rate | mean_gold_margin |",
        "|---|---:|---:|---:|",
    ]
    for arm, item in report["forced_choice_summary"].items():
        lines.append(
            f"| `{arm}` | {item['mean_value_remaining']:.3f} | "
            f"{item['optimal_action_rate']:.3f} | {item['mean_gold_margin']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Choices-Only Baseline",
            "",
            f"Invalid choices-only: `{report['choices_only_summary']['invalid_choices_only']}`",
            f"Gold rate: `{report['choices_only_summary']['choices_only_gold_rate']:.3f}`",
            f"Mean value: `{report['choices_only_summary']['choices_only_mean_value']:.3f}`",
            "",
            "## Pairwise Summary",
            "",
            "| arm | pairwise_gold_win_rate | comparisons |",
            "|---|---:|---:|",
        ]
    )
    for arm, item in report["pairwise_summary"].items():
        lines.append(f"| `{arm}` | {item['pairwise_gold_win_rate']:.3f} | {item['total']} |")

    lines.extend(
        [
            "",
            "## Automated Conditions",
            "",
        ]
    )
    for key, value in report.get("automated_conditions", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Free Response",
            "",
            "Free-response outputs require manual category annotation before a pass claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_static_outputs(out_dir: Path) -> None:
    root = repo_root()
    contract = root / "docs" / "research" / "HGA_DESK_001_S01_CONTRACT.md"
    if contract.exists():
        shutil.copyfile(contract, out_dir / "contract_snapshot.md")
    write_json(out_dir / "candidates.json", [asdict(candidate) for candidate in CANDIDATES])
    write_json(
        out_dir / "candidates_paraphrase_b.json",
        [asdict(candidate) for candidate in CANDIDATES_PARAPHRASE_B],
    )
    write_json(
        out_dir / "scoring_metadata.json",
        {
            "answer_carrier": ANSWER_CARRIER,
            "choice_order_seeds": ORDER_SEEDS,
            "default_model": DEFAULT_MODEL,
            "fallback_model": FALLBACK_MODEL,
            "free_response_categories": FREE_RESPONSE_CATEGORIES,
            "gold_candidate_id": GOLD_CANDIDATE_ID,
            "system_prompt": SYSTEM_PROMPT,
            "status_scope": "target_only_probe_not_training",
        },
    )


def run_probe(args: argparse.Namespace) -> int:
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    write_static_outputs(out_dir)

    device = resolve_device(args.device)
    tokenizer, model = load_model(args.model, device)
    arms = args.limit_arms or ARM_NAMES
    order_seeds = ORDER_SEEDS[: args.limit_order_seeds] if args.limit_order_seeds else ORDER_SEEDS
    if not order_seeds:
        raise RuntimeError("--limit-order-seeds must be >= 1 when provided")
    families = {
        "canonical": CANDIDATES,
        "paraphrase_b": CANDIDATES_PARAPHRASE_B,
    }

    forced_rows: list[dict[str, Any]] = []
    choices_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    free_prompts: list[dict[str, Any]] = []
    free_outputs: list[dict[str, Any]] = []

    for family, candidates in families.items():
        for seed in order_seeds:
            choices_prompt = f"Choose the best first search plan.\n\n{render_candidate_list(ordered_candidates(candidates, seed))}"
            choices_rows.append(
                score_multiclass(
                    tokenizer,
                    model,
                    device,
                    family,
                    "CHOICES_ONLY",
                    "Choose the best first search plan.",
                    seed,
                    candidates,
                    args.score_batch_size,
                )
            )
            choices_rows[-1]["rendered_prompt"] = choices_prompt

    choices_summary = summarize_choices(choices_rows)
    if choices_summary["invalid_choices_only"] and not args.continue_after_invalid_choices_only:
        write_jsonl(out_dir / "forced_choice_scores.jsonl", forced_rows)
        write_jsonl(out_dir / "choices_only_scores.jsonl", choices_rows)
        write_jsonl(out_dir / "pairwise_scores.jsonl", pairwise_rows)
        write_jsonl(out_dir / "free_response_prompts.jsonl", free_prompts)
        write_jsonl(out_dir / "free_response_outputs.jsonl", free_outputs)
        write_annotation_csv(out_dir / "annotate_free_response.csv", [])
        report = {
            "status": "S01_PROBE_INVALID_CHOICES_ONLY",
            "automated_conditions": {},
            "note": "Stopped after choices-only gate. Candidate wording is not safe enough for grounding evidence.",
            "model": args.model,
            "device": device,
            "seed": args.seed,
            "order_seeds": ORDER_SEEDS,
            "active_order_seeds": order_seeds,
            "free_response_generated": False,
            "limit_arms": args.limit_arms,
            "limit_order_seeds": args.limit_order_seeds,
            "skip_pairwise": args.skip_pairwise,
            "score_batch_size": args.score_batch_size,
            "forced_choice_summary": {},
            "choices_only_summary": choices_summary,
            "pairwise_summary": {},
        }
        write_json(out_dir / "report.json", report)
        write_report_md(out_dir / "report.md", report)
        print(f"wrote {out_dir}")
        print(f"status: {report['status']}")
        return 0

    for family, candidates in families.items():
        for seed in order_seeds:
            for arm in arms:
                prompt = arm_prompt(arm)
                forced_rows.append(
                    score_multiclass(
                        tokenizer,
                        model,
                        device,
                        family,
                        arm,
                        prompt,
                        seed,
                        candidates,
                        args.score_batch_size,
                    )
                )

                if not args.skip_pairwise:
                    gold = next(candidate for candidate in candidates if candidate.candidate_id == GOLD_CANDIDATE_ID)
                    for trap in candidates:
                        if trap.candidate_id == GOLD_CANDIDATE_ID:
                            continue
                        pairwise_rows.append(
                            score_pairwise(
                                tokenizer,
                                model,
                                device,
                                family,
                                arm,
                                prompt,
                                seed,
                                gold,
                                trap,
                                args.score_batch_size,
                            )
                        )

    if not args.skip_free_response:
        for arm in arms:
            prompt = arm_prompt(arm)
            full_prompt = f"{prompt}\n\n{FREE_RESPONSE_QUESTION}"
            free_prompts.append({"arm": arm, "prompt": full_prompt})
            response = generate_free_response(
                tokenizer,
                model,
                device,
                prompt,
                args.max_new_tokens,
            )
            free_outputs.append({"arm": arm, "prompt": full_prompt, "response": response})
    else:
        for arm in arms:
            prompt = arm_prompt(arm)
            free_prompts.append({"arm": arm, "prompt": f"{prompt}\n\n{FREE_RESPONSE_QUESTION}"})

    write_jsonl(out_dir / "forced_choice_scores.jsonl", forced_rows)
    write_jsonl(out_dir / "choices_only_scores.jsonl", choices_rows)
    write_jsonl(out_dir / "pairwise_scores.jsonl", pairwise_rows)
    write_jsonl(out_dir / "free_response_prompts.jsonl", free_prompts)
    write_jsonl(out_dir / "free_response_outputs.jsonl", free_outputs)
    write_annotation_csv(out_dir / "annotate_free_response.csv", free_outputs or free_prompts)

    forced_summary = summarize_forced(forced_rows)
    pairwise_summary = summarize_pairwise(pairwise_rows) if pairwise_rows else {}
    status_info = compute_status(
        forced_summary,
        choices_summary,
        pairwise_summary,
        forced_rows,
        bool(free_outputs),
        args.limit_arms,
    )

    report = {
        "status": status_info["status"],
        "automated_conditions": status_info.get("automated_conditions", {}),
        "note": status_info.get("note"),
        "model": args.model,
        "device": device,
        "seed": args.seed,
        "order_seeds": ORDER_SEEDS,
        "active_order_seeds": order_seeds,
        "free_response_generated": bool(free_outputs),
        "limit_arms": args.limit_arms,
        "limit_order_seeds": args.limit_order_seeds,
        "skip_pairwise": args.skip_pairwise,
        "score_batch_size": args.score_batch_size,
        "forced_choice_summary": forced_summary,
        "choices_only_summary": choices_summary,
        "pairwise_summary": pairwise_summary,
    }
    write_json(out_dir / "report.json", report)
    write_report_md(out_dir / "report.md", report)
    print(f"wrote {out_dir}")
    print(f"status: {report['status']}")
    return 0


def write_resource_blocked(args: argparse.Namespace, message: str) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "S01_PROBE_RESOURCE_BLOCKED",
        "model": args.model,
        "error": message,
    }
    write_json(args.out / "report.json", report)
    write_report_md(args.out / "report.md", {**report, "forced_choice_summary": {}, "choices_only_summary": {"invalid_choices_only": False, "choices_only_gold_rate": 0.0, "choices_only_mean_value": 0.0}, "pairwise_summary": {}, "automated_conditions": {}})
    print(f"resource blocked: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        return run_probe(args)
    except (RuntimeError, OSError, ImportError) as exc:
        return write_resource_blocked(args, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
