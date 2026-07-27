#!/usr/bin/env python3
"""Run deterministic HuggingFace inference for AWFT-001 prompts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel


STANDARD_SYSTEM_PROMPT = "Return only valid JSON. No markdown. No explanation."
HARDENED_SYSTEM_PROMPT = "Return only one JSON object. No markdown. No schema. No explanation."
LIST_FIELDS = ["high_salience", "low_salience", "symbol_attach", "symbol_reject"]
STRING_FIELDS = ["first_action", "commitment_level", "claim_boundary"]
REQUIRED_MODEL_FIELDS = STRING_FIELDS[:1] + LIST_FIELDS + ["counterfactual_answers"] + STRING_FIELDS[1:]
ALLOWED_COMMITMENT_LEVELS = {
    "confirmed_same_place",
    "rejected_same_place",
    "defer_and_disambiguate",
    "premature_commit",
}
ALLOWED_COUNTERFACTUAL_LABELS = {
    "strengthens_match",
    "weakens_match",
    "neutral",
    "requires_disambiguation",
    "rejects_match",
    "confirms_match",
}
REASON_PRECEDENCE = [
    "parse_error",
    "extra_wrapper_or_schema_output",
    "missing_field",
    "wrong_type",
    "list_field_not_list",
    "counterfactual_answers_bad_shape",
    "invalid_enum",
    "other",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AWFT-001 HF inference.")
    parser.add_argument("--model", required=True, help="HuggingFace causal LM id.")
    parser.add_argument("--adapter", type=Path, default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--prompts", required=True, type=Path, help="Eval prompts JSONL.")
    parser.add_argument("--out", required=True, type=Path, help="Predictions JSONL.")
    parser.add_argument("--invalid", required=True, type=Path, help="Invalid JSON/shape log JSONL.")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt-mode", choices=["standard", "hardened"], default="standard")
    return parser.parse_args(argv)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, adapter: Path | None):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def extract_counterfactual_ids(prompt: str) -> list[str]:
    return re.findall(r"^\s*-\s*(cf_[A-Za-z0-9_]+)\s*:", prompt, flags=re.MULTILINE)


def hardened_contract(prompt: str) -> str:
    cf_ids = extract_counterfactual_ids(prompt)
    cf_keys = ", ".join(cf_ids) if cf_ids else "<none found in prompt>"
    enum_values = ", ".join(sorted(ALLOWED_COUNTERFACTUAL_LABELS))
    commitment_values = ", ".join(sorted(ALLOWED_COMMITMENT_LEVELS))
    return (
        "\n\nOutput contract:\n"
        "- Return exactly one JSON object and nothing else.\n"
        "- Do not include markdown fences, prose, a schema, or scenario_id.\n"
        "- The runner supplies scenario_id externally from the eval row.\n"
        "- Required fields: first_action, high_salience, low_salience, symbol_attach, "
        "symbol_reject, counterfactual_answers, commitment_level, claim_boundary.\n"
        "- high_salience, low_salience, symbol_attach, symbol_reject must be arrays of strings.\n"
        "- claim_boundary and first_action must be strings.\n"
        f"- commitment_level must be one of: {commitment_values}.\n"
        f"- counterfactual_answers must use exactly these keys from the prompt: {cf_keys}.\n"
        f"- each counterfactual answer value must be one of: {enum_values}.\n"
        "- Do not use booleans, null, nested objects, or natural-language counterfactual answers.\n"
    )


def render_chat_prompt(tokenizer: Any, prompt: str, prompt_mode: str) -> str:
    system_prompt = HARDENED_SYSTEM_PROMPT if prompt_mode == "hardened" else STANDARD_SYSTEM_PROMPT
    user_prompt = prompt + hardened_contract(prompt) if prompt_mode == "hardened" else prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: "


def extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return {
            "parsed": parsed if isinstance(parsed, dict) else None,
            "parse_error": None if isinstance(parsed, dict) else "raw JSON value is not an object",
            "raw_json_object": isinstance(parsed, dict),
            "structural_extract": False,
        }
    except json.JSONDecodeError as exc:
        raw_error = exc.msg

    start = stripped.find("{")
    if start == -1:
        return {
            "parsed": None,
            "parse_error": raw_error,
            "raw_json_object": False,
            "structural_extract": False,
        }

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    return {
                        "parsed": None,
                        "parse_error": exc.msg,
                        "raw_json_object": False,
                        "structural_extract": True,
                    }
                return {
                    "parsed": parsed if isinstance(parsed, dict) else None,
                    "parse_error": None
                    if isinstance(parsed, dict)
                    else "extracted JSON value is not an object",
                    "raw_json_object": False,
                    "structural_extract": True,
                }
    return {
        "parsed": None,
        "parse_error": "unterminated JSON object",
        "raw_json_object": False,
        "structural_extract": False,
    }


def default_prediction(scenario_id: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "first_action": "",
        "high_salience": [],
        "low_salience": [],
        "symbol_attach": [],
        "symbol_reject": [],
        "counterfactual_answers": {},
        "commitment_level": "",
        "claim_boundary": "",
    }


def schema_like_output(parsed: dict[str, Any] | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    schema_keys = {"$schema", "type", "properties", "required", "additionalProperties"}
    return bool(schema_keys & set(parsed)) and not any(field in parsed for field in REQUIRED_MODEL_FIELDS)


def issue(reason: str, field: str, detail: str) -> dict[str, str]:
    return {"reason": reason, "field": field, "detail": detail}


def primary_reason(parse_error: str | None, issues: list[dict[str, str]]) -> str | None:
    reasons = [item["reason"] for item in issues]
    if parse_error:
        reasons.append("parse_error")
    for reason in REASON_PRECEDENCE:
        if reason in reasons:
            return reason
    return None


def coerce_prediction(
    scenario_id: str,
    parsed: dict[str, Any] | None,
    expected_cf_ids: list[str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    prediction = default_prediction(scenario_id)
    issues: list[dict[str, str]] = []
    if parsed is None:
        issues.append(issue("parse_error", "<object>", "no JSON object available"))
        return prediction, issues

    if schema_like_output(parsed):
        issues.append(issue("extra_wrapper_or_schema_output", "<object>", "schema-like JSON object"))

    for field in STRING_FIELDS:
        if field not in parsed:
            issues.append(issue("missing_field", field, "required string field missing"))
            continue
        value = parsed.get(field)
        if isinstance(value, str):
            prediction[field] = value
        else:
            issues.append(issue("wrong_type", field, f"expected string, got {type(value).__name__}"))

    for field in LIST_FIELDS:
        if field not in parsed:
            issues.append(issue("missing_field", field, "required list field missing"))
            continue
        value = parsed.get(field)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            prediction[field] = value
        elif not isinstance(value, list):
            issues.append(
                issue("list_field_not_list", field, f"expected list[str], got {type(value).__name__}")
            )
        else:
            issues.append(issue("wrong_type", field, "list contains non-string item"))

    if "commitment_level" in parsed and isinstance(parsed.get("commitment_level"), str):
        if parsed["commitment_level"] not in ALLOWED_COMMITMENT_LEVELS:
            issues.append(
                issue("invalid_enum", "commitment_level", f"invalid value {parsed['commitment_level']!r}")
            )

    if "counterfactual_answers" not in parsed:
        issues.append(
            issue("missing_field", "counterfactual_answers", "required counterfactual object missing")
        )
        return prediction, issues

    answers = parsed.get("counterfactual_answers")
    if isinstance(answers, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in answers.items()
    ):
        prediction["counterfactual_answers"] = answers
        expected = set(expected_cf_ids)
        actual = set(answers)
        if expected and actual != expected:
            issues.append(
                issue(
                    "counterfactual_answers_bad_shape",
                    "counterfactual_answers",
                    f"expected keys {sorted(expected)}, got {sorted(actual)}",
                )
            )
        invalid_values = {
            key: value for key, value in answers.items() if value not in ALLOWED_COUNTERFACTUAL_LABELS
        }
        if invalid_values:
            issues.append(
                issue(
                    "invalid_enum",
                    "counterfactual_answers",
                    f"invalid enum values {invalid_values}",
                )
            )
    else:
        issues.append(
            issue(
                "counterfactual_answers_bad_shape",
                "counterfactual_answers",
                "expected object mapping counterfactual id to enum string",
            )
        )

    return prediction, issues


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    prompt_mode: str,
) -> str:
    rendered = render_chat_prompt(tokenizer, prompt, prompt_mode)
    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    rows = read_jsonl(args.prompts, args.max_eval_rows)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, args.adapter)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid_count = 0
    raw_json_object_count = 0
    structural_extract_count = 0
    usable_prediction_count = 0
    parse_error_count = 0
    invalid_breakdown: Counter[str] = Counter()

    with args.out.open("w", encoding="utf-8") as pred_handle, args.invalid.open(
        "w", encoding="utf-8"
    ) as invalid_handle:
        for index, row in enumerate(rows, start=1):
            scenario_id = row.get("scenario_id")
            prompt = row.get("prompt")
            if not isinstance(scenario_id, str) or not isinstance(prompt, str):
                raise ValueError(f"{args.prompts}:{index}: scenario_id and prompt are required")
            print(f"[{index}/{len(rows)}] {scenario_id}", flush=True)
            raw = generate_one(model, tokenizer, prompt, args.max_new_tokens, args.prompt_mode)
            parsed_info = extract_first_json_object(raw)
            parsed = parsed_info["parsed"]
            parse_error = parsed_info["parse_error"]
            raw_json_object = bool(parsed_info["raw_json_object"])
            structural_extract = bool(parsed_info["structural_extract"])
            raw_json_object_count += 1 if raw_json_object else 0
            structural_extract_count += 1 if structural_extract else 0
            parse_error_count += 1 if parse_error else 0

            expected_cf_ids = extract_counterfactual_ids(prompt)
            prediction, issues = coerce_prediction(scenario_id, parsed, expected_cf_ids)
            reason = primary_reason(parse_error, issues)
            if reason:
                invalid_count += 1
                invalid_breakdown[reason] += 1
                invalid_handle.write(
                    compact_json(
                        {
                            "scenario_id": scenario_id,
                            "primary_reason": reason,
                            "parse_error": parse_error,
                            "raw_json_object": raw_json_object,
                            "structural_extract": structural_extract,
                            "expected_counterfactual_ids": expected_cf_ids,
                            "issues": issues,
                            "parsed": parsed,
                            "coerced_prediction": prediction,
                            "raw": raw,
                        }
                    )
                    + "\n"
                )
            else:
                usable_prediction_count += 1
            pred_handle.write(compact_json(prediction) + "\n")

    prompt_count = len(rows)
    summary = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "prompt_mode": args.prompt_mode,
        "scenario_id_source": "eval_row_external",
        "prompts": prompt_count,
        "predictions": prompt_count,
        "invalid": invalid_count,
        "invalid_shape_count": invalid_count,
        "valid_predictions": usable_prediction_count,
        "usable_prediction_count": usable_prediction_count,
        "usable_prediction_rate": usable_prediction_count / prompt_count if prompt_count else 0.0,
        "raw_json_object_count": raw_json_object_count,
        "raw_json_object_rate": raw_json_object_count / prompt_count if prompt_count else 0.0,
        "structural_extract_count": structural_extract_count,
        "parse_error_count": parse_error_count,
        "invalid_breakdown": dict(sorted(invalid_breakdown.items())),
        "repaired_json": structural_extract_count,
        "out": str(args.out),
        "invalid_out": str(args.invalid),
    }
    write_json(args.summary_out or (args.out.parent / "inference_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_inference(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
