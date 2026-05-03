#!/usr/bin/env python3
"""
D31A C-block stream tokenizer / embedder probe.

Shape:
    raw bytes -> sliding 8-byte AB/B64 windows -> C tokenizer -> TokenEvent stream

This is not a language worker. It proves that the frozen AB/B64 surface can feed
a stream tokenizer that emits word/number/operator/punctuation events plus C64
token embeddings and ALU route hints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ab_window_codec import ABWindowCodec, WINDOW_BYTES, verify_artifact  # noqa: E402


DEFAULT_SEED = 20260503
TOKEN_KINDS = ("WORD", "NUMBER", "OP", "PUNCT", "NEWLINE", "UNKNOWN")
ROUTES = ("LANG", "ALU", "MEM", "TRANSFORM", "UNKNOWN")
OP_WORDS = {
    "PLUS": "OP_ADD",
    "ADD": "OP_ADD",
    "MINUS": "OP_SUB",
    "SUBTRACT": "OP_SUB",
    "TIMES": "OP_MUL",
    "MULTIPLY": "OP_MUL",
    "MULTIPLIED": "OP_MUL",
}
OP_SYMBOLS = {
    "+": "OP_ADD",
    "-": "OP_SUB",
    "*": "OP_MUL",
    "^": "OP_XOR",
    "&": "OP_AND",
    "|": "OP_OR",
}
PUNCT = {",", ".", ":", ";", "(", ")"}
ASCII_SHADE = " .:-=+*#%@"


@dataclass(frozen=True)
class TokenEvent:
    sample_id: int
    token_index: int
    kind: str
    raw_text: str
    normalized: str
    start_byte: int
    end_byte: int
    route_hint: str
    source_window_start: int
    source_window_end: int
    c64_embedding: str


@dataclass(frozen=True)
class AluCall:
    sample_id: int
    call_index: int
    left: int
    op: str
    right: int
    result_mod256: int
    token_start_index: int
    token_end_index: int


@dataclass(frozen=True)
class SampleCase:
    sample_id: int
    text: str
    category: str
    is_boundary_case: bool


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def stable_hash_bytes(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def sparse_c64(kind: str, normalized: str, route_hint: str) -> list[int]:
    vec = [0] * 64
    if kind in TOKEN_KINDS:
        vec[TOKEN_KINDS.index(kind)] = 1
    if route_hint in ROUTES:
        vec[8 + ROUTES.index(route_hint)] = 1
    if kind == "NUMBER":
        vec[16] = 1
        try:
            value = int(normalized)
        except ValueError:
            value = 0
        vec[17] = 1 if value == 0 else -1
        vec[18] = 1 if value % 2 == 0 else -1
        vec[19] = 1 if value < 256 else -1
    elif kind == "OP":
        op_offset = {
            "OP_ADD": 20,
            "OP_SUB": 21,
            "OP_MUL": 22,
            "OP_AND": 23,
            "OP_OR": 24,
            "OP_XOR": 25,
        }.get(normalized, 26)
        vec[op_offset] = 1
    elif kind == "PUNCT":
        vec[27] = 1
    elif kind == "NEWLINE":
        vec[28] = 1
    elif kind == "WORD":
        vec[29] = 1
    else:
        vec[30] = 1

    digest = stable_hash_bytes(f"{kind}:{normalized}:{route_hint}")
    used: set[int] = set()
    byte_idx = 0
    while len(used) < 6 and byte_idx < len(digest):
        lane = 32 + (digest[byte_idx] % 32)
        if lane not in used:
            sign = 1 if (digest[byte_idx] & 0x80) else -1
            vec[lane] = sign
            used.add(lane)
        byte_idx += 1
    return vec


def sparse_string(vec: Sequence[int]) -> str:
    return " ".join(f"{idx}:{value:+d}" for idx, value in enumerate(vec) if int(value) != 0)


def token_route(kind: str, normalized: str) -> str:
    if kind in ("NUMBER", "OP") and (kind != "OP" or normalized.startswith("OP_")):
        return "ALU"
    if kind == "WORD":
        return "LANG"
    if kind == "UNKNOWN":
        return "UNKNOWN"
    return "UNKNOWN"


def source_window_bounds(start_byte: int, end_byte: int, text_len: int) -> tuple[int, int]:
    if text_len <= 0:
        return 0, 0
    return max(0, min(start_byte, text_len - 1)), max(0, min(max(end_byte - 1, start_byte), text_len - 1))


def reference_tokenizer(text: str, sample_id: int = 0) -> list[TokenEvent]:
    events: list[TokenEvent] = []
    i = 0
    token_index = 0
    text_len = len(text)
    while i < text_len:
        ch = text[i]
        if ch in " \t\r":
            i += 1
            continue
        start = i
        if ch == "\n":
            raw = ch
            kind = "NEWLINE"
            normalized = "NEWLINE"
            i += 1
        elif ch.isalpha():
            while i < text_len and text[i].isalpha():
                i += 1
            raw = text[start:i]
            upper = raw.upper()
            if upper in OP_WORDS:
                kind = "OP"
                normalized = OP_WORDS[upper]
            else:
                kind = "WORD"
                normalized = upper
        elif ch.isdigit():
            while i < text_len and text[i].isdigit():
                i += 1
            raw = text[start:i]
            kind = "NUMBER"
            normalized = str(int(raw))
        elif ch in OP_SYMBOLS:
            i += 1
            raw = ch
            kind = "OP"
            normalized = OP_SYMBOLS[ch]
        elif ch in PUNCT:
            i += 1
            raw = ch
            kind = "PUNCT"
            normalized = ch
        else:
            i += 1
            raw = ch
            kind = "UNKNOWN"
            normalized = ch
        route = token_route(kind, normalized)
        source_start, source_end = source_window_bounds(start, i, text_len)
        events.append(
            TokenEvent(
                sample_id=sample_id,
                token_index=token_index,
                kind=kind,
                raw_text=raw,
                normalized=normalized,
                start_byte=start,
                end_byte=i,
                route_hint=route,
                source_window_start=source_start,
                source_window_end=source_end,
                c64_embedding=sparse_string(sparse_c64(kind, normalized, route)),
            )
        )
        token_index += 1
    return events


def alu_result(left: int, op: str, right: int) -> int:
    left &= 0xFF
    right &= 0xFF
    if op == "OP_ADD":
        return (left + right) & 0xFF
    if op == "OP_SUB":
        return (left - right) & 0xFF
    if op == "OP_MUL":
        return (left * right) & 0xFF
    if op == "OP_AND":
        return left & right
    if op == "OP_OR":
        return left | right
    if op == "OP_XOR":
        return left ^ right
    raise ValueError(f"unsupported ALU op: {op}")


def extract_alu_calls(events: Sequence[TokenEvent], sample_id: int) -> list[AluCall]:
    calls: list[AluCall] = []
    for idx in range(0, max(0, len(events) - 2)):
        a, op, b = events[idx], events[idx + 1], events[idx + 2]
        if (
            a.kind == "NUMBER"
            and op.kind == "OP"
            and b.kind == "NUMBER"
            and a.normalized.isdigit()
            and b.normalized.isdigit()
            and op.normalized.startswith("OP_")
        ):
            left = int(a.normalized)
            right = int(b.normalized)
            calls.append(
                AluCall(
                    sample_id=sample_id,
                    call_index=len(calls),
                    left=left,
                    op=op.normalized,
                    right=right,
                    result_mod256=alu_result(left, op.normalized, right),
                    token_start_index=idx,
                    token_end_index=idx + 2,
                )
            )
    return calls


def sliding_b64(text: str, codec: ABWindowCodec) -> list[list[int]]:
    raw = text.encode("ascii", errors="ignore")
    if not raw:
        raw = b" "
    latents: list[list[int]] = []
    for start in range(len(raw)):
        window = raw[start : start + WINDOW_BYTES]
        if len(window) < WINDOW_BYTES:
            window = window + b" " * (WINDOW_BYTES - len(window))
        latents.append(codec.encode_window_b64(window))
    return latents


def reconstruct_text_from_b64_stream(latents: Sequence[Sequence[int]], codec: ABWindowCodec) -> str:
    chars: list[int] = []
    for latent in latents:
        window = codec.decode_b64_to_window(latent)
        chars.append(int(window[0]))
    return bytes(chars).decode("ascii", errors="ignore")


def sparse_c_tokenizer(text: str, sample_id: int, codec: ABWindowCodec) -> list[TokenEvent]:
    return reference_tokenizer(reconstruct_text_from_b64_stream(sliding_b64(text, codec), codec), sample_id)


def shuffled_window_tokenizer(text: str, sample_id: int, codec: ABWindowCodec, rng: random.Random) -> list[TokenEvent]:
    latents = sliding_b64(text, codec)
    rng.shuffle(latents)
    return reference_tokenizer(reconstruct_text_from_b64_stream(latents, codec), sample_id)


def random_b64_tokenizer(text: str, sample_id: int, codec: ABWindowCodec, rng: random.Random) -> list[TokenEvent]:
    latents = sliding_b64(text, codec)
    random_latents = [[rng.choice((-1, 1)) for _ in latent] for latent in latents]
    return reference_tokenizer(reconstruct_text_from_b64_stream(random_latents, codec), sample_id)


def label_shuffle_events(events: Sequence[TokenEvent], rng: random.Random) -> list[TokenEvent]:
    kinds = [event.kind for event in events]
    if len(kinds) > 1:
        rng.shuffle(kinds)
    shuffled: list[TokenEvent] = []
    for event, kind in zip(events, kinds):
        route = token_route(kind, event.normalized)
        shuffled.append(
            TokenEvent(
                sample_id=event.sample_id,
                token_index=event.token_index,
                kind=kind,
                raw_text=event.raw_text,
                normalized=event.normalized,
                start_byte=event.start_byte,
                end_byte=event.end_byte,
                route_hint=route,
                source_window_start=event.source_window_start,
                source_window_end=event.source_window_end,
                c64_embedding=sparse_string(sparse_c64(kind, event.normalized, route)),
            )
        )
    return shuffled


def event_signature(events: Sequence[TokenEvent]) -> list[tuple[str, str, int, int]]:
    return [(event.kind, event.normalized, event.start_byte, event.end_byte) for event in events]


def alu_signature(calls: Sequence[AluCall]) -> list[tuple[int, str, int, int]]:
    return [(call.left, call.op, call.right, call.result_mod256) for call in calls]


def compare_events(expected: Sequence[TokenEvent], actual: Sequence[TokenEvent]) -> dict[str, float]:
    expected_sig = event_signature(expected)
    actual_sig = event_signature(actual)
    stream_exact = float(expected_sig == actual_sig)
    n = max(len(expected), len(actual), 1)
    boundary = 0
    kind = 0
    normalized = 0
    for idx in range(n):
        if idx < len(expected) and idx < len(actual):
            boundary += int((expected[idx].start_byte, expected[idx].end_byte) == (actual[idx].start_byte, actual[idx].end_byte))
            kind += int(expected[idx].kind == actual[idx].kind)
            normalized += int(expected[idx].normalized == actual[idx].normalized)
    return {
        "token_stream_exact": stream_exact,
        "token_boundary_acc": boundary / n,
        "token_kind_acc": kind / n,
        "normalized_acc": normalized / n,
    }


def make_word(rng: random.Random) -> str:
    return rng.choice(
        [
            "Hello",
            "world",
            "Give",
            "me",
            "apples",
            "I",
            "need",
            "exactly",
            "blue",
            "cat",
            "river",
            "signal",
            "token",
        ]
    )


def make_expression(rng: random.Random) -> tuple[str, int]:
    left = rng.randrange(0, 256)
    right = rng.randrange(0, 256)
    op_text, op_norm = rng.choice(
        [
            ("+", "OP_ADD"),
            ("-", "OP_SUB"),
            ("*", "OP_MUL"),
            ("times", "OP_MUL"),
            ("plus", "OP_ADD"),
            ("minus", "OP_SUB"),
            ("^", "OP_XOR"),
            ("&", "OP_AND"),
            ("|", "OP_OR"),
        ]
    )
    if op_text.isalpha():
        text = f"{left} {op_text} {right}"
    elif rng.random() < 0.5:
        text = f"{left}{op_text}{right}"
    else:
        text = f"{left} {op_text} {right}"
    return text, alu_result(left, op_norm, right)


def build_cases(samples: int, boundary_cases: int, seed: int) -> list[SampleCase]:
    rng = random.Random(seed)
    cases: list[SampleCase] = []
    fixed = [
        ("Hello, world.", "word_punct", False),
        ("25 times 7", "alu_word_op", False),
        ("25*7", "alu_symbol_op", False),
        ("25 + 7", "alu_symbol_op", False),
        ("Give me apples, i need EXACTLY 25 times 7...", "embedded_alu", True),
        ("THE+CAT", "ambiguous", False),
        ("ABC123", "ambiguous", False),
        ("Line one\nLine two.", "newline", False),
    ]
    for text, category, is_boundary in fixed:
        cases.append(SampleCase(len(cases), text, category, is_boundary))

    while len(cases) < samples:
        choice = rng.randrange(5)
        if choice == 0:
            text = f"{make_word(rng)}, {make_word(rng)}."
            category = "word_punct"
        elif choice == 1:
            expression, _ = make_expression(rng)
            text = expression
            category = "alu_expr"
        elif choice == 2:
            expression, _ = make_expression(rng)
            text = f"{make_word(rng)} {make_word(rng)}, i need exactly {expression}."
            category = "embedded_alu"
        elif choice == 3:
            text = f"{make_word(rng)}\n{make_word(rng)}."
            category = "newline"
        else:
            text = rng.choice(["THE+CAT", "ABC123", "A+BIRD", "@#??!!", "cat_42"])
            category = "ambiguous"
        cases.append(SampleCase(len(cases), text, category, False))

    for idx in range(boundary_cases):
        expression, _ = make_expression(rng)
        prefix_len = (idx % 8) + 1
        prefix = "x" * prefix_len
        suffix = rng.choice([" done.", " please.", " now.", "..."])
        text = f"{prefix} {expression}{suffix}"
        cases.append(SampleCase(len(cases), text, "boundary_alu", True))
    return cases


def evaluate_cases(cases: Sequence[SampleCase], codec: ABWindowCodec) -> tuple[list[dict[str, object]], list[TokenEvent], list[dict[str, object]], list[AluCall]]:
    sample_rows: list[dict[str, object]] = []
    token_rows: list[TokenEvent] = []
    span_rows: list[dict[str, object]] = []
    alu_rows: list[AluCall] = []
    for case in cases:
        expected = reference_tokenizer(case.text, case.sample_id)
        actual = sparse_c_tokenizer(case.text, case.sample_id, codec)
        expected_alu = extract_alu_calls(expected, case.sample_id)
        actual_alu = extract_alu_calls(actual, case.sample_id)
        cmp = compare_events(expected, actual)
        alu_exact = float(alu_signature(expected_alu) == alu_signature(actual_alu))
        row = {
            "sample_id": case.sample_id,
            "category": case.category,
            "is_boundary_case": case.is_boundary_case,
            "text": case.text,
            "token_count": len(actual),
            "alu_call_count": len(actual_alu),
            "alu_call_exact": alu_exact,
            **cmp,
        }
        sample_rows.append(row)
        token_rows.extend(actual)
        span_rows.append(
            {
                "sample_id": case.sample_id,
                "category": case.category,
                "text": case.text,
                "tokens": " | ".join(f"{event.kind}:{event.normalized}" for event in actual),
                "alu_calls": " | ".join(f"{call.left} {call.op} {call.right} -> {call.result_mod256}" for call in actual_alu),
            }
        )
        alu_rows.extend(actual_alu)
    return sample_rows, token_rows, span_rows, alu_rows


def control_rows(cases: Sequence[SampleCase], codec: ABWindowCodec, repeats: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    control_names = ("window_shuffle", "random_b64_projection", "label_shuffle")
    for repeat in range(repeats):
        rng = random.Random(seed + 1000 + repeat)
        for control_name in control_names:
            exact = 0
            alu_exact = 0
            alu_positive_exact = 0
            alu_positive_count = 0
            boundary_acc = 0.0
            kind_acc = 0.0
            normalized_acc = 0.0
            for case in cases:
                expected = reference_tokenizer(case.text, case.sample_id)
                if control_name == "window_shuffle":
                    actual = shuffled_window_tokenizer(case.text, case.sample_id, codec, rng)
                elif control_name == "random_b64_projection":
                    actual = random_b64_tokenizer(case.text, case.sample_id, codec, rng)
                else:
                    actual = label_shuffle_events(sparse_c_tokenizer(case.text, case.sample_id, codec), rng)
                cmp = compare_events(expected, actual)
                exact += int(cmp["token_stream_exact"] == 1.0)
                boundary_acc += cmp["token_boundary_acc"]
                kind_acc += cmp["token_kind_acc"]
                normalized_acc += cmp["normalized_acc"]
                expected_alu = extract_alu_calls(expected, case.sample_id)
                actual_alu = extract_alu_calls(actual, case.sample_id)
                alu_matches = int(alu_signature(expected_alu) == alu_signature(actual_alu))
                alu_exact += alu_matches
                if expected_alu:
                    alu_positive_exact += alu_matches
                    alu_positive_count += 1
            count = max(1, len(cases))
            token_stream_exact_acc = exact / count
            alu_call_exact_acc = alu_exact / count
            alu_positive_exact_acc = alu_positive_exact / max(1, alu_positive_count)
            rows.append(
                {
                    "control_name": control_name,
                    "repeat": repeat,
                    "sample_count": count,
                    "token_stream_exact_acc": token_stream_exact_acc,
                    "token_boundary_acc": boundary_acc / count,
                    "token_kind_acc": kind_acc / count,
                    "normalized_acc": normalized_acc / count,
                    "alu_call_exact_acc": alu_call_exact_acc,
                    "alu_positive_count": alu_positive_count,
                    "alu_positive_exact_acc": alu_positive_exact_acc,
                    "status": "CONTROL_PASS"
                    if token_stream_exact_acc <= 0.25 and alu_positive_exact_acc <= 0.35
                    else "CONTROL_LEAK",
                }
            )
    return rows


def aggregate(sample_rows: Sequence[dict[str, object]], controls: Sequence[dict[str, object]]) -> dict[str, object]:
    n = max(1, len(sample_rows))
    boundary_rows = [row for row in sample_rows if bool(row["is_boundary_case"])]
    normal = {
        "sample_count": n,
        "token_stream_exact_acc": sum(float(row["token_stream_exact"]) for row in sample_rows) / n,
        "token_boundary_acc": sum(float(row["token_boundary_acc"]) for row in sample_rows) / n,
        "token_kind_acc": sum(float(row["token_kind_acc"]) for row in sample_rows) / n,
        "normalized_acc": sum(float(row["normalized_acc"]) for row in sample_rows) / n,
        "alu_call_exact_acc": sum(float(row["alu_call_exact"]) for row in sample_rows) / n,
        "boundary_case_exact_acc": sum(float(row["token_stream_exact"]) for row in boundary_rows) / max(1, len(boundary_rows)),
        "boundary_case_count": len(boundary_rows),
        "max_control_token_exact": max((float(row["token_stream_exact_acc"]) for row in controls), default=0.0),
        "max_control_alu_exact": max((float(row["alu_call_exact_acc"]) for row in controls), default=0.0),
        "max_control_alu_positive_exact": max((float(row["alu_positive_exact_acc"]) for row in controls), default=0.0),
    }
    if (
        normal["token_stream_exact_acc"] == 1.0
        and normal["token_boundary_acc"] == 1.0
        and normal["token_kind_acc"] == 1.0
        and normal["normalized_acc"] == 1.0
        and normal["alu_call_exact_acc"] == 1.0
        and normal["boundary_case_exact_acc"] == 1.0
        and normal["max_control_token_exact"] <= 0.25
        and normal["max_control_alu_positive_exact"] <= 0.35
    ):
        verdict = "D31A_CBLOCK_TOKENIZER_PASS"
    elif any(str(row["status"]) == "CONTROL_LEAK" for row in controls):
        verdict = "D31A_CBLOCK_CONTROL_LEAK"
    elif normal["token_stream_exact_acc"] == 1.0 and normal["boundary_case_exact_acc"] < 1.0:
        verdict = "D31A_CBLOCK_WEAK_PASS"
    else:
        verdict = "D31A_CBLOCK_B64_ROUTE_FAIL"
    normal["verdict"] = verdict
    return normal


def write_csv(path: Path, rows: Sequence[dict[str, object] | TokenEvent | AluCall]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    dict_rows: list[dict[str, object]] = [asdict(row) if not isinstance(row, dict) else row for row in rows]
    fields: list[str] = []
    for row in dict_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict_rows)


def write_report(path: Path, summary: dict[str, object], controls: Sequence[dict[str, object]], examples: Sequence[dict[str, object]]) -> None:
    lines = [
        "# D31A C-Block Stream Tokenizer Report",
        "",
        "## Verdict",
        "",
        "```text",
        str(summary["verdict"]),
        "```",
        "",
        "## Summary",
        "",
        "```text",
        f"sample_count: {summary['sample_count']}",
        f"token_stream_exact_acc: {summary['token_stream_exact_acc']:.6f}",
        f"token_boundary_acc: {summary['token_boundary_acc']:.6f}",
        f"token_kind_acc: {summary['token_kind_acc']:.6f}",
        f"normalized_acc: {summary['normalized_acc']:.6f}",
        f"alu_call_exact_acc: {summary['alu_call_exact_acc']:.6f}",
        f"boundary_case_exact_acc: {summary['boundary_case_exact_acc']:.6f}",
        f"max_control_token_exact: {summary['max_control_token_exact']:.6f}",
        f"max_control_alu_exact: {summary['max_control_alu_exact']:.6f}",
        f"max_control_alu_positive_exact: {summary['max_control_alu_positive_exact']:.6f}",
        "```",
        "",
        "## Control Rows",
        "",
        "```text",
    ]
    for row in controls:
        lines.append(
            f"{row['control_name']} repeat={row['repeat']} "
            f"token_exact={float(row['token_stream_exact_acc']):.4f} "
            f"alu_positive_exact={float(row['alu_positive_exact_acc']):.4f} {row['status']}"
        )
    lines.extend(["```", "", "## Integration Examples", "", "```text"])
    for row in examples:
        lines.append(f"{row['text']} -> {row['tokens']} :: {row['alu_calls']}")
    lines.extend(
        [
            "```",
            "",
            "## Interpretation",
            "",
            "D31A proves that the frozen B64 stream can feed a tokenizer/embedder layer.",
            "It is still a probe/reference implementation, not a language worker.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shade(value: float) -> str:
    idx = min(len(ASCII_SHADE) - 1, max(0, int(round(value * (len(ASCII_SHADE) - 1)))))
    return ASCII_SHADE[idx]


def heatmap(summary: dict[str, object], controls: Sequence[dict[str, object]]) -> str:
    lines = ["D31A C-block tokenizer: brighter = better pass metric"]
    metrics = [
        ("token_stream", float(summary["token_stream_exact_acc"])),
        ("boundary", float(summary["token_boundary_acc"])),
        ("kind", float(summary["token_kind_acc"])),
        ("normalized", float(summary["normalized_acc"])),
        ("alu_calls", float(summary["alu_call_exact_acc"])),
        ("boundary_cases", float(summary["boundary_case_exact_acc"])),
    ]
    for name, value in metrics:
        lines.append(f"{name:<16} [{shade(value) * 20:<20}] {value:.4f}")
    lines.append("controls lower is better")
    for row in controls:
        value = float(row["token_stream_exact_acc"])
        lines.append(f"{str(row['control_name'])[:16]:<16} [{shade(1.0 - value) * 20:<20}] exact={value:.4f}")
    return "\n".join(lines)


def integration_cases(raw_examples: str) -> list[SampleCase]:
    examples = parse_csv(raw_examples)
    return [SampleCase(idx, text, "integration", "times" in text or "*" in text or "+" in text) for idx, text in enumerate(examples)]


def run(args: argparse.Namespace) -> dict[str, object]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = Path(args.artifact)
    checked_artifact(artifact_path)
    codec = ABWindowCodec()
    if args.mode == "integration-smoke":
        cases = integration_cases(args.examples)
        repeats = 1
    elif args.mode == "smoke":
        cases = build_cases(int(args.samples), int(args.boundary_cases), int(args.seed))
        repeats = max(1, int(args.control_repeats))
    else:
        cases = build_cases(int(args.samples), int(args.boundary_cases), int(args.seed))
        repeats = max(1, int(args.control_repeats))

    sample_rows, token_rows, span_rows, alu_rows = evaluate_cases(cases, codec)
    controls = control_rows(cases, codec, repeats, int(args.seed)) if args.mode != "integration-smoke" else []
    summary = aggregate(sample_rows, controls)
    examples = span_rows[: min(24, len(span_rows))]

    write_csv(out / "token_events.csv", token_rows)
    write_csv(out / "span_events.csv", span_rows)
    write_csv(out / "alu_calls.csv", alu_rows)
    write_csv(out / "token_embedding_samples.csv", token_rows[: min(512, len(token_rows))])
    write_csv(out / "tokenizer_controls.csv", controls)
    payload = {
        **summary,
        "mode": args.mode,
        "artifact": str(artifact_path),
        "c64_layout": {
            "0_7": "token kind lanes",
            "8_15": "route lanes",
            "16_31": "operator/punctuation/numeric feature lanes",
            "32_63": "stable sparse signed hash lanes",
        },
    }
    (out / "cblock_top.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out / "D31A_CBLOCK_STREAM_TOKENIZER_REPORT.md", summary, controls, examples)
    (out / "cblock_heatmap.txt").write_text(heatmap(summary, controls) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(heatmap(summary, controls))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "main", "integration-smoke"), required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--boundary-cases", type=int, default=4096)
    parser.add_argument("--control-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact", default="tools/ab_window_codec_v1.json")
    parser.add_argument("--examples", default="25 times 7,25*7,Give me exactly 25 times 7.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = run(args)
    if args.mode in ("smoke", "main") and str(payload["verdict"]) not in ("D31A_CBLOCK_TOKENIZER_PASS", "D31A_CBLOCK_WEAK_PASS"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
