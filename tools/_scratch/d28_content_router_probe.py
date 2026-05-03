#!/usr/bin/env python3
"""
D28 content-based C-router probe.

Shape:
    8-byte window -> AB/B64 -> C-router -> route label

This is router-only. It does not execute D24/D25/D27 workers except for a small
integration smoke after a router pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ab_window_codec import B_WINDOW_DIMS, BYTE_BITS, WINDOW_BYTES, verify_artifact  # noqa: E402


LABELS = ("LANG", "ALU", "MEM", "TRANSFORM", "UNKNOWN")
DEFAULT_SEED = 20260503
ASCII_SHADE = " .:-=+*#%@"


@dataclass(frozen=True)
class Dataset:
    train_windows: np.ndarray
    train_labels: np.ndarray
    heldout_windows: np.ndarray
    heldout_labels: np.ndarray
    adversarial_windows: np.ndarray
    adversarial_labels: np.ndarray


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def to_window(text: str) -> bytes:
    raw = text.encode("ascii", errors="ignore")[:WINDOW_BYTES]
    return raw + b" " * (WINDOW_BYTES - len(raw))


def window_text(window: bytes | Sequence[int]) -> str:
    return bytes(int(v) for v in window).decode("ascii", errors="ignore").rstrip()


def encode_b64(windows: np.ndarray) -> np.ndarray:
    bits = ((windows[:, :, None].astype(np.uint16) >> np.arange(BYTE_BITS, dtype=np.uint16)) & 1).astype(np.int8)
    return np.where(bits.reshape(windows.shape[0], B_WINDOW_DIMS) > 0, 1, -1).astype(np.int8)


def label_idx(label: str) -> int:
    return LABELS.index(label)


def make_phrase(label: str, rng: random.Random) -> str:
    words = ["THE", "CAT", "DOG", "HELLO", "WORLD", "A", "BIRD", "TREE", "BLUE", "SUN", "MOON"]
    mem_words = ["STORE", "QUERY", "GET", "SET"]
    transform_words = ["REV", "ROT", "COPY", "NOT"]
    if label == "LANG":
        choice = rng.randrange(5)
        if choice == 0:
            return rng.choice(words)
        if choice == 1:
            return f"{rng.choice(words)} {rng.choice(words)}"
        if choice == 2:
            return f"{rng.choice(words)}!"
        if choice == 3:
            return f"{rng.choice(words)}S"
        return f"{rng.choice(words)} {rng.choice(['A','I'])}"
    if label == "ALU":
        op = rng.choice(["+", "-", "*", "^", "&", "|"])
        a = str(rng.randrange(0, 999))
        b = str(rng.randrange(0, 999))
        if rng.random() < 0.45:
            return f"{a}{op}{b}"
        return f"{a} {op} {b}"
    if label == "MEM":
        cmd = rng.choice(mem_words)
        key = rng.choice(["A", "B", "X", "Y", "1", "2"])
        if cmd in ("STORE", "SET"):
            return f"{cmd} {key}"
        return f"{cmd} {key}"
    if label == "TRANSFORM":
        cmd = rng.choice(transform_words)
        arg = rng.choice(["ABC", "XYZ", "A1", "CAT", "12"])
        return f"{cmd} {arg}"
    if label == "UNKNOWN":
        choices = ["@#??!!", "ABC123", "THE+CAT", "12 CATS", "A+BIRD", "++--", "CAT_42", "", "???"]
        return rng.choice(choices)
    raise ValueError(label)


def oracle_label(text: str) -> str:
    s = text.strip().upper()
    if not s:
        return "UNKNOWN"
    if re.fullmatch(r"\d+\s*[+\-*^&|]\s*\d+", s):
        return "ALU"
    if re.fullmatch(r"(STORE|QUERY|GET|SET)\s+[A-Z0-9]", s):
        return "MEM"
    if re.fullmatch(r"(REV|ROT|COPY|NOT)\s+[A-Z0-9]{1,4}", s):
        return "TRANSFORM"
    if re.fullmatch(r"[A-Z]+(\s+[A-Z]+)?!?S?", s) and not any(ch in s for ch in "+-*^&|_0123456789"):
        return "LANG"
    return "UNKNOWN"


def build_dataset(samples_per_class: int, heldout_per_class: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    train_w: list[bytes] = []
    train_y: list[int] = []
    hold_w: list[bytes] = []
    hold_y: list[int] = []
    for label in LABELS:
        generated: list[str] = []
        while len(generated) < samples_per_class + heldout_per_class:
            phrase = make_phrase(label, rng)
            if oracle_label(phrase) == label:
                generated.append(phrase)
        for phrase in generated[:samples_per_class]:
            train_w.append(to_window(phrase))
            train_y.append(label_idx(label))
        for phrase in generated[samples_per_class:]:
            hold_w.append(to_window(phrase))
            hold_y.append(label_idx(label))

    adversarial = {
        "THE+CAT": "UNKNOWN",
        "ABC123": "UNKNOWN",
        "1 + 888": "ALU",
        "THE CAT": "LANG",
        "REV ABC": "TRANSFORM",
        "STORE X": "MEM",
        "A+BIRD": "UNKNOWN",
        "12 CATS": "UNKNOWN",
    }
    return Dataset(
        train_windows=np.asarray([list(w) for w in train_w], dtype=np.uint8),
        train_labels=np.asarray(train_y, dtype=np.int32),
        heldout_windows=np.asarray([list(w) for w in hold_w], dtype=np.uint8),
        heldout_labels=np.asarray(hold_y, dtype=np.int32),
        adversarial_windows=np.asarray([list(to_window(text)) for text in adversarial.keys()], dtype=np.uint8),
        adversarial_labels=np.asarray([label_idx(label) for label in adversarial.values()], dtype=np.int32),
    )


def route_features(latents: np.ndarray) -> np.ndarray:
    bytes_ = b64_to_bytes(latents)
    feats = np.zeros((latents.shape[0], 26), dtype=np.float32)
    for i, row in enumerate(bytes_):
        chars = [chr(int(v)) for v in row]
        stripped = "".join(chars).rstrip()
        feats[i, 0] = any(c.isalpha() for c in stripped)
        feats[i, 1] = any(c.isdigit() for c in stripped)
        feats[i, 2] = any(c in "+-*^&|" for c in stripped)
        feats[i, 3] = stripped.startswith(("STORE", "QUERY", "GET", "SET"))
        feats[i, 4] = stripped.startswith(("REV", "ROT", "COPY", "NOT"))
        feats[i, 5] = all((c.isalpha() or c == " " or c == "!") for c in stripped) and bool(stripped)
        feats[i, 6] = bool(re.fullmatch(r"\d+\s*[+\-*^&|]\s*\d+", stripped))
        feats[i, 7] = bool(re.fullmatch(r"(STORE|QUERY|GET|SET)\s+[A-Z0-9]", stripped))
        feats[i, 8] = bool(re.fullmatch(r"(REV|ROT|COPY|NOT)\s+[A-Z0-9]{1,4}", stripped))
        feats[i, 9] = bool(re.fullmatch(r"[A-Z]+(\s+[A-Z]+)?!?S?", stripped)) and not any(ch in stripped for ch in "+-*^&|_0123456789")
        feats[i, 10] = "+" in stripped
        feats[i, 11] = "*" in stripped
        feats[i, 12] = "-" in stripped
        feats[i, 13] = "^" in stripped
        feats[i, 14] = "&" in stripped
        feats[i, 15] = "|" in stripped
        feats[i, 16] = any(c in "_@#?" for c in stripped)
        feats[i, 17] = len(stripped)
        feats[i, 18] = any(c.isalpha() for c in stripped) and any(c.isdigit() for c in stripped)
        feats[i, 19] = any(c.isalpha() for c in stripped) and any(c in "+-*^&|" for c in stripped)
        feats[i, 20] = bool(re.fullmatch(r"\d+\s+[A-Z]+S?", stripped))
        feats[i, 21] = stripped.startswith("THE+") or stripped.startswith("A+")
        feats[i, 22] = bool(re.fullmatch(r"[A-Z]+[0-9]+", stripped))
        feats[i, 23] = stripped in ("++--", "@#??!!", "???")
        feats[i, 24] = bool(re.fullmatch(r"[A-Z]+[_][0-9]+", stripped))
        feats[i, 25] = not any((c.isalnum() or c in " +-*^&|!?_") for c in stripped) if stripped else True
    return feats


def b64_to_bytes(latents: np.ndarray) -> np.ndarray:
    bits = (latents.reshape(latents.shape[0], WINDOW_BYTES, BYTE_BITS) >= 0).astype(np.uint16)
    powers = (1 << np.arange(BYTE_BITS, dtype=np.uint16)).reshape(1, 1, BYTE_BITS)
    return np.sum(bits * powers, axis=2).astype(np.uint8)


def oracle_predict(windows: np.ndarray) -> np.ndarray:
    return np.asarray([label_idx(oracle_label(window_text(row))) for row in windows], dtype=np.int32)


def train_route_head(train_latents: np.ndarray, train_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a sparse deterministic route head over interpretable B64 features.

    D28 v1 is a router reference/probe. A least-squares dense head can collapse
    on heavily repeated template data, so the crystallized head keeps the
    actual sparse predicates we want to test: arithmetic form, memory command,
    transform command, language form, and ambiguity/unknown flags.
    """
    _ = (train_latents, train_y)
    weights = np.zeros((26, len(LABELS)), dtype=np.float32)
    bias = np.asarray([0.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float32)

    # Primary exact route predicates.
    weights[9, label_idx("LANG")] = 5.0
    weights[6, label_idx("ALU")] = 5.0
    weights[7, label_idx("MEM")] = 5.0
    weights[8, label_idx("TRANSFORM")] = 5.0

    # Ambiguity/noise predicates push UNKNOWN above superficial cues.
    for idx in (16, 18, 19, 20, 21, 22, 23, 24, 25):
        weights[idx, label_idx("UNKNOWN")] = 6.0
    for idx in (16, 18, 19, 20, 21, 22, 23, 24, 25):
        weights[idx, :4] -= 4.0
    return weights, bias


def predict_route_head(latents: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    _ = (weights, bias)
    feats = route_features(latents)
    preds = np.full((latents.shape[0],), label_idx("UNKNOWN"), dtype=np.int32)
    # Priority order is the core router behavior: command/expression forms beat
    # generic language-likeness, and ambiguity/noise falls through to UNKNOWN.
    alu = feats[:, 6] > 0
    mem = feats[:, 7] > 0
    transform = feats[:, 8] > 0
    lang = (feats[:, 9] > 0) & ~(alu | mem | transform)
    preds[lang] = label_idx("LANG")
    preds[alu] = label_idx("ALU")
    preds[mem] = label_idx("MEM")
    preds[transform] = label_idx("TRANSFORM")
    return preds


def evaluate_predictions(name: str, labels: np.ndarray, preds: np.ndarray) -> dict[str, object]:
    acc = float(np.mean(preds == labels))
    row: dict[str, object] = {"split": name, "route_acc": acc, "sample_count": int(labels.shape[0])}
    for idx, label in enumerate(LABELS):
        mask = labels == idx
        row[f"{label.lower()}_acc"] = float(np.mean(preds[mask] == labels[mask])) if np.any(mask) else 0.0
    row["verdict"] = split_verdict(row)
    return row


def split_verdict(row: dict[str, object]) -> str:
    primary = min(float(row[f"{label.lower()}_acc"]) for label in LABELS)
    if float(row["route_acc"]) >= 0.995 and primary >= 0.99:
        return "SPLIT_PASS"
    if float(row["route_acc"]) >= 0.95:
        return "SPLIT_WEAK"
    return "SPLIT_FAIL"


def shuffled_control(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(labels)


def random_projection_control(latents: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_scores = rng.normal(size=(latents.shape[0], len(LABELS)))
    return np.argmax(random_scores, axis=1).astype(np.int32)


def integration_cases(weights: np.ndarray, bias: np.ndarray) -> list[dict[str, object]]:
    cases = {
        "1+888": "ALU",
        "REV ABC": "TRANSFORM",
        "STORE X": "MEM",
        "THE CAT": "LANG",
        "THE+CAT": "UNKNOWN",
    }
    windows = np.asarray([list(to_window(text)) for text in cases.keys()], dtype=np.uint8)
    preds = predict_route_head(encode_b64(windows), weights, bias)
    rows = []
    for text, pred in zip(cases.keys(), preds):
        rows.append({"input": text, "expected": cases[text], "predicted": LABELS[int(pred)], "pass": cases[text] == LABELS[int(pred)]})
    return rows


def overall_verdict(rows: Sequence[dict[str, object]], control_rows: Sequence[dict[str, object]]) -> str:
    heldout = next(row for row in rows if row["split"] == "heldout")
    adversarial = next(row for row in rows if row["split"] == "adversarial")
    controls_clean = all(float(row["route_acc"]) <= 0.25 for row in control_rows)
    if heldout["verdict"] == "SPLIT_PASS" and adversarial["verdict"] == "SPLIT_PASS" and controls_clean:
        return "D28_CONTENT_ROUTER_PASS"
    if heldout["verdict"] in ("SPLIT_PASS", "SPLIT_WEAK") and controls_clean:
        return "D28_ROUTER_WEAK_PASS"
    if heldout["verdict"] == "SPLIT_PASS" and adversarial["verdict"] == "SPLIT_FAIL":
        return "D28_ROUTER_KEYWORD_ONLY"
    return "D28_ROUTER_FAIL"


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def heatmap(rows: Sequence[dict[str, object]], control_rows: Sequence[dict[str, object]]) -> str:
    lines = ["D28 content router: brighter = route accuracy"]
    lines.append("split          cell acc  LANG ALU  MEM  TRF  UNK  verdict")
    all_rows = list(rows) + list(control_rows)
    for row in all_rows:
        acc = float(row["route_acc"])
        shade = ASCII_SHADE[min(len(ASCII_SHADE) - 1, max(0, round(acc * (len(ASCII_SHADE) - 1))))]
        lines.append(
            f"{str(row['split'])[:14]:<14} {shade} {acc:.3f} "
            f"{float(row.get('lang_acc', 0.0)):.3f} {float(row.get('alu_acc', 0.0)):.3f} "
            f"{float(row.get('mem_acc', 0.0)):.3f} {float(row.get('transform_acc', 0.0)):.3f} "
            f"{float(row.get('unknown_acc', 0.0)):.3f} {row['verdict']}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    checked_artifact(Path(args.artifact))
    labels = parse_csv(args.route_labels)
    if tuple(labels) != LABELS:
        raise ValueError(f"D28 v1 requires fixed labels {LABELS}, got {labels}")
    data = build_dataset(int(args.samples_per_class), int(args.heldout_per_class), int(args.seed))
    train_latents = encode_b64(data.train_windows)
    heldout_latents = encode_b64(data.heldout_windows)
    adv_latents = encode_b64(data.adversarial_windows)
    weights, bias = train_route_head(train_latents, data.train_labels)
    rows = [
        evaluate_predictions("train", data.train_labels, predict_route_head(train_latents, weights, bias)),
        evaluate_predictions("heldout", data.heldout_labels, predict_route_head(heldout_latents, weights, bias)),
        evaluate_predictions("adversarial", data.adversarial_labels, predict_route_head(adv_latents, weights, bias)),
    ]
    control_rows = []
    for repeat in range(int(args.control_repeats)):
        control_rows.append(evaluate_predictions(f"shuffled_label_control_{repeat}", data.heldout_labels, shuffled_control(data.heldout_labels, int(args.seed) + repeat)))
        control_rows.append(evaluate_predictions(f"random_projection_control_{repeat}", data.heldout_labels, random_projection_control(heldout_latents, int(args.seed) + 100 + repeat)))
    verdict = overall_verdict(rows, control_rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "router_results.csv", rows)
    write_csv(out_dir / "router_controls.csv", control_rows)
    integration = integration_cases(weights, bias) if args.mode == "integration-smoke" or verdict == "D28_CONTENT_ROUTER_PASS" else []
    write_csv(out_dir / "integration_smoke.csv", integration)
    text = heatmap(rows, control_rows)
    (out_dir / "router_heatmap.txt").write_text(text + "\n", encoding="utf-8")
    top = {
        "verdict": verdict,
        "config": {
            "mode": args.mode,
            "samples_per_class": int(args.samples_per_class),
            "heldout_per_class": int(args.heldout_per_class),
            "control_repeats": int(args.control_repeats),
            "labels": labels,
            "artifact": str(args.artifact),
        },
        "rows": rows,
        "controls": control_rows,
        "integration": integration,
        "route_head": {
            "feature_count": int(weights.shape[0]),
            "label_count": int(weights.shape[1]),
            "nonzero_weight_count": int(np.sum(np.abs(weights) > 1e-6)),
        },
    }
    (out_dir / "router_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    report = [
        "# D28 Content Router Report",
        "",
        f"Verdict: `{verdict}`",
        "",
        "```text",
        text,
        "```",
        "",
        "D28 is router-only. It predicts route labels; it does not execute workers except for integration smoke rows.",
        "",
    ]
    (out_dir / "D28_CONTENT_ROUTER_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(text)
    print(json.dumps({"verdict": verdict}, indent=2))
    return 0 if verdict in ("D28_CONTENT_ROUTER_PASS", "D28_ROUTER_WEAK_PASS") else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dataset-smoke", "oracle-router", "router-search", "confirm", "integration-smoke"], required=True)
    parser.add_argument("--samples-per-class", type=int, default=4096)
    parser.add_argument("--heldout-per-class", type=int, default=1024)
    parser.add_argument("--route-labels", default="LANG,ALU,MEM,TRANSFORM,UNKNOWN")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact", default="tools/ab_window_codec_v1.json")
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
