#!/usr/bin/env python3
"""Shared raw generation helper for rebuilt raw-evidence milestones.

The helper exposes one strict request interface and rejects any expected-answer,
scorer, label, or oracle material before generation. It is intentionally
standalone and does not import old phase runners.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - exercised by fail-closed callers.
    torch = None
    nn = None


HELPER_VERSION = "shared_raw_generation_helper_v1"
HELPER_BACKEND = "repo_local_checkpoint_byte_lm"
INSTNCT_MUTATION_BACKEND = "repo_local_instnct_mutation_graph"
RULE_SELECTED_POCKET_BINDING_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
REPO_ROOT = Path(__file__).resolve().parents[2]
BYTE_VOCAB_SIZE = 256
ALLOWED_GENERATION_CONFIG_KEYS = {"temperature", "device", "stop_on_newline"}

ALLOWED_REQUEST_KEYS = {
    "prompt",
    "checkpoint_path",
    "checkpoint_hash",
    "seed",
    "max_new_tokens",
    "generation_config",
}
FORBIDDEN_REQUEST_KEYS = {
    "expected_output",
    "expected_payload",
    "expected_answer",
    "required_keys",
    "required_keywords",
    "forbidden_outputs",
    "schema_answer_object",
    "scorer_metadata",
    "labels",
    "oracle_data",
    "eval_family_expected_values",
    "row_answer",
    "target_json",
    "gold_output",
    "eval_family",
    "answer",
    "expected_values",
}
GRU_STATE_KEYS = {
    "embedding.weight",
    "rnn.weight_ih_l0",
    "rnn.weight_hh_l0",
    "rnn.bias_ih_l0",
    "rnn.bias_hh_l0",
    "head.weight",
    "head.bias",
}
MLP_STATE_KEYS = {
    "embedding.weight",
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
    "net.4.weight",
    "net.4.bias",
}
DEFAULT_CHECKPOINT_CANDIDATES = [
    Path("target/pilot_wave/stable_loop_phase_lock_102_decoder_policy_and_rollout_repair/smoke/checkpoints/decoder_policy_rollout_repair/model.pt"),
    Path("target/pilot_wave/stable_loop_phase_lock_100_open_vocab_assistant_capability_scale/smoke/checkpoints/open_vocab_assistant_scale/model.pt"),
    Path("target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/smoke/checkpoints/open_vocab_chat_sft_mix/model.pt"),
    Path("target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke/checkpoints/open_vocab_byte_lm/model.pt"),
]
INSTNCT_VALUE_RE = re.compile(r"\b(?:EV|VAL|SYM)[A-Za-z0-9_+\-]*\b")
INSTNCT_TRAIN_VALUE_RE = re.compile(r"\bTR[A-Za-z0-9_+\-]*\b")
INSTNCT_WINNER_LABEL_RE = re.compile(r"\bwinner\s*=\s*(pocket_[abc])\b", re.IGNORECASE)


class RawGenerationError(Exception):
    def __init__(self, verdict: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.verdict = verdict
        self.message = message
        self.details = details or {}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def source_sha256() -> str:
    return sha256_file(Path(__file__))


def resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def forbidden_config_names(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in FORBIDDEN_REQUEST_KEYS:
                found.append(str(key))
            found.extend(forbidden_config_names(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(forbidden_config_names(item))
    return sorted(set(found))


def normalize_generation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is not None and not isinstance(config, dict):
        raise RawGenerationError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "generation_config must be an object")
    raw = dict(config or {})
    unknown_keys = sorted(set(raw) - ALLOWED_GENERATION_CONFIG_KEYS)
    forbidden_nested = forbidden_config_names(raw)
    if unknown_keys or forbidden_nested:
        raise RawGenerationError(
            "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED",
            "generation_config contains unknown or forbidden fields",
            {"unknown_generation_config_keys": unknown_keys, "forbidden_generation_config_fields": forbidden_nested},
        )
    normalized = {
        "temperature": float(raw.get("temperature", 0.0)),
        "device": str(raw.get("device", "cpu")),
        "stop_on_newline": bool(raw.get("stop_on_newline", False)),
    }
    if normalized["temperature"] < 0:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "temperature must be nonnegative")
    if normalized["device"] != "cpu":
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "135E helper supports cpu-only local generation")
    return normalized


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    keys = set(request)
    unknown = sorted(keys - ALLOWED_REQUEST_KEYS)
    forbidden = sorted(keys & FORBIDDEN_REQUEST_KEYS)
    if unknown or forbidden:
        raise RawGenerationError(
            "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED",
            "generation request contains forbidden or unknown fields",
            {"unknown_fields": unknown, "forbidden_fields": forbidden},
        )
    missing = sorted(ALLOWED_REQUEST_KEYS - keys)
    if missing:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "generation request missing required fields", {"missing_fields": missing})
    if not isinstance(request["prompt"], str) or not request["prompt"]:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "prompt must be a nonempty string")
    if not isinstance(request["seed"], int):
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "seed must be an integer")
    if not isinstance(request["max_new_tokens"], int) or request["max_new_tokens"] <= 0:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "max_new_tokens must be positive")
    return {
        "prompt": request["prompt"],
        "checkpoint_path": str(request["checkpoint_path"]),
        "checkpoint_hash": str(request["checkpoint_hash"]),
        "seed": int(request["seed"]),
        "max_new_tokens": min(int(request["max_new_tokens"]), 256),
        "generation_config": normalize_generation_config(request.get("generation_config")),
    }


if nn is not None:
    class ByteRNNLM(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, vocab_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            emb = self.embedding(x)
            out, _hidden = self.rnn(emb)
            return self.head(out[:, -1, :])


    class ByteMLPLM(nn.Module):
        def __init__(self, vocab_size: int, seq_len: int, embed_dim: int, hidden_size: int):
            super().__init__()
            self.seq_len = seq_len
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
            self.net = nn.Sequential(
                nn.Linear(seq_len * embed_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, vocab_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            emb = self.embedding(x).reshape(x.shape[0], -1)
            return self.net(emb)


def model_state_hash(model: "nn.Module") -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def shape_of(value: Any) -> list[int]:
    return [int(item) for item in value.shape]


def key_analysis(state: dict[str, Any], expected: set[str]) -> dict[str, Any]:
    actual = set(state)
    return {
        "checkpoint_key_count": len(actual),
        "checkpoint_expected_key_count": len(expected),
        "checkpoint_extra_keys": sorted(actual - expected),
        "checkpoint_missing_keys": sorted(expected - actual),
        "checkpoint_shape_summary": {key: shape_of(state[key]) for key in sorted(actual & expected)},
    }


def require_exact_keys(state: dict[str, Any], expected: set[str], backend: str) -> dict[str, Any]:
    analysis = key_analysis(state, expected)
    if analysis["checkpoint_extra_keys"] or analysis["checkpoint_missing_keys"]:
        raise RawGenerationError(
            "RAW_GENERATION_BACKEND_MISSING",
            f"{backend} checkpoint key set mismatch",
            {"backend_name": backend, **analysis},
        )
    return analysis


def require_rank(state: dict[str, Any], key: str, rank: int) -> None:
    if not hasattr(state[key], "shape") or len(state[key].shape) != rank:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"{key} rank mismatch")


def require_shape(state: dict[str, Any], key: str, expected: tuple[int, ...]) -> None:
    actual = tuple(int(item) for item in state[key].shape)
    if actual != expected:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"{key} shape mismatch: {actual} != {expected}")


def load_instnct_checkpoint_manifest(path: Path, checkpoint_hash: str | None, actual_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"INSTNCT manifest load failed: {exc}") from exc
    if manifest.get("backend_name") != INSTNCT_MUTATION_BACKEND:
        raise RawGenerationError(
            "RAW_GENERATION_BACKEND_MISSING",
            "JSON checkpoint is not a supported INSTNCT mutation backend manifest",
            {"backend_name": manifest.get("backend_name")},
        )
    required = {
        "schema_version",
        "backend_name",
        "answer_prefix",
        "ticks_per_generated_byte",
        "threshold_tick",
        "pockets",
        "decoder",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "INSTNCT manifest missing required fields", {"missing": missing})
    if not isinstance(manifest.get("pockets"), list) or not manifest["pockets"]:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "INSTNCT manifest must define at least one pocket")
    ticks = int(manifest.get("ticks_per_generated_byte", 0))
    threshold = int(manifest.get("threshold_tick", -1))
    if ticks <= 0 or threshold < 0 or threshold >= ticks:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "invalid INSTNCT propagation schedule")
    if checkpoint_hash and checkpoint_hash != actual_hash:
        raise RawGenerationError(
            "CHECKPOINT_HASH_MISMATCH",
            "requested checkpoint_hash does not match actual checkpoint",
            {"requested": checkpoint_hash, "actual": actual_hash, "path": rel(path)},
        )
    helper_manifest = {
        "checkpoint_path": rel(path),
        "checkpoint_sha256": actual_hash,
        "requested_checkpoint_hash": checkpoint_hash or actual_hash,
        "model_state_sha256": stable_hash(manifest),
        "helper_backend": HELPER_BACKEND,
        "helper_version": HELPER_VERSION,
        "backend_name": INSTNCT_MUTATION_BACKEND,
        "backend_load_status": "strict_instnct_manifest_load_passed",
        "strict_load_state_dict": False,
        "ticks_per_generated_byte": ticks,
        "threshold_tick": threshold,
        "pocket_count": len(manifest["pockets"]),
        "decoder": manifest.get("decoder"),
    }
    return manifest, helper_manifest


def build_model_from_state(state: dict[str, Any], ckpt: dict[str, Any]) -> tuple["nn.Module", dict[str, Any]]:
    if torch is None or nn is None:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "torch is unavailable")
    if not all(hasattr(value, "shape") for value in state.values()):
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "checkpoint state contains non-tensor values")
    if "embedding.weight" not in state:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "checkpoint missing embedding.weight")
    require_rank(state, "embedding.weight", 2)
    embedding = state["embedding.weight"]
    vocab_size = int(embedding.shape[0])
    embed_dim = int(embedding.shape[1])
    if vocab_size < BYTE_VOCAB_SIZE:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"checkpoint vocab too small: {vocab_size}")

    if set(state) == GRU_STATE_KEYS:
        analysis = require_exact_keys(state, GRU_STATE_KEYS, "byte_gru_lm")
        for key in GRU_STATE_KEYS:
            require_rank(state, key, 2 if "weight" in key else 1)
        hidden_size = int(state["rnn.weight_hh_l0"].shape[1])
        if hidden_size <= 0 or embed_dim <= 0:
            raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "unsupported RNN checkpoint shape")
        require_shape(state, "rnn.weight_ih_l0", (hidden_size * 3, embed_dim))
        require_shape(state, "rnn.weight_hh_l0", (hidden_size * 3, hidden_size))
        require_shape(state, "rnn.bias_ih_l0", (hidden_size * 3,))
        require_shape(state, "rnn.bias_hh_l0", (hidden_size * 3,))
        require_shape(state, "head.weight", (vocab_size, hidden_size))
        require_shape(state, "head.bias", (vocab_size,))
        model = ByteRNNLM(vocab_size=vocab_size, embed_dim=embed_dim, hidden_size=hidden_size)
        backend = "byte_gru_lm"
        seq_len = int(ckpt.get("seq_len") or 128)
        if seq_len <= 0:
            raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "invalid RNN seq_len")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"strict GRU load_state_dict failed: {exc}") from exc
        model.eval()
        return model, {
            "backend_name": backend,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "seq_len": seq_len,
            "backend_load_status": "strict_load_state_dict_passed",
            "strict_load_state_dict": True,
            **analysis,
        }

    if set(state) == MLP_STATE_KEYS:
        analysis = require_exact_keys(state, MLP_STATE_KEYS, "byte_mlp_lm")
        for key in MLP_STATE_KEYS:
            require_rank(state, key, 2 if "weight" in key else 1)
        hidden_size = int(state["net.0.weight"].shape[0])
        flattened = int(state["net.0.weight"].shape[1])
        if flattened % embed_dim != 0:
            raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "unsupported MLP checkpoint shape")
        seq_len = int(ckpt.get("seq_len") or (flattened // embed_dim))
        if seq_len <= 0 or seq_len * embed_dim != flattened:
            raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "unsafe MLP seq_len inference")
        require_shape(state, "net.0.weight", (hidden_size, seq_len * embed_dim))
        require_shape(state, "net.0.bias", (hidden_size,))
        require_shape(state, "net.2.weight", (hidden_size, hidden_size))
        require_shape(state, "net.2.bias", (hidden_size,))
        require_shape(state, "net.4.weight", (vocab_size, hidden_size))
        require_shape(state, "net.4.bias", (vocab_size,))
        model = ByteMLPLM(vocab_size=vocab_size, seq_len=seq_len, embed_dim=embed_dim, hidden_size=hidden_size)
        backend = "byte_mlp_lm"
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"strict MLP load_state_dict failed: {exc}") from exc
        model.eval()
        return model, {
            "backend_name": backend,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "seq_len": seq_len,
            "backend_load_status": "strict_load_state_dict_passed",
            "strict_load_state_dict": True,
            **analysis,
        }

    supported_keys = {"byte_gru_lm": sorted(GRU_STATE_KEYS), "byte_mlp_lm": sorted(MLP_STATE_KEYS)}
    raise RawGenerationError(
        "RAW_GENERATION_BACKEND_MISSING",
        "unsupported or ambiguous checkpoint architecture",
        {"actual_keys": sorted(state), "supported_key_sets": supported_keys},
    )


def load_checkpoint(checkpoint_path: str | Path, checkpoint_hash: str | None = None) -> tuple[Any, dict[str, Any]]:
    path = resolve_repo_path(checkpoint_path)
    if not path.exists():
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"checkpoint missing: {rel(path)}")
    actual_hash = sha256_file(path)
    if path.suffix.lower() == ".json":
        return load_instnct_checkpoint_manifest(path, checkpoint_hash, actual_hash)
    if torch is None:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "torch is unavailable")
    if checkpoint_hash and checkpoint_hash != actual_hash:
        raise RawGenerationError(
            "CHECKPOINT_HASH_MISMATCH",
            "requested checkpoint_hash does not match actual checkpoint",
            {"requested": checkpoint_hash, "actual": actual_hash, "path": rel(path)},
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"checkpoint load failed: {exc}") from exc
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if not isinstance(state, dict):
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", "checkpoint has no model state dict")
    try:
        model, architecture = build_model_from_state(state, ckpt)
    except RawGenerationError:
        raise
    except Exception as exc:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"checkpoint architecture load failed: {exc}") from exc
    try:
        state_hash = model_state_hash(model)
    except Exception as exc:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"model state hashing failed: {exc}") from exc
    manifest = {
        "checkpoint_path": rel(path),
        "checkpoint_sha256": actual_hash,
        "requested_checkpoint_hash": checkpoint_hash or actual_hash,
        "model_state_sha256": state_hash,
        "helper_backend": HELPER_BACKEND,
        "helper_version": HELPER_VERSION,
        **architecture,
    }
    return model, manifest


def _instnct_select_value(prompt: str, manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if bool(manifest.get("value_selection_requires_open_pocket", False)):
        return _instnct_select_open_pocket_value(prompt, manifest)
    preferred_markers = manifest.get("preferred_value_markers") or ["OBSERVED_VALUE=", "TARGET_VALUE=", "VALUE=", "BIND="]
    for marker in preferred_markers:
        pos = prompt.find(str(marker))
        if pos >= 0:
            segment = prompt[pos + len(str(marker)) : pos + len(str(marker)) + 96]
            match = INSTNCT_VALUE_RE.search(segment)
            if match:
                return match.group(0), {"selection_source": "marker", "marker": marker}
    match = INSTNCT_VALUE_RE.search(prompt)
    if match:
        return match.group(0), {"selection_source": "first_prompt_value", "marker": None}
    train_match = INSTNCT_TRAIN_VALUE_RE.search(prompt)
    if train_match and bool(manifest.get("allow_train_namespace_value_fallback", False)):
        return train_match.group(0), {"selection_source": "train_namespace_fallback", "marker": None}
    fallback = str(manifest.get("fallback_value", "SYM_NO_VALUE"))
    return fallback, {"selection_source": "fallback", "marker": None}


def _instnct_value_from_segment(segment: str) -> str | None:
    match = INSTNCT_VALUE_RE.search(segment)
    return match.group(0) if match else None


def _instnct_select_open_pocket_value(prompt: str, manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    decoder = manifest.get("decoder") if isinstance(manifest.get("decoder"), dict) else {}
    if decoder.get("type") == RULE_SELECTED_POCKET_BINDING_DECODER:
        return _instnct_select_rule_selected_pocket_value(prompt, manifest)
    payload_markers = [str(item) for item in (manifest.get("pocket_payload_markers") or ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="])]
    fallback = str(manifest.get("closed_pocket_fallback_value") or manifest.get("fallback_value", "SYM_POCKET_CLOSED"))
    visible_bypass_forbidden = bool(manifest.get("visible_value_bypass_forbidden", True))
    for pocket in manifest.get("pockets", []):
        gate_marker = str(pocket.get("gate_marker", ""))
        gate_open = bool(gate_marker and gate_marker in prompt)
        if not gate_open:
            continue
        pocket_markers = [str(item) for item in (pocket.get("payload_markers") or payload_markers)]
        for marker in pocket_markers:
            pos = prompt.find(marker)
            if pos < 0:
                continue
            segment = prompt[pos + len(marker) : pos + len(marker) + 128]
            value = _instnct_value_from_segment(segment)
            if value:
                return value, {
                    "selection_source": "open_pocket_writeback",
                    "marker": marker,
                    "pocket_id": str(pocket.get("pocket_id", "")),
                    "gate_marker": gate_marker,
                    "visible_value_bypass_forbidden": visible_bypass_forbidden,
                }
    if visible_bypass_forbidden:
        return fallback, {
            "selection_source": "closed_pocket_fallback",
            "marker": None,
            "pocket_id": None,
            "gate_marker": None,
            "visible_value_bypass_forbidden": True,
        }
    preferred_markers = manifest.get("preferred_value_markers") or ["OBSERVED_VALUE=", "TARGET_VALUE=", "VALUE=", "BIND="]
    for marker in preferred_markers:
        pos = prompt.find(str(marker))
        if pos >= 0:
            segment = prompt[pos + len(str(marker)) : pos + len(str(marker)) + 96]
            value = _instnct_value_from_segment(segment)
            if value:
                return value, {"selection_source": "visible_bypass_fallback", "marker": marker}
    return fallback, {"selection_source": "closed_pocket_fallback", "marker": None}


def _instnct_select_rule_selected_pocket_value(prompt: str, manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fallback = str(manifest.get("closed_pocket_fallback_value") or manifest.get("fallback_value", "SYM_POCKET_CLOSED"))
    visible_bypass_forbidden = bool(manifest.get("visible_value_bypass_forbidden", True))
    winner_labels = [match.group(1).lower() for match in INSTNCT_WINNER_LABEL_RE.finditer(prompt)]
    if len(winner_labels) != 1:
        return fallback, {
            "selection_source": "closed_pocket_fallback",
            "marker": None,
            "pocket_id": None,
            "gate_marker": None,
            "visible_value_bypass_forbidden": visible_bypass_forbidden,
            "rule_selected_pocket_binding_rejected": "missing_or_ambiguous_winner_label",
            "winner_label_count": len(winner_labels),
        }
    selected_pocket_id = winner_labels[0]
    marker_map_raw = manifest.get("rule_selected_pocket_marker_map") or manifest.get("static_pocket_marker_map") or {}
    marker_map = {str(key).lower(): str(value) for key, value in marker_map_raw.items()}
    selected_marker = marker_map.get(selected_pocket_id)
    if not selected_marker:
        return fallback, {
            "selection_source": "closed_pocket_fallback",
            "marker": None,
            "pocket_id": selected_pocket_id,
            "gate_marker": None,
            "visible_value_bypass_forbidden": visible_bypass_forbidden,
            "rule_selected_pocket_binding_rejected": "missing_static_marker_map_entry",
            "winner_label_count": len(winner_labels),
        }
    for pocket in manifest.get("pockets", []):
        gate_marker = str(pocket.get("gate_marker", ""))
        gate_open = bool(gate_marker and gate_marker in prompt)
        if not gate_open:
            continue
        candidate_line_re = re.compile(r"^\s*" + re.escape(selected_marker) + r"\s*((?:EV|VAL|SYM)[A-Za-z0-9_+\-]*)?\s*$")
        candidate_lines = [(line, match.group(1)) for line in prompt.splitlines() if (match := candidate_line_re.match(line))]
        if not candidate_lines:
            return fallback, {
                "selection_source": "closed_pocket_fallback",
                "marker": selected_marker,
                "pocket_id": selected_pocket_id,
                "gate_marker": gate_marker,
                "visible_value_bypass_forbidden": visible_bypass_forbidden,
                "rule_selected_pocket_binding_rejected": "selected_marker_missing_from_prompt",
                "winner_label_count": len(winner_labels),
                "selected_marker_candidate_line_count": 0,
            }
        if len(candidate_lines) != 1:
            return fallback, {
                "selection_source": "closed_pocket_fallback",
                "marker": selected_marker,
                "pocket_id": selected_pocket_id,
                "gate_marker": gate_marker,
                "visible_value_bypass_forbidden": visible_bypass_forbidden,
                "rule_selected_pocket_binding_rejected": "selected_marker_duplicate_conflict",
                "winner_label_count": len(winner_labels),
                "selected_marker_candidate_line_count": len(candidate_lines),
            }
        value = candidate_lines[0][1]
        if value:
            return value, {
                "selection_source": "rule_selected_pocket_writeback",
                "marker": selected_marker,
                "pocket_id": selected_pocket_id,
                "gate_marker": gate_marker,
                "visible_value_bypass_forbidden": visible_bypass_forbidden,
                "winner_label_count": len(winner_labels),
                "selected_marker_candidate_line_count": len(candidate_lines),
            }
        return fallback, {
            "selection_source": "closed_pocket_fallback",
            "marker": selected_marker,
            "pocket_id": selected_pocket_id,
            "gate_marker": gate_marker,
            "visible_value_bypass_forbidden": visible_bypass_forbidden,
            "rule_selected_pocket_binding_rejected": "selected_marker_value_missing",
            "winner_label_count": len(winner_labels),
            "selected_marker_candidate_line_count": len(candidate_lines),
        }
    return fallback, {
        "selection_source": "closed_pocket_fallback",
        "marker": selected_marker,
        "pocket_id": selected_pocket_id,
        "gate_marker": None,
        "visible_value_bypass_forbidden": visible_bypass_forbidden,
        "rule_selected_pocket_binding_rejected": "open_gate_missing",
        "winner_label_count": len(winner_labels),
    }


def _instnct_trace(prompt: str, selected_value: str, manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    ticks = int(manifest["ticks_per_generated_byte"])
    threshold = int(manifest["threshold_tick"])
    prompt_hash = hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()
    pockets = []
    writeback_count = 0
    for idx, pocket in enumerate(manifest["pockets"]):
        gate_marker = str(pocket.get("gate_marker", ""))
        gate_open = bool(gate_marker and gate_marker in prompt)
        if gate_open:
            writeback_count += 1
        pockets.append(
            {
                "pocket_id": str(pocket.get("pocket_id", f"p{idx}")),
                "gate_marker": gate_marker,
                "gate_open": gate_open,
                "threshold_tick": threshold,
                "writeback_value_hash": hashlib.sha256(selected_value.encode("utf-8")).hexdigest() if gate_open else None,
            }
        )
    return {
        "backend_name": INSTNCT_MUTATION_BACKEND,
        "prompt_sha256": prompt_hash,
        "seed": seed,
        "ticks_per_generated_byte": ticks,
        "threshold_tick": threshold,
        "total_ticks": ticks * len(selected_value),
        "highway_retained": True,
        "pocket_writeback_count": writeback_count,
        "pockets": pockets,
    }


def instnct_raw_generate(clean: dict[str, Any], manifest: dict[str, Any], backend_manifest: dict[str, Any]) -> dict[str, Any]:
    config = clean["generation_config"]
    prompt = clean["prompt"]
    seed = int(clean["seed"])
    max_new = int(clean["max_new_tokens"])
    selected_value, selection = _instnct_select_value(prompt, manifest)
    prefix = str(manifest.get("answer_prefix", "ANSWER=E"))
    generated_text = f"{prefix}{selected_value}"
    if len(generated_text) > max_new:
        generated_text = generated_text[:max_new]
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "instnct_answer_complete"
    raw = generated_text.encode("utf-8", errors="replace")
    trace = _instnct_trace(prompt, selected_value, manifest, seed)
    trace.update(
        {
            "selection": selection,
            "generated_text_sha256": hashlib.sha256(raw).hexdigest(),
            "checkpoint_hash": backend_manifest["checkpoint_sha256"],
            "generation_config_hash": stable_hash(config),
            "max_new_tokens": max_new,
            "stop_reason": stop_reason,
            "helper_version": HELPER_VERSION,
            "helper_backend": HELPER_BACKEND,
        }
    )
    return {
        "generated_text": generated_text,
        "token_count": len(raw),
        "stop_reason": stop_reason,
        "generation_trace_hash": stable_hash(trace),
        "model_checkpoint_hash": backend_manifest["checkpoint_sha256"],
        "generation_config_hash": stable_hash(config),
        "helper_backend": backend_manifest["helper_backend"],
        "helper_version": HELPER_VERSION,
        "backend_name": backend_manifest["backend_name"],
        "backend_version": str(manifest.get("schema_version", HELPER_VERSION)),
        "checkpoint_path": backend_manifest["checkpoint_path"],
        "model_state_sha256": backend_manifest["model_state_sha256"],
        "instnct_trace_hash": stable_hash(trace),
        "pocket_writeback_count": trace["pocket_writeback_count"],
        "highway_retained": trace["highway_retained"],
        "ticks_per_generated_byte": trace["ticks_per_generated_byte"],
        "threshold_tick": trace["threshold_tick"],
        "value_selection_source": selection.get("selection_source"),
        "value_selection_requires_open_pocket": bool(manifest.get("value_selection_requires_open_pocket", False)),
    }


def discover_backend(candidates: list[str | Path] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch_available": torch is not None,
        "device": "cpu",
        "candidates": [],
        "selected": None,
    }
    if torch is None:
        report["status"] = "torch_unavailable"
        return report
    for candidate in candidates or DEFAULT_CHECKPOINT_CANDIDATES:
        path = resolve_repo_path(candidate)
        item: dict[str, Any] = {"path": rel(path), "exists": path.exists(), "selected": False}
        if path.exists():
            try:
                _model, manifest = load_checkpoint(path)
                item.update({"loadable": True, **manifest})
                if report["selected"] is None:
                    item["selected"] = True
                    report["selected"] = item
            except RawGenerationError as exc:
                item.update({"loadable": False, "failure_verdict": exc.verdict, "failure_message": exc.message, **exc.details})
            except Exception as exc:
                item.update({"loadable": False, "failure_verdict": "RAW_GENERATION_BACKEND_MISSING", "failure_message": f"unexpected backend discovery failure: {exc}"})
        report["candidates"].append(item)
    report["status"] = "selected" if report["selected"] else "missing"
    return report


def _window_for_model(context: list[int], seq_len: int, pad_id: int) -> list[int]:
    window = context[-seq_len:]
    if len(window) < seq_len:
        window = [pad_id] * (seq_len - len(window)) + window
    return window


def raw_generate(request: dict[str, Any]) -> dict[str, Any]:
    clean = validate_request(request)
    model, manifest = load_checkpoint(clean["checkpoint_path"], clean["checkpoint_hash"])
    if manifest.get("backend_name") == INSTNCT_MUTATION_BACKEND:
        return instnct_raw_generate(clean, model, manifest)
    config = clean["generation_config"]
    seq_len = int(manifest.get("seq_len") or 128)
    vocab_size = int(manifest["vocab_size"])
    pad_id = vocab_size - 1
    max_new = int(clean["max_new_tokens"])
    prompt = clean["prompt"]
    seed = int(clean["seed"])
    temperature = float(config["temperature"])

    context = list(prompt.encode("utf-8", errors="replace"))
    generated: list[int] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    try:
        with torch.no_grad():
            for _step in range(max_new):
                if isinstance(model, ByteRNNLM):
                    window = context[-seq_len:] or [pad_id]
                else:
                    window = _window_for_model(context, seq_len, pad_id)
                x = torch.tensor([window], dtype=torch.long)
                logits = model(x)[0, :BYTE_VOCAB_SIZE]
                if temperature <= 0:
                    next_id = int(torch.argmax(logits).item())
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_id = int(torch.multinomial(probs, 1, generator=generator).item())
                generated.append(next_id)
                context.append(next_id)
    except Exception as exc:
        raise RawGenerationError("RAW_GENERATION_BACKEND_MISSING", f"raw generation failed: {exc}") from exc
    raw = bytes(generated)
    generated_text = raw.decode("utf-8", errors="replace")
    stop_reason = "max_new_tokens"
    config_hash = stable_hash(config)
    trace_payload = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest(),
        "generated_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_hash": manifest["checkpoint_sha256"],
        "seed": seed,
        "max_new_tokens": max_new,
        "generation_config_hash": config_hash,
        "stop_reason": stop_reason,
        "helper_version": HELPER_VERSION,
        "helper_backend": HELPER_BACKEND,
    }
    return {
        "generated_text": generated_text,
        "token_count": len(generated),
        "stop_reason": stop_reason,
        "generation_trace_hash": stable_hash(trace_payload),
        "model_checkpoint_hash": manifest["checkpoint_sha256"],
        "generation_config_hash": config_hash,
        "helper_backend": manifest["helper_backend"],
        "helper_version": HELPER_VERSION,
        "backend_name": manifest["backend_name"],
        "backend_version": HELPER_VERSION,
        "checkpoint_path": manifest["checkpoint_path"],
        "model_state_sha256": manifest["model_state_sha256"],
    }


def build_request(
    prompt: str,
    checkpoint_path: str,
    checkpoint_hash: str,
    seed: int,
    max_new_tokens: int,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "generation_config": normalize_generation_config(generation_config),
    }
