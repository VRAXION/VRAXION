"""Unified checkpoint save/load.

Merges monolithic and modular checkpoint logic into one clean interface.
Modular layout (stable contract):
  <base_dir>/
    system/router.state      # core model state + optimizer/scaler + training metadata
    experts/expert_###.pt    # per-expert tensors (sliced from head.experts.*)
    experts/meta.json        # lightweight expert lifecycle metadata
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .log import log


# Callers may override at runtime.
USE_AMP = False
DEVICE: Any = "cpu"
EXPERT_HEADS = 1


def _torch_load_compat(path: str, *, map_location: Any = "cpu", weights_only: Optional[bool] = None) -> Any:
    """torch.load wrapper compatible across torch versions."""
    if weights_only is None:
        return torch.load(path, map_location=map_location)
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        if weights_only:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
        raise


def _atomic_torch_save(obj: Any, path: str) -> None:
    """Atomically torch.save(obj) -> path (temp file + os.replace)."""
    dirpth = os.path.dirname(path) or "."
    os.makedirs(dirpth, exist_ok=True)

    fdpair = tempfile.mkstemp(prefix=f".tmp.{os.path.basename(path)}.", dir=dirpth)
    os.close(fdpair[0])
    tmppth = fdpair[1]

    try:
        with open(tmppth, "wb") as filobj:
            torch.save(obj, filobj)
            filobj.flush()
            try:
                os.fsync(filobj.fileno())
            except OSError:
                pass
        os.replace(tmppth, path)
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except OSError:
                pass


def _atomic_json_dump(obj: Any, path: str, *, indent: int = 2) -> None:
    """Atomically json.dump(obj) -> path (temp file + os.replace)."""
    dirpth = os.path.dirname(path) or "."
    os.makedirs(dirpth, exist_ok=True)

    fdpair = tempfile.mkstemp(prefix=f".tmp.{os.path.basename(path)}.", dir=dirpth)
    os.close(fdpair[0])
    tmppth = fdpair[1]

    try:
        with open(tmppth, "w", encoding="utf-8") as filobj:
            json.dump(obj, filobj, indent=indent)
            filobj.flush()
            try:
                os.fsync(filobj.fileno())
            except OSError:
                pass
        os.replace(tmppth, path)
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except OSError:
                pass


def _hash_state_dict(state: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not state:
        return None
    hasher = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key]
        hasher.update(key.encode("utf-8"))
        if torch.is_tensor(tensor):
            tenval = tensor.detach().contiguous().cpu()
            try:
                arr = tenval.numpy()
            except Exception:
                arr = tenval.to(dtype=torch.float32).numpy()
            hasher.update(arr.tobytes())
        else:
            hasher.update(repr(tensor).encode("utf-8"))
    return hasher.hexdigest()


# ---- Modular layout helpers ----


def _modular_paths(base_dir: str) -> Tuple[str, str, str]:
    system_dir = os.path.join(base_dir, "system")
    experts_dir = os.path.join(base_dir, "experts")
    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(experts_dir, exist_ok=True)

    router_path = os.path.join(system_dir, "router.state")
    meta_path = os.path.join(experts_dir, "meta.json")
    return router_path, experts_dir, meta_path


def _split_model_state_dict(state_dict: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    core: Dict[str, Any] = {}
    experts: Dict[int, Dict[str, Any]] = {}

    prefix = "head.experts."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]
            parts = rest.split(".", 1)
            if len(parts) != 2:
                core[key] = value.detach().cpu() if torch.is_tensor(value) else value
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                core[key] = value.detach().cpu() if torch.is_tensor(value) else value
                continue
            expert_state = experts.setdefault(idx, {})
            expert_state[parts[1]] = value.detach().cpu() if torch.is_tensor(value) else value
        else:
            core[key] = value.detach().cpu() if torch.is_tensor(value) else value

    return core, experts


def _coerce_int(valobj: Any, defval: int = 0) -> int:
    try:
        return int(valobj)
    except Exception:
        return int(defval)


def _coerce_flt(valobj: Any, defval: float = 0.0) -> float:
    try:
        return float(valobj)
    except Exception:
        return float(defval)


def _coerce_bol(valobj: Any, defval: bool = False) -> bool:
    try:
        if isinstance(valobj, str):
            lowval = valobj.strip().lower()
            if lowval in {"true", "1", "yes", "y", "t"}:
                return True
            if lowval in {"false", "0", "no", "n", "f"}:
                return False
        return bool(valobj)
    except Exception:
        return bool(defval)


# ---- Save ----


def save_modular_checkpoint(
    model: Any,
    optimizer: Any,
    scaler: Any,
    step: int,
    losses: Any,
    base_dir: str,
    contrib_thresh: float = 0.0,
    probation_steps: int = 0,
    ttl_steps: int = 0,
    gc_enabled: bool = False,
) -> None:
    """Save a modular checkpoint with separate expert shards."""
    router_path, experts_dir, meta_path = _modular_paths(base_dir)
    state_dict = model.state_dict()
    core_state, expert_states = _split_model_state_dict(state_dict)

    num_experts = getattr(getattr(model, "head", None), "num_experts", EXPERT_HEADS)

    payload = {
        "model": core_state,
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if USE_AMP and scaler is not None else None,
        "step": int(step),
        "losses": list(losses),
        "update_scale": 1.0,
        "ptr_inertia": getattr(model, "ptr_inertia", 0.0),
        "ptr_inertia_ema": getattr(model, "ptr_inertia_ema", getattr(model, "ptr_inertia", 0.0)),
        "ptr_inertia_floor": getattr(model, "ptr_inertia_floor", 0.0),
        "agc_scale_max": 1.0,
        "ground_speed_ema": getattr(model, "ground_speed_ema", None),
        "ground_speed_limit": getattr(model, "ground_speed_limit", None),
        "num_experts": int(num_experts),
        "param_names": [name for name, _ in model.named_parameters()],
    }

    _atomic_torch_save(payload, router_path)

    # Build expert lifecycle metadata.
    _ensure_expert_tracking(model, int(num_experts), step)
    expert_meta = _build_expert_meta(model, step, int(num_experts), contrib_thresh, probation_steps)

    deleted = []
    for idx, state in expert_states.items():
        if idx >= int(num_experts):
            continue

        if gc_enabled and ttl_steps > 0:
            tenured = bool(model.expert_tenured[idx])
            idle = int(step) - int(model.expert_last_used[idx])
            age = int(step) - int(model.expert_created_step[idx])
            probation = age < probation_steps
            if not tenured and not probation and idle >= ttl_steps:
                deleted.append(idx)
                continue

        path = os.path.join(experts_dir, f"expert_{idx:03d}.pt")
        _atomic_torch_save(state, path)

    meta = {
        "num_experts": int(num_experts),
        "step": int(step),
        "experts": expert_meta,
        "deleted": deleted,
    }
    _atomic_json_dump(meta, meta_path, indent=2)


# ---- Load ----


def load_modular_checkpoint(model: Any, optimizer: Any, scaler: Any, base_dir: str) -> Dict[str, Any]:
    """Load a modular checkpoint, restoring model + optimizer + scaler + expert shards."""
    router_path, experts_dir, meta_path = _modular_paths(base_dir)
    ckpt = _torch_load_compat(router_path, map_location=DEVICE, weights_only=False)

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        log(f"Modular load strict failed: {exc}; retrying strict=False")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            log(f"Modular load missing keys: {missing}")
        if unexpected:
            log(f"Modular load unexpected keys: {unexpected}")

    head = getattr(model, "head", None)
    if head is not None and getattr(head, "experts", None):
        for idx, expert in enumerate(head.experts):
            path = os.path.join(experts_dir, f"expert_{idx:03d}.pt")
            if not os.path.exists(path):
                continue
            try:
                state = _torch_load_compat(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                log(f"Expert load failed (ignored): {path} ({exc})")
                continue
            if not isinstance(state, dict):
                log(f"Expert snapshot is not a dict (ignored): {path}")
                continue
            try:
                expert.load_state_dict(state, strict=False)
            except Exception as exc:
                log(f"Expert load_state_dict failed (ignored): {path} ({exc})")

    if ckpt.get("optim") is not None and optimizer is not None:
        optimizer.load_state_dict(ckpt["optim"])
    if ckpt.get("scaler") is not None and USE_AMP and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])

    _load_modular_meta(model, meta_path)
    return ckpt


def resolve_modular_resume_dir(resume_path: Optional[str]) -> Optional[str]:
    """Resolve a modular checkpoint directory from a resume path."""
    if resume_path and os.path.isdir(resume_path):
        router_path = os.path.join(resume_path, "system", "router.state")
        if os.path.exists(router_path):
            return resume_path

    if resume_path and resume_path.endswith(".pt"):
        candidate = os.path.splitext(resume_path)[0] + "_modular"
        router_path = os.path.join(candidate, "system", "router.state")
        if os.path.exists(router_path):
            return candidate

    return None


# ---- Expert lifecycle tracking ----


def _ensure_expert_tracking(model: Any, num_experts: int, step: int) -> None:
    contrib = getattr(model, "expert_contrib", None)
    last_used = getattr(model, "expert_last_used", None)
    created = getattr(model, "expert_created_step", None)
    tenured = getattr(model, "expert_tenured", None)

    if contrib is None or last_used is None or created is None or tenured is None:
        model.expert_contrib = [0.0 for _ in range(num_experts)]
        model.expert_last_used = [int(step) for _ in range(num_experts)]
        model.expert_created_step = [int(step) for _ in range(num_experts)]
        model.expert_tenured = [False for _ in range(num_experts)]
        return

    if len(contrib) < num_experts:
        missing = num_experts - len(contrib)
        model.expert_contrib.extend([0.0 for _ in range(missing)])
        model.expert_last_used.extend([int(step) for _ in range(missing)])
        model.expert_created_step.extend([int(step) for _ in range(missing)])
        model.expert_tenured.extend([False for _ in range(missing)])


def _build_expert_meta(model: Any, step: int, num_experts: int, contrib_thresh: float, probation_steps: int):
    _ensure_expert_tracking(model, num_experts, step)
    meta = []
    for idx in range(num_experts):
        contrib = float(model.expert_contrib[idx])
        created = int(model.expert_created_step[idx])
        last_used = int(model.expert_last_used[idx])
        age = max(0, int(step) - created)
        probation = age < probation_steps
        tenured = contrib >= contrib_thresh
        model.expert_tenured[idx] = bool(tenured)

        meta.append({
            "id": idx,
            "contrib": contrib,
            "created_step": created,
            "last_used_step": last_used,
            "age_steps": age,
            "probation": bool(probation),
            "tenured": bool(tenured),
        })
    return meta


def _load_modular_meta(model: Any, meta_path: str) -> None:
    if not os.path.exists(meta_path):
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as filobj:
            data = json.load(filobj)
    except Exception as exc:
        log(f"Invalid modular meta.json (ignored): {meta_path} ({exc})")
        return

    if not isinstance(data, dict):
        return
    experts = data.get("experts", [])
    if not isinstance(experts, list) or not experts:
        return

    model.expert_contrib = [_coerce_flt(item.get("contrib", 0.0)) if isinstance(item, dict) else 0.0 for item in experts]
    model.expert_last_used = [_coerce_int(item.get("last_used_step", 0)) if isinstance(item, dict) else 0 for item in experts]
    model.expert_created_step = [_coerce_int(item.get("created_step", 0)) if isinstance(item, dict) else 0 for item in experts]
    model.expert_tenured = [_coerce_bol(item.get("tenured", False)) if isinstance(item, dict) else False for item in experts]
