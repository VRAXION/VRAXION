"""
Live controls for Diamond Code swarm training.

Reads logs/swarm/controls.json every N steps. If the file doesn't exist
or has parse errors, uses defaults silently.

Edit controls.json while training runs to adjust LR, data mixing, think ticks,
batch size, LCX on/off, and per-being states (null/active/frozen).

Effort tier system (Greek alphabet):
    To change tier, edit controls.json: {"effort": "Gamma"}
    apply_controls() auto-sets tt, batch, use_lcx from the tier definition.

Format:
    {
        "lr": 0.0001,
        "effort": "Beta",
        "being_states": {"0": "null", "6": "active"},
        "data_weights": {"gold_origin_echo.traindat": 1.0}
    }

Effort tiers:
    Alpha(Reflex):    tt=0,  batch=500, lcx=OFF  (pure feedforward)
    Beta(Recall):     tt=1,  batch=500, lcx=ON   (first memory, L0)
    Gamma(Reason):    tt=2,  batch=500, lcx=ON   (deeper retrieval)
    Delta(Depth):     tt=4,  batch=500, lcx=ON   (multi-pass reasoning)
    Epsilon(Emerge):  tt=8,  batch=500, lcx=ON   (extended contemplation)
    Zeta(Zenith):     tt=16, batch=250, lcx=ON   (maximum think depth)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn


# Greek alphabet effort tiers — canonical definition.
# To advance: edit controls.json {"effort": "Gamma"} and apply_controls() does the rest.
DEFAULT_EFFORT_TIERS = {
    "Alpha":   {"tt": 0,  "lcx": False, "batch": 500, "name": "Reflex"},
    "Beta":    {"tt": 1,  "lcx": True,  "batch": 500, "name": "Recall"},
    "Gamma":   {"tt": 2,  "lcx": True,  "batch": 500, "name": "Reason"},
    "Delta":   {"tt": 4,  "lcx": True,  "batch": 500, "name": "Depth"},
    "Epsilon": {"tt": 8,  "lcx": True,  "batch": 500, "name": "Emergence"},
    "Zeta":    {"tt": 16, "lcx": True,  "batch": 250, "name": "Zenith"},
}

VALID_EFFORTS = set(DEFAULT_EFFORT_TIERS.keys())


DEFAULT_CONTROLS = {
    "lr": None,
    "think_ticks": None,
    "batch_size": None,
    "use_lcx": None,
    "stage": None,
    "effort": None,
    "effort_name": None,
    "checkpoint_every": None,
    "eval_every": None,
    "eval_samples": None,
    "temporal_fibonacci": None,
    "effort_mode": None,
    "effort_tiers": None,
    "being_states": None,
    "data_weights": {},
    "agc_enabled": True,
    "agc_low": 0.5,
    "agc_high": 1.0,
    # Dreaming phase defaults
    "dream_enabled": False,
    "dream_frequency": 10,
    "dream_steps": 3,
    "dream_think_ticks": 2,
    "dream_mode": "consolidation",
    "dream_binarize": True,
    "dream_lr_scale": 0.1,
}


def write_default_controls(path: str, lr: float, data_weights: Dict[str, float],
                           think_ticks: int = 0, checkpoint_every: int = 50,
                           eval_every: int = 10, batch_size: int = 32,
                           use_lcx: bool = True, stage: str = "INFANT",
                           effort: str = "Beta"):
    """Write initial controls.json at training start.

    If controls.json already exists, PRESERVE existing data_weights and eval_every
    (user may have configured these via the control panel). Other fields are
    overwritten from run args to ensure effort tier consistency.
    """
    # Preserve user-configured fields from existing controls.json
    existing_data_weights = None
    existing_eval_every = None
    existing_tiers = None
    if Path(path).exists():
        try:
            with open(path, 'r') as f:
                existing = json.load(f)
            if 'data_weights' in existing and isinstance(existing['data_weights'], dict):
                existing_data_weights = existing['data_weights']
            if 'eval_every' in existing:
                existing_eval_every = existing['eval_every']
            if 'effort_tiers' in existing and isinstance(existing['effort_tiers'], dict):
                existing_tiers = existing['effort_tiers']
        except Exception:
            pass

    # Look up effort tier — override tt/lcx/batch from tier definition
    # Prefer tiers from existing controls.json (user may have customized batch sizes)
    tiers = existing_tiers or DEFAULT_EFFORT_TIERS
    tier = tiers.get(effort)
    if tier:
        think_ticks = tier["tt"]
        use_lcx = tier["lcx"]
        batch_size = tier["batch"]
        effort_name = tier["name"]
        stage = effort.upper()
    else:
        effort_name = ""

    # Use existing data_weights if available, otherwise use loader defaults.
    # Also merge any NEW files from loader that aren't in the existing config.
    if existing_data_weights is not None:
        merged_weights = dict(existing_data_weights)
        for k, v in data_weights.items():
            if k not in merged_weights:
                merged_weights[k] = 0  # new files default to OFF
        data_weights = merged_weights

    controls = {
        "lr": lr,
        "think_ticks": think_ticks,
        "batch_size": batch_size,
        "use_lcx": use_lcx,
        "stage": stage,
        "effort": effort,
        "effort_name": effort_name,
        "effort_lock": "off",
        "checkpoint_every": checkpoint_every,
        "eval_every": existing_eval_every if existing_eval_every is not None else eval_every,
        "data_weights": data_weights,
        "effort_tiers": tiers,
        "agc_enabled": True,
        "agc_low": 0.5,
        "agc_high": 1.0,
        # Dreaming phase (disabled by default — turn on via controls.json or Grafana)
        "dream_enabled": False,
        "dream_frequency": 10,
        "dream_steps": 3,
        "dream_think_ticks": 2,
        "dream_mode": "consolidation",
        "dream_binarize": True,
        "dream_lr_scale": 0.1,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(controls, f, indent=2)


def read_controls(path: str) -> Dict[str, Any]:
    """
    Read controls.json. Returns DEFAULT_CONTROLS on any error.

    Intentionally fault-tolerant: file missing, parse errors, partial
    writes from concurrent edits -- all return safe defaults.
    """
    try:
        if not os.path.exists(path):
            return dict(DEFAULT_CONTROLS)

        with open(path, 'r') as f:
            data = json.load(f)

        result = dict(DEFAULT_CONTROLS)
        if isinstance(data, dict):
            if 'lr' in data and isinstance(data['lr'], (int, float)):
                result['lr'] = float(data['lr'])
            if 'think_ticks' in data and isinstance(data['think_ticks'], int):
                result['think_ticks'] = int(data['think_ticks'])
            if 'batch_size' in data and isinstance(data['batch_size'], int):
                result['batch_size'] = int(data['batch_size'])
            if 'use_lcx' in data and isinstance(data['use_lcx'], bool):
                result['use_lcx'] = data['use_lcx']
            if 'stage' in data and isinstance(data['stage'], str):
                result['stage'] = data['stage']
            if 'checkpoint_every' in data and isinstance(data['checkpoint_every'], int):
                result['checkpoint_every'] = int(data['checkpoint_every'])
            if 'eval_samples' in data and isinstance(data['eval_samples'], int):
                result['eval_samples'] = int(data['eval_samples'])
            if 'eval_every' in data and isinstance(data['eval_every'], int):
                result['eval_every'] = int(data['eval_every'])
            if 'temporal_fibonacci' in data and isinstance(data['temporal_fibonacci'], bool):
                result['temporal_fibonacci'] = data['temporal_fibonacci']
            if 'effort_mode' in data:
                _em = data['effort_mode']
                if isinstance(_em, bool) or _em in ('auto', 'off', None):
                    result['effort_mode'] = _em
            if 'effort' in data and isinstance(data['effort'], str):
                if data['effort'] in VALID_EFFORTS:
                    result['effort'] = data['effort']
            if 'effort_name' in data and isinstance(data['effort_name'], str):
                result['effort_name'] = data['effort_name']
            if 'effort_tiers' in data and isinstance(data['effort_tiers'], dict):
                result['effort_tiers'] = data['effort_tiers']
            if 'effort_lock' in data and isinstance(data['effort_lock'], str):
                if data['effort_lock'] in ('off', 'random', 'fast', 'medium', 'slow'):
                    result['effort_lock'] = data['effort_lock']
            if 'being_states' in data and isinstance(data['being_states'], dict):
                result['being_states'] = {
                    k: v for k, v in data['being_states'].items()
                    if isinstance(v, str) and v in ('null', 'active', 'frozen')
                }
            if 'data_weights' in data and isinstance(data['data_weights'], dict):
                result['data_weights'] = {
                    k: float(v) for k, v in data['data_weights'].items()
                    if isinstance(v, (int, float))
                }
            if 'data_sequential' in data and isinstance(data['data_sequential'], bool):
                result['data_sequential'] = data['data_sequential']
            if 'data_seq_steps' in data and isinstance(data['data_seq_steps'], (int, float)):
                result['data_seq_steps'] = int(data['data_seq_steps'])
            if 'agc_enabled' in data and isinstance(data['agc_enabled'], bool):
                result['agc_enabled'] = data['agc_enabled']
            if 'agc_low' in data and isinstance(data['agc_low'], (int, float)):
                result['agc_low'] = float(data['agc_low'])
            if 'agc_high' in data and isinstance(data['agc_high'], (int, float)):
                result['agc_high'] = float(data['agc_high'])
            # Dreaming phase controls
            if 'dream_enabled' in data and isinstance(data['dream_enabled'], bool):
                result['dream_enabled'] = data['dream_enabled']
            if 'dream_frequency' in data and isinstance(data['dream_frequency'], int):
                result['dream_frequency'] = max(1, int(data['dream_frequency']))
            if 'dream_steps' in data and isinstance(data['dream_steps'], int):
                result['dream_steps'] = max(1, min(20, int(data['dream_steps'])))
            if 'dream_think_ticks' in data and isinstance(data['dream_think_ticks'], int):
                result['dream_think_ticks'] = max(0, int(data['dream_think_ticks']))
            if 'dream_mode' in data and isinstance(data['dream_mode'], str):
                if data['dream_mode'] in ('consolidation', 'rehearsal'):
                    result['dream_mode'] = data['dream_mode']
            if 'dream_binarize' in data and isinstance(data['dream_binarize'], bool):
                result['dream_binarize'] = data['dream_binarize']
            if 'dream_lr_scale' in data and isinstance(data['dream_lr_scale'], (int, float)):
                result['dream_lr_scale'] = float(max(0.001, min(1.0, data['dream_lr_scale'])))
        return result
    except Exception:
        return dict(DEFAULT_CONTROLS)


def apply_controls(controls: Dict[str, Any], optimizer, loader=None, model=None):
    """
    Apply controls to optimizer, loader, and model.
    Returns (optimizer, description_of_changes).

    Safe to call every N steps. Only applies non-None values.
    The optimizer may be rebuilt if being states change.

    Effort tier auto-application: when 'effort' field changes (e.g. "Beta" -> "Gamma"),
    the tier's tt/lcx/batch are applied automatically. Edit one field, get all three.
    """
    changes = []

    # --- Effort tier auto-application (must run BEFORE individual field checks) ---
    if controls.get('effort') is not None and model is not None:
        new_effort = controls['effort']
        old_effort = getattr(model, '_current_effort', None)
        if old_effort != new_effort and new_effort in VALID_EFFORTS:
            tiers = controls.get('effort_tiers') or DEFAULT_EFFORT_TIERS
            tier = tiers.get(new_effort)
            if tier:
                old_name = getattr(model, '_current_effort_name', '?')
                new_name = tier.get('name', new_effort)
                # Override the individual controls with tier values
                controls['think_ticks'] = tier['tt']
                controls['use_lcx'] = tier['lcx']
                controls['batch_size'] = tier['batch']
                controls['stage'] = new_effort.upper()
                controls['effort_name'] = new_name
                model._current_effort = new_effort
                model._current_effort_name = new_name
                changes.append(f"EFFORT: {old_effort}({old_name}) -> {new_effort}({new_name})")

    if controls.get('lr') is not None:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = controls['lr']
        if abs(current_lr - new_lr) > 1e-10:
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            changes.append(f"lr: {current_lr:.6f} -> {new_lr:.6f}")

    if controls.get('think_ticks') is not None and model is not None:
        current_tt = getattr(model, 'think_ticks', 0)
        new_tt = controls['think_ticks']
        if current_tt != new_tt:
            model.think_ticks = new_tt
            changes.append(f"think_ticks: {current_tt} -> {new_tt}")

    # LCX on/off toggle (sets _lcx_hash_mode which gates all LCX operations)
    if controls.get('use_lcx') is not None and model is not None:
        new_lcx = bool(controls['use_lcx'])
        old_lcx = getattr(model, '_lcx_hash_mode', False)
        if old_lcx != new_lcx:
            model._lcx_hash_mode = new_lcx
            changes.append(f"use_lcx: {old_lcx} -> {new_lcx}")

    # Stage name (informational, logged to InfluxDB)
    if controls.get('stage') is not None and model is not None:
        new_stage = controls['stage']
        old_stage = getattr(model, '_current_stage', None)
        if old_stage != new_stage:
            model._current_stage = new_stage
            changes.append(f"stage: {old_stage} -> {new_stage}")

    # Zoom LCX effort controls
    if controls.get('effort_mode') is not None and model is not None:
        model._effort_auto = (controls['effort_mode'] == 'auto')
    if controls.get('allowed_levels') is not None and model is not None:
        model._allowed_levels = controls['allowed_levels']
    if controls.get('max_zoom_level') is not None and model is not None:
        model._allowed_levels = list(range(controls['max_zoom_level'] + 1))

    if controls.get('temporal_fibonacci') is not None and model is not None:
        enabled = controls['temporal_fibonacci']
        has_periods = getattr(model, 'tick_periods', None) is not None
        if enabled and not has_periods:
            orig = getattr(model, '_original_tick_periods', None)
            if orig is not None:
                model.tick_periods = orig
                changes.append("temporal_fibonacci: OFF -> ON")
        elif not enabled and has_periods:
            if not hasattr(model, '_original_tick_periods') or model._original_tick_periods is None:
                model._original_tick_periods = model.tick_periods.clone()
            model.tick_periods = None
            changes.append("temporal_fibonacci: ON -> OFF (all beings fire every tick)")

    # Per-being state changes (may rebuild optimizer)
    if controls.get('being_states') is not None and model is not None:
        lr = optimizer.param_groups[0]['lr']
        optimizer, state_changes = apply_being_states(
            model, controls['being_states'], optimizer, lr
        )
        changes.extend(state_changes)

    if controls.get('data_weights') and loader is not None:
        loader.update_weights(controls['data_weights'])

    # Sequential dataset cycling
    if loader is not None and hasattr(loader, 'set_sequential'):
        seq_enabled = controls.get('data_sequential', False)
        seq_steps = controls.get('data_seq_steps', 100)
        if seq_enabled != getattr(loader, '_sequential', False) or \
           seq_steps != getattr(loader, '_seq_steps', 100):
            loader.set_sequential(seq_enabled, seq_steps)
            mode = f"SEQ({seq_steps} steps/dataset)" if seq_enabled else "RANDOM"
            changes.append(f"data_mode: {mode}")

    return optimizer, ", ".join(changes) if changes else ""


# ---------------------------------------------------------------------------
# Per-being state management
# ---------------------------------------------------------------------------

def apply_being_states(model, new_states: Dict[str, str], optimizer, lr: float):
    """Apply being state changes. Returns (new_optimizer, changes_list)."""
    changes = []
    rebuild_needed = False

    for idx_str, new_state in new_states.items():
        idx = int(idx_str)
        if idx < 0 or idx >= len(model.beings):
            continue
        old_state = model.being_states.get(idx, 'null')
        if old_state == new_state:
            continue

        if new_state == 'active':
            _set_being_grad(model, idx, True)
            rebuild_needed = True
            changes.append(f"B{idx}: {old_state} -> active")

        elif new_state == 'frozen':
            _set_being_grad(model, idx, False)
            rebuild_needed = True
            changes.append(f"B{idx}: {old_state} -> frozen")

        elif new_state == 'null':
            _reinit_being(model, idx)
            _set_being_grad(model, idx, False)
            rebuild_needed = True
            changes.append(f"B{idx}: {old_state} -> null (reinit)")

        model.being_states[idx] = new_state

    if rebuild_needed:
        trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
        if trainable:
            new_optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        else:
            # No trainable params — keep old optimizer (will be rebuilt when a being activates)
            new_optimizer = optimizer
        return new_optimizer, changes
    return optimizer, changes


def _set_being_grad(model, idx: int, enabled: bool):
    """Set requires_grad for all parameters of being idx."""
    for p in model.beings[idx].parameters():
        p.requires_grad_(enabled)
    for proj_list in [model.being_input_projs, model.being_output_projs]:
        if proj_list is not None:
            for p in proj_list[idx].parameters():
                p.requires_grad_(enabled)
    if getattr(model, 'capacity_fibonacci', False) and getattr(model, 'ring_read_projs', None):
        for proj_list in [model.ring_read_projs, model.ring_write_projs]:
            for p in proj_list[idx].parameters():
                p.requires_grad_(enabled)
    if getattr(model, 'being_processing_layers', None) is not None:
        for layer in model.being_processing_layers[idx]:
            for p in layer.parameters():
                p.requires_grad_(enabled)


def _reinit_being(model, idx: int):
    """Reinitialize a being's weights to random init."""
    being = model.beings[idx]
    with torch.no_grad():
        being.pointer_destinations.copy_(
            torch.rand_like(being.pointer_destinations) * model.num_memory_positions
        )
        nn.init.constant_(being.jump_gate.bias, 0.5)
        nn.init.normal_(being.jump_gate.weight, 0, 0.01)
        being.context_strength.fill_(0.2)
    for proj_list in [model.being_input_projs, model.being_output_projs]:
        nn.init.xavier_uniform_(proj_list[idx].weight)
        nn.init.zeros_(proj_list[idx].bias)
    if getattr(model, 'capacity_fibonacci', False) and getattr(model, 'ring_read_projs', None):
        for proj_list in [model.ring_read_projs, model.ring_write_projs]:
            nn.init.xavier_uniform_(proj_list[idx].weight)
            nn.init.zeros_(proj_list[idx].bias)
    if getattr(model, 'being_processing_layers', None) is not None:
        for layer in model.being_processing_layers[idx]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
