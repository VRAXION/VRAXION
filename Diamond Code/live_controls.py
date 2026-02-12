"""
Live controls for Diamond Code swarm training.

Reads logs/swarm/controls.json every N steps. If the file doesn't exist
or has parse errors, uses defaults silently.

Edit controls.json while training runs to adjust LR, data mixing, think ticks,
and per-being states (null/active/frozen).

Format:
    {
        "lr": 0.0001,
        "think_ticks": 0,
        "being_states": {"0": "null", "6": "active"},
        "data_weights": {
            "shakespeare.traindat": 3.0,
            "xor.traindat": 1.0
        }
    }
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn


DEFAULT_CONTROLS = {
    "lr": None,
    "think_ticks": None,
    "checkpoint_every": None,
    "eval_every": None,
    "eval_samples": None,
    "temporal_fibonacci": None,
    "being_states": None,
    "data_weights": {},
}


def write_default_controls(path: str, lr: float, data_weights: Dict[str, float],
                           think_ticks: int = 0, checkpoint_every: int = 50,
                           eval_every: int = 10):
    """Write initial controls.json at training start. Preserves existing file."""
    if os.path.exists(path):
        return  # Preserve user edits
    controls = {
        "lr": lr,
        "think_ticks": think_ticks,
        "checkpoint_every": checkpoint_every,
        "eval_every": eval_every,
        "data_weights": data_weights,
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
            if 'checkpoint_every' in data and isinstance(data['checkpoint_every'], int):
                result['checkpoint_every'] = int(data['checkpoint_every'])
            if 'eval_samples' in data and isinstance(data['eval_samples'], int):
                result['eval_samples'] = int(data['eval_samples'])
            if 'temporal_fibonacci' in data and isinstance(data['temporal_fibonacci'], bool):
                result['temporal_fibonacci'] = data['temporal_fibonacci']
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
        return result
    except Exception:
        return dict(DEFAULT_CONTROLS)


def apply_controls(controls: Dict[str, Any], optimizer, loader=None, model=None):
    """
    Apply controls to optimizer, loader, and model.
    Returns (optimizer, description_of_changes).

    Safe to call every N steps. Only applies non-None values.
    The optimizer may be rebuilt if being states change.
    """
    changes = []

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
