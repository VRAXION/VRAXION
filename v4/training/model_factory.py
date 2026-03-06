"""Model factory for deterministic checkpoint reconstruction.

Centralizes model build specs so train/eval/resume instantiate the exact
architecture encoded in checkpoint metadata, not whatever current YAML defaults
or code defaults happen to be.
"""

from __future__ import annotations

from pathlib import Path
import sys
import yaml

# model/ imports (train.py / eval.py add this path too, but keep it local-safe)
_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model'
_MODEL_DIR_STR = str(_MODEL_DIR)
if _MODEL_DIR_STR not in sys.path:
    sys.path.insert(0, _MODEL_DIR_STR)

from instnct import INSTNCT  # type: ignore[import-not-found]  # noqa: E402
from tiny_transformer import TinyTransformer  # type: ignore[import-not-found]  # noqa: E402


def load_model_config(v4_root: Path) -> dict:
    """Load model section from vraxion_config.yaml with strict shape checks."""
    cfg_path = v4_root / 'config' / 'vraxion_config.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, encoding='utf-8') as f:
        root = yaml.safe_load(f)
    if not isinstance(root, dict) or not isinstance(root.get('model'), dict):
        raise RuntimeError(f"Missing/invalid 'model' section in {cfg_path}")
    return root['model']


def _build_instnct_spec(embed_mode: bool, model_config: dict) -> dict:
    d_legacy = model_config.get('D', 132)
    return {
        'M': int(model_config.get('M', 256)),
        'embed_dim': None,
        'hidden_dim': int(model_config.get('hidden_dim', d_legacy)),
        'slot_dim': int(model_config.get('slot_dim', d_legacy)),
        'N': int(model_config.get('N', 2)),
        'R': int(model_config.get('R', 1)),
        'B': 8,
        'embed_mode': bool(embed_mode),
        'kernel_mode': model_config.get('kernel_mode', 'vshape'),
        'checkpoint_chunks': int(model_config.get('checkpoint_chunks', 0)),
        'expert_weighting': bool(model_config.get('expert_weighting', False)),
        'embed_encoding': model_config.get('embed_encoding', 'learned'),
        'output_encoding': model_config.get('output_encoding', 'learned'),
        'pointer_mode': model_config.get('pointer_mode', 'sequential'),
        'write_mode': model_config.get('write_mode', 'accumulate'),
        'replace_impl': model_config.get('replace_impl', 'dense'),
        'bb_enabled': bool(model_config.get('bb_enabled', False)),
        'bb_gate_bias': float(model_config.get('bb_gate_bias', 0.0)),
        'bb_scale': float(model_config.get('bb_scale', 0.1)),
        'bb_tau': float(model_config.get('bb_tau', 4.0)),
        'bb_gate_mode': model_config.get('bb_gate_mode', 'learned'),
        'topk_K': int(model_config.get('topk_K', 8)),
        's_constraint': model_config.get('s_constraint', 'softplus'),
    }


def _build_transformer_spec(embed_mode: bool, training_config: dict) -> dict:
    return {
        'embed_mode': bool(embed_mode),
        'd_model': int(training_config.get('transformer_d_model', 128)),
        'n_layers': int(training_config.get('transformer_n_layers', 3)),
        'n_heads': int(training_config.get('transformer_n_heads', 4)),
        'd_ff': int(training_config.get('transformer_d_ff', 576)),
        'max_seq': int(training_config.get('transformer_max_seq', 512)),
        'dropout': float(training_config.get('transformer_dropout', 0.0)),
    }


def build_model_spec(model_type: str, embed_mode: bool, model_config: dict, training_config: dict) -> dict:
    """Create deterministic model record (type/module/class/build_spec)."""
    mt = model_type.lower()
    if mt == 'instnct':
        return {
            'type': 'instnct',
            'module': 'instnct',
            'class_name': 'INSTNCT',
            'build_spec': _build_instnct_spec(embed_mode, model_config),
        }
    if mt == 'transformer':
        return {
            'type': 'transformer',
            'module': 'tiny_transformer',
            'class_name': 'TinyTransformer',
            'build_spec': _build_transformer_spec(embed_mode, training_config),
        }
    raise ValueError(f"Unsupported model_type: {model_type!r}")


def build_model_from_spec(model_record: dict, device: str):
    """Instantiate model exactly from checkpoint model record."""
    mtype = model_record.get('type')
    spec = model_record.get('build_spec', {})
    if not isinstance(spec, dict):
        raise TypeError('model.build_spec must be a dict')

    if mtype == 'instnct':
        model = INSTNCT(**spec)
    elif mtype == 'transformer':
        model = TinyTransformer(**spec)
    else:
        raise ValueError(f"Unsupported checkpoint model.type: {mtype!r}")
    return model.to(device)
