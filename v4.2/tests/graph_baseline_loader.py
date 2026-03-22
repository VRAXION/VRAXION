"""Helpers to load deterministic graph baselines for v4.2 benchmarking.

We compare two CPU baselines explicitly:
  - CPU_DENSE_COMMITTED: exact `origin/main:v4.2/model/graph.py`
  - CPU_SPARSE_LOCAL: current local `v4.2/model/graph.py`

This avoids accidentally benchmarking "whatever happens to be in the worktree"
without naming it.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from types import ModuleType
import importlib.util
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_GRAPH_PATH = REPO_ROOT / "v4.2" / "model" / "graph.py"
COMMITTED_GRAPH_REF = "origin/main"
COMMITTED_GRAPH_REPO_PATH = "v4.2/model/graph.py"


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_module_from_source(name: str, source: str, file_hint: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__file__ = file_hint
    exec(compile(source, file_hint, "exec"), mod.__dict__)
    sys.modules[name] = mod
    return mod


@lru_cache(maxsize=1)
def load_dense_committed_module() -> ModuleType:
    src = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{COMMITTED_GRAPH_REF}:{COMMITTED_GRAPH_REPO_PATH}"],
        text=True,
    )
    return _load_module_from_source(
        "v42_dense_committed_graph",
        src,
        f"{COMMITTED_GRAPH_REF}:{COMMITTED_GRAPH_REPO_PATH}",
    )


@lru_cache(maxsize=1)
def load_sparse_local_module() -> ModuleType:
    return _load_module_from_path("v42_sparse_local_graph", LOCAL_GRAPH_PATH)


def clone_common_state(src_net, dst_net) -> None:
    """Copy the shared state fields between the dense committed and local sparse nets."""
    dst_net.mask[:] = src_net.mask
    if hasattr(dst_net, "resync_alive"):
        dst_net.resync_alive()
    dst_net.state[:] = src_net.state
    dst_net.charge[:] = src_net.charge
    if hasattr(src_net, "input_projection") and hasattr(dst_net, "input_projection"):
        if src_net.input_projection.shape == dst_net.input_projection.shape:
            dst_net.input_projection[:] = src_net.input_projection
    if hasattr(src_net, "output_projection") and hasattr(dst_net, "output_projection"):
        if src_net.output_projection.shape == dst_net.output_projection.shape:
            dst_net.output_projection[:] = src_net.output_projection
    if hasattr(src_net, "theta") and hasattr(dst_net, "theta"):
        if src_net.theta.shape == dst_net.theta.shape:
            dst_net.theta[:] = src_net.theta
    if hasattr(src_net, "decay") and hasattr(dst_net, "decay"):
        if src_net.decay.shape == dst_net.decay.shape:
            dst_net.decay[:] = src_net.decay
    if hasattr(src_net, "loss_pct") and hasattr(dst_net, "loss_pct"):
        dst_net.loss_pct = np.int8(src_net.loss_pct)
    if hasattr(src_net, "drive") and hasattr(dst_net, "drive"):
        dst_net.drive = np.int8(src_net.drive)
    if hasattr(src_net, "mood_x") and hasattr(dst_net, "mood_x"):
        dst_net.mood_x = src_net.mood_x
    if hasattr(src_net, "mood_z") and hasattr(dst_net, "mood_z"):
        dst_net.mood_z = src_net.mood_z
    if hasattr(src_net, "leak") and hasattr(dst_net, "leak"):
        dst_net.leak = src_net.leak
    if hasattr(dst_net, "_weff_dirty"):
        dst_net._weff_dirty = True


def build_paired_nets(vocab: int, neurons: int, density: float, seed: int):
    """Instantiate both baselines with identical initial state."""
    dense_mod = load_dense_committed_module()
    sparse_mod = load_sparse_local_module()

    dense_mod.np.random.seed(seed)
    dense_mod.random.seed(seed)
    dense_net = dense_mod.SelfWiringGraph(neurons, vocab, density=density)

    sparse_mod.np.random.seed(seed)
    sparse_mod.random.seed(seed)
    sparse_net = sparse_mod.SelfWiringGraph(neurons, vocab, density=density)
    clone_common_state(dense_net, sparse_net)

    return dense_mod, dense_net, sparse_mod, sparse_net
