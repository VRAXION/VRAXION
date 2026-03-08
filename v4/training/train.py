"""INSTNCT v4 Training Harness — byte-level model training with masked supervision.

Connects the data pipeline (.traindat + .mask files) to the INSTNCT model.
Supports both synthetic data (echo, denoise from generate.py) and real-world
data (text/code shards from convert.py) in the same format.

Two modes:
    binary (default): each byte → 8-bit float vector, MSE loss
    embed (--embed):  each byte → token index 0-255, CrossEntropy loss

Usage:
    python train.py --data training_data --steps 10000
    python train.py --data training_data/echo256.traindat --embed --steps 500
    python train.py --resume training_output/ckpt_latest.pt --steps 20000
    python train.py --data training_data --device cuda --batch 64 --seq 256
"""

import argparse
import bisect
import csv
import hashlib
import math
import os
import platform
import random
import socket
import subprocess
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── Model import ──────────────────────────────────────────────
# INSTNCT lives in model/instnct.py — a sibling directory to training/.
# Python can't find it without help, so we add model/ to the import path.
# This is a standard R&D pattern; a proper package install replaces it later.

# resolve() normalizes the path (symlinks, case) — robust across platforms;
# parent.parent walks: training/train.py → training/ → v4/ — the project root;
# then we append 'model' to land in v4/model/ where instnct.py lives.
_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model'

# fail fast with a clear message if the directory structure has changed —
# without this, the import below would raise a confusing ModuleNotFoundError
if not _MODEL_DIR.is_dir():
    raise FileNotFoundError(f"Model directory not found: {_MODEL_DIR}")

# convert to string for sys.path compatibility, then guard against duplicate
# entries (matters if this module is somehow imported more than once)
_MODEL_DIR_STR = str(_MODEL_DIR)
if _MODEL_DIR_STR not in sys.path:
    sys.path.insert(0, _MODEL_DIR_STR)  # index 0 = highest priority on the search path

# now the import resolves to v4/model/instnct.py — the INSTNCT ring-buffer network;
# type: ignore silences Pylance (can't follow runtime sys.path);
# noqa: E402 silences flake8 (import not at top-of-file, because sys.path must come first)
from model_factory import (  # type: ignore[import-not-found]  # noqa: E402
    build_model_from_spec,
    build_model_spec,
    load_model_config,
)


# ── Config ────────────────────────────────────────────────────
# vraxion_config.yaml is the single source of truth for all runtime parameters.
# This loader mirrors the one in instnct.py — both read the same YAML,
# just different sections (model vs training).

def _load_yaml(section: str) -> dict:
    """Load one section from vraxion_config.yaml. Fails loud if missing."""

    # navigate: training/train.py → training/ → v4/ → config/vraxion_config.yaml
    cfg_path = Path(__file__).parent.parent / 'config' / 'vraxion_config.yaml'

    # existence check — fail immediately with a clear path, not a buried IOError
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # safe_load blocks arbitrary Python execution (unlike yaml.load) — always use safe;
    # YAMLError catch turns cryptic scanner exceptions into a one-line message
    try:
        with open(cfg_path, encoding='utf-8') as f:
            root = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Corrupted YAML in {cfg_path}: {e}") from e

    # empty YAML returns None, broken YAML might return a list — guard both
    if not isinstance(root, dict):
        raise RuntimeError(f"Expected dict in {cfg_path}, got {type(root).__name__}")

    # tell the user what sections DO exist — saves a round-trip to open the file manually
    if section not in root:
        raise KeyError(f"Missing '{section}' in {cfg_path} (have: {list(root.keys())})")

    # guard against `training: 42` or `training: [list]` — must be a key-value mapping;
    # without this, the caller would get a confusing TypeError deep in config access
    result = root[section]
    if not isinstance(result, dict):
        raise TypeError(
            f"Section '{section}' in {cfg_path} must be a dict, "
            f"got {type(result).__name__}: {result!r}"
        )

    return result


# checkpoint format version — bump when adding/removing keys from the saved dict;
# func_loadckpt_dct can compare this to decide if a checkpoint is compatible
CKPT_VERSION = 2


# ── Data Discovery ────────────────────────────────────────────
# The data pipeline produces .traindat + .mask file pairs (generate.py, convert.py).
# This function finds them, no matter how the user points at them:
# a single file, a flat directory, or nested shard subdirectories.

def func_discover_dat(data_path_str: str) -> list[tuple[Path, Path, int]]:
    """Discover .traindat + .mask file pairs from a path.

    Handles three cases:
        1. Specific .traindat file → single pair
        2. Flat directory with .traindat files → all pairs
        3. Directory with shard subdirs (real_text/, real_code/) → recurse

    Returns list of (traindat_path, mask_path, byte_count) sorted by path.
    Raises FileNotFoundError if nothing found."""

    # CLI passes strings; convert once at the boundary so the rest works with Path
    data_path = Path(data_path_str)

    # ── Case 1: explicit .traindat file path ──
    if data_path.is_file() and data_path.suffix == '.traindat':
        # convention: echo256.traindat always has a matching echo256.mask next to it
        mask_path = data_path.with_suffix('.mask')
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file missing: {mask_path}")

        # reject 0-byte files — same guard as the directory branch below;
        # without this, ByteDataset would create a 0-sample dataset
        sz = data_path.stat().st_size
        if sz == 0:
            raise ValueError(f"Training data file is empty (0 bytes): {data_path}")

        # mask must be exactly the same size as traindat — one mask byte per data byte;
        # a mismatch means stale mask from a previous generation run, and would cause
        # silent out-of-bounds reads or garbage supervision signals in ByteDataset
        mask_sz = mask_path.stat().st_size
        if mask_sz != sz:
            raise ValueError(
                f"Size mismatch: {data_path.name} is {sz} bytes "
                f"but {mask_path.name} is {mask_sz} bytes"
            )

        return [(data_path, mask_path, sz)]

    # ── Case 2/3: directory (flat or with shard subdirs) ──
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # rglob recurses into subdirectories — finds shards in real_text/, real_code/ etc.;
    # sorted() guarantees deterministic ordering across runs (reproducibility)
    pairs = []
    for td in sorted(data_path.rglob('*.traindat')):
        mask = td.with_suffix('.mask')

        # warn if a .traindat has no matching .mask — likely a generation bug;
        # we skip it rather than crash, but the user should know
        if not mask.exists():
            print(f'  [WARN] skipping {td.name} — no matching .mask file')
            continue

        # skip empty files — they contribute 0 samples and waste a memmap slot
        sz = td.stat().st_size
        if sz == 0:
            continue

        # mask must be byte-for-byte aligned with traindat — one mask byte per data byte;
        # a stale or truncated mask would silently corrupt supervision in ByteDataset
        mask_sz = mask.stat().st_size
        if mask_sz != sz:
            print(f'  [WARN] skipping {td.name} — size mismatch '
                  f'(data={sz}, mask={mask_sz})')
            continue

        pairs.append((td, mask, sz))

    if not pairs:
        raise FileNotFoundError(f"No .traindat/.mask pairs found in {data_path}")

    return pairs


# ── Dataset ───────────────────────────────────────────────────
# The core data feeder: maps raw bytes from disk into tensors the model can eat.
# Uses memory-mapped I/O (np.memmap) so multi-GB datasets don't fill RAM —
# the OS pages in bytes on demand, and we never hold the full file in memory.

class ByteDataset:
    """Memory-mapped random-offset dataset for .traindat + .mask files.

    Concatenates multiple files into a virtual byte stream. Samples
    random byte positions and returns (x, y, mask) tuples ready for
    the model.

    Binary mode: x/y = (batch, seq_len, 8) float32
    Embed mode:  x/y = (batch, seq_len) int64
    Mask always: (batch, seq_len) or (batch, seq_len, 1) float32
    """

    def __init__(self, file_pairs: list[tuple[Path, Path, int]],
                 seq_len: int, embed_mode: bool,
                 seed: int | None = None):
        self.seq_len = seq_len
        self.embed_mode = embed_mode
        self.file_pairs = file_pairs
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # cumulative byte offsets: [0, sz1, sz1+sz2, ...] — treats N files as one
        # contiguous stream; bisect_right finds which file a global offset falls in
        self._cum = [0]
        for _, _, sz in file_pairs:
            self._cum.append(self._cum[-1] + sz)

        # mmaps are opened lazily on first access and cached for the session;
        # dict (not list) because file indices may not be contiguous in future
        self._data_mmaps: dict[int, np.ndarray] = {}
        self._mask_mmaps: dict[int, np.ndarray] = {}

    def get_rng_state(self) -> dict:
        """Return dataset RNG state (serializable dict)."""
        return self.rng.bit_generator.state

    def set_rng_state(self, state: dict):
        """Restore dataset RNG state from checkpoint payload."""
        self.rng.bit_generator.state = state

    @property
    def total_bytes(self) -> int:
        return self._cum[-1]

    @property
    def n_samples(self) -> int:
        """Number of valid random start positions.
        Each sample reads seq_len+1 bytes (x overlaps y by 1), so the last
        valid offset is total_bytes - seq_len - 1, giving total_bytes - seq_len
        valid positions (0 through total_bytes - seq_len - 1 inclusive)."""
        return max(0, self.total_bytes - self.seq_len)

    def _get_mmap(self, file_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get or create mmap for file at index."""
        if file_idx not in self._data_mmaps:
            td_path, mask_path, _ = self.file_pairs[file_idx]
            # mode='r' = read-only; the OS handles paging from disk transparently
            self._data_mmaps[file_idx] = np.memmap(td_path, dtype=np.uint8, mode='r')
            self._mask_mmaps[file_idx] = np.memmap(mask_path, dtype=np.uint8, mode='r')
        return self._data_mmaps[file_idx], self._mask_mmaps[file_idx]

    def _read_bytes(self, global_offset: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Read `length` bytes of data + mask starting at global_offset.
        Handles spans crossing file boundaries and circular wrap-around
        at the end of the dataset (offset modulo n_samples)."""

        # pre-allocate output buffers — filled in chunks if the read spans files
        data_buf = np.empty(length, dtype=np.uint8)
        mask_buf = np.empty(length, dtype=np.uint8)
        written = 0

        while written < length:
            # wrap around the dataset — circular corpus semantics
            pos = (global_offset + written) % self.n_samples
            # bisect_right on cumulative offsets → O(log n) file lookup;
            # subtract 1 because cum[i] is the START of file i
            file_idx = bisect.bisect_right(self._cum, pos) - 1
            # local_off = position within this specific file
            local_off = pos - self._cum[file_idx]
            data_mm, mask_mm = self._get_mmap(file_idx)
            # how many bytes remain in this file from local_off onward
            available = len(data_mm) - local_off
            # take as many as we need, but don't exceed what's available
            take = min(length - written, available)

            data_buf[written:written + take] = data_mm[local_off:local_off + take]
            mask_buf[written:written + take] = mask_mm[local_off:local_off + take]
            written += take
            # if take < remaining, the loop continues (next file or wrap-around)

        return data_buf, mask_buf

    def sample_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of sequences.

        Returns (x, y, supervision_mask):
            binary: x/y = (B, T, 8) float, mask = (B, T, 1) float
            embed:  x/y = (B, T) long,     mask = (B, T) float
        """
        T = self.seq_len
        # uniform random offsets into the virtual byte stream;
        # each offset yields a (T+1)-byte window: x = [0:T], y = [1:T+1]
        offsets = self.rng.integers(0, self.n_samples, size=batch_size)

        # read raw bytes for all samples: T+1 because y is shifted by 1
        data_all = np.empty((batch_size, T + 1), dtype=np.uint8)
        mask_all = np.empty((batch_size, T + 1), dtype=np.uint8)
        for i, off in enumerate(offsets):
            data_all[i], mask_all[i] = self._read_bytes(int(off), T + 1)

        # supervision mask aligned with targets — mask[1:] because we predict
        # byte[t+1] from byte[t], so supervision applies to the prediction target
        sup = mask_all[:, 1:].astype(np.float32)  # (batch, T)

        if self.embed_mode:
            # embed mode: each byte is a token index (0–255), model outputs (B, T, 256)
            x = torch.from_numpy(data_all[:, :T].astype(np.int64)).to(device)
            y = torch.from_numpy(data_all[:, 1:T + 1].astype(np.int64)).to(device)
            mask_t = torch.from_numpy(sup).to(device)          # (B, T) — flat for CE loss
        else:
            # binary mode: unpack each byte into 8 float bits (MSB first)
            flat = np.unpackbits(data_all.reshape(-1))         # flatten → unpack all
            bits = flat.reshape(batch_size, T + 1, 8).astype(np.float32)
            # .copy() is critical — without it, torch.from_numpy shares the numpy
            # buffer, and any later numpy operation could silently corrupt the tensor
            x = torch.from_numpy(bits[:, :T].copy()).to(device)
            y = torch.from_numpy(bits[:, 1:T + 1].copy()).to(device)
            # mask is (B, T, 1) for broadcasting against (B, T, 8) predictions
            mask_t = torch.from_numpy(sup).unsqueeze(-1).to(device)  # (B, T, 1)

        return x, y, mask_t

    def init_sequential(self, batch_size: int):
        """Initialize sequential offsets — evenly spaced across the dataset.
        Called once before the training loop when sequential mode is on."""
        n = self.n_samples
        self._seq_offsets = np.linspace(0, n - 1, batch_size, dtype=np.int64)

    def sample_batch_sequential(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read next seq_len+1 bytes sequentially from each batch element's offset.

        Same return format as sample_batch(). After reading, each offset
        advances by seq_len (no overlap, no gap). Wraps around at dataset end.
        """
        T = self.seq_len
        n = self.n_samples

        # lazy init if not already set (e.g. resumed without saved offsets)
        if not hasattr(self, '_seq_offsets') or len(self._seq_offsets) != batch_size:
            self.init_sequential(batch_size)

        data_all = np.empty((batch_size, T + 1), dtype=np.uint8)
        mask_all = np.empty((batch_size, T + 1), dtype=np.uint8)
        for i in range(batch_size):
            data_all[i], mask_all[i] = self._read_bytes(int(self._seq_offsets[i]), T + 1)
            self._seq_offsets[i] = (self._seq_offsets[i] + T) % n

        sup = mask_all[:, 1:].astype(np.float32)

        if self.embed_mode:
            x = torch.from_numpy(data_all[:, :T].astype(np.int64)).to(device)
            y = torch.from_numpy(data_all[:, 1:T + 1].astype(np.int64)).to(device)
            mask_t = torch.from_numpy(sup).to(device)
        else:
            flat = np.unpackbits(data_all.reshape(-1))
            bits = flat.reshape(batch_size, T + 1, 8).astype(np.float32)
            x = torch.from_numpy(bits[:, :T].copy()).to(device)
            y = torch.from_numpy(bits[:, 1:T + 1].copy()).to(device)
            mask_t = torch.from_numpy(sup).unsqueeze(-1).to(device)

        return x, y, mask_t


# ── Masked Loss ───────────────────────────────────────────────
# Two loss functions — one per training mode (binary / embed).
# Both return (raw_loss, masked_loss):
#   raw_loss   = unmasked average over ALL positions (for logging context)
#   masked_loss = average over SUPERVISED positions only (the actual training signal)
# The mask (0=unsupervised, 1=supervised) controls which positions contribute
# to gradients. Unsupervised positions still see raw_loss for diagnostics.

def func_maskloss_mse(pred, target, mask):
    """Masked MSE loss for binary mode.

    Args:
        pred:   (B, T, 8) float — model output, 8 bits per position
        target: (B, T, 8) float — ground truth bits
        mask:   (B, T, 1) float — 0=unsupervised, 1=supervised

    Returns (raw_loss, masked_loss) — both scalar tensors."""

    # raw_loss: standard unmasked MSE — logs how the model is doing on everything,
    # including unsupervised positions; useful for detecting overfitting to mask patterns
    raw_loss = F.mse_loss(pred, target)

    # manual per-element squared error — we need the un-reduced tensor
    # to apply the mask before averaging (F.mse_loss reduces internally)
    sq_err = (pred - target) ** 2               # (B, T, 8)

    # zero out unsupervised positions: mask (B,T,1) broadcasts across the 8-bit dim;
    # only supervised positions contribute to the numerator
    masked_sq = sq_err * mask                   # zero where unsupervised

    # denominator: total supervised ELEMENTS (not positions) — mask.sum() counts
    # supervised positions, ×8 because each position has 8 bit-values;
    # clamp(min=1) prevents 0/0 = NaN when an entire batch is unsupervised
    n_sup = mask.sum() * pred.shape[-1]         # total supervised elements
    masked_loss = masked_sq.sum() / n_sup.clamp(min=1)

    return raw_loss, masked_loss


def func_maskloss_ce(pred, target, mask):
    """Masked CrossEntropy loss for embed mode.

    Args:
        pred:   (B, T, 256) float logits — one logit per possible byte value
        target: (B, T) long — ground truth byte values 0-255
        mask:   (B, T) float — 0=unsupervised, 1=supervised

    Returns (raw_loss, masked_loss) — both scalar tensors."""

    # F.cross_entropy expects class dim at index 1: (B, V, T) not (B, T, V).
    # transpose(1,2) swaps dims without copying data (stride metadata only) —
    # this avoids the reshape(-1, V) → reshape(B, T) round-trip which could
    # trigger a hidden contiguous() copy if pred's memory layout isn't C-contiguous.
    # The output is already (B, T), no reshape needed.
    per_pos = F.cross_entropy(
        pred.transpose(1, 2), target, reduction='none'
    )  # (B, T) — one loss value per position, no reshape needed
    raw_loss = per_pos.mean()                   # unmasked average for logging

    # mask out unsupervised positions — per_pos is already (B, T)
    masked_ce = per_pos * mask                  # zero where unsupervised

    # average over supervised positions only; clamp prevents NaN on all-zero masks
    n_sup = mask.sum().clamp(min=1)
    masked_loss = masked_ce.sum() / n_sup

    return raw_loss, masked_loss


# ── Accuracy ─────────────────────────────────────────────────
# Two accuracy functions — one per training mode (binary / embed).
# Both return (raw_acc, masked_acc):
#   raw_acc    = fraction correct over ALL positions (for logging context)
#   masked_acc = fraction correct over SUPERVISED positions only
# Computed from the same pred/target tensors used for loss — zero extra cost.
# These run inside the training loop but do NOT contribute to gradients.

@torch.no_grad()
def func_accuracy_bin(pred, target, mask):
    """Bit-level accuracy for binary mode.

    pred/target are (B, T, 8) float tensors. A bit is "correct" when
    round(pred) == round(target) — both snapped to {0, 1}.
    mask is (B, T, 1) — already broadcastable against (B, T, 8).
    Returns (raw_acc, masked_acc) — both float scalars."""

    # round both to {0,1} and compare element-wise; shape (B, T, 8)
    correct = (pred.round() == target.round()).float()

    # raw: fraction correct over all B×T×8 bits
    raw_acc = correct.mean().item()

    # masked: only count supervised positions (mask=1)
    # mask arrives as (B, T, 1) in binary mode — broadcasts against (B, T, 8) naturally
    n_sup = mask.sum() * pred.shape[-1]                # total supervised bits
    masked_acc = (correct * mask).sum() / n_sup.clamp(min=1)

    return raw_acc, masked_acc.item()


@torch.no_grad()
def func_accuracy_emb(pred, target, mask):
    """Byte-level accuracy for embed mode.

    pred is (B, T, 256) float logits — argmax gives the predicted byte.
    target is (B, T) long — the actual byte value.
    Returns (raw_acc, masked_acc) — both float scalars."""

    # argmax over vocab dim → predicted byte index; shape (B, T)
    correct = (pred.argmax(-1) == target).float()      # 1.0 = exact match, 0.0 = miss

    # raw: fraction correct over all B×T positions
    raw_acc = correct.mean().item()

    # masked: only count supervised positions (mask=1); mask is already (B, T)
    n_sup = mask.sum().clamp(min=1)
    masked_acc = (correct * mask).sum() / n_sup

    return raw_acc, masked_acc.item()


# ── LR Schedule ──────────────────────────────────────────────
# Warmup is stateless (depends only on step number).
# After warmup, LR is managed by _PlateauTracker (if lr_schedule=plateau)
# or stays constant (if lr_schedule=constant).

def _compute_lr(base_lr: float, step: int, warmup: int) -> float:
    """Warmup phase only. After warmup, LR is managed by plateau tracker."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


class _PlateauTracker:
    """ReduceLROnPlateau — stateful, checkpoint-friendly.

    Tracks best loss and reduces LR by `factor` when loss stalls for
    `patience` steps. Minimum LR is floored at `min_lr`.
    """

    def __init__(self, base_lr: float, factor: float = 0.5,
                 patience: int = 150, min_lr: float = 1e-5):
        self.base_lr = base_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.stale_steps = 0
        self.multiplier = 1.0

    def step(self, loss: float) -> float:
        """Call every training step. Returns current LR."""
        if loss < self.best_loss:
            self.best_loss = loss
            self.stale_steps = 0
        else:
            self.stale_steps += 1

        if self.stale_steps >= self.patience:
            new_mult = self.multiplier * self.factor
            if self.base_lr * new_mult >= self.min_lr:
                self.multiplier = new_mult
                self.stale_steps = 0
                print(f'  [LR] Plateau detected — LR reduced to '
                      f'{self.base_lr * self.multiplier:.2e} '
                      f'(x{self.multiplier:.4f})')

        return max(self.base_lr * self.multiplier, self.min_lr)

    def state_dict(self) -> dict:
        return {
            'best_loss': self.best_loss,
            'stale_steps': self.stale_steps,
            'multiplier': self.multiplier,
        }

    def load_state_dict(self, d: dict):
        self.best_loss = d.get('best_loss', float('inf'))
        self.stale_steps = d.get('stale_steps', 0)
        self.multiplier = d.get('multiplier', 1.0)


def _sha256_head(path: Path, n_bytes: int = 1 << 20) -> str:
    """Fast content fingerprint: SHA-256 over the first N bytes."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _build_file_manifest(file_pairs: list[tuple[Path, Path, int]]) -> list[dict]:
    """Build immutable-ish dataset manifest for drift detection."""
    manifest = []
    for td_path, mask_path, sz in file_pairs:
        td_stat = td_path.stat()
        mk_stat = mask_path.stat()
        manifest.append({
            'path': str(td_path.resolve()),
            'size': int(sz),
            'mtime_ns': int(td_stat.st_mtime_ns),
            'sha256_head_1mb': _sha256_head(td_path),
            'mask_path': str(mask_path.resolve()),
            'mask_size': int(mask_path.stat().st_size),
            'mask_mtime_ns': int(mk_stat.st_mtime_ns),
            'mask_sha256_head_1mb': _sha256_head(mask_path),
        })
    return manifest


def _compare_manifest(expected: list[dict], actual: list[dict]) -> list[str]:
    """Return human-readable drift messages."""
    msgs = []
    if len(expected) != len(actual):
        msgs.append(f'file count mismatch: checkpoint={len(expected)}, runtime={len(actual)}')
        return msgs
    for i, (e, a) in enumerate(zip(expected, actual)):
        for k in ('path', 'size', 'mtime_ns', 'sha256_head_1mb',
                  'mask_path', 'mask_size', 'mask_mtime_ns', 'mask_sha256_head_1mb'):
            if e.get(k) != a.get(k):
                msgs.append(f'file[{i}] {k} mismatch: ckpt={e.get(k)!r}, runtime={a.get(k)!r}')
    return msgs


def _get_git_commit(v4_root: Path) -> str:
    """Best-effort git hash for experiment provenance."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(v4_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return 'unknown'


def _capture_env_metadata(v4_root: Path) -> dict:
    return {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda': torch.version.cuda or '',
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'git_commit': _get_git_commit(v4_root),
    }


def _capture_rng_state(dataset: ByteDataset, eval_seed: int) -> dict:
    """Collect all RNG sources required for deterministic resume."""
    out = {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_cpu_rng_state': torch.get_rng_state(),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'torch_deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
        'dataset_rng_state': dataset.get_rng_state(),
        'eval_seed': int(eval_seed),
    }
    if torch.cuda.is_available():
        out['torch_cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng_state(rng_state: dict, dataset: ByteDataset):
    """Restore RNG state from checkpoint payload."""
    random.setstate(rng_state['python_random_state'])
    np.random.set_state(rng_state['numpy_random_state'])
    cpu_rng = rng_state['torch_cpu_rng_state']
    if not isinstance(cpu_rng, torch.ByteTensor):
        cpu_rng = torch.ByteTensor(cpu_rng.cpu().numpy().astype('uint8'))
    torch.set_rng_state(cpu_rng)
    if torch.cuda.is_available() and 'torch_cuda_rng_state_all' in rng_state:
        cuda_states = []
        for s in rng_state['torch_cuda_rng_state_all']:
            if not isinstance(s, torch.ByteTensor):
                s = torch.ByteTensor(s.cpu().numpy().astype('uint8'))
            cuda_states.append(s)
        torch.cuda.set_rng_state_all(cuda_states)
    torch.backends.cudnn.benchmark = bool(rng_state.get('cudnn_benchmark', False))
    torch.backends.cudnn.deterministic = bool(rng_state.get('cudnn_deterministic', False))
    if rng_state.get('torch_deterministic_algorithms', False):
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)
    if 'dataset_rng_state' in rng_state:
        dataset.set_rng_state(rng_state['dataset_rng_state'])


# ── Checkpoint ────────────────────────────────────────────────
# V2 saves decision-complete state for deterministic resume:
# - model build spec + state_dict
# - optimizer state
# - resolved train/model config snapshots
# - dataset manifest + sequential offsets
# - sequence state (ring/ptr/hidden/bb_*)
# - RNG state (python/numpy/torch/cuda/dataset)
# - runtime env metadata
# CKPT_VERSION gates loading to avoid silent schema drift.

def func_saveckpt_non(path: str | Path, model, optimizer, step: int, best_loss: float,
                      run_id: str, model_record: dict, train_config_resolved: dict,
                      model_config_resolved: dict, data_state: dict,
                      rng_state: dict, env_meta: dict,
                      sequence_state: dict | None = None,
                      lr_plateau_state: dict | None = None):
    """Save v2 training checkpoint with full deterministic resume payload."""

    tmp_path = str(path) + '.tmp'

    # sequence/TBPTT state is optional; when absent this remains an empty dict.
    seq_payload: dict = {}
    if sequence_state is not None:
        for key, value in sequence_state.items():
            if torch.is_tensor(value):
                seq_payload[key] = value.cpu()
            else:
                seq_payload[key] = value

    save_dict = {
        'ckpt_version': CKPT_VERSION,
        'run_id': run_id,
        'step': int(step),
        'best_loss': float(best_loss),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model': {
            'type': model_record['type'],
            'module': model_record['module'],
            'class_name': model_record['class_name'],
            'build_spec': model_record['build_spec'],
            'state_dict': model.state_dict(),
        },
        'optimizer': {
            'class_name': optimizer.__class__.__name__,
            'state_dict': optimizer.state_dict(),
        },
        'train_config_resolved': train_config_resolved,
        'model_config_resolved': model_config_resolved,
        'data_state': data_state,
        'sequence_state': seq_payload,
        'rng_state': rng_state,
        'env': env_meta,
    }
    if lr_plateau_state is not None:
        save_dict['lr_plateau_state'] = lr_plateau_state
    # AMP GradScaler state is passed via train_config_resolved['_amp_scaler_state']
    # (optional — only present when use_amp=True)
    _amp_state = train_config_resolved.get('_amp_scaler_state')
    if _amp_state is not None:
        save_dict['amp_scaler_state'] = _amp_state

    torch.save(save_dict, tmp_path)
    os.replace(tmp_path, str(path))


def func_loadckpt_dct(path, device):
    """Load checkpoint from disk. Returns raw dict.

    Validates checkpoint format version against CKPT_VERSION — a mismatch
    means the checkpoint layout changed and blindly resuming could load
    wrong keys or missing state, silently corrupting the training run."""

    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False because we save config (a plain dict), not just tensors;
    # this is safe here because checkpoints are self-generated, not untrusted input
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # guard against loading a non-checkpoint file (e.g. a raw tensor or model-only export);
    # without this, the .get() below would throw a cryptic AttributeError
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Expected checkpoint dict, got {type(ckpt).__name__}. "
            f"Is {path} a valid training checkpoint?"
        )

    # version gate — this build only supports v2 checkpoints.
    saved_ver = ckpt.get('ckpt_version', 0)
    if saved_ver != CKPT_VERSION:
        raise RuntimeError(
            f"Checkpoint version mismatch: file v{saved_ver}, code v{CKPT_VERSION}. "
            "Legacy checkpoints are intentionally unsupported in this build."
        )

    # strict schema validation (decision-complete resume contract)
    required_top = (
        'run_id', 'step', 'best_loss', 'timestamp_utc',
        'model', 'optimizer',
        'train_config_resolved', 'model_config_resolved',
        'data_state', 'sequence_state', 'rng_state', 'env',
    )
    missing_top = [k for k in required_top if k not in ckpt]
    if missing_top:
        raise RuntimeError(f"Invalid v2 checkpoint: missing keys: {missing_top}")

    model_obj = ckpt['model']
    for k in ('type', 'module', 'class_name', 'build_spec', 'state_dict'):
        if k not in model_obj:
            raise RuntimeError(f"Invalid v2 checkpoint: model.{k} missing")

    opt_obj = ckpt['optimizer']
    for k in ('class_name', 'state_dict'):
        if k not in opt_obj:
            raise RuntimeError(f"Invalid v2 checkpoint: optimizer.{k} missing")

    data_obj = ckpt['data_state']
    for k in ('data_dir', 'seq_len', 'batch_size', 'embed_mode', 'sequential', 'file_manifest'):
        if k not in data_obj:
            raise RuntimeError(f"Invalid v2 checkpoint: data_state.{k} missing")

    rng_obj = ckpt['rng_state']
    required_rng = (
        'python_random_state',
        'numpy_random_state',
        'torch_cpu_rng_state',
        'cudnn_benchmark',
        'cudnn_deterministic',
        'torch_deterministic_algorithms',
        'dataset_rng_state',
        'eval_seed',
    )
    for k in required_rng:
        if k not in rng_obj:
            raise RuntimeError(f"Invalid v2 checkpoint: rng_state.{k} missing")

    return ckpt


# ── CSV Logger ────────────────────────────────────────────────
# Lightweight append-only CSV for training curves. One row per log interval.
# The file is human-readable and trivially loadable by pandas/Excel/gnuplot.
# Flush-per-row minimizes data loss on crash — at most one row can be lost
# (flush pushes to OS cache, not physical disk, but that's good enough for CSV logs).

class CSVLogger:
    """Append-only CSV logger for training metrics."""

    # column order is the contract — tools that parse train_log.csv depend on this.
    # accuracy = fraction of correct predictions over ALL positions (raw).
    # masked_acc = fraction of correct predictions over SUPERVISED positions only.
    COLUMNS = ['step', 'raw_loss', 'masked_loss', 'accuracy', 'masked_acc',
               'lr', 'elapsed_s', 'samples_seen', 'mask_frac',
               'ring_norm', 'ring_slot_mean',
               'alpha_0_mean', 'alpha_0_min', 'alpha_0_max',
               'alpha_1_mean', 'alpha_1_min', 'alpha_1_max',
               'input_norm_0', 'input_norm_1',
               'ring_signal_norm_0', 'ring_signal_norm_1',
               'blended_norm_0', 'blended_norm_1',
               'hidden_norm_0', 'hidden_norm_1',
               'hidden_final_norm_0', 'hidden_final_norm_1',
               'ptr_pos_0', 'ptr_pos_1',
               # ── BB telemetry (per-expert, flowchart order) ──
               'bb_beta_0', 'bb_beta_1',                   # gate openness
               'bb_ctx_raw_norm_0', 'bb_ctx_raw_norm_1',   # cache read (before gate)
               'bb_ctx_scaled_norm_0', 'bb_ctx_scaled_norm_1',  # cache read (after gate+scale)
               'bb_attn_entropy_0', 'bb_attn_entropy_1',   # attention sharpness
               'bb_query_norm_0', 'bb_query_norm_1',       # expert query strength
               'bb_key_norm',                               # written key strength
               'bb_ctx_vs_input_0', 'bb_ctx_vs_input_1',   # ratio: bb_ctx / input_vec
               'bb_ctx_vs_ring_0', 'bb_ctx_vs_ring_1',     # ratio: bb_ctx / blended_ring
               ]

    def __init__(self, path: Path):
        self.path = path

        # write header only on first creation or if the file is empty;
        # on resume, the file already has a header + previous rows — just append
        write_header = not path.exists() or path.stat().st_size == 0

        # newline='' is required by the csv module on Windows to prevent double \r\n;
        # append mode ('a') ensures resume doesn't overwrite previous log entries
        self._f = open(path, 'a', newline='', encoding='utf-8')
        self._w = csv.writer(self._f)
        if write_header:
            self._w.writerow(self.COLUMNS)
            self._f.flush()

    def log(self, step, raw_loss, masked_loss, accuracy, masked_acc,
            lr, elapsed_s, samples_seen, mask_frac, diag=None):
        """Append one row. Flushes immediately so the last step is never lost."""
        d = diag or {}
        self._w.writerow([
            step,
            f'{raw_loss:.6f}',
            f'{masked_loss:.6f}',
            f'{accuracy:.6f}',
            f'{masked_acc:.6f}',
            f'{lr:.6f}',
            f'{elapsed_s:.1f}',
            samples_seen,
            f'{mask_frac:.4f}',
            f'{d.get("ring_norm", 0):.4f}',
            f'{d.get("ring_slot_mean", 0):.4f}',
            f'{d.get("alpha_0_mean", 0):.4f}',
            f'{d.get("alpha_0_min", 0):.4f}',
            f'{d.get("alpha_0_max", 0):.4f}',
            f'{d.get("alpha_1_mean", 0):.4f}',
            f'{d.get("alpha_1_min", 0):.4f}',
            f'{d.get("alpha_1_max", 0):.4f}',
            f'{d.get("input_norm_0", 0):.4f}',
            f'{d.get("input_norm_1", 0):.4f}',
            f'{d.get("ring_signal_norm_0", 0):.4f}',
            f'{d.get("ring_signal_norm_1", 0):.4f}',
            f'{d.get("blended_norm_0", 0):.4f}',
            f'{d.get("blended_norm_1", 0):.4f}',
            f'{d.get("hidden_norm_0", 0):.4f}',
            f'{d.get("hidden_norm_1", 0):.4f}',
            f'{d.get("hidden_final_norm_0", 0):.4f}',
            f'{d.get("hidden_final_norm_1", 0):.4f}',
            f'{d.get("ptr_pos_0", 0):.2f}',
            f'{d.get("ptr_pos_1", 0):.2f}',
            # ── BB telemetry ──
            f'{d.get("bb_beta_0", 0):.4f}',
            f'{d.get("bb_beta_1", 0):.4f}',
            f'{d.get("bb_ctx_raw_norm_0", 0):.4f}',
            f'{d.get("bb_ctx_raw_norm_1", 0):.4f}',
            f'{d.get("bb_ctx_scaled_norm_0", 0):.4f}',
            f'{d.get("bb_ctx_scaled_norm_1", 0):.4f}',
            f'{d.get("bb_attn_entropy_0", 0):.4f}',
            f'{d.get("bb_attn_entropy_1", 0):.4f}',
            f'{d.get("bb_query_norm_0", 0):.4f}',
            f'{d.get("bb_query_norm_1", 0):.4f}',
            f'{d.get("bb_key_norm", 0):.4f}',
            f'{d.get("bb_ctx_vs_input_0", 0):.4f}',
            f'{d.get("bb_ctx_vs_input_1", 0):.4f}',
            f'{d.get("bb_ctx_vs_ring_0", 0):.4f}',
            f'{d.get("bb_ctx_vs_ring_1", 0):.4f}',
        ])
        self._f.flush()

    def close(self):
        self._f.close()


# ── Boot Info ─────────────────────────────────────────────────
# One-shot diagnostics printed at startup. No state mutation — purely informational.
# Intended for the user to sanity-check before a long training run:
# "is this the right GPU? right data? right mode?"

def func_bootinfo_non(config, model, dataset, device):
    """Print boot diagnostics — environment, model, data, and config at a glance."""
    model_name = config.get('model_type', 'instnct').upper()
    print(f'VRAXION v4 — {model_name} Training Harness')
    print(f'{"=" * 52}')
    print(f'[BOOT] Python     {sys.version.split()[0]}')
    print(f'[BOOT] PyTorch    {torch.__version__}')
    print(f'[BOOT] Device     {device}')
    # GPU info only if CUDA is actually in play — str() wraps in case device is torch.device.
    # Derive GPU index from device string ("cuda:1" → 1, "cuda" → 0) so multi-GPU
    # setups report the ACTUAL device, not always GPU 0.
    if 'cuda' in str(device) and torch.cuda.is_available():
        gpu_idx = torch.device(device).index or 0
        print(f'[BOOT] GPU        {torch.cuda.get_device_name(gpu_idx)}')
        vram = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
        print(f'[BOOT] VRAM       {vram:.1f} GB')
    # exact trainable param count (requires_grad only — excludes fixed buffers like _R_eff)
    def _count_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_total = _count_trainable(model)
    print(f'[BOOT] Params     {n_total:,} trainable')
    # subsystem breakdown — exact counts, no formulas
    n_inp = _count_trainable(model.inp) if hasattr(model, 'inp') and isinstance(model.inp, torch.nn.Module) else 0
    n_out = _count_trainable(model.out) if hasattr(model, 'out') and isinstance(model.out, torch.nn.Module) else 0
    n_core = n_total - n_inp - n_out
    print(f'[BOOT] ParamsBy   input={n_inp:,} core={n_core:,} output={n_out:,}')
    if hasattr(model, 'hidden_dim'):
        print(f'[BOOT] Hidden     {model.hidden_dim}  Slot {model.slot_dim}')
    elif hasattr(model, 'd_model'):
        print(f'[BOOT] d_model    {model.d_model}')
    mode = 'embed (256 tokens, CE loss)' if config['embed_mode'] else 'binary (8-bit, MSE loss)'
    print(f'[BOOT] Mode       {mode}')
    print(f'[BOOT] Data       {len(dataset.file_pairs)} file(s), '
          f'{dataset.total_bytes / 1024**2:.1f} MB, '
          f'{dataset.n_samples:,} samples')
    print(f'[BOOT] Batch      {config["batch_size"]} x {config["seq_len"]} bytes')
    print(f'[BOOT] Steps      {config["steps"]:,}')
    print(f'[BOOT] LR         {config["lr"]}')
    _c19_names = {'c19_rho_input', 'c19_rho_hidden', 'c19_C_input', 'c19_C_hidden'}
    n_c19 = sum(p.numel() for n, p in model.named_parameters()
                if n in _c19_names or 'raw_rho' in n or 'raw_C' in n)
    print(f'[BOOT] C19 meta   {n_c19:,} params (constant LR, no decay)')
    warmup = config.get('warmup_steps', 0)
    if warmup > 0:
        sched = config.get('lr_schedule', 'plateau')
        print(f'[BOOT] Warmup     {warmup} steps (then {sched})')
    else:
        sched = config.get('lr_schedule', 'plateau')
        print(f'[BOOT] Schedule   {sched} (no warmup)')
    ckpt_chunks = getattr(model, 'checkpoint_chunks', 0)
    if ckpt_chunks > 0:
        print(f'[BOOT] GradCkpt   chunk_size={ckpt_chunks} (recompute to save VRAM)')
    if getattr(model, 'expert_weighting', False):
        print(f'[BOOT] XprtConf   gradient-based write weighting (1-frame delay)')
    enc = getattr(model, 'embed_encoding', 'learned')
    print(f'[BOOT] EmbedEnc   {enc} ({n_inp:,} params)')
    out_enc = getattr(model, 'output_encoding', 'learned')
    print(f'[BOOT] OutEnc     {out_enc} ({n_out:,} params)')
    grad_clip = config.get('max_grad_norm', 0)
    if grad_clip > 0:
        print(f'[BOOT] Grad clip  {grad_clip}')
    patience = config.get('patience', 0)
    if patience > 0:
        print(f'[BOOT] Patience   {patience} save intervals')
    print(f'[BOOT] Save every {config["save_every"]} steps')
    print(f'[BOOT] Log every  {config["log_every"]} steps')
    if config.get('sequential', False):
        print(f'[BOOT] DataMode   sequential (state persists across sequences)')
    if config.get('use_amp', False) and 'cuda' in str(device):
        print(f'[BOOT] AMP        fp16 mixed precision (GradScaler enabled)')
    print()


def _build_optimizer(model, model_type: str, lr: float):
    """Create optimizer matching model type and expected param groups."""
    if model_type == 'transformer':
        return torch.optim.Adam(model.parameters(), lr=lr)

    # INSTNCT: keep c19 meta params on a constant-LR group.
    _c19_meta_names = {'c19_rho_input', 'c19_rho_hidden', 'c19_C_input', 'c19_C_hidden'}
    c19_params = []
    normal_params = []
    for name, param in model.named_parameters():
        if name in _c19_meta_names or 'raw_rho' in name or 'raw_C' in name:
            c19_params.append(param)
        else:
            normal_params.append(param)
    return torch.optim.Adam([
        {'params': normal_params},
        {'params': c19_params, 'lr': lr},
    ], lr=lr)


# ── Training Loop ─────────────────────────────────────────────
# Orchestrates the full training lifecycle: device selection, data loading,
# model creation, optional checkpoint resume, and the step loop itself.
# Stability guards: NaN/Inf detection, gradient clipping, LR warmup + cosine
# decay, and optional early stopping (patience). No DDP — single GPU only.

def train(config):
    """Main training loop."""
    # ── Device ──
    # "auto" → prefer GPU, fall back to CPU. Explicit "cuda"/"cpu" bypasses.
    dev_str = config.get('device', 'auto')
    if dev_str == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = dev_str

    # ── CPU threads ──
    cpu_threads = config.get('cpu_threads', 0)
    if cpu_threads > 0 and device == 'cpu':
        torch.set_num_threads(cpu_threads)
        print(f'[BOOT] Threads    {cpu_threads} (torch CPU threads)')

    v4_root = Path(__file__).resolve().parent.parent
    deterministic_resume = bool(config.get('deterministic_resume', True))
    if deterministic_resume:
        config['hot_reload_enabled'] = False  # hard guard requested by spec

    # Optional seed for fresh runs (resume path restores exact RNG state).
    if config.get('seed') is not None and not config.get('resume'):
        seed = int(config['seed'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model_cfg_resolved = load_model_config(v4_root)

    # ── Resume payload (loaded early so config/data/model can be reconstructed from it) ──
    ckpt = None
    if config.get('resume'):
        resume_path = config['resume']
        if resume_path == 'latest':
            resume_path = str(Path(config['out_dir']) / 'ckpt_latest.pt')
        ckpt = func_loadckpt_dct(resume_path, device)

        if deterministic_resume:
            # Restore deterministic-critical runtime knobs from checkpoint.
            saved_cfg = ckpt['train_config_resolved']
            preserved = {'resume', 'device', 'out_dir', 'steps', 'log_every', 'save_every', 'heartbeat_every',
                         'lr_schedule', 'lr_patience', 'lr_factor', 'lr_min',
                         'seq_len', 'batch_size'}
            for k, v in saved_cfg.items():
                if k not in preserved:
                    config[k] = v

    # ── Data ──
    files = func_discover_dat(config['data_dir'])
    dataset = ByteDataset(
        files,
        config['seq_len'],
        config['embed_mode'],
        seed=config.get('dataset_seed'),
    )
    file_manifest = _build_file_manifest(files)

    # n_samples = total_bytes - seq_len. For a full batch we need
    # n_samples >= batch_size, i.e. total_bytes >= seq_len + batch_size.
    if dataset.n_samples < config['batch_size']:
        raise ValueError(
            f"Not enough data: {dataset.n_samples} samples < batch_size {config['batch_size']}. "
            f"Need at least {config['seq_len'] + config['batch_size']} bytes of training data."
        )

    # Guard against zero-interval configs — these cause ZeroDivisionError on modulo.
    for k in ('heartbeat_every', 'log_every', 'save_every'):
        if config.get(k, 1) < 1:
            raise ValueError(f"Config '{k}' must be >= 1, got {config[k]}")

    # ── Model ──
    # Always build from YAML config (not checkpoint) so architecture changes
    # (e.g. slot_dim 64→256) take effect on resume. Old weights load via strict=False.
    model_record = build_model_spec(
        model_type=config.get('model_type', 'instnct'),
        embed_mode=config['embed_mode'],
        model_config=model_cfg_resolved,
        training_config=config,
    )
    model = build_model_from_spec(model_record, device=device)
    # torch.compile: +22% step/sec, -55% VRAM after ~73s warmup. Breakeven ~1460 steps.
    if config.get('compile', False) and device.startswith('cuda') and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("[compile] torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"[compile] torch.compile failed, running eager: {e}")
    opt = _build_optimizer(model, model_record['type'], config['lr'])

    # ── Output dir ──
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ──
    # start_step=0 means fresh run; overridden below if resuming.
    # best_loss=inf guarantees the first step always becomes the new best.
    start_step = 0
    best_loss = float('inf')
    run_id = str(uuid.uuid4())
    eval_seed = int(config.get('eval_seed', 1337))
    _resumed_state = None  # set by resume block if checkpoint has sequential state

    if ckpt is not None:
        resume_path = config['resume']
        # Filter out shape-mismatched params (e.g. slot_dim 64→256 resizes write/read_proj)
        ckpt_sd = ckpt['model']['state_dict']
        model_sd = model.state_dict()
        _shape_mismatch = []
        filtered_sd = {}
        for k, v in ckpt_sd.items():
            if k in model_sd and v.shape != model_sd[k].shape:
                _shape_mismatch.append(f'{k}: ckpt={list(v.shape)} model={list(model_sd[k].shape)}')
            else:
                filtered_sd[k] = v
        if _shape_mismatch:
            print(f'  [RESUME] Shape-changed params (re-init): {_shape_mismatch}')
        _missing, _unexpected = model.load_state_dict(filtered_sd, strict=False)
        _needs_reset = bool(_missing) or bool(_shape_mismatch)
        if _missing:
            print(f'  [RESUME] New params (init from scratch): {_missing}')
        if _unexpected:
            print(f'  [RESUME] Dropped params (no longer in model): {_unexpected}')
        if _needs_reset:
            print(f'  [RESUME] Optimizer reset (architecture changed — Adam state incompatible)')
        else:
            opt.load_state_dict(ckpt['optimizer']['state_dict'])
        start_step = int(ckpt['step'])
        best_loss = float(ckpt['best_loss'])
        run_id = str(ckpt['run_id'])
        eval_seed = int(ckpt.get('rng_state', {}).get('eval_seed', eval_seed))

        # Manifest drift detection.
        expected_manifest = ckpt.get('data_state', {}).get('file_manifest', [])
        drift_msgs = _compare_manifest(expected_manifest, file_manifest) if expected_manifest else []
        if drift_msgs:
            msg = '; '.join(drift_msgs[:3])
            if config.get('manifest_strict', False):
                raise RuntimeError(f'Data manifest mismatch on resume: {msg}')
            print(f'[WARN] Data manifest mismatch: {msg}')

        # Restore sequential/TBPTT state
        # Validate shapes: batch/slot_dim/hidden_dim may have changed via config.
        seq_state = ckpt.get('sequence_state', {})
        if seq_state and not _needs_reset:
            _resumed_state = {}
            for k in ('ring_state', 'ptr_state', 'hidden_state', 'bb_buf', 'bb_keys'):
                if k in seq_state:
                    alias = k.replace('_state', '') if k.endswith('_state') else k
                    _resumed_state[alias] = seq_state[k].to(device)
            for k in ('bb_write_ptr', 'bb_steps'):
                if k in seq_state:
                    _resumed_state[k] = seq_state[k]
        elif seq_state and _needs_reset:
            print(f'  [RESUME] Sequential state discarded (architecture changed — shapes incompatible)')
            _resumed_state = None

        saved_seq_offsets = ckpt.get('data_state', {}).get('seq_offsets')
        if saved_seq_offsets is not None:
            dataset._seq_offsets = np.array(saved_seq_offsets, dtype=np.int64)

        if deterministic_resume:
            _restore_rng_state(ckpt['rng_state'], dataset)

        print(f'[RESUME] step={start_step}  best_loss={best_loss:.6f}  from {resume_path}')

    # Use checkpoint-resolved model config on resume so saves remain self-consistent.
    if ckpt is not None and 'model_config_resolved' in ckpt:
        model_cfg_resolved = ckpt['model_config_resolved']

    env_meta = _capture_env_metadata(v4_root)
    # Snapshot runtime config (decision-complete run context).
    train_config_resolved = dict(config)
    train_config_resolved['device'] = device
    train_config_resolved['deterministic_resume'] = deterministic_resume
    train_config_resolved['hot_reload_enabled'] = bool(config.get('hot_reload_enabled', False))

    # ── Plateau LR tracker ──
    if config.get('lr_schedule', 'plateau') == 'plateau':
        plateau = _PlateauTracker(
            base_lr=config['lr'],
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 150),
            min_lr=config.get('lr_min', 1e-5),
        )
        if ckpt is not None and 'lr_plateau_state' in ckpt:
            plateau.load_state_dict(ckpt['lr_plateau_state'])
            print(f'[RESUME] plateau LR state restored: mult={plateau.multiplier:.4f} '
                  f'stale={plateau.stale_steps}/{plateau.patience}')
    else:
        plateau = None

    # ── AMP (mixed precision) ──
    use_amp = bool(config.get('use_amp', False)) and 'cuda' in device
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    if use_amp and ckpt is not None and 'amp_scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['amp_scaler_state'])
        print(f'[RESUME] AMP GradScaler state restored')

    # ── Boot ──
    func_bootinfo_non(config, model, dataset, device)

    if config['steps'] <= start_step:
        print(f'[DONE] Already at step {start_step} >= target {config["steps"]}. Nothing to do.')
        return

    # ── Logger ──
    csv_log = CSVLogger(out_dir / 'train_log.csv')
    # Select loss AND accuracy functions once, not per-step.
    # loss_fn returns (raw_loss, masked_loss). acc_fn returns (raw_acc, masked_acc).
    loss_fn = func_maskloss_ce if config['embed_mode'] else func_maskloss_mse
    acc_fn  = func_accuracy_emb if config['embed_mode'] else func_accuracy_bin

    # ── Loop ──
    t0 = time.perf_counter()
    model.train()  # enable dropout/batchnorm train behavior (INSTNCT has neither, but convention)

    # NaN/Inf guard state — consecutive NaN count; 10 in a row = divergence, abort
    nan_count = 0
    s = start_step  # last completed step (1-indexed); updated inside loop
    lv = float('inf')  # loss value from previous step; init to inf for plateau tracker

    # Early stopping state — tracked at save intervals, not per-step (too noisy)
    prev_best = best_loss
    stale_saves = 0

    # Reset CUDA peak stats so the final report shows THIS run's peak, not
    # accumulated from previous PyTorch work in the same process.
    if 'cuda' in device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ── Sequential / stateful mode ──
    sequential = config.get('sequential', False)
    # Use resumed state if available (from checkpoint), otherwise None (fresh).
    state = _resumed_state

    def _make_data_state(seq_offsets):
        payload = {
            'data_dir': config['data_dir'],
            'seq_len': int(config['seq_len']),
            'batch_size': int(config['batch_size']),
            'embed_mode': bool(config['embed_mode']),
            'sequential': bool(sequential),
            'file_manifest': file_manifest,
        }
        if seq_offsets is not None:
            payload['seq_offsets'] = seq_offsets.tolist()
        return payload

    def _make_sequence_state(current_state):
        if not sequential or current_state is None:
            return {}
        seq_payload = {}
        if 'ring' in current_state:
            seq_payload['ring_state'] = current_state['ring']
        if 'ptr' in current_state:
            seq_payload['ptr_state'] = current_state['ptr']
        if 'hidden' in current_state:
            seq_payload['hidden_state'] = current_state['hidden']
        if 'bb_buf' in current_state:
            seq_payload['bb_buf'] = current_state['bb_buf']
            seq_payload['bb_keys'] = current_state['bb_keys']
            seq_payload['bb_write_ptr'] = current_state['bb_write_ptr']
            seq_payload['bb_steps'] = current_state['bb_steps']
        return seq_payload

    for step in range(start_step, config['steps']):
        # 1-indexed for display — used by heartbeat, log, save, and NaN guard messages
        s = step + 1

        # ── LR Schedule ──
        # Warmup ramps linearly to base_lr. After warmup, plateau tracker
        # manages reductions; if schedule=constant, LR stays at base_lr.
        if step < config['warmup_steps']:
            lr = _compute_lr(config['lr'], step, config['warmup_steps'])
        elif plateau is not None:
            lr = plateau.step(lv)
        else:
            lr = config['lr']
        opt.param_groups[0]['lr'] = lr
        # group 1 (c19 rho/C): constant LR — no decay, always adaptive

        # ── Forward ──
        if sequential:
            xb, yb, mask = dataset.sample_batch_sequential(config['batch_size'], device)
        else:
            xb, yb, mask = dataset.sample_batch(config['batch_size'], device)

        _ring_pre = state['ring'].norm().item() if state is not None and 'ring' in state else -1

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred, state = model(xb, state=state if sequential else None)
            # loss_fn returns two values: raw_loss (all positions) and masked_loss
            # (supervised positions only). We optimize masked_loss but log both.
            raw_loss, masked_loss = loss_fn(pred, yb, mask)

        _ring_post = state['ring'].norm().item() if state is not None and 'ring' in state else -1
        # Temporary ring delta probe — remove once ring freeze is resolved
        if s <= start_step + 5:
            print(f'  [RING-PROBE] step {s}: pre={_ring_pre:.4f} post={_ring_post:.4f} delta={_ring_post - _ring_pre:.6f}')

        # ── NaN/Inf guard ──
        # Silent NaN/Inf in loss would corrupt gradients, optimizer state, checkpoint.
        # Detect and skip the entire step. One bad batch is recoverable;
        # corrupted optimizer momentum buffers are not.
        if not torch.isfinite(masked_loss):
            nan_count += 1
            print(f'  [NAN] step {s} -- loss={masked_loss.item()}, skipping ({nan_count} consecutive)')
            if nan_count >= 10:
                raise RuntimeError(
                    f'NaN/Inf loss in {nan_count} consecutive steps -- training is diverging. '
                    f'Try lower lr or check data.'
                )
            continue
        nan_count = 0  # reset on healthy step

        # ── Backward ──
        opt.zero_grad()
        if scaler is not None:
            scaler.scale(masked_loss).backward()
            if hasattr(model, 'update_expert_conf'):
                model.update_expert_conf()
            if config['max_grad_norm'] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(opt)
            scaler.update()
        else:
            masked_loss.backward()
            if hasattr(model, 'update_expert_conf'):
                model.update_expert_conf()
            if config['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            opt.step()

        # .item() detaches from the graph — safe to hold across iterations.
        lv = masked_loss.item()
        if lv < best_loss:
            best_loss = lv

        # ── Heartbeat ──
        if s % config['heartbeat_every'] == 0:
            elapsed = time.perf_counter() - t0
            sps = s - start_step
            sec_per_step = elapsed / sps if sps > 0 else 0
            xc = getattr(model, '_expert_conf', None)
            xc_str = f'  conf=[{" ".join(f"{c:.4f}" for c in xc.tolist())}]' if xc is not None else ''
            # diagnostics from model forward pass
            diag = getattr(model, '_diag', {})
            alpha_str = ''
            if 'alpha_0_mean' in diag:
                alphas = ' '.join(f'{diag.get(f"alpha_{i}_mean", 0):.3f}' for i in range(model.N))
                alpha_str = f'  alpha=[{alphas}]'
            ring_str = f'  ring={diag.get("ring_norm", 0):.2f}' if 'ring_norm' in diag else ''
            bb_str = ''
            if 'bb_beta_0' in diag:
                betas = ' '.join(f'{diag.get(f"bb_beta_{i}", 0):.3f}' for i in range(model.N))
                ent = f'{diag.get("bb_attn_entropy_0", 0):.2f}'
                ratio = f'{diag.get("bb_ctx_vs_ring_0", 0):.2f}'
                bb_str = f'  bb=[{betas}] ent={ent} ctx/ring={ratio}'
            plat_str = ''
            if plateau is not None:
                plat_str = f'  stale={plateau.stale_steps}/{plateau.patience} mult={plateau.multiplier:.2f}'
            bpc = lv * 1.4427  # nats -> bits (log2(e))
            print(f'  [{s}/{config["steps"]}] loss={lv:.6f}  bpc={bpc:.3f}  best={best_loss:.6f}'
                  f'  {elapsed:.1f}s  {sec_per_step:.1f}s/step  lr={lr:.2e}'
                  f'{plat_str}{xc_str}{alpha_str}{ring_str}{bb_str}')

        # ── CSV Log ──
        # Log at regular intervals AND on the very first step (to capture initial loss).
        # acc_fn reuses the pred/yb/mask tensors already in memory — zero extra forward pass.
        if s % config['log_every'] == 0 or s == start_step + 1:
            raw_acc, masked_acc = acc_fn(pred, yb, mask)
            elapsed = time.perf_counter() - t0
            mf = mask.mean().item()
            csv_log.log(s, raw_loss.item(), lv, raw_acc, masked_acc,
                        lr, elapsed, s * config['batch_size'], mf,
                        diag=getattr(model, '_diag', {}))

        # ── Checkpoint ──
        if s % config['save_every'] == 0:
            ckpt_path = out_dir / f'ckpt_step_{s:06d}.pt'
            seq_off = getattr(dataset, '_seq_offsets', None) if sequential else None
            if scaler is not None:
                train_config_resolved['_amp_scaler_state'] = scaler.state_dict()
            func_saveckpt_non(
                ckpt_path,
                model=model,
                optimizer=opt,
                step=s,
                best_loss=best_loss,
                run_id=run_id,
                model_record=model_record,
                train_config_resolved=train_config_resolved,
                model_config_resolved=model_cfg_resolved,
                data_state=_make_data_state(seq_off),
                sequence_state=_make_sequence_state(state),
                rng_state=_capture_rng_state(dataset, eval_seed),
                env_meta=env_meta,
                lr_plateau_state=plateau.state_dict() if plateau else None,
            )
            shutil.copy2(ckpt_path, out_dir / 'ckpt_latest.pt')
            print(f'  [SAVE] {ckpt_path}')

            # ── Hot-reload config ──
            # Re-read YAML at every save point. Only apply safe params that
            # don't change model architecture or dataset structure.
            try:
                if not config.get('hot_reload_enabled', False):
                    raise RuntimeError('disabled')
                _hot = _load_yaml('training')
                _safe_keys = {
                    'batch_size': int, 'lr': float, 'log_every': int,
                    'save_every': int, 'heartbeat_every': int,
                    'max_grad_norm': float, 'patience': int,
                }
                _changed = []
                for k, cast in _safe_keys.items():
                    new_val = cast(_hot[k]) if k in _hot else None
                    if new_val is not None and new_val != config.get(k):
                        _changed.append(f'{k}: {config[k]} -> {new_val}')
                        config[k] = new_val
                        train_config_resolved[k] = new_val
                if _changed:
                    print(f'  [HOT-RELOAD] {", ".join(_changed)}')
            except Exception as e:
                if str(e) != 'disabled':
                    print(f'  [HOT-RELOAD] failed: {e}')

            # ── Early stopping ──
            # Measured at save intervals (not per-step) to smooth out noise.
            # If best_loss hasn't improved for `patience` consecutive save checks,
            # training has plateaued — stop and save the user's GPU time.
            if config['patience'] > 0:
                if best_loss < prev_best:
                    prev_best = best_loss
                    stale_saves = 0
                else:
                    stale_saves += 1
                    if stale_saves >= config['patience']:
                        print(f'  [EARLY STOP] No improvement for {config["patience"]} '
                              f'save intervals (best={best_loss:.6f})')
                        break

    # ── Final ──
    # s = last completed step (1-indexed). If early stopping triggered, the
    # checkpoint was already saved inside the save block (early stop is always
    # at a save point). If normal completion, save only if the last step wasn't
    # already a save point — avoids rewriting an identical checkpoint.
    if s % config['save_every'] != 0:
        final_path = out_dir / f'ckpt_step_{s:06d}.pt'
        seq_off = getattr(dataset, '_seq_offsets', None) if sequential else None
        if scaler is not None:
            train_config_resolved['_amp_scaler_state'] = scaler.state_dict()
        func_saveckpt_non(
            final_path,
            model=model,
            optimizer=opt,
            step=s,
            best_loss=best_loss,
            run_id=run_id,
            model_record=model_record,
            train_config_resolved=train_config_resolved,
            model_config_resolved=model_cfg_resolved,
            data_state=_make_data_state(seq_off),
            sequence_state=_make_sequence_state(state),
            rng_state=_capture_rng_state(dataset, eval_seed),
            env_meta=env_meta,
            lr_plateau_state=plateau.state_dict() if plateau else None,
        )
        shutil.copy2(final_path, out_dir / 'ckpt_latest.pt')
    csv_log.close()

    # ── Summary ──
    # Mirror the boot info block: give the user a quick glance at what happened.
    elapsed = time.perf_counter() - t0
    total_steps = s - start_step
    rate = total_steps / elapsed if elapsed > 0 else 0

    print()
    sec_per_step = elapsed / total_steps if total_steps > 0 else 0
    print(f'[DONE] {total_steps} steps in {elapsed:.1f}s ({sec_per_step:.1f}s/step)')
    print(f'       best_loss={best_loss:.6f}')
    if 'cuda' in device:
        # max_memory_allocated tracks the high-water mark since reset_peak_memory_stats().
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f'       peak_vram={peak:.0f} MB')
    print(f'       checkpoint: {out_dir / "ckpt_latest.pt"}')
    print(f'       log: {out_dir / "train_log.csv"}')


# ── CLI ───────────────────────────────────────────────────────
# Config merge: CLI args > YAML file > hardcoded defaults.
# Most keys go through all three tiers. Exceptions:
#   - `resume` is CLI-only (no YAML default — intentional).
#   - `heartbeat_every` is YAML-only (no CLI flag — too niche).
# All args use default=None so we can distinguish "user didn't pass this"
# from "user explicitly set this". The merge logic below relies on that.

def func_parsecli_dct() -> dict:
    """Parse CLI args, merge with YAML config. CLI overrides YAML."""
    # Load YAML defaults — missing file or missing section is fine, just use {}.
    try:
        yaml_cfg = _load_yaml('training')
    except (FileNotFoundError, KeyError):
        yaml_cfg = {}

    parser = argparse.ArgumentParser(
        description='INSTNCT v4 Training Harness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python train.py --data training_data --steps 10000
  python train.py --data training_data/echo256.traindat --embed --steps 500
  python train.py --resume latest --steps 20000""",
    )

    parser.add_argument('--data', default=None,
                        help='path to training data dir or .traindat file')
    # No `choices` constraint — allows cuda:0, cuda:1, etc. for multi-GPU.
    # Invalid strings (e.g. "banana") crash at model.to(device) with a clear error.
    parser.add_argument('--device', default=None,
                        help='device (default: auto). e.g. cuda, cpu, cuda:1')
    parser.add_argument('--steps', type=int, default=None,
                        help='total training steps')
    parser.add_argument('--batch', type=int, default=None,
                        help='batch size')
    parser.add_argument('--seq', type=int, default=None,
                        help='sequence length in bytes')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--embed', action='store_true', default=None,
                        help='use embed mode (256 tokens, CrossEntropy)')
    parser.add_argument('--resume', default=None,
                        help='checkpoint path to resume from ("latest" = auto)')
    parser.add_argument('--log-every', type=int, default=None,
                        help='CSV logging interval (steps)')
    parser.add_argument('--save-every', type=int, default=None,
                        help='checkpoint save interval (steps)')
    parser.add_argument('--out', default=None,
                        help='output directory for checkpoints + logs')
    parser.add_argument('--warmup', type=int, default=None,
                        help='LR warmup steps (default: from config or 0)')
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='gradient clipping max norm (0=disabled, default: from config or 1.0)')
    parser.add_argument('--patience', type=int, default=None,
                        help='early stopping: save intervals without improvement (0=disabled)')
    parser.add_argument('--model', default=None, choices=['instnct', 'transformer'],
                        help='model architecture (default: instnct)')
    parser.add_argument('--seed', type=int, default=None,
                        help='global random seed for fresh runs (default: from config)')
    parser.add_argument('--compile', action='store_true', default=None,
                        help='enable torch.compile (reduce-overhead). +22%% speed after ~73s warmup')

    args = parser.parse_args()

    # Merge: CLI > YAML > hardcoded defaults.
    # All string args use `is not None` (not `or`) to avoid falsy-short-circuit
    # on empty strings. `--embed` is store_true so it's True or None — no way
    # to force False from CLI if YAML sets embed_mode: true (rare edge case).
    config = {
        'data_dir':        args.data if args.data is not None else yaml_cfg.get('data_dir', 'training_data'),
        'device':          args.device if args.device is not None else yaml_cfg.get('device', 'auto'),
        'steps':           args.steps if args.steps is not None else yaml_cfg.get('steps', 10000),
        'batch_size':      args.batch if args.batch is not None else yaml_cfg.get('batch_size', 32),
        'seq_len':         args.seq if args.seq is not None else yaml_cfg.get('seq_len', 128),
        'lr':              args.lr if args.lr is not None else yaml_cfg.get('lr', 1e-3),
        'embed_mode':      args.embed if args.embed is not None else yaml_cfg.get('embed_mode', False),
        'resume':          args.resume,
        'log_every':       args.log_every if args.log_every is not None else yaml_cfg.get('log_every', 100),
        'save_every':      args.save_every if args.save_every is not None else yaml_cfg.get('save_every', 1000),
        'heartbeat_every': yaml_cfg.get('heartbeat_every', 10),
        'out_dir':         args.out if args.out is not None else yaml_cfg.get('out_dir', 'training_output'),
        'warmup_steps':    args.warmup if args.warmup is not None else yaml_cfg.get('warmup_steps', 0),
        'max_grad_norm':   args.grad_clip if args.grad_clip is not None else yaml_cfg.get('max_grad_norm', 1.0),
        'patience':        args.patience if args.patience is not None else yaml_cfg.get('patience', 0),
        'cpu_threads':     yaml_cfg.get('cpu_threads', 0),
        'model_type':      args.model if args.model is not None else yaml_cfg.get('model_type', 'instnct'),
        'seed':            args.seed if args.seed is not None else yaml_cfg.get('seed', 1337),
        'dataset_seed':    yaml_cfg.get('dataset_seed', args.seed if args.seed is not None else yaml_cfg.get('seed', 1337)),
        'deterministic_resume': bool(yaml_cfg.get('deterministic_resume', True)),
        'hot_reload_enabled': bool(yaml_cfg.get('hot_reload_enabled', False)),
        'eval_seed':       int(yaml_cfg.get('eval_seed', 1337)),
        'manifest_strict': bool(yaml_cfg.get('manifest_strict', False)),
        # Adaptive LR (plateau tracker)
        'lr_schedule':     yaml_cfg.get('lr_schedule', 'plateau'),
        'lr_patience':     int(yaml_cfg.get('lr_patience', 150)),
        'lr_factor':       float(yaml_cfg.get('lr_factor', 0.5)),
        'lr_min':          float(yaml_cfg.get('lr_min', 1e-5)),
        # Sequential / stateful training (TBPTT): ring/hidden/ptr persist across batches
        'sequential':      bool(yaml_cfg.get('sequential', False)),
        # Mixed precision (AMP): fp16 forward/backward, fp32 optimizer
        'use_amp':         bool(yaml_cfg.get('use_amp', False)),
        'compile':         args.compile if args.compile is not None else bool(yaml_cfg.get('compile', False)),
    }

    # Resolve relative paths from v4/ root so `python training/train.py --data foo`
    # works the same as `cd v4 && python training/train.py --data foo`.
    v4_root = Path(__file__).parent.parent
    for key in ('data_dir', 'out_dir'):
        p = Path(config[key])
        if not p.is_absolute():
            config[key] = str(v4_root / p)

    return config


# ── Entry Point ───────────────────────────────────────────────
# Top-level exception handling: KeyboardInterrupt is NOT an error (exit 0),
# everything else prints the full traceback and exits with code 1 so that
# shell scripts and CI can detect failure.

if __name__ == '__main__':
    cfg = func_parsecli_dct()

    try:
        train(cfg)
    except KeyboardInterrupt:
        print('\n[INTERRUPTED] Training stopped by user.')
        sys.exit(0)
    except Exception as e:
        print(f'[FATAL] {type(e).__name__}: {e}')
        traceback.print_exc()
        sys.exit(1)
