"""
Universal .traindat loader for Diamond Code swarm training.

.traindat files are raw bytes -- no header, no encoding.
Gray code scalar encoding: each byte -> 1 scalar in (0, 1).
num_bits channels per position = num_bits bytes of data per position.

Usage:
    loader = TraindatLoader("data/traindat/")
    x, y = loader.sample_batch(batch_size=1, seq_len=62, num_bits=8, seed=42)
"""

import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


# --- Gray code lookup tables (256-entry, precomputed) ---
# GRAY_ENCODE[position] = byte value at that position in Gray code order
GRAY_ENCODE = [i ^ (i >> 1) for i in range(256)]
# GRAY_POSITION[byte_value] = position in Gray code order (inverse of GRAY_ENCODE)
GRAY_POSITION = [0] * 256
for _i in range(256):
    GRAY_POSITION[GRAY_ENCODE[_i]] = _i
# GRAY_SCALAR[byte_value] = scalar in (0.00195, 0.998) via (position + 0.5) / 256
GRAY_SCALAR = [(_p + 0.5) / 256.0 for _p in GRAY_POSITION]


def byte_to_gray_scalar(byte_val: int) -> float:
    """Convert a byte value (0-255) to its Gray-coded scalar in (0, 1)."""
    return float(GRAY_SCALAR[byte_val])


def gray_scalar_to_byte(scalar: float) -> int:
    """Convert a Gray-coded scalar back to the original byte value."""
    pos = max(0, min(255, round(scalar * 256.0 - 0.5)))
    return GRAY_ENCODE[pos]


def generate_batch_from_bytes(
    corpus: bytes,
    n_samples: int,
    seq_len: int = 16,
    num_bits: int = 64,
    seed=None,
    role_map: Optional[np.ndarray] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate next-chunk prediction batch from raw bytes using Gray code scalar encoding.

    Each position holds num_bits bytes, each Gray-coded to a scalar in (0, 1).
    Input = chunk[0:seq_len], Target = chunk[1:seq_len+1] (shifted by num_bits bytes).

    Args:
        role_map: Optional numpy array from golden_disc.build_role_map().
            If provided, returns a 3-tuple with mask tensor.

    Returns:
        Without role_map: (x, y)
        With role_map: (x, y, mask) where mask is [n_samples, seq_len] uint8
            mask values: 0=separator, 1=question, 2=answer, 3=type, 4=freetext
            mask[i,t] = role of the TARGET chunk at position t.
    """
    if seed is not None:
        random.seed(seed)

    bytes_per_pos = num_bits  # each channel = 1 byte (was num_bits // 8)
    chunk_len = (seq_len + 1) * bytes_per_pos
    max_start = len(corpus) - chunk_len - bytes_per_pos

    if max_start < 0:
        raise ValueError(
            f"Corpus too small ({len(corpus)} bytes) for "
            f"seq_len={seq_len}, num_bits={num_bits} (need >= {chunk_len + bytes_per_pos})"
        )

    x = torch.zeros(n_samples, seq_len, num_bits)
    y = torch.zeros(n_samples, seq_len, num_bits)
    if role_map is not None:
        mask = torch.zeros(n_samples, seq_len, dtype=torch.uint8)

    for i in range(n_samples):
        start = random.randint(0, max_start // bytes_per_pos) * bytes_per_pos
        chunk = corpus[start:start + chunk_len + bytes_per_pos]

        for t in range(seq_len):
            offset = t * bytes_per_pos
            for b in range(num_bits):
                x[i, t, b] = GRAY_SCALAR[chunk[offset + b]]

            target_offset = offset + bytes_per_pos
            for b in range(num_bits):
                y[i, t, b] = GRAY_SCALAR[chunk[target_offset + b]]

            # Mask: role of the target chunk (what model is predicting)
            if role_map is not None:
                target_byte_pos = start + target_offset
                if target_byte_pos < len(role_map):
                    mask[i, t] = role_map[target_byte_pos]

    if role_map is not None:
        return x, y, mask
    return x, y


class TraindatLoader:
    """
    Loads .traindat files from a directory with weighted sampling.

    Files are lazily loaded into memory and cached.
    Weights can be updated mid-run via update_weights() (live controls hook).
    """

    def __init__(self, data_dir: str, weights: Optional[Dict[str, float]] = None):
        self.data_dir = Path(data_dir)
        self._corpora: Dict[str, bytes] = {}
        self._role_maps: Dict[str, np.ndarray] = {}
        self._file_list: List[str] = []
        self._weights: Dict[str, float] = weights or {}
        self._sequential = False
        self._seq_steps = 100
        self._seq_index = 0
        self._seq_counter = 0
        self._seq_active: List[str] = []
        self._discover_files()

    def _discover_files(self):
        """Scan directory for .traindat files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self._file_list = sorted([
            f.name for f in self.data_dir.iterdir()
            if f.suffix == '.traindat' and f.is_file()
        ])

        if not self._file_list:
            raise FileNotFoundError(f"No .traindat files in {self.data_dir}")

        for fname in self._file_list:
            if fname not in self._weights:
                self._weights[fname] = 1.0

    def _load_corpus(self, filename: str) -> bytes:
        """Lazily load and cache a .traindat file."""
        if filename not in self._corpora:
            path = self.data_dir / filename
            with open(path, 'rb') as f:
                self._corpora[filename] = f.read()
        return self._corpora[filename]

    def _get_role_map(self, filename: str) -> np.ndarray:
        """Lazily build and cache role map for a .traindat file."""
        if filename not in self._role_maps:
            from golden_disc import build_role_map
            corpus = self._load_corpus(filename)
            self._role_maps[filename] = build_role_map(corpus)
        return self._role_maps[filename]

    def update_weights(self, new_weights: Dict[str, float]):
        """Update sampling weights (from live controls). Re-scans directory for new files."""
        self._discover_files()
        self._weights.update(new_weights)
        # Clean stale cached corpora for deleted files
        stale = [k for k in self._corpora if k not in self._file_list]
        for k in stale:
            del self._corpora[k]
        # Rebuild sequential active list when weights change
        if self._sequential:
            self._rebuild_seq_active()

    def set_sequential(self, enabled: bool, steps_per_dataset: int = 100):
        """Enable/disable sequential dataset cycling.

        When enabled, pick_file() cycles through active datasets (weight > 0)
        in sorted order, spending steps_per_dataset steps on each before
        advancing. When the last dataset is reached, wraps back to first.
        """
        was_enabled = self._sequential
        self._sequential = enabled
        self._seq_steps = max(1, steps_per_dataset)
        if enabled and not was_enabled:
            self._seq_index = 0
            self._seq_counter = 0
            self._rebuild_seq_active()

    def _rebuild_seq_active(self):
        """Rebuild the sorted list of active files for sequential mode."""
        active = sorted([f for f in self._file_list if self._weights.get(f, 0) > 0])
        if active != self._seq_active:
            self._seq_active = active
            if self._seq_index >= len(self._seq_active):
                self._seq_index = 0
            if self._seq_active:
                print(f"  [SEQ] datasets: {' -> '.join(f.replace('.traindat','') for f in self._seq_active)} "
                      f"({self._seq_steps} steps each, cycling)")

    @property
    def current_dataset(self) -> Optional[str]:
        """Return the current dataset name in sequential mode, or None."""
        if self._sequential and self._seq_active:
            return self._seq_active[self._seq_index]
        return None

    @property
    def files(self) -> List[str]:
        return list(self._file_list)

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def pick_file(self) -> str:
        """Select next file. Sequential mode cycles through active datasets;
        random mode uses weighted sampling. Falls back to uniform if all weights are zero."""
        if self._sequential and self._seq_active:
            fname = self._seq_active[self._seq_index]
            self._seq_counter += 1
            if self._seq_counter >= self._seq_steps:
                self._seq_counter = 0
                old_idx = self._seq_index
                self._seq_index = (self._seq_index + 1) % len(self._seq_active)
                old_name = self._seq_active[old_idx].replace('.traindat', '')
                new_name = self._seq_active[self._seq_index].replace('.traindat', '')
                print(f"  [SEQ] switching: {old_name} -> {new_name} "
                      f"({self._seq_index + 1}/{len(self._seq_active)})")
            return fname
        names = self._file_list
        w = [self._weights.get(n, 1.0) for n in names]
        if sum(w) <= 0:
            return random.choice(names)
        return random.choices(names, weights=w, k=1)[0]

    def sample_batch(
        self,
        n_samples: int,
        seq_len: int,
        num_bits: int,
        seed: int = None,
        return_mask: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample a batch from a weighted-random .traindat file.

        Args:
            return_mask: If True, returns (x, y, mask) with Golden Disc role mask.
                mask is [n_samples, seq_len] uint8 with role of each target byte.

        Returns (x, y) or (x, y, mask) if return_mask=True.
        """
        filename = self.pick_file()
        corpus = self._load_corpus(filename)
        rm = self._get_role_map(filename) if return_mask else None
        return generate_batch_from_bytes(corpus, n_samples, seq_len, num_bits, seed, role_map=rm)
