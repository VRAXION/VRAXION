"""
Universal .traindat loader for Diamond Code swarm training.

.traindat files are raw bytes -- no header, no encoding.
The model sees byte sequences and predicts next bytes.

Usage:
    loader = TraindatLoader("data/traindat/")
    x, y = loader.sample_batch(batch_size=1, seq_len=128, num_bits=256, seed=42)
"""

import random
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def generate_batch_from_bytes(
    corpus: bytes,
    n_samples: int,
    seq_len: int = 16,
    num_bits: int = 64,
    seed=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate next-byte prediction batch from raw bytes.

    Each byte is expanded to 8 bits (LSB first). num_bits/8 bytes per position.
    Input = bytes[0:seq_len], Target = bytes[1:seq_len+1] (shifted by bytes_per_pos).

    Returns:
        x: [n_samples, seq_len, num_bits] float32
        y: [n_samples, seq_len, num_bits] float32
    """
    if seed is not None:
        random.seed(seed)

    bytes_per_pos = num_bits // 8
    chunk_len = (seq_len + 1) * bytes_per_pos
    max_start = len(corpus) - chunk_len - bytes_per_pos

    if max_start < 0:
        raise ValueError(
            f"Corpus too small ({len(corpus)} bytes) for "
            f"seq_len={seq_len}, num_bits={num_bits} (need >= {chunk_len + 2})"
        )

    x = torch.zeros(n_samples, seq_len, num_bits)
    y = torch.zeros(n_samples, seq_len, num_bits)

    for i in range(n_samples):
        start = random.randint(0, max_start)
        chunk = corpus[start:start + chunk_len + bytes_per_pos]

        for t in range(seq_len):
            offset = t * bytes_per_pos
            for b in range(bytes_per_pos):
                byte_val = chunk[offset + b]
                for bit in range(8):
                    x[i, t, b * 8 + bit] = float((byte_val >> bit) & 1)

            target_offset = offset + bytes_per_pos
            for b in range(bytes_per_pos):
                byte_val = chunk[target_offset + b]
                for bit in range(8):
                    y[i, t, b * 8 + bit] = float((byte_val >> bit) & 1)

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
        self._file_list: List[str] = []
        self._weights: Dict[str, float] = weights or {}
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

    def update_weights(self, new_weights: Dict[str, float]):
        """Update sampling weights (from live controls). Re-scans directory for new files."""
        self._discover_files()
        self._weights.update(new_weights)
        # Clean stale cached corpora for deleted files
        stale = [k for k in self._corpora if k not in self._file_list]
        for k in stale:
            del self._corpora[k]

    @property
    def files(self) -> List[str]:
        return list(self._file_list)

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def pick_file(self) -> str:
        """Weighted random file selection. Falls back to uniform if all weights are zero."""
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from a weighted-random .traindat file.

        Returns (x, y) both [n_samples, seq_len, num_bits].
        """
        filename = self.pick_file()
        corpus = self._load_corpus(filename)
        return generate_batch_from_bytes(corpus, n_samples, seq_len, num_bits, seed)
