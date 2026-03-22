"""
Small corpus and bigram helpers for self-wiring graph toy evaluations.
Lazy-loaded: the bigram distribution is computed on first access, not at import.
"""

import os
from pathlib import Path

import numpy as np
from collections import Counter

TEXT = (
    "a stitch in time saves nine. "
    "the early bird catches the worm. "
    "all that glitters is not gold. "
    "actions speak louder than words. "
    "fortune favors the bold. "
    "knowledge is power. "
    "practice makes perfect. "
    "the pen is mightier than the sword. "
    "where there is a will there is a way. "
    "birds of a feather flock together. "
    "every cloud has a silver lining. "
    "honesty is the best policy. "
    "look before you leap. "
    "better late than never. "
    "two wrongs do not make a right. "
    "when in rome do as the romans do. "
    "the squeaky wheel gets the grease. "
    "you can lead a horse to water but you cannot make it drink. "
    "curiosity killed the cat. "
    "do not put all your eggs in one basket. "
    "if you want something done right do it yourself. "
    "the apple does not fall far from the tree. "
    "people who live in glass houses should not throw stones. "
)

chars = sorted(set(TEXT))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB = len(chars)

# Lazy bigram cache
_bigram_cache = {}

FINEWEB_FILENAME = "fineweb_edu.traindat"
REPO_ROOT = Path(__file__).resolve().parents[2]
V42_ROOT = REPO_ROOT / "v4.2"
CANONICAL_TRAINDAT_DIR = V42_ROOT / "data" / "traindat"
CANONICAL_FINEWEB_PATH = CANONICAL_TRAINDAT_DIR / FINEWEB_FILENAME
LEGACY_FINEWEB_PATH = REPO_ROOT / "Diamond Code" / "data" / "traindat" / FINEWEB_FILENAME
FINEWEB_ENV_VAR = "VRAXION_FINEWEB_PATH"


def fineweb_candidate_paths(repo_root: str | os.PathLike | None = None) -> list[Path]:
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    return [
        root / "v4.2" / "data" / "traindat" / FINEWEB_FILENAME,
        root / "Diamond Code" / "data" / "traindat" / FINEWEB_FILENAME,
    ]


def resolve_fineweb_path(
    repo_root: str | os.PathLike | None = None,
    env_var: str = FINEWEB_ENV_VAR,
) -> Path:
    override = os.environ.get(env_var)
    if override:
        override_path = Path(override).expanduser().resolve()
        if override_path.exists():
            return override_path
        raise FileNotFoundError(
            f"{env_var} points to a missing corpus file: {override_path}. "
            f"Preferred canonical local path: {CANONICAL_FINEWEB_PATH}"
        )

    candidates = fineweb_candidate_paths(repo_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find fineweb_edu.traindat. "
        f"Preferred canonical local path: {candidates[0]}. "
        f"Legacy fallback checked: {candidates[1]}. "
        f"Optional override: set {env_var} to the corpus path."
    )


def load_fineweb_bytes(
    max_bytes: int | None = None,
    repo_root: str | os.PathLike | None = None,
    env_var: str = FINEWEB_ENV_VAR,
) -> np.ndarray:
    path = resolve_fineweb_path(repo_root=repo_root, env_var=env_var)
    with path.open("rb") as handle:
        raw = handle.read() if max_bytes is None else handle.read(max_bytes)
    return np.frombuffer(raw, dtype=np.uint8)


def compute_bigram_dist():
    """Compute bigram probability distribution from TEXT. Cached after first call."""
    if 'dist' in _bigram_cache:
        return _bigram_cache['dist'], _bigram_cache['active']

    bigrams = {}
    for i in range(len(TEXT) - 1):
        c1, c2 = TEXT[i], TEXT[i + 1]
        if c1 in char_to_idx and c2 in char_to_idx:
            idx1 = char_to_idx[c1]
            bigrams.setdefault(idx1, []).append(char_to_idx[c2])

    dist = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i, nexts in bigrams.items():
        counts = Counter(nexts)
        total = sum(counts.values())
        for j, count in counts.items():
            dist[i, j] = count / total

    active = [i for i in range(VOCAB) if dist[i].sum() > 0.01]

    _bigram_cache['dist'] = dist
    _bigram_cache['active'] = active
    return dist, active


def bigram_targets():
    """Most common next char for each char. Returns (VOCAB,) int array."""
    dist, _ = compute_bigram_dist()
    targets = np.zeros(VOCAB, dtype=int)
    for i in range(VOCAB):
        if dist[i].sum() > 0:
            targets[i] = np.argmax(dist[i])
        else:
            targets[i] = (i + 1) % VOCAB
    return targets
