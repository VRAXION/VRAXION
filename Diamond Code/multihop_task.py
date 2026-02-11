"""
Multi-hop Reasoning Task Generator

Task: Compositional memory access via chained lookups.

Example:
  Facts: A→B, B→C, C→D
  Query: "Given A, what is the final value after 2 hops?"
  Answer: C (A→B→C)

This tests memory capacity via compositional reasoning:
- Must store multiple independent chains (A→B→C, D→E→F, etc.)
- Must chain lookups to find answer (not just single recall)
- Harder chains (3-hop) require more memory capacity

Task format:
  Input: [A, B, LINK, B, C, LINK, C, D, LINK, ..., QUERY, A, ?]
  Output: Final entity after N hops

Special tokens:
  LINK = 0.0 (separator between facts)
  QUERY = -999.0 (query marker)
  Entity IDs = [1.0, 2.0, ..., vocab_size]

Difficulty scaling:
  1-hop: 10 chains × 1 hop = 10 facts (~30 tokens)
  2-hop: 10 chains × 2 hops = 20 facts (~60 tokens)
  3-hop: 10 chains × 3 hops = 30 facts (~90 tokens)
"""

import torch
import random
from typing import Tuple, Dict, List


def generate_multihop(
    n_samples: int = 100,
    num_chains: int = 10,
    chain_length: int = 2,
    vocab_size: int = 100,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Generate multi-hop reasoning dataset.

    Args:
        n_samples: Number of sequences to generate
        num_chains: Number of independent fact chains per sequence
        chain_length: Number of hops in each chain (1, 2, 3, etc.)
        vocab_size: Number of unique entity IDs available
        seed: Random seed for reproducibility

    Returns:
        x: [n_samples, seq_len, 1] - Input token sequences
        y: [n_samples] - Target entity ID after chain_length hops
        metadata: Dict with chain structures for debugging

    Sequence structure:
        [fact1_key, fact1_val, LINK, fact2_key, fact2_val, LINK, ..., QUERY, query_key, ?]

    Example (2-hop, 2 chains):
        Chain 0: 5→17→23 (entities 5, 17, 23)
        Chain 1: 8→42→91 (entities 8, 42, 91)

        Sequence: [5, 17, LINK, 17, 23, LINK, 8, 42, LINK, 42, 91, LINK, QUERY, 5, ?]
        Target: 23 (start at 5, hop to 17, hop to 23)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Special tokens
    LINK_TOKEN = 0.0
    QUERY_TOKEN = -999.0

    # Each chain: chain_length facts, each fact is 3 tokens (key, val, LINK)
    # Plus final QUERY phase: 3 tokens (QUERY, query_key, ?)
    tokens_per_chain = chain_length * 3
    seq_len = num_chains * tokens_per_chain + 3

    # Initialize tensors
    x = torch.zeros((n_samples, seq_len, 1), dtype=torch.float32)
    y = torch.zeros((n_samples,), dtype=torch.long)

    metadata = {
        'chains': [],  # List of chain structures per sample
        'query_chains': [],  # Which chain was queried per sample
        'vocab_size': vocab_size,
        'num_chains': num_chains,
        'chain_length': chain_length,
        'seq_len': seq_len,
    }

    for idx in range(n_samples):
        # Generate independent chains
        chains = []
        used_entities = set()

        for chain_idx in range(num_chains):
            # Create chain of length (chain_length + 1) entities
            # Example: 2-hop needs 3 entities (A→B→C)
            chain = []
            for _ in range(chain_length + 1):
                # Sample entity ID not yet used in this sequence
                # Use 0-indexed IDs (0 to vocab_size-1) for PyTorch cross_entropy
                attempts = 0
                while attempts < 1000:
                    entity_id = random.randint(0, vocab_size - 1)
                    if entity_id not in used_entities:
                        used_entities.add(entity_id)
                        chain.append(entity_id)
                        break
                    attempts += 1
                if attempts >= 1000:
                    raise RuntimeError(
                        f"Could not generate unique entities for sample {idx}, chain {chain_idx}. "
                        f"Try increasing vocab_size (current: {vocab_size})"
                    )
            chains.append(chain)

        # Build input sequence: store all facts
        pos = 0
        for chain in chains:
            # Each chain stores facts: A→B, B→C, C→D, etc.
            for hop_idx in range(chain_length):
                key = float(chain[hop_idx])
                val = float(chain[hop_idx + 1])
                x[idx, pos, 0] = key
                x[idx, pos + 1, 0] = val
                x[idx, pos + 2, 0] = LINK_TOKEN
                pos += 3

        # Query phase: pick random chain and ask for final value
        query_chain_idx = random.randint(0, num_chains - 1)
        query_chain = chains[query_chain_idx]
        query_key = query_chain[0]  # Start of chain
        target_val = query_chain[chain_length]  # End of chain (after N hops)

        x[idx, pos, 0] = QUERY_TOKEN
        x[idx, pos + 1, 0] = float(query_key)
        x[idx, pos + 2, 0] = 0.0  # Placeholder for answer (model predicts this)

        y[idx] = target_val  # Target is the final entity ID

        # Store metadata
        metadata['chains'].append(chains)
        metadata['query_chains'].append(query_chain_idx)

    return x, y, metadata


def verify_multihop_task():
    """
    Verification test for multi-hop task generator.
    Run this to check correctness.
    """
    print("=" * 70)
    print("Multi-hop Task Generator - Verification Test")
    print("=" * 70)
    print()

    # Test 2-hop, 3 chains, small vocab
    x, y, meta = generate_multihop(
        n_samples=5,
        num_chains=3,
        chain_length=2,
        vocab_size=50,
        seed=42,
    )

    print(f"Generated {x.shape[0]} samples")
    print(f"Sequence length: {x.shape[1]}")
    print(f"Vocab size: {meta['vocab_size']}")
    print(f"Num chains: {meta['num_chains']}")
    print(f"Chain length (hops): {meta['chain_length']}")
    print()

    # Show first sample in detail
    sample_idx = 0
    print(f"Sample {sample_idx} breakdown:")
    print("-" * 70)

    chains = meta['chains'][sample_idx]
    query_chain_idx = meta['query_chains'][sample_idx]

    for chain_idx, chain in enumerate(chains):
        chain_str = " -> ".join([str(e) for e in chain])
        marker = " (QUERIED)" if chain_idx == query_chain_idx else ""
        print(f"  Chain {chain_idx}: {chain_str}{marker}")

    print()
    print(f"Query: Start at {chains[query_chain_idx][0]}, find value after {meta['chain_length']} hops")
    print(f"Expected answer: {y[sample_idx].item()}")
    print(f"Chain path: {' -> '.join([str(e) for e in chains[query_chain_idx]])}")
    print()

    # Show raw sequence
    print("Raw sequence (first 30 tokens):")
    seq = x[sample_idx, :30, 0].tolist()
    print("  " + ", ".join([f"{t:.1f}" for t in seq]))
    print()

    # Verify correctness
    print("Verification checks:")
    print("-" * 70)

    # Check 1: Target matches chain end
    expected = chains[query_chain_idx][meta['chain_length']]
    actual = y[sample_idx].item()
    check1 = expected == actual
    print(f"  [{'OK' if check1 else 'X'}] Target matches chain end: {expected} == {actual}")

    # Check 2: Sequence length correct
    expected_len = meta['num_chains'] * meta['chain_length'] * 3 + 3
    actual_len = x.shape[1]
    check2 = expected_len == actual_len
    print(f"  [{'OK' if check2 else 'X'}] Sequence length: {expected_len} == {actual_len}")

    # Check 3: LINK tokens in right places
    link_positions = [i for i in range(x.shape[1]) if x[sample_idx, i, 0] == 0.0]
    expected_links = meta['num_chains'] * meta['chain_length']
    check3 = len(link_positions) >= expected_links  # At least this many (placeholder may be 0)
    print(f"  [{'OK' if check3 else 'X'}] LINK tokens found: {len(link_positions)} >= {expected_links}")

    # Check 4: QUERY token present
    query_positions = [i for i in range(x.shape[1]) if x[sample_idx, i, 0] == -999.0]
    check4 = len(query_positions) == 1
    print(f"  [{'OK' if check4 else 'X'}] QUERY token found: {len(query_positions)} == 1")

    print()
    if all([check1, check2, check3, check4]):
        print("[OK] All verification checks passed!")
    else:
        print("[X] Some checks failed - investigate")

    print("=" * 70)
    print()

    return x, y, meta


if __name__ == "__main__":
    verify_multihop_task()
