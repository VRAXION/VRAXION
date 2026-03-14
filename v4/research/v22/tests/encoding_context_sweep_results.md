# Encoding & Context Sweep Results (2026-03-14)

## Encoding Type (8 bits, 29 chars, 4K budget, 3 seeds)

| Encoding | Bits | Bit Acc | Exact |
|---|---|---|---|
| byte_16 | 16 | 90.0% | 31.7% |
| byte_12 | 12 | 86.2% | 28.6% |
| byte_8 | 8 | 79.4% | 28.6% |
| hadamard_32 | 32 | 65.0% | 28.6% |
| spread_8 | 8 | 65.1% | 14.3% |
| random_8 | 8 | 62.3% | 7.9% |
| byte_4 | 4 | 61.9% | 28.6% |

**Winner: byte encoding.** Hadamard/spread/random all worse.

## Bit Width (byte encoding, 4K budget, 3 seeds)

| Bits | N | Bit Acc | Exact |
|---|---|---|---|
| 4 | 120 | 61.9% | 28.6% |
| 6 | 124 | 72.8% | 30.2% |
| 8 | 128 | 79.4% | 28.6% |
| 10 | 132 | 83.3% | 28.6% |
| 12 | 136 | 86.2% | 28.6% |
| 16 | 144 | 90.0% | 31.7% |
| 20 | 152 | 91.7% | 28.6% |
| 24 | 160 | 93.1% | 28.6% |
| 32 | 176 | 94.8% | 28.6% |

**Finding: bit accuracy scales with width, exact match plateaus at ~29%.**
More bits = easier per-bit, but network can only learn ~8 correct bits per char.
Optimal: 8 bits (no wasted neurons).

## Context Length (byte encoding, 6K budget, 3 seeds, longer text)

| Context | Input bits | Samples | N | Bit Acc | Exact |
|---|---|---|---|---|---|
| 1 char | 8 | 27 | 128 | 78.9% | 32.1% |
| 2 chars | 16 | 53 | 136 | 78.6% | 22.6% |
| 3 chars | 24 | 58 | 144 | 78.5% | 22.4% |

**Finding: more context HURTS.** Network builds lookup tables, not generalizations.
More context = more unique samples = less accuracy at fixed budget.
