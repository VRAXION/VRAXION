# v22 — Ternary Activation Test Results

## Setup
- 32-class A→B association, 160 neurons, 8 ticks
- Pure hill-climbing mutation (v18 framework, 20 mutations/step)
- Self-wiring with 3D+1D addresses
- seed=42, max 30,000 attempts
- **Note**: Neither config reached 100% — the user's offline v18 implementation
  (which did reach 100% @ 8k) likely had a different mutation strategy or
  hyperparameter set that we don't have in the repo.

## Summary Table

| Config              | Solved | Best Acc | Step   | Conns | Time   |
|---------------------|--------|----------|--------|-------|--------|
| leaky_relu          | NO     | 34.4%    | 30,000 | 1,694 | 38.9s  |
| ternary_soft t=0.1  | NO     | 31.2%    | 30,000 | 1,452 | 119.3s |
| ternary_soft t=0.3  | NO     | 25.0%    | 30,000 | 1,535 | 117.3s |
| ternary_soft t=0.5  | NO     | 21.9%    | 30,000 | 1,582 | 106.4s |
| ternary_soft t=1.0  | NO     | **40.6%** | 30,000 | 1,912 | 117.0s |

## Key Finding: -1 State IS Active at 160 Neurons!

Unlike the 16-class/80-neuron offline test where -1 never activated,
**at 160 neurons the -1 state is consistently used**:

| Config     | Avg +1 neurons | Avg -1 neurons | Avg 0 neurons | Notes                    |
|------------|----------------|----------------|---------------|--------------------------|
| t=0.1      | 34.6           | 24.7           | 36.7          | Very active, almost binary |
| t=0.3      | 27.9           | 21.3           | 46.8          | Balanced three-state      |
| t=0.5      | 24.4           | 16.2           | 55.4          | Sparse, more zeros        |
| t=1.0      | 11.7           | 9.0            | 75.3          | Very sparse, few active   |

(Out of 96 internal neurons total)

## Analysis

1. **t=1.0 wins on accuracy (40.6%)** — the highly selective threshold creates
   sparse codes that are easier for hill-climbing to optimize. This matches the
   offline 16-class result where t=1.0 was fastest.

2. **t=0.1 wins on connection efficiency (1,452 conns)** — fewer connections
   than leaky_relu (1,694) at similar accuracy (31.2% vs 34.4%). The dense
   ternary codes pack more information per connection.

3. **-1 inhibition scales with network size** — at 80 neurons (offline), -1
   never activated. At 160 neurons, 9-25 neurons actively inhibit depending
   on threshold. This suggests inhibition becomes more useful as networks grow.

4. **Speed penalty**: ternary_soft is ~3x slower than leaky_relu due to
   per-sample activation (can't fully vectorize the three-way branch).

5. **None reached 100%** — the hill-climbing implementation here is weaker than
   the user's offline v18. The relative ordering (t=1.0 > leaky_relu > t=0.1 >
   t=0.3 > t=0.5) should be valid for comparison though.

## Next Steps
- Get the actual v18 mutation code to reach 100% baseline
- If t=1.0 maintains its lead at 100%, test 64/128-class scaling
- Investigate why t=0.3 was best at 16-class but worst here at 32-class
  (hypothesis: the optimal threshold scales with network size)
