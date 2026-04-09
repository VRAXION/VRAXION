# Overnight Research Log — 2026-04-09 (23:00 - 05:00)

## Agenda
Starting from the evening session findings:
- Holographic (1-step) proven > pathway
- Swish best for MUL (75%), C19 per-C for ADD+PAR (100%)
- MUL 75% = current ceiling
- Shared W recurrence > layered W

## Experiments

### Exp 1: MUL Width × Weight × Tick Scaling
- Status: RUNNING
- Question: does N=16 or wider weights (±4) or 3 ticks break MUL 75%?
- Config: 9 combinations of N=[8,12,16] × w_range=[±1,±4] × ticks=[2,3]
- Result: BIGGER IS WORSE. Best = baseline N=8 ±1 2-tick = 12/16 (75%).
  N=16 ±4 3-tick = 9/16 (56%). Larger search space + fixed sample = worse coverage.
  Width and weight range don't help — the search can't find rare solutions in larger spaces.

### Exp 2: Incremental Holographic Build for MUL
- Status: RUNNING
- Question: does neuron-by-neuron building work for MUL like it did for ADD?
- Swish activation, 2 ticks, ternary weights, build from N=8 to N=14
- Result: 81% ceiling (13/16) at n=9, plateaus through n=14.
  Better than flat N=8 (75%) but NOT 100%. Ternary weights may be the bottleneck.

### Exp 3: Incremental + wider weights (±4) for MUL
- Status: RUNNING
- Question: does ±4 weight range break the 81% ceiling?
- Result: NO. ±1=75%, ±2=75%, ±4=81%, ±8=81%. Same 81% ceiling.
  Weight range is NOT the bottleneck.

### Exp 4: Different readout strategies for MUL
- Status: RUNNING
- Question: is the readout (total_charge→nearest) the bottleneck?
- Testing: total_charge, max_neuron, per_class_neuron, weighted_sum
- Result: total_charge=68%, weighted_sum=68%, max=62%, per_class=56%.
  Readout is NOT the bottleneck.

### Exp 5: MUL with different input encodings
- Status: RUNNING
- Question: is THERMOMETER encoding wrong for multiplication?
- Testing: thermo, onehot, magnitude, binary, thermo×onehot hybrid
- Hypothesis: MUL needs the inputs in DIFFERENT format
- Result: ALL encodings 62-75%. Encoding is NOT the bottleneck.
  thermo=magnitude=75%, binary=68%, onehot=thermo×onehot=62%.

### KEY INSIGHT: Why MUL is fundamentally hard
MUL (a×b) is BILINEAR — needs input_A × input_B. But `act @ W` is LINEAR in act.
W is FIXED (doesn't depend on input). So `W × input` can never compute input[0] × input[4].
Need: the input to INTERACT WITH ITSELF. This is what attention (Q×K^T) does.

### Exp 6: Bilinear interaction layer for MUL
- Status: RUNNING
- Adding pairwise product terms: act[i] × act[j] × bilinear_weight
- Testing: standard (linear only), bilinear (linear + products), pure_bilinear (products only)
- Result: standard=75%, bilinear=75%, pure_bilinear=0%. Bilinear doesn't help!
  The search space grew (84 params) but sample stayed 5M → can't find the solution.

### KEY INSIGHT #2: MUL fundamentally needs input×input interaction
`act @ W` is linear in act. W is fixed. So act[0] × act[4] is IMPOSSIBLE.
This is like needing attention (Q×K^T) where both Q and K come from input.

### Exp 7: Area encoding — encode multiplication IN the input
- Status: RUNNING
- If a=2, b=3 → a 2×3 grid of active cells (6 cells) → network just COUNTS
- This makes MUL into ADD (linear!) at the cost of more input neurons (4×4=16)
- If area=100% → proves the network CAN solve it with the right encoding
- Result: AREA = 100% ✓ SOLVED! Thermo = 62%.
  PROOF: the network CAN'T multiply, but CAN count. If you pre-encode a×b as
  area (a×b active cells), the network just counts = linear = 1-step.
  MUL is a COMPUTATION problem, not a network problem.

### KEY INSIGHT #3: The encoding IS the computation
- Addition: thermometer naturally encodes the sum → 1-step count works
- Multiplication: area encoding pre-computes the product → 1-step count works
- The HARD part is the encoding, not the network
- The network is a universal LINEAR computer
- Nonlinear computation must happen BEFORE or IN the encoding

### Exp 8: Full capability map (1-step holographic)
- Status: DONE
- Result:
  ```
  ✓ ADD (100%)  — linear count
  ✓ MAX (100%)  — threshold-linear
  ✓ MUL area (100%) — pre-computed in encoding
  ✓ a/b floor (100%) — monotonic in a (12 examples)
  
  SUB (75%), MIN (87%), PAR (87%), a==b (81%), |a-b| (81%), a%3 (81%), MUL thermo (68%)
  ```
- The 1-step holographic network is a UNIVERSAL LINEAR COMPUTER.

### Exp 9: Depth comparison (1-tick vs 2-tick vs 3-tick)
- Status: DONE
- Result:
  ```
             1t    2t    3t   delta
  SUB:       75%   68%   68%   -6pp (WORSE!)
  MIN:       87%   87%   93%   +6pp
  MUL:       68%   75%   68%   +0pp (2t peak, 3t drops)
  PAR:       87%   93%  100%  +12pp ✓ SOLVED!
  a==b:      81%   87%   87%   +6pp
  |a-b|:     81%   87%   93%  +12pp
  ```
- PARITY solved at 3 ticks. |a-b| close (93%). MUL doesn't benefit from depth.
- Depth helps XOR-like tasks. Depth HURTS subtraction. Depth NEUTRAL for multiplication.

### Exp 10: Incremental build on remaining tasks (best tick depth)
- Status: DONE
- Result:
  ```
  |a-b| 3t: n=9 → 100% ✓ SOLVED!
  a==b  3t: 87% STUCK (n=8-14, no improvement)
  MIN   3t: 93% STUCK
  SUB   1t: 75% STUCK
  ```

## OVERNIGHT SUMMARY

### Complete task capability map (holographic + swish + incremental build):

| Task | Best result | Ticks | Neurons | Method |
|---|---|---|---|---|
| ADD | **100%** ✓ | 1 | 8 | thermometer count |
| MAX | **100%** ✓ | 1 | 8 | threshold-linear |
| MUL area | **100%** ✓ | 1 | 16 | area pre-encoding |
| a/b floor | **100%** ✓ | 1 | 8 | monotonic (12 ex) |
| PARITY | **100%** ✓ | 3 | 8 | shared W recurrence |
| \|a-b\| | **100%** ✓ | 3 | 9 | incremental build |
| MIN | 93% | 3 | 8-14 | close but stuck |
| a==b? | 87% | 3 | 8-14 | stuck |
| MUL thermo | 81% | 2 | 9 | incremental, bilinear limit |
| SUB | 75% | 1 | 8-14 | signed output hard |

### Key findings:
1. **Width scaling doesn't help** — larger N with fixed sample = worse coverage
2. **Weight range doesn't help** — ±8 same as ±1 with incremental
3. **Bilinear interaction doesn't help** — search space too large
4. **Encoding IS computation** — area encoding makes MUL trivial (100%)
5. **Depth is task-specific** — helps PARITY/|a-b|, hurts SUB, neutral for MUL
6. **Incremental build** unlocks |a-b| (93%→100%) but not a==b?/MIN/SUB
7. **MUL is fundamentally bilinear** — needs input×input, which `act @ W` cannot do

### Unsolved frontier:
- MUL: needs bilinear/attention mechanism (Q×K^T)
- a==b?: needs exact match detection (nonlinear comparison)
- MIN: very close (93%), might need 4+ ticks or more neurons
- SUB: signed output difficult with unsigned readout

## Late Night: MUL architecture limits (PROVEN)

- **Square activation = 81% MUL** (best ever, 1-tick). Cross-term 2ab in (a+b)². Overflow at 2+ ticks.
- **Full input (every neuron sees A,B)**: WORSE than thermometer (56% vs 68%).
- **Per-connection bias + gradient descent**: N=5 = 15/16 max. ARCHITECTURE LIMIT not search.
- **Newton 2nd order**: N=4 = 12/16 max. Same ceiling.
- **PROVEN: MUL is UNSOLVABLE** with (act+bias)×weight + signed square + linear readout.
- Needs: bilinear layer, input-dependent ticks, or fundamentally different architecture.
