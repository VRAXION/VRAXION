# STABLE_LOOP_PHASE_LOCK_007_INSTNCT_PHASE_LANE_WAVEFIELD Result

Status: implemented, static validation complete, sanity and bounded 3-seed smoke complete.

## Verdict

```text
INSTNCT_PHASE_LANE_TASK_VALID
FIXED_PHASE_LANE_ONLY
```

Rejected for this implementation:

```text
INSTNCT_MUTATION_RESCUES_PHASE_CREDIT
SEEDED_PRIMITIVE_REQUIRED
CHANNEL_MUTATION_REQUIRED
POLARITY_MUTATION_REQUIRED
LOOPS_REQUIRED
DIRECT_SHORTCUT_CONTAMINATION
```

## Runs

Static:

```powershell
cargo check -p instnct-core --example phase_lane_wavefield_grower
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

All passed.

Sanity:

```powershell
cargo run -p instnct-core --example phase_lane_wavefield_grower --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_007_instnct_phase_lane_wavefield/sanity ^
  --seeds 2026 ^
  --steps 100 ^
  --eval-examples 256 ^
  --width 8 ^
  --ticks 6 ^
  --jackpot 6 ^
  --heartbeat-sec 15
```

Result: 10/10 jobs, 4,200 candidate rows, 2.0 seconds after release build.

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_wavefield_grower --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_007_instnct_phase_lane_wavefield/smoke ^
  --seeds 2026,2027,2028 ^
  --steps 600 ^
  --eval-examples 1024 ^
  --width 12 ^
  --ticks 6 ^
  --jackpot 9 ^
  --heartbeat-sec 30
```

Result: 30/30 jobs, 113,400 candidate rows, 76.1 seconds.

## Bounded Smoke Summary

| Arm | Accuracy | Correct target probability | Candidate nonzero delta | Candidate positive delta | Accepted | Nonlocal edges |
|---|---:|---:|---:|---:|---:|---:|
| ORACLE_PHASE_LANE_WIRING | 100.0% | 97.0% | 0.0% | 0.0% | 0 | 0 |
| PARTICLE_FRONTIER_004_BASELINE | 22.3% | 22.4% | 0.0% | 0.0% | 0 | 0 |
| RANDOM_PHASE_LANE_NETWORK | 27.1% | 25.0% | 0.0% | 0.0% | 0 | 0 |
| INSTNCT_GROWER_STRICT_K9 | 27.1% | 25.0% | 49.8% | 0.0% | 0 | 0 |
| INSTNCT_GROWER_TIES_K9 | 27.1% | 25.0% | 54.7% | 0.0% | 1777 | 0 |
| INSTNCT_GROWER_ZEROP_K9 | 27.1% | 25.1% | 51.3% | 0.0% | 571 | 0 |
| NO_CHANNEL_MUTATION_ABLATION | 27.1% | 25.0% | 54.4% | 0.0% | 0 | 0 |
| NO_POLARITY_MUTATION_ABLATION | 27.1% | 25.0% | 49.5% | 0.0% | 0 | 0 |
| NO_LOOP_MUTATION_ABLATION | 27.1% | 25.4% | 36.7% | 0.0% | 1 | 0 |
| SEEDED_PHASE_LANE_MOTIF_GROWER | 27.3% | 25.3% | 62.2% | 0.0% | 4 | 0 |

The fixed oracle/reference validates the task:

```text
ORACLE_PHASE_LANE_WIRING:
  accuracy = 100.0%
  correct target probability = 97.0%
  same-target counterfactual = 100.0%
  gate-shuffle collapse = 100.0%
```

The canonical mutation arms did not rescue phase credit:

```text
best mutable accuracy = 27.3%
random baseline accuracy = 27.1%
best mutable correct probability = 25.4%
random baseline correct probability = 25.0%
positive candidate delta fraction = 0.0%
```

This is important: the candidate landscape was not silent. Roughly half of evaluated candidates produced nonzero fitness deltas, but those deltas did not become positive improvements under the current phase-lane encoding and jackpot mutation schedule.

## Operator Summary

Across smoke candidate logs:

| Operator | Seen | Accepted | Accept rate |
|---|---:|---:|---:|
| add_edge | 11958 | 19 | 0.2% |
| remove_edge | 11981 | 545 | 4.5% |
| rewire | 11729 | 18 | 0.2% |
| reverse | 11980 | 353 | 2.9% |
| mirror | 11979 | 353 | 2.9% |
| enhance | 11873 | 17 | 0.1% |
| theta | 11935 | 560 | 4.7% |
| channel | 10208 | 479 | 4.7% |
| loop2 | 10027 | 9 | 0.1% |
| loop3 | 9730 | 0 | 0.0% |

`NO_POLARITY_MUTATION_ABLATION` is an audit row. The current canonical jackpot schedule does not expose polarity as a sampled public operator, so this run cannot support a polarity-specific conclusion.

## Interpretation

This is a clean negative for the current canonical INSTNCT phase-lane grower:

```text
fixed/reference phase-lane task: works
canonical local jackpot mutation: did not discover phase transport
seeded same-phase motif: did not rescue
channel/loop ablations: no decisive difference because no mutable arm lifted off
shortcut/locality contamination: not observed
```

The result does not contradict 005. The fixed wavefield idea still works. It says that mapping the wavefield into this first integer phase-lane `Network` encoding is not enough for canonical mutation to discover the transport/interference rule from target probability.

Likely blocker:

```text
The substrate exposes phase lanes and local gates, but the available graph mutations
do not easily create the required conditional phase rotation / coincidence motif.
```

The next useful tests should not rerun this bigger. They should isolate the missing construct:

```text
1. add an explicit local coincidence/gating motif primitive and test whether mutation can tune it
2. test a tiny single-cell/two-cell phase-lane gate composition circuit in instnct-core
3. add polarity as an explicit canonical mutation operator only if the core theory requires it
4. compare against a small hand-built integer phase-rotation motif before spatial transfer
```

## Claim Boundary

This probe only tests canonical INSTNCT phase-lane constructability on a toy spatial phase-lock setting.

It does not prove consciousness, full VRAXION validity, language grounding, Prismion uniqueness, or physical quantum behavior.

It also does not disprove VRAXION mutation generally. It falsifies this particular first mapping:

```text
4 phase lanes per cell
local gate token neurons
canonical jackpot graph/threshold/channel mutations
final target-lane probability fitness
```
