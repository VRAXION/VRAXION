# STABLE_LOOP_PHASE_LOCK_004_MUTATION_CREDIT_ASSIGNMENT Result

Status: implemented, sanity run complete, bounded multi-seed smoke complete.

## Final Verdict

Manual verdict:

```text
PHASE_CREDIT_ASSIGNMENT_NOT_SOLVED
PHASE_TRANSPORT_IS_BLOCKER
```

Rejected positive claims:

```text
MUTATION_RESCUES_PHASE_CREDIT_ASSIGNMENT
GATED_POCKETS_UNIQUELY_HELPFUL
UNGATED_POCKETS_SUFFICIENT
UNRESTRICTED_GRAPH_SUFFICIENT
```

The probe is useful because it separates a valid spatial phase-lock task from the mutation-search result. The oracle and fixed local complex reference solve the task perfectly, while mutable local pocket/circuit search stays near chance under the no-private-field, local-only constraint.

## Run Blocks

Static validation before runs:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_mutation_credit_assignment_probe.py
git diff --check
```

Sanity:

```powershell
python scripts\probes\run_stable_loop_phase_lock_mutation_credit_assignment_probe.py `
  --out target\pilot_wave\stable_loop_phase_lock_004_mutation_credit_assignment\sanity_cand1 `
  --seeds 2026 `
  --steps 200 `
  --eval-examples 512 `
  --width 16 `
  --pockets 4 `
  --jackpot 6 `
  --jobs 6 `
  --device cpu `
  --heartbeat-sec 15
```

Result: 10/10 jobs, 6,000 candidate rows, 37.5 seconds.

Bounded smoke:

```powershell
python scripts\probes\run_stable_loop_phase_lock_mutation_credit_assignment_probe.py `
  --out target\pilot_wave\stable_loop_phase_lock_004_mutation_credit_assignment\smoke_bounded `
  --seeds 2026,2027,2028 `
  --steps 400 `
  --eval-examples 1024 `
  --width 24 `
  --pockets 4 `
  --jackpot 6 `
  --jobs 6 `
  --device cpu `
  --heartbeat-sec 20
```

Result: 30/30 jobs, 36,000 candidate rows, 373.8 seconds.

The full planned smoke was not expanded because the bounded 3-seed smoke already falsified the positive gate: no mutable local arm showed nontrivial phase-credit signal, pocket ablation was negligible, and destructive mutation rates rose in the less protected topologies.

## Bounded Smoke Metrics

| Arm | Phase final | Heldout path | Counterfactual | Gate shuffle acc | Highway retention | Pocket drop | Wall leak | Forbidden/private leak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ORACLE_SPATIAL_PHASE_LOCK | 100.0% | 100.0% | 100.0% | 33.6% | 100.0% | 0.0pp | 0.0% | 0.0% |
| FIXED_COMPLEX_MULTIPLY_LOCAL_REFERENCE | 100.0% | 100.0% | 100.0% | 33.7% | 100.0% | 0.0pp | 0.0% | 0.0% |
| HIGHWAY_ONLY_PHASE | 25.3% | 25.2% | 0.0% | 25.3% | 100.0% | 0.0pp | 0.0% | 0.0% |
| HIGHWAY_WITH_RANDOM_POCKETS_NO_WRITEBACK_PHASE | 25.3% | 25.2% | 0.0% | 25.3% | 100.0% | 0.0pp | 0.0% | 0.0% |
| HIGHWAY_WITH_GATED_POCKETS_PHASE | 24.8% | 24.6% | 0.0% | 25.0% | 93.5% | 0.0pp | 0.0% | 0.0% |
| HIGHWAY_WITH_UNGATED_POCKETS_PHASE | 23.2% | 22.9% | 0.0% | 23.2% | 53.1% | 0.0pp | 0.0% | 0.0% |
| UNRESTRICTED_GRAPH_MUTATION_PHASE | 23.4% | 23.4% | 0.7% | 23.0% | 31.1% | 0.4pp | 0.0% | 0.0% |

Chance for K=4 phase labels is about 25%. The mutable arms do not beat the highway-only baseline by the required +5pp, do not pass paired counterfactuals, and do not show nontrivial pocket-ablation drops.

## Credit Split

| Split arm | Phase final | Interpretation |
|---|---:|---|
| MUTABLE_ROUTING_ORACLE_PHASE | 100.0% | If phase transport is supplied by the oracle/reference, routing is not the blocker. |
| ORACLE_ROUTING_MUTABLE_PHASE_POCKET | 20.7% | If routing is supplied but phase transport is mutable, the mutable phase pocket still fails. |
| MUTABLE_ROUTING_MUTABLE_PHASE | 25.1% | Fully mutable local search remains at chance. |

Aggregated gaps:

```text
phase_transport_credit_gap = 0.749
routing_credit_gap = -0.044
routing_phase_interaction_gap = 0.251
```

This points to phase-transport primitive acquisition as the blocker, not path routing alone.

## Shortcut And Locality Audit

The implementation enforces `PublicCase` / `PrivateCase` separation:

```text
PublicCase:
  wall/free mask
  source location and source phase
  target marker
  per-cell local gate vectors

PrivateCase:
  label
  true_path
  path_phase_total
  gate_sum
  oracle routing info
```

Candidate prediction receives only `PublicCase`. The evaluator reads `PrivateCase.label` only for scoring.

Observed bounded-smoke audit:

```text
uses_forbidden_private_field = 0.0
locality_audit_fail_rate     = 0.0
direct_output_leak_rate      = 0.0
wall_leak_rate               = 0.0
```

The oracle/reference also show gate dependence: gate shuffling drops them from 100.0% to about 33.6-33.7%. That means the task is not solved by target-local or fixed-position cues in the oracle/reference path. The mutable arms stay near chance before and after gate shuffle because they never learned the phase transport behavior.

## Interpretation

This 004 run does not reproduce the `HIGHWAY_POCKET_MUTATION_001` positive result under the stricter spatial/local setup. That is the point of the falsification probe:

```text
HIGHWAY_POCKET_MUTATION_001:
  rule-selection smoke positive

STABLE_LOOP_PHASE_LOCK_004:
  raw local spatial phase credit assignment not solved
```

The result is not a task failure. Oracle and fixed local complex multiplication solve it cleanly, paired counterfactuals pass for those references, gate shuffle degrades them, and no forbidden-field leakage is detected.

The negative result is specific:

```text
Mutation-selection over the current local circuit/pocket parameterization did not discover the spatial phase-transport primitive from final target labels.
```

## What This Does Not Prove

This does not prove consciousness.
This does not prove full VRAXION.
This does not prove language grounding.
This does not disprove phase-lock as a useful primitive.
This does not disprove mutation-selection in general.

It only shows that this runner-local, final-label-only, local circuit mutation setup did not solve spatial phase-lock credit assignment without direct oracle fields or named phase rules.

## Next Blocker

The next technical question is no longer whether `gate_sum` shortcuts were contaminating the result. They were removed and audited.

The blocker is now:

```text
How can a mutable local cell acquire or be initialized with the local complex phase-transport primitive without reintroducing direct oracle rules?
```

Likely next probes should focus on primitive acquisition or curriculum inside the local pocket, not another broad topology sweep.
