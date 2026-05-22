# STABLE_LOOP_PHASE_LOCK_011_SPARSE_CORE_TRANSFER Result

Status: implemented, static validation complete, sanity complete, 3-seed smoke
complete, 5-seed confirm complete.

## Verdict

```text
COMMON_CORE_WAS_ONE_MOTIF_SHORT
SPARSE_CORE_TRANSFER_POSITIVE
WIDTH_TRANSFER_PASSES
LONG_PATH_TRANSFER_PASSES
LAYOUT_TRANSFER_PASSES
RANDOM_MOTIF_CONTROL_FAILS
EXPERIMENTAL_MUTATION_LANE_SUPPORTED
PRODUCTION_API_NOT_READY
```

Interpretation:

```text
the 010 sparse motif core transfers, but the 15-motif common core is not the
complete reusable local rule

adding the missing 1_2_3 motif restores full per-pair transfer

the transferable rule is therefore the completed 16-pair local coincidence
phase transport template, not the raw 15-pair common core alone
```

## Grounding

010 found a stable sparse core with 15 motif types common across seeds and one
seed-specific missing motif:

```text
missing motif: 1_2_3
```

011 directly tests whether that missing motif matters under transfer. It uses
new deterministic spatial families across widths 8, 10, 12, and 16, with no
new growth or pruning.

The path generator is capped to the recurrent settling horizon where
`DENSE_009_REFERENCE` and `FULL_16_RULE_TEMPLATE` remain valid. This prevents a
sparse-core claim from being judged on buckets where the fixed spatial
reference itself fails.

## Runs

Static:

```powershell
cargo check -p instnct-core --example phase_lane_sparse_core_transfer
cargo test -p instnct-core jackpot_traced_emits_candidate_rows_and_accept_invariants
git diff --check
```

Sanity:

```powershell
cargo run -p instnct-core --example phase_lane_sparse_core_transfer --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_011_sparse_core_transfer/sanity ^
  --seeds 2026 ^
  --eval-examples 256 ^
  --widths 8,10 ^
  --ticks 16 ^
  --baseline-steps 100 ^
  --heartbeat-sec 15
```

Smoke:

```powershell
cargo run -p instnct-core --example phase_lane_sparse_core_transfer --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_011_sparse_core_transfer/smoke ^
  --seeds 2026,2027,2028 ^
  --eval-examples 1024 ^
  --widths 8,10,12,16 ^
  --ticks 24 ^
  --baseline-steps 400 ^
  --heartbeat-sec 30
```

Confirm:

```powershell
cargo run -p instnct-core --example phase_lane_sparse_core_transfer --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_011_sparse_core_transfer/confirm ^
  --seeds 2026-2030 ^
  --eval-examples 2048 ^
  --widths 8,10,12,16 ^
  --ticks 32 ^
  --baseline-steps 400 ^
  --heartbeat-sec 30
```

## Confirm Summary

5-seed means:

| Arm | Acc | Prob | CF | Pair min | Gate collapse | Long | Layout |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIXED_PHASE_LANE_REFERENCE | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DENSE_009_REFERENCE | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| COMMON_CORE_15 | 0.97 | 0.92 | 1.00 | 0.68 | 0.96 | 1.00 | 0.95 |
| COMMON_CORE_15_PLUS_MISSING_1_2_3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| FULL_16_RULE_TEMPLATE | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| SEED_SPECIFIC_CORE_REINSERTED | 0.98 | 0.95 | 1.00 | 0.81 | 0.98 | 1.00 | 0.97 |
| CANONICAL_JACKPOT_007_BASELINE | 0.25 | 0.25 | 0.25 | 0.03 | 0.00 | 0.25 | 0.24 |
| RANDOM_MATCHED_15_MOTIF_CONTROL | 0.26 | 0.25 | 0.33 | 0.02 | 0.03 | 0.31 | 0.25 |
| RANDOM_MATCHED_16_MOTIF_CONTROL | 0.24 | 0.23 | 0.26 | 0.00 | 0.02 | 0.23 | 0.22 |
| RANDOM_MATCHED_25_MOTIF_CONTROL | 0.32 | 0.30 | 0.32 | 0.00 | 0.08 | 0.34 | 0.31 |

## Missing-Motif Diagnostic

The common 15-motif template is strong on aggregate:

```text
COMMON_CORE_15:
  accuracy ~= 97%
  correct probability ~= 92%
  counterfactual = 100%
  gate-shuffle collapse ~= 96%
```

But it fails the per-pair rule gate:

```text
min_per_pair_accuracy ~= 68% < required 80%
min_per_pair_probability = 25%
```

Adding only the missing motif fixes the issue:

```text
COMMON_CORE_15_PLUS_MISSING_1_2_3:
  accuracy = 100%
  per-pair min = 100%
  counterfactual = 100%
  gate-shuffle collapse = 100%
```

This supports:

```text
COMMON_CORE_WAS_ONE_MOTIF_SHORT
```

not:

```text
COMMON_TEMPLATE_WORKS
```

## Audits

```text
forbidden_private_field_leak = 0
nonlocal_edge_count = 0
direct_output_leak_rate = 0
random matched controls stay far below sparse/full templates
gate-shuffle collapse is preserved for sparse/full templates
same-target counterfactual passes for sparse/full templates
```

Required output files were produced:

```text
queue.json
progress.jsonl
metrics.jsonl
family_metrics.jsonl
template_metrics.jsonl
per_pair_metrics.jsonl
random_control_metrics.jsonl
counterfactual_metrics.jsonl
locality_audit.jsonl
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

## Claim Boundary

This supports:

```text
the sparse phase-lane core transfers when completed to the full local
phase_i + gate_g -> phase_(i+g) coincidence rule

010's 15-motif common core was distribution-pruned and one motif short

the audited coincidence lane remains experimentally useful
```

This does not support:

```text
production architecture
full VRAXION validity
consciousness
language grounding
Prismion uniqueness
```
