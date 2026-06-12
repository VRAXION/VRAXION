# E35 Pocket Transfer Integrity Audit Contract

## Summary

E35 tests whether a learned Pocket Operator can become a reusable library
artifact rather than a one-off probe result.

Core question:

```text
Can a learned/exported Ingress Codec pocket be frozen, archived, imported into
a related but non-identical active-evidence world, adapted with a small
codebook adapter, and verified by ablation and wrong-pocket controls?
```

## Scope

E35 does not try to solve bit-slip stream reassembly. It separates:

```text
Protocol/framing pocket:
  START/LENGTH/CRC/END/requested-feature commit hygiene

World-specific codebook adapter:
  raw feature code -> target Flow Field feature id
```

This avoids treating one large binary-ingress bundle as a magical transferable
skill.

## Systems

```text
scratch_no_pocket
frozen_import_pocket
imported_plus_small_adapter
full_retrain_from_import
wrong_pocket_negative_control
protocol_ablation_no_import
oracle_invalid_control
```

## Required Pocket Archive

The runner must export a reusable pocket package under:

```text
docs/research/pocket_archive/e35_transfer_smoke/
```

Required files:

```text
pocket_manifest.json
pocket_contract.md
frozen_params.json
lineage.json
source_metrics.json
transfer_tests.json
safety_report.json
```

## Metrics

```text
same_codebook_zero_shot
shifted_codebook_frozen
shifted_codebook_adapter
stable_target_success
bitslip_target_success
closed_loop_success
wrong_feature_write_rate
false_frame_commit_rate
localized_ablation_drop
wrong_pocket_target_success
adapter accepted/rejected/rollback counts
source accepted/rejected/rollback counts
deterministic replay
checker failure count
```

## Decision Labels

```text
e35_pocket_transfer_integrity_confirmed
e35_transfer_partial
e35_no_transfer_detected
e35_negative_transfer_detected
e35_invalid_artifact_or_oracle_detected
```

## Positive Signal

Positive or partial-positive transfer requires:

```text
frozen same-codebook import works
shifted-codebook adapter improves target transfer
wrong pocket does not silently match the real import
protocol ablation causes localized degradation
wrong feature write stays low
adapter path has accepted/rejected/rollback evidence
checker and sample-only checker pass
```

## Hard Requirements

```text
real row-level eval
exported pocket archive
frozen params and manifest hashes
wrong-pocket negative control
ablation control
mutation/rollback counts
progress.jsonl and hardware_heartbeat.jsonl
deterministic replay
sample-only checker
no gradient descent
no optimizer/backprop
no AGI/consciousness/model-scale claims
```

Boundary: E35 is a controlled Pocket Operator transfer audit. It does not prove
general reusable intelligence, raw language reasoning, AGI, consciousness,
deployed-model behavior, or model-scale behavior.
