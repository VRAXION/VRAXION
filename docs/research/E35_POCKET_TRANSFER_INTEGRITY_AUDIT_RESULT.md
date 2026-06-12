# E35 Pocket Transfer Integrity Audit Result

Status: complete.

Decision:

```text
e35_transfer_partial
```

## Summary

E35 tested whether a learned/exported Pocket Operator can be treated as a
reusable library artifact instead of a one-off probe state.

The result is a partial positive:

```text
Reusable protocol/framing pocket: yes, under stable framing conditions.
World-specific codebook transfer without adapter: no.
Small target codebook adapter: yes.
Bit-slip recovery: no, still the E34D bottleneck.
Dirty scratch decoder still gets higher raw accuracy, but with worse safety.
```

The key architectural result is that the transferable unit should not be one
large `BinaryIngressPocket`. It should be split:

```text
ProtocolFramingIngressPocket
  transferable frozen anchor

TargetCodebookAdapter
  small world-specific adapter
```

## Primary Run

Run root:

```text
target/pilot_wave/e35_pocket_transfer_integrity_audit
```

Pocket archive:

```text
docs/research/pocket_archive/e35_transfer_smoke/binary_ingress/protocol_framing_ingress_v001/
```

Primary decision:

```text
e35_transfer_partial
```

Primary metrics:

```text
frozen_same_world_success              = 1.000000
frozen_target_world_success            = 0.000000
adapter_stable_target_success          = 1.000000
adapter_target_world_success           = 0.741111
adapter_bitslip_target_success         = 0.352778
adapter_target_wrong_feature_write     = 0.000000
adapter_target_false_frame_commit      = 0.006612
wrong_pocket_target_world_success      = 0.000000
scratch_target_world_success           = 0.967778
target_safety_gain_vs_ablation         = 0.010648
target_false_frame_gain_vs_ablation    = 0.019685
source_policy_accepted_mutations       = 9
target_adapter_accepted_mutations      = 4
target_adapter_rejected_mutations      = 32
repair_accepted_mutations              = 0
repair_rejected_mutations              = 288
deterministic_replay_match_rate        = 1.000000
```

Interpretation:

```text
Frozen import works perfectly when the target uses the same codebook.
Frozen import fails when the target codebook changes.
A small adapter restores stable target transfer to 1.0.
Wrong-pocket import does not silently help.
Full retrain/repair does not improve over adapter import.
```

## Confirm And Stress Seeds

All confirm/stress lanes reproduced the same decision.

```text
seed35002: decision=e35_transfer_partial, stable=1.000000, target=0.742857, bitslip=0.357143, wrong=0.000000, wrong_pocket=0.000000
seed35003: decision=e35_transfer_partial, stable=1.000000, target=0.760000, bitslip=0.400000, wrong=0.000000, wrong_pocket=0.005714
seed35004: decision=e35_transfer_partial, stable=1.000000, target=0.755714, bitslip=0.389286, wrong=0.000000, wrong_pocket=0.000000
seed35005: decision=e35_transfer_partial, stable=1.000000, target=0.736000, bitslip=0.340000, wrong=0.000000, wrong_pocket=0.000000
seed35006: decision=e35_transfer_partial, stable=1.000000, target=0.746000, bitslip=0.365000, wrong=0.000000, wrong_pocket=0.000000
```

This makes the result stable enough to treat as a real signal:

```text
Protocol/framing transfer: confirmed on stable target families.
Full binary ingress transfer: not confirmed because bit-slip remains.
```

## Pocket Archive Files

The E35 run exported:

```text
pocket_manifest.json
pocket_contract.md
frozen_params.json
lineage.json
source_metrics.json
transfer_tests.json
safety_report.json
```

The exported pocket explicitly declares that it does not own world-specific
feature-codebook mapping. That boundary is intentional.

## Checker

Primary target checker:

```text
passed = true
failure_count = 0
```

Primary sample-only checker:

```text
passed = true
failure_count = 0
```

Static validation:

```text
py_compile passed
git diff --check passed
```

## What This Proves

E35 supports this:

```text
A Pocket Operator can be exported, archived, frozen, imported, and reused
across related worlds when its contract boundary is clean.
```

Specifically:

```text
ProtocolFramingIngressPocket transfers.
TargetCodebookAdapter must remain world-specific.
Wrong pocket import is harmful/blocked by metrics.
Adapter training has real accepted/rejected/rollback evidence.
```

## What This Does Not Prove

E35 does not prove:

```text
bitstream-to-text solved
raw language reasoning
general reusable intelligence
automatic pocket discovery
bit-slip tolerant stream reassembly
```

It also does not prove that every pocket is transferable. It proves one important
class:

```text
protocol/framing ingress pockets with explicit contracts.
```

## Scientific Interpretation

The strongest result is not raw score. The strongest result is lifecycle:

```text
train source pocket
export frozen anchor
archive with manifest/contract/tests
import into target
learn small adapter
verify wrong-pocket and ablation controls
```

That is the first concrete step toward a real Pocket Library.

The caveat is equally important:

```text
Do not archive fused, world-specific pockets as universal skills.
Archive transferable mechanism pockets and keep adapters separate.
```

## Recommended Next Step

Two paths now make sense:

```text
E34E_BITSLIP_TOLERANT_STREAM_REASSEMBLY_PROBE
```

to repair the remaining binary stream bottleneck, and:

```text
E36_POCKET_LIBRARY_IMPORT_RUNTIME_PROBE
```

to make archived pockets first-class runtime inputs rather than merely archived
research artifacts.

Practical recommendation:

```text
Do E34E next if the immediate goal is binary ingress quality.
Do E36 next if the immediate goal is long-term skill-library infrastructure.
```

Boundary: E35 is a controlled Pocket Operator transfer audit. It does not make
language, AGI, consciousness, deployed-model, or model-scale claims.
