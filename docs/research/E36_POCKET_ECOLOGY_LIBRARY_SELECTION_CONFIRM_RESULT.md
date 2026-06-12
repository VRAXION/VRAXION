# E36 Pocket Ecology Library Selection Confirm Result

Status: complete.

Decision:

```text
e36_pocket_ecology_selection_partial
```

## Summary

E36 tested whether a shared Pocket Library can act as an evolutionary selection
layer across fresh target worlds:

```text
candidate pockets
-> paired ablation / random import / toxic control
-> pocket_value
-> lifecycle status
-> promoted stable library subset
```

The result is a partial positive:

```text
Pocket Ecology selection works for clean stable-family mechanism transfer.
It correctly promotes ProtocolFramingIngressPocket.
It bans toxic/wrong-codebook imports.
It deprecates dormant AFK pockets.
It does not solve the E35 bit-slip bottleneck.
It does not beat dirty scratch on full target raw success.
```

## Primary Run

Run root:

```text
target/pilot_wave/e36_pocket_ecology_library_selection_confirm
```

Canonical ecology library snapshot:

```text
docs/research/pocket_ecology/e36_selection/
```

Primary decision:

```text
e36_pocket_ecology_selection_partial
```

Primary metrics:

```text
evaluated_selected_candidate          = protocol_framing_ingress_v001
promoted_pockets                      = protocol_framing_ingress_v001
banned_pockets                        = protocol_framing_no_adapter, wrong_rotated_codebook_pocket
deprecated_pockets                    = dormant_unused_pocket

evaluated_stable_target_success       = 1.000000
evaluated_target_world_success        = 0.746042
evaluated_bitslip_target_success      = 0.365104
evaluated_wrong_feature_write_rate    = 0.000000
evaluated_false_frame_commit_rate     = 0.005082

random_target_world_success           = 0.428750
scratch_target_world_success          = 0.969583
scratch_wrong_feature_write_rate      = 0.010122
unfiltered_target_world_success       = 0.969583
toxic_target_world_success            = 0.000000

deterministic_replay_match_rate       = 1.000000
target_checker_failure_count          = 0
sample_only_checker_failure_count     = 0
```

## Pocket Lifecycle Outcomes

```text
protocol_framing_ingress_v001:
  status = stable
  reason = safe stable-target transfer despite unsolved bit-slip family
  stable_target_success = 1.000000
  wrong_feature_write_rate = 0.000000
  bitslip_target_success = 0.365104

dirty_start_only_decoder:
  status = staging
  reason = useful but not stable/core
  target_world_success = 0.969583
  wrong_feature_write_rate = 0.010122
  false_frame_commit_rate = 0.026500

protocol_framing_no_adapter:
  status = banned
  reason = negative transfer or wrong commits
  target_world_success = 0.000000
  wrong_feature_write_rate = 0.094405

wrong_rotated_codebook_pocket:
  status = banned
  reason = negative transfer or wrong commits
  target_world_success = 0.000000
  wrong_feature_write_rate = 0.090242

dormant_unused_pocket:
  status = deprecated
  reason = no useful activations
```

The important detail is that the raw best unfiltered candidate was the dirty
scratch decoder, but the evaluator did not promote it as a stable/core reusable
library pocket. This is the intended distinction:

```text
raw local score != reusable clean pocket value
```

## Confirm Seeds

All confirm lanes reproduced the same decision.

```text
seed36002: decision=e36_pocket_ecology_selection_partial, stable=1.000000, target=0.731111, bitslip=0.327778, wrong=0.000000, random=0.413704, toxic=0.000000
seed36003: decision=e36_pocket_ecology_selection_partial, stable=1.000000, target=0.745926, bitslip=0.364815, wrong=0.000000, random=0.414815, toxic=0.000000
seed36004: decision=e36_pocket_ecology_selection_partial, stable=1.000000, target=0.751111, bitslip=0.377778, wrong=0.000000, random=0.432593, toxic=0.000000
seed36005: decision=e36_pocket_ecology_selection_partial, stable=1.000000, target=0.743333, bitslip=0.358333, wrong=0.000000, random=0.422963, toxic=0.000000
seed36006: decision=e36_pocket_ecology_selection_partial, stable=1.000000, target=0.735556, bitslip=0.338889, wrong=0.000000, random=0.418889, toxic=0.000000
seed36007: decision=e36_pocket_ecology_selection_partial, stable=1.000000, target=0.739630, bitslip=0.349074, wrong=0.000000, random=0.426667, toxic=0.008519
seed36008: decision=e36_pocket_ecology_selection_partial, stable=1.000000, target=0.745556, bitslip=0.363889, wrong=0.000000, random=0.421852, toxic=0.000000
seed36009: decision=e36_pocket_ecology_selection_partial, stable=1.000000, target=0.742593, bitslip=0.356481, wrong=0.000000, random=0.441481, toxic=0.000000
```

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

E36 supports this:

```text
A Pocket Library can score reusable pocket candidates across fresh worlds,
promote a clean transferable mechanism pocket, and quarantine toxic/stale
candidate pockets.
```

Specifically:

```text
ProtocolFramingIngressPocket becomes stable.
Wrong-codebook/no-adapter imports are banned.
Dormant/AFK pockets are deprecated.
Random library import is much worse than evaluated library import.
Unfiltered raw-score import selects the dirty scratch decoder, which is not
safe enough to become stable/core.
```

## What This Does Not Prove

E36 does not prove:

```text
full bitstream transfer solved
bit-slip tolerant stream reassembly
automatic discovery of new reusable pockets
raw language reasoning
AGI/consciousness/model-scale behavior
```

It also does not prove that the evaluated library is better than scratch on
raw target accuracy. Scratch still wins raw target success, but with dirtier
commit behavior and no clean reusable mechanism boundary.

## Scientific Interpretation

The main result is lifecycle discipline:

```text
Do not keep pockets because they won one run.
Keep pockets that survive cross-world safety and lifecycle selection.
```

This makes the Pocket Library direction viable, but only as a guarded ecology:

```text
mechanism pocket = reusable library candidate
adapter = local world-specific learnable part
dirty monolith = useful diagnostic, not stable/core library skill
toxic/AFK pockets = banned/deprecated
```

## Recommended Next Step

The next scientific bottleneck is still:

```text
bit-slip tolerant stream reassembly
```

Best next:

```text
E34E_BITSLIP_TOLERANT_STREAM_REASSEMBLY_PROBE
```

Secondary infrastructure next:

```text
E37_POCKET_ECOLOGY_CANDIDATE_GENERATION_AND_ARCHIVE_LOOP
```

That would test whether fresh runs can generate new candidate pockets, not just
score a fixed E35 candidate set.

Boundary: E36 is a controlled Pocket Ecology selection probe over symbolic
binary-ingress worlds. It does not prove raw language reasoning, AGI,
consciousness, deployed-model behavior, or model-scale behavior.
