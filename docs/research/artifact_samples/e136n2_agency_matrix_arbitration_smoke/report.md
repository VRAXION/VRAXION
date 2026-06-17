# E136N2 Agency Matrix Arbitration Smoke

```text
decision = e136n2_agency_matrix_arbitration_smoke_confirmed
next     = E136O_CHALLENGER_OOD_RUNTIME_REPLACEMENT_GAUNTLET
```

E136N2 tests a small trained Agency Matrix over the crystallized E136N
primary/secondary proposal surface. It compares matrix arbitration against a
first-valid sequential baseline and allows Flow chunk commits only when
compatible primary proposals support the same relation group.

## Result

```text
input_e136n_operator_count = 34
training_example_count = 118
training_epochs_completed = 2
training_converged_epoch = 2
training_final_epoch_updates = 0

case_count = 146
baseline_accuracy = 0.232877
agency_matrix_accuracy = 1.000000
baseline_unsafe_commit_count = 34
agency_matrix_unsafe_commit_count = 0

expected_child_check_count = 36
agency_matrix_child_check_count = 36
expected_flow_chunk_count = 10
agency_matrix_flow_chunk_count = 10

baseline_child_call_proxy = 336
agency_matrix_child_call_proxy = 202
child_call_proxy_reduction = 134
child_call_proxy_reduction_ratio = 0.398810

challenger_promoted_count = 0
lineage_hold_promoted_count = 0
destructive_delete_count = 0
```

## Interpretation

The existing Flow/Ground/Proposal/Agency structure does not need a separate
hand-written hierarchy registry for this smoke. A trained Agency Matrix can
arbitrate proposal bundles, hold challenger/lineage variants, reject unsafe
direct-write proposals, and commit compatible Flow chunks.

## Boundary

This is arbitration metadata over existing variants. It does not train a neural
model, discover new operators, promote held challengers, or destructively delete
anything.
