# E7A10 Binary Scale Overhead And Bit-Budget Audit Contract

## Purpose

E7A10 tests whether the E7A9 binary matrix-core result is a real low-bit path or mostly a scale-overhead artifact. The probe keeps the E7 matrix-core architecture fixed and compares binary scale policies against int4/int3/ternary baselines.

## Core Question

If binary receives either better scale policy or more hidden width under the same measured bit budget, does it match or beat the int4 width-32 reference?

## Systems

- `float32_matrix_core`
- `int8_direct`
- `int4_direct`
- `int3_direct`
- `ternary_block_scale_qat`
- `binary_direct_block_scale`
- `binary_minimal_scale_qat`
- `binary_global_scale_qat`
- `binary_block_scale_qat`
- `binary_channel_scale_qat`
- `binary_channel_scale_qat_paramwise_freeze`

## Required Metrics

- row-level train/validation/heldout/OOD/counterfactual/adversarial accuracy
- scale policy and stored scale bit cost
- total bit cost including stored float32 scales
- compression versus float32
- same-width comparison at reference width
- same-bit-budget comparison versus int4 reference
- accepted/rejected/rollback counts for mutation/freeze repair
- deterministic replay hash match

## Decision Rules

- `e7a10_binary_same_budget_preferred`: best binary within int4 bit budget beats the int4 reference.
- `e7a10_global_or_block_binary_viable`: global/block binary matches int4 within the threshold.
- `e7a10_binary_scale_overhead_required`: binary only matches when expensive scale overhead is used.
- `e7a10_binary_width_scaling_not_worth_it`: wider binary does not improve meaningfully over same-width binary.
- `e7a10_int4_quality_path_preferred`: int4 remains the better quality path.
- `e7a10_ternary_balanced_path_preferred`: ternary beats binary under the relevant budget.
- `e7a10_invalid_artifact_detected`: artifact contract, replay, leakage-style, or synthetic metric gate fails.

## Boundary

This is a controlled symbolic/numeric matrix-core compression audit. It makes no broad claim about natural language systems, model-scale behavior, or consciousness.
