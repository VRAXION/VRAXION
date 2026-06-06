# E7A12 Binary Mutation-Only Viability Audit Contract

## Purpose

E7A12 tests whether binary matrix-core can work through mutation-only search, rather than relying on backprop/QAT as the main training path.

## Core Questions

- Can random binary matrix-core be solved from scratch by mutation-only search?
- Can QAT-seeded binary be improved by local mutation-only repair?
- Can progressive-freeze seeded binary be improved by mutation-only repair?
- Does mutation-only work only locally around a good seed, or can it discover a solution?

## Systems

- `float32_matrix_core_reference`
- `int4_reference`
- `binary_qat_reference`
- `random_binary_from_scratch_mutation`
- `sensitivity_guided_binary_from_scratch_mutation`
- `qat_seeded_binary_local_mutation`
- `progressive_freeze_seeded_binary_local_mutation`
- `binary_mutation_with_scale_only`
- `binary_mutation_bits_plus_scale`
- `random_mutation_control`

## Mutation Operators

- single-bit flip
- k-bit flip
- block flip
- row/channel flip
- targeted flip using train/validation error proxy
- scale mutation where scale state exists
- bits plus scale mutation

## Required Metrics

- heldout/OOD/counterfactual/adversarial accuracy
- eval average
- improvement over seed
- gap to QAT
- gap to int4
- accepted/rejected/rollback counts
- mutation attempts
- accepted/rejected counts by operator
- local search radius proxy
- deterministic replay hash match

## Decision Rules

- `e7a12_binary_mutation_from_scratch_viable`: from-scratch binary mutation solves and stays close to QAT.
- `e7a12_binary_local_mutation_repair_viable`: from-scratch fails but local mutation around a good seed works.
- `e7a12_progressive_seed_mutation_bridge_viable`: progressive-freeze seeded mutation improves enough to count as a bridge.
- `e7a12_binary_scale_mutation_only_positive`: scale-only mutation gives the meaningful improvement.
- `e7a12_binary_mutation_repair_no_advantage`: mutation-only cannot improve good seeds.
- `e7a12_mutation_policy_artifact_or_task_too_easy`: random-control mutation matches guided mutation.
- `e7a12_invalid_artifact_detected`: checker, replay, row-level, or mutation-only policy gate fails.

## Boundary

This is a controlled binary matrix-core mutation audit. It does not make claims about natural-language reasoning, AGI, consciousness, or model-scale behavior.
