# STABLE_LOOP_PHASE_LOCK_140F_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRM Contract

140F is the scale confirm after the positive 140A noisy-marker bridge probe.

It tests whether the 140A result survives more rows, more seeds, more families, more scaffold variants, more noisy distractors, more reduced-marker variants, and more partial-candidate ablations.

## Boundary

140F may call `scripts/probes/shared_raw_generation_helper.py` for helper-only eval.

140F must not train, mutate source checkpoints, modify `shared_raw_generation_helper.py`, modify backend/runtime/release/product surfaces, change public request keys, start services, deploy, or change root `LICENSE`.

140F is not GPT-like readiness, not broad assistant capability, not production readiness, not public API readiness, not deployment readiness, and not safety alignment.

## Required Upstream

140F requires 140A:

```text
decision = instnct_pocket_gated_noisy_marker_bridge_probe_positive
next = 140F_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRM
```

140A must have passed the noisy bridge gates: high main accuracy, high pocket writeback, closed-pocket ablation failure, visible bypass control failure, noisy distractor control failure, reduced-marker dominance, and deterministic replay.

## Positive Gates

Positive 140F requires:

- eval rows at least `2000`
- family count at least `6`
- scaffold variant count at least `20`
- main answer value accuracy at least `0.95`
- main exact answer accuracy at least `0.95`
- main pocket writeback rate at least `0.95`
- main contrast group accuracy at least `0.95`
- closed-pocket ablation answer value accuracy at most `0.05`
- pocket ablation delta at least `0.90`
- reduced-marker row rate at least `0.85`
- direct `POCKET_VALUE=` marker rate at most `0.15`
- visible bypass violation rate `0.0`
- noisy distractor violation rate `0.0`
- visible bypass control fails
- noisy distractor control fails
- every seed independently passes
- deterministic replay passes
- expected-output canary passes
- AST shortcut scan passes
- generated text exists before scoring
- helper requests use only public allowed keys

## Clean Negative Routes

140F clean negatives:

- `marker_dependency_too_strong -> 140B_MARKER_DEPENDENCY_ANALYSIS`
- `pocket_ablation_not_decision_critical -> 140C_POCKET_CAUSALITY_FAILURE_ANALYSIS`
- `noisy_prompt_breaks_value_binding -> 140D_NOISY_PROMPT_VALUE_BINDING_ANALYSIS`
- `mutation_search_fails_to_select_open_pocket -> 140E_MUTATION_SELECTION_FAILURE_ANALYSIS`
- `helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`

If positive:

```text
decision = instnct_pocket_gated_noisy_marker_bridge_scale_confirmed
next = 140G_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PLAN
```

The positive is constrained pocket-mechanism scale evidence only. It is not a general value-grounding, broad assistant, or architecture-superiority claim.
