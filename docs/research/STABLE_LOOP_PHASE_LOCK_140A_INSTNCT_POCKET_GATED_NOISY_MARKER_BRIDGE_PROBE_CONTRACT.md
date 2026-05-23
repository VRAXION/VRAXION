# STABLE_LOOP_PHASE_LOCK_140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE Contract

140A is the executable bridge probe after 139YS.

It tests whether the existing pocket-gated INSTNCT helper manifest path still works when prompt scaffolding is less sterile: fewer direct `POCKET_VALUE=` rows, more `POCKET_BIND=` / `POCKET_TABLE_ROW=` rows, visible distractors, noisy distractor values, and natural-ish carrier text around the pocket payload.

## Boundary

140A may call `scripts/probes/shared_raw_generation_helper.py` for final helper-only eval.

140A must not train, mutate source checkpoints, modify `shared_raw_generation_helper.py`, modify backend/runtime/release/product surfaces, change public request keys, start services, deploy, or change root `LICENSE`.

140A is not GPT-like readiness, not broad assistant capability, not production readiness, not public API readiness, not deployment readiness, and not safety alignment.

## Required Upstream

140A requires 139YS:

```text
decision = real_task_bridge_recommended
next = 140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE
```

The 139YS target plan must require reduced explicit markers, noisy distractors, natural-ish value carrier text, a required pocket gate, visible bypass rejection, closed-pocket ablation failure, mutation selection, helper/canary/AST/leakage/controls/determinism gates, and clean-negative routes.

## Positive Gates

Positive 140A requires:

- main answer value accuracy at least `0.80`
- main pocket writeback rate at least `0.90`
- closed-pocket ablation answer value accuracy at most `0.10`
- pocket ablation delta at least `0.50`
- reduced-marker row rate at least `0.60`
- direct `POCKET_VALUE=` marker rate at most `0.40`
- visible bypass control fails
- noisy distractor control fails
- deterministic replay passes
- expected-output canary passes
- AST shortcut scan passes
- generated text exists before scoring
- helper requests use only public allowed keys

## Clean Negative Routes

140A clean negatives:

- `marker_dependency_too_strong -> 140B_MARKER_DEPENDENCY_ANALYSIS`
- `pocket_ablation_not_decision_critical -> 140C_POCKET_CAUSALITY_FAILURE_ANALYSIS`
- `noisy_prompt_breaks_value_binding -> 140D_NOISY_PROMPT_VALUE_BINDING_ANALYSIS`
- `mutation_search_fails_to_select_open_pocket -> 140E_MUTATION_SELECTION_FAILURE_ANALYSIS`
- `helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL`

If positive:

```text
decision = instnct_pocket_gated_noisy_marker_bridge_probe_positive
next = 140F_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRM
```

The positive is constrained pocket-mechanism evidence only. It is not a general value-grounding or broad architecture-superiority claim.
