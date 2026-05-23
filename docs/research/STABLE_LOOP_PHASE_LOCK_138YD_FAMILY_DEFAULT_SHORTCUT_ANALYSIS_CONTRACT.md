# STABLE_LOOP_PHASE_LOCK_138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS Contract

138YD is artifact-only analysis after the completed 138YH high-frequency replay forensics.

Required upstream route:

```text
138YH decision = high_frequency_value_replay_analysis_complete
138YH root_cause = family_default_shortcut_replay
138YH next = 138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS
```

138YD investigates why `ANSWER=E` rollout falls into family-default wrong values. It must distinguish template-induced shortcut, objective weakness, contrastive-objective weakness, dataset diversity weakness, scorer weakness, and model output-behavior attractor. It must not claim global memorized lookup, top-k train frequency replay, measured output-head/logit prior, hidden-state mechanism, grower behavior, or scout behavior unless explicit artifacts support that claim.

This phase is artifact-only. It does not train, repair, run new inference, call `shared_raw_generation_helper.py`, run torch forward passes, mutate checkpoints, modify helper/backend code, import old runners, delete or consolidate files, start services, deploy, modify runtime/service/deploy/product/release surfaces, modify `docs/product`, modify `docs/releases`, modify SDK exports, or change root `LICENSE`.

The generated reports must include family default mapping, default value origin, template shortcut analysis, contrast-group default failure analysis, objective shortcut reward analysis, scorer/dataset shortcut analysis, root cause, next recommendation, diagnostic gaps, risk register, decision, summary, and report.

Root cause must be exactly one of:

- `template_induced_family_default_shortcut`
- `objective_allows_family_default_shortcut`
- `contrastive_objective_too_weak`
- `dataset_low_intra_family_value_diversity`
- `scorer_family_default_weakness`
- `model_family_default_attractor_output_behavior`
- `mixed_family_default_shortcut`
- `family_default_shortcut_ambiguous`

All capability flags remain false: reasoning restored, reasoning subtrack partial evidence, raw assistant, structured/tool, GPT-like, open-domain, production chat, public API, deployment, and safety alignment.

Raw assistant capability remains quarantined. Structured/tool capability remains invalidated. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
