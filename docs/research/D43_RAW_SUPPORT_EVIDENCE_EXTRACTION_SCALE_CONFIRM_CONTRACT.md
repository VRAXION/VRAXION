# D43 Raw Support Evidence Extraction Scale Confirm Contract

## Goal

D43 scale-confirms the D42 raw symbolic support-evidence extractor on a controlled symbolic pocket task.

The bounded pipeline is:

raw symbolic support boards with completed centers -> learned equality/evidence extractor -> rule-family scores -> previously confirmed support-rule-router path -> query formula channel -> selected target pocket.

## Boundary

A positive D43 proves only that learned raw symbolic support-evidence extraction scale-confirms on a controlled symbolic task with fixed formula primitive candidates available.

It does not prove raw visual Raven reasoning, formula primitive discovery, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.

## Dataset

Each row contains raw support boards, one query board with hidden target, nine candidate pockets, one unique target pocket, and an intended family in `{row,col,pair,mirror,diag}`.

Family formulas:

- `row`: `(b[1][0] + b[1][2]) % 9`
- `col`: `(b[0][1] + b[2][1]) % 9`
- `pair`: `(b[0][0] + b[2][2]) % 9`
- `mirror`: `(b[2][0] + b[0][2]) % 9`
- `diag`: `(b[0][0] + b[1][2] + b[2][1]) % 9`

Support counts are mixed across `1,2,3,5`. Support strata include clean/high/low/noisy majority cases, with ambiguous rows rejected.

## Anti-Shortcut Audits

D43 adds explicit audits for the D42 caveat:

- `cold_init_accuracy_report.json`
- `initial_equality_kernel_report.json`
- `equality_kernel_ablation_report.json`
- `equality_kernel_shuffle_report.json`
- `no_prebaked_equality_audit.json`
- `learned_extractor_input_audit.json`

The learned extractor must not receive a precomputed support-evidence vector, a boolean equality feature, the true family label, the expected selected pocket, or the query target.

The learned equality kernel is used as a row-normalized learned match distribution over formula-candidate values. This keeps the extractor learned and raw-symbolic while avoiding a magnitude shortcut where one learned symbol diagonal can dominate support-count evidence solely because of scale.

## Positive Decision Gate

Positive decision requires passing dataset invariants and OOD oracle audits, learned raw extractor test/OOD and rule-family test/OOD at least `0.98`, min seed test/OOD at least `0.95`, min support-count and margin-strata accuracy at least `0.95`, controls collapsed, counterfactual controls passed, wrong-support behavior causal, and equality-kernel ablation/shuffle producing a strong drop.
