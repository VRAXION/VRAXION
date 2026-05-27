# D43 Raw Support Evidence Extraction Scale Confirm Result

Status: positive.

D43 is a scale confirm of D42 with added cold-initialization, equality-kernel ablation, equality-kernel shuffle, no-prebaked-equality, and learned-input audits.

Artifact path: `target/pilot_wave/d43_raw_support_evidence_extraction_scale_confirm/smoke`

Decision: `raw_support_evidence_extraction_scale_confirmed`

Verdict: `D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_SCALE_CONFIRMED`

Next: `D44_FORMULA_PRIMITIVE_DISCOVERY_PLAN`

## Key Metrics

`MUTABLE_LEARNED_RAW_SUPPORT_EVIDENCE_EXTRACTOR`:

- train/test/OOD selected-pocket accuracy = `0.999375 / 0.9995833333333334 / 0.9992708333333333`
- rule-family train/test/OOD accuracy = `0.999375 / 0.9995833333333334 / 0.9992708333333333`
- min seed test/OOD = `0.9966666666666667 / 0.9975`
- min support-count accuracy = `0.9991666666666666`
- min margin-strata accuracy = `0.9977777777777778`

Controls:

- query-only test accuracy = `0.20114583333333333`
- shuffled-center test accuracy = `0.07802083333333333`
- shuffled-formula-candidate test accuracy = `0.0`
- no-center test accuracy = `0.2`
- no-formula-candidate test accuracy = `0.2`
- wrong-support selected-pocket test accuracy = `0.0`
- wrong-support follow rate = `1.0`
- same-query-different-raw-support accuracy = `0.9989583333333334`

Anti-shortcut audits:

- cold-init train accuracy mean = `0.2`
- cold-init solved seed count = `0`
- equality-kernel ablation test/OOD = `0.2 / 0.2`
- equality-kernel shuffle test/OOD = `0.08364583333333334 / 0.07708333333333334`
- equality-kernel argmax mapping = identity over symbols `0..8`
- channel-gate identity alignment = `1.0`

## Boundary

A positive D43 proves only that learned raw symbolic support-evidence extraction scale-confirms on a controlled symbolic task with fixed formula primitive candidates available.

It does not prove raw visual Raven reasoning, formula primitive discovery, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.

