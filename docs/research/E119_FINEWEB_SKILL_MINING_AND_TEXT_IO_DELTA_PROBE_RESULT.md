# E119 FineWeb Skill Mining And Text IO Delta Probe Result

```text
decision = e119_fineweb_skill_mining_positive
checker_failure_count = 0
```

Boundary:

```text
not Gemma-style language model
not free-form next-token chatbot
not final training
not PermaCore
not TrueGolden
```

## Key Metrics

```text
rows_seen = 100000
operator_count = 136
actual_300k_operator_count = 77

current_text_io_accuracy = 1.000000
legacy_text_io_accuracy  = 0.482290
none_text_io_accuracy    = 0.000000
current_minus_legacy     = +0.517710

current_hard_negative_count = 0
skill_candidate_count = 11
farm_candidate_count = 8
```

The current E118 Operator Library substantially improves canonical text-IO
action selection over the older grounding-only subset. The improvement comes
from the E100-E106 evidence/answer/progress operators, not from free-form
language generation.

## Text IO Distribution

```text
ANSWER_WITH_TRACE              = 25224
ASK_FOR_EVIDENCE              = 2314
DEFER_CONFLICT                = 37325
DEFER_UNSAFE                  = 33
OBSERVE_AND_GROUND            = 10871
UPDATE_PROGRESS_LEDGER        = 1823
VALIDATE_VISIBLE_CALC_TRACE   = 22410
```

Current library action accuracy was 1.0 on this controlled canonical action
classification harness. Legacy grounding subset stayed useful for observation,
conflict, and unsafe-defer cases, but could not cover answer-with-trace,
ask-for-evidence, progress-ledger, or visible-calc validation actions.

## Top Farm Candidates

```text
definition_term_anchor_lens
  support = 94350
  current coverage = 0.00

named_entity_anchor_lens
  support = 86231
  current coverage = 0.00

causal_relation_lens
  support = 37516
  current coverage = 0.00

date_entity_timeline_lens
  support = 56220
  current coverage = 0.50

comparison_quantifier_guard
  support = 23899
  current coverage = 0.19

procedure_step_parser_lens
  support = 42870
  current coverage = 0.50

safety_domain_caution_guard
  support = 16352
  current coverage = 0.00

quote_speaker_attribution_lens
  support = 13314
  current coverage = 0.50
```

Already-covered watch candidates:

```text
document_structure_lens
hedge_uncertainty_strength_t_stab
number_unit_grounding_alpha_syncer
```

## Generation Readiness

```text
grounded_canonical_text_io_ready = true
freeform_gemma_style_generation_ready = false
```

The system can select trace-aware canonical actions:

```text
ANSWER_WITH_TRACE
ASK_FOR_EVIDENCE
DEFER_CONFLICT
OBSERVE_AND_GROUND
VALIDATE_VISIBLE_CALC_TRACE
UPDATE_PROGRESS_LEDGER
```

It still lacks the pieces needed for a Gemma-style assistant:

```text
open-vocabulary semantic compression
free-form fluent decoder
broad world knowledge memory
long-context discourse planner
```

## Interpretation

E119 confirms that the current skill library is meaningfully better at safe,
trace-aware FineWeb text IO than the earlier grounding-only state. It does not
prove general natural-language understanding or fluent generation. The best
next step is to farm the eight high-support FineWeb candidates as scoped
Operators, starting with definition/term anchoring and named-entity anchoring,
because they appear constantly in real text and are currently under-covered.

## Artifacts

```text
target/pilot_wave/e119_fineweb_skill_mining_and_text_io_delta_probe/
```
