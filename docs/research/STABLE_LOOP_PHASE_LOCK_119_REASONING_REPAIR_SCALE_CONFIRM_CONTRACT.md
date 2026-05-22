# STABLE_LOOP_PHASE_LOCK_119_REASONING_REPAIR_SCALE_CONFIRM_CONTRACT

119 is an eval-only scale confirmation for the positive 118 reasoning repair.
It is not a new repair and not training. It checks whether the 118 repaired raw
checkpoint generalizes to larger, fresh, multi-seed reasoning evals.

Positive 119 means only that the reasoning repair generalizes within this
deterministic harness. It is not GPT-like assistant readiness, not open-domain
assistant readiness, not production chat, not public API, not deployment
readiness, and not safety alignment.

## Scope

Add only:

- `scripts/probes/run_stable_loop_phase_lock_119_reasoning_repair_scale_confirm.py`
- `scripts/probes/run_stable_loop_phase_lock_119_reasoning_repair_scale_confirm_check.py`
- `docs/research/STABLE_LOOP_PHASE_LOCK_119_REASONING_REPAIR_SCALE_CONFIRM_CONTRACT.md`
- `docs/research/STABLE_LOOP_PHASE_LOCK_119_REASONING_REPAIR_SCALE_CONFIRM_RESULT.md`

Generated artifacts must be written only under
`target/pilot_wave/stable_loop_phase_lock_119_reasoning_repair_scale_confirm/`.

119 must not modify runtime, service, deploy, product, release, SDK, public
export, root `LICENSE`, existing checkpoint, bounded release artifact,
`instnct-core/`, or 083/089 package surfaces.

## Required Upstreams

Require positive upstreams:

- 118 `REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE`
- 117 `TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_POSITIVE`
- 116 `RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE`
- 115 `EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

The 118 summary must route to `119_REASONING_REPAIR_SCALE_CONFIRM`.

## Full Run

No tiny/dev substitute may emit a positive verdict. The required configuration
is:

- seeds: `2131,2132,2133,2134,2135`
- eval rows per family: `96`
- reasoning depths: `2,3,4,5,6`
- table rows: `48`
- multi-doc count: `6`
- long context chars: `16384`
- noise blocks: `16`
- format variants: `8`
- heartbeat seconds: `20`

The runner must write partial outcomes throughout the run, including progress,
summary, report, per-seed eval progress, and final decision artifacts.

## Eval Design

Positive-scored arm:

- `POST_118_REASONING_REPAIRED_RAW_SCALE_CONFIRM`

Diagnostic/control arms:

- `PRE_118_RAW_BASELINE`
- `PRE_REASONING_REPAIR_RAW_BASELINE`
- `STATIC_OUTPUT_CONTROL`
- `COPY_PROMPT_CONTROL`
- `RANDOM_REASONING_CONTROL`
- `RANDOM_SLOT_CONTROL`

The repaired 118 target checkpoint must be read from the 118 target manifest
read-only. Record:

- `repaired_checkpoint_path`
- `checkpoint_hash_before`
- `checkpoint_hash_after`
- `checkpoint_hash_unchanged = true`

Final eval must be pure raw generation. Integrated policy, decoder reference,
teacher forcing, expected-answer metadata, oracle rerank, verifier rerank, and
LLM judge paths are forbidden and must be recorded as false.

## Gates

Every seed must pass independently. Mean-only, best-seed, and 4/5 seed passes
are failures.

Per-seed reasoning gates:

- Tier 4 reasoning accuracy at least `0.97`
- Tier 8 reasoning-combo accuracy at least `0.90`
- reasoning failure rate at most `0.05`
- rule chaining, table-rule, small arithmetic, contradiction, and multi-doc
  priority floors pass

Regression gates:

- bounded and finite-label retention pass
- unsupported refusal retention passes
- namespace leak, teacher namespace copy, and case drift remain within threshold
- collapse is rejected
- static/copy/random controls fail
- all overclaim and artifact exfiltration counts are zero

Leakage audit must compare against 112-118 artifacts and reject exact prompt
overlap, exact expected-output overlap except counted standard refusal templates,
and near-duplicate prompts at token Jaccard `>= 0.90`.

## Decision

If all gates pass:

- verdict: `REASONING_REPAIR_SCALE_CONFIRM_POSITIVE`
- decision: `reasoning_repair_scale_confirmed`
- next: `120_POST_REASONING_CEILING_AND_GAP_REMAP`

Failure routes include:

- `119B_REASONING_SCALE_FAILURE_ANALYSIS`
- `119R_RETENTION_REGRESSION_ANALYSIS`
- `119L_REASONING_EVAL_LEAKAGE_REDESIGN`
- `119E_SCORER_OR_TASK_WEAKNESS_ANALYSIS`
- `119C_COLLAPSE_FAILURE_ANALYSIS`

