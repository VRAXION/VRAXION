# STABLE_LOOP_PHASE_LOCK_118_REASONING_FIRST_RAW_ASSISTANT_REPAIR_CONTRACT

118 is a targeted research repair milestone after positive 117. It repairs the
first 116 breakpoint: `TIER_4_MULTI_STEP_REASONING`.

This is targeted research repair, not general training, not deploy polish, not
an architecture pivot, not production packaging, not GPT-like assistant
readiness, not open-domain assistant readiness, not production chat, not public
API, not deployment readiness, and not safety alignment.

## Scope

Add only:

- `scripts/probes/run_stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair.py`
- `scripts/probes/run_stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair_check.py`
- `docs/research/STABLE_LOOP_PHASE_LOCK_118_REASONING_FIRST_RAW_ASSISTANT_REPAIR_CONTRACT.md`
- `docs/research/STABLE_LOOP_PHASE_LOCK_118_REASONING_FIRST_RAW_ASSISTANT_REPAIR_RESULT.md`

Generated artifacts must be written only under
`target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/`.

The runner must not modify runtime, service, deploy, product, release, SDK,
public export, root `LICENSE`, existing checkpoint, bounded release artifact,
`instnct-core/`, or 083/089 package surfaces.

## Required Upstreams

118 requires positive upstreams:

- 117 `TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_POSITIVE`
- 116 `RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE`
- 115 `EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

117 must select `118_REASONING_FIRST_RAW_ASSISTANT_REPAIR`.

## Full Run

No tiny/dev substitute may emit a positive verdict. The full configured run is:

- seeds: `2121,2122,2123`
- steps: `12000`
- batch size: `64`
- sequence length: `256`
- train examples: `120000`
- eval rows per family: `64`
- FineWeb replay tokens: `1000000`
- rollout eval cadence: `50`
- heartbeat seconds: `20`

The runner must write partial outcomes throughout the run, including progress,
summary, report, training heartbeat, and rollout metrics.

## Repair Design

The training mix targets reasoning-first repair:

- 35% provided-fact multi-step reasoning
- 15% table + rule reasoning
- 10% small arithmetic over supplied values
- 10% contradiction resolution
- 10% rollout hard-negative / anti-memorization rows
- 8% bounded + finite-label retention
- 7% refusal / boundary / unsupported facts
- 5% FineWeb replay

Anti-111 safeguards are hard gates:

- scheduled sampling or rollout loss must actually run
- train/eval namespaces must be disjoint
- anti-memorization rows must be present
- leakage audit must compare against 112-117 artifacts
- final eval must be raw-only
- teacher-forcing-only success must fail

## Arms

Positive-scored arm:

- `POST_118_REASONING_REPAIRED_RAW`

Comparison/control arms:

- `PRE_118_RAW_BASELINE`
- `NO_ROLLOUT_OBJECTIVE_CONTROL`
- `GENERAL_SFT_ONLY_CONTROL`
- `STATIC_OUTPUT_CONTROL`
- `COPY_PROMPT_CONTROL`
- `RANDOM_REASONING_CONTROL`

Helper, decoder, oracle, expected-answer metadata, teacher forcing, LLM judge,
and verifier rerank paths are forbidden during final eval.

## Gates

Positive 118 requires actual repair work:

- `train_step_count > 0`
- `optimizer_step_count > 0`
- `target_118_checkpoint_changed = true`
- source 100 and 102 checkpoints unchanged
- bounded release artifact unchanged
- packaged winner hash unchanged
- final train loss lower than initial train loss
- raw rollout reasoning metrics improve

Reasoning gates:

- post Tier 4 reasoning accuracy at least `0.995`
- post Tier 4 reasoning accuracy at least pre + `0.005`
- post Tier 8 reasoning-combo accuracy at least `0.93`
- post Tier 8 reasoning-combo accuracy at least pre + `0.05`
- post reasoning failure count at most 25% of pre
- table+rule, small arithmetic, rule chaining, and contradiction gates pass

Regression gates:

- bounded and finite-label retention preserved
- unsupported refusal retention preserved
- namespace leak, teacher namespace copy, and case drift stay within threshold
- collapse rejected
- static/copy/random controls fail
- all overclaim and artifact exfiltration counts are zero

## Decisions

If all gates pass, write:

- verdict: `REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE`
- next: `119_REASONING_REPAIR_SCALE_CONFIRM`

Failure routes include:

- `118B_REASONING_REPAIR_PARTIAL_ANALYSIS`
- `118A_REASONING_TARGET_REVALIDATION`
- `118G_ROLLOUT_OBJECTIVE_FAILURE_ANALYSIS`
- `118R_RETENTION_OR_COLLAPSE_REGRESSION_ANALYSIS`
- `118L_REASONING_DATA_LEAKAGE_REDESIGN`
- `118C_BOUNDARY_FAILURE_ANALYSIS`

