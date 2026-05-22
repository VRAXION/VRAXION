# STABLE_LOOP_PHASE_LOCK_120_POST_REASONING_CEILING_AND_GAP_REMAP_CONTRACT

120 is an eval-only ceiling/gap remap after positive 119. It remaps the
capability ceiling after the 118 reasoning repair and 119 scale confirmation to
find the new first breakpoint.

Positive 120 means the post-reasoning ceiling/gap map is complete. It is not
GPT-like assistant readiness, not open-domain assistant readiness, not
production chat, not public API, not deployment readiness, not safety alignment,
and not Hungarian assistant readiness.

## Scope

Add only:

- `scripts/probes/run_stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap.py`
- `scripts/probes/run_stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap_check.py`
- `docs/research/STABLE_LOOP_PHASE_LOCK_120_POST_REASONING_CEILING_AND_GAP_REMAP_CONTRACT.md`
- `docs/research/STABLE_LOOP_PHASE_LOCK_120_POST_REASONING_CEILING_AND_GAP_REMAP_RESULT.md`

Generated artifacts must be written only under
`target/pilot_wave/stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap/`.

120 must not modify runtime, service, deploy, product, release, SDK, public
export, root `LICENSE`, existing checkpoint, bounded release artifact,
`instnct-core/`, or 083/089 package surfaces.

## Required Upstreams

Require positive upstreams:

- 119 `REASONING_REPAIR_SCALE_CONFIRM_POSITIVE`
- 118 `REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE`
- 116 `RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE`
- 115 `EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE`
- 112 `CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE`
- 099 `BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE`

The 119 summary must route to `120_POST_REASONING_CEILING_AND_GAP_REMAP`.

## Full Run

No tiny/dev substitute may emit a positive verdict. The required configuration
is:

- seeds: `2141,2142,2143,2144`
- rows per family per tier: `48`
- max context chars: `49152`
- noise blocks: `48`
- format variants: `16`
- table rows: `96`
- multi-doc count: `10`
- multi-turn depth: `8`
- heartbeat seconds: `20`

The runner must write partial outcomes throughout the run, including progress,
summary, report, checkpoint provenance, dataset build, leakage audit, tier/seed
eval blocks, aggregate analysis, decision writing, and final verdict.

## Eval Design

Positive-scored arm:

- `POST_119_REASONING_REPAIRED_CEILING_MAP`

Controls:

- `PRE_REASONING_REPAIR_BASELINE`
- `STATIC_OUTPUT_CONTROL`
- `COPY_PROMPT_CONTROL`
- `RANDOM_FACT_CONTROL`
- `RANDOM_SLOT_CONTROL`

The 118 repaired checkpoint provenance must be read through the 119/118 manifests
read-only. The runner records:

- `repaired_checkpoint_path`
- `checkpoint_hash_before`
- `checkpoint_hash_after`
- `checkpoint_hash_unchanged = true`

Use the configured eight tiers and seventeen families. Hungarian remains
diagnostic only unless it causes UTF-8 corruption, collapse, retention
regression, overclaim, or exfiltration.

Scoring is deterministic only: exact values, regex, JSON/schema validity,
slot/case correctness, refusal markers, provided-fact grounding, and collapse
metrics. No LLM judge, no subjective scoring, and no current-world facts.

## Gates

Positive 120 does not require perfect accuracy. It requires:

- all artifacts written
- all tiers evaluated
- post-reasoning failure map complete
- first breakpoint identified or `ceiling_not_reached_within_config` recorded
- `unknown_failure_rate <= 0.10`
- reasoning regression rejected
- retention preserved
- collapse rejected
- controls fail
- leakage rejected
- checkpoint hash unchanged
- all overclaim/exfiltration counts are zero

Reasoning regression is a hard failure:

- Tier 4 reasoning accuracy at least `0.97`
- Tier 8 reasoning-combo accuracy at least `0.90`
- reasoning failure rate at most `0.05`

Every failed row must have one primary label from the allowed failure taxonomy.

## Decision

If all gates pass:

- verdict: `POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE`
- decision: `post_reasoning_ceiling_gap_map_complete`
- next: `121_TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN`

Failure routes include:

- `120R_REASONING_REGRESSION_ANALYSIS`
- `120L_BENCHMARK_LEAKAGE_REDESIGN`
- `120E_SCORER_OR_TASK_WEAKNESS_ANALYSIS`
- `120T_RETENTION_REGRESSION_ANALYSIS`
- `120C_BOUNDARY_FAILURE_ANALYSIS`
- `120B_FAILURE_MAP_INCOMPLETE_ANALYSIS`

