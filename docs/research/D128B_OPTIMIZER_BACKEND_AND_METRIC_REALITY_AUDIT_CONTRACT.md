# D128B Optimizer Backend and Metric Reality Audit Contract

Purpose: audit the current D-series probe runners and artifacts to determine whether reported training, optimizer, mutation, and metric-improvement language is backed by real parameter optimization or by deterministic synthetic report harness logic.

Boundary: D128B is audit-only. It performs no training, no repair, no model update, no adapter mutation, no dataset mutation, no natural-language pretraining, no tokenizer or next-token objective, no raw text corpus, no raw Raven work, no Gemma-class training, and no AGI or production-readiness claim.

Scope: inspect D100-D128X probe runners/checkers, D120-D128X runners/checkers, relevant probe utilities, recent target/pilot_wave artifacts when present, and research docs/results when needed for claim-vs-code comparison.

Audit questions: identify optimizer/backprop imports and calls, mutation/evolution operators and invocations, parameter tensor or checkpoint diffs, hardcoded/static metric dictionaries, seeded deterministic formulas, row-level prediction loops, artifact replay, and checker-only validation behavior.

Required outputs: D128B writes static code inventory, backend classification, claim-vs-code, metric source, parameter diff, mutation algorithm, gradient/backprop, synthetic harness, deterministic replay, aggregate metrics, decision, summary, and report artifacts.

Decision target: d128b_synthetic_harness_backend_confirmed -> D128_CONTROLLED_SYMBOLIC_BRIDGE_FRONTIER_CONSOLIDATION_WITH_BACKEND_BOUNDARY if the audited D-series runners are synthetic deterministic report harnesses and no real optimizer, mutation backend, gradient backend, or parameter-diff evidence is found.

Required wording discipline: D128B must state explicitly when metrics are hardcoded, formulaic, replayed, or synthetic and must not describe adapter-only training as actual trainable tensor updates unless code evidence proves real parameter updates.
