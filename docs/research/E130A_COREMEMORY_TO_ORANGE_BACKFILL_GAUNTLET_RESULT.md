# E130A CoreMemoryCandidate To Orange Backfill Gauntlet Result

```text
decision = e130a_corememory_to_orange_backfill_confirmed
next = E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET
boundary = scoped Operator rank backfill only; not PermaCore or TrueGolden

candidate_count = 136
orange_legendary_candidate_count = 136
qualified_activation_before_total = 13,877,699
qualified_activation_add_total = 27,158,734
qualified_activation_total = 41,036,433
qualified_activation_min = 300,623
family_coverage_min = 20
campaign_count_min = 8

hard_negative_total = 0
false_commit_total = 0
wrong_scope_call_total = 0
unsupported_answer_total = 0
negative_transfer_total = 0
direct_flow_write_total = 0

reload_match_rate = 1.000000
negative_scope_pass_rate = 1.000000
challenger_pass_rate = 1.000000
prune_pass_rate = 1.000000
mean_selected_prune_ratio = 0.746176
deterministic_replay_pass = true
checker_failure_count = 0
```

## Summary

E130A tests the exact condition the rank dashboard exposed: the E112 lila
CoreMemoryCandidate pool was below Orange/Legendary activation threshold even
though its safety, coverage, reload, challenger, and prune evidence was already
clean.

The run backfills the missing activation pressure and re-checks the stricter
Orange/Legendary gate. All 136 prior CoreMemoryCandidate Operators reached
OrangeLegendaryCandidate without hard negatives, wrong-scope calls,
unsupported answers, negative transfers, or direct Flow writes.

## What Was Learned

The strongest interpretation is:

```text
The prior CoreMemoryCandidate pool can be promoted to scoped
OrangeLegendaryCandidate when it receives enough additional qualified
activation evidence and still passes the E121-style no-harm, reload,
negative-scope, challenger, and prune gates.
```

This is not just a label change. E130A requires:

```text
activation backfill
family/campaign coverage
mutation pressure
selected prune ratio >= 0.60
negative-scope pass
reload shadow pass
challenger pass
direct-write zero
deterministic replay
checker failure_count = 0
```

## Dashboard State

After merging E130A into the dashboard payload:

```text
Dashboard operator count = 530
Orange/LegendaryCandidate = 527
CoreMemoryCandidate = 0
Deprecated = 3
E130A rows = 136
```

## What Is Not Claimed

E130A does not claim:

```text
PermaCore
TrueGolden
production assistant behavior
open-domain LLM/chatbot readiness
Gemma/GPT-like generation
natural-language word-problem solving
final training completion
trained model weights
```

## Artifacts

```text
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/report.md
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/summary.json
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/decision.json
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/operator_orange_results.json
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/variant_report.json
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/backfill_report.json
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/mutation_summary.json
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/deterministic_replay.json
docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet/row_level_samples.jsonl
```

## Reproduce

```powershell
python scripts/probes/run_e130a_corememory_to_orange_backfill_gauntlet.py --out target/pilot_wave/e130a_corememory_to_orange_backfill_gauntlet --sample-out docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet
```
