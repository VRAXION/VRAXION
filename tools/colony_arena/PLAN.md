# Colony Arena Plan

## Research Input

- Screeps uses a tick loop: the player script reads current state, issues intents,
  and the world applies the resulting changes on the next tick. This supports a
  clean separation between observation, decision, and world update.
- Screeps script modules can be organized like Node modules and can use
  WebAssembly. That makes a future VRAXION policy export plausible, but not
  necessary for the first local prototype.
- Quality-diversity / MAP-Elites style work supports treating behavior as an
  evolvable object evaluated across many scenarios instead of one scripted demo.

## First Prototype Decision

Build a local deterministic arena, not a Steam-game adapter yet.

```text
world state -> D51 evidence gate -> policy intent -> world tick -> metrics/replay
```

The first behavior is movement:

- reach green goals,
- avoid red threat,
- avoid walls,
- leave visible route traces,
- train/evaluate over multiple deterministic scenarios.

The current D51 bridge ports the latest completed D51 best arm,
`MUTABLE_RULE_TABLE_CONTROLLER`, as an evidence gate. It maps arena state to
confidence/margin/entropy/collision/support features, then chooses DECIDE,
counter checks, joint counter checks, external route scans, or ABSTAIN before a
movement action is applied.

## Adversarial Sanity Checks

- **Animation-only failure**: avoided by sharing `src/sim.js` between viewer and
  smoke trainer.
- **Lucky-seed failure**: avoided by evaluating a scenario set, not one replay.
- **Black-box run failure**: smoke writes `queue.json`, per-generation
  `progress.jsonl`, `partial_summary.json`, final summary, policy, replay, and
  report.
- **Reward hacking failure**: first smoke exposed safe non-goal orbiting. Fixed
  by increasing real goal collection weight and adding a no-goal distance
  penalty.
- **Visual-test overload**: headless browser initially timed out due auto-train.
  Fixed with `?test=1` mode that disables auto-train and uses a smaller
  deterministic population for screenshots.
- **D51 over-query failure**: first D51 port requested extra evidence on almost
  every tick and slowed smoke runs. Fixed by making dominant/external features
  hazard-sensitive and by selecting the best held-out validation policy rather
  than the last generation.

## Current Gates

```text
npm run smoke
```

Pass conditions:

- score improvement >= 4.0,
- final test success rate >= 0.50,
- final test caught rate <= 0.55,
- final adversarial success rate >= 0.35,
- final adversarial caught rate <= 0.60,
- final D51 non-DECIDE rate >= 0.05.

Current verified result:

```text
PASS
improvement = +109.1590
final_test_success_rate = 0.50
final_test_caught_rate = 0.00
final_adversarial_success_rate = 0.60
final_adversarial_caught_rate = 0.00
final_test_non_decide_rate = 0.7446
```
