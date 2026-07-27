Original prompt: jah okes akkor kezdetnek csinaljunk valamit itt. Lenyeg hogy legyen egy szep ablak amin latom mi tortenik - pl valami valahova mozog valami elol megprobal elmenni meg ilyenek, mennyire logikus utvonalakat valaszt, javul e barmi etc - az nem mindig kiolvashato outputbol.

## 2026-05-29

- Scope picked: build a local visual "colony arena" MVP under `tools/colony_arena`.
- Research direction: use a Screeps-like tick/intent separation, but keep the first prototype local and deterministic instead of integrating a Steam game.
- Sanity fixes before implementation:
  - Viewer must use the same simulation/trainer modules as smoke tests.
  - Smoke training must write partial artifacts continuously.
  - Training must evaluate multiple deterministic scenarios, not one lucky replay.
  - The first target is movement/pathing/avoidance learning, not full colony behavior.

## TODO

- Implemented deterministic arena sim, evolutionary policy trainer, web viewer, and smoke CLI.
- First smoke failed usefully: score improved +73.57 but test success stayed at 37.5%, so the reward was too permissive for safe non-goal orbiting. Adjusted goal radius and score shaping to make actual collection matter more.
- Smoke checks now pass:
  - `npm run smoke:fast` PASS, improvement +129.44, final test success 0.50, caught 0.00.
  - `npm run smoke` PASS, improvement +130.90, final test success 0.50, caught 0.00.
- Browser Playwright client initially timed out because the page kept auto-training during headless visual checks. Added `?test=1` mode to reduce population and disable auto-train for deterministic screenshot runs.
- Browser checks passed:
  - skill Playwright client produced canvas screenshots and `render_game_to_text` states under `target/colony_arena_browser_v2/`.
  - full-page Playwright screenshot rendered canvas + side panel + chart with no console errors.
  - Train/Pause/Reset controls changed state correctly.

## Next Suggestions

- Move the arena policy genome closer to INSTNCT-native graph/projection export.
- Add a replay loader for `sample_replay.json` so smoke artifacts can be inspected visually after a run.
- Add a second scenario with multiple workers/resources once movement/avoidance is stable above this smoke gate.

## 2026-05-30

- Continued from prompt: use the existing GUI sim plus the latest completed D51 brain to build a working adversarially checked sim.
- Ported the D51 `MUTABLE_RULE_TABLE_CONTROLLER` as a local evidence gate:
  - Game state -> D51-style confidence/margin/entropy/collision/support features.
  - D51 evidence action -> DECIDE / counter check / joint counter / external scan / abstain.
  - Evidence actions now accrue `infoCost` and are visible in `render_game_to_text`.
- Added adversarial scenario generation with threat-on-route and bait-wall layouts, separate from normal mixed test scenarios.
- Added D51 GUI visibility: brain action, info cost, adversarial scenario kind, evidence halo, and D51 feature dump.
- First D51 smoke attempts failed usefully:
  - Too much external/joint evidence caused very slow runs and low collection.
  - Recalibrated feature mapping so dominant/external evidence only fires on hazard/route-block signals.
  - Separated normal/mixed test from adversarial gate so the report tells two different things.
- Current smoke gate passed:
  - `npm run smoke:fast`
  - Output: `S:\Git\VRAXION\target\colony_arena_smoke\20260530_040319`
  - final_test_success_rate = 0.50
  - final_test_caught_rate = 0.00
  - final_adversarial_success_rate = 0.60
  - final_adversarial_caught_rate = 0.00
  - final_test_non_decide_rate = 0.7446

## TODO

- Browser/GUI verification passed:
  - Playwright client run wrote screenshots/state to `S:\Git\VRAXION\target\colony_arena_browser_d51`.
  - `full-page-fixed.png` visually checked: canvas and panel render at top of viewport, no overlapping controls, D51 metrics visible.
  - Console/page errors: none.
  - Pause/Resume, Train 1, Reset, `window.advanceTime`, and `window.render_game_to_text` were checked through Playwright.
- Longer training should use a lower-cost evaluator or fewer validation episodes per generation; the current smoke is correct but not cheap.
