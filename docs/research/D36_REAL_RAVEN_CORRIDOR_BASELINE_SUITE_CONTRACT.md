# D36 Real Raven Corridor Baseline Suite Contract

D36 implements a real (non-synthetic) baseline suite after D35.

- Fix D35 duplicate-target risk by enforcing exactly one target-containing pocket.
- Keep label rule identical across train/test/OOD; OOD uses held-out permutations/templates, not a changed answer rule.
- Run layered arms:
  - TARGET_GIVEN_ROUTING
  - RULE_KNOWN_ROUTING
  - RULE_HIDDEN_ROUTING
- Required methods: random_baseline and direct_mutation.
- Optional methods are marked unavailable unless real implementation exists.

Non-claims:
- no solved claim
- no architecture superiority claim
- no natural-language reasoning claim
