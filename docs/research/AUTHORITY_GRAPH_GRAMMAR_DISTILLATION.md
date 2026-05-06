# Authority Graph Grammar Distillation

## Goal

The developmental search and quick authority-gradient A/B test showed that the hand-seeded authority graph works, while random and weak grammar priors do not reliably rediscover the mechanism. Authority-shaped fitness did not fix the failure.

This note distills what is present in the working hand-seeded and damaged-success graphs, but missing or unreliable in the failed grammar arms. It does not run a new search and does not claim that the proposed Grammar v2 works yet.

Sources:

- `scripts/run_authority_graph_pilot.py`
- `scripts/run_authority_graph_developmental_search.py`
- `docs/research/AUTHORITY_GRAPH_MINIMALITY_EVOLUTION.md`
- `docs/research/AUTHORITY_GRAPH_DEVELOPMENTAL_SEARCH.md`
- `docs/research/AUTHORITY_GRADIENT_SEARCH_QUICK_TEST.md`
- `target/context-cancellation-probe/authority-graph-developmental-search/`
- `target/context-cancellation-probe/authority-gradient-search-quick/`

Machine-readable summary:

- `target/context-cancellation-probe/authority-graph-grammar-distillation/authority_graph_grammar_distillation_summary.json`

## Prior Result

Final-test summary from the developmental search:

| Graph class | Accuracy | Temporal | Authority | Wrong-frame drop | Edges |
|---|---:|---:|---:|---:|---:|
| `hand_seeded` best | `0.991536` | `1.000000` | `0.358725` | `0.359375` | `91` |
| `damaged_hand_seeded_50` best | `0.966146` | `1.000000` | `0.298404` | `0.363281` | `51` |
| `route_gate_grammar_graph` best | `0.832031` | `0.625000` | `0.227883` | `0.304688` | `37` |
| `route_gate_recurrence_grammar` best | `0.820964` | `0.750000` | `0.079391` | `0.300781` | `43` |
| `route_gate_hub_grammar` best | `0.833333` | `0.875000` | `0.089167` | `0.242188` | `57` |
| `route_grammar_graph` best | `0.723958` | `0.500000` | `0.089758` | `0.230469` | `24` |
| `random_graph` best | `0.820964` | `0.625000` | `0.065436` | `0.250000` | `102` |

The strongest damaged graph is important: with only `51` edges it nearly reaches the hand graph. That means the full hand graph is not minimal, but the surviving edges are highly structured.

## Implementation Caveat

The current pilot contains `readout_positive` and `readout_negative` nodes, but the static task score is read directly from the active route state in `static_score(...)`. In other words, static authority readout is currently route-state readout, not an explicit route-to-readout edge circuit.

This matters for Grammar v2: either keep route-state readout as the formal mechanism, or add explicit route-to-readout edges and make the evaluator use them. The current evidence supports route-state authority, not yet a separate readout-node circuit.

## Node Inventory

All analyzed graphs share the same node scaffold:

| Node type | Count |
|---|---:|
| `token_input` | `23` |
| `shared_hub` | `3` |
| `frame_route` | `5` |
| `suppressor` | `4` |
| `temporal_role` | `11` |
| `readout` | `2` |
| total | `48` |

The difference is therefore not node availability. The failure is in edge coverage, edge signs, recurrent/temporal wiring, and suppressor routing.

## Edge Inventory

| Graph | Edges | Token->route | Token->hub | Hub->route | Route self-loop | Suppressor->route | Temporal role edges | Token->readout | Route->readout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hand_seeded` | `91` | `29` | `17` | `12` | `4` | `16` | `12/12 aligned` | `0` | `0` |
| `damaged_hand_seeded_50` | `51` | `16` | `9` | `6` | `2` | `8` | `7/7 aligned` | `0` | `0` |
| `route_gate_grammar_graph` | `37` | `19` | `1` | `4` | `3` | `3` | `2/3 aligned` | `0` | `0` |
| `route_gate_recurrence_grammar` | `43` | `19` | `0` | `3` | `4` | `4` | `3/7 aligned` | `0` | `0` |
| `route_gate_hub_grammar` | `57` | `16` | `13` | `7` | `4` | `4` | `6/6 aligned` | `0` | `0` |
| `random_graph` | `102` | `2` | `3` | `4` | `2` | `8` | `1/1 aligned` | `0` | `0` |

Main observation:

- Hand and damaged-success graphs have fewer random-looking accidents and more complete functional wiring.
- The grammar arms may have many token-route edges, but their coverage is patchy and often not connected to the needed relation.
- The hand graph has a full suppressor-to-route matrix: `4` same-frame negative suppressor edges and `12` cross-frame positive support/suppression-routing edges. The damaged success still has `8` such edges. Failed grammar arms usually have only `2-4`.
- Hub edges help only when they bridge relevant tokens to routes. The `route_gate_hub_grammar` has many hub edges, but still misses critical task paths.

## Route Coverage

Expected-token route coverage compares whether known toy-relevant tokens have direct or hub-mediated paths into their route. This is a diagnostic of scaffold coverage, not a claim that Grammar v2 may hardcode exact labels.

| Graph | Danger coverage | Friendship coverage | Sound coverage | Environment coverage | Direct alignment |
|---|---:|---:|---:|---:|---:|
| `hand_seeded` | `7/7` | `7/7` | `6/6` | `5/5` | `1.000000` |
| `damaged_hand_seeded_50` | `5/7 path, 1 direct` | `3/7` | `6/6 path, 5 direct` | `5/5` | `1.000000` |
| `route_gate_grammar_graph` | `3/7` | `2/7` | `2/6` | `1/5` | `0.750000` |
| `route_gate_recurrence_grammar` | `0/7` | `1/7` | `2/6` | `1/5` | `0.750000` |
| `route_gate_hub_grammar` | `2/7` | `5/7 path, 1 direct` | `2/6` | `4/5 path, 1 direct` | `0.800000` |
| `route_grammar_graph` | `1/7` | `2/7` | `2/6` | `2/5` | `0.571429` |
| `random_graph` | `0/7` | `0/7` | `2/6 path, 1 direct` | `0/5` | `0.000000` |

This is the clearest distillation signal. The working graphs have coverage of the relevant token groups. Failed grammar arms often contain route nodes and gates, but do not guarantee that the right evidence can reach the right route at all.

## Path-Length Findings

Relation path diagnostic:

| Relation | Hand-seeded | Damaged success | Common failed-arm issue |
|---|---|---|---|
| `dog + bite -> danger` | both tokens reach `danger_route`, avg path `1.0`, `4` short paths | both tokens reach, avg path `2.0`, `2` short paths | `route_gate_recurrence` and `route_gate_hub` best graphs have no correct short path |
| `dog + owner -> friendship` | both tokens reach, avg path `1.0`, `3` short paths | both tokens reach, avg path `1.0`, `3` short paths | some grammar arms only route one of two tokens |
| `dog + bark -> sound` | both tokens reach, avg path `1.0`, `4` short paths | both tokens reach, avg path `1.5`, `2` short paths | grammar arms often route only one token |
| `street + car_noise -> environment` | both tokens reach, avg path `1.0`, `4` short paths | both tokens reach, avg path `1.0`, `2` short paths | `route_gate_grammar` has no correct path; `route_gate_recurrence` has no correct path |

The damaged success graph can lose many edges and still work because each major relation keeps at least a surviving route. The failed grammar arms often fail before learning starts: important relations are not connected to a usable route.

## Temporal Role Channel

Temporal order contrast depends on role-state edges into `temporal_route`.

| Graph | Expected role edges present | Aligned role edges | Temporal accuracy |
|---|---:|---:|---:|
| `hand_seeded` | `12` | `12` | `1.000000` |
| `damaged_hand_seeded_50` | `7` | `7` | `1.000000` |
| `route_gate_hub_grammar` | `6` | `6` | `0.875000` |
| `route_gate_recurrence_grammar` | `7` | `3` | `0.750000` |
| `route_gate_grammar_graph` | `3` | `2` | `0.625000` |
| `random_graph` | `1` | `1` | `0.625000` |
| `route_grammar_graph` | `0` | `0` | `0.500000` |

The route-gate-hub grammar can sometimes recover temporal behavior because it has some aligned temporal role edges, but it still lacks authority geometry. Temporal role wiring is therefore necessary for the sequence task, but not sufficient for refraction.

## Missing Ingredients In Failed Grammar Arms

### `route_grammar_graph`

Missing:

- frame gates,
- reliable temporal role channel,
- full suppressor matrix,
- token-to-hub bridge,
- broad token-to-route coverage.

It can produce modest static accuracy but temporal order stays at chance.

### `route_gate_grammar_graph`

Missing:

- environment relation path in the best graph,
- strong hub bridge,
- recurrent temporal role memory,
- full suppressor routing.

It has the best authority score among failed grammar arms (`0.227883`), but temporal accuracy is only `0.625000`.

### `route_gate_recurrence_grammar`

Missing:

- `dog+bite->danger` path in the best graph,
- `street+car_noise->environment` path in the best graph,
- token-to-hub bridge,
- aligned temporal role signs.

It has recurrence, but recurrence cannot rescue absent evidence paths.

### `route_gate_hub_grammar`

Missing:

- `dog+bite->danger` path in the best graph,
- full suppressor matrix,
- reliable route-specific coverage.

It has hubs and decent temporal accuracy (`0.875000`), but authority remains weak (`0.089167`). Hubs without route coverage and suppression are traffic, not authority routing.

### `random_graph`

Missing:

- nearly all route coverage,
- stable authority geometry,
- structured temporal channel,
- meaningful hub-route grammar despite high edge count.

Its edge count is high (`102`), but authority is low (`0.065436`). More edges are not the missing ingredient.

## Distilled Structural Rules

### Rule 1: Route Scaffold

One route group per frame is necessary, but not enough. Each route needs:

- a recurrent self-loop or local recurrence,
- token/group candidate inputs,
- suppression/gating edges,
- a formal route-state readout or explicit route-to-readout edge.

Evidence:

- `no_frame_routes` killed authority in minimality.
- Failed grammar arms contain route nodes but lack usable evidence paths.

### Rule 2: Token-to-Route Candidate Coverage

Grammar v1 used random token-route wiring. That is too weak. Grammar v2 should guarantee coverage at the feature-group level:

- actor/action tokens have candidate paths to danger/social/sound-style routes,
- place/noise tokens have candidate paths to environment routes,
- sound tokens have candidate paths to sound routes,
- relation tokens have candidate paths to friendship/social routes.

This should be a coverage scaffold, not exact label hardcoding. The grammar should not encode `dog+bite=danger` by name. It should guarantee that relevant token groups can reach plausible routes, while signs/gains remain mutable.

Evidence:

- Hand: complete expected route coverage.
- Damaged success: partial but still sufficient coverage.
- Failed grammar arms: often no path for key relations.

### Rule 3: Shared Hub Bridge

At least one shared hub bridge should collect token/context signal and distribute it to routes.

The bridge must be connected on both sides:

- token -> hub,
- hub -> route.

Evidence:

- Hand has `17` token->hub and `12` hub->route edges.
- Damaged success keeps `9` token->hub and `6` hub->route edges.
- `route_gate_recurrence_grammar` has `0` token->hub edges and fails to recover authority.
- `route_gate_hub_grammar` has hub edges, but misses route coverage; hubs are helpful only when paired with coverage.

### Rule 4: Frame Gate Placement

Frame/control should affect routes early, not only output. In the current pilot, frame gate injection lands directly on the active route. Grammar v2 should preserve that early control placement.

Evidence:

- `no_frame_gates` was necessary-core in minimality.
- Wrong-frame drop is high in hand and damaged success.

### Rule 5: Cross-Route Suppression

The hand graph uses a full suppressor-to-route matrix:

- inactive frame suppressors inhibit their own inactive routes,
- the active route receives structured contrast from suppressor state,
- suppression is route-level, not output-only.

Grammar v2 should instantiate a complete or near-complete suppressor scaffold by default. Mutations can tune signs/gains, but the matrix should not be left to random discovery.

Evidence:

- Hand has `16` suppressor->route edges.
- Damaged success keeps `8`.
- Failed grammar arms usually have `2-4`.
- `no_suppressors` mainly hurts authority/refraction, not raw accuracy.

### Rule 6: Temporal Role Channel

Temporal order contrast requires stateful role edges:

- subject token memory,
- verb memory,
- object token memory,
- directed convergence into `temporal_route`.

Grammar v2 should guarantee the role-channel scaffold without hardcoding task labels. For example, it can wire subject/object/verb role nodes into a temporal route with mutable signs/gains.

Evidence:

- Hand has `12/12` aligned temporal role edges.
- Damaged success has `7/7` and still reaches temporal accuracy `1.0`.
- Route grammar without role edges stays at `0.5`.

### Rule 7: Readout Authority

The current evaluator reads route state directly. Grammar v2 should make this explicit:

- either keep route-state readout as the official design,
- or add explicit route->readout edges and update evaluation to use readout nodes.

Avoid direct token->readout shortcuts. All analyzed graph exports have `0` token->readout edges.

### Rule 8: Mutation-Friendly Redundancy

The graph should not depend on a single critical edge per relation. Grammar v2 needs redundant weak candidate paths:

- direct token/group -> route candidates,
- token -> hub -> route candidates,
- route recurrence,
- suppressor route contrast.

Evidence:

- The hand graph has multiple short paths per relation.
- Damaged success survives with roughly half the edges because key paths remain.
- Failed grammar arms often have one partial path or no path, so mutation has no smooth hill to climb.

## Proposed Grammar v2

Grammar v2 should be explicit about structure while avoiding exact task-label solutions.

1. Route scaffold:
   one route group per frame; each route has recurrence and a readout port.

2. Group-level token coverage:
   connect token groups to plausible route families. Do not wire exact token-label rules like `dog+bite->danger`.

3. Shared hub bridge:
   add token->hub and hub->route coverage for all route families.

4. Early frame gates:
   frame/control gates route activation before route competition/readout.

5. Full route-level suppressor matrix:
   create suppressor nodes and complete suppressor->route candidate edges; mutate signs/gains.

6. Temporal role channel:
   create subject/verb/object memory nodes and route them into `temporal_route`; mutate signs/gains.

7. Explicit readout policy:
   decide whether route state itself is the readout, or implement actual route->readout edges.

8. Redundant weak paths:
   initialize multiple low-gain candidates so mutation can tune rather than invent the whole circuit.

## Proposed Next Experiment

Do not run this yet without a separate prompt.

`AUTHORITY_GRAPH_GRAMMAR_V2_SEARCH` should compare:

- Grammar v1,
- Grammar v2 route scaffold only,
- Grammar v2 route + hub bridge,
- Grammar v2 route + suppressor matrix,
- Grammar v2 route + temporal role channel,
- full Grammar v2,
- damaged-hand and hand-seeded upper bounds.

The key success criterion should be final-test authority/refraction plus temporal order, not raw accuracy alone.

## Verdict

```json
{
  "missing_structural_rule_identified": true,
  "grammar_v1_failure_explained": true,
  "grammar_v2_proposed": true,
  "evidence_for_route_scaffold": true,
  "evidence_for_hub_bridge": true,
  "evidence_for_cross_route_suppression": true,
  "evidence_for_temporal_role_channel": true
}
```

## Safe Interpretation

The latest failure is not mainly a fitness-shaping problem. The current grammar prior does not reliably generate the structural preconditions required for authority switching:

- enough token/group route coverage,
- hub bridge coverage,
- route-level suppressor contrast,
- temporal role memory,
- explicit route-state readout policy,
- redundant mutation-friendly paths.

The hand graph works because these structures are already present. Damaged hand graphs can recover because enough of the skeleton remains. Weak grammar arms fail because they are often missing entire usable paths.

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or that Grammar v2 works before it is tested.
