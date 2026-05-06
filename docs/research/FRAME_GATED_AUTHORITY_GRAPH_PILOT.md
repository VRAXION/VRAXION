# Frame-Gated Authority Graph Pilot

## Why This Test Exists

The neural toy produced a shared-core plus frame-specific-route mechanism with hub/integrator and suppressor/gate candidates. This pilot asks whether that mechanism requires neural layers, or whether a smaller explicit recurrent authority graph can reproduce the same behavior without backprop through the graph internals.

## Minimal Mechanism Tested

- token input nodes inject observed symbols
- shared hub nodes integrate common actor/action/context evidence
- frame route nodes receive frame-gated evidence
- suppressor nodes inhibit inactive routes
- recurrent scalar states settle over a few steps
- readout uses active-route authority against inactive-route competition

## Graph Architecture

Update rule:

```text
state[t+1] = tanh(decay * state[t] + signed_edge_sum + input_injection + frame_gate_modulation + bias)
```

No gradient or backprop is used. The `evolved_graph_small` mode uses mutation/hillclimb over edge gains and scalar gate/suppressor strengths.

## Run Configuration

```json
{
  "seeds": 3,
  "steps": 5,
  "samples": 160,
  "mutation_steps": 60,
  "mutation_scale": 0.18,
  "random_graphs": 3,
  "decay": 0.35,
  "smoke": false
}
```

## Task Results

| Mode | Accuracy | Latent | Multi-Aspect | Temporal Order | Authority | Refraction |
|---|---:|---:|---:|---:|---:|---:|
| `hand_seeded_graph` | `0.988194` | `0.977083` | `0.987500` | `1.000000` | `0.363368` | `0.363368` |
| `random_graph_baseline` | `0.468056` | `0.445139` | `0.445139` | `0.513889` | `-0.054453` | `-0.054453` |
| `evolved_graph_small` | `0.680556` | `0.658333` | `0.633333` | `0.750000` | `0.046189` | `0.046189` |

## Control Results

| Mode | Wrong Frame Drop | No Frame Gate Drop | No Suppressor Acc Drop | No Suppressor Authority Drop | Inactive Influence Rise | No Recurrence Drop |
|---|---:|---:|---:|---:|---:|---:|
| `hand_seeded_graph` | `0.350000` | `0.131250` | `-0.010417` | `0.020759` | `0.010562` | `0.500000` |
| `random_graph_baseline` | `-0.005556` | `-0.058333` | `-0.013889` | `-0.011571` | `0.002816` | `0.013889` |
| `evolved_graph_small` | `0.016667` | `0.056250` | `0.110417` | `0.020805` | `0.021266` | `0.250000` |

## Mutation / Evolution Result

- mutation steps requested: `60`
- accepted mutations mean: `11.666667`
- evolved graph accuracy mean: `0.680556`

## What Neural Components Were Unnecessary

- Dense neural hidden layers were not required for the hand-seeded pilot graph to show frame-gated authority routing.
- Backprop through the graph internals was not required; the evolved variant used only mutation/hillclimb.

## What Still Required Dynamics

- Frame gates remain necessary in the hand graph when the same observations are evaluated under different task frames.
- Suppressor nodes matter when inactive route competition would otherwise leak into the active decision.
- Temporal order contrast depends on recurrent carry / role memory; final-token-only evaluation is weaker.

## Verdict

```json
{
  "supports_non_neural_authority_graph": "true",
  "supports_frame_gated_routes": "true",
  "supports_suppressor_nodes": "true",
  "supports_recurrence_for_order_binding": "true",
  "neural_net_necessity_reduced": "true",
  "evolved_graph_improves_over_random": "true"
}
```

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.
