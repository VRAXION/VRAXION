# E8D Substrate-First RAM Language Pretraining Probe Result

## Decision

```text
decision = e8d_substrate_language_not_helpful
primary checker failure_count = 0
gpu confirm checker failure_count = 0
primary deterministic replay = passed
gpu confirm deterministic replay = passed
```

E8D tested whether learning a shared RAM/substrate language before pocket
composition makes producer/consumer pocket composition easier. The answer in
this controlled proxy is: not enough. The substrate models learned valid RAM
state geometry, but did not materially improve downstream composition.

## Primary Evidence Run

Artifact root:

```text
target/pilot_wave/e8d_substrate_first_ram_language_pretraining_probe/
```

Key primary means:

```text
no_substrate_baseline                    useful 0.403462  write 0.735075  read 0.894402
bridge_only_baseline                     useful 0.336408  write 0.788356  read 0.906389
substrate_autoencoder                    useful 0.406913  write 0.735570  read 0.894773  valid 0.957623
substrate_transition_model               useful 0.406786  write 0.732644  read 0.893438  valid 0.927781
low_bit_substrate_codebook               useful 0.404624  write 0.728648  read 0.891292  valid 0.884403
frozen_substrate_then_producer           useful 0.403721  write 0.726494  read 0.889670  valid 0.884403
frozen_substrate_then_consumer           useful 0.369381  write 0.765241  read 0.905437  valid 0.884403
frozen_substrate_then_pocket_composition useful 0.355918  write 0.762265  read 0.901984  valid 0.884403
jointly_mutable_substrate_and_pockets    useful 0.374050  write 0.766049  read 0.905793  valid 0.884403
oracle_substrate_reference               useful 0.608516  write 0.884665  read 0.952594
dense_graph_danger_control               useful 0.387123  write 0.620451  read 0.714873
```

Best substrate gain over the best baseline was only:

```text
substrate_autoencoder - no_substrate_baseline = +0.003451
```

That is far below the `+0.03` positive threshold.

## GPU Confirm

Artifact root:

```text
target/pilot_wave/e8d_substrate_first_ram_language_pretraining_probe_gpu_confirm/
```

GPU confirm reproduced the same interpretation:

```text
no_substrate_baseline                    useful 0.489325  write 0.660428  read 0.856505
bridge_only_baseline                     useful 0.300887  write 0.761674  read 0.893737
substrate_autoencoder                    useful 0.495188  write 0.660820  read 0.856536  valid 0.955870
substrate_transition_model               useful 0.493174  write 0.659897  read 0.855988  valid 0.925740
low_bit_substrate_codebook               useful 0.487328  write 0.660145  read 0.856129  valid 0.884094
frozen_substrate_then_producer           useful 0.487328  write 0.660145  read 0.856129  valid 0.884094
frozen_substrate_then_consumer           useful 0.413474  write 0.669448  read 0.860231  valid 0.884094
frozen_substrate_then_pocket_composition useful 0.419146  write 0.668389  read 0.859050  valid 0.884094
jointly_mutable_substrate_and_pockets    useful 0.413474  write 0.669448  read 0.860231  valid 0.884094
oracle_substrate_reference               useful 0.637835  write 0.884486  read 0.952556
dense_graph_danger_control               useful 0.306892  write 0.623010  read 0.627238
```

Best GPU-confirm substrate gain:

```text
substrate_autoencoder - no_substrate_baseline = +0.005864
```

This also fails the positive threshold.

## Interpretation

E8D falsified the simple substrate-first hypothesis in this implementation. The
substrate itself learned stable-looking RAM codes:

```text
primary substrate_autoencoder validity = 0.957623
gpu substrate_autoencoder validity     = 0.955870
```

But a valid-looking RAM state space did not by itself become a better pocket
composition substrate. In other words:

```text
valid RAM state geometry != useful shared pocket ABI
```

The bridge-only baseline improved write/read compatibility but hurt composition,
which suggests that post-hoc translation can overfit the local write shape while
damaging the route-level state dynamics.

Dense graph did not win in the evidence run or GPU confirm. This keeps the
negative result clean: the substrate-first variants failed by insufficient
usefulness gain, not because graph-soup dominated the proxy.

## What We Learned

E8C said target decomposition was not enough. E8D says substrate pretraining
alone is also not enough.

The remaining likely bottleneck is not just:

```text
producer target shape
```

and not just:

```text
valid RAM state distribution
```

It is probably the stricter interface problem:

```text
learn a callable ABI where:
  producer writes
  substrate canonicalizes/integrates
  consumer reads
  route-level dynamics remain useful
```

## Next Step

Recommended next experiment:

```text
E8E_FLOW_ABI_INTEGRATOR_AND_STATE_DYNAMICS_PROBE
```

Focus: test a mechanical flow integrator/ABI layer that is trained on local
write validity and route-level next-state usefulness together. It should not be
a semantic lane system, a new router, or a dense graph. The falsification target
should be whether integration plus state-dynamics supervision beats both
substrate-first and bridge-only while preserving OOD/counterfactual/adversarial
behavior.
