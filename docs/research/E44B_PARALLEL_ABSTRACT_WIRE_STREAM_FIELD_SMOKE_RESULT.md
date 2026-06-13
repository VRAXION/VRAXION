# E44B Parallel Abstract Wire Stream Field Smoke Result

## Decision

```text
decision = e44b_parallel_serial_capacity_detected
required_capacity_bits = 5
run_id = 250b6de07f521c04
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
```

E44B tested the stricter wire-stream version of the E44 question:

```text
Wire#01: 010001...
Wire#02: 010010...
...
```

The result is clean: for this 32-intent abstract Proposal payload task, what
matters is total collision-free capacity. Any `wire_count x bits_per_wire`
shape with at least 5 total bits passed. Every shape below 5 total bits failed.

## Primary 6x6 Table

```text
| wires \ bits | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 1 | fail 0.688 | fail 0.750 | fail 0.812 | fail 0.812 | PASS 1.000 | PASS 1.000 |
| 2 | fail 0.750 | fail 0.812 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 |
| 3 | fail 0.812 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 |
| 4 | fail 0.812 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 |
| 5 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 |
| 6 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 | PASS 1.000 |
```

## Key Comparisons

```text
1 wire x 5 bits  = PASS
5 wires x 1 bit  = PASS
2 wires x 3 bits = PASS
3 wires x 2 bits = PASS

1 wire x 4 bits  = fail
2 wires x 2 bits = fail
3 wires x 1 bit  = fail
4 wires x 1 bit  = fail
```

This means the first-order smoke result is not "must be many wires" and not
"must be one long wire." It is:

```text
the abstract payload must have at least 5 bits of usable capacity
```

Within this clean abstract setup, serial and parallel layouts were equivalent
once total capacity reached the required threshold.

## Controls

```text
oracle_success = 1.000
headerless_5x1_success = 0.500
random_decoder_5x1_success = 0.625
```

The headerless control did not pass, so the fixed mechanical Proposal header is
still required. The random decoder did not pass, so capacity alone is not enough;
the stream code must be learned/matched to the Proposal ABI.

## Confirm Seeds

```text
seed 44502: e44b_parallel_serial_capacity_detected, required bits 5
seed 44503: e44b_parallel_serial_capacity_detected, required bits 5
```

## Interpretation

In this abstract smoke, a wire stream behaves like a small anonymous bus:

```text
wire_count = how many parallel lanes
bits_per_wire = how long each lane's local stream is
capacity_bits = wire_count * bits_per_wire
```

The system did not care whether the 5 bits arrived as:

```text
1 x 5
5 x 1
2 x 3
3 x 2
```

The break was exactly the collision boundary for 32 intents.

## Boundary

This is a controlled symbolic/numeric Proposal ABI smoke. It does not prove raw
language reasoning, deployed AI assistant behavior, model-scale behavior, AGI,
or consciousness.
