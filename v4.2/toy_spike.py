"""
Toy Network — Integrate-and-Fire Spike Test
=============================================
8 neurons, hand-wired edges, NO training.
Test: does spiking create logical gates (AND, OR)?

  [0] input A        [1] input B
   |                   |
   v (+1.5)            v (+1.5)
  [2] A amplifier     [3] B amplifier
   |         \         |        /
   v (+1.0)   \(+1.5)  v(+1.0) /(+1.5)
  [4] AND gate        [5] OR gate
   |
   v (+1.5)
  [6] delay
   |
   v (+1.5)
  [7] output
"""
import numpy as np

H = 8
NAMES = ['inpA', 'inpB', 'ampA', 'ampB', 'AND ', 'OR  ', 'dely', 'out ']

# Hand-wired edges: (src, tgt, weight)
EDGES = [
    (0, 2, +1.5),   # input A -> amplifier A
    (1, 3, +1.5),   # input B -> amplifier B
    (2, 4, +1.0),   # ampA -> AND (needs both)
    (3, 4, +1.0),   # ampB -> AND (needs both)
    (2, 5, +1.5),   # ampA -> OR (either enough)
    (3, 5, +1.5),   # ampB -> OR (either enough)
    (4, 6, +1.5),   # AND -> delay
    (6, 7, +1.5),   # delay -> output
]

rs = np.array([e[0] for e in EDGES])
cs = np.array([e[1] for e in EDGES])
ws = np.array([e[2] for e in EDGES], dtype=np.float32)

RET = 0.85


def run_test(name, inject_A, inject_B, threshold, n_ticks=50):
    """Run toy network and print tick-by-tick state."""
    charge = np.zeros(H, dtype=np.float32)
    act = np.zeros(H, dtype=np.float32)
    spike_log = {i: [] for i in range(H)}

    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"  inject A={inject_A}, B={inject_B}, threshold={threshold}, ret={RET}")
    print(f"{'='*80}")

    # Header
    hdr = "tick "
    for i in range(H):
        hdr += f" {NAMES[i]:6s}"
    print(hdr)
    print("-" * len(hdr))

    for tick in range(n_ticks):
        # 1. INJECT
        act[0] += inject_A
        act[1] += inject_B

        # 2. PROPAGATE
        raw = np.zeros(H, dtype=np.float32)
        np.add.at(raw, cs, act[rs] * ws)

        # 3. ACCUMULATE + DECAY
        charge += raw
        charge *= RET

        # 4. SPIKE CHECK
        fired = charge > threshold
        spike_neurons = np.where(fired)[0]

        # 5. ReLU
        act = np.maximum(charge, 0.0)
        charge = np.maximum(charge, 0.0)

        # 6. RESET fired neurons
        charge[fired] = 0.0
        act[fired] = 0.0

        # Log spikes
        for n in spike_neurons:
            spike_log[n].append(tick)

        # Print state
        line = f"{tick:3d}  "
        for i in range(H):
            if i in spike_neurons:
                line += f" *{charge[i]:4.1f}* "
            else:
                line += f"  {charge[i]:4.2f} "
        if spike_neurons.size > 0:
            line += f"  FIRE: {[NAMES[n] for n in spike_neurons]}"
        print(line)

    # Summary
    print()
    print(f"  Spike summary:")
    for i in range(H):
        if spike_log[i]:
            freq = len(spike_log[i]) / n_ticks
            intervals = np.diff(spike_log[i]) if len(spike_log[i]) > 1 else []
            avg_int = np.mean(intervals) if len(intervals) > 0 else 0
            print(f"    [{NAMES[i]}] fired {len(spike_log[i])}x"
                  f" (freq={freq:.2f}/tick, avg interval={avg_int:.1f} ticks)"
                  f" at ticks: {spike_log[i][:10]}{'...' if len(spike_log[i])>10 else ''}")
        else:
            print(f"    [{NAMES[i]}] never fired")
    print()


if __name__ == "__main__":
    print("TOY SPIKE NETWORK")
    print(f"Neurons: {H}, Edges: {len(EDGES)}")
    print(f"Topology:")
    for s, t, w in EDGES:
        print(f"  [{NAMES[s]}] --({w:+.1f})--> [{NAMES[t]}]")

    # Test 1: Only A input
    run_test("TEST 1: Only A (should fire ampA, OR, NOT AND)",
             inject_A=1.0, inject_B=0.0, threshold=3.0, n_ticks=30)

    # Test 2: Both A and B
    run_test("TEST 2: Both A+B (should fire AND too!)",
             inject_A=1.0, inject_B=1.0, threshold=3.0, n_ticks=30)

    # Test 3: Weak input (should fire slower)
    run_test("TEST 3: Weak A only (slower firing)",
             inject_A=0.5, inject_B=0.0, threshold=3.0, n_ticks=40)

    # Test 4: Threshold sweep
    for thr in [2.0, 5.0, 10.0]:
        run_test(f"TEST 4: Both A+B, threshold={thr}",
                 inject_A=1.0, inject_B=1.0, threshold=thr, n_ticks=30)
