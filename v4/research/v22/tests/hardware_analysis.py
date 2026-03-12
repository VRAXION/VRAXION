"""
HARDWARE FEASIBILITY ANALYSIS
===============================
What's the minimum hardware for v18 self-wiring graph?

Key requirements:
- Matrix multiply: N×N @ N (the forward pass core)
- Memory: store W, mask, state, addresses (all N×N or N×4)
- Integer/float math: at least float32, ideally float64
- No GPU needed, no framework needed

Let's compute exact requirements per network size.
"""

import numpy as np

print("="*70)
print("v18 HARDWARE REQUIREMENTS ANALYSIS")
print("="*70)

print("\n--- MEMORY REQUIREMENTS (bytes) ---")
print(f"{'Neurons':>8} {'W (f32)':>10} {'mask':>10} {'state':>8} {'addr':>8} {'TOTAL':>10} {'TOTAL':>8}")
print(f"{'':>8} {'':>10} {'':>10} {'':>8} {'':>8} {'bytes':>10} {'KB':>8}")
print("-"*70)

for N in [16, 32, 48, 64, 80, 128, 160, 256, 320, 640]:
    w_bytes = N * N * 4        # float32
    mask_bytes = N * N * 1     # uint8 (binary)
    state_bytes = N * 4        # float32
    addr_bytes = N * 4 * 4     # 4D float32
    target_bytes = N * 4 * 4   # target_W float32
    total = w_bytes + mask_bytes + state_bytes + addr_bytes + target_bytes
    print(f"{N:>8} {w_bytes:>10,} {mask_bytes:>10,} {state_bytes:>8,} "
          f"{addr_bytes:>8,} {total:>10,} {total/1024:>7.1f}")

print("\n--- COMPUTE REQUIREMENTS (per forward pass) ---")
print(f"{'Neurons':>8} {'MatMul ops':>12} {'Per tick':>12} {'8 ticks':>12} {'Full eval':>14}")
print(f"{'':>8} {'(multiply)':>12} {'(+relu)':>12} {'':>12} {'(32-class)':>14}")
print("-"*70)

for N in [16, 32, 48, 64, 80, 128, 160, 256, 320, 640]:
    matmul = N * N          # one matrix-vector multiply
    per_tick = matmul + N   # + relu
    eight_ticks = per_tick * 8
    # Full eval: 32 items × 2 passes × 8 ticks
    n_classes = max(16, N // 5)  # rough
    full_eval = eight_ticks * n_classes * 2
    print(f"{N:>8} {matmul:>12,} {per_tick:>12,} {eight_ticks:>12,} {full_eval:>14,}")

print("\n--- HARDWARE OPTIONS ---\n")

hardware = [
    ("Arduino Uno (ATmega328P)",    "8-bit 16MHz",    2*1024,     16_000_000,   "int8/int16 only"),
    ("Arduino Mega (ATmega2560)",   "8-bit 16MHz",    8*1024,     16_000_000,   "int8/int16 only"),
    ("Arduino Due (ARM Cortex-M3)", "32-bit 84MHz",   96*1024,    84_000_000,   "float32 (slow, no FPU)"),
    ("Teensy 4.0 (Cortex-M7)",     "32-bit 600MHz",  1024*1024,  600_000_000,  "float32 (HW FPU!)"),
    ("Teensy 4.1 (Cortex-M7)",     "32-bit 600MHz",  1024*1024,  600_000_000,  "float32 (HW FPU!) +8MB PSRAM"),
    ("ESP32",                       "32-bit 240MHz",  520*1024,   240_000_000,  "float32 (slow FPU), WiFi!"),
    ("ESP32-S3",                    "32-bit 240MHz",  512*1024,   240_000_000,  "float32, WiFi+BT, +8MB PSRAM"),
    ("STM32F4 (Cortex-M4)",        "32-bit 168MHz",  192*1024,   168_000_000,  "float32 (HW FPU)"),
    ("STM32H7 (Cortex-M7)",        "32-bit 480MHz",  1024*1024,  480_000_000,  "float32 (HW FPU, SIMD)"),
    ("RPi Pico (RP2040)",          "32-bit 133MHz",  264*1024,   133_000_000,  "float32 (soft FPU)"),
    ("RPi Pico 2 (RP2350)",        "32-bit 150MHz",  520*1024,   150_000_000,  "float32 (HW FPU!)"),
    ("RPi Zero 2W (ARM A53)",      "64-bit 1GHz",    512*1024*1024, 1_000_000_000, "float64, Linux, WiFi"),
    ("RPi 4 (ARM A72)",            "64-bit 1.5GHz",  1024*1024*1024, 1_500_000_000, "float64, Linux, Gigabit"),
    ("RPi 5 (ARM A76)",            "64-bit 2.4GHz",  2048*1024*1024, 2_400_000_000, "float64, Linux, fastest"),
]

# Figure out max neurons for each hardware
print(f"{'Hardware':<35} {'RAM':>8} {'Max N':>6} {'Est speed':>20} {'Swarm?':>8}")
print("-"*85)

for name, arch, ram, mhz, notes in hardware:
    # Max neurons: N*N*4 (W) + N*N (mask) = 5*N*N bytes roughly
    max_n_mem = int(np.sqrt(ram / 5))  # memory limited
    max_n = min(max_n_mem, 1280)  # cap at reasonable

    # Speed estimate: assume 1 FLOP per cycle for simple chips, 2-4 for FPU chips
    if "HW FPU" in notes or "float64" in notes:
        flops_per_cycle = 2.0
    elif "slow" in notes or "soft" in notes:
        flops_per_cycle = 0.1
    else:
        flops_per_cycle = 0.5

    if "int8" in notes:
        # Can do int8 math natively, but need to quantize
        flops_per_cycle = 1.0  # int ops are fast

    flops = mhz * flops_per_cycle

    # Time for one full eval of 32-class with 160 neurons
    # = 32 * 2 * 8 * 160^2 = 13M ops
    ops_eval = 32 * 2 * 8 * min(160, max_n)**2
    time_eval_ms = ops_eval / flops * 1000 if flops > 0 else float('inf')
    evals_per_sec = 1000 / time_eval_ms if time_eval_ms > 0 else 0

    swarm = "YES" if "WiFi" in notes or "Linux" in notes or "Gigabit" in notes else "wired"

    if ram < 1024:
        ram_str = f"{ram}B"
    elif ram < 1024*1024:
        ram_str = f"{ram//1024}KB"
    elif ram < 1024*1024*1024:
        ram_str = f"{ram//(1024*1024)}MB"
    else:
        ram_str = f"{ram//(1024*1024*1024)}GB"

    print(f"{name:<35} {ram_str:>8} {max_n:>6} "
          f"{evals_per_sec:>8.0f} eval/sec     {swarm:>8}")

print(f"\n--- BEST OPTIONS BY USE CASE ---\n")

print("CHEAPEST SWARM NODE ($2-5 each):")
print("  ESP32-S3 — 512KB RAM + 8MB PSRAM, WiFi built-in, ~$3")
print("  Can run 80-neuron net (16-class), ~100 eval/sec")
print("  10x ESP32 swarm = ~$30, wireless mesh, 1000 eval/sec combined")
print("  PERFECT for: small-scale experiments, proof of concept")

print("\nFASTEST MICROCONTROLLER ($20-30 each):")
print("  Teensy 4.1 — 1MB RAM + 8MB PSRAM, 600MHz Cortex-M7 with HW FPU")
print("  Can run 160-neuron net (32-class), ~400 eval/sec")
print("  No WiFi — needs wired communication (SPI/UART/Ethernet shield)")
print("  PERFECT for: fast dedicated compute, wired cluster")

print("\nBEST OVERALL SWARM ($15-35 each):")
print("  RPi Pico 2 (RP2350) — 520KB, HW FPU, $5")
print("  RPi Zero 2W — 512MB, WiFi, Linux, $15")
print("  RPi 4 2GB — full Linux, Gigabit, $35")
print("  PERFECT for: scalable swarm with real networking")

print("\n--- QUANTIZATION OPPORTUNITY ---\n")
print("The v18 mask is already binary (0/1) — 1 bit per connection.")
print("The ternary activation is {-1, 0, +1} — 2 bits per neuron.")
print("If we quantize weights to int8 (256 levels):")
print("  160 neurons: W=25KB, mask=3.2KB, total <30KB")
print("  This fits in an ARDUINO DUE (96KB RAM)!")
print("  Or 5+ instances in a single ESP32 (520KB)")
print()
print("With ternary weights {-1, 0, +1} (2 bits each):")
print("  160 neurons: W=6.4KB, mask=3.2KB, total <10KB")
print("  This fits in an ARDUINO MEGA (8KB RAM) barely!")
print("  Or 50+ instances in a single ESP32")
print()
print("TERNARY EVERYTHING (weights + activations):")
print("  The entire forward pass becomes bitwise operations")
print("  Multiply: AND/XOR gates (no floating point needed)")
print("  Accumulate: popcount (hardware instruction on most CPUs)")
print("  This would be THOUSANDS of times faster than float math")
print("  An ESP32 could potentially run MILLIONS of eval/sec")
