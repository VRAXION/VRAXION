# Overnight Sweep Log — 2026-04-08

## Plan
1. **Sweep 1**: ListNet full sweep — 5 seeds, 120s/seed, H=256-4096, edge_cap=300
2. **Sweep 2**: Edge cap sweep — H=1024, caps=[100,200,300,500,1000], 5 seeds, 60s/seed
3. **Sweep 3**: ListNet vs INSTNCT head-to-head — H=256+2048, 5 seeds, 120s/seed
4. **Sweep 4**: Long run best config — 300s/seed, 5 seeds
5. **Sweep 5**: Cache A/B/C/D with ListNet

## Status
- [x] Sweep 1+2 started at ~00:15 (combined in one binary)
- 00:30 check: H=256 done (20.2% best, 3813 step/s). H=512 running.
- 01:00 check: H=256-2048 done. H=512 best: **23.8%**! H=4096 running, sweep 2 next.
- [x] Sweep 1+2 DONE at ~01:15
  - Sweep 1 winner: H=512 @ 23.8% best / 21.1% mean
  - Sweep 2 winner: edge_cap=100 (21.6% best, 20.7% mean) — smallest cap wins!
- [x] Sweep 3 started at ~01:15
- [x] Sweep 3 DONE at ~02:00
  - H=256: ListNet 3847 step/s vs INSTNCT 564 = **6.8x**, accuracy identical (20.4% vs 20.6%)
  - H=2048: ListNet 571 step/s vs INSTNCT 233 = **2.5x**, accuracy identical (21.6% both)
- [x] Sweep 4 started at ~02:00 (long run H=512, 300s/seed)
- [x] Sweep 4 DONE at ~02:30
  - H=512 long run (300s): best=20.8%, mean=20.0%, spread=1.4pp, 2.0 µs/tok
  - More time does NOT improve accuracy — the 20% band is the ceiling for 1+1 ES
- [x] Sweep 5 started at ~02:30
- [x] Sweep 5 DONE at ~03:05
  - No sharp cache cliff with ListNet — smooth linear scaling
  - H=1024 (L1): best=21.4%, mean=20.6%, 1095 step/s
  - H=8192 (L2): still 20.0% mean at 148 step/s
- [x] All sweeps complete. Compiling report...
