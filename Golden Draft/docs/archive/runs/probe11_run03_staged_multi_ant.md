# Probe 11 — Run 03: Staged Multi-Ant Activation

**Date:** 2026-02-09
**Task:** Harmonic XOR (sin(F=1) + sin(F=7), label = XOR(sign_slow, sign_fast))
**Script:** `tools/probe11_fib_volume_weight.py`
**Device:** CUDA (RTX)

## Run Phases

### Phase 1: Single Ant (steps 1–850)
- **Config:** `--active-ants 1`, ant[0] solo (ring=1071, weight=0.505)
- **Result:** Plateaued at **86.6% MA100** by step 850
- Gradient norms calmed from ~5000 (early) to ~10-50 (converged)

### Phase 2: Staged Activation (steps 851–1260)
- **Config:** `--active-ants 2 --freeze-ants 0` — ant[0] frozen (still votes), ant[1] learns residual
- **Result:** Peak **87.9% MA100** at step ~1238 (+1.3% over Phase 1)
- ant[1] (ring=535) learned the residual errors ant[0] missed
- Gnorms: ant[0]=0.0 (frozen), ant[1]=8-12 with spikes to 120+
- Msg norms: both ants at ~4.0 (ant[1] caught up to ant[0]'s signal strength)
- Conclusion: ant[1] squeezed +1.3% but plateaued — limited capacity (535 params vs 1071)

### Phase 3: Both Unfrozen (steps 1261–1669)
- **Config:** `--active-ants 2` — both ants learning, fresh optimizer
- **Result:**
  - Initial dip from 87.9% → 81.2% (optimizer reset, co-adaptation shock)
  - Recovery to **~92% MA100** around step 1600 (broke through Phase 2 ceiling!)
  - Then unstable decline to **83.5% MA100** by step 1669
  - Gnorm ratio wildly volatile: 0.02x to 197x (gradient starvation flashes)
- **Per-ant solo accuracy** (from step 1488+):
  - ant[0] solo: ~55% MA100 (was 87% when it was the only ant)
  - ant[1] solo: ~52% MA100 (barely above coin-flip)
  - Combined swarm: 83-92% — **neither ant is good alone, but together they work**

## Key Findings

1. **Staged activation works mechanically.** Freeze/unfreeze, optimizer state handling, checkpoint resume — all solid.

2. **Volume weighting prevents ant death** but doesn't prevent gradient starvation flashes. Both ants maintained msg_norm ~4.0 throughout (no suppression). But gnorm ratio swings 0.02x–197x indicate one ant dominates gradient on any given step.

3. **Co-adaptation is unstable without LR scheduling or AGC.** Phase 3 showed promising peak (92%) but couldn't hold it — gradient spikes destabilized the learned representations.

4. **Swarm specialization confirmed.** Per-ant accuracy shows neither ant predicts well alone (55%, 52%), but combined they hit 83-92%. They're genuinely learning different aspects of the signal.

5. **Big-first may be suboptimal.** The big ant (ring=1071) learned 87% of the task, leaving only a thin residual for the small ant. Small-first activation might allow better capacity allocation.

## Metrics at Kill

| Metric | Value |
|--------|-------|
| Step | 1669 |
| MA100 | 83.5% |
| MA50 | 80.1% |
| MA10 | 82.5% |
| ant[0] solo MA100 | 55.0% |
| ant[1] solo MA100 | 52.5% |
| Loss | 0.598 |
| Logit norm | 5.5 |

## Checkpoints

- 167 step checkpoints (every 10 steps) in `logs/probe/checkpoints/`
- Range: `probe11_step_00010.pt` through `probe11_step_01660.pt`
- Plus `best.pt` and `latest.pt`

## Code Changes This Session

| File | Change |
|------|--------|
| `hallway.py` | Store `_last_per_ant_logits` during forward pass |
| `probe11_fib_volume_weight.py` | `--freeze-ants` feature, optimizer resume fix, per-ant accuracy (MA100/50/10) |
| `probe11_dashboard.py` | Full rewrite: plotly, frozen/active detection, per-ant deep dive, status badges |
| `requirements.txt` | Uncommented streamlit/plotly/pandas |

## Next Steps

- [ ] Try small-first activation (ant[1] solo → freeze → activate ant[0] to learn residual)
- [ ] Wire AGC into probe11 training loop to dampen gradient spikes during co-adaptation
- [ ] Consider LR warmup/scheduling for newly unfrozen ants
- [ ] Test with 3+ active ants
