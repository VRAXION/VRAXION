# Learned Pointer Movement — Research Digest (2026-02-27)

## Problem Statement

A pointer fix phi-palyan mozog: `ptr = 0.5 * phi_jump + 0.5 * walk`.
Minden expert, minden timestep, minden input eseten ugyanaz. Input-fuggetlen.
Diagnostic mutatta: pointer std=0.00 minden batch elemre — determinisztikus palya.

**Kovetkezmeny:** aligned echo 83%, random offset 63%. A modell poziciora tanult, nem mintafelismeresre. A pointer a bottleneck.

---

## Korabbi VRAXION kiserletek (Golden Draft/tests/)

| Kiserlet | Eredmeny |
|----------|----------|
| **Stride gate** (`sigmoid(logits/TAU)`) | Tesztelve, eltavolitva — fix prob-ra cserelve |
| **Learned destinations** (nn.Parameter [N,M]) | 120 futas, phi legyozte — eldobva |
| **Soft index** (linearis interpolacio floor/ceil kozott) | Tesztelve, nem adoptalva (sebesseg miatt) |
| **future_ideas.md**: gravitational / content-based pull | Dokumentalva v5-re |

A stride gate VOLT, de per-expert fix prob-ot hasznalt (0.9/0.1), nem input-fuggo gate-et.

---

## Deep Research: Bevett megoldasok

### 1. NTM Interpolation Gate (legkozelebbi analog)
```
w_g = g * w_content + (1-g) * w_prev
```
Ahol `g = sigmoid(linear(controller_output))` — tanult scalar (0,1).
Ha g→0: location mode (elozo pozicio). Ha g→1: content lookup.

### 2. Skip RNN (ICLR 2018) — legpraktikusabb referencia
```python
update_prob = sigmoid(W_u @ h_t + b_u)  # scalar (0,1)
u_t = binarize_STE(update_prob)          # STE: floor forward, identity backward
# u_t=1: update state, u_t=0: copy previous
```
Pontosan a jump gate otlet. A paper megerositette hogy:
- Folytonos gate (nem binarizalt) is mukodik es stabilabb
- STE jobb mint REINFORCE binaris dontesekhez
- Budget constraint opcionalis: `loss += cost * sum(u_t)`

### 3. P-NTM (2025) — ring-buffer specifikus
Location-only shift: s ∈ (0,1)^3 = [bal, marad, jobb].
Content addressing eltavolitva. Ez a legkozelebbi az INSTNCT architekturához.

---

## Gradient Flow Problema

`.long()` elvagja a gradienst — PyTorch nem enged `requires_grad=True` integer tensoron.

**Megoldasok:**
1. **Soft addressing**: pointer = float sulyvektor [M], olvasas = weighted sum. O(M), elmos.
2. **Straight-Through Estimator**: `y = (y_hard - y_soft).detach() + y_soft`. Forward: hard, backward: soft.
3. **Gumbel-Softmax**: `F.gumbel_softmax(logits, tau, hard=True)`. Jo ha tobb opcio kozott kell valasztani.

**INSTNCT-re**: a gradient a **gate-en** keresztul folyik, nem a pointer indexen at.
A `.long()` csak az olvasas/iras pillanataban kell. A jelenlegi architektura mar igy mukodik:
```python
# ptr float-ban marad
ptr_tns[i] = gate * jump_target + (1-gate) * walk_target  # float
# .long() csak itt:
center = ptr_tns[i].long().clamp(0, M - 1)  # indexeleshez
```

---

## Javasolt Implementacio

### Minimal jump gate (10 sor valtozas)

```python
# __init__-ben:
self.jump_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(N)])

# _process_chunk-ban, a move_pointer resz:
gate = torch.sigmoid(self.jump_gate[i](hidden_lst[i]))  # (B, 1) -> (B,)
gate = gate.squeeze(-1)
jump_target = self.dests[i][ptr_tns[i].long().clamp(0, M-1)].float()
walk_target = (ptr_tns[i] + 1) % M
ptr_tns[i] = gate * jump_target + (1 - gate) * walk_target
```

**Param cost:** N * (hidden_dim + 1) = 2 * 2049 = ~4K param (elhanyagolhato).

### Alternativ: 3-way gate (bal/marad/jobb, mint P-NTM)
```python
self.shift_gate = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(N)])
# softmax([left, stay, right]) → weighted blend
```
**Param cost:** N * (hidden_dim*3 + 3) = ~12K param.

---

## Gotchak

1. **Gradient crosstalk**: N expert shared ringre ir — expert i irasa felulirhatja expert j tartalmat.
   Nem uj problema (mar most is igy van), de tanult gate-tel intenzivebb lehet.

2. **no_grad kontaminacio**: ha a gate szamitas `torch.no_grad()` blokk alatt fut,
   gradient csendben nullazodik. Grad checkpoint hasznalatanal figyelni.

3. **Elmosott iras (blurry write)**: ha a pointer sokat mozog, az iras szetszorodhat.
   NTM sharpening (gamma) erre valo. Egyelore nem kell.

4. **Init**: a gate bias-t erdemes 0-ra init-elni → sigmoid(0)=0.5 → induloertek = jelenlegi 50-50.
   Igy a tanulas a jelenlegi viselkedesbol indul, nem random gate-bol.

---

## Forrasok

- NTM PyTorch: github.com/loudinthecloud/pytorch-ntm
- Skip RNN (ICLR 2018): arxiv.org/abs/1708.06834
- P-NTM (2025): arxiv.org/html/2602.18508
- Baby-NTM: arxiv.org/abs/1911.03329
- DNC equations: jaspock.github.io/funicular/dnc.html
- Stable NTM: isg.beel.org/blog/2018/08/01/a-stable-neural-turing-machine-ntm-implementation
- Gumbel-Softmax: neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch

---

## Status: PINNED — I/O layer prioritas, pointer utana jon.
