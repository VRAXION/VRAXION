# Signal Propagation Fix — Implementation Plan

## Problem Summary

A jel az első tick után meghal mert az effektív gain per hop < 1.0:
```
signal_out = max(signal_in × DRIVE × retention - THRESHOLD, 0)
```
DRIVE=0.6, THRESHOLD=0.5 → a jel 1-2 hop után garantáltan eltűnik.
A hálózat effektív mélysége max 1-2, bármennyi tick-et adsz neki.

## Plan

### 1. lépés: Gain-analízis teszt
- Kiszámolni az **effektív gain-t** (output/input) hop-onként
- Sweep: DRIVE × THRESHOLD × retention kombók
- Cél: megtalálni hol lesz gain ≈ 1.0 (stabil terjedés, nem robban, nem hal meg)
- Kulcs kombók:
  - `DRIVE=0.6, THRESH=0.1` → gain ~0.83 (lassan csillapít)
  - `DRIVE=1.0, THRESH=0.5` → gain ~0.83
  - `DRIVE=1.5, THRESH=0.5` → gain ~1.0 (stabil!)
  - `DRIVE=2.0, THRESH=0.5` → gain > 1 (robbanhat?)
  - `DRIVE=0.6, THRESH=0.0` → gain 0.6 (de legalább nem nullázódik)

### 2. lépés: Signal trace vizualizáció
- Minden DRIVE/THRESH kombóra: 64 tick-en keresztül nézni
  - Hány neuron aktív tick-enként
  - Átlagos/max aktiváció
  - Eléri-e az output réteget
  - Stabil marad-e vagy robban/hal meg
- Ez megmutatja melyik kombó ad **stabil terjedést**

### 3. lépés: Raven benchmark a legjobb kombókkal
- A 2x2 diagonal + row=shape puzzle-ökön tesztelni
- Minden ígéretes DRIVE/THRESH kombóra:
  - Train accuracy, test accuracy
  - Budget: 20k mutation
  - Ticks: 8 és 32 (hogy lássuk a tick-hatást is)
- Kontroll: eredeti DRIVE=0.6, THRESH=0.5

### 4. lépés: Stabilitás-ellenőrzés
- A nyertes kombóval futtatni a meglévő benchmark-okat:
  - `benchmark_expressiveness.py` taskok (identity, perm, XOR, stb.)
  - `real_text_test.py` (English bigrams)
- Ha a régi taskok romlanak → a gain túl nagy, vagy a threshold túl alacsony
- Cél: Raven javul ÉS a többi nem romlik

### 5. lépés: Konstansok frissítése
- Ha van egyértelmű nyertes: `graph.py` DRIVE/THRESHOLD update
- Ha trade-off van: learnable threshold/drive marad opció (már van `drive` drift)
- Commit + push

## Nem csinálunk
- Kódolás-változtatás (one-hot marad egyelőre)
- NV_RATIO változtatás (az ortogonális kérdés)
- Új architektúra (highway, skip connection) — először a legegyszerűbb fix

## Sorrend
1 → 2 → 3 → 4 → 5, szekvenciálisan, mert minden lépés függ az előzőtől.
