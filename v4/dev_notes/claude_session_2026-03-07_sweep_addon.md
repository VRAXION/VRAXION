# Claude Session Addon — 2026-03-07
# INSTNCT v4 Parameter Sweep & Optimization Report

> **Cél:** Ez a dokumentum a GPT nightly branch kezelő AI számára készült.
> Tartalmazza az összes tesztelési eredményt, módszertant, következtetést és ajánlást,
> hogy beolvasztható legyen a VRAXION tudástárba.

---

## 1. Mi történt ebben a sessionben

Két nagy benchmark suite-ot futtattunk le CPU-n:

1. **CPU Parameter Sweep** (`v4/tests/bench_param_sweep.py`) — sebesség-orientált mérés
2. **Minmaxed IQ Sweep** (`v4/sweeps/sweep_minmax_iq.py`) — IQ/loss/accuracy optimalizáció minimális compute-tal

Mindkettő az INSTNCT v4 modellt teszteli (`v4/model/instnct.py`), byte-level tokenizációval.

---

## 2. Tesztelési módszertan

### 2.1 CPU Parameter Sweep (`bench_param_sweep.py`)

**Cél:** Megmérni minden egyes hiperparaméter hatását a CPU tanítási sebességre (steps/sec), izoláltan és párosan.

**Módszer:**
- Baseline config: `hidden_dim=256, slot_dim=64, M=128, N=1, R=1, B=16, seq=64`
- Kernel: vshape, pointer: sequential, embed: bitlift, output: lowrank_c19, write: replace
- 2 warmup step + 5 bench step per konfig (rövid de megbízható CPU mérésnél)
- `torch.set_num_threads(16)` — konzisztens CPU threading

**Sweepelt paraméterek (single-axis):**
| Paraméter | Értékek | Cél |
|-----------|---------|-----|
| hidden_dim | 128, 256, 512, 1024, 2048 | Brain width hatása sebességre |
| slot_dim | 32, 64, 128, 256 | Ring slot width hatása |
| M (ring slots) | 64, 128, 256, 512, 1024 | Memória kapacitás vs sebesség |
| seq_len | 32, 64, 128, 256 | Szekvencia hossz hatása |
| batch_size | 4, 8, 16, 32, 64 | Batch méret skálázódás |
| N (experts) | 1, 2, 4, 6 | Expert szám overhead |
| R (attn radius) | 0, 1, 2, 4 | Figyelmi ablak méret |
| pointer_mode | sequential, learned, pilot | Pointer mozgás overhead |

**Interakciós sweepek (párban):**
- slot_dim × seq_len (3×3 = 9 config)
- hidden_dim × slot_dim (4×3 = 12 config)
- M × kernel_mode [vshape, dotprod] (4×2 = 8 config)
- N × seq_len (3×3 = 9 config)
- hidden_dim × pointer_mode (4×2 = 8 config)
- batch_size × hidden_dim (3×4 = 12 config)

**Összesen: ~80+ egyedi konfiguráció lemérve.**

### 2.2 Minmaxed IQ Sweep (`sweep_minmax_iq.py`)

**Cél:** Maximális "intelligencia" (loss, accuracy) minimális compute-tal. N=1 (legolcsóbb), csak a magas IQ/alacsony cost paraméterek variálva.

**Módszer:**
- Baseline: `M=128, hidden=256, slot=32, N=1, R=1, vshape, sequential, bitlift, learned output`
- 150 training step per konfig, batch=8, seq=32
- Két task: `echo` (könnyű, ismétlés) + `delay_echo` (nehéz, memória kell)
- Mérések: best_loss, best_acc (byte-level), steps/sec, param count

**Sweepelt paraméterek (IQ impact rating):**
| Paraméter | IQ Impact | Cost Impact | Értékek |
|-----------|-----------|-------------|---------|
| output_encoding | ★★★★☆ | ★☆ | learned, lowrank_c19 |
| kernel_mode | ★★★☆ | ★☆ | vshape, gaussian, uniform |
| embed_encoding | ★★☆ | ★☆ | learned, bitlift |
| pointer_mode | ★★☆ | ★★☆ | sequential, pilot |
| S (context_scale) | ★★☆ | ☆ | 0.0, 0.05, 0.1, 0.3, dotprod |
| hidden_dim | ★★★★★ | ★★★★☆ | 128, 256, 512 |
| R (attn radius) | ★★★☆ | ★★☆ | 0, 1, 2 |

**Phase 2 — Kombinált kandidátusok:**
- Baseline (referencia)
- MinCost+MaxIQ: lowrank + vshape + bitlift + S=dotprod
- MinCost+MaxIQ+Pilot: + pilot pointer
- BrainBoost: H=512 + lowrank + vshape + bitlift + S=dotprod
- BrainBoost+Pilot: + pilot pointer
- Cheapest: H=128, S=0.05 (minimális erőforrás)

---

## 3. Eredmények és következtetések

### 3.1 CPU Sebesség (bench_param_sweep)

**Legnagyobb sebesség-hatás:**
1. **hidden_dim** — domináns faktor. H=128→2048: ~10-20× lassulás. Legjobb ár/érték: H=256-512.
2. **batch_size** — lineáris skálázódás; nagyobb batch = jobb throughput/step de több RAM.
3. **seq_len** — lineáris hatás; seq=32 dupla gyors vs seq=128.
4. **N (experts)** — ~lineáris overhead; N=1 a leggyorsabb.
5. **M (ring slots)** — meglepően alacsony hatás M≤512-ig; M=1024-nél érezhető lassulás.
6. **pointer_mode** — pilot minimális overhead a sequential-hoz képest.
7. **slot_dim** — mérsékelt hatás; slot=32 gyors, slot=256 lassú.
8. **R (attn radius)** — elhanyagolható R≤2-ig.

**Interakciós meglepetések:**
- `hidden_dim × slot_dim`: slot_dim hatása **nő** ahogy hidden_dim nő (nem additív)
- `hidden_dim × pointer_mode`: pilot overhead **nem nő** hidden_dim-mel (jó hír)
- `M × kernel_mode`: dotprod kernel **drágább** nagy M-nél (O(M) vs O(1) vshape)

**Optimális CPU konfig kandidátusok:**
| Profil | Config | Megjegyzés |
|--------|--------|------------|
| MaxSpeed | H=128, SD=32, M=256, seq=32, B=32 | Gyors iteráció, alacsony kapacitás |
| Balanced | H=512, SD=64, M=512, seq=64, B=16 | Legjobb ár/érték |
| HighCap | H=1024, SD=64, M=512, seq=64, B=8 | Magas kapacitás, elfogadható sebesség |
| Prod-Small | H=2048, SD=128, M=1024, seq=128, B=4 | Production-közelítés CPU-n |

### 3.2 IQ/Loss Optimalizáció (sweep_minmax_iq)

**Echo task (könnyű — ismétlés detektálás):**
- Szinte minden config könnyen megtanulja (~99%+ acc)
- **output_encoding**: lowrank_c19 ≈ learned (echo-nál nincs nagy különbség)
- **kernel_mode**: vshape a legjobb, uniform a legrosszabb
- **S (context_scale)**: dotprod és S=0.3 a legerősebb
- **hidden_dim**: H=512 >> H=256 > H=128

**Delay_echo task (nehéz — memória szükséges):**
- **output_encoding**: lowrank_c19 **egyértelműen jobb** mint learned (+loss csökkenés)
- **kernel_mode**: vshape > gaussian >> uniform
- **pointer_mode**: pilot ≈ sequential (150 step kevés a pilot előnyéhez)
- **S (context_scale)**: dotprod gate a legjobb (tanult, adaptív)
- **hidden_dim**: H=512 **jelentősen jobb** loss/acc — de 2× lassabb
- **R**: R=1 legjobb; R=0 túl szűk, R=2 túl zajos rövid seq-nél

**Phase 2 kombinált eredmények:**
- **BrainBoost+Pilot** (H=512, lowrank, vshape, bitlift, pilot, S=dotprod) = legjobb loss+acc
- **MinCost+MaxIQ** (H=256, lowrank, vshape, bitlift, seq, S=dotprod) = legjobb ár/érték
- **Cheapest** (H=128, S=0.05) = meglepően erős echo-n, gyenge delay_echo-n

---

## 4. Korábbi session eredmények összefoglalása (kontextus)

### 4.1 C19 Dual-Phi A/B teszt (2026-03-05)
- **GPU**: RTX 4070 Ti SUPER, WikiText-103 byte-level
- **Dual-phi aktiváció**: +1.5% acc vs baseline, +1.65% vs neg-phi-only
- **Gradiens stabilitás**: 2× alacsonyabb max grad norm (3.5 vs 7.4)
- **Döntés**: Dual-phi beolvasztva main-be ✅
- Fájlok: `v4/tests/ab_c19_dualphi_wikitext.py`, `v4/tests/ab_c19_negphi_vs_dualphi.py`

### 4.2 Replace Write mód (2026-03-01)
- Scatter_add → HDD-stílusú replace write
- Blob eliminálva: adj_cos 0.99 → 0.78 (N=1)
- **N=2 még mindig blobosodik** — future: write buffer kell

### 4.3 Pilot Seek-First pointer (2026-03-01)
- SEEK → READ → HIDDEN → WRITE sorrend (pilot esetén)
- ~3× gyorsabb tanulás a régi READ-first-höz képest
- Csak szintetikus taszkon validálva, WikiText pending

---

## 5. Fájlok és hivatkozások

| Fájl | Leírás |
|------|--------|
| `v4/sweeps/sweep_minmax_iq.py` | IQ sweep script (N=1, echo + delay_echo) |
| `v4/tests/bench_param_sweep.py` | CPU sebesség benchmark (single + interaction) |
| `v4/model/instnct.py` | Fő modell implementáció |
| `v4/config/vraxion_config.yaml` | Központi konfiguráció (production értékek) |
| `v4/tests/ab_c19_dualphi_wikitext.py` | C19 dual-phi A/B teszt script |
| `v4/dev_notes/ab_c19_dualphi_results_2026-03-05.md` | Dual-phi eredmények részletesen |
| `v4/dev_notes/experiment_log.md` | Teljes kísérleti napló (legrégebbitől) |

---

## 6. Ajánlások a nightly branch-nek

### Azonnali beolvasztás (high confidence):
1. **lowrank_c19 output encoding** — egyértelműen jobb mint learned, kevesebb param
2. **vshape kernel** — konzisztensen a legjobb pozicionális kernel
3. **bitlift embed encoding** — 28× kevesebb param mint learned, azonos teljesítmény
4. **S=dotprod** (tanult gate) — legjobb S érték, adaptív

### Továbbvizsgálandó:
1. **hidden_dim=512 vs 2048** — 512 jobb ár/érték CPU-n, de 2048 a production GPU config
2. **pilot pointer** — rövid teszten nincs előnye, de korábbi szintetikus teszten 3× gyorsabb tanulás → hosszabb futtatás kell
3. **N=2+ expert support** — replace write N=2-vel még blobosodik, write buffer implementáció szükséges
4. **slot_dim × hidden_dim interakció** — nem additív, érdemes újrasweepelni GPU-n

### Nem ajánlott:
1. **uniform kernel** — konzisztensen a legrosszabb
2. **S=0.0** (no context) — a ring read blend nélkül a modell nem használja a memóriát
3. **R=0** (no attention window) — túl szűk, legalább R=1 kell

---

## 7. Technikai megjegyzések

- **Környezet**: CPU-only tesztelés (Linux 4.4.0, PyTorch, 16 thread)
- **Reprodukálhatóság**: seed=42, determinisztikus data generátor
- **Limitáció**: CPU tesztek 150-step limit — a valós GPU futtatás (500-50000 step) más rangsort hozhat a lassabban konvergáló konfigoknál
- **Kód minőség**: mindkét sweep script self-contained, argparse CLI, futtatható `python v4/sweeps/sweep_minmax_iq.py` ill. `python v4/tests/bench_param_sweep.py`

---

*Generálta: Claude (Opus 4.6) — 2026-03-07*
*Session: claude/update-nightly-branch-1pysT*
