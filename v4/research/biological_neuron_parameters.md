# Biological Neuron Electrophysiology Parameters
## Reference for Capacitor Neuron Model Mapping

> Compiled from neuroscience literature. Focus: cortical neurons (pyramidal cells, interneurons).
> All values include min/typical/max ranges where available.

---

## 1. MEMBRANE CAPACITANCE

### Specific Capacitance (per unit area)

| Neuron Type             | Species | Cm (µF/cm²)  | Source |
|-------------------------|---------|-------------|--------|
| Generic biological membrane | —   | ~1.0        | Classical "universal" value |
| Cortical pyramidal (soma) | Rodent | 0.9 ± 0.1  | [Gentet et al., 2000](https://pmc.ncbi.nlm.nih.gov/articles/PMC1300935/) |
| Cortical pyramidal (range) | Rodent | 0.8–1.1    | Nucleated patch measurements |
| Spinal cord neurons     | Rodent  | ~0.9        | Gentet et al., 2000 |
| Hippocampal neurons      | Rodent  | ~0.9        | Gentet et al., 2000 |
| GABAergic interneurons   | Rodent  | ~0.9        | Modeling (Nörenberg et al., 2010) |
| L2/3 pyramidal neurons   | Human   | ~0.5        | [Eyal et al., 2016 (eLife)](https://elifesciences.org/articles/16553) |

**Key finding:** Human cortical neurons have ~0.5 µF/cm² — half the universal value. This enhances
synaptic charge-transfer and spike propagation speed.

### Total Whole-Cell Capacitance

| Neuron Type                    | Total Cm (pF)    | Notes |
|--------------------------------|-----------------|-------|
| Cortical pyramidal (L2/3)      | 100–300 pF      | Depends on dendritic arborization |
| Cortical pyramidal (L5, large) | 200–500 pF      | Extensive dendritic tree |
| Fast-spiking interneuron       | 15–50 pF        | Smaller soma, fewer dendrites |
| Computational model (pyramidal)| 500 pF (0.5 nF) | Wang, 2002 |
| Computational model (interneuron)| 200 pF (0.2 nF)| Wang, 2002 |

**Daily oscillation:** Pyramidal cell capacitance oscillates ~60–100% over the light-dark cycle
in mouse V1 ([PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11744780/)).
PV+ interneurons show NO such oscillation.

---

## 2. RESTING POTENTIAL & THRESHOLD POTENTIAL

### Resting Membrane Potential (Vrest)

| Neuron Type                       | Vrest (mV)       | Source |
|-----------------------------------|-----------------|--------|
| Generic cortical neuron           | −65 to −70      | Textbook consensus |
| L5 pyramidal (soma)               | −76.6 ± 0.3     | [Bhatt et al., 2018 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5866248/) |
| L5 pyramidal (dendrites)          | −77.0 ± 0.2     | Same study |
| L5 pyramidal (axon)               | −80.0 ± 0.4     | Same study (more hyperpolarized) |
| Fast-spiking interneuron          | ~−70            | Model convention |
| Full biological range             | −85 to −60      | Across neuron types |

### Threshold Potential (Vthresh)

| Neuron Type                       | Vthresh (mV)     | ΔV from rest | Source |
|-----------------------------------|-----------------|-------------|--------|
| Generic textbook neuron           | −55             | ~15 mV      | Classical |
| Cortical pyramidal (typical)      | −50 to −55      | ~15–20 mV   | Multiple sources |
| Fast-spiking interneuron (L2/3)   | −42.9 ± 5.7     | ~27 mV      | [Ohana et al., 2012](https://academic.oup.com/cercor/article/25/11/4415/2367212) |
| Human cortical neuron (estimated) | ~−35            | ~30 mV      | Eyal et al., 2016 |
| Computational model (Wang, 2002)  | −50             | 20 mV       | Wang, 2002 |

**Mapping note:** The gap between rest and threshold is typically **15–30 mV**.
This is the "charge bucket" that must be filled by EPSPs.

---

## 3. SINGLE EPSP AMPLITUDE

### Voltage Contribution

| Measurement Location | Amplitude (mV)  | Source |
|---------------------|-----------------|--------|
| Soma (typical unitary EPSP) | 0.2–1.0 mV | Multiple cortical studies |
| Soma (common estimate)      | ~0.5 mV     | [eNeuro, 2016](https://www.eneuro.org/content/3/2/ENEURO.0050-15.2016) |
| Dendritic spine (proximal)  | 13.0 mV (range 6.5–30.8) | Same eNeuro study |
| Spine-to-soma attenuation   | ~25× | Spine/soma ratio |
| GC-BC synapse (hippocampus) | 0.7–3.0 mV  | [Bhatt et al.](https://www.cell.com/neuron/fulltext/S0896-6273(00)80339-6) |
| Neuromuscular junction (comparison) | ~50 mV | Suprathreshold (1 input = 1 fire) |

### Synaptic Current & Charge

| Parameter                    | Value            | Source |
|-----------------------------|-----------------|--------|
| Unitary EPSC amplitude (AMPA) | 10–30 pA       | Cortical patch-clamp literature |
| EPSC duration (half-width)    | 2–5 ms         | AMPA-mediated |
| Estimated charge per EPSP     | 0.02–0.15 pC   | Current × duration |
| AMPA conductance per synapse  | 0.5–1.5 nS     | Human cortical: 0.88 nS (Eyal et al.) |
| NMDA conductance per synapse  | ~1.31 nS        | Human cortical (Eyal et al.) |

**Mapping note:** Each EPSP contributes ~0.5 mV at soma. Need ~30–60 simultaneous EPSPs
to bridge ~15–30 mV gap to threshold. This is the biological "spatial summation" count.

---

## 4. MEMBRANE TIME CONSTANT (τm)

τm = Rm × Cm — determines how fast the "bucket" leaks.

| Neuron Type                    | τm (ms)        | Source |
|-------------------------------|----------------|--------|
| Cortical pyramidal (in vitro) | 20–30          | [Koch, 1996 (Cerebral Cortex)](https://academic.oup.com/cercor/article-pdf/6/2/93/968752/6-2-93.pdf) |
| Cortical pyramidal (in vivo)  | 5–15           | Reduced by synaptic bombardment |
| Fast-spiking interneuron      | 8–15           | Smaller cells, less Rm |
| Historical early estimates    | 3–5            | 1950s recordings (underestimates) |
| Modern consensus range        | 10–40          | Depends on recording conditions |

### Charge Decay Per Millisecond

| τm (ms) | Charge remaining after 1 ms | Charge lost per ms |
|---------|---------------------------|-------------------|
| 10      | e^(−1/10) = 90.5%         | ~9.5%             |
| 15      | e^(−1/15) = 93.5%         | ~6.5%             |
| 20      | e^(−1/20) = 95.1%         | ~4.9%             |
| 30      | e^(−1/30) = 96.7%         | ~3.3%             |

**Mapping note:** Your capacitor neuron `leak=0.85` means 15% lost per tick.
Biological equivalent: τm ≈ 6 ms (fast, like an interneuron under heavy synaptic load).
`leak=0.90` → τm ≈ 10 ms (more pyramidal-like in vivo).

---

## 5. AFTERHYPERPOLARIZATION (AHP)

Three distinct components, each with different mechanisms and timescales:

### Fast AHP (fAHP)
| Parameter           | Value          | Mechanism |
|--------------------|---------------|-----------|
| Duration           | 2–5 ms        | BK (big-K) channels |
| Amplitude          | 5–10 mV below rest | Voltage + Ca²⁺ gated |
| Role               | Spike repolarization, enables fast firing | |

### Medium AHP (mAHP)
| Parameter           | Value          | Mechanism |
|--------------------|---------------|-----------|
| Duration           | 10–100 ms (peak at 50–125 ms) | SK2 channels, Kv7/KCNQ |
| Amplitude          | 5–10 mV below rest | Ca²⁺-dependent, NOT voltage-dependent |
| Role               | Sets interspike interval, spike-frequency adaptation | |

### Slow AHP (sAHP)
| Parameter           | Value          | Mechanism |
|--------------------|---------------|-----------|
| Duration           | 100 ms to several seconds (decay τ ~ seconds) | Second-messenger cascade |
| Amplitude          | 5–15 mV below rest | Via hippocalcin / neurocalcin δ |
| Role               | Prevents runaway firing, regulates excitability | |

### Quantitative AHP Values (Layer 5 Pyramidal)
| Condition          | Peak AHP (mV below rest) | Source |
|-------------------|------------------------|--------|
| Control (50 Hz train) | 8.1 ± 0.3 mV        | [Bhatt et al., J Neurosci 2013](https://www.jneurosci.org/content/33/32/13025) |
| With cadmium       | 7.3 ± 0.3 mV          | Same study (Na-pump mediated) |
| With BAPTA         | 9.8 ± 0.3 mV          | Same study |
| Equilibrium potential| 10–15 mV below rest   | Classical measurements |

**Mapping note:** After firing, membrane goes ~5–15 mV below rest for 2 ms to several seconds.
For a capacitor neuron, this means the reset value should be BELOW the baseline (not just back to 0).

---

## 6. REFRACTORY PERIODS

### Absolute Refractory Period (ARP)
| Source / Context     | Duration (ms)   |
|---------------------|----------------|
| CNS typical          | 0.5–1.0        |
| Broader range        | 0.4–2.0        |
| Computational models | 1–2            |
| Na⁺ channel full recovery | 3–4 ms   |

During ARP: **impossible** to fire another AP regardless of stimulus strength.
Mechanism: Na⁺ channel inactivation gates closed.

### Relative Refractory Period (RRP)
| Source / Context     | Duration (ms)   |
|---------------------|----------------|
| CNS typical          | ~10            |
| Range                | 2–15           |
| Some recent findings | up to 20+      |

During RRP: firing **possible** but requires stronger-than-normal stimulus.
Mechanism: Na⁺ channels recovering; K⁺ channels still partially open.

### Theoretical Maximum Firing Rate from ARP
| ARP (ms) | Max Hz   |
|---------|---------|
| 0.5     | 2000    |
| 1.0     | 1000    |
| 2.0     | 500     |

**Mapping note:** The absolute refractory period sets the hard ceiling on firing rate.
In practice, AHP and adaptation limit rates far below this ceiling.

---

## 7. ACTION POTENTIAL DYNAMICS

| Parameter               | Min    | Typical | Max    | Source |
|------------------------|--------|---------|--------|--------|
| Resting potential       | −85 mV | −70 mV  | −60 mV | Multiple |
| Threshold               | −55 mV | −50 mV  | −35 mV | Multiple |
| Peak voltage            | +20 mV | +30 mV  | +40 mV | [Wikipedia: AP](https://en.wikipedia.org/wiki/Action_potential) |
| Total AP amplitude      | 80 mV  | 100 mV  | 130 mV | Rest to peak |
| AP duration (half-width)| 0.3 ms | 1.0 ms  | 2.5 ms | Varies by cell type |
| Repolarization target   | −70 mV | −70 mV  | −70 mV | Returns to rest |
| Undershoot (AHP)        | −75 mV | −80 mV  | −85 mV | K⁺ overshoot |
| Reset (computational)   | —      | −60 mV  | —      | Wang, 2002 model |

### AP Sequence (cortical pyramidal)
1. **Rest:** −70 mV
2. **Threshold crossing:** −55 mV (sodium channels open)
3. **Rapid depolarization:** 0.2–0.5 ms to peak
4. **Peak:** +30 to +40 mV
5. **Repolarization:** K⁺ channels open, Na⁺ inactivate → 0.5–1.0 ms
6. **Undershoot:** −75 to −85 mV (K⁺ channels slow to close)
7. **Recovery to rest:** 2–5 ms back to −70 mV

---

## 8. SYNAPTIC INTEGRATION

### Spatial Summation (how many simultaneous inputs to fire)

| Parameter                        | Value          | Source |
|---------------------------------|---------------|--------|
| Single EPSP at soma              | ~0.5 mV       | Cortical consensus |
| Gap to threshold (rest→thresh)   | 15–30 mV      | Depends on neuron type |
| Estimated inputs needed          | **30–60 simultaneous** | 15 mV / 0.5 mV = 30 |
| Total synapses per neuron        | 5,000–10,000  | [NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK11104/) |
| Synapses on dendrites            | ~80%          | |
| Synapses on soma                 | ~20%          | Proximal = stronger effect |

### Temporal Summation

| Parameter                        | Value          | Source |
|---------------------------------|---------------|--------|
| EPSP duration                    | 15–20 ms      | Time for single EPSP to decay |
| Temporal summation window        | 15–20 ms      | Must arrive within this window |
| Effective window (in vivo)       | 5–10 ms       | Shorter due to synaptic bombardment |
| EPSP rise time                   | 1–3 ms        | AMPA-mediated |
| EPSP decay time                  | 5–20 ms       | τ-dependent |

### Integration Rules
- **Linear summation** at low input rates (subthreshold)
- **Sublinear** at high rates (conductance shunting)
- **Supralinear** in active dendrites (NMDA spikes, Ca²⁺ spikes)
- Dendritic location matters: proximal inputs ~25× more effective at soma than distal

---

## 9. FIRING RATES

### By Neuron Type

| Cell Type                 | In Vivo Typical | Maximum Rate | Source |
|--------------------------|----------------|-------------|--------|
| Cortical pyramidal        | 0.1–10 Hz      | 25–60 Hz    | [Frontiers, 2016](https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2016.00239/full) |
| Fast-spiking interneuron  | 10–110 Hz      | 200–450 Hz (mean) | Same |
| FS instantaneous max      | —              | 450–611 Hz  | Human/monkey |
| Average cortical neuron   | ~0.16 Hz       | —           | Energy budget estimate |
| Hippocampal pyramidal     | 1–2 Hz         | ~20 Hz      | In vivo rat |

### Species Comparison (Fast-Spiking Max)

| Species | Max Mean Freq | Max Instantaneous |
|---------|--------------|-------------------|
| Mouse   | 215 Hz       | 342 Hz            |
| Human   | 338 Hz       | 453 Hz            |
| Monkey  | 450 Hz       | 611 Hz            |

**Key insight:** ~90% of cortical neurons are "silent" (fire <0.1 Hz).
Active pyramidal cells typically fire 1–10 Hz. Only FS interneurons sustain >100 Hz.

---

## 10. LEAK CONDUCTANCE

### Values

| Parameter                        | Value          | Source |
|---------------------------------|---------------|--------|
| Cortical pyramidal gL (total)    | 10–50 nS      | [Scholarpedia: High-conductance state](http://www.scholarpedia.org/article/High-conductance_state) |
| Spinal motor neuron gL           | 300–1700 nS   | Much larger cells |
| HH model gL (specific)           | 0.3 mS/cm²    | Hodgkin & Huxley, 1952 |
| Leak reversal potential           | Near Vrest     | K⁺ dominated |

### Charge Loss Rate (derived from τm)

| Condition              | τm (ms)  | gL effective | % charge lost/ms |
|-----------------------|---------|-------------|-----------------|
| In vitro (quiet)       | 20–30   | Low         | 3–5%            |
| In vivo (active)       | 5–15    | High (synaptic) | 7–20%        |
| High-conductance state | 3–5     | Very high   | 20–33%          |

**In vivo reality:** The effective leak is NOT just passive channels. Ongoing synaptic
bombardment adds ~3× the resting conductance, dramatically shortening τm.

**Molecular basis:** TASK (K⁺), ClC-2 (Cl⁻), NALCN (Na⁺) two-pore leak channels.

---

## 11. IZHIKEVICH NEURON MODEL

### Equations
```
dv/dt = 0.04v² + 5v + 140 − u + I
du/dt = a(bv − u)
if v ≥ 30 mV → v ← c, u ← u + d
```

v = membrane potential (mV), u = recovery variable, I = input current.

### Parameters by Cortical Neuron Type

| Type                      | a    | b    | c (mV) | d   | Behavior |
|--------------------------|------|------|--------|-----|----------|
| **Regular Spiking (RS)**  | 0.02 | 0.2  | −65    | 8   | Spike-frequency adaptation |
| **Intrinsically Bursting (IB)** | 0.02 | 0.2 | −55 | 4 | Initial burst → RS |
| **Chattering (CH)**       | 0.02 | 0.2  | −50    | 2   | Fast rhythmic bursting |
| **Fast Spiking (FS)**     | 0.1  | 0.2  | −65    | 2   | No adaptation, high freq |
| **Low-Threshold Spiking (LTS)** | 0.02 | 0.25 | −65 | 2 | Rebound bursts |
| **Thalamo-cortical (TC)** | 0.02 | 0.25 | −65    | 0.05| Tonic/burst modes |
| **Resonator (RZ)**        | 0.1  | 0.26 | −65    | 2   | Subthreshold oscillations |

### Parameter Meaning
- **a:** Recovery time constant (larger = faster recovery). FS: 0.1 (fast), RS: 0.02 (slow)
- **b:** Sensitivity of u to v (subthreshold oscillations if b > a critical value)
- **c:** Reset voltage after spike. −65 = deep reset (RS), −50 = shallow reset (CH)
- **d:** After-spike reset of u. d=8 (strong adaptation), d=2 (weak adaptation)

Source: [Izhikevich, 2003 (IEEE Trans Neural Networks)](https://www.izhikevich.org/publications/spikes.pdf)

---

## 12. HODGKIN-HUXLEY MODEL PARAMETERS

### Original 1952 Squid Giant Axon Values

| Parameter         | Symbol | Value      | Units    |
|------------------|--------|-----------|----------|
| Membrane capacitance | Cm  | 1.0       | µF/cm²   |
| Max Na⁺ conductance  | ḡNa | 120       | mS/cm²   |
| Max K⁺ conductance   | ḡK  | 36        | mS/cm²   |
| Leak conductance      | gL  | 0.3       | mS/cm²   |
| Na⁺ reversal potential | ENa | +115*     | mV       |
| K⁺ reversal potential  | EK  | −12*      | mV       |
| Leak reversal potential | EL | +10.6*    | mV       |
| Axon radius            | a   | 238       | µm       |
| Axoplasm resistivity   | ρ   | 35.4      | Ω·cm     |

*Original HH convention: voltages as deviations from rest (depolarization positive).
In modern convention: ENa ≈ +50 mV, EK ≈ −77 mV, EL ≈ −54.4 mV, Vrest ≈ −65 mV.

### Conductance Equations
```
INa = ḡNa · m³ · h · (V − ENa)    (3 activation gates, 1 inactivation gate)
IK  = ḡK  · n⁴ · (V − EK)          (4 activation gates)
IL  = gL  · (V − EL)                (always open, constant)
```

### Key Ratios
- ḡNa / ḡK = 3.33 (Na much larger → fast depolarization)
- ḡNa / gL = 400 (active conductance >> leak)
- ḡK / gL = 120

### AP Properties from HH Model
- Spike amplitude: ~100 mV
- Spike half-width: ~2.5 ms (squid, slower than mammalian cortex)
- Resting potential: −65 mV (modern convention)

Source: [Hodgkin & Huxley, 1952](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model),
[Neuronal Dynamics (EPFL) Ch 2.2](https://neuronaldynamics.epfl.ch/online/Ch2.S2.html)

---

## 13. MAPPING GUIDE: BIOLOGY → CAPACITOR NEURON

### Direct Analogies

| Biological Parameter | Capacitor Neuron Analog | Biological Value | Your Model |
|---------------------|------------------------|-----------------|------------|
| Membrane capacitance | Charge accumulator capacity | 100–500 pF | Implicit in state variable |
| Resting potential → threshold gap | Threshold parameter | 15–30 mV | `threshold=0.3–0.5` |
| Leak (τm-based)     | `leak` parameter       | 3–10% lost/ms (in vivo) | `leak=0.85–0.90` |
| Single EPSP         | One input contribution | 0.5 mV (~2% of gap) | Weight × activation |
| Spatial summation count | Inputs to fire      | 30–60 simultaneous | Determined by threshold/weight ratio |
| AHP reset           | Post-fire charge reset | 5–15 mV below rest | Reset to 0 (or negative) |
| Refractory period   | Post-fire dead time    | 1–2 ms absolute | Could add 1-tick dead time |
| Temporal summation window | Integration ticks | 15–20 ms (5–10 in vivo) | ~3–5 ticks at leak=0.85 |

### Effective Temporal Window (ticks until charge decays to 1/e)

| leak param | Effective τ (ticks) | If 1 tick = 1 ms | If 1 tick = 5 ms |
|-----------|-------------------|-----------------|-----------------|
| 0.80      | 4.5 ticks          | 4.5 ms          | 22.5 ms         |
| 0.85      | 6.2 ticks          | 6.2 ms          | 31 ms           |
| 0.90      | 9.5 ticks          | 9.5 ms          | 47.5 ms         |
| 0.95      | 19.5 ticks         | 19.5 ms         | 97.5 ms         |

Formula: τ_eff = −1 / ln(leak)

### What Biology Suggests for Your Model
1. **leak=0.85–0.90** is biologically plausible for in-vivo cortical neurons (τm ≈ 6–10 ms)
2. **threshold=0.3–0.5** is reasonable — biology requires ~30–60 inputs out of thousands
3. Consider adding **negative reset** after firing (AHP analog) — go below 0, not just to 0
4. Consider **refractory period** — 1–2 tick dead time after firing
5. **Bipolar variant** (cap_bipolar) maps to excitatory/inhibitory balance in biology
6. FS interneurons recover ~5× faster than pyramidal cells (a=0.1 vs 0.02 in Izhikevich)

---

## SOURCES

1. [Gentet et al., 2000 — Direct measurement of specific membrane capacitance](https://pmc.ncbi.nlm.nih.gov/articles/PMC1300935/)
2. [Eyal et al., 2016 — Unique membrane properties in human neocortical neurons (eLife)](https://elifesciences.org/articles/16553)
3. [Bhatt et al., 2018 — Differential control of resting potential (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5866248/)
4. [Koch, 1996 — A brief history of time constants (Cerebral Cortex)](https://academic.oup.com/cercor/article-pdf/6/2/93/968752/6-2-93.pdf)
5. [Izhikevich, 2003 — Simple model of spiking neurons (IEEE)](https://www.izhikevich.org/publications/spikes.pdf)
6. [Hodgkin & Huxley, 1952 — HH model (Wikipedia summary)](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model)
7. [Neuronal Dynamics (EPFL) — Ch 2.2 HH Model](https://neuronaldynamics.epfl.ch/online/Ch2.S2.html)
8. [Frontiers, 2016 — Firing frequency maxima of fast-spiking neurons](https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2016.00239/full)
9. [NCBI Bookshelf — Summation of synaptic potentials](https://www.ncbi.nlm.nih.gov/books/NBK11104/)
10. [Scholarpedia — High-conductance state](http://www.scholarpedia.org/article/High-conductance_state)
11. [Ohana et al., 2012 — Synaptic conductance estimates (Cerebral Cortex)](https://academic.oup.com/cercor/article/25/11/4415/2367212)
12. [eNeuro, 2016 — EPSPs in proximal dendritic spines](https://www.eneuro.org/content/3/2/ENEURO.0050-15.2016)
13. [Bhatt et al., 2013 — Na-pump-mediated AHP (J Neurosci)](https://www.jneurosci.org/content/33/32/13025)
14. [Daily oscillations of neuronal membrane capacitance (Cell Reports, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11744780/)
15. [PhysiologyWeb — Refractory periods](https://www.physiologyweb.com/lecture_notes/neuronal_action_potential/neuronal_action_potential_refractory_periods.html)
16. [AI Impacts — Neuron firing rates in humans](https://aiimpacts.org/rate-of-neuron-firing/)
17. [Wikipedia — Action potential](https://en.wikipedia.org/wiki/Action_potential)
18. [StatPearls — Physiology, Action Potential](https://www.ncbi.nlm.nih.gov/books/NBK538143/)
