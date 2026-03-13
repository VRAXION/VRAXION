# A CSUKLYÁS — Hullámok és Rezonancia (Brainstorm)

**Forrás:** Notion Introspekciós Napló + HORN Paper Analízis (2026-03-13)
**Státusz:** Brainstorming — NEM validált, tesztelésre vár

---

## A Felfedezés

Minden ide vezet vissza: **rezonancia, agyhullámok, aktivációs hullámfüggvények, sinus görbék, interferencia.**

A rendszer NEM egy gráf amin jelek terjednek. A rendszer egy **KÖZEG** amin **HULLÁMOK** terjednek és **INTERFERÁLNAK**.

### A bizonyítékok (meglévő v22 eredményekből):

| Jelenség | Hullám értelmezés |
|----------|-------------------|
| Ternary mask (+1/0/-1) | konstruktív interferencia / nincs jel / destruktív interferencia |
| Capacitor neuron | hullám akkumuláció időben |
| 64-class fal | destruktív interferencia (minták kioltják egymást) |
| Dream output [0,0,0,1,0,0] | tökéletes konstruktív interferencia EGY ponton |
| 70/30 forward/backward | állóhullám (oda-vissza verődés) |
| Flip mutáció (+1↔-1) | FÁZIS megfordítása (a legsikeresebb mutáció!) |
| "Budapest" azonnali felismerés | REZONÁNS FREKVENCIA a közegben |
| "17+28" nem azonnali | NINCS rezonáns frekvencia, lépések kellenek |
| Delta hullámok ébren | lassabb hullámfrekvencia = mélyebb integráció |

### Implikációk az architektúrára:

- A connectionok nem "vezetékek" hanem a **KÖZEG TULAJDONSÁGAI**
- A neuronok nem "feldolgozó egységek" hanem **REZONÁTOROK**
- A tickek nem "feldolgozási lépések" hanem **HULLÁM CIKLUSOK**
- A mutáció nem "connection módosítás" hanem **KÖZEG HANGOLÁS**
- A fázis (sign) fontosabb mint az amplitúdó (weight) — ezért nyert a ternary mask!
- **A tanulás = a közeg hangolása** úgy hogy a helyes inputra konstruktív interferencia legyen a helyes outputon, és destruktív mindenhol máshol.

---

## HORN Paper Analízis

**Paper:** "The functional role of oscillatory dynamics in neocortical circuits: A computational perspective"
**Szerzők:** Felix Effenberger, Pedro Carvalho, Igor Dubinin, Wolf Singer (Ernst Strüngmann Institute, Frankfurt)
**Megjelenés:** PNAS, 2025 január

### Mi a HORN?

Harmonic Oscillator Recurrent Network — rekurrens hálózat ahol minden neuron egy **csillapított harmonikus oszcillátor (DHO)**. Három paraméter per neuron:
- **ω (omega):** természetes frekvencia
- **γ (gamma):** csillapítás
- **α (alpha):** gerjeszthetőség

A HORN **VERI** az összes nem-oszcilláló architektúrát (tanh RNN, GRU, LSTM) tanulási sebességben, zajtűrésben és paraméter hatékonyságban.

### Egyezések a mi rendszerünkkel

| HORN | VRAXION v22 | Megjegyzés |
|------|-------------|------------|
| DHO oszcillátor neuron | Capacitor neuron | Hasonló! A capacitor = primitív DHO |
| Fázis kódolás | Ternary mask (+1/-1) | Ugyanaz! A +1/-1 = fázisban/ellenfázisban |
| Állóhullámok | Tesztelve és detektálva | 22/48 neuron mindig csendes = csomópont |
| Interferencia minták | Konstruktív interferencia mérve | Tesztelt és igazolt |
| Oszcilláció a tickekben | UP/DOWN/UP mintázat mérve | Tesztelt és igazolt |
| Fázis > amplitúdó | Flip mutáció 1.5x erősebb hatás | Tesztelt és igazolt |
| Rekurrens topológia | Flat gráf 70/30 fwd/bwd | Hasonló arány |

### Ami a HORN-ban van de NÁLUNK HIÁNYZIK

**1. Explicit DHO neuron model:**
A mi capacitor neuronunk primitív — nincs benne explicit frekvencia (ω). A HORN DHO-ja gazdagabb: frekvencia + csillapítás + gerjeszthetőség. Az explicit frekvencia azt jelenti, hogy a neuron REZONÁL bizonyos input frekvenciákra és másokat elszűr.
→ **TODO: capacitor → DHO upgrade**

**2. Heterogén frekvenciák:**
A HORN-ban minden neuron MÁS frekvenciával oszcillál. Ez növeli a dimenzionalitást és a memória időskálákat. Nálunk minden neuron UGYANAZZAL a leak rate-tel dolgozik.
→ **TODO: minden neuron kap egy saját ω-t, evolúció válogatja**

**3. Vezetési késleltetések (conduction delays):**
A connectionokön a jel nem azonnal hanem késleltetve érkezik. Ez még több időbeli struktúrát ad.
→ **TODO: delay per connection, evolúció válogatja**

### Ami NÁLUNK van de a HORN-ban NINCS

**1. Backprop-mentes tanulás:**
A HORN BPTT-t (backprop through time) használ! Mi mutáció + szelekció. Ez a FŐ KÜLÖNBSÉG. Ha a HORN eredményeit el tudjuk érni backprop nélkül, az áttörés.

**2. Topológia evolúció:**
A HORN fix topológiát használ (all-to-all), a weight-eket tanítja. Mi a TOPOLÓGIÁT evolváljuk — connectionoköt ad/vesz/rewire. Ez biológiailag hiteleseebb (szinaptogenezis + prunálás).

**3. Ternary mask (fázis a topológiában):**
A HORN-ban a fázis a WEIGHT-ben van. Nálunk a MASK-ban — a topológia MAGA hordozza a fázis információt. Ez egyszerűbb és hardware-barátabb.

**4. Introspekció-alapú tervezés:**
Ők fizikából és neurobiológiából indultak ki. Mi a BELSŐ TAPASZTALATBÓL. Az eredmények konvergálnak — az introspekció megbízható forrás.

---

## Következő Lépések (TODO — tesztelésre vár)

1. **Capacitor → DHO neuron** (ω, γ, α paraméterekkel)
2. **Heterogén ω**: minden neuron más frekvenciával, evolúció válogatja
3. **Conduction delay**: per-connection késleltetés
4. Mindez **BACKPROP NÉLKÜL** — mutáció + szelekció + combined scoring
5. Ha ez működik: a HORN benchmark-okon (sMNIST) összehasonlítani

---

## Valós Tudományos Pozíció

A hullám-alapú neurális számítás ISMERT terület (Singer 2025, Marcucci 2020, Papp 2021).

Ami **POTENCIÁLISAN ÚJ**:
- Hullám-közeg evolúciós hangolása (backprop helyett)
- Topológia evolúció + hullám interferencia kombinációja
- Ternary mask mint fázis kódolás a topológiában
- Introspekció-alapú architektúra tervezés (neurophenomenológia)
- Combined scoring (accuracy + target_prob) mint evolúciós fitness
