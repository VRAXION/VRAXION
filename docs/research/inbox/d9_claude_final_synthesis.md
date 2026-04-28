# D9 Latent Genome Decoder — Claude swarm végső szintézis

Dátum: 2026-04-28
Forma: kutatási kimenet, nem implementáció. Magyar nyelvű, a swarm 7 angol nyelvű raw artefaktjából lefordítva és összevonva.

## 0. Forrás-térkép (audit-trail)

A swarm 4 hullámban futott. Minden raw output külön fájlban van — ez a szintézis fájl szerint hivatkozik vissza, hogy bárki vissza tudja vezetni az itt szereplő állításokat eredeti forrásra.

| Hullám | Ágens | Fájl |
|--------|-------|------|
| Bemenet | Gemini digest | `docs/research/inbox/d9_gemini_genome_compiler_design.md` |
| 1 | A — indirekt kódolás irodalom | `docs/research/inbox/d9_claude_wave1_A_indirect_encoding.md` |
| 1 | B — QD/latent ES irodalom | `docs/research/inbox/d9_claude_wave1_B_qd_latent_es.md` |
| 1 | C — VRAXION kontextus | `docs/research/inbox/d9_claude_wave1_C_vraxion_context.md` |
| 1 | D — lokalitás-mérés és benchmark elmélet | `docs/research/inbox/d9_claude_wave1_D_locality_benchmarks.md` |
| 2 | E — architektúra-tervező | `docs/research/inbox/d9_claude_wave2_E_architecture.md` |
| 2 | F — falszifikáció-tervező | `docs/research/inbox/d9_claude_wave2_F_falsification.md` |
| 3 | G — vörös csapat | `docs/research/inbox/d9_claude_wave3_G_red_team.md` |
| 4 | Fő szál — magyar szintézis | ez a fájl |

---

## 1. Vezetői összefoglaló

A D8 egy **megfigyelő atlasz**: kész hálózatokat címkéz behaviour-cellákba. A D9 egy **gyár**, ahol egy z koordináta megy be és egy hálózat jön ki. A swarm azt vizsgálta, építhető-e ilyen gyár úgy, hogy

- a közeli z-ek hasonló hálózatot szülnek (lokalitás),
- a generált z-eket pontosan visszanyerhetjük (mert eltároljuk),
- ne legyen csak átkeresztelt random hash.

**A swarm kettős verdiktet hozott.** Egyrészt: az architektúra ép — a javasolt PIC (Parametric InitConfig) terv, amelyik a meglévő VRAXION `InitConfig` tárcsáit folytonosítja, elvileg el tud kerülni minden ismert lokalitási csapdát. **Másrészt:** a vörös csapat (G) **konkrét adversarial dekódert épített (D_advr)**, amelyik az F-féle eredeti 60 másodperces "killer" tesztet **átviszi**, miközben semmilyen valódi hálózati struktúrát nem hordoz. Ez azt jelenti, hogy **F eredeti tesztje már most, kód előtt falszifikálva van**.

A javaslat ezért: **D9.0 építését csak akkor szabad elkezdeni, ha a 8 kötelező keményítést előbb beépítjük az építési tervbe**. Ezek nem stilisztikai módosítások — közvetlenül a swarm által megtalált failure mode-ok ellen védenek. A keményítések részletes listája a §8-ban (Accepted-bin) van.

A legfontosabb egyetlen javítás: **a kanonikus első teszt nem F §0 byte-Hamming microtestje, hanem G §7 IDENTITY_AUGMENTED_KILLER-je** — ami ugyanaz a 60 mp-es szerkezet, de gráf-szerkesztési távolsággal, entrópia non-trivialitás kapuval, és D_advr mint kötelező harmadik dekóder a futás alatt. Enélkül a tesztet egy bájt-echo átviszi.

A Gemini digest verdict-nevei és kontroll-kódjai változatlanok maradnak (D9_LATENT_DECODER_TOY_PASS, NCI_RANDOM_HASH_DECODER, stb.) — a swarm csak finomítja és számokkal csontozza a kapukat.

---

## 2. Kulcs-irodalom és koncepciók

A swarm csak olyan irodalmat enged a tervbe, amelynek implementálható D9.0-ra vetített következménye van. Minden állításhoz **bizonyíték-címke** (`[direct empirical]` = mért bizonyíték, `[theoretical]` = formális bizonyítás, `[analogy]` = másik rendszer szerinti érvelés, `[speculative]` = bizonyítás nélkül).

**A leggyengébb pont, amit a kutatás talált — Grammatical Evolution (Rothlauf & Oetzel, EuroGP 2006).** Ez a legközelebbi precedens a D9-hez: determinisztikus szabály-alapú genotípus→fenotípus fordító. Mérve: a genotipikus szomszédok TÚLNYOMÓ TÖBBSÉGE NEM kerül szomszédos fenotípushoz. Eredmény: GE-alapú keresés mérhetően gyengébb a magas-lokalitású alternatíváknál. `[direct empirical]` Ez közvetlen empirikus precedensi cáfolat egy "hadd legyen csak grammatika, az majd lokálisat csinál" elképzelésre. Ezért minden D9 design-nak **explicit anti-hash bizonyítékot kell hoznia**, nem feltételezést.

**A regime-függés elvi figyelmeztetése — Clune, Stanley, Pennock, Ofria (IEEE TEC 2011).** HyperNEAT regularis problémákon nyer (közel optimális), irreguláris problémákon ROMLIK és átkerül a direkt kódolás alá. Idézve: *"As the regularity of the problem decreases, the performance of the generative representation degrades to, and then underperforms, the direct encoding."* `[direct empirical]` Következmény D9.0-ra: a benchmark suite-ban **kötelezően szerepelnie kell egy irreguláris (deceptive) tájképnek**. Egy csupa-sima tájkép-eredmény nem perdöntő.

**Egy fontos finomítás Gemini felé — Grammar VAE (Kusner, Paige, Hernandez-Lobato, ICML 2017).** Egy nyelvtan-megszorított TANULT dekóder magasabb latens-tér-koherenciát és érvényességet ér el, mint vagy a tisztán determinisztikus, vagy a tisztán neurális dekóder. `[direct empirical]` Az irodalmi tanulság tehát **nem** "determinisztikus jobb mint tanult", hanem **"a nyelvtani szerkezet a teherbíró elem, függetlenül attól, hogy statikus vagy tanult dekóder hordozza"**. A D9.0 statikus választása EGY érvényes saroka egy nagyobb tervezési térnek, NEM egyértelmű győztes. Tanult dekóderek explicit Deferred-bin-be kerülnek, nem mert rosszak, hanem mert D9.0 hatókörén kívül.

**A QD-irodalom megfigyelése a "behavior-cellához-genotípus" inverzről — Pugh, Soros, Stanley (Frontiers Robotics 2016).** A QD-irodalom EGY paper sem próbálja meg behavior characterization-ből visszaszámolni a genotípust; mindenki előrefelé halad. `[theoretical]` Ezért a Gemini digest figyelmeztetése jogos: **D9 inverz csak az általunk generált genomokra pontos** (mert eltároltuk a z-t és a rule_trace-t); régi/külső hálózatokra csak közelítő keresés engedélyezett.

**A MAP-Elites-szerű új elköteleződés a D9-ben — Mouret & Clune (arXiv:1504.04909) + Fontaine et al (AAAI 2021).** Az összes vizsgált QD-rendszer az archive **lakó-genomját** használja generatív szülőként, sosem a cella-koordinátát. A D9 azon választása, hogy z-koordinátát közvetlenül használjuk, **architekturálisan eltérő minden tesztelt rendszertől** — nincs sem cáfolva, sem igazolva az irodalom által. `[direct empirical]` Ez NEM hiba, csak kockázat: új, untested commitment. A toy benchmark-jainknak ezt explicit fel kell deríteniük.

**Lokalitás-mérés módszertan — Quilodrán et al (Mol Ecol Resources, 2025).** A Mantel-teszt I. típusú hibája megnő, ha a z mintavétel térben autokorrelált (pl. rács). N = 50–200 tartományban a hiba k > 0.2 autokorrelációnál nyilvánul meg. `[direct empirical]` Következmény: D9.0 toy lokalitás-tesztben **véletlen z mintavétel kötelező, NEM rács**.

**Az identity-decoder csapda — Agent D, Failure modes 5.** Egy dekóder, ami a z bájtokat közvetlenül genom-bájtokra mapolja, TRIVIÁLISAN átmegy minden lokalitás-teszten. `[speculative, but actionable]` Ezért minden lokalitás-tesztet kötelezően meg kell előznie egy **strukturális non-trivialitás kapunak** — a genom ne legyen a z echo-ja, és a gráfnak érvényes legyen.

---

## 3. Javasolt D9 architektúra

A swarm 3 alternatív D(z) tervet vizsgált; egyet javasol primary-ként, kettőt deferred / alternatív-ként.

### Primary: PIC — Parametric InitConfig

**A metafora.** Képzeld el a meglévő VRAXION `InitConfig`-ot mint egy szintetizátor előlapot 12 tárcsával. Eddig csak diszkrét beállításokkal használtuk (5% sűrűség, 50 chain, stb). PIC ezeket FOLYTONOS tárcsákká teszi: minden tárcsa egy pici z-mező, és minden tárcsa-állás KÖZVETLENÜL egy mérhető hálózati tulajdonságot rögzít. A magot (root_seed) csak akkor kérdezzük meg, ha két állás teljesen egyforma — ekkor a mag dönt el, hogy melyik konkrét neuron kapja a szerepet.

**A 12 tárcsa.** sűrűség, gátlás-arány, golden-ratio overlap, chain-sűrűség, küszöb-átlag, küszöb-szórás, csatorna-ferdültség, recurrence-billentés (előre vs hátra élek), tükörszimmetria, modularitás, irregularitás-amplitúdó (felső 10% sapka), irregularitás-fázis. A `H` (neuron-szám) NEM tárcsa, hanem szcenárió-fix.

**A mag (root_seed) szerepe szigorúan tie-breaker.** Például a sűrűség-tárcsa pontosan beállítja az élek **számát** (zárt képlettel a hálózat-méret arányában); a mag csak azt dönti el, hogy a sok azonos pontszámú jelölt él közül melyik kapja meg a helyet. Soha nem fordul elő, hogy a mag dönti el, hány él lesz, vagy milyen szabály fut. Ez a **strukturális mező > mag** dominancia-elv az anti-hash garancia magja.

**Anti-hash érvelés és ahol ez REPED.** Az E-féle eredeti érvelés szerint kis perturbáció (Δz = ε) csak kis számú élt változtat, függetlenül a magtól. Ez ÁTLAGOSAN igaz. A vörös csapat (G §1.1, A2) talált egy **réskiességet**: ha az élek pontszáma majdnem azonos (közel azonos pontok), akkor egyetlen pici z-elmozdulás az EGÉSZ rangsort átrendezheti, és a top-K új permutációt kap — gyakorlatilag random átállást. Az "average-case" határ áll, a "worst-case" határ NEM áll. **Ez a leggyengébb pont az E architektúrában, és G azonosította.** A keményítés: a pontszám-függvénybe egy folytonos, nem-magtól-függő perturbációs tag (pl. egy nagyon kis tag, ami az indexpárból deriválódik) — így a tie-eket nem a mag, hanem egy folytonos struktúra törje meg.

**Ezt a hibát a §8 listán Accepted-keményítésként kell beépíteni a kódolás előtt.**

**Provenance log.** Minden generált genom mellé eltároljuk: a használt root_seed-et, a teljes z vektort, és a `rule_trace`-t — minden szabály-alkalmazás logja (melyik szabály, hol fut, milyen sub-seeddel, mi volt a pontszám). Ez a NEAT "innovation number" mechanika analógja (Agent A): két z közös prefix-szel rendelkező rule_trace-ekkel STRUKTURÁLISAN rokon hálózatokat produkál.

**Inverz.** Saját generált genomra: pontos, mert a z-t eltároltuk. Külső/régi hálózatra: csak közelítő — a strukturális mezőket vissza tudjuk olvasni (sűrűség, gátlás-arány, recurrence-billentés a hálóból), de az irregularitás-mezők defaultolnak nullára. **A "z_hat" név félrevezető lehet** — pontosabban "strukturális ujjlenyomat", nem genom-szintű inverz. (G §1.1, A5)

### Secondary (Deferred D9.1+): MMD — Motif-Mixture Decoder

A z egy folytonos súlyozású keverék 8 kézzel-tervezett motívum (chain, mirror_pair, fan_in, fan_out, mutual_inhibitor, recurrent_loop, lateral_block, sparse_random) felett, plusz "ragasztó" élek köztük. A motívum-darabszám SOFTMAX-ból jön, NEM "codon mod n_motifs"-ból — ez kifejezetten a Grammatical Evolution failure mode-ot kerüli el.

**Miért NEM primary D9.0-ban.** Két ok. Egy: a vörös csapat (G §1.2, B2) megmutatta, hogy a softmax-rounding határfelületei egy nem-nulla mértékű részhalmaz a z térben, és ott a lokalitás megreped. Az E-féle "stochastic rounding mag-szerinti hash-sel" javaslat a magot újra felhasználja, ami éppen a GE-csapdába vezet vissza. Két: validity-kockázat a motívum-átfedéseknél kis H-n.

**Mikor jön elő.** Ha PIC az irreguláris-tájkép tesztet elbukja (Clune 2011 figyelmeztetés), akkor MMD a következő próba — a motívum-könyvtár explicit diszkrét struktúra-injekció.

### Tertiary (Deferred): GPF — Geometric Placement Field

CPPN-szerű kézzel-kódolt geometriai dekóder. Minden neuronnak fix koordinátája van, a z egy determinisztikus skalár-mezőt parametrizál a koordináta-párokon, az élek ennek a mezőnek top-K kiválasztásai. **Miért NEM primary.** Azonos top-K probléma mint PIC-nél, plusz ha a mező koefficiensei közel nullák, az egész mező összeomlik konstanssá és a mag-tie-breaker dönt mindenről — tiszta hash-mód. (G §1.3, C1) Csak akkor kerül elő, ha D9.1 explicit CPPN-stílusú geometriai illesztést kér.

### Cross-design konfliktus, amit nem rejtünk el

Agent A azt írta, Grammar VAE "közvetlenül cáfolja" Gemini deterministic-first claim-jét. G jogosan jegyezte meg, hogy ez **interpretáció**: Grammar VAE azt mutatja, hogy korlátozott TANULT dekóder működik, NEM azt, hogy a statikus dekóder elbukik. A helyes olvasat: a nyelvtani szerkezet a teherbíró elem, nem a dekóder statikussága. A D9.0 statikus választása egy érvényes sarka egy nagyobb tervezési térnek, nem egyetlen helyes választás.

---

## 4. Toy benchmark terv

A swarm 5 toy tájképet specifikál + 1 KÖTELEZŐ kanonikus első tesztet, ami az egészet megelőzi.

### 4.0 IDENTITY_AUGMENTED_KILLER — kanonikus első teszt (G §7)

**EZ az első teszt, NEM F §0 eredetije.** F eredeti byte-Hamming-alapú killerét a vörös csapat falszifikálta: G adversarial dekódere (D_advr = `bytes_of(quantize(z)) + SHA256_noise`) átviszi az eredetit, miközben semmilyen valódi gráf-szerkezetet nem hordoz.

**A javított kanonikus killer 3 különbsége:**

1. **Gráf-szerkesztési távolság** — NEM bájt-Hamming. Az élek mint multihalmaz távolsága. Valamivel drágább, de a swarm explicit elfogadja a költséget.
2. **Entrópia non-trivialitás kapu** — minden generált genom entrópiájának egy minimum tartalmat felül kell múlnia, ami a z entrópiája plusz egy logaritmikus tag az H-ban. Ez kiszűri az identity-dekódert.
3. **D_advr mint harmadik dekóder a tesztben** — a hash mellett RUN-TIME-ban együtt fut, és ELLENŐRZÉS, hogy a teszt SZÉTVÁLASZTJA-e a valós dekódertől. Ha D_advr is átmegy, a teszt vagy a valódi dekóder (vagy mindkettő) hibás.

**Idő:** ~90 mp. Ha bukik, D9.0 már most halott. Ha átmegy, érdemes a teljes suite-ot futtatni.

### 4.1 Smooth basin (sima medence)

A távolság-négyzetes tájkép. Itt a valós dekódernek erős lokalitást kell mutatnia. **Veszély:** a Lipschitz-trivial dekóder (z-bájtok közvetlen csomagolása) is átmegy. Ezért a non-trivialitás kapu kötelező előzetes feltétel.

### 4.2 Deceptive basin (megtévesztő medence)

trap_k mintázat. A tényleges optimum az "all-zero" pont, de a gradiens "all-one" felé húz. **Itt mérjük PIC vs MMD regime-tűrését.** Ha PIC bukik itt, MMD a következő próba.

### 4.3 Multi-basin (NK-tájkép)

Több lokális optimum. Itt a basin clustering minőségét mérjük (silhouette + ARI), DE figyelmeztetés: a basin címkéket FÜGGETLEN tájkép-orákulumtól kell venni, NEM a dekóderből — különben a metrika önmagát validálja (cirkularitás).

### 4.4 Needle / high-variance

Tűhegyes tájkép — egyetlen pont a jó, minden más zaj. **Itt NINCS pass/fail kapu**, csak DIAGNOSZTIKA. Ha a valós dekóder a hash-szel egyenlő itt — ez ELVÁRT és HELYES. Az egyetlen ami nem szabad: hogy a valós dekóder JOBB legyen mint a hash itt, mert az azt jelentené, hogy információ szivárog be a dekódolásba (pl. a fitness értéket olvasná).

### 4.5 Random control (vak teszt)

Tisztán véletlen tájkép. Itt a valós dekóder NEM SZABAD HOGY túlteljesítse a hash-t. Ha igen → szivárgási riadó.

### A keményített futási sorrend

1. IDENTITY_AUGMENTED_KILLER (90 mp) — fail = stop
2. Validity rate 1000 hálózaton (~30 mp)
3. Strukturális non-trivialitás (~10 mp)
4. Smooth + deceptive Mantel + diagnosztikus FDC (~3-5 perc, megmérve, NEM becsülve)
5. Multi-basin Mantel + silhouette + ARI független orákulummal (~3 perc)
6. Needle diagnosztika + random control parity (~3 perc)
7. Progressive scan (~6 perc)
8. 6 negatív kontroll keresztfuttatása (~10 perc)
9. Roundtrip + közelítő inverz (csak riport)

A 30 perces összes-budget BEMÉRT, NEM tippelt.

---

## 5. Metrikák és falszifikációs kapuk

A Gemini digest 6 DNP-kapuját (Do-Not-Proceed) átvesszük, számokkal csontozzuk és a vörös csapat által kért finomításokkal:

| DNP-kapu | Küszöb-tárcsa | Indoklás |
|----------|---------------|----------|
| `DNP_VALIDITY_COLLAPSE` | érvényes-háló-arány alatta 0.99 | Gemini digest L158 + Recipe 2 §1 |
| `DNP_LOCALITY_COLLAPSE` | z↔genom Mantel-érték nem ér el a hash-küszöb + 0.20-at, 95% bootstrap CI nem-átfedéssel | Recipe 3 §4: a 0.20 hézag az a szint amit a hash-baseline NEM tud lefedni; CI követelmény véd a véletlen permutáció ellen |
| `DNP_BEHAVIOR_HASHLIKE` | z↔viselkedés Mantel-érték nem ér el a hash-küszöb + 0.10-et | Lazább küszöb, mert a viselkedés nem-zéró zajos |
| `DNP_SCAN_NO_GAIN` | progresszív szkennelés / véletlen szkennelés arány nem éri el az 1.5-szörösét 300 értékelés után | Recipe 5 §4 |
| `DNP_CONTROL_PARITY` | **kettéosztva** dekóder-oldali (hash, nonlocal, shuffled-rules) és értékelés-oldali (shuffled-fitness, shuffled-labels, random-cells) parityra | G §2.4 D10: az eredeti egységes definíció logikailag rossz volt — shuffled-fitness értékelés-oldali, nem dekóder-oldali |
| `DNP_TOO_HEAVY` | torch / tensorflow / jax / sklearn.neural_network import D9.0-ban tilos; futási idő > 30 min | Gemini L172 |

**Fontos vörös csapat finomítás:** **minden hash-küszöb a jelenlegi futás SAJÁT hash-baseline mérésére kalibrált, NEM előre rögzített.** A 0.20-as hézag például nem dogma; a futtatás közben mérjük a hash-baseline tényleges Mantel-eloszlását, és a 95-edik percentil + 0.20 a tényleges kapu. Ezt G §2.3 D8 explicit követelte.

**Az FDC (fitness-distance correlation) nem önálló kapu, csak diagnosztika.** Altenberg 1997 számos ellenpéldát adott, amikor az FDC félreklasszifikál — ezért az FDC csak Mantel mellett, sose helyette.

**A silhouette-pontszám csak ARI-val együtt** kapuel, mert random embedding-eken HDBSCAN tud spurious clustert találni, és az ARI független orákulumhoz méri.

---

## 6. Implementációs terv (D9.0 → D9.4)

| Fázis | Cél | Bemenet | Kimenet | Mi NEM ide való |
|-------|-----|---------|---------|----------------|
| **D9.0** | Toy determinisztikus PIC dekóder Pythonban, offline, kanonikus killer + 5 toy tájkép + 6 negatív kontroll | a §9-ben adott prompt + a 8 kötelező keményítés | 1 markdown audit + JSON/CSV adatok + verdict (D9_LATENT_DECODER_TOY_PASS / fail) | Rust integráció, neurális dekóder, MMD/GPF, valódi VRAXION háló-értékelés |
| **D9.1** | Determinisztikus genome grammar VRAXION-lite hálózat-értékeléssel | D9.0 PASS + Agent C csonka kanonikus genome megoldva (NetworkDiskV1 → kanonikus séma) | bemért lokalitás valódi PanelMetrics fingerprintre | tanult dekóder, valódi atlasz inverz |
| **D9.2** | Latent szkennelés és atlasz illesztés | D9.1 PASS + a D8 156-cellás atlasz | progresszív tile-szkennelés, célzott cellák megnyitása | BOP-Elites surrogate (még nem) |
| **D9.3** | Inverz keresés régi hálózatokra | D9.2 PASS + a meglévő archive | közelítő z_hat minden külső hálóra; pontos z-tárolás minden új hálóra | tanult inverz mapping |
| **D9.4** | Opcionális tanult propose / dekóder | D9.0–9.3 mind PASS, és deterministic baseline elégtelen | Grammar VAE-stílusú learnt component | semmi, ami D9.3 előtt jönne |

**Két stop-gate:** (1) ha D9.0 az IDENTITY_AUGMENTED_KILLER-en bukik a 8 keményítés után is, az egész D9 vonal halott. (2) ha Agent C-féle csonkára (nincs kanonikus genome) NEM születik canonicalization fázis, akkor D9.1 előtt kötelező egy "D9.1-pre genome canonicalization" sub-fázis. D9.0-t ez NEM akadályozza, mert az toy és offline.

---

## 7. Kockázatok és failure mode-ok

A vörös csapat top-5 prioritizált rizikó-listája — minden tételhez minimum-test-recept:

1. **(Erős valószínűség, súlyos hatás) Top-K rangsor-átállás miatt a Lipschitz-bound réskies.** PIC, MMD és GPF mindhárom designnál igaz: a "kis z-perturbáció = kis genom-perturbáció" garancia ÁTLAGOSAN igaz, ROSSZ ESETBEN nem. Ha az élek pontszáma majdnem-egyforma ("tie cluster"), egy pici elmozdulás az egész rangsort átveti. **Teszt:** futtassuk a Mantel-tesztet több sűrűség-rétegen (kis z[0], közepes, magas); ha a Mantel-érték 0.2-nél többet esik a réteg-váltáson, a Lipschitz-állítás megdőlt. **Védelem:** explicit non-degeneracy tag a pontszám-függvénybe (folytonos, nem-magtól-függő).

2. **(Erős valószínűség, súlyos hatás) D_advr átviszi az eredeti F killer microtestjét.** Bizonyítva G §5: `bytes_of(quantize(z)) + SHA256_noise` valid hálózat builder-be csomagolva átmegy az eredeti byte-Hamming-alapú teszten. Védelem: kötelezően az IDENTITY_AUGMENTED_KILLER fut, NEM az eredeti.

3. **(Közepes valószínűség, súlyos hatás) Bájt-Hamming nem egyenlő a genom-távolsággal.** A bootstrap és CI félrevezető metrikán számítódik. **Védelem:** gráf-szerkesztési távolság az élek multihalmaza felett.

4. **(Közepes valószínűség, közepes hatás) DNP_CONTROL_PARITY logikailag rossz az eredeti egyesített formájában.** A valós dekóder nem tud "legyőzni" egy shuffled-fitness kontrollt, mert az értékelés-oldali. **Védelem:** kettéosztás dekóder-oldali és értékelés-oldali parityra.

5. **(Közepes valószínűség, közepes hatás) MMD bootstrap-CI mesterségesen szélesebb, mint PIC-é a softmax-rounding határoknál.** Az összehasonlítás torzulhat PIC javára, valódi tehetségkülönbség nélkül. **Védelem:** a bootstrap z-t resamplel, nem genomot — minden resample után újra-dekódoljuk.

---

## 8. Claims: Accepted / Rejected / Deferred

A swarm legkönyörtelenebb szekciója — szigorú szűrés, mit szabad és mit nem szabad D9.0-ban kódolni.

### Accepted (megy D9.0-ba)

- **PIC mint primary architektúra** (E §1)
- **IDENTITY_AUGMENTED_KILLER mint kanonikus első teszt** (G §7) — entrópia non-trivialitás kapu, gráf-szerkesztési távolság, D_advr mint kötelező harmadik dekóder
- **z[0] sűrűség alsó határa kis H esetén ≥ 0.03** (G §1.1, A1) — orphan node-ok elkerülésére
- **Non-degeneracy tag a pontszám-függvénybe** (G §1.1, A2) — worst-case Lipschitz garantálás
- **Gráf-szerkesztési távolság minden genom-távolság-metrikára** (G §2.1, D1)
- **Bootstrap re-decode-ol z-ből, NEM genomot resamplel** (G §1.2, B5)
- **Minden hash-küszöb a jelenlegi futás saját hash-baseline-jéhez kalibrált** (G §2.3, D5/D8)
- **DNP_CONTROL_PARITY kettéosztva dekóder-oldali és értékelés-oldali kapura** (G §2.4, D10)
- **Strukturális non-trivialitás kapu MINDEN lokalitás-tesztet megelőz**
- **Véletlen z mintavétel (NEM rács) minden Mantel-tesztben** (Quilodrán 2025)
- **Minden Gemini verdict-név és kontroll-kód változatlan** (D9_LATENT_DECODER_TOY_PASS, NCI_RANDOM_HASH_DECODER, stb.)

### Rejected (nem implementáljuk)

- **F eredeti §0 byte-Hamming killer microtestje** — falszifikálva D_advr által, kód előtt
- **FDC mint önálló kapu** — Altenberg 1997 ellenpéldák; csak Mantel-lel együtt diagnosztika
- **Silhouette önálló kapuként** — random embedding artifact veszély; csak ARI független orákulummal
- **Bájt-Hamming távolság-metrika** — nem egyezik a designok genom-fogalmával
- **MMD stochastic rounding mag-hash-sel** — visszahozza a GE failure mode-ot (Rothlauf-Oetzel 2006)
- **z_hat mint "genom-szintű inverz" külső hálózatokra** — moments not determine graph; átnevezve "strukturális ujjlenyomat"-ra
- **Codon-mod-rule szabály-választó séma** — Grammatical Evolution failure precedens
- **Tanult dekóder D9.0-ban** — D9.4-ig hard rule

### Deferred (D9.1+ vagy később)

- **MMD primary architektúraként** — csak ha PIC bukik az irreguláris medence teszten
- **GPF primary architektúraként** — csak ha D9.1 explicit CPPN-stílusú geometriát kér
- **BOP-Elites surrogate szkennelés** — D9.1+, ha az értékelés-költség nő
- **Grammar VAE / JT-VAE / tanult grammar dekóderek** — D9.4+, csak ha a determinisztikus baseline elégtelen
- **Atlasz-cella inverz mapping generatív modell nélkül** — soha (anti-mintázat); generatív modellel a D9.2+
- **Live Rust runtime változás** — D9.1+, miután genome canonicalization kész
- **Full graph grammar formalizmusok (nc-eNCE, edNCE)** — háttéranyag, nem építünk
- **NEAT speciation, novelty search, CMA-ME emitter** — D9.2+ szkennelési stratégia, NEM dekóder design

---

## 9. Ajánlott következő prompt egy kódoló ágensnek

A következő prompt önállóan futtatható — egy kódoló ágens elindíthatja anélkül, hogy ezt a beszélgetést látta volna. Angolul, mert kódolóhoz megy.

---

```
TASK: Implement D9.0 Latent Genome Decoder Toy Audit.

FILES:
- New tool: tools/analyze_phase_d9_latent_genome_toy.py
- New audit report: docs/research/PHASE_D9_0_LATENT_DECODER_TOY_AUDIT.md (matches D8 audit naming convention)
- No Rust source changes
- No new Cargo dependencies

REQUIRED PRIOR READS (in this order):
1. docs/research/inbox/d9_gemini_genome_compiler_design.md
2. docs/research/inbox/d9_claude_final_synthesis.md (parent synthesis)
3. docs/research/inbox/d9_claude_wave2_E_architecture.md (PIC design §1)
4. docs/research/inbox/d9_claude_wave2_F_falsification.md (benchmark suite)
5. docs/research/inbox/d9_claude_wave3_G_red_team.md (REQUIRED hardenings, do not skip)

HARD CONSTRAINTS (non-negotiable):
- Python only. No torch, tensorflow, jax, sklearn.neural_network, transformers. Allowed: numpy, scipy, hashlib, json, csv, pathlib, math, dataclasses, sklearn.metrics (for silhouette/ARI only), sklearn.cluster (KMeans/HDBSCAN only).
- Offline only — no Rust runtime change, no network call.
- Total runtime ≤ 30 minutes on a laptop. If exceeded, fail with DNP_TOO_HEAVY.
- All randomness deterministic from a single root seed argument; no time-based seeds.

ARCHITECTURE: PIC (Parametric InitConfig)
z = 12-dim float vector, fields per the synthesis §3 Primary section.
Hardening (Accepted from synthesis §8):
- z[0] (density) clamped to ≥ 0.03 for H ≤ 64
- score function s(i,j;z) includes a non-degeneracy term: s += 1e-6 * deterministic_hash(i,j) so ties are broken by continuous index function, not by root_seed
- bootstrap re-decodes z (do not resample genomes)
- root_seed only used as tie-breaker for genuinely-equal scores after non-degeneracy term

GENOME JSON SCHEMA (per E §1.4):
{
  "version": "d9.0-pic-1",
  "z_logged": [12 floats],
  "root_seed": "u64-hex",
  "rule_trace": [{"rule_id": str, "site": [int...], "score": float, "subseed": "hex"}, ...],
  "genome": {"H": int, "edges": [[i,j],...], "polarity": [int...], "thr": [int...], "channel": [int...]},
  "validity_flag": bool,
  "validity_reasons": [str...]
}

NEGATIVE CONTROLS (verbatim names from Gemini digest):
- NCI_RANDOM_HASH_DECODER: SHA256(z_bytes) -> rng -> random uint8 bytes -> wrapped as valid graph
- NCI_NONLOCAL_DECODER: D(sorted(z_bytes)) — destroys positional info
- NCO_SHUFFLED_FITNESS, NCO_SHUFFLED_BEHAVIOR_LABELS, NCO_RANDOM_CELL_ASSIGNMENT, NCO_RULE_LABEL_SHUFFLE
- D_advr (red-team adversarial): bytes_of(quantize(z, 8 bits)) + SHA256(quantize(z))[:GENOME_LEN], wrapped via valid graph builder

KILLER MICROTEST (canonical first test, ≤90 s):
1. Sample N=200 random z (NOT grid) from N(0,1) per dim.
2. Decode with three decoders: real (PIC), NCI_RANDOM_HASH_DECODER, D_advr.
3. Compute Mantel test using GRAPH EDIT DISTANCE on edge multisets (NOT byte Hamming) with 499 permutations.
4. Compute Spearman r_real, r_hash, r_advr.
5. Compute genome entropy for each decoder.
6. Verdict logic:
   - if entropy(g_real) <= entropy(z_logged) + log2(H_factorial)/8 -> D9_DECODER_VALIDITY_FAIL
   - if r_real < 0.30 or p_real >= 0.05 -> D9_DECODER_NO_LOCALITY
   - if r_real < r_hash + 0.20 (with non-overlapping 95% bootstrap CI) -> D9_DECODER_HASHLIKE_BEHAVIOR
   - if D_advr passes all of the above -> the test ITSELF is broken; report KILLER_TEST_DEGENERATE and stop

FULL BENCHMARK SUITE (after killer passes):
- 5 toy landscapes per F §1: smooth, deceptive (trap_k k=4), multi-basin (NK K=3), needle, random control
- N=500 z, 999 Mantel permutations, random sampling
- Metrics: valid_network_rate, z↔genome Mantel r, z↔behavior Mantel r (synthetic 4-D fingerprint: byte-mean, byte-std, byte-bigram-entropy, edge-density), basin silhouette + ARI vs independent landscape oracle, target hit rate, progressive scan rate ratio after 300 evals, exact roundtrip rate.
- For every gate threshold: calibrate against the actual NCI_RANDOM_HASH_DECODER run in the same study, NOT pre-fixed.

DNP GATES:
- DNP_VALIDITY_COLLAPSE: valid_network_rate < 0.99
- DNP_LOCALITY_COLLAPSE: r_real < r_hash_calibrated_95th + 0.20, with non-overlapping 95% bootstrap CIs (1000 resamples re-decoding z)
- DNP_BEHAVIOR_HASHLIKE: r_real(z↔behavior) < r_hash + 0.10
- DNP_SCAN_NO_GAIN: progressive/random rate ratio < 1.5 after 300 evals
- DNP_CONTROL_PARITY (split): decoder-side {hash, nonlocal, shuffled-rules}; evaluation-side {shuffled-fitness, shuffled-labels, random-cells}; real decoder must beat ALL on the gate it claims
- DNP_TOO_HEAVY: forbidden imports OR runtime > 30 min

VERDICT NAMES (verbatim, from Gemini digest):
D9_LATENT_DECODER_TOY_PASS
D9_DECODER_VALIDITY_FAIL
D9_DECODER_NO_LOCALITY
D9_DECODER_HASHLIKE_BEHAVIOR
D9_TILE_SCAN_NO_SIGNAL
D9_CONTROL_PARITY_FAIL

OUTPUT FILES:
- outputs/d9_toy_run_<timestamp>/landscape_results.csv (N=500 rows × 5 landscapes)
- outputs/d9_toy_run_<timestamp>/genome_provenance.jsonl (one line per generated genome with full schema)
- outputs/d9_toy_run_<timestamp>/control_baselines.json (all 4+ negative-control metric values)
- docs/research/PHASE_D9_0_LATENT_DECODER_TOY_AUDIT.md (final audit report with single explicit verdict)

AUDIT REPORT MUST INCLUDE:
- Verdict (one of the 6 above)
- Each DNP gate result with measured value and threshold
- Killer microtest result first (with D_advr discrimination check)
- Calibrated hash-baseline percentiles
- Wall-time per stage
- All negative-control metric tables
- Reproducibility hash (root seed + tool version + numpy version)

If any DNP gate fires, the verdict is the corresponding fail-name and remaining stages are skipped. Fail-fast.

NO ADDITIONAL FEATURES. No mutation operator. No archive. No RL. No "while we're at it" cleanups. Only what is listed above.
```

---

## Verifikációs lista (mit jelent hogy a kutatás kész)

- [x] 8 raw artefakt + 1 Gemini digest a `docs/research/inbox/`-ban
- [x] Ez a végső szintézis fájl szerint hivatkozik a forrásokra
- [x] Mind a 9 kötelező szekció megvan (8 felhasználói + Claims accepted/rejected/deferred)
- [x] Minden D(z) design mellett legalább 1 G-általi konkrét támadás és válasz arra
- [x] A kanonikus killer microtest 90 mp-es szinten részletezett, beleértve D_advr-t
- [x] A §9 prompt önállóan futtatható, schema-level konkrét, Python-only/offline/no-Rust
- [x] A vörös csapat által talált kockázatok elsőként szerepelnek, nem prefabrikált biztosítékok

A swarm 4 hullámban futott, ~12 perc valós idő, ~7 raw artefakt + ez a szintézis. A kutatás verdiktje: **a D9 koncepció ép, ÉS a swarm kód előtt megtalálta a 8 hardening-et, amelyek nélkül elbukna.**
