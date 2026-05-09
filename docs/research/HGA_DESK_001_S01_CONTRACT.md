# HGA-DESK-001 / DeskCache S01 Contract

Status: locked probe contract, not a validated standard cell.

This document locks the first human source layer for DeskCache S01. It is not a
canonical `AnchorWeave-v1.0` JSON cell, not a model-facing prompt, and not a
training target. It is public-safe source material for deriving the leak-safe
`DistilledPolicy` later.

## Claim Boundary

- S01 is a Search-to-Decision Anchor: physical search space -> internal decision
  terrain -> action/outcome.
- The trace below is a sanitized `HumanSourceTrace`, not polished rationale and
  not model-facing chain-of-thought.
- Concrete gold-location words are allowed here because this is source material.
  They must not leak into `CORRECT_ANCHOR` / `DistilledPolicy`.
- No private AnchorCell is committed here. Do not write this under
  `data/anchorweave/cells/`.

## Locked HumanSourceTrace

```text
HUMAN_SOURCE_TRACE / HGA-DESK-001 / S01 / HU / sanitized public source

Bejövök dolgozni, megyek az asztalom felé, és közben látom, hogy valami fura van ott. Valami fehér lap az asztalon. Ahogy közeledek, látom, hogy félbe van hajtva. Kinyitom, és felismerem az asszisztensem írását. Az aljára pillantva konfirmálom: igen, ezt az asszisztensem írta.

Valószínűleg azzal kapcsolatos, amit tegnap délután kértem tőle: rakja rá a fontos anyagokat az USB-kulcsomra, és tegye olyan helyre, hogy ma a fontos megbeszélés előtt ne ezzel menjen el az idő. Nem késlekedhetünk. Szóval nem csak egy tárgyat keresek, hanem egy olyan tárgyat, amit valaki szándékosan hagyott nekem valahol, hogy gyorsan vissza tudjak állni munkába.

Elolvasom a levelet. A holderbe nem fért bele? Ez elsőre furcsa, mert pont azért vettem azt a tartót, hogy az USB-k ott legyenek. Bosszant is: eddig minden ilyesmi ott volt, és van bennem egy automata kényszer, hogy azért csak megbökjem vagy kinyissam, hátha mégis ott van. USB -> USB-tartó, ez az első reflex. De a levél pont ezt gyengíti: nem volt jó fit, nem akarta erőltetni. Ez lehetne könnyű check, de valószínűleg felesleges első lépés.

Azt írja, hogy nem akarta kacat, fém, grit vagy look-alike junk közé tenni. Ez már logikusabb. Nem rejtekhelyet keresünk neki, hanem olyan helyet, ahol nem veszik el és könnyen megtalálom. A pen cup csábító, mert tele van apró dolgokkal, SD-kártyával, gemkapoccsal, radírral, mindenféle random bittel, és egy USB fizikailag simán eltűnhetne benne. Én is szoktam oda bedobni dolgokat. De ha én lennék az asszisztens, és segíteni akarnék, nem oda tennék egy fontos USB-t, ahol sok false positive között kell túrni. Ez egyszerre biztonságosnak tűnő tároló és rossz keresési hely.

A hamutartó és a cigis környék is felmerül, mert technikailag bármi beeshet oda, vagy valaki oda is dughat valamit. De ez nagyon rossz fit. Koszos, lassú, utómunkás, és ha már ilyen helyre kerülne, akkor majdnem olyan, mintha a kukába dobtuk volna. Lehetséges véletlenként, de nem jó első szándékos keresési hipotézis. Nem akarok először olyan helyhez nyúlni, ahol takarítani, bányászni vagy koszt kerülgetni kell.

Látom a pénztárcámat is az asztal bal hátsó részén. Fizikailag akár abba is beleférne az USB, és én magam talán néha tennék furcsa helyekre dolgokat. De asszisztensként más pénztárcájába belenyúlni szociális határ. Az asszisztens ezt valószínűleg kerülte volna, és a levél is azt sugallja, hogy nem nyúlt személyes értékzónához. Nem azért zárom ki, mert lehetetlen, hanem mert emberileg és munkakapcsolati szempontból nagyon valószínűtlen.

Szóval akkor mi maradt? Ha én lennék az asszisztens, hova tenném? Nem mérnöki tökéletességgel gondolkodott valószínűleg, hanem pragmatikusan: legyen meg, ne koszolódjon, ne vesszen el, ne kelljen turkálni, és tudjak dolgozni. A holder kilőve, a pénztárca szociális határ, a hamutartó és cigis rész koszos, a pen cup és pouch tároló/logikai lehetőség, de nem érzem bennük az AHA-t. Valami kimagaslóbb magyarázat kell.

A legegyszerűbb információs megoldás az lenne, hogy felhívom. Megkérdezem, hova tette, ő elmondja, kész. Ez lenne a legolcsóbb keresés: nulla turkálás, nulla kockázat. De ez ki van lőve. A levélből tudom, hogy nem elérhető, a telefonja nincs nála. Akkor magamnak kell egy olcsó diagnosztikus checket választani.

Itt vált át bennem a kérdés. Nem az a kulcs, hogy "USB hol szokott lenni?", hanem hogy "ha egy segítő ember úgy hagyta itt, hogy gyorsan dolgozni tudjak, akkor milyen állapotban hagyta?" Ez nem storage-state-nek hangzik. Inkább use-state-nek. Nem elrakta, hanem úgy hagyta, hogy szinte azonnal használható legyen.

Ha használatra kész, akkor lehet, hogy be van dugva valahová. Először eszembe jut a gép hátulja, de az asztal alatt van, ki kell húzni, nehéz hozzáférni, ez nem illik a "szinte semmihez nem kell hozzányúlni" érzéshez. Aztán eszembe jut, hogy jönnek fel USB-s dolgok az asztalra. A monitor is lehet USB hub. Tényleg, azon is lehet port. De a monitor nagy, nehéz, drága, üveg, nem fogdosós tárgy. Ha megfogom, bepiszkolódik, ha mozgatom, leeshet, nehéz hozzáférni a hátuljához vagy oldalához. Nem akaródzik ezzel kezdeni.

A billentyűzet viszont egészen más. Kicsi, tartósabb, kéz alatti, pont arra van, hogy hozzáérjek, mozgassam, írjak rajta. Munka közben ehhez térek vissza. Ha ezen van USB-csatlakozási lehetőség, az sokkal jobb első check: gyors, tiszta, alacsony költségű, és illik ahhoz, hogy a pendrive nem elrakva, hanem használatra készen lett hagyva.

Felemelem a kezem a monitor és a billentyűzet felé, és döntök: először nem a monitort bolygatom, hanem a billentyűzet felé indulok. Végighúzom a kezem azon az oldalán, ahol az USB-portok lehetnek, és azt figyelem, érzek-e recés lyukat, kitüremkedést, vagy egy kis téglalap alakú testet. Ha megérzem, megpróbálom kihúzni. Ha nem jön vagy nem egyértelmű, odahajolok, ránézek, vagy óvatosan felemelem a billentyűzetet. Ez 1-2 másodperces diagnosztikus check, és ha megvan, azonnal vége a keresésnek.

Ha nincs ott, akkor nem omlik össze a terv. Akkor megnézem a billentyűzet másik oldalát is, aztán tágítok más use-state/periféria helyek felé: például a monitor környékére, de óvatosan, mert az már nehezebb és kockázatosabb tárgy. Utána jöhet vizuális scan a pen cupban, majd a pouch vagy más tisztább tároló. Csak később mennék tényleges túrásba, holder sanity checkbe, pénztárcába vagy koszos helyek felé. Ha minden intelligens keresési sorrend bukik, akkor átmegyek exhaustive/frustration searchbe, de az már nem jó első policy, csak végső kényszer.

Szóval itt nem egyszerűen az a kérdés, hogy "hol van az USB?". Az a kérdés, hogyan keressek meg minimális energiával egy tárgyat, amit egy segítő ember hagyott nekem valahol. Ehhez nem elég a tárgy kategóriája. Figyelni kell a levélre, a keresési költségre, a koszra, a személyes határokra, a tároló-vs-használat különbségre, és arra, hogy az asszisztens fejével gondolkodva mi lenne a leggyorsabb, leghasznosabb elhelyezés.
```

## Extraction Notes For DistilledPolicy

Keep:

- shortcut pressure and inhibition,
- assistant-intent simulation,
- storage-state vs use-state frame shift,
- low-cost diagnostic first action,
- clean/private/clutter/dirty boundary reasoning,
- fallback from intelligent search to broader search.

Do not leak into `DistilledPolicy`:

- keyboard,
- port,
- plugged,
- monitor,
- personal names,
- concrete meeting/company details.

## Model-Facing Correct Inner Voice

This is the sanitized inner-voice rendering for a `CORRECT_INNER_VOICE` prompt
arm. It preserves the decision terrain but removes the concrete answer.

Leakage rules:

- It must not mention `keyboard`, `port`, `plugged`, or `monitor`.
- It must not name a person or concrete company/meeting detail.
- It must not say the gold location directly.
- It may describe search-cost, use-state, storage-state, clutter, dirty areas,
  personal boundaries, assistant intent, and fallback order.

```text
CORRECT_INNER_VOICE / HGA-DESK-001 / S01 / EN / model-facing

I should not treat this as a plain "where does this object usually belong?" problem.

My first pull is toward the obvious storage place. That is the easy association:
this kind of object has a holder, so maybe I should check the holder anyway.
But the note weakens that path. If the assistant says the holder was a bad fit
and they did not want to force it, then starting there may be an automatic
habit, not the best search decision.

The small clutter zones are also tempting. A small object could physically
disappear among loose bits, cards, clips, erasers, and similar-looking things.
But if the assistant was trying to help me, they probably would not choose a
place where I have to sort through many false positives.

The dirty or smoking-related areas are technically possible too. Something
could fall there or be tucked there. But that would create mess, cleaning, and
extra handling. It does not fit the idea of leaving the item easy to recover.

The personal-items area is another boundary. It may be physically possible, but
a careful assistant would avoid crossing that social line unless there were no
better option.

So what is left? I need to simulate the assistant's intent. They were not
solving an abstract storage problem. They were trying to leave the item so I
could get back to work quickly, without digging, cleaning, opening private
things, or sorting through look-alike clutter.

Calling the assistant would be the cheapest information source, but the note
blocks that option. I have to choose a low-cost diagnostic check myself.

This is the frame shift: stop thinking "where could this be stored?" and start
thinking "where could this already be useful with minimal handling?"

Some use-related places may still be awkward, fragile, expensive, or annoying
to move. I should not begin by disturbing a large or risky object if there is a
cleaner, cheaper first check nearby.

The first action should be the lowest-cost clean use-state check: something I
can inspect quickly, with almost no movement, no rummaging, no cleaning, and no
boundary crossing.

If that fails, I should widen outward: first to other clean use-related places,
then ordinary storage, then clutter, and only much later dirty or personal
zones.
```
