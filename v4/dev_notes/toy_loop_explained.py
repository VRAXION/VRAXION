"""
INSTNCT belso loop — "TOY" verzio, magyarul

Kepzeld el:
- Van egy KORAGOLYOS POLC (ring) — 256 rekesz koralakban
- Van 2 DOLGOZO (expert) — mindegyiknek van egy UJJA ami a polc egy rekeszere mutat
- Van egy BEJOVO LEVEL (input) — minden timestep-ben erkezik egy
- Van egy EMLEKEZET (hidden) — minden dolgozo fejeben egy kis jegyzetfuzet

Minden orautes (timestep):
  Minden dolgozo (expert):
    1. OLVAS a polcrol ahol az ujja mutat
    2. GONDOLKODIK (input + olvasott + emlekezet → uj emlekezet)
    3. IR a polcra amit kigondolt
    4. MOZGATJA az ujjat (ugrik vagy lep egyet)
  Vegul: a ket dolgozo gondolatait atlagoljuk → ez a valasz
"""

# ============================================================
#  JELENLEGI KOD (soros — egymas utan)
# ============================================================

def egy_timestep_SOROS(ring, pointerek, emlekezetelk, input_vec, expertek):
    """
    Amit MOST csinal a kod — Expert₀ ELOSZOR, Expert₁ MASODSZOR.
    Expert₁ LATJA amit Expert₀ irt a polcra.
    """

    # --- Expert₀ dolgozik ---

    olvasott_0 = OLVASD_A_POLCOT(ring, ahonnan=pointerek[0])
        # Expert₀ megnezi a polcot ahol az ujja mutat
        # → kap egy "slot_dim" meretu vektort (pl 32 szam)

    emlekezetelk[0] = GONDOLKODJ(
        bejovo   = input_vec,          # a mai level
        olvasott = olvasott_0,         # amit a polcrol olvasott
        regi_mem = emlekezetelk[0],    # amit tegnap gondolt
    )
        # osszeadja mind a harmat, atnyomja tanh-on
        # → uj emlekezet (hidden_dim meretu, pl 1024 szam)

    ring = IRJ_A_POLCRA(ring, amit=emlekezetelk[0], ahova=pointerek[0])
        # Expert₀ visszairja a gondolatat a polc rekeszebe
        # !!!! A RING MEGVALTOZOTT !!!!

    pointerek[0] = MOZGASD_AZ_UJJAT(pointerek[0])
        # ugrik (phi-jump) vagy lep egyet (+1 walk)

    # --- Expert₁ dolgozik ---

    olvasott_1 = OLVASD_A_POLCOT(ring, ahonnan=pointerek[1])
        # Expert₁ MOST olvassa a polcot
        # → Ha Expert₀ epp IDEirt, Expert₁ LATJA Expert₀ irasat!
        # → Ha Expert₀ masik rekeszbe irt, Expert₁ NEM latja
        #   (ez tortenik az ido 73%-aban)

    emlekezetelk[1] = GONDOLKODJ(
        bejovo   = input_vec,
        olvasott = olvasott_1,
        regi_mem = emlekezetelk[1],
    )

    ring = IRJ_A_POLCRA(ring, amit=emlekezetelk[1], ahova=pointerek[1])
    pointerek[1] = MOZGASD_AZ_UJJAT(pointerek[1])

    # --- Valasz ---
    valasz = ATLAGOLD(emlekezetelk[0], emlekezetelk[1])
        # a ket dolgozo gondolatanak atlaga → ez megy ki

    return ring, pointerek, emlekezetelk, valasz


# ============================================================
#  BATCHED KOD (parhuzamos — egyszerre)
# ============================================================

def egy_timestep_BATCHED(ring, pointerek, emlekezetelk, input_vec, expertek):
    """
    Amit csinalni AKARUNK — mindket expert EGYSZERRE dolgozik.
    Expert₁ NEM latja amit Expert₀ irt, mert meg nem irta.
    """

    # Elmentjuk a polc JELENLEGI allapotat
    ring_EREDETI = ring
        # mindket dolgozo UGYANAZT a regi polcot olvassa

    # --- Mindketto EGYSZERRE olvas (a REGI polcbol) ---

    olvasott_0 = OLVASD_A_POLCOT(ring_EREDETI, ahonnan=pointerek[0])
    olvasott_1 = OLVASD_A_POLCOT(ring_EREDETI, ahonnan=pointerek[1])
        # ^^^ EZ A KULONBSEG! Expert₁ a REGI polcot olvassa,
        #     nem azt amibe Expert₀ epp most irt.
        #     Az ido 73%-aban ez UGYANAZ (mert mas rekeszt neznek).
        #     Az ido 27%-aban ez MAS (mert atfednek).

    # --- Mindketto EGYSZERRE gondolkodik ---

    emlekezetelk[0] = GONDOLKODJ(
        bejovo=input_vec, olvasott=olvasott_0, regi_mem=emlekezetelk[0])
    emlekezetelk[1] = GONDOLKODJ(
        bejovo=input_vec, olvasott=olvasott_1, regi_mem=emlekezetelk[1])
        # GPU-n ez EGY NAGY MATMUL ket kicsi helyett → 2× gyorsabb!

    # --- Mindketto EGYSZERRE ir ---

    ring = IRJ_A_POLCRA(ring, amit=emlekezetelk[0], ahova=pointerek[0])
    ring = IRJ_A_POLCRA(ring, amit=emlekezetelk[1], ahova=pointerek[1])
        # mindketto irasa rakerulm a polcra

    # --- Mindketto EGYSZERRE mozog ---

    pointerek[0] = MOZGASD_AZ_UJJAT(pointerek[0])
    pointerek[1] = MOZGASD_AZ_UJJAT(pointerek[1])

    # --- Valasz ---
    valasz = ATLAGOLD(emlekezetelk[0], emlekezetelk[1])

    return ring, pointerek, emlekezetelk, valasz


# ============================================================
#  MIERT GYORSABB A BATCHED?
# ============================================================
"""
SOROS:
  GPU kap feladatot: "szorozz meg egy (32, 32) matrixot"     → Expert₀ read_proj
  GPU kiszamolja...var...kesz
  GPU kap feladatot: "szorozz meg egy (32, 32) matrixot"     → Expert₁ read_proj
  GPU kiszamolja...var...kesz
  = 2 kulon "kernel launch" — a GPU minden alkalommal bemelgit, kiszamolja, lehul

BATCHED:
  GPU kap feladatot: "szoroz meg egy (64, 32) matrixot"      → mindketto egyszerre
  GPU kiszamolja...kesz
  = 1 kernel launch, dupla adat, UGYANANNYI ido

  A GPU olyan mint egy gyar: ha 10 dobozt kell festeni, jobb ha
  EGYSZERRE betolod mind a 10-et a festovonalon, nem egyessevel.
"""


# ============================================================
#  A KERDES
# ============================================================
"""
A 27% overlap-nel Expert₁ MAST lat a batched verzioban:
  - SOROS:   Expert₁ olvassa Expert₀ FRISS irasat → "kollegam mar megirta, latom"
  - BATCHED: Expert₁ a REGI polcot olvassa → "meg nem latom kollegam irasat"

Ez JOBB vagy ROSSZABB? Nem tudjuk elore. Lehetsegesek:
  - JOBB:  Expert₁ fuggetlenul gondolkodik → tobb diverzitas → jobb
  - ROSSZABB: Expert₁ nem latja Expert₀ jelzeset → info vesztes → rosszabb
  - UGYANOLYAN: a 27% tul kicsi ahhoz hogy szamitson

Csak benchmark dontheti el. De a batched 2× gyorsabb per timestep.
"""
