# VRAXION — Claude Code resume primer

> Másold be ezt a fájlt egy friss Claude Code session első promptjába (vagy csak: `cat docs/wiki/RESUME_PRIMER.md` → paste). Célja: nulláról is tudjam, miben dolgozunk, hol áll a repo, mik a frissen megoldott dolgok, és hogy a user hogyan szereti a munkamenetet.
>
> Naprakész: **2026-04-28**, INSTNCT `v5.0.0-beta.6`. Ha ez a primer >1 hete készült, nézd meg a `git log --oneline -30`-at mielőtt terveket írnál.

---

## Ki a user, mit csinál

- Tulajdonos: kenessy.dani (Windows fő gép, Steam Deck úton).
- Architektúra-vonal: **INSTNCT** — byte-level lexikai→neurális pipeline Rust-ban + Python deploy SDK-val.
- Fejfájdalom: képletek és matek notáció NEM — gép/tárcsa/víz/gyár metafórákkal gondolkodunk, rövid chunkolt magyar + angol technikai terms.
- Munkamód: "nyomjad", persze, auto mode; ritkán interaktív plan mód; gyors iteráció.
- Preferenciák röviden (részletek a memory-ban `MEMORY.md`):
  - mindent push-ol minden commit után
  - `--no-verify` tilos
  - background task után nem polling — auto-értesítést várunk
  - checkpoint kötelező minden hosszú kísérletben
  - "growing/persistent" = checkpointet **betölti startkor**, nem csak menti

## Repo gyors-anatómia

```
VRAXION/
├── Rust/                  # deploy SDK (pure std + serde), Block A + B
├── Python/                # numpy deploy SDK, Block A + B
├── instnct-core/          # Rust grower mainline (neuron_grower.rs)
├── tools/                 # diag + runner + sync scripts
├── output/                # champion artifactok (IN GIT)
├── docs/
│   ├── index.html         # Pages home
│   ├── blocks/*.html      # 5 block + 1 variant (native-7bit)
│   ├── wiki/*.md          # GitHub Wiki mirror forrás
│   └── playground/        # interactive visualizer-ek
└── VALIDATED_FINDINGS.md  # nyilvános evidence register
```

Champions (NE töröld, CI is őrzi):

| fájl | méret | mi |
|---|---|---|
| `output/merger_single_w_huffman_pack/packed_model.bin` | **3440 B exact** | L1 merger champion (Cluster 13) |
| `output/merger_single_w_huffman_pack/summary.json` | — | champion metadata |
| `output/byte_unit_champion_binary_c19_h16/byte_embedder_lut_int8.json` | — | L0 byte unit LUT (Cluster 9) |
| `tools/byte_embedder_lut.h` | — | C header deploy |
| `tools/byte_embedder_lut_int8.json` | — | tools/ mirror |

## Mit csináltunk kb. az elmúlt hétben

1. **Cluster 18 — Native 7-bit identity merger** (GPT + lokális kísérlet)
   - H=120, 3,421 B bake-probe (kb. 19 B-tal alacsonyabb mint a 3440-es Huffman champion)
   - Identity linear autoencoder, 100% lossless all 65,536 byte pair-en
   - C19 aktiváció nem post-hoc kvantizálható — ezt a próba megerősítette
   - Dedikált site: `docs/blocks/b-merger-native-7bit.html` (amber téma, 8 slide)

2. **Public-release cleanup (Fázis 5–7)**
   - ~60 Python diag + ~233 Rust example archiválva `archives/*` tagekbe
   - 3 archive tag a remote-on:
     - `archives/python-research-20260420`
     - `archives/tools-legacy-diag-20260420`
     - `archives/public-readiness-residue-20260420`
   - tag = immutable, mindig visszaállítható `git checkout <tag>`-al
   - Version bump: `v5.0.0-beta.1` → `v5.0.0-beta.2` → … → `v5.0.0-beta.6` (README, VERSION.json, Cargo.toml, CITATION, BETA)
   - `instnct-core/results/` untrack-elve (36 run-artifact state.json)
   - `.gitignore` frissítve (OS cruft, IDE, ML weights, Python caches)

3. **Website polish + bugfixek (2026-04-20)**
   - Homepage "Explore the blocks" CTA most a grid-re ugrik, nem direkt Block A-ra
   - Block A page kapott block-nav-ot (előtte zsákutca volt)
   - Block B variant chip (amber) a homepage Block B kártyáján
   - B merger page linkel a Native 7-bit variant page-re
   - `check_public_surface.py` most champion artifact méret-lock-ot is ellenőriz (3440 B exact)

4. **Test suite bővítés (parity symmetry)**
   - Python Block A: 9→11 test (determinism + empty-bytes)
   - Python Block B: 9→11 test (middle-pair spot-check + forward determinism)
   - Rust Block A: 5→7 test (ugyanaz a két új pattern)

5. **Wiki sync**
   - `docs/wiki/COMPRESSION_LOOP.md` bekerült a sync pipelinebe és fel van töltve a GitHub Wiki-re
   - Timeline-Archive új bejegyzésekkel: mind a 4 cleanup fázis + a 3 archive tag hivatkozás

## Pipeline jelenlegi állapota

```
A Byte Unit (L0)  →  B Merger (L1)  →  C Tokenizer (L2?)  →  D Embedder  →  E Brain
   FROZEN              FROZEN            ACTIVE              SCAFFOLD       SCAFFOLD
   256/256             3440 B            word V1 52%         várj            várj
   lossless            100% pairs        (FineWeb-EDU)
```

- **L0 (Block A)**: binary weights + C19 + H=16, teljes 256/256 byte lossless round-trip. Pure LUT lookup deploy (O(1)).
- **L1 (Block B)**: single-W tied mirror H=81, Huffman-packed, 3440 B, 100% lossless 65,536 byte pair-en. Egy variant frontier: **Native 7-bit** (H=120, 3,421 B potenciál).
- **L2 (Tokenizer)**: word tokenizer V1 52% raw / 34.6% byte fallback FineWeb-EDU-n. Commit `e82a029`.
- **L2 frontier**: `tools/diag_byte_l2_merger.py` + `tools/diag_byte_l2_phase0_geometry_probe.py` — open exploration, nincs bakelt eredmény.

## Parancsok (copy-paste, Linux)

```bash
# fresh clone → working state
git clone https://github.com/VRAXION/VRAXION.git
cd VRAXION

# Python SDK teszt
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pytest
pytest Python/                    # 22 test, mind zöld

# Rust SDK teszt
cd Rust && cargo test --release   # Block A + B tests
cd ..

# Public surface gate (champion artifactokat is ellenőriz)
python tools/check_public_surface.py

# Grower regression (B0 freeze gate, hosszabb)
python tools/run_grower_regression.py
```

## Nyitott frontierek (ha szabad kezed van)

1. **L2 merger baseline** — `diag_byte_l2_merger.py` + phase0 geometry probe; hol van a ceiling?
2. **Native 7-bit QAT finalize** — bake → QAT Adam → QAT LBFGS recipe végigfuttatása a H=120 variantra, Huffman-pack, konkrét `.bin` artifact
3. **Word tokenizer V2** — jelenleg 52% raw; subword fallback stratégia javítás
4. **Embedder scaffold** (Block D) — még csak placeholder; input: L1 packed bytes, output: semantic latent
5. **Nano Brain** (Block E) — scaffold, mainline Rust grower alapra kell illeszteni

## Ha elakadsz

- **VALIDATED_FINDINGS.md** — nyilvános evidence lista, soha ne másolj be belőle nem-validált állítást
- **docs/wiki/Timeline-Archive.md** — teljes kutatási idővonal, minden cluster commit hash-el
- **docs/wiki/COMPRESSION_LOOP.md** — Cluster 18 részletes writeup
- **BETA.md** — B0 gate kontraktus
- **MEMORY.md** — user preferenciák + architektúra findings (auto-load)
- `git log --oneline -50` — mi volt mostanában
- `git tag --list "archives/*"` — miből kutatható vissza történelem

## Steam Deck specifikus

- Repo-ban nincs Windows-only path (ellenőrizve)
- `env!("CARGO_MANIFEST_DIR")` joinokkal dolgozik Rust oldalon, Linuxon is működik
- Sync script (`tools/sync_wiki_from_repo.py`) Linuxon is futtatható
- VRAXION.wiki remote külön repo: `https://github.com/VRAXION/VRAXION.wiki.git` — klónozd a VRAXION-nal szomszédos könyvtárba ha wiki-t akarsz pusholni

---

**Amikor ezt primerként kapod**: először `git status` + `git log --oneline -10` + `python tools/check_public_surface.py` — ha mind zöld, készen állunk. Ha nem zöld, jelents vissza és ne kezdj javítani vakon.
