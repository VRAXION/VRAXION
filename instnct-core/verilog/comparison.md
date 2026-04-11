# 8-bit ALU Gate Count Comparison: C19 LutGate vs Standard Verilog

**Tool:** Yosys 0.45+139 (OSS CAD Suite 2024-09-22)  
**Target:** Generic CMOS gates → LUT4 mapping (ABC lutpack)  
**Date:** 2026-04-11

---

## Summary Table

| Metric                    | C19 LutGate (`alu8_c19.v`) | Standard (`alu8_standard.v`) | Delta (C19 vs Std) |
|---------------------------|:--------------------------:|:-----------------------------:|:-------------------:|
| **Logic gates (pre-ABC)** | **665**                    | **692**                       | -27 (-3.9%)         |
| **LUT4 cells (post-ABC)** | **258**                    | **267**                       | -9 (-3.4%)          |
| Total cells (post-ABC)    | 496                        | 267                           | +229 (+86%)         |
| Wires                     | 1401                       | 676                           | +725 (+107%)        |
| Wire bits                 | 2248                       | 729                           | +1519 (+208%)       |

> **"Total cells (post-ABC)"** includes `$scopeinfo` markers for the C19 version (238 scope
> annotations from the many named sub-modules). These are not logic — they are debug metadata.
> The **LUT4 count** is the correct apples-to-apples comparison.

---

## Pre-ABC Gate Breakdown

After `synth -flatten` and before LUT packing, Yosys maps to 2-input primitives:

| Gate type    | C19 LutGate | Standard |
|--------------|:-----------:|:--------:|
| `$_ANDNOT_`  |   46        |   58     |
| `$_AND_`     |   83        |   62     |
| `$_MUX_`     |   45        |   17     |
| `$_NAND_`    |   14        |   25     |
| `$_NOR_`     |  208        |  232     |
| `$_NOT_`     |   52        |   42     |
| `$_ORNOT_`   |   71        |  106     |
| `$_OR_`      |   14        |   17     |
| `$_XNOR_`    |  110        |  121     |
| `$_XOR_`     |   22        |   12     |
| **TOTAL**    | **665**     | **692**  |

---

## LUT4 Mapping (ABC lutpack -S 1)

| Design         | LUT4 cells | Notes                                  |
|----------------|:----------:|----------------------------------------|
| `alu8_c19`     |    258     | C19 primitives → LUT4 with packing     |
| `alu8_standard`|    267     | Standard operators → LUT4 with packing |
| **Difference** |    **-9**  | C19 uses **3.4% fewer LUT4s**          |

---

## Interpretation

### Why C19 is (marginally) smaller

The C19 LutGate representation encodes each gate as an explicit LUT truth table. This
structural approach makes certain optimisations transparent to Yosys:

1. **Explicit carry chain structure** — the ripple-carry adder with named `c19_full_adder`
   primitives gives ABC a clean signal graph. The optimizer can pack sum+carry into a single
   LUT4 easily.

2. **Explicit MUX primitives** — `c19_mux2` (3-input LUT) translates directly to a LUT4
   without the mux-inference overhead the standard `?:` operator sometimes carries.

3. **No arithmetic inference overhead** — Yosys's arithmetic inference (`+`, `-`, `*`) adds
   carry-propagation logic that the explicit ripple-carry C19 adder avoids.

### Why the wire count is much higher for C19

The C19 version has a deeply modular hierarchy (18 sub-modules). After flattening, each
internal connection becomes a named wire, inflating the wire count. These are the same
physical signals — just more named. This does **not** affect synthesis quality.

### Multiplication is the equaliser

Both designs implement 8×8 multiplication. The C19 `c19_mul8` uses Verilog's `+` operator
for partial-product accumulation (Yosys infers the adder tree either way). This is where
the standard version catches up — the synthesizer is equally efficient regardless of how
the partial products are expressed.

---

## Files

| File                       | Description                                    |
|----------------------------|------------------------------------------------|
| `alu8_c19.v`               | C19 LutGate ALU — explicit LUT primitives      |
| `alu8_standard.v`          | Standard Verilog ALU — operators               |
| `c19_synth_report.txt`     | Full Yosys log for C19 (synth + LUT4 ABC)      |
| `standard_synth_report.txt`| Full Yosys log for standard (synth + LUT4 ABC) |

---

## Conclusion

The C19 LutGate style produces a **marginally smaller circuit** (3–4% fewer gates/LUTs)
compared to a standard Verilog description of the same 8-bit ALU. The difference is not
dramatic because modern synthesis tools (Yosys + ABC) are excellent at inferring efficient
implementations from behavioral descriptions.

The real advantage of the C19 representation is **design intent clarity**: each primitive
maps one-to-one to a physical LUT, making gate-level reasoning and formal verification
more tractable. The wire count overhead from explicit modularity is purely cosmetic and
disappears after place-and-route.
