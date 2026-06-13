# VRAXION License v1.0 — Adversarial Review

Date: 2026-06-13
Scope: license model for broad free community use + 19% royalty on monetized third-party access.

## Baseline repo state checked

Current public repo state observed before this package:

- `LICENSE` is PolyForm Noncommercial 1.0.0.
- Current `legal/LEGAL.md` says commercial use requires a separate written license.
- Current `README.md` points users to `LICENSE` and `legal/LEGAL.md`.

This package changes the model from "noncommercial only unless separately licensed" to "free community and internal use, paid royalty for monetized third-party access."

## Core target

```text
Free:
- personal
- research
- education
- nonprofit
- internal use
- community forks
- free demos
- self-hosting

Paid:
- hosted SaaS
- paid API
- subscription
- paid managed deployment
- paid app feature
- white-label/resale
- ad-supported/sponsored monetized access
```

Royalty target:

```text
19% of Attributable Net Revenue
  = 1% Founder Allocation
  + 18% VRAXION Forever Prize Allocation

After founder death/designated trigger:
19% VRAXION Forever Prize Allocation
```

## Attack tests

### 1. "We have no profit"

Defense: royalty is based on Net Revenue, not profit.

Net Revenue excludes only refunds, chargebacks, transaction taxes, payment fees, limited platform fees, and user-facing discounts. It does not deduct cloud compute, salaries, R&D, marketing, internal fees, or overhead.

### 2. "We bundled it inside a bigger subscription"

Defense: Attributable Net Revenue uses the greatest of reasonable allocation, usage-share allocation, or 30% fallback if VRAXION-powered functionality is a material paid feature.

Risk: 30% fallback may be negotiated down by large buyers. Good for leverage; may be too aggressive for enterprise adoption.

### 3. "We route revenue through a reseller/affiliate"

Defense: affiliate aggregation and no-avoidance clauses ignore below-market transfer pricing and affiliate routing.

### 4. "We call the fee compute/support/platform/membership"

Defense: labels do not control. If third parties receive monetized access to VRAXION-powered functionality, it is Royalty Use.

### 5. "We give access for free but monetize ads/data/sponsorship"

Defense: ad-supported, sponsored, data-resale, lead-generation, affiliate, and non-cash consideration models count as Royalty Use.

### 6. "We fork and remove the VRAXION name"

Defense: fork is allowed, but license notices must remain. Trademark restriction blocks false official branding.

Risk: if someone clean-room rewrites ideas without copying protected Project Materials, license cannot stop that. Patent protection would be needed for stronger algorithmic moat.

### 7. "We use it internally at a company"

Decision: allowed for free unless third parties are charged for access to VRAXION-powered functionality.

Rationale: this maximizes adoption and keeps the royalty focused on actual commercialization of access.

Tradeoff: a company can get indirect internal economic benefit for free.

### 8. "We use VRAXION to train another model, then sell the other model"

Defense: derivative checkpoints, distillations, adaptations, conversions, and functionality materially enabled by Project Materials are included.

Risk: if they only learn general ideas and independently implement from scratch, copyright license reach is limited.

### 9. "We say only 1% of revenue is VRAXION-attributable"

Defense: if no separate price and the feature is material, fallback attribution applies.

Risk: "material feature" can be fact-intensive. Keep usage logs and public marketing evidence.

### 10. "Founder dies; who gets the 1%?"

Defense: Founder Redirect Event redirects the Founder Allocation to the Prize.

Legal hardening needed: Daniel should execute a will/trust/assignment/foundation document so heirs, estate, or payment processors cannot create ambiguity.

### 11. "AI controls the money"

Defense: the license and charter make AI an allocator/advisor unless law permits direct autonomous control. Human/legal entity executes payments.

Rationale: avoids legal-personhood and fiduciary ambiguity.

### 12. "Contributors later block commercial licensing"

Defense: CONTRIBUTING.md grants broad contributor rights, including commercial relicensing and Prize funding.

Critical: do not accept substantial external contributions unless the contributor terms are clearly visible and enforceable.

### 13. "GitHub does not detect the custom license"

Expected. This is a custom LicenseRef, not a standard SPDX listed/OSI license.

Mitigation:
- keep LICENSE at repo root;
- put summary in README;
- use `SPDX-License-Identifier: LicenseRef-VRAXION-Community-Source-1.0`;
- add legal/LEGAL.md for human-readable commercial rules.

### 14. "Regulator/export control blocks model anyway"

License cannot override law. Compliance with law section pushes deployment responsibility to deployers/licensees.

### 15. "Some old contributions were under PolyForm Noncommercial"

Risk: relicensing old third-party contributions requires rights or consent.

Mitigation:
- verify contributor list;
- get consent/CLA for meaningful contributions;
- if uncertain, keep old files under old terms or dual-license only owned portions.

## Main legal risks

1. Custom license may be harder for enterprises to approve than standard licenses.
2. "Attributable Net Revenue" in bundles can become a negotiation fight.
3. Founder Redirect Event needs estate/foundation hardening outside the repo.
4. AI allocator needs a real legal entity/fiduciary wrapper.
5. Copyright does not protect pure ideas or clean-room reimplementation.
6. Existing contributors must have granted sufficient rights.
7. GitHub may not display a nice license badge for a custom license.

## Recommended final hardening

Before relying on this for serious money:

1. create VRAXION legal entity or foundation;
2. assign IP to that entity;
3. make Daniel the Founder Payee by contract, not by informal expectation;
4. create a signed Founder Redirect Instrument;
5. define payment accounts for Founder Allocation and Prize Allocation;
6. require contributor terms before merging outside PRs;
7. have counsel review the license for Slovak/EU law and target-market enforceability.
