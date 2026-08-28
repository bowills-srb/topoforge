# Regeneration ledger — the `kind='stable'` fix

Every number that appears anywhere in `topoforge_preprint.md`, with its value
before the fix and after. "Before" is whatever environment originally produced
it (mixed NumPy 1.26.4 / 2.5.1 — see the note below). "After" is deterministic:
identical under both NumPy versions, which is the point of the change.

## What the fix is

`np.argsort(..., kind='stable')` at 21 sites across the load-bearing path. The
rewire step ranks candidate connections by a score array in which most entries
are exactly zero; NumPy's default sort is unstable, so *which* of the tied-at-
zero connections got replaced depended on sort internals that changed between
NumPy 1.26.4 and 2.5.1. It is not an RNG change — the random draws are
byte-identical before and after. It changes which ties win.

Verified before regenerating anything:

| Check | Result |
|---|---|
| V1 cross-version determinism (PLB seed 0, both NumPy versions) | 2978 / 702 / 2758 under **both** — previously 2955/682 vs 2979/724 |
| V2 structural gates (`audit_deployed.py`): SpatialGrid, SparsePairState, SpiNeCluster optimum, placement geometry, CLI equality | unchanged, all pass |
| V3 `exp39 --verify-equiv` bit-for-bit physics equality vs the original implementation | `identical=True` |

V2 and V3 are the meaningful STEP-1 checks. The task framing asked to confirm
"the fix only changes which random stream is used" — that property cannot hold,
because no random stream is involved. What V2/V3 establish instead is that the
physics, the primitives, the placement geometry and the cross-implementation
equivalence relationships are all preserved; only tie-breaking moved.

## Scope note: which files were patched

Patched (21 sites): the experiments whose numbers appear in the preprint —
`exp29_ant_colony`, `exp32b_benchmark`, `exp32b_scaled`, `exp33_shd_topology`,
`exp35b_itdp_fusion_fixed`, `exp36_locality_without_geometry`,
`exp37b_v2_real_data`, `exp39_kclass_real_data` — plus `topoforge_run.py`
(the CLI must match the benchmark), `engine.py` and `engine_fast.py`.

Deliberately NOT patched: superseded experiments (exp22-28, exp30, exp31,
exp32_truenorth, exp34, exp35, exp35c-g, exp37, exp37b_dual) and `audit.py`.
Editing code whose recorded output will not be regenerated would make those
results unreproducible from their own source, which is worse than leaving them
consistent with the history that produced them.

## The diff

Status legend: `done` = regenerated and recorded; `running`; `pending`.

### Abstract

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| synthetic penalty range | 3.8–4.0x | | | pending |
| geometry-removed ratio | 2.81x | | | pending |
| real-data range | 3.6x–7.3x | | | pending |
| real-data headline | 7.25x, CI [5.05, 11.48], p=3.2e-14 | | | pending |
| SpiNeMap map-time penalty | 3.3x, 8% of the way | | | pending |
| sparse-fabric advantage | 1.46x beyond rho 1.5 | | | pending |
| energy premium | identical frozen, 1.4% plastic | | | pending |

### Section 3.4 (the reproducibility note itself — must be rewritten, not just updated)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| numpy 1.26.4 seed 0 | 2955 / 682 | n/a — fix removes the divergence | — | done |
| numpy 2.5.1 seed 0 | 2979 / 724 | n/a | — | done |
| stable-sort value | 2978 / 702 | now the only value | — | done |
| headline ratio spread | 4.33x / 4.03x / 4.24x | single value | — | pending |

### Section 4.1 — PLB (exp32b_benchmark, exp32b_scaled)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| N=900 segregated | 731.8 ± 6.0 | 703.3 ± 1.9 | −3.9% | done |
| N=900 interleaved | 2952.9 ± 29.1 | 2950.1 ± 37.7 | −0.1% | done |
| N=900 random | 2733.2 ± 25.9 | 2733.7 ± 23.3 | +0.0% | done |
| N=900 ratio | 4.04x | 4.20x | +3.9% | done |
| N=900 t / df | t(18) = 236.7 | Welch t(9.0) = 188.0 | see note | done |
| N=900 p | 1.1e-19 | 1.5e-17 (Welch) | see note | done |
| N=900 Cohen's d | 105.9 | 84.1 | −20.6% | done |

Note on the test statistic. Section 3.4 states the method is Welch's t-test
(unequal variances), but the reported `t(18)` is Student's pooled degrees of
freedom — with these variances Welch gives df ≈ 9. The regenerated row reports
Welch consistently with the stated method. For reference the pooled statistic on
the new data is t(18) = 188.0, p = 4.2e-31. Cohen's d falls because the
segregated s.d. tightened from 6.0 to 1.9, not because the effect shrank; the
ratio grew.
| N=5000 segregated | 4032.8 ± 26.8 | | | pending |
| N=5000 interleaved | 15349.6 ± 216.7 | | | pending |
| N=5000 ratio | 3.81x | | | pending |
| N=5000 t(18) | 163.9 | | | pending |
| N=5000 p | 2.3e-17 | | | pending |
| N=5000 Cohen's d | 73.3 | | | pending |

### Section 4.2 — persistence and detection (exp29_ant_colony)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| spatial-coherence ratio | 4.7x | 4.43x (C_pairs 1731 vs 391) | −5.7% | done |

Claim intact. Note for whoever reruns exp29: its printed verdict says "NO
CONSENSUS EFFECT -- coherent and random produced similar structural change",
which is assessing the edge-change metric (Δinternal −64 vs +4, Δtouching −222
vs +52 -- which are plainly not similar), not the C_pairs ratio the preprint
cites. The published figure is unaffected, but the script's self-assessment
contradicts the use the paper makes of it.

### Section 4.3 — geometry removed (exp36)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| segregated | 726 ± 9 | 701 ± 0 | −3.4% | done |
| interleaved | 2037 ± 18 | 2045 ± 10 | +0.4% | done |
| ratio | 2.81x | 2.92x | +3.9% | done |

The segregated s.d. collapsing to 0 is a real consequence, not a defect: with
zero cross-type reach nothing is learned, so taught mass is fixed by the seeded
initial connectivity, and the previous ±9 was unstable tie-breaking being read
as seed variance. The spatial PLB retains segregated variance because its
candidate score carries a 1/(1+0.05d^2) term that breaks ties by distance.

### Section 4.4 — external ITDP rule (exp35b)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| fusion mass beyond radius | 0.00 ± 0.00 (5 seeds) | 0.00 ± 0.00 (5 seeds) | **0%** | done |
| near placement (context) | not published | 5436.98 ± 1522.42 | — | done |
| no-timing control (context) | not published | 4039.42 ± 524.30 | — | done |
| inverted-U jitter sweep | qualitative | 192.7 / 3473 / 5667 / 4747 / 4265 | shape intact | done |

Exactly preserved, and necessarily so: if no cross-channel pair lies within the
deposit radius there are no fusion events at all, hence no ties to break. The
mechanism-level claim of 4.4 is the one result in the paper that the sort
change *cannot* touch.

### Section 4.5 — real SHD speech (exp37c, exp39)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| segregated bridge mass | 1161 ± 822 | 1135.1 ± 372.8 | −2.2% (s.d. −55%) | done |
| interleaved bridge mass | 8423 ± 1089 | 9329.0 ± 919.4 | +10.8% | done |
| headline ratio | 7.25x | **8.22x** | +13.4% | done |
| bootstrap 95% CI | [5.05x, 11.48x] | [7.03x, 10.05x] | narrower | done |
| Welch p | 3.2e-14 | 3.53e-14 | ~0 | done |
| Wilcoxon p | 4.9e-4 | 4.88e-4 | ~0 | done |
| Cohen's d | 7.53 | 11.68 | +55% | done |

**A CLAIM BREAKS HERE, not just a number.** Section 4.5 argues that sample
diversity is not the driver of the effect, on the grounds that the 4-sample
config (7.22x) and the 20-sample config (7.25x) were statistically
indistinguishable. Regenerated, those are 16.24x [11.70, 26.26] and 8.22x
[7.03, 10.05] — non-overlapping CIs. The claim as written is falsified by its
own experiment.

The defensible reading is NOT that sample diversity matters after all. It is
that the 4-sample ratio does not reliably estimate anything: its denominator is
311.8 of near-pure tie noise, so the original 7.22x/7.25x agreement was two
unreliable quantities happening to coincide. Under any tie-break the 20-sample,
12-seed configuration is the better-determined of the two. That paragraph needs
rewriting, not renumbering.
| 4-sample re-verification | 7.22x | **16.24x** | **+125%** | done |
| 4-sample segregated | not published | 311.8 ± 148.7 | — | done |
| 4-sample interleaved | not published | 5064.5 ± 240.0 | — | done |
| 4-sample CI | not published | [11.70x, 26.26x] | — | done |
| 2-epoch ratio | 3.62x | **5.09x** | +40.6% | done |
| 2-epoch CI | [2.70x, 5.42x] | [3.93x, 7.27x] | — | done |
| 2-epoch p | 1.3e-7 | 2.10e-14 (Welch) | — | done |
| 2-epoch d | 4.9 | 7.37 | +50% | done |
| 2-epoch segregated | not published | 1194.7 ± 719.7 | — | done |
| 2-epoch interleaved | not published | 6084.2 ± 601.9 | — | done |
| segregated RSD | 60–70% | 32.8% (matched) / 60.2% (2-epoch) | — | done |
| stated real-data range | 3.6x–7.3x | **5.1x–8.2x** | — | done |
| K=4 generalization (context) | not in preprint | 2.63x [2.43, 2.87] | — | done |

**Section 4.5 needs the most rewriting of any section. Four statements fail:**

1. *Sample diversity is not the driver* — rested on 7.22x vs 7.25x being
   indistinguishable. Now 16.24x [11.70, 26.26] vs 8.22x [7.03, 10.05],
   non-overlapping.
2. *Both endpoints comfortably bracket the 3.8–4.0x synthetic result* — the
   regenerated real-data range is 5.09x–8.22x and the regenerated synthetic
   value is 4.20x, so the real range now lies entirely ABOVE the synthetic one
   rather than bracketing it.
3. *Two of twelve seeds produced bridge mass of essentially zero* — the
   regenerated 2-epoch per-seed minimum is 274.1; no seed is near zero.
   Per-seed: 402.3, 1094.0, 1222.6, 622.1, 608.8, 2233.4, 274.1, 1447.4,
   1181.1, 1718.9, 908.2, 2623.3.
4. *Relative standard deviation 60–70% regardless of seed count* — true for the
   2-epoch regime (60.2%) but not the matched regime (32.8%).

What survives: segregation harms real-data learning, substantially and with
overwhelming significance, in every regime (5.09x–8.22x, all p < 1e-13), and
segregated remains far more variable than interleaved (60.2%/32.8% vs 9.9%).
The reliability claim holds in direction; its specific numbers do not.

### Attribution control for the Section 4.5 swing

The 4-sample real-data ratio moved 7.22x -> 16.24x (+125%), far beyond every
other change. Because that moves the headline in the *favourable* direction, it
was verified rather than assumed: the same patched code was re-run with the sort
forced back to the old unstable default.

| Configuration | segregated | interleaved | ratio |
|---|---|---|---|
| patched code + unstable sort | 688.0 ± 130.3 | 4968.5 ± 263.4 | **7.22x** |
| patched code + stable sort | 311.8 ± 148.7 | 5064.5 ± 240.0 | **16.24x** |

The control reproduces the published 7.22x exactly, so the swing is attributable
to tie-breaking alone and no defect was introduced by the patch. Interleaved
differs 1.9% between sorts; segregated differs 55%.

The consequence is larger than a changed number. Both values are legitimate
outputs of identical physics, differing only in which tied-at-zero connections
were replaced. `kind='stable'` makes that choice reproducible but not
principled — it picks index order. The published ± seed variance never contained
this uncertainty, so the precision of the real-data magnitude was overstated in a
way that adding seeds could not have revealed.

Recommended follow-up (NOT done here, outside this task's scope): break ties with
a per-seed randomization rather than deterministically, so that reported seed
variance absorbs tie uncertainty and the error bars mean what they appear to
mean. That is a change to the experiment's statistical design, not a bug fix,
and should be a deliberate decision.

### Section 4.6 — SpiNeMap (exp38)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| Table 3 adjacency, SpiNeMap-pop | 71.1% / 66.7% | 71.1% / 66.7% | 0% (placement-only) | done |
| Table 4 segregated | 731.8 ± 6.0 | 703.3 ± 1.9 | −3.9% | done |
| Table 4 SpiNeMap-population | 903.8 ± 61.2 | 876.3 ± 49.8 | −3.0% | done |
| Table 4 random | 2733.2 ± 25.9 | 2733.7 ± 23.3 | +0.0% | done |
| Table 4 interleaved | 2952.9 ± 29.1 | 2950.1 ± 37.7 | −0.1% | done |
| Table 4 SpiNeMap-functional | 2980.6 ± 16.6 | 2946.5 ± 26.1 | −1.1% | done |
| pop-vs-interleaved ratio | 3.27x | 3.37x | +3.1% | done |
| pop-vs-interleaved p, d | 9.9e-20, 42.8 | 4.6e-25, 46.9 | — | done |
| pop-vs-segregated | 1.24x | 1.25x | +0.8% | done |
| positions on axis | 7.7% / 90.1% / 100% / 101.2% | 7.7% / 90.4% / 100% / 99.8% | ~0% | done |

The functional-graph row improves the paper's wording. It was 101.2% with a
nominally significant p = 0.02 that had to be explained away as a low-noise
artifact; it is now 99.8% with p = 0.807 — genuinely indistinguishable from
interleaved, and the hedge can be deleted. But this weakens Section 4.9's
Pareto claim: at 0.999x interleaved learning, SpiNeMap-functional no longer
strictly DOMINATES interleaved (which required learning >= 1.0x). The claim
becomes lower energy at statistically equal learning. Confirmed against the
4.9 re-run before rewriting.

### Section 4.7 — dose-response and dissociation (exp40, exp41)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| Table 5, 11 rows (reach + normalized learning) | see preprint | | | pending |
| Spearman rho K=4 / K=2 | 1.00 / 0.89 | | | pending |
| dose-response rise K=4 / K=2 | 3.2x / 3.9x | | | pending |
| pooled mediation R2 / rho | 0.90 / 0.93 | | | pending |
| Table 6, 4 separations x 2 arrangements | see preprint | | | pending |
| reach ladder | 71.0 / 57.0 / 22.5 / 2.6 | unchanged (placement-only) | 0% | done |
| reach effect | 77x, rho=1.00, R2=0.98 | | | pending |
| arrangement p range | 0.09–0.28 | | | pending |

### Section 4.8 — fabric sparsity and mediator (exp42, exp42b, exp43)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| Table 7, 6 pitches x 4 conditions | see preprint | | | pending |
| interleaved decline | 1.49x | | | pending |
| interleaved plateau | 2035 / 2039 / 2039 | | | pending |
| count form pooled R2 | 0.266 | | | pending |
| coverage pooled R2 / rho | 0.832 / 0.922 | | | pending |
| incremental R2 (count / coverage) | +0.004 / +0.172 | | | pending |
| Table 8 C1 / C2 / C3 | 2909±27 / 2739±63 / 2082±36 | 2925±35 / 2719±53 / 2096±44 | +0.6 / −0.7 / +0.7% | done |
| C2-vs-C3 | 1.32x, p=3.6e-11, d=12.8 | 1.30x, p=7.3e-13, d=12.8 | −1.5% | done |
| C1-vs-C3 | 1.40x, p=1.5e-16 | 1.40x, p=1.7e-15, d=20.7 | 0.0% | done |
| C1-vs-C2 | 1.06x, p=5.0e-5, d=3.5 | 1.08x, p=8.4e-7, d=4.6 | +1.9% | done |

All three registered verdicts are unchanged (count REFUTED, coverage confirmed
in isolation, frac non-zero). Note the pattern this establishes: every condition
here learns substantially (2000-2900) and every one is stable to within 1%. The
large swings in this regeneration are confined to conditions that learn almost
nothing, where the ratio's denominator is dominated by tie-breaking.
| density control | 0.4%, p=0.72 | | | pending |
| mediator metrics (count/frac/cover) | placement-only | unchanged | 0% | done |

### Section 4.9 — energy frontier (exp44)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| Table 9 segregated | 2.529e6 / 687 | 2.525e6 / 704 | −0.2% / +2.5% | done |
| Table 9 SpiNeMap-population | 2.530e6 / 861 | 2.497e6 / 875 | −1.3% / +1.6% | done |
| Table 9 random | 2.564e6 / 2741 | 2.556e6 / 2730 | −0.3% / −0.4% | done |
| Table 9 interleaved | 2.565e6 / 2935 | 2.566e6 / 2959 | +0.0% / +0.8% | done |
| Table 9 SpiNeMap-functional | 2.493e6 / 2952 | 2.501e6 / 2951 | +0.3% / −0.0% | done |
| frozen energy (bit-identical) | 5.01906e6 | 5.01906e6 | 0% (structural) | done |
| interleaved/segregated energy | 1.0143x, p=3.6e-5 | 1.0162x, p=1.9e-9 | +0.2pp | done |
| energy per unit taught | 3680 vs 874 | 3589 vs 867 | −2.5% / −0.8% | done |
| Pareto vs interleaved | 0.972x / 1.006x = **dominates** | 0.975x / **0.997x** = **trades off** | claim breaks | done |
| Pareto vs segregated | 0.986x / 4.296x = dominates | 0.990x / 4.195x = dominates | holds | done |
| Pareto vs SpiNeMap-pop | 0.985x / 3.430x = dominates | **1.001x** / 3.373x = **trades off** | claim breaks | done |

**A SECOND CLAIM BREAKS.** Section 4.9 states the association-aware mapper "is
Pareto-dominant over every other condition tested" and "there is no axis on
which it is worse". That is now false: it dominates only the segregated
baseline. Against interleaved it buys 2.5% lower energy for a 0.3% learning
deficit; against population-graph SpiNeMap it is 0.1% HIGHER energy, because
that condition's energy fell 2.530e6 -> 2.497e6 and overtook it. The practical
recommendation survives in weaker form (~3.4x the learning of the map-time
graph at essentially equal energy) but the dominance sentence must go.
