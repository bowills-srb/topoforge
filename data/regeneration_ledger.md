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
| N=900 segregated | 731.8 ± 6.0 | | | running |
| N=900 interleaved | 2952.9 ± 29.1 | | | running |
| N=900 ratio | 4.04x | | | running |
| N=900 t(18) | 236.7 | | | running |
| N=900 p | 1.1e-19 | | | running |
| N=900 Cohen's d | 105.9 | | | running |
| N=5000 segregated | 4032.8 ± 26.8 | | | pending |
| N=5000 interleaved | 15349.6 ± 216.7 | | | pending |
| N=5000 ratio | 3.81x | | | pending |
| N=5000 t(18) | 163.9 | | | pending |
| N=5000 p | 2.3e-17 | | | pending |
| N=5000 Cohen's d | 73.3 | | | pending |

### Section 4.2 — persistence and detection (exp29_ant_colony)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| spatial-coherence ratio | 4.7x | | | pending |

### Section 4.3 — geometry removed (exp36)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| segregated | 726 ± 9 | | | pending |
| interleaved | 2037 ± 18 | | | pending |
| ratio | 2.81x | | | pending |

### Section 4.4 — external ITDP rule (exp35b)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| fusion mass beyond radius | 0.00 ± 0.00 (5 seeds) | | | pending |

### Section 4.5 — real SHD speech (exp37c, exp39)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| segregated bridge mass | 1161 ± 822 | | | pending |
| interleaved bridge mass | 8423 ± 1089 | | | pending |
| headline ratio | 7.25x | | | pending |
| bootstrap 95% CI | [5.05x, 11.48x] | | | pending |
| Welch p | 3.2e-14 | | | pending |
| Wilcoxon p | 4.9e-4 | | | pending |
| Cohen's d | 7.53 | | | pending |
| 4-sample re-verification | 7.22x | | | pending |
| 2-epoch ratio | 3.62x | | | pending |
| 2-epoch CI | [2.70x, 5.42x] | | | pending |
| 2-epoch p | 1.3e-7 | | | pending |
| 2-epoch d | 4.9 | | | pending |
| segregated RSD | 60–70% | | | pending |

### Section 4.6 — SpiNeMap (exp38)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| Table 3 adjacency, SpiNeMap-pop | 71.1% / 66.7% | | | running |
| Table 4 segregated | 731.8 ± 6.0 | | | running |
| Table 4 SpiNeMap-population | 903.8 ± 61.2 | | | running |
| Table 4 random | 2733.2 ± 25.9 | | | running |
| Table 4 interleaved | 2952.9 ± 29.1 | | | running |
| Table 4 SpiNeMap-functional | 2980.6 ± 16.6 | | | running |
| pop-vs-interleaved ratio | 3.27x | | | running |
| pop-vs-interleaved p, d | 9.9e-20, 42.8 | | | running |
| pop-vs-segregated | 1.24x | | | running |
| positions on axis | 7.7% / 90.1% / 100% / 101.2% | | | running |

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
| Table 8 C1 / C2 / C3 | 2909±27 / 2739±63 / 2082±36 | | | pending |
| C2-vs-C3 | 1.32x, p=3.6e-11, d=12.8 | | | pending |
| C1-vs-C3 | 1.40x, p=1.5e-16 | | | pending |
| C1-vs-C2 | 1.06x, p=5.0e-5, d=3.5 | | | pending |
| density control | 0.4%, p=0.72 | | | pending |
| mediator metrics (count/frac/cover) | placement-only | unchanged | 0% | done |

### Section 4.9 — energy frontier (exp44)

| Figure | Before | After | Δ | Status |
|---|---|---|---|---|
| Table 9, 5 conditions x 4 columns | see preprint | | | pending |
| frozen energy (bit-identical) | 5.01906e6 | | | pending |
| interleaved/segregated energy | 1.0143x, p=3.6e-5 | | | pending |
| learning gain | 4.27x | | | pending |
| energy per unit taught | 3680 vs 874 | | | pending |
| Pareto ratios vs interleaved | 0.972x / 1.006x | | | pending |
| Pareto ratios vs segregated | 0.986x / 4.296x | | | pending |
