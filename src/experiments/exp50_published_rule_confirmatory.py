"""Exp 50: Confirmatory test that the segregation penalty survives under a
PUBLISHED, value-free, correlation-free structural-plasticity rule.

MOTIVATION. Exp 38 closed the "is the baseline a strawman?" gap on the
MAPPER side: a real published partitioner (SpiNeMap) reproduces the
pathological segregated placement, so the segregated condition is what a
real energy-only tool actually emits. The matching gap on the GROWTH-RULE
side stayed open: every number in this project comes from `LocalLife`'s
own C/E/V machinery (correlation, eligibility, reward-attributed value on
candidate pairs), so an obvious objection remains -- is the penalty a fact
about PLACEMENT, or an artifact of THIS rule's candidate-scoring design?

`tinker_published_rule_butz_vanooyen.py` (exploratory, 5 seeds 0-4,
no pre-registration) reimplemented the Butz & van Ooyen (2013) homeostatic
model -- Gaussian growth curve F(rate) = 2*exp(-((rate-eps)/zeta)^2) - 1
driving per-neuron axonal/dendritic element budgets from each neuron's OWN
firing rate, with synapse formation decided purely by a spatial Gaussian
exp(-d^2/2sigma^2). Nothing in it is type-aware, correlation-aware, or
reward-aware. It found segregated cross-type edge fraction 0.364 +/- 0.008
vs interleaved 0.810 +/- 0.003 (Welch p = 3.9e-10), and specific taught
pairs 5.4 +/- 2.4 vs 897.8 +/- 41.8.

That pass is not sufficient on its own, for the reasons this project has
already learned twice (Exp 43 after Exp 42b; Exp 45c after Exp 45/45b):
5 seeds, no registered endpoint, no registered decision rule, and -- most
importantly -- the EPOCH constant was changed mid-debugging after seeing
that EPOCH=40 produced zero edges. Parameters touched while looking at
output cannot also serve as a pre-registered test. This script re-runs the
identical mechanism on FRESH SEEDS with the endpoint, thresholds, and
decision rule all fixed in the text below before execution.

HONESTY ABOUT WHAT KIND OF TEST THIS IS. The predictions below are DERIVED
FROM the exploratory pass -- they are not blind. This is a pre-registered
REPLICATION on disjoint seeds, exactly as Exp 45c was for Exp 45/45b. Its
value is that the endpoint and thresholds were committed before these
particular seeds were run, not that the direction was unknown.

DESIGN.
  Mechanism: imported verbatim from tinker_published_rule_butz_vanooyen
  (run_butz_vanooyen, calibrate_target, measure, and every constant).
  Importing rather than re-typing is deliberate: it makes a transcription
  divergence between the exploratory and confirmatory runs impossible.

  Conditions: SEGREGATED = make_placement("vlsi"), INTERLEAVED =
  make_placement("topoforge"), from exp32b_benchmark.

  Seeds: 100-111 (12 seeds), DISJOINT from the exploratory run's 0-4, and
  matching exp37c's 12-seed standard. No seed contributing to the numbers
  quoted above contributes to the numbers below.

STRUCTURAL PARITY (why this comparison cannot be adjacency-confounded).
CLAUDE.md's gotcha list records a bogus "1,448x" that came from comparing
placements with ~57x different cross-type adjacency counts -- ratio
reflected opportunity, not arrangement. That failure mode is impossible
here, and section [0] asserts it rather than assuming it: exp32b builds
both placements from the same `neurons_in_core(core_id, ...)` call with
the same seeded RNG, so the two conditions have BIT-IDENTICAL coordinates
and identical per-type population counts (180 of each of 5 types). Only
the assignment of types to those fixed positions differs. Every spatial
quantity the rule can see -- inter-neuron distances, the pairing kernel,
neighbour density -- is therefore exactly equal across conditions.

PRIMARY ENDPOINT, REGISTERED: cross-type FRACTION of final edges
(fraction of grown edges whose two endpoints have different types).
Registered as primary over the specific-taught-pair count for three
reasons:
  (a) Rule-agnostic. Butz & van Ooyen has no term that privileges
      exp32b's particular pattern-coupled pairs (0,3) and (1,4) over any
      other cross-type pair. Scoring only those pairs would import this
      project's own pattern schedule into a test whose whole purpose is
      independence from this project's design choices.
  (b) Robust to the total-edge asymmetry (see LIMITATIONS): a fraction
      normalises edge count out; a raw count does not.
  (c) It is the measure on which the ceiling argument below is stateable.

THE CEILING, AND HOW IT SHAPES THE PREDICTION. With 180 neurons in each
of NC=5 types and type-blind pairing, the expected cross-type fraction for
a random partner is (900-180)/(900-1) = 0.801. Interleaved's exploratory
0.810 sits AT that chance ceiling. So the correct scientific statement is
NOT "interleaving boosts cross-type wiring" -- it cannot exceed chance,
and the ratio between conditions is bounded above by ~0.801/x. The claim
is the mirror image: SEGREGATION SUPPRESSES cross-type wiring far BELOW
what type-blind chance would produce, because same-type blocks dominate
each neuron's spatial neighbourhood and the pairing kernel is local. The
predictions are worded as a suppression test accordingly.

PREDICTIONS, REGISTERED BEFORE RUNNING:
  P1 (manipulation checks, not the test -- both parts must hold):
    P1a STRUCTURAL PARITY: the two conditions' coordinate arrays are
        exactly equal, and per-type population counts are exactly equal
        (180 each). If this fails, exp32b's placement code has changed
        under this script and the comparison is confounded.
    P1b HOMEOSTASIS SETTLES: mean final edge count is nonzero and not
        saturated in BOTH conditions -- registered band [1000, 20000]
        (hard ceiling is N*MAX_ELEMENTS = 900*25 = 22500). This guards
        the exact failure already hit once: with a mis-set EPOCH the rate
        went bimodal, homeostasis never settled, and every condition
        returned ZERO edges. A zero or saturated run must not be read as
        a placement effect.
  P2 (the actual test, primary endpoint = cross-type edge fraction):
    P2a interleaved > segregated, Welch p < 0.05.
    P2b segregated mean cross-fraction < 0.60, i.e. suppressed well below
        the 0.801 type-blind chance ceiling (exploratory: 0.364; 0.60 is
        a deliberately lenient bound that still demands real suppression).
    P2c interleaved mean cross-fraction > 0.70, i.e. at or near that
        ceiling (exploratory: 0.810).

DECISION RULE:
  - P1 fails: the run is COMPROMISED. Do not interpret P2 at all until
    the parity or settling failure is understood.
  - P1 holds and P2a+P2b+P2c all hold: CONFIRMS, at confirmatory-grade
    rigor and on disjoint seeds, that the segregation penalty is not an
    artifact of this project's own value/correlation rewire scoring. The
    generality claim becomes reportable.
  - P1 holds, P2a holds, but P2b and/or P2c fails: PARTIAL. The
    directional effect is real and significant at matched geometry, but
    the "suppressed far below chance / saturated at chance" framing is
    mis-calibrated; report the measured fractions, not the framing.
  - P2a fails (p >= 0.05, or wrong direction): REFUTED at this power.
    The 5-seed exploratory result did not replicate on fresh seeds, and
    the "survives under a published rule" claim must be withdrawn.

SECONDARY ENDPOINT (reported, not decisive): specific taught pairs
(0,3)+(1,4), for continuity with exp32b's own metric.

LIMITATIONS, STATED BEFORE RUNNING:
  1. TOTAL-EDGE ASYMMETRY. The exploratory pass found segregated ended
     with MORE total edges than interleaved (6585 vs 5014) and did not
     explain why (plausibly within-block firing correlations shifting
     local rate dynamics, hence element budgets -- uninvestigated). The
     primary endpoint is a FRACTION, so this asymmetry is normalised out
     of it. It is NOT normalised out of the secondary raw-count endpoint
     -- but there it runs AGAINST the effect (segregated has more edges
     yet fewer cross-type ones), making the secondary conservative rather
     than inflated. Section [1] reports total edges per condition so this
     is visible rather than buried.
  2. MECHANISM SIMPLIFICATION vs. the published model, inherited
     unchanged from the exploratory script and disclosed there: element
     quotas are recomputed and the full edge set re-derived each epoch,
     rather than the paper's incremental per-element binding/unbinding.
     The two properties the test needs (local rate-driven budgets,
     type-blind spatial pairing) are preserved; synapse-by-synapse
     persistence across epochs is not.
  3. A single homeostatic target (eps, zeta) is calibrated once on a
     feedforward-only pilot and shared by both conditions. This is the
     fair choice -- per-condition calibration would let the target absorb
     the very difference under test -- but it does mean neither condition
     is individually tuned.
  4. One rule, one geometry family (exp32b's PLB core layout), N=900.
     Generality across other published plasticity rules is untested.

Run: python src/experiments/exp50_published_rule_confirmatory.py
"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from scipy import stats

from exp32b_benchmark import make_placement, NC, N
from exp37c_real_data_scaled import cohens_d_pooled, bootstrap_ratio_ci
from tinker_published_rule_butz_vanooyen import (
    run_butz_vanooyen, calibrate_target, measure,
    N_EPOCHS, STEPS, EPOCH, MAX_ELEMENTS, FORM_SIGMA,
)

SEEDS = list(range(100, 112))          # disjoint from exploratory 0-4

EDGE_BAND = (1000.0, 20000.0)          # P1b registered settling band
SEG_CEILING_MAX = 0.60                 # P2b registered
INT_CEILING_MIN = 0.70                 # P2c registered


def chance_cross_fraction(cids):
    """Expected cross-type fraction for a type-blind random partner:
    for a neuron of type t, (N - count(t)) / (N - 1), averaged over
    neurons. With balanced types this is (NC-1)/NC up to the -1."""
    counts = np.bincount(cids, minlength=NC).astype(float)
    per_neuron = (len(cids) - counts) / (len(cids) - 1.0)
    return float((counts * per_neuron).sum() / len(cids))


def bootstrap_diff_ci(a, b, n_boot=10000, seed=12345):
    """Percentile bootstrap 95% CI on mean(a) - mean(b). Reported
    alongside the ratio CI because the primary endpoint is a fraction
    bounded above by the chance ceiling, where a difference is the more
    interpretable contrast."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    diffs = [a[rng.integers(0, n, n)].mean() - b[rng.integers(0, n, n)].mean()
             for _ in range(n_boot)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def sweep(coords, cids, eps, zeta, seeds=SEEDS):
    rows = []
    for s in seeds:
        src, dst = run_butz_vanooyen(coords, cids, s, eps, zeta)
        rows.append(measure(src, dst, cids))
        print("    seed {:>3}: edges={:>7d}  cross_frac={:.3f}  taught={:>6d}".format(
            s, rows[-1]["n_edges"], rows[-1]["cross_frac"], rows[-1]["taught"]))
        sys.stdout.flush()
    return rows


def cols(rows):
    return {k: np.array([r[k] for r in rows], dtype=float)
            for k in ("n_edges", "taught", "cross_frac", "cross_count")}


if __name__ == "__main__":
    print("=" * 88)
    print("EXP 50: CONFIRMATORY -- does the segregation penalty survive a PUBLISHED,")
    print("value-free, correlation-free plasticity rule? (predictions fixed above)")
    print("=" * 88)

    coords_seg, cids_seg = make_placement("vlsi")
    coords_int, cids_int = make_placement("topoforge")

    print("\n[0] Structural audit (P1a: parity between conditions)")
    coords_identical = np.array_equal(coords_seg, coords_int)
    counts_seg = np.bincount(cids_seg, minlength=NC)
    counts_int = np.bincount(cids_int, minlength=NC)
    counts_identical = np.array_equal(np.sort(counts_seg), np.sort(counts_int))
    print("  N={}  NC={}  epochs={}  steps={}  form_sigma={}".format(
        N, NC, N_EPOCHS, STEPS, FORM_SIGMA))
    print("  coordinates bit-identical across conditions : {}".format(coords_identical))
    print("  per-type counts  segregated : {}".format(counts_seg.tolist()))
    print("  per-type counts  interleaved: {}".format(counts_int.tolist()))
    print("  per-type counts identical                   : {}".format(counts_identical))
    ceiling = chance_cross_fraction(cids_int)
    print("  type-blind chance cross-fraction ceiling    : {:.3f}".format(ceiling))
    p1a = bool(coords_identical and counts_identical)
    print("  P1a {}".format("HOLDS" if p1a else "FAILS -- comparison is confounded"))

    eps, zeta = calibrate_target(coords_int, cids_int)
    print("\n  calibrated homeostatic target: eps={:.4f}  zeta={:.4f}".format(eps, zeta))
    print("  (feedforward-only pilot; calibrated once, shared by both conditions)")

    print("\n[1] Running {} epochs ({} steps), grown from zero, {} seeds/condition".format(
        N_EPOCHS, STEPS, len(SEEDS)))
    print("\n  SEGREGATED (vlsi)")
    c_seg = cols(sweep(coords_seg, cids_seg, eps, zeta))
    print("\n  INTERLEAVED (topoforge)")
    c_int = cols(sweep(coords_int, cids_int, eps, zeta))

    print("\n" + "-" * 88)
    print("  {:<24} {:>10} {:>18} {:>18}".format(
        "condition", "edges", "cross_frac", "taught(0,3/1,4)"))
    for label, c in (("SEGREGATED", c_seg), ("INTERLEAVED", c_int)):
        print("  {:<24} {:>10.1f} {:>11.3f}+/-{:.3f} {:>11.1f}+/-{:.1f}".format(
            label, c["n_edges"].mean(),
            c["cross_frac"].mean(), c["cross_frac"].std(ddof=1),
            c["taught"].mean(), c["taught"].std(ddof=1)))

    print("\n[2] P1b (manipulation check): does homeostasis settle in both conditions?")
    print("    registered band: mean edges in [{:.0f}, {:.0f}]  (hard ceiling N*MAX_ELEMENTS={:.0f})".format(
        EDGE_BAND[0], EDGE_BAND[1], N * MAX_ELEMENTS))
    p1b = True
    for label, c in (("SEGREGATED", c_seg), ("INTERLEAVED", c_int)):
        ok = EDGE_BAND[0] <= c["n_edges"].mean() <= EDGE_BAND[1]
        p1b = p1b and ok
        print("    {:<14} mean edges = {:>8.1f}   {}".format(
            label, c["n_edges"].mean(), "in band" if ok else "OUT OF BAND"))
    print("  P1b {}".format("HOLDS" if p1b else "FAILS -- homeostasis did not settle"))
    p1 = p1a and p1b
    print("  P1 OVERALL: {}".format("HOLDS" if p1 else "FAILS"))

    print("\n[3] P2 (the actual test) -- PRIMARY ENDPOINT: cross-type edge fraction")
    cf_i, cf_s = c_int["cross_frac"], c_seg["cross_frac"]
    t_ind, p_ind = stats.ttest_ind(cf_i, cf_s, equal_var=False)
    d = cohens_d_pooled(cf_i, cf_s)
    r_lo, r_hi = bootstrap_ratio_ci(cf_i, cf_s)
    d_lo, d_hi = bootstrap_diff_ci(cf_i, cf_s)
    ratio = cf_i.mean() / max(cf_s.mean(), 1e-9)
    print("    segregated  : {:.3f} +/- {:.3f}".format(cf_s.mean(), cf_s.std(ddof=1)))
    print("    interleaved : {:.3f} +/- {:.3f}".format(cf_i.mean(), cf_i.std(ddof=1)))
    print("    chance ceiling (type-blind): {:.3f}".format(ceiling))
    print("    Welch t = {:.2f}, p = {:.3e}".format(t_ind, p_ind))
    print("    Cohen's d (pooled)         : {:.2f}".format(d))
    print("    ratio interleaved/segregated: {:.2f}x   bootstrap 95% CI [{:.2f}x, {:.2f}x]".format(
        ratio, r_lo, r_hi))
    print("    difference (int - seg)      : {:+.3f}  bootstrap 95% CI [{:+.3f}, {:+.3f}]".format(
        cf_i.mean() - cf_s.mean(), d_lo, d_hi))

    p2a = bool(cf_i.mean() > cf_s.mean() and p_ind < 0.05)
    p2b = bool(cf_s.mean() < SEG_CEILING_MAX)
    p2c = bool(cf_i.mean() > INT_CEILING_MIN)
    print("\n    P2a  interleaved > segregated, p < 0.05      : {}".format(p2a))
    print("    P2b  segregated  < {:.2f} (suppressed below chance): {}".format(SEG_CEILING_MAX, p2b))
    print("    P2c  interleaved > {:.2f} (at/near chance)        : {}".format(INT_CEILING_MIN, p2c))

    print("\n[4] SECONDARY ENDPOINT (reported, not decisive): specific taught pairs (0,3)+(1,4)")
    tg_i, tg_s = c_int["taught"], c_seg["taught"]
    t2, p2_ = stats.ttest_ind(tg_i, tg_s, equal_var=False)
    print("    segregated  : {:.1f} +/- {:.1f}".format(tg_s.mean(), tg_s.std(ddof=1)))
    print("    interleaved : {:.1f} +/- {:.1f}".format(tg_i.mean(), tg_i.std(ddof=1)))
    print("    Welch t = {:.2f}, p = {:.3e}   Cohen's d = {:.2f}".format(
        t2, p2_, cohens_d_pooled(tg_i, tg_s)))
    if tg_s.mean() > 0:
        print("    ratio interleaved/segregated: {:.1f}x".format(tg_i.mean() / tg_s.mean()))
    else:
        print("    segregated is at the floor (0.0) -- ratio undefined, floor effect.")
    print("    NOTE: raw counts are NOT normalised for the total-edge asymmetry below;")
    print("    that asymmetry runs AGAINST the effect, so this endpoint is conservative.")
    print("    total edges: segregated={:.1f}  interleaved={:.1f}  (seg/int = {:.2f}x)".format(
        c_seg["n_edges"].mean(), c_int["n_edges"].mean(),
        c_seg["n_edges"].mean() / max(c_int["n_edges"].mean(), 1e-9)))

    print("\n" + "=" * 88)
    print("[5] VERDICT (by the decision rule registered above)")
    print("=" * 88)
    if not p1:
        print("  COMPROMISED -- P1 failed. Do not interpret P2 until the parity or")
        print("  settling failure is understood.")
    elif p2a and p2b and p2c:
        print("  CONFIRMED at confirmatory-grade rigor, on seeds disjoint from the")
        print("  exploratory pass: segregated placement suppresses cross-type connectivity")
        print("  far below the type-blind chance ceiling, while interleaved sits at it,")
        print("  under a PUBLISHED plasticity rule containing no value, correlation, or")
        print("  reward term. The penalty is not an artifact of this project's own C/E/V")
        print("  rewire scoring. Reportable.")
    elif p2a:
        print("  PARTIAL confirmation: the directional effect is real and significant at")
        print("  bit-identical geometry (P2a holds), but the registered calibration failed")
        print("  (P2b={}, P2c={}). Report the measured fractions, not the".format(p2b, p2c))
        print("  'suppressed far below / saturated at chance' framing.")
    else:
        print("  REFUTED at this power: the exploratory 5-seed result did NOT replicate on")
        print("  fresh seeds. The 'survives under a published rule' claim should be")
        print("  withdrawn pending investigation.")
