"""Exp 45c: Confirmatory test of the filament/blob topology-fragility finding.

MOTIVATION. Exp 45/45b found that connected filament networks collapse to
net-negative growth at low coverage while isolated blobs stay robust, and
ruled out the obvious confound (the effect is not an artifact of the
"gel" shape rework -- both polygonal and smooth filament shapes show it).
But that sweep was built ADAPTIVELY -- parameters tuned live against
results as they came in -- and count was only approximately matched
between the blob and filament families at each coverage rung, not tightly
controlled. Per this project's own standard (Exp 43 was built the same
way after Exp 42b's coverage form was chosen post-hoc), an adaptively-built
finding needs a confirmatory run: placements fixed in advance, predictions
and a decision rule registered before execution, count matched tightly
this time rather than approximately.

DESIGN. A 2x2: topology (blob / filament) x coverage level (high ~1.0 /
low ~0.4), four placements, all N=900, built once, before any of this
script's output was seen:
  BLOB-HIGH      = Exp 43's "dilute30" (C2), verbatim.        count=6.00 cover=1.000
  FILAMENT-HIGH  = density field, alpha=1.0, sigma=1.6, thr=0.35.  count=6.60 cover=0.996
  BLOB-LOW       = Exp 43's "clumped30" (C3), verbatim.        count=6.25 cover=0.417
  FILAMENT-LOW   = density field, alpha=0.0, sigma=2.4, thr=1.0,
                   bias_strength=0.99 (found by a small grid search over
                   sigma/threshold BEFORE this script was run, searching
                   only for count/coverage match to clumped30 -- the
                   search did not look at growth outcomes).  count=6.55 cover=0.391
Count matches within 9% at both rungs (6.00 vs 6.60; 6.25 vs 6.55);
coverage matches within 6% (1.000 vs 0.996; 0.417 vs 0.391) -- both
substantially tighter than Exp 45/45b's approximate matching.

PREDICTIONS, REGISTERED BEFORE RUNNING:
  P1 (manipulation check, not the test): at HIGH coverage, BLOB-HIGH and
     FILAMENT-HIGH growth are statistically indistinguishable (Welch
     p > 0.05). This has now been found twice (Exp 45 v2, Exp 45b); if it
     fails to replicate a third time, something about this run's setup is
     wrong and the LOW-coverage result below should not be trusted either.
  P2 (the actual test): at LOW coverage, matched count and coverage,
     FILAMENT-LOW growth is significantly BELOW BLOB-LOW growth (Welch
     p < 0.05), AND FILAMENT-LOW's mean growth is negative while
     BLOB-LOW's is positive.

DECISION RULE:
  - P1 holds and P2 fully holds (significant gap, FILAMENT-LOW < 0 <
    BLOB-LOW): CONFIRMS the topology-fragility finding at
    confirmatory-grade rigor. This becomes reportable as a real result.
  - P1 holds but P2's significance holds while the sign prediction does
    not (FILAMENT-LOW < BLOB-LOW but both positive, or both negative):
    PARTIAL confirmation -- topology has a real, matched-count/coverage
    effect, but the sharp "crosses to net loss" framing from the
    exploratory pass was too strong; report the calibrated effect size
    instead.
  - P2's significance test fails (p > 0.05): REFUTES the standing
    finding at this coverage level -- the Exp 45/45b gap was driven by
    residual count/coverage mismatch, not a genuine topology effect, and
    the topology-fragility claim should be withdrawn or narrowed.
  - P1 fails: the run is compromised (an unnoticed geometry or harness
    change); do not interpret P2 until P1 is understood.

8 seeds, matching this project's standard for this benchmark family.

Run: python src/experiments/exp45c_topology_confirmatory.py
"""
import numpy as np
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats

from exp43_coverage_confirmatory import make_condition, mediators
from exp45_cosmic_web_topology import make_filament_density_biased, growth_of, SEEDS

CELLS = {
    "BLOB-HIGH": ("blob", "dilute30", None),
    "FILAMENT-HIGH": ("filament", None, dict(alpha=1.0, sigma=1.6, threshold=0.35, bias_strength=0.85)),
    "BLOB-LOW": ("blob", "clumped30", None),
    "FILAMENT-LOW": ("filament", None, dict(alpha=0.0, sigma=2.4, threshold=1.0, bias_strength=0.99)),
}


def build(cell):
    kind, blob_name, fkwargs = CELLS[cell]
    if kind == "blob":
        return make_condition(blob_name, seed=0)
    return make_filament_density_biased(seed=0, **fkwargs)


if __name__ == "__main__":
    print("=" * 78)
    print("EXP 45c: CONFIRMATORY topology-fragility test (predictions above, fixed before running)")
    print("=" * 78)

    print("\n[0] Structural audit: does the geometry match what's registered above?")
    results = {}
    for cell in ("BLOB-HIGH", "FILAMENT-HIGH", "BLOB-LOW", "FILAMENT-LOW"):
        coords, cids = build(cell)
        m = mediators(coords, cids)
        results[cell] = {"coords": coords, "cids": cids, "mediators": m}
        print("  {:<15} count={:.2f}  cover={:.3f}".format(cell, m["count"], m["cover"]))

    print("\n[1] Running the benchmark (growth = plastic - frozen taught mass), {} seeds each".format(len(SEEDS)))
    for cell in ("BLOB-HIGH", "FILAMENT-HIGH", "BLOB-LOW", "FILAMENT-LOW"):
        g = growth_of(results[cell]["coords"], results[cell]["cids"], seeds=SEEDS)
        results[cell]["growth"] = g
        print("  {:<15} growth={:+8.1f} +/- {:<6.1f}".format(cell, g.mean(), g.std()))

    print("\n" + "=" * 78)
    print("[2] P1 (manipulation check): BLOB-HIGH vs FILAMENT-HIGH")
    print("=" * 78)
    t1, p1 = stats.ttest_ind(results["FILAMENT-HIGH"]["growth"], results["BLOB-HIGH"]["growth"], equal_var=False)
    print("  BLOB-HIGH      growth={:+.1f}+/-{:.1f}".format(results["BLOB-HIGH"]["growth"].mean(), results["BLOB-HIGH"]["growth"].std()))
    print("  FILAMENT-HIGH  growth={:+.1f}+/-{:.1f}".format(results["FILAMENT-HIGH"]["growth"].mean(), results["FILAMENT-HIGH"]["growth"].std()))
    print("  Welch t={:.2f}, p={:.3f}".format(t1, p1))
    p1_holds = p1 > 0.05
    print("  P1 {}".format("HOLDS (indistinguishable, as predicted)" if p1_holds else "FAILS (unexpected difference at high coverage)"))

    print("\n" + "=" * 78)
    print("[3] P2 (the actual test): BLOB-LOW vs FILAMENT-LOW")
    print("=" * 78)
    t2, p2 = stats.ttest_ind(results["FILAMENT-LOW"]["growth"], results["BLOB-LOW"]["growth"], equal_var=False)
    print("  BLOB-LOW      growth={:+.1f}+/-{:.1f}".format(results["BLOB-LOW"]["growth"].mean(), results["BLOB-LOW"]["growth"].std()))
    print("  FILAMENT-LOW  growth={:+.1f}+/-{:.1f}".format(results["FILAMENT-LOW"]["growth"].mean(), results["FILAMENT-LOW"]["growth"].std()))
    print("  Welch t={:.2f}, p={:.2e}".format(t2, p2))
    sig = p2 < 0.05
    sign_pred = results["FILAMENT-LOW"]["growth"].mean() < 0 < results["BLOB-LOW"]["growth"].mean()
    print("  significant gap: {}".format(sig))
    print("  sign prediction (filament<0<blob): {}".format(sign_pred))

    print("\n" + "=" * 78)
    print("[4] VERDICT")
    print("=" * 78)
    if not p1_holds:
        print("  P1 FAILED -- do not trust the P2 interpretation below without investigating why")
        print("  the high-coverage manipulation check broke.")
    elif sig and sign_pred:
        print("  CONFIRMED at confirmatory-grade rigor: topology-fragility is real, matched")
        print("  count and coverage, both predictions held. Reportable.")
    elif sig and not sign_pred:
        print("  PARTIAL confirmation: topology has a real, significant, matched-count/coverage")
        print("  effect, but the sharp 'crosses to net loss' framing was too strong at this")
        print("  specific coverage level -- report the calibrated effect size, not the sign flip.")
    else:
        print("  REFUTED at this coverage level: no significant difference at matched count")
        print("  and coverage -- the Exp 45/45b gap was likely driven by residual mismatch,")
        print("  not a genuine topology effect. The topology-fragility claim should be")
        print("  withdrawn or substantially narrowed.")
