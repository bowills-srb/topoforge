"""DIAGNOSTIC (descriptive, post-hoc) -- why did the interleaved arm's
bridge-vs-capability correlation flip sign between Exp 62's discovery and
confirmation blocks?

THIS IS NOT A RESULT AND CANNOT RESCUE P1. Exp 62's pre-registered primary
FAILED (discovery partial r=+0.361 -> confirmation -0.140) and that verdict
stands however this file reads. Everything below is post-hoc analysis chosen
after seeing the failure -- exactly the forking path this project has been
caught by before (Exp 49, Exp 57). Its only legitimate output is a HYPOTHESIS
to be pre-registered and tested on a third disjoint seed block.

THE HYPOTHESIS UNDER EXAMINATION. Exp 62's P4 established a log10 functional
form, which implies saturation. If drsq saturates (or turns over) in bridge
mass, two disjoint samples drawn at different points on that curve can show
opposite local slopes while lying on ONE monotone-then-flat relationship.
Confirmation's interleaved bridge mean (8747) exceeds discovery's (7897),
which is the direction that hypothesis needs.

THE TEST THAT CAN KILL IT -- COMMON SUPPORT. If the sign flip persists when
both blocks are restricted to the bridge range they SHARE, then the two blocks
disagree on identical inputs and "different regimes" is not the explanation;
it is sampling noise plus winner's curse, and the hypothesis is dead.

Run:  venv/Scripts/python.exe <this file>
"""
import os, sys, json
sys.path.insert(0, "src")
sys.path.insert(0, os.path.join("src", "experiments"))

import numpy as np
from scipy import stats

from exp52_accuracy_endpoint import load_split
from exp62_structure_performance_law import (
    discovery_rows, confirm_rows, _f_load, BRIDGE_PATH, CONDITIONS,
)


def own_r(B, D):
    if len(B) < 4:
        return float("nan"), float("nan")
    return stats.pearsonr(B, D)


def boot_r_ci(B, D, n_boot=20000, seed=17):
    rng = np.random.default_rng(seed)
    n = len(B)
    if n < 4:
        return float("nan"), float("nan")
    v = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(B[idx]) < 1e-12 or np.std(D[idx]) < 1e-12:
            v[i] = 0.0
        else:
            v[i] = stats.pearsonr(B[idx], D[idx])[0]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    top, avail, plastic, evalset = load_split()
    labels = np.array([0] * len(evalset[top[0]]) + [1] * len(evalset[top[1]]))

    Bd, Dd, Gd = discovery_rows(labels)
    store = _f_load()
    bridge = json.load(open(BRIDGE_PATH))
    Bc, Dc, Gc = confirm_rows(store, bridge, labels)

    B = np.concatenate([Bd, Bc])
    D = np.concatenate([Dd, Dc])
    G = np.concatenate([Gd, Gc])
    BLK = np.array(["discovery"] * len(Bd) + ["confirm"] * len(Bc))

    print("=" * 92)
    print("DIAGNOSTIC -- Exp 62 interleaved sign flip   [DESCRIPTIVE, POST-HOC, NOT A RESULT]")
    print("=" * 92)
    print("  Exp 62 P1 FAILED and stays failed. This only asks whether the flip")
    print("  is a sampling-range artifact or a genuine disagreement.")

    # ---------------------------------------------------------- [1] blocks
    print("\n[1] BRIDGE-MASS DISTRIBUTIONS BY BLOCK")
    print("    {:<12} {:<10} {:>4} {:>9} {:>9} {:>9} {:>9} {:>8}".format(
        "condition", "block", "n", "mean", "median", "min", "max", "d_rsq"))
    for cond, _ in CONDITIONS:
        for blk in ("discovery", "confirm"):
            m = (G == cond) & (BLK == blk)
            print("    {:<12} {:<10} {:>4d} {:>9.0f} {:>9.0f} {:>9.0f} {:>9.0f} {:>8.4f}"
                  .format(cond, blk, m.sum(), B[m].mean(), np.median(B[m]),
                          B[m].min(), B[m].max(), D[m].mean()))
        mi = (G == cond) & (BLK == "discovery")
        mc = (G == cond) & (BLK == "confirm")
        t, pt = stats.ttest_ind(B[mi], B[mc], equal_var=False)
        ks, pks = stats.ks_2samp(B[mi], B[mc])
        print("      block bridge distributions differ?  Welch p={:.3f}   KS p={:.3f}"
              .format(pt, pks))

    # ------------------------------------------------- [2] common support
    print("\n[2] COMMON-SUPPORT TEST -- THE ONE THAT CAN KILL THE HYPOTHESIS")
    print("    Restrict both blocks to the bridge range they share, then re-correlate.")
    print("    If the flip SURVIVES here, the blocks disagree on identical inputs")
    print("    and the regime story is dead.")
    for cond, _ in CONDITIONS:
        mi = (G == cond) & (BLK == "discovery")
        mc = (G == cond) & (BLK == "confirm")
        lo = max(B[mi].min(), B[mc].min())
        hi = min(B[mi].max(), B[mc].max())
        print("\n    {}  common range [{:.0f}, {:.0f}]".format(cond, lo, hi))
        for blk, m in (("discovery", mi), ("confirm", mc)):
            k = m & (B >= lo) & (B <= hi)
            r, p = own_r(B[k], D[k])
            clo, chi = boot_r_ci(B[k], D[k])
            full_r = own_r(B[m], D[m])[0]
            print("      {:<10} n={:>3d}  r={:+.3f} (p={:.3f})  CI [{:+.3f}, {:+.3f}]"
                  "   [full-range r was {:+.3f}]"
                  .format(blk, k.sum(), r, p, clo, chi, full_r))

    # -------------------------------------------------- [3] pooled binning
    print("\n[3] POOLED SHAPE -- both blocks together, binned by bridge quartile")
    print("    A monotone-then-flat (or turning) curve would show here.")
    for cond, _ in CONDITIONS:
        m = G == cond
        b, d = B[m], D[m]
        qs = np.percentile(b, [0, 25, 50, 75, 100])
        print("\n    {}  (n={})".format(cond, m.sum()))
        print("      {:<22} {:>4} {:>10} {:>10}".format("bridge bin", "n", "mean B", "mean drsq"))
        for i in range(4):
            if i == 3:
                k = (b >= qs[i]) & (b <= qs[i + 1])
            else:
                k = (b >= qs[i]) & (b < qs[i + 1])
            if k.sum():
                print("      [{:>7.0f}, {:>7.0f}) {:>4d} {:>10.0f} {:>10.4f}"
                      .format(qs[i], qs[i + 1], k.sum(), b[k].mean(), d[k].mean()))
        med = np.median(b)
        for tag, k in (("low half ", b < med), ("high half", b >= med)):
            r, p = own_r(b[k], d[k])
            print("      {}  n={:>3d}  r={:+.3f} (p={:.3f})".format(tag, k.sum(), r, p))

    # ------------------------------------------------ [4] quadratic / form
    print("\n[4] IS THERE CURVATURE? quadratic term on pooled within-condition data")
    for cond, _ in CONDITIONS:
        m = G == cond
        b, d = B[m], D[m]
        x = (b - b.mean()) / b.std()
        X = np.column_stack([np.ones_like(x), x, x ** 2])
        beta, *_ = np.linalg.lstsq(X, d, rcond=None)
        resid = d - X @ beta
        dof = len(d) - 3
        s2 = resid @ resid / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        se2 = np.sqrt(cov[2, 2])
        tq = beta[2] / se2
        pq = 2 * (1 - stats.t.cdf(abs(tq), dof))
        vertex = -beta[1] / (2 * beta[2]) if abs(beta[2]) > 1e-12 else np.nan
        vertex_b = vertex * b.std() + b.mean() if np.isfinite(vertex) else np.nan
        inside = (np.isfinite(vertex_b) and b.min() <= vertex_b <= b.max())
        print("    {:<12} quad coef {:+.5f}  t={:+.2f}  p={:.3f}   vertex at bridge {:.0f} {}"
              .format(cond, beta[2], tq, pq, vertex_b,
                      "(inside data range)" if inside else "(outside range)"))

    # ------------------------------------------------------ [5] read-off
    print("\n[5] READ-OFF")
    print("    Interpret ONLY via [2]. If the confirm-block interleaved correlation stays")
    print("    negative on common support with a CI excluding zero, the two blocks")
    print("    disagree where they overlap -> NOT a range artifact, and no third seed")
    print("    block is justified on this hypothesis.")
    print("    If it moves toward zero / positive on common support, the flip is")
    print("    partly a range effect -> a THIRD disjoint block at controlled bridge")
    print("    mass is worth pre-registering.")
    print("\n    Either way Exp 62's P1 and P3 verdicts are unchanged.")


if __name__ == "__main__":
    main()
