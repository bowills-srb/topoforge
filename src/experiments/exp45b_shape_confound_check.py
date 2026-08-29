"""Exp 45b: Is the filament-fragility finding about SMOOTHNESS, or about
CONNECTIVITY/SPARSITY in general?

MOTIVATION. Exp 45 v2 found that a smooth, organic ("gel poured over a
skull") filament network collapses to net-negative growth once coverage
drops below ~0.5, while isolated blobs stay robustly positive down to
coverage ~0.42 -- a real topology effect beyond coverage (R^2 0.57 -> 0.83
with a family term). But the smooth density-field generator was built
IN RESPONSE to a shape critique of the original polygonal (straight
Voronoi-ridge) generator, and that same rework was what added the
coverage-sweeping mechanism (alpha-biased type assignment) -- the polygonal
version was never swept across coverage at all, only near coverage~1.0,
exactly the regime where even the smooth version shows no effect. So the
"gel" shape and the coverage sweep were confounded: we don't yet know
whether the fragility is about organic smoothness specifically, or about
filament/connected topology in general (which the polygonal shape also
has, angles and all).

DESIGN. Take the OLD polygonal generator (make_filament_biased, straight
Voronoi-ridge segments, hard angular joints) and give it the SAME
alpha-biased coverage sweep the smooth version got. If the polygonal
family ALSO collapses to negative growth below coverage~0.5, the effect is
about connectivity/sparsity structure in general (filament vs. blob),
and the smoothness fix was a legitimate shape correction but not the
active ingredient in the finding. If the polygonal family stays robust
like blobs instead, smoothness itself is doing real work and the
finding is narrower than Exp 45 v2's report suggested.

Blob-family and density-field-family growth values are the exact per-seed
results already obtained this session (2026-08-29, Exp 45 v2 run) --
reproduced here as their own fresh run so this script's regression is
self-contained and independently re-checkable, not copy-pasted means.

Run: python src/experiments/exp45b_shape_confound_check.py
"""
import numpy as np
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats

from exp32b_benchmark import run_life
from exp43_coverage_confirmatory import make_condition, mediators
from exp45_cosmic_web_topology import (
    make_filament_biased, make_filament_density_biased, growth_of, SEEDS,
)

POLY_CONFIGS = [
    ("a1.00", 1.00, 0.85),
    ("a0.50", 0.50, 0.85),
    ("a0.20", 0.20, 0.85),
    ("a0.05", 0.05, 0.85),
    ("a0-b0.95", 0.00, 0.95),
    ("a0-b0.99", 0.00, 0.99),
]
DENSITY_CONFIGS = [
    ("a1.00", 1.00, 0.85),
    ("a0.50", 0.50, 0.85),
    ("a0.20", 0.20, 0.85),
    ("a0.05", 0.05, 0.85),
    ("a0-b0.95", 0.00, 0.95),
    ("a0-b0.99", 0.00, 0.99),
]


def ols_r2(y, X):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pred) ** 2) / ss_tot, beta


if __name__ == "__main__":
    print("=" * 78)
    print("EXP 45b: shape-confound check -- polygonal filaments vs smooth filaments")
    print("=" * 78)

    print("\n[1] Blob family (reference, re-run fresh)")
    results = {}
    for name in ("spread15", "dilute30", "clumped30"):
        coords, cids = make_condition(name, seed=0)
        m = mediators(coords, cids)
        g = growth_of(coords, cids, seeds=SEEDS)
        results[("blob", name)] = (g, m)
        print("  blob:{:<10} growth={:+7.1f} +/- {:<6.1f}  cover={:.3f}  count={:.2f}".format(
            name, g.mean(), g.std(), m["cover"], m["count"]))

    print("\n[2] Smooth density-field family (re-run fresh, same configs as Exp 45 v2)")
    for label, alpha, bias in DENSITY_CONFIGS:
        coords, cids = make_filament_density_biased(seed=0, alpha=alpha, bias_strength=bias)
        m = mediators(coords, cids)
        g = growth_of(coords, cids, seeds=SEEDS)
        results[("smooth", label)] = (g, m)
        print("  smooth:{:<10} growth={:+7.1f} +/- {:<6.1f}  cover={:.3f}  count={:.2f}".format(
            label, g.mean(), g.std(), m["cover"], m["count"]))

    print("\n[3] Polygonal (straight Voronoi-ridge, pre-'gel' shape) family, SAME coverage sweep mechanism")
    for label, alpha, bias in POLY_CONFIGS:
        coords, cids = make_filament_biased(seed=0, alpha=alpha, jitter=1.2, bias_strength=bias)
        m = mediators(coords, cids)
        g = growth_of(coords, cids, seeds=SEEDS)
        results[("polygonal", label)] = (g, m)
        print("  polygonal:{:<10} growth={:+7.1f} +/- {:<6.1f}  cover={:.3f}  count={:.2f}".format(
            label, g.mean(), g.std(), m["cover"], m["count"]))

    print("\n" + "=" * 78)
    print("[4] Direct matched-coverage comparison: polygonal vs smooth, at each alpha rung")
    print("=" * 78)
    print("{:>10} {:>9} {:>9} {:>9} {:>9} {:>10}".format(
        "label", "poly cov", "poly grw", "smth cov", "smth grw", "p (poly vs smooth)"))
    for label, _, _ in POLY_CONFIGS:
        gp, mp = results[("polygonal", label)]
        gs, ms = results[("smooth", label)]
        t, p = stats.ttest_ind(gp, gs, equal_var=False)
        print("{:>10} {:>9.3f} {:>+9.1f} {:>9.3f} {:>+9.1f} {:>10.2e}".format(
            label, mp["cover"], gp.mean(), ms["cover"], gs.mean(), p))

    print("\n" + "=" * 78)
    print("[5] Pooled regression: growth ~ coverage, family = {blob, polygonal, smooth}")
    print("=" * 78)
    all_cover, all_growth, all_is_poly, all_is_smooth = [], [], [], []
    for (family, _key), (g, m) in results.items():
        for gi in g:
            all_cover.append(m["cover"]); all_growth.append(gi)
            all_is_poly.append(1.0 if family == "polygonal" else 0.0)
            all_is_smooth.append(1.0 if family == "smooth" else 0.0)
    all_cover = np.array(all_cover); all_growth = np.array(all_growth)
    all_is_poly = np.array(all_is_poly); all_is_smooth = np.array(all_is_smooth)

    X_cov = np.column_stack([np.ones_like(all_cover), all_cover])
    r2_cov, _ = ols_r2(all_growth, X_cov)

    X_full = np.column_stack([np.ones_like(all_cover), all_cover, all_is_poly, all_is_smooth])
    r2_full, beta_full = ols_r2(all_growth, X_full)

    print("  n points = {}".format(len(all_cover)))
    print("  R^2 (coverage only)                         = {:.4f}".format(r2_cov))
    print("  R^2 (coverage + polygonal-term + smooth-term) = {:.4f}".format(r2_full))
    print("  polygonal coefficient (vs blob, at fixed coverage) = {:+.1f}".format(beta_full[2]))
    print("  smooth    coefficient (vs blob, at fixed coverage) = {:+.1f}".format(beta_full[3]))
    diff = beta_full[3] - beta_full[2]
    print("  smooth - polygonal (at fixed coverage)              = {:+.1f}".format(diff))
    print()
    print("  If polygonal and smooth coefficients are close (small |diff|), the fragility")
    print("  effect is about FILAMENT/CONNECTED TOPOLOGY in general, not shape smoothness --")
    print("  the 'gel' fix was a legitimate correction but not the active ingredient.")
    print("  If they differ a lot, smoothness itself is doing real work.")

    print("\n" + "=" * 78)
    print("[6] Does the polygonal family ALSO collapse to negative growth at low coverage?")
    print("=" * 78)
    low_cov_labels = [l for l, a, b in POLY_CONFIGS if a == 0.0]
    for label in low_cov_labels:
        gp, mp = results[("polygonal", label)]
        sign = "NET LOSS" if gp.mean() < 0 else "net gain"
        print("  polygonal:{:<10} cover={:.3f}  growth={:+.1f}+/-{:.1f}  -> {}".format(
            label, mp["cover"], gp.mean(), gp.std(), sign))
