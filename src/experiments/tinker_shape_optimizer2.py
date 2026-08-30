"""Shape optimizer v2: fixes the v1 flaw (it let coverage vary freely and
just rediscovered "more coverage is better"). This version holds coverage
FIXED at blob clumped30's scarce-regime value (~0.417 -- the coverage
level where Exp 45c found the actual topology effect; at high coverage
shape doesn't matter at all, already established) and searches the
remaining shape parameters (K, sigma, density_mult) for the
growth-maximizing configuration AT THAT FIXED COVERAGE. For each sampled
(K, sigma, density_mult) triple, n_matched_frac is locally calibrated
(fast, mediator-only, no simulation) to land coverage as close to the
0.417 target as achievable, before growth is ever evaluated.

Run: python src/experiments/tinker_shape_optimizer2.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from tinker_fragmentation3 import make_fragmented_clumped
from exp43_coverage_confirmatory import mediators, make_condition
from exp32b_benchmark import run_life, N

FAST_SEEDS = [0, 1, 2]
REFINE_SEEDS = list(range(8))
N_SHAPES = 18
THRESHOLD = 0.6
TARGET_COVER = 0.417
COVER_TOL = 0.06  # accept 0.357-0.477; report the achieved miss for transparency

SHAPE_RANGES = {
    "K": (1, 50),
    "density_mult": (1.0, 15.0),
    "sigma": (1.0, 4.0),
}


def sample_shape(rng, i, n_total):
    cfg = {}
    for name, (lo, hi) in SHAPE_RANGES.items():
        stratum = rng.permutation(n_total)[i]
        frac = (stratum + rng.uniform(0, 1)) / n_total
        val = lo + frac * (hi - lo)
        cfg[name] = int(round(val)) if name == "K" else float(val)
    cfg["K"] = max(1, cfg["K"])
    return cfg


def calibrate_coverage(cfg, seed=0):
    """Fast, mediator-only search over n_matched_frac to hit TARGET_COVER."""
    K = cfg["K"]
    best = None
    grid = np.linspace(0.0, 1.0, 11)
    for frac in grid:
        n_matched = int(round(frac * K))
        try:
            coords, cids = make_fragmented_clumped(
                K, sigma=cfg["sigma"], thr=THRESHOLD,
                density_mult=cfg["density_mult"], n_matched=n_matched, seed=seed)
        except Exception:
            continue
        m = mediators(coords, cids)
        d = abs(m["cover"] - TARGET_COVER)
        if best is None or d < best[0]:
            best = (d, n_matched, m, coords, cids)
    return best  # (dist, n_matched, mediators, coords, cids) or None


def growth_of(coords, cids, seeds):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


if __name__ == "__main__":
    print("=" * 100)
    print("SHAPE OPTIMIZER v2: coverage held near {:.3f} (blob clumped30's value), searching K/density/sigma".format(TARGET_COVER))
    print("=" * 100)

    rng = np.random.default_rng(7742)
    candidates = []
    for i in range(N_SHAPES):
        shape_cfg = sample_shape(rng, i, N_SHAPES)
        cal = calibrate_coverage(shape_cfg)
        if cal is None:
            continue
        dist, n_matched, m, coords, cids = cal
        if dist > COVER_TOL:
            print("  K={:<3} dmult={:<5.1f} sigma={:.2f}  best achievable cover={:.3f} (miss={:.3f}) -- SKIPPED (out of tolerance)".format(
                shape_cfg["K"], shape_cfg["density_mult"], shape_cfg["sigma"], m["cover"], dist))
            continue
        g = growth_of(coords, cids, FAST_SEEDS)
        candidates.append((shape_cfg, n_matched, m, g.mean()))
        print("  K={:<3} dmult={:<5.1f} sigma={:.2f}  n_matched={:<3} count={:<6.2f} cover={:<6.3f}  growth(fast)={:+7.1f}".format(
            shape_cfg["K"], shape_cfg["density_mult"], shape_cfg["sigma"], n_matched,
            m["count"], m["cover"], g.mean()))
        sys.stdout.flush()

    if not candidates:
        print("\nNo candidates landed within coverage tolerance -- widen COVER_TOL or search range.")
        sys.exit(1)

    candidates.sort(key=lambda x: -x[3])
    print("\n" + "=" * 100)
    print("TOP 5 (at matched coverage ~{:.3f}) -- refining with {} seeds each".format(TARGET_COVER, len(REFINE_SEEDS)))
    print("=" * 100)
    refined = []
    for shape_cfg, n_matched, m_fast, _ in candidates[:5]:
        coords, cids = make_fragmented_clumped(
            shape_cfg["K"], sigma=shape_cfg["sigma"], thr=THRESHOLD,
            density_mult=shape_cfg["density_mult"], n_matched=n_matched, seed=0)
        m = mediators(coords, cids)
        g = growth_of(coords, cids, REFINE_SEEDS)
        refined.append((shape_cfg, n_matched, m, g))
        print("  K={:<3} dmult={:<5.1f} sigma={:.2f}  n_matched={:<3} count={:<6.2f} cover={:<6.3f}  growth={:+7.1f} +/- {:<6.1f}".format(
            shape_cfg["K"], shape_cfg["density_mult"], shape_cfg["sigma"], n_matched,
            m["count"], m["cover"], g.mean(), g.std()))
        sys.stdout.flush()

    refined.sort(key=lambda x: -x[3].mean())
    best_cfg, best_nm, best_m, best_g = refined[0]

    print("\n" + "=" * 100)
    print("BEST SEARCHED SHAPE vs blob clumped30, at matched coverage")
    print("=" * 100)
    print("  BEST SEARCHED   K={} dmult={:.1f} sigma={:.2f} n_matched={}  count={:.2f} cover={:.3f}  growth={:+.1f} +/- {:.1f}".format(
        best_cfg["K"], best_cfg["density_mult"], best_cfg["sigma"], best_nm,
        best_m["count"], best_m["cover"], best_g.mean(), best_g.std()))
    c, ci = make_condition("clumped30", seed=0)
    m_b = mediators(c, ci)
    g_b = growth_of(c, ci, REFINE_SEEDS)
    print("  blob clumped30                                         count={:.2f} cover={:.3f}  growth={:+.1f} +/- {:.1f}".format(
        m_b["count"], m_b["cover"], g_b.mean(), g_b.std()))
    from scipy import stats
    t, p = stats.ttest_ind(best_g, g_b, equal_var=False)
    print("\n  Welch t={:.2f}, p={:.3e}  (best searched vs blob, at matched coverage)".format(t, p))
    if best_g.mean() > g_b.mean() and p < 0.05:
        print("  The searched shape BEATS blob at matched coverage -- a genuinely new, better design found.")
    elif p >= 0.05:
        print("  Statistically indistinguishable from blob at matched coverage -- blob remains the best known shape,")
        print("  not beaten, but matched by at least one alternative design.")
    else:
        print("  Blob remains superior -- the search did not find anything better at this coverage level.")
