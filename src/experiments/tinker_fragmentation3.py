"""Fast tinkering, v3: does the clumped-style (discrete matched/mismatched
component) trick, which decouples count from coverage far better than
continuous alpha blending did, let a K=30 fragmented filament network
finally reach blob's actual operating point (count~6.25, coverage~0.417)?
If it does, does growth then also close the gap to blob? NOT a formal
experiment -- reduced seeds for speed.

Run: python src/experiments/tinker_fragmentation3.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from exp45_cosmic_web_topology import _voronoi_segments, _field
from exp43_coverage_confirmatory import mediators, make_condition
from exp32b_benchmark import run_life, N

QUICK_SEEDS = [0, 1, 2, 3, 4]
TARGET_DENSITY_BASE = 900.0 / (88.0 * 44.0)
MARGIN = 20.0


def layout(k, ncols, side):
    pitch = side + MARGIN
    cx = (k % ncols) * pitch
    cy = (k // ncols) * pitch
    return (cx, cx + side, cy, cy + side)


def make_fragmented_clumped(nk, sigma, thr, density_mult, n_matched, seed=0):
    ncols = int(np.ceil(np.sqrt(nk)))
    per = N // nk
    side = np.sqrt(per / (TARGET_DENSITY_BASE * density_mult))
    nvs = max(10, per // 10)
    coords_list, cids_list = [], []
    got = 0
    rng = np.random.default_rng(seed)
    for k in range(nk):
        this_n = per if k < nk - 1 else (N - got)
        bbox = layout(k, ncols, side)
        rngk = np.random.default_rng(seed * 97 + k)
        x0, x1, y0, y1 = bbox
        seed_pts = rngk.uniform([x0, y0], [x1, y1], size=(nvs, 2))
        seg_a, seg_b = _voronoi_segments(seed_pts)
        accepted = []
        for _ in range(200):
            cand = rngk.uniform([x0, y0], [x1, y1], size=(2000, 2))
            f = _field(cand, seg_a, seg_b, sigma)
            keep = cand[f > thr]
            accepted.append(keep)
            if sum(len(a) for a in accepted) >= this_n:
                break
        c = np.concatenate(accepted, axis=0)[:this_n]
        if len(c) < this_n:
            c = np.tile(c, (this_n // max(len(c), 1) + 1, 1))[:this_n]
        types = ([0, 3] if k % 2 == 0 else [1, 4]) if k < n_matched else ([0, 4] if k % 2 == 0 else [3, 1])
        ci = np.array([types[i % 2] for i in range(this_n)])
        rng.shuffle(ci)
        coords_list.append(c)
        cids_list.append(ci)
        got += this_n
    return np.concatenate(coords_list), np.concatenate(cids_list)


def growth_quick(coords, cids, seeds=QUICK_SEEDS):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


if __name__ == "__main__":
    print("=" * 100)
    print("TINKER v3: clumped-style K=30 fragmented filament vs blob, both count AND coverage matched")
    print("=" * 100)
    coords, cids = make_fragmented_clumped(30, sigma=2.2, thr=0.9, density_mult=3.0, n_matched=13)
    m = mediators(coords, cids)
    g = growth_quick(coords, cids)
    print("  fragmented-clumped K=30  count={:.2f} cover={:.3f}  growth={:+.1f} +/- {:.1f}".format(
        m["count"], m["cover"], g.mean(), g.std()))

    coords_b, cids_b = make_condition("clumped30", seed=0)
    m_b = mediators(coords_b, cids_b)
    g_b = growth_quick(coords_b, cids_b)
    print("  blob clumped30           count={:.2f} cover={:.3f}  growth={:+.1f} +/- {:.1f}".format(
        m_b["count"], m_b["cover"], g_b.mean(), g_b.std()))

    from scipy import stats
    t, p = stats.ttest_ind(g, g_b, equal_var=False)
    print("\n  Welch t={:.2f}, p={:.3e}, ratio(blob/fragmented)={:.2f}x".format(
        t, p, g_b.mean() / g.mean() if g.mean() != 0 else float("nan")))
