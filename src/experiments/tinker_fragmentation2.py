"""Fast tinkering, v2: fixes a real bug in tinker_fragmentation.py -- that
version held each component's BBOX SIZE fixed while shrinking the number
of points per component as K grew, which silently increases density (and
therefore coverage) along with fragmentation, confounding the two. This
version holds POINT DENSITY fixed (component area scales with per-component
N, calibrated against the K=1 reference point that gave cover=0.427 in the
original single-network sweep) so K varies with density and coverage held
approximately constant, isolating fragmentation as intended.

Run: python src/experiments/tinker_fragmentation2.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from exp45_cosmic_web_topology import make_filament_density_biased
from exp43_coverage_confirmatory import mediators, make_condition
from exp32b_benchmark import run_life, N, NC

QUICK_SEEDS = [0, 1, 2]
TARGET_DENSITY = 900.0 / (88.0 * 44.0)  # calibrated from the K=1 reference (sigma=2.2, thr=0.25, cover=0.427)
MARGIN = 20.0  # component-to-component gap beyond each component's own side length, >> deposit radius 5.0


def component_layout(k, ncols, side):
    pitch = side + MARGIN
    cx = (k % ncols) * pitch
    cy = (k // ncols) * pitch
    return (cx, cx + side, cy, cy + side)


def make_fragmented(n_components, sigma, threshold, alpha, bias_strength, seed=0):
    ncols = int(np.ceil(np.sqrt(n_components)))
    per_comp = N // n_components
    side = np.sqrt(per_comp / TARGET_DENSITY)
    coords_list, cids_list = [], []
    got = 0
    for k in range(n_components):
        this_n = per_comp if k < n_components - 1 else (N - got)
        bbox = component_layout(k, ncols, side)
        c, ci = make_filament_density_biased(
            seed=seed * 97 + k, sigma=sigma, threshold=threshold,
            alpha=alpha, bias_strength=bias_strength, bbox=bbox,
            n_voronoi_seeds=max(8, per_comp // 15))
        coords_list.append(c[:this_n])
        cids_list.append(ci[:this_n])
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
    print("TINKER v2: fragmentation at held-fixed DENSITY (target={:.4f} pts/unit^2)".format(TARGET_DENSITY))
    print("sigma=2.2 threshold=0.25 alpha=0 bias=0.99")
    print("=" * 100)
    for k in [1, 2, 3, 5, 10, 20, 30]:
        coords, cids = make_fragmented(k, sigma=2.2, threshold=0.25, alpha=0.0, bias_strength=0.99, seed=0)
        m = mediators(coords, cids)
        g = growth_quick(coords, cids)
        print("  K={:<4} (n={:>3}/comp)  count={:<6.2f} cover={:<6.3f}  growth={:+8.1f} +/- {:<6.1f}".format(
            k, N // k, m["count"], m["cover"], g.mean(), g.std()))
        sys.stdout.flush()

    coords, cids = make_condition("clumped30", seed=0)
    m = mediators(coords, cids)
    g = growth_quick(coords, cids)
    print("  [blob clumped30, K=30 isolated cliques]  count={:<6.2f} cover={:<6.3f}  growth={:+8.1f} +/- {:<6.1f}".format(
        m["count"], m["cover"], g.mean(), g.std()))
