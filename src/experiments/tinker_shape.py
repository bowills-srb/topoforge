"""Fast, low-rigor shape tinkering. NOT a formal experiment -- 3 seeds
instead of 8, no pre-registration, just quick iteration to find which
knobs move growth before committing to a rigorous sweep of the promising
ones. Reuses exp45's density-field generator (sigma=tube thickness/
redundancy, threshold=void cutoff, n_voronoi_seeds=mesh complexity,
alpha/bias_strength=type-segregation intensity -> coverage).

Run: python src/experiments/tinker_shape.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from exp45_cosmic_web_topology import make_filament_density_biased
from exp43_coverage_confirmatory import mediators, make_condition
from exp32b_benchmark import run_life

QUICK_SEEDS = [0, 1, 2]


def growth_quick(coords, cids, seeds=QUICK_SEEDS):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


def probe(label, **kwargs):
    coords, cids = make_filament_density_biased(seed=0, **kwargs)
    m = mediators(coords, cids)
    g = growth_quick(coords, cids)
    print("  {:<28} sigma={:<5.2f} thr={:<5.2f} nvs={:<3} alpha={:<5.2f} bias={:<5.2f}  "
          "count={:<6.2f} cover={:<6.3f}  growth={:+7.1f} +/- {:<5.1f}".format(
              label, kwargs.get("sigma", 1.6), kwargs.get("threshold", 0.35),
              kwargs.get("n_voronoi_seeds", 14), kwargs.get("alpha", 1.0),
              kwargs.get("bias_strength", 0.85), m["count"], m["cover"], g.mean(), g.std()))
    return m, g


if __name__ == "__main__":
    print("=" * 100)
    print("TINKER: reference points")
    print("=" * 100)
    for name in ("spread15", "dilute30", "clumped30"):
        coords, cids = make_condition(name, seed=0)
        m = mediators(coords, cids)
        g = growth_quick(coords, cids)
        print("  blob:{:<20} count={:<6.2f} cover={:<6.3f}  growth={:+7.1f} +/- {:<5.1f}".format(
            name, m["count"], m["cover"], g.mean(), g.std()))

    print("\n" + "=" * 100)
    print("TINKER: sigma (tube thickness / local redundancy), alpha=0 bias=0.99 held fixed")
    print("=" * 100)
    for sigma in [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.6]:
        probe("sigma={:.1f}".format(sigma), sigma=sigma, threshold=0.35, alpha=0.0, bias_strength=0.99)

    print("\n" + "=" * 100)
    print("TINKER: threshold (void cutoff), sigma=1.6 alpha=0 bias=0.99 held fixed")
    print("=" * 100)
    for thr in [0.15, 0.25, 0.35, 0.5, 0.7, 0.9]:
        probe("thr={:.2f}".format(thr), sigma=1.6, threshold=thr, alpha=0.0, bias_strength=0.99)

    print("\n" + "=" * 100)
    print("TINKER: n_voronoi_seeds (mesh complexity / branching), sigma=1.6 thr=0.35 alpha=0 bias=0.99")
    print("=" * 100)
    for nvs in [6, 10, 14, 20, 28, 40]:
        probe("nvs={}".format(nvs), sigma=1.6, threshold=0.35, n_voronoi_seeds=nvs, alpha=0.0, bias_strength=0.99)
