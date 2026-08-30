"""A real search over shape-space, not more hand-picked probes. Reuses the
clumped-style fragmented-filament generator (tinker_fragmentation3.py),
which is the best mechanism found tonight for reaching blob-competitive
performance from a filament-family geometry, and searches its parameter
space for the growth-maximizing configuration.

SEARCH SPACE:
  K            number of disconnected components (1-60)
  n_matched_frac  fraction of components that get BOTH associated types
                (controls coverage; the "clumped30 trick")
  density_mult    local point density multiplier (controls count)
  sigma        filament tube thickness
  threshold    fixed at 0.6 (mid-range; today's sweeps showed no dominant
               independent effect once density/sigma are covered)

METHOD: random search (30 configs, LatinHypercube-style stratified sampling
per dimension) at reduced seed count (2) for a fast first pass, then the
top 5 candidates by mean growth are re-evaluated at 8 seeds for a reliable
final ranking. This is a real, if simple, systematic search -- honest
about being random search + refinement, not Bayesian optimization.

Run: python src/experiments/tinker_shape_optimizer.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from tinker_fragmentation3 import make_fragmented_clumped
from exp43_coverage_confirmatory import mediators, make_condition
from exp32b_benchmark import run_life, N

FAST_SEEDS = [0, 1]
REFINE_SEEDS = list(range(8))
N_RANDOM = 30
THRESHOLD = 0.6

PARAM_RANGES = {
    "K": (1, 60),                 # integer
    "n_matched_frac": (0.0, 1.0),
    "density_mult": (1.0, 15.0),
    "sigma": (1.0, 4.0),
}


def sample_config(rng, i, n_total):
    """Stratified (Latin-hypercube-ish) sample: divide each dimension's
    range into n_total strata, draw a random point within stratum i for
    each dimension independently (dimensions shuffled per-call so strata
    don't correlate across dimensions)."""
    cfg = {}
    for name, (lo, hi) in PARAM_RANGES.items():
        stratum = rng.permutation(n_total)[i]
        frac = (stratum + rng.uniform(0, 1)) / n_total
        val = lo + frac * (hi - lo)
        cfg[name] = int(round(val)) if name == "K" else float(val)
    cfg["K"] = max(1, cfg["K"])
    return cfg


def growth_of(coords, cids, seeds):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


def build_and_eval(cfg, seeds):
    n_matched = max(0, min(cfg["K"], int(round(cfg["n_matched_frac"] * cfg["K"]))))
    coords, cids = make_fragmented_clumped(
        cfg["K"], sigma=cfg["sigma"], thr=THRESHOLD,
        density_mult=cfg["density_mult"], n_matched=n_matched, seed=0)
    m = mediators(coords, cids)
    g = growth_of(coords, cids, seeds)
    return m, g, coords, cids


if __name__ == "__main__":
    print("=" * 100)
    print("SHAPE OPTIMIZER: random search ({} configs, {} fast seeds) + refine top 5 ({} seeds)".format(
        N_RANDOM, len(FAST_SEEDS), len(REFINE_SEEDS)))
    print("=" * 100)

    rng = np.random.default_rng(2026)
    candidates = []
    for i in range(N_RANDOM):
        cfg = sample_config(rng, i, N_RANDOM)
        try:
            m, g, _, _ = build_and_eval(cfg, FAST_SEEDS)
        except Exception as e:
            print("  config {} FAILED: {} -- skipping".format(cfg, e))
            continue
        candidates.append((cfg, m, g.mean()))
        print("  K={:<3} nmf={:.2f} dmult={:<5.1f} sigma={:.2f}  count={:<6.2f} cover={:<6.3f}  "
              "growth(fast)={:+7.1f}".format(
                  cfg["K"], cfg["n_matched_frac"], cfg["density_mult"], cfg["sigma"],
                  m["count"], m["cover"], g.mean()))
        sys.stdout.flush()

    candidates.sort(key=lambda x: -x[2])
    print("\n" + "=" * 100)
    print("TOP 5 from fast pass -- refining with {} seeds each".format(len(REFINE_SEEDS)))
    print("=" * 100)
    refined = []
    for cfg, m_fast, _ in candidates[:5]:
        m, g, coords, cids = build_and_eval(cfg, REFINE_SEEDS)
        refined.append((cfg, m, g))
        print("  K={:<3} nmf={:.2f} dmult={:<5.1f} sigma={:.2f}  count={:<6.2f} cover={:<6.3f}  "
              "growth={:+7.1f} +/- {:<6.1f}".format(
                  cfg["K"], cfg["n_matched_frac"], cfg["density_mult"], cfg["sigma"],
                  m["count"], m["cover"], g.mean(), g.std()))
        sys.stdout.flush()

    refined.sort(key=lambda x: -x[2].mean())
    best_cfg, best_m, best_g = refined[0]

    print("\n" + "=" * 100)
    print("BEST FOUND vs known reference points")
    print("=" * 100)
    print("  BEST SEARCHED  K={} nmf={:.2f} dmult={:.1f} sigma={:.2f}  count={:.2f} cover={:.3f}  "
          "growth={:+.1f} +/- {:.1f}".format(
              best_cfg["K"], best_cfg["n_matched_frac"], best_cfg["density_mult"], best_cfg["sigma"],
              best_m["count"], best_m["cover"], best_g.mean(), best_g.std()))
    for name in ("spread15", "dilute30", "clumped30"):
        c, ci = make_condition(name, seed=0)
        m = mediators(c, ci)
        g = growth_of(c, ci, REFINE_SEEDS)
        print("  blob:{:<10} count={:.2f} cover={:.3f}  growth={:+.1f} +/- {:.1f}".format(
            name, m["count"], m["cover"], g.mean(), g.std()))
