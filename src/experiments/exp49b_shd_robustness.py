"""Exp 49b: is Exp 49's real-data partial-knowledge curve a genuine trend,
or an artifact of using a single clustering/placement draw per knowledge
level?

MOTIVATION. Exp 49 found the partial-knowledge cliff did NOT replicate on
real SHD-derived association weights -- knowledge_frac=25% appeared to
recover 135% of the population-to-functional gap (i.e. it beat full
knowledge outright), with variances roughly half the mean at every point.
That is a red flag, not a clean result: SpiNeCluster (multi-restart
Kernighan-Lin) and SpiNePlacer (particle-swarm) are both randomized
heuristics, and Exp 49 used exactly ONE clustering/placement seed per
knowledge level. A single placement draw can land in a good or bad local
optimum for reasons unrelated to the knowledge level itself, especially
with real, non-uniform channel weights making the optimization landscape
rougher than the synthetic graph's uniform one. This experiment re-runs
the same knowledge levels with THREE independent placement seeds each, to
see whether the 25% anomaly (and the concave-recovery conclusion built on
it) survives, or was a fluke of one lucky clustering draw.

DESIGN. Same real target/distractor channel-activity graph as Exp 49
(loaded once, held fixed -- only the placement algorithm's own internal
randomness varies). For each knowledge_frac, 3 independent placement
seeds x 4 learning-dynamics seeds = 12 runs (vs Exp 49's 1 placement seed
x 8 dynamics seeds = 8 runs), trading dynamics-seed depth for placement-
seed breadth, which is what this question actually needs. Reports both
the pooled curve (all 12 runs per knowledge_frac as one sample) and a
per-placement-seed breakdown, so a placement-specific outlier is visible
rather than averaged away.

DECISION RULE, registered before running:
  - If each individual placement seed's mean is reasonably close to the
    others at a given knowledge_frac (no single seed dominating the
    pooled mean), the pooled curve is trustworthy as a real trend.
  - If knowledge_frac=25%'s outlier behavior (beating 100%) is driven by
    ONE placement seed while the other two look unremarkable, Exp 49's
    "cliff does not replicate" conclusion does not survive and should be
    withdrawn in favor of "inconclusive, dominated by placement-search
    variance" pending a design that controls for it (e.g. averaging over
    many more placement seeds, or a placement algorithm whose output
    quality is less draw-dependent).

Run: python src/experiments/exp49b_shd_robustness.py
"""
import numpy as np
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats

from exp49_shd_partial_knowledge import (
    load_two_class_samples, channel_activity, build_real_graph, make_placement_real,
    bridge_of, SAMPLES_PER_CLASS, POOL_SIZE, KNOWLEDGE_FRACS,
)

PLACEMENT_SEEDS = [0, 1, 2]
DYN_SEEDS = list(range(4))

if __name__ == "__main__":
    print("=" * 78)
    print("EXP 49b: placement-seed robustness of the real-data partial-knowledge curve")
    print("=" * 78)

    print("\n[0] Loading real SHD samples (same as Exp 49)")
    t0 = time.time()
    top, avail, target_samples, distractor_samples = load_two_class_samples(SAMPLES_PER_CLASS, POOL_SIZE)
    print("  classes: target={}, distractor={}  (loaded in {:.0f}s)".format(top[0], top[1], time.time() - t0))
    act_t = channel_activity(target_samples)
    act_d = channel_activity(distractor_samples)

    print("\n[1] Growth per (knowledge_frac, placement_seed), {} dynamics seeds each".format(len(DYN_SEEDS)))
    per_placement = {}   # (kf, pseed) -> array of dynamics-seed bridge values
    for kf in KNOWLEDGE_FRACS:
        W = build_real_graph(act_t, act_d, kf)
        for pseed in PLACEMENT_SEEDS:
            coords, cids = make_placement_real(W, seed=pseed)
            g = bridge_of(coords, cids, target_samples, distractor_samples, seeds=DYN_SEEDS)
            per_placement[(kf, pseed)] = g
            print("  kf={:>5.0%}  placement_seed={}  bridge={:+8.1f} +/- {:<6.1f}".format(
                kf, pseed, g.mean(), g.std()))

    print("\n" + "=" * 78)
    print("[2] Per-knowledge_frac: pooled mean vs individual placement-seed means")
    print("=" * 78)
    pooled = {}
    for kf in KNOWLEDGE_FRACS:
        seed_means = [per_placement[(kf, ps)].mean() for ps in PLACEMENT_SEEDS]
        all_vals = np.concatenate([per_placement[(kf, ps)] for ps in PLACEMENT_SEEDS])
        pooled[kf] = all_vals
        spread = max(seed_means) - min(seed_means)
        rel_spread = spread / max(abs(np.mean(seed_means)), 1e-9)
        print("  kf={:>5.0%}  pooled mean={:+8.1f}   per-seed means={}   spread={:.0f} ({:.0%} of mean)".format(
            kf, all_vals.mean(), ["{:+.0f}".format(m) for m in seed_means], spread, rel_spread))

    print("\n" + "=" * 78)
    print("[3] Recovery curve on the POOLED (placement-seed-averaged) data")
    print("=" * 78)
    pop_g = pooled[0.0].mean()
    fn_g = pooled[1.0].mean()
    span = fn_g - pop_g
    print("  gap: population {:+.1f} -> functional {:+.1f}  (span {:.1f})".format(pop_g, fn_g, span))
    frac_at_50 = None
    for kf in KNOWLEDGE_FRACS:
        g = pooled[kf]
        pct = (g.mean() - pop_g) / span if span != 0 else float("nan")
        t, p = stats.ttest_ind(g, pooled[0.0], equal_var=False)
        print("  {:>13.0%} {:>+10.1f} {:>11.1%} {:>14.2e}".format(kf, g.mean(), pct, p))
        if frac_at_50 is None and pct >= 0.5:
            frac_at_50 = kf

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    kf_check = 0.25
    seed_means_25 = [per_placement[(kf_check, ps)].mean() for ps in PLACEMENT_SEEDS]
    m25 = np.mean(seed_means_25)
    spread25 = max(seed_means_25) - min(seed_means_25)
    dominated_by_one = spread25 > 0.5 * abs(m25) if m25 != 0 else True
    print("  knowledge_frac=25% per-seed means: {}".format(["{:+.0f}".format(x) for x in seed_means_25]))
    if dominated_by_one:
        print("  The 25% anomaly IS placement-seed-dependent (spread {:.0f} vs mean {:.0f}) --".format(spread25, m25))
        print("  Exp 49's 'cliff does not replicate' conclusion does NOT survive. Verdict:")
        print("  INCONCLUSIVE, dominated by placement-search variance, not a real knowledge")
        print("  effect. A design averaging over many more placement seeds (or a less")
        print("  draw-dependent partitioner) would be needed before trusting a shape here.")
    else:
        print("  The 25% result is consistent across placement seeds (spread {:.0f} vs mean {:.0f}).".format(spread25, m25))
        print("  Exp 49's conclusion survives this check: real, graded channel activity")
        print("  genuinely gives partial knowledge more traction than the uniform synthetic")
        print("  graph did, and this is not a single-seed artifact.")
