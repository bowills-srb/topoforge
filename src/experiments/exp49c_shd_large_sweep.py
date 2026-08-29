"""Exp 49c: the resolving version of Exp 49/49b -- 10 independent placement
seeds per knowledge level instead of 1 (Exp 49) or 3 (Exp 49b), to get a
real-data partial-knowledge curve with enough placement-seed averaging to
trust its shape.

MOTIVATION. Exp 49 (1 placement seed) found a 25%-knowledge point that
appeared to beat full knowledge outright. Exp 49b (3 placement seeds)
showed that anomaly was driven by a single lucky clustering/PSO draw and
declared the question INCONCLUSIVE -- 3 seeds is enough to catch a wild
outlier but not enough to trust a curve shape in a region with 80-200%
relative spread. This experiment scales placement-seed averaging up to 10
per knowledge level (60 independent placements total, 240 runs), which
should be enough to distinguish a real concave/convex/cliff shape from
placement-search noise.

DECISION RULE, registered before running:
  - Compute the standard error of the pooled mean at each knowledge_frac
    (using the 10 placement-seed means as the unit of replication, not
    the 40 raw dynamics-seed samples, since placement seed is the
    dominant noise source per Exp 49b). If adjacent knowledge_frac points
    are separated by less than 2 standard errors, they are NOT
    distinguishable and the curve between them should be reported as
    flat, not interpreted.
  - Report where 50% and 90% recovery are first reached with this
    tightened uncertainty, or state that they are not resolvable within
    this sweep's precision if the curve is too noisy even at n=10.

Run: python src/experiments/exp49c_shd_large_sweep.py
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

PLACEMENT_SEEDS = list(range(10))
DYN_SEEDS = list(range(4))

if __name__ == "__main__":
    print("=" * 78)
    print("EXP 49c: large placement-seed sweep (10 seeds x {} knowledge levels)".format(len(KNOWLEDGE_FRACS)))
    print("=" * 78)

    print("\n[0] Loading real SHD samples (same as Exp 49/49b)")
    t0 = time.time()
    top, avail, target_samples, distractor_samples = load_two_class_samples(SAMPLES_PER_CLASS, POOL_SIZE)
    print("  classes: target={}, distractor={}  (loaded in {:.0f}s)".format(top[0], top[1], time.time() - t0))
    act_t = channel_activity(target_samples)
    act_d = channel_activity(distractor_samples)

    print("\n[1] Growth per (knowledge_frac, placement_seed), {} dynamics seeds each -- {} total runs".format(
        len(DYN_SEEDS), len(KNOWLEDGE_FRACS) * len(PLACEMENT_SEEDS) * len(DYN_SEEDS)))
    per_placement = {}
    seed_means_by_kf = {}
    for kf in KNOWLEDGE_FRACS:
        W = build_real_graph(act_t, act_d, kf)
        seed_means = []
        t0 = time.time()
        for pseed in PLACEMENT_SEEDS:
            coords, cids = make_placement_real(W, seed=pseed)
            g = bridge_of(coords, cids, target_samples, distractor_samples, seeds=DYN_SEEDS)
            per_placement[(kf, pseed)] = g
            seed_means.append(g.mean())
        seed_means_by_kf[kf] = np.array(seed_means)
        print("  kf={:>5.0%}  placement-seed means: mean={:+8.1f}  sd={:>7.1f}  se={:>6.1f}  ({:.0f}s)".format(
            kf, seed_means_by_kf[kf].mean(), seed_means_by_kf[kf].std(ddof=1),
            seed_means_by_kf[kf].std(ddof=1) / np.sqrt(len(PLACEMENT_SEEDS)), time.time() - t0))
        sys.stdout.flush()

    print("\n" + "=" * 78)
    print("[2] Recovery curve, unit of replication = placement seed (n={})".format(len(PLACEMENT_SEEDS)))
    print("=" * 78)
    pop_means = seed_means_by_kf[0.0]
    fn_means = seed_means_by_kf[1.0]
    pop_g, fn_g = pop_means.mean(), fn_means.mean()
    span = fn_g - pop_g
    print("  gap: population {:+.1f} -> functional {:+.1f}  (span {:.1f})".format(pop_g, fn_g, span))
    print("  {:>10} {:>10} {:>8} {:>12} {:>14}".format("knowledge", "mean", "se", "%recovered", "Welch p vs pop"))
    results_summary = []
    for kf in KNOWLEDGE_FRACS:
        m = seed_means_by_kf[kf]
        se = m.std(ddof=1) / np.sqrt(len(PLACEMENT_SEEDS))
        pct = (m.mean() - pop_g) / span if span != 0 else float("nan")
        t, p = stats.ttest_ind(m, pop_means, equal_var=False)
        results_summary.append((kf, m.mean(), se, pct, p))
        print("  {:>9.0%} {:>+10.1f} {:>8.1f} {:>11.1%} {:>14.2e}".format(kf, m.mean(), se, pct, p))

    print("\n" + "=" * 78)
    print("[3] Adjacent-point resolvability (2-SE rule, registered above)")
    print("=" * 78)
    resolvable_curve = []
    for i in range(1, len(results_summary)):
        kf0, m0, se0, _, _ = results_summary[i - 1]
        kf1, m1, se1, _, _ = results_summary[i]
        diff = m1 - m0
        combined_se = np.sqrt(se0 ** 2 + se1 ** 2)
        resolvable = abs(diff) > 2 * combined_se
        resolvable_curve.append(resolvable)
        print("  {:>5.0%} -> {:>5.0%}: diff={:+8.1f}  2*SE={:>7.1f}  {}".format(
            kf0, kf1, diff, 2 * combined_se, "RESOLVABLE increase" if resolvable else "NOT resolvable (flat)"))

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    frac_at_50 = next((kf for kf, m, se, pct, p in results_summary if pct >= 0.5), None)
    frac_at_90 = next((kf for kf, m, se, pct, p in results_summary if pct >= 0.9), None)
    n_resolvable = sum(resolvable_curve)
    print("  {} of {} adjacent steps are statistically resolvable at n={} placement seeds.".format(
        n_resolvable, len(resolvable_curve), len(PLACEMENT_SEEDS)))
    if frac_at_50 is not None:
        print("  50% recovery first reached at knowledge_frac={:.0%}.".format(frac_at_50))
    if frac_at_90 is not None:
        print("  90% recovery first reached at knowledge_frac={:.0%}.".format(frac_at_90))
    if n_resolvable <= 1:
        print("  Even at n=10 placement seeds, the curve is mostly NOT resolvable step-to-step.")
        print("  Read this as: population and functional are reliably different (endpoints),")
        print("  but the shape of the path between them is not resolvable with this harness'")
        print("  placement-search variance -- report the two endpoints as established and the")
        print("  intermediate shape as genuinely open, not as either a cliff or a smooth curve.")
    elif frac_at_50 is not None and frac_at_50 <= 0.5:
        print("  CONCAVE, and resolvable: real, graded channel activity gives partial")
        print("  knowledge real traction well before full certainty -- the Exp 48 cliff does")
        print("  NOT replicate on real data, with adequate placement-seed averaging this time.")
    else:
        print("  The curve is resolvable and CONFIRMS a late, cliff-like jump: most of the")
        print("  benefit still requires knowledge_frac>50%, replicating Exp 48's qualitative")
        print("  shape on real data with a properly-powered design.")
