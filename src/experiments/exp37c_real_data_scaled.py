"""Exp 37c: Strengthen the Exp 37b real-data placement result.

Motivation (preprint Section 6, "Real-data result variance and scope"):
the 9.31x interleaved/segregated ratio rested on TWO examples per class
x 4 samples, 5 seeds, with +/-42% relative variance in the segregated
condition. That makes 9.31x an order-of-magnitude claim, not a tight
estimate. This experiment tightens it by:
  1. Using many more DISTINCT real SHD samples per class (SAMPLES_PER_CLASS).
  2. Running more seeds (SEEDS) to shrink the CI on the ratio.
  3. Reporting mean, std, a paired significance test, a nonparametric
     check, Cohen's d, and a bootstrap CI on the ratio -- not just a ratio.

WHAT IS KEPT EXACTLY (so the validated physics is not disturbed):
  - The adjacency-matched geometry (make_placement_segregated /
    make_placement_interleaved) and the run_life() loop are IMPORTED
    verbatim from exp37b_v2_real_data.py. Nothing is re-implemented, so
    the PROJECT_HISTORY gotchas (unreachable rewire block, reward-timing,
    always-true triggers, decay/life mismatch) cannot be re-introduced by
    a copy-paste edit.
  - V decay stays at 0.99995 (set inside the imported run_life).

DECAY / LIFE-LENGTH DISCIPLINE (gotcha #2):
  V retention = 0.99995 ** life_steps, and life_steps grows with
  samples_per_class * n_epochs. The validated config (4 samples x 5
  epochs ~= 8,600 steps) retains ~0.65. To add distinct samples WITHOUT
  driving retention toward zero, we trade epochs down as samples go up so
  life stays in the validated band. Retention is printed and asserted to
  stay >= MIN_RETENTION so a bad config fails loudly instead of silently
  returning decayed-to-noise results.

FINDINGS (12 seeds each; totals are bridge mass, mean +/- sample SD):
  1. AUDIT (4 samples x 5 epochs, n=5): reproduces the committed exp37b
     BIT-IDENTICALLY -> 7.22x. NOTE: the committed code yields 7.22x, not
     the 9.31x in the commit message / preprint. Adjacency (1198 vs 2268)
     and V retention (0.651) match the commit exactly, so the geometry and
     decay are unchanged; the 9.31x was a different, luckier draw of the
     4 specific samples -- i.e. exactly the small-sample instability this
     experiment set out to quantify.
  2. MATCHED-LIFE, the headline (20 distinct samples x 1 epoch, SAME
     ~8,600-step life / 0.651 retention as the validated original):
       segregated 1161 +/- 822 (rel std 71%), interleaved 8423 +/- 1089
       ratio = 7.25x, bootstrap 95% CI [5.05x, 11.48x]
       paired t p = 4.3e-09, Welch p = 3.2e-14, Wilcoxon p = 4.9e-04,
       Cohen's d = 7.53
     => 5x more distinct real samples gives the SAME ratio as 4 repeated
        samples (7.25x vs 7.22x). The effect is NOT a 4-sample artifact.
  3. LONGER-LIFE (20 samples x 2 epochs, ~17,200 steps, retention 0.42):
     ratio compresses to 3.62x [2.70x, 5.42x] (still p = 1.3e-07, d = 4.9).
     The MAGNITUDE depends on the training-exposure / V-retention regime,
     not on sample diversity; both 3.6x and 7.3x bracket the synthetic
     ~4x PLB reference.
  4. The segregated condition is intrinsically high-variance (rel std
     60-70% across every multi-seed run); more seeds REVEAL this rather
     than shrink it. The wide CI reflects a real property, not noise.

Run (default = matched-life headline):
             python src/experiments/exp37c_real_data_scaled.py
Run (matched, explicit alias of default):
             python src/experiments/exp37c_real_data_scaled.py --matched
Run (longer-life, 2 epochs): ... --full
Run (smoke): ... --smoke
Run (audit, reproduce original 4-sample config): ... --audit
"""
import numpy as np
import time
import sys
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats
from shd_loader import load_shd_samples
# Import the validated physics verbatim -- do NOT re-implement.
from exp37b_v2_real_data import (
    make_placement_segregated, make_placement_interleaved, run_life,
    INPUT_TYPE, N,
)

# ---- Full-scale configuration ------------------------------------------
# samples_per_class * n_epochs is held near the validated ~20 sample-
# presentations/class so life length (and thus V retention) stays in the
# regime the 0.99995 decay was calibrated for.
SAMPLES_PER_CLASS = 20      # was 4 in exp37b -> 5x more distinct real samples
N_EPOCHS = 2                # was 5; 20*2 presentations vs 4*5, life ~2x validated
SEEDS = list(range(12))     # was 5 seeds -> 12 for a tighter CI
MIN_RETENTION = 0.25        # fail loudly if the config decays V too far
STEPS_PER_SAMPLE = 215      # ~200 stimulus steps + 15 rest, for the estimate

# run_life expects the input dimensionality the placement uses; N_INPUT in
# exp37b is 140, so we must request 140 channels from the loader.
INPUT_TYPE_CHANNELS = 140


def load_two_class_samples(samples_per_class, pool_size, max_steps=200):
    """Load a large SHD pool and return the first `samples_per_class`
    DISTINCT samples from each of the two most-common classes. Deterministic
    for a fixed pool_size (loader seed is fixed inside load_shd_samples)."""
    stimuli, labels = load_shd_samples(
        n_samples=pool_size, n_channels_out=INPUT_TYPE_CHANNELS,
        bin_ms=4.0, max_steps=max_steps)
    counts = Counter(labels)
    top = [c for c, _ in counts.most_common(2)]
    avail = {c: counts[c] for c in top}
    tgt = [s for s, l in zip(stimuli, labels) if l == top[0]][:samples_per_class]
    dis = [s for s, l in zip(stimuli, labels) if l == top[1]][:samples_per_class]
    return top, avail, tgt, dis


def cohens_d_pooled(a, b):
    """Standard (pooled-SD) Cohen's d between two independent samples,
    matching the effect-size style used elsewhere in the project."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) /
                 (na + nb - 2))
    if sp == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return (a.mean() - b.mean()) / sp


def bootstrap_ratio_ci(inter, seg, n_boot=10000, seed=12345):
    """Percentile bootstrap 95% CI on mean(inter)/mean(seg), resampling
    seeds with replacement. Crude at small n but standard; reported so the
    ratio comes with an interval rather than as a point estimate."""
    rng = np.random.default_rng(seed)
    inter, seg = np.asarray(inter, float), np.asarray(seg, float)
    n = len(inter)
    ratios = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        si = inter[idx].mean()
        ss = max(seg[idx].mean(), 1e-9)
        ratios.append(si / ss)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return lo, hi


def run_config(samples_per_class, n_epochs, seeds, pool_size, label):
    print("=" * 74)
    print("EXP 37c: {}".format(label))
    print("=" * 74)

    top, avail, target_samples, distractor_samples = load_two_class_samples(
        samples_per_class, pool_size)
    got = min(len(target_samples), len(distractor_samples))
    print("Classes: target={} (avail {}), distractor={} (avail {})".format(
        top[0], avail[top[0]], top[1], avail[top[1]]))
    print("Using {} samples/class x {} classes x {} epochs".format(
        len(target_samples), 2, n_epochs))
    if got < samples_per_class:
        print("  WARNING: only {} samples/class available (< {} requested); "
              "raise pool_size.".format(got, samples_per_class))

    est_steps = 2 * len(target_samples) * n_epochs * STEPS_PER_SAMPLE
    retention = 0.99995 ** est_steps
    print("Life length: ~{:,} steps  |  V retention at end: {:.3f} "
          "(validated ~0.65 at 8,600 steps)".format(est_steps, retention))
    assert retention >= MIN_RETENTION, (
        "V retention {:.4f} < {} -- config would decay early structure to "
        "noise (gotcha #2). Reduce samples_per_class*n_epochs.".format(
            retention, MIN_RETENTION))

    # Adjacency parity check (gotcha #4): both conditions must have
    # comparable input-output opportunity, else the ratio is rigged.
    from spatial import SpatialGrid
    strategies = [("segregated", make_placement_segregated),
                  ("interleaved", make_placement_interleaved)]
    print("\nAdjacency parity (input-output pairs within radius 6.0):")
    for name, fn in strategies:
        c, ci = fn()
        g = SpatialGrid(c, 6.0)
        inp = set(np.where(ci == INPUT_TYPE)[0].tolist())
        out = set(np.where(ci != INPUT_TYPE)[0].tolist())
        adj = sum(1 for i in inp for j in g.within(i, 6.0) if int(j) in out)
        print("  {:>12}: {:,} adjacent pairs".format(name, adj))

    results = {}
    for name, fn in strategies:
        print("\n  {}".format(name.upper()))
        seed_totals = []
        for s in seeds:
            t0 = time.time()
            coords, cids = fn()
            ba, bb = run_life(coords, cids, s, target_samples,
                              distractor_samples, n_epochs=n_epochs)
            seed_totals.append(ba + bb)
            print("    seed {:>2}: bridge_A={:>7.1f}  bridge_B={:>7.1f}  "
                  "total={:>7.1f}  ({:.0f}s)".format(s, ba, bb, ba + bb,
                                                     time.time() - t0))
        results[name] = np.array(seed_totals, dtype=float)
    return results


def report_stats(results):
    inter = results["interleaved"]
    seg = results["segregated"]

    print("\n" + "=" * 74)
    print("RESULTS on REAL SHD DATA  (n = {} seeds/condition)".format(len(inter)))
    print("-" * 74)
    for name, vals in (("segregated", seg), ("interleaved", inter)):
        rel = 100 * vals.std(ddof=1) / max(vals.mean(), 1e-9)
        print("  {:>12}: total = {:>8.1f} +/- {:>7.1f}  (rel std {:>5.1f}%)".format(
            name, vals.mean(), vals.std(ddof=1), rel))

    ratio = inter.mean() / max(seg.mean(), 1e-9)
    lo, hi = bootstrap_ratio_ci(inter, seg)

    # Paired test (same seed drives input noise in both conditions).
    t_rel, p_rel = stats.ttest_rel(inter, seg)
    # Unpaired Welch as a robustness check (does not assume pairing).
    t_ind, p_ind = stats.ttest_ind(inter, seg, equal_var=False)
    # Nonparametric paired check -- robust to non-normality at small n.
    try:
        w_stat, p_w = stats.wilcoxon(inter, seg)
    except ValueError as e:
        w_stat, p_w = float("nan"), float("nan")
        print("  (Wilcoxon skipped: {})".format(e))
    d = cohens_d_pooled(inter, seg)

    print("-" * 74)
    print("  Interleaved / Segregated ratio : {:.2f}x".format(ratio))
    print("  Bootstrap 95% CI on ratio      : [{:.2f}x, {:.2f}x]".format(lo, hi))
    print("  Paired t-test                  : t = {:.3f}, p = {:.2e}".format(
        t_rel, p_rel))
    print("  Welch unpaired t-test          : t = {:.3f}, p = {:.2e}".format(
        t_ind, p_ind))
    print("  Wilcoxon signed-rank (paired)  : W = {:.1f}, p = {:.2e}".format(
        w_stat, p_w))
    print("  Cohen's d (pooled)             : {:.2f}".format(d))
    print("=" * 74)
    print("Reference: exp37b was 9.31x, 5 seeds, 4 samples/class, +/-42% rel std.")
    return ratio, (lo, hi), p_rel, d


if __name__ == "__main__":
    if "--audit" in sys.argv:
        # Reproduce the ORIGINAL exp37b config through the imported run_life
        # to confirm the physics is unchanged before trusting new numbers.
        res = run_config(samples_per_class=4, n_epochs=5,
                         seeds=[0, 1, 2, 3, 4], pool_size=200,
                         label="AUDIT -- reproduce original 4-sample config")
        report_stats(res)
    elif "--matched" in sys.argv:
        # Disentangle sample-diversity from life-length. 20 samples/class x
        # 1 epoch = ~8,600 steps = the EXACT validated life/retention regime
        # (0.651) of the original 4-sample x 5-epoch config, but with 20
        # distinct samples (each seen once) instead of 4 repeated 5x. Any
        # ratio change vs the original is then attributable to sample
        # diversity, not to a longer life or lower V retention.
        res = run_config(samples_per_class=20, n_epochs=1,
                         seeds=list(range(12)), pool_size=1500,
                         label="MATCHED-LIFE -- 20 samples/class, 1 epoch, 12 seeds")
        report_stats(res)
    elif "--full" in sys.argv:
        # Longer-life variant: 2 epochs doubles the life (~17,200 steps,
        # retention 0.42). Ratio compresses to ~3.6x -- reported to show the
        # magnitude's dependence on the exposure/decay regime, NOT as the
        # headline (its life sits outside the validated 0.65-retention band).
        res = run_config(samples_per_class=SAMPLES_PER_CLASS,
                         n_epochs=N_EPOCHS, seeds=SEEDS, pool_size=1500,
                         label="LONGER-LIFE -- {} samples/class, {} epochs, {} seeds".format(
                             SAMPLES_PER_CLASS, N_EPOCHS, len(SEEDS)))
        report_stats(res)
    elif "--smoke" in sys.argv:
        # Tiny, fast: confirm nothing is broken and the effect direction
        # holds before committing to the full run.
        res = run_config(samples_per_class=6, n_epochs=2,
                         seeds=[0, 1, 2], pool_size=400,
                         label="SMOKE TEST -- 6 samples/class, 2 epochs, 3 seeds")
        report_stats(res)
    else:
        # DEFAULT = matched-life headline: 20 distinct samples/class x 1
        # epoch = the validated ~8,600-step / 0.651-retention regime, with
        # 5x more distinct real samples than the original. Ratio 7.25x.
        res = run_config(samples_per_class=20, n_epochs=1,
                         seeds=list(range(12)), pool_size=1500,
                         label="MATCHED-LIFE (default) -- 20 samples/class, 1 epoch, 12 seeds")
        report_stats(res)
