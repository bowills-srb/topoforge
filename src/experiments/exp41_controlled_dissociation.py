"""Exp 41: controlled dissociation -- is reach CAUSAL, or just correlated?

Exp 40 showed learning collapses onto one curve as a function of a placement-
only "reach" metric (mean input neurons within the plasticity radius of each
output neuron). But in that sweep reach co-varied with other geometric
properties, so the evidence was correlational. This experiment dissociates
reach from the most obvious confound -- whether output CLASSES are spatially
clustered or mixed -- and manipulates each independently.

Two orthogonal factors:
  REACH: the output population is a rigid blob translated a distance `offset`
    from the input blob. offset=0 -> fully overlapping (high reach);
    large offset -> separated (low reach). Rigid translation preserves
    input-input and output-output structure exactly; ONLY input<->output
    distances change.
  ARRANGEMENT: for a FIXED set of output positions, class labels are assigned
    either 'clustered' (contiguous angular sectors) or 'mixed' (shuffled).
    This is a pure RELABELING -- the geometry, and therefore reach, is
    byte-identical between the two; only which position belongs to which
    class changes.

Pre-registered predictions:
  (Q1) ARRANGEMENT is null at matched reach: clustered vs mixed learning is
       statistically indistinguishable at every offset (identical geometry).
  (Q2) REACH is causal: learning changes strongly and monotonically with
       offset (reach), at fixed arrangement.
  If both hold, reach -- not class arrangement -- is the operative variable,
  upgrading Exp 40's mediation from correlational to causal.

Life held at the validated ~8,600-step / 0.651-retention regime (K=2, 20
samples/class, 1 epoch).

  python src/experiments/exp41_controlled_dissociation.py --smoke
  python src/experiments/exp41_controlled_dissociation.py
"""
import numpy as np
import time
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats
from exp39_kclass_real_data import (
    N_INPUT, N_OUTPUT_PER_CLASS, INPUT_TYPE, DEPOSIT_RADIUS, V_DECAY,
    STEPS_PER_SAMPLE, MIN_RETENTION, run_life_kclass, load_k_class_samples,
)
from exp40_dose_response import reach_metrics

POP_R = 7.0          # radius of each population blob
CENTER = 20.0        # input blob centred here; output blob at (CENTER+offset, CENTER)


def make_dissociation_placement(K, offset, arrangement, pos_seed=11, label_seed=0):
    """Input blob at CENTER; output blob (rigid) at CENTER+offset. Output class
    labels assigned 'clustered' (angular sectors) or 'mixed' (shuffled) over
    the SAME positions -- so reach is identical across arrangement, and varies
    only with offset."""
    rng = np.random.default_rng(pos_seed)

    in_coords = np.empty((N_INPUT, 2))
    for i in range(N_INPUT):
        th = rng.uniform(0, 2 * np.pi); r = POP_R * np.sqrt(rng.uniform(0, 1))
        in_coords[i] = [CENTER + r * np.cos(th), CENTER + r * np.sin(th)]

    n_out = K * N_OUTPUT_PER_CLASS
    out_coords = np.empty((n_out, 2))
    ocx = CENTER + offset
    for i in range(n_out):
        th = rng.uniform(0, 2 * np.pi); r = POP_R * np.sqrt(rng.uniform(0, 1))
        out_coords[i] = [ocx + r * np.cos(th), CENTER + r * np.sin(th)]

    labels = np.empty(n_out, dtype=int)
    if arrangement == "clustered":
        ang = np.arctan2(out_coords[:, 1] - CENTER, out_coords[:, 0] - ocx)
        order = np.argsort(ang)
        per = n_out // K
        for k in range(K):
            labels[order[k * per:(k + 1) * per]] = k + 1
    elif arrangement == "mixed":
        labels = np.repeat(np.arange(K) + 1, N_OUTPUT_PER_CLASS)
        np.random.default_rng(label_seed).shuffle(labels)
    else:
        raise ValueError(arrangement)

    coords = np.vstack([in_coords, out_coords])
    cids = np.concatenate([np.full(N_INPUT, INPUT_TYPE), labels])
    return coords, cids


def run(K, samples_per_class, offsets, arrangements, seeds, pool_size):
    top, avail, class_samples = load_k_class_samples(K, samples_per_class, pool_size)
    est_steps = K * samples_per_class * STEPS_PER_SAMPLE
    retention = V_DECAY ** est_steps
    print("K={} classes {}, {} samples/class -> life ~{:,} steps, retention {:.3f}".format(
        K, top, samples_per_class, est_steps, retention))
    assert retention >= MIN_RETENTION

    # --- audit: reach identical across arrangement, varies with offset ---
    print("\n" + "-" * 74)
    print("REACH AUDIT (mean input neighbors per output neuron)")
    print("-" * 74)
    print("  {:>8} {:>16} {:>16}".format("offset", "clustered reach", "mixed reach"))
    reach_of = {}
    for off in offsets:
        rc = reach_metrics(*make_dissociation_placement(K, off, "clustered"))[0]
        rm = reach_metrics(*make_dissociation_placement(K, off, "mixed"))[0]
        reach_of[off] = rc
        flag = "" if abs(rc - rm) < 1e-9 else "  !! reach differs across arrangement"
        print("  {:>8.0f} {:>16.2f} {:>16.2f}{}".format(off, rc, rm, flag))
    print("  (clustered and mixed reach must be identical at each offset)")

    # --- learning runs ---
    results = {}
    for arr in arrangements:
        for off in offsets:
            coords, cids = make_dissociation_placement(K, off, arr)
            totals = []
            for s in seeds:
                br = run_life_kclass(coords, cids, K, s, class_samples, n_epochs=1)
                totals.append(br.sum())
            results[(arr, off)] = np.array(totals)
    return results, reach_of


def report(results, reach_of, offsets, arrangements):
    print("\n" + "=" * 74)
    print("LEARNING (total bridge mass, mean +/- std over seeds)")
    print("-" * 74)
    print("  {:>8} {:>8} {:>16} {:>16}".format("offset", "reach", "clustered", "mixed"))
    for off in offsets:
        c = results[("clustered", off)]; m = results[("mixed", off)]
        print("  {:>8.0f} {:>8.2f} {:>10.0f} +/-{:>4.0f} {:>10.0f} +/-{:>4.0f}".format(
            off, reach_of[off], c.mean(), c.std(ddof=1), m.mean(), m.std(ddof=1)))

    print("\n" + "=" * 74)
    print("Q1 -- ARRANGEMENT null at matched reach? (clustered vs mixed per offset)")
    print("-" * 74)
    for off in offsets:
        c = results[("clustered", off)]; m = results[("mixed", off)]
        t, p = stats.ttest_rel(c, m)
        verdict = "n.s. (arrangement doesn't matter)" if p > 0.05 else "SIGNIFICANT"
        print("  offset {:>2.0f} (reach {:>5.2f}): clustered {:>7.0f} vs mixed {:>7.0f}"
              "  paired p={:.3f}  -> {}".format(
                  off, reach_of[off], c.mean(), m.mean(), p, verdict))

    print("\n" + "=" * 74)
    print("Q2 -- REACH causal? (learning vs offset/reach, at fixed arrangement)")
    print("-" * 74)
    for arr in arrangements:
        means = [results[(arr, off)].mean() for off in offsets]
        reaches = [reach_of[off] for off in offsets]
        rho, p = stats.spearmanr(reaches, means)
        print("  {:>10}: learning across reach {} "
              "(hi->lo reach: {:.0f} -> {:.0f}, {:.1f}x)  Spearman rho={:.3f}".format(
                  arr, np.array2string(np.array(means), precision=0, floatmode="fixed"),
                  means[0], means[-1], means[0] / max(means[-1], 1e-9), rho))

    # pooled mediation: learning vs reach, both arrangements on one axis
    xr, yl = [], []
    for arr in arrangements:
        for off in offsets:
            xr.append(reach_of[off]); yl.append(results[(arr, off)].mean())
    pr, pp = stats.pearsonr(xr, yl)
    print("\n  Pooled (both arrangements) learning vs reach: "
          "Pearson r={:.3f} (R^2={:.3f})".format(pr, pr ** 2))
    print("\nCONCLUSION: if Q1 is n.s. at every offset and Q2 shows strong")
    print("reach-dependence, then reach is CAUSAL and class arrangement is not --")
    print("upgrading Exp 40's mediation from correlational to controlled.")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        print("=" * 74)
        print("SMOKE: reach identical across arrangement; varies with offset")
        print("=" * 74)
        K = 2
        for off in (0, 8, 16):
            rc, fc = reach_metrics(*make_dissociation_placement(K, off, "clustered"))
            rm, fm = reach_metrics(*make_dissociation_placement(K, off, "mixed"))
            print("  offset {:>2}: clustered reach {:6.2f}  mixed reach {:6.2f}  "
                  "(identical: {})".format(off, rc, rm, abs(rc - rm) < 1e-9))
        print("\n  quick learning (2 seeds), offset 0 (high reach) vs 16 (low):")
        top, avail, csamp = load_k_class_samples(2, 20, 1500)
        for off in (0, 16):
            for arr in ("clustered", "mixed"):
                coords, cids = make_dissociation_placement(K, off, arr)
                tot = [run_life_kclass(coords, cids, K, s, csamp, n_epochs=1).sum()
                       for s in (0, 1)]
                print("    offset {:>2} {:>10}: totals {}".format(
                    off, arr, np.array2string(np.array(tot), precision=0, floatmode="fixed")))
        print("\n  Expect: high reach >> low reach; clustered ~ mixed at each offset.")
        sys.exit(0)

    t0 = time.time()
    offsets = [0.0, 5.0, 10.0, 15.0]
    arrangements = ["clustered", "mixed"]
    results, reach_of = run(K=2, samples_per_class=20, offsets=offsets,
                            arrangements=arrangements, seeds=list(range(8)),
                            pool_size=1500)
    report(results, reach_of, offsets, arrangements)
    print("\n(total {:.0f}s)".format(time.time() - t0))
