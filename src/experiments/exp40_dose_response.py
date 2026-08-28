"""Exp 40: dose-response and mediation -- is reachable-correlated-fraction THE
variable that placement acts through?

Every prior placement result is a two-point contrast (segregated vs
interleaved). Exp 39's uniform-adjacency control revealed that per-class
outcomes track a placement-only adjacency metric with a threshold
nonlinearity. This experiment turns the contrast into a causal dose-response
and tests the mediation claim directly:

  Knob: alpha in [0, 1] interpolates placement from fully SEGREGATED
  (alpha=0, Exp 39's hub-and-wedges) to fully INTERLEAVED (alpha=1). Each
  neuron independently keeps its segregated position with prob 1-alpha, or is
  scattered to a random disc position with prob alpha. alpha=0/1 reproduce the
  two validated endpoints.

  Mediator (placement-only, measured BEFORE learning): the mean number of
  input neurons within the plasticity radius of each output neuron -- the
  structural "reach" that lets input->output bridges form at all.

Two pre-registered, falsifiable predictions:
  (P1) Dose-response: total learned bridge mass increases monotonically in
       alpha. Falsified if non-monotonic (Spearman rho < ~0.9).
  (P2) Mediation / collapse: normalized learning is a single function of the
       reach metric -- the alpha-sweep points at K=2 AND K=4 fall on ONE
       curve. Falsified if K=2 and K=4 do not collapse, or reach fails to
       predict learning (low rank correlation).

If both hold, placement matters exactly insofar as it sets the reachable
correlated fraction; this unifies the segregated/interleaved/random/SpiNeMap
(Exp 38) and K=2/K=4 (Exp 37c/39) results as points on one curve.

Life is held in the validated ~8,600-step / 0.651-retention regime
(K=2: 20 samples/class; K=4: 10 samples/class; 1 epoch).

Audit-before-trust:
  python src/experiments/exp40_dose_response.py --smoke   # endpoints match Exp 39?
  python src/experiments/exp40_dose_response.py            # full sweep
"""
import numpy as np
import time
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats
from spatial import SpatialGrid
from exp39_kclass_real_data import (
    N_INPUT, N_OUTPUT_PER_CLASS, INPUT_TYPE, DISC_R, DISC_CX, DISC_CY,
    DEPOSIT_RADIUS, V_DECAY, STEPS_PER_SAMPLE, MIN_RETENTION,
    run_life_kclass, load_k_class_samples,
    make_placement_segregated, make_placement_interleaved,
)


def make_placement_alpha(K, alpha, seed=11):
    """Interpolate between segregated (alpha=0) and interleaved (alpha=1).
    Each neuron keeps its segregated position with prob 1-alpha, else is
    scattered to a uniform-area random disc position. Types are fixed; only
    positions move. alpha=0 -> hub-and-wedges; alpha=1 -> random scatter."""
    rng = np.random.default_rng(seed)
    r_in = np.sqrt(DISC_R ** 2 * N_INPUT / (N_INPUT + N_OUTPUT_PER_CLASS * K))
    types = [INPUT_TYPE] * N_INPUT
    for k in range(K):
        types += [k + 1] * N_OUTPUT_PER_CLASS

    def seg_pos(t):
        if t == INPUT_TYPE:                                  # central hub
            th = rng.uniform(0, 2 * np.pi)
            r = r_in * np.sqrt(rng.uniform(0, 1))
        else:                                                # annular wedge
            k = t - 1
            th = rng.uniform(2 * np.pi * k / K, 2 * np.pi * (k + 1) / K)
            r = np.sqrt(r_in ** 2 + rng.uniform(0, 1) * (DISC_R ** 2 - r_in ** 2))
        return DISC_CX + r * np.cos(th), DISC_CY + r * np.sin(th)

    def rand_pos():
        th = rng.uniform(0, 2 * np.pi)
        r = DISC_R * np.sqrt(rng.uniform(0, 1))
        return DISC_CX + r * np.cos(th), DISC_CY + r * np.sin(th)

    coords, cids = [], []
    for t in types:
        scatter = rng.uniform(0, 1) < alpha
        x, y = rand_pos() if scatter else seg_pos(t)
        coords.append([x, y]); cids.append(t)
    return np.array(coords), np.array(cids)


def reach_metrics(coords, cids, radius=DEPOSIT_RADIUS):
    """Placement-only 'reach': for each OUTPUT neuron, how many INPUT neurons
    lie within the plasticity radius. Returns (mean_count, frac_with_any).
    mean_count is the graded mediator; frac is a saturating companion."""
    g = SpatialGrid(coords, radius)
    out_idx = np.where(cids != INPUT_TYPE)[0]
    counts = []
    for i in out_idx:
        c = sum(1 for j in g.within(int(i), radius) if int(cids[int(j)]) == INPUT_TYPE)
        counts.append(c)
    counts = np.array(counts)
    return counts.mean(), float((counts > 0).mean())


def sweep(K, samples_per_class, alphas, seeds, pool_size, place_seed=11):
    top, avail, class_samples = load_k_class_samples(K, samples_per_class, pool_size)
    n_total = K * samples_per_class
    est_steps = n_total * STEPS_PER_SAMPLE
    retention = V_DECAY ** est_steps
    print("\n### K = {}  ({} classes {}, {} samples/class, 1 epoch)".format(
        K, K, top, samples_per_class))
    print("    life ~{:,} steps, V retention {:.3f}".format(est_steps, retention))
    assert retention >= MIN_RETENTION, "retention {:.3f} too low".format(retention)

    rows = []
    for a in alphas:
        coords, cids = make_placement_alpha(K, a, seed=place_seed)
        mean_reach, frac_reach = reach_metrics(coords, cids)
        totals = []
        for s in seeds:
            br = run_life_kclass(coords, cids, K, s, class_samples, n_epochs=1)
            totals.append(br.sum())
        totals = np.array(totals)
        rows.append({"K": K, "alpha": a, "mean_reach": mean_reach,
                     "frac_reach": frac_reach, "learn_mean": totals.mean(),
                     "learn_std": totals.std(ddof=1)})
        print("    alpha={:.2f}  reach(mean/frac)={:5.2f}/{:.2f}  "
              "learn={:8.0f} +/- {:6.0f}".format(
                  a, mean_reach, frac_reach, totals.mean(), totals.std(ddof=1)))
    return rows


def report(all_rows):
    print("\n" + "=" * 78)
    print("DOSE-RESPONSE (P1): learning monotonic in alpha?")
    print("-" * 78)
    for K in sorted(set(r["K"] for r in all_rows)):
        rk = [r for r in all_rows if r["K"] == K]
        a = [r["alpha"] for r in rk]; l = [r["learn_mean"] for r in rk]
        rho, p = stats.spearmanr(a, l)
        lo, hi = rk[0]["learn_mean"], rk[-1]["learn_mean"]
        print("  K={}: Spearman(alpha, learning) rho = {:.3f} (p={:.2e})  "
              "alpha0->1: {:.0f} -> {:.0f} ({:.1f}x)".format(
                  K, rho, p, lo, hi, hi / max(lo, 1e-9)))

    print("\n" + "=" * 78)
    print("MEDIATION / COLLAPSE (P2): does reach predict learning across K?")
    print("-" * 78)
    # normalize learning per K by that K's alpha=1 (interleaved) ceiling so
    # K=2 and K=4 share a [0,1] axis; the test is whether the MIDDLE points
    # coincide as a single function of reach.
    for K in sorted(set(r["K"] for r in all_rows)):
        ceil = max(r["learn_mean"] for r in all_rows
                   if r["K"] == K and r["alpha"] == max(
                       rr["alpha"] for rr in all_rows if rr["K"] == K))
        for r in all_rows:
            if r["K"] == K:
                r["learn_norm"] = r["learn_mean"] / max(ceil, 1e-9)

    reach = np.array([r["mean_reach"] for r in all_rows])
    lnorm = np.array([r["learn_norm"] for r in all_rows])
    pr, pp = stats.pearsonr(reach, lnorm)
    sr, sp = stats.spearmanr(reach, lnorm)
    print("  pooled points (both K):")
    print("  {:>4} {:>7} {:>12} {:>14}".format("K", "alpha", "mean_reach", "learn_norm"))
    for r in sorted(all_rows, key=lambda x: x["mean_reach"]):
        print("  {:>4} {:>7.2f} {:>12.2f} {:>14.3f}".format(
            r["K"], r["alpha"], r["mean_reach"], r["learn_norm"]))
    print("-" * 78)
    print("  Pearson  r = {:.3f} (R^2 = {:.3f}), p = {:.2e}".format(pr, pr ** 2, pp))
    print("  Spearman rho = {:.3f}, p = {:.2e}".format(sr, sp))
    print("\n  P1 (dose-response) holds if both Spearman rho ~ 1.")
    print("  P2 (collapse) holds if pooled reach->learn is tight (high R^2/rho)")
    print("  AND the K=2 and K=4 points interleave rather than forming two")
    print("  separate curves.")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        # Audit-before-trust: endpoints must reproduce Exp 39's segregated /
        # interleaved reach levels, and the midpoint must fall between.
        print("=" * 78)
        print("SMOKE / ENDPOINT AUDIT: alpha endpoints vs Exp 39 placements")
        print("=" * 78)
        K = 2
        for a in (0.0, 0.5, 1.0):
            c, ci = make_placement_alpha(K, a)
            mr, fr = reach_metrics(c, ci)
            print("  alpha={:.1f}: mean_reach={:5.2f}  frac_reach={:.2f}".format(a, mr, fr))
        cs, cis = make_placement_segregated(K)
        cin, ciin = make_placement_interleaved(K)
        print("  Exp39 segregated : mean_reach={:5.2f}  (should ~ alpha=0)".format(
            reach_metrics(cs, cis)[0]))
        print("  Exp39 interleaved: mean_reach={:5.2f}  (should ~ alpha=1)".format(
            reach_metrics(cin, ciin)[0]))

        print("\n  quick learning at endpoints (2 seeds), K=2, 20 samples/class:")
        top, avail, csamp = load_k_class_samples(2, 20, 1500)
        for a in (0.0, 1.0):
            c, ci = make_placement_alpha(K, a)
            tot = [run_life_kclass(c, ci, K, s, csamp, n_epochs=1).sum() for s in (0, 1)]
            print("    alpha={:.1f}: totals {}".format(
                a, np.array2string(np.array(tot), precision=0, floatmode="fixed")))
        print("\n  Expect alpha=0 ~ segregated (low), alpha=1 ~ interleaved (high).")
        sys.exit(0)

    t0 = time.time()
    all_rows = []
    all_rows += sweep(K=2, samples_per_class=20,
                      alphas=[0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0],
                      seeds=list(range(8)), pool_size=1500)
    all_rows += sweep(K=4, samples_per_class=10,
                      alphas=[0.0, 0.33, 0.67, 1.0],
                      seeds=list(range(8)), pool_size=1500)
    report(all_rows)
    print("\n(total {:.0f}s)".format(time.time() - t0))
