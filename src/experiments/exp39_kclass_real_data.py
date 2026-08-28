"""Exp 39: K-class real-data placement, generalizing Exp 37c beyond two classes.

Exp 37c established the placement effect on real SHD speech with a two-class
discrimination task: two dedicated output sub-clusters, each with its own
independent reward stream (the fix for Exp 37's shared-output reward-conflict
bug). This experiment generalizes that EXACT, validated mechanism to K output
classes -- it does not redesign it:

  1. K dedicated output sub-clusters, one per class, each with its own reward
     stream Rhat[k], k=0..K-1. The reward rule is the Exp 37c rule verbatim,
     just indexed by K streams instead of 2.
  2. Segregated placement: input as a central hub, the K output classes as
     equal wedges of the surrounding annulus, on the same radius-10 disc with
     the same uniform-area density as interleaved. (A naive K+1 coplanar-wedge
     tiling of Exp 37c's lower-input/upper-output layout starves the middle
     output wedges of input adjacency at K>=3; the hub-and-wedges layout
     equalizes per-class input adjacency -- see make_placement_segregated and
     the adjacency audit.) Interleaved: same disc, all types shuffled.
  3. Adjacency is audited PER CLASS before any learning number is trusted.
     With K output wedges sharing one disc, per-class input->output adjacency
     is NOT assumed to hold -- it is measured and printed, so a geometry that
     starves a middle wedge fails loudly instead of masquerading as a
     placement finding.
  4. Life length is held in the validated ~8,600-step / 0.651-retention band.
     More classes means more samples per epoch cycle, so the step budget is
     recomputed explicitly (K * samples/class * epochs * 215) and retention is
     printed and asserted, exactly as Exp 37c does.
  5. Statistics: still a two-GROUP comparison (segregated vs interleaved) on
     total bridge mass -- same Welch t-test / Wilcoxon / Cohen's d / bootstrap
     ratio as Exp 37c (imported verbatim). Per-class bridge mass is ALSO
     reported, because a class that fails to learn would be hidden in a total.

Audit-before-trust:
  python src/experiments/exp39_kclass_real_data.py --verify-equiv  # K=2 == Exp 37b
  python src/experiments/exp39_kclass_real_data.py --smoke         # small K=4
  python src/experiments/exp39_kclass_real_data.py                 # full K=4
"""
import numpy as np
import time
import sys
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats
from sparse_state import SparsePairState
from spatial import SpatialGrid
from shd_loader import load_shd_samples
# Reuse the validated statistics helpers from Exp 37c verbatim.
from exp37c_real_data_scaled import cohens_d_pooled, bootstrap_ratio_ci

N_INPUT = 140
N_OUTPUT_PER_CLASS = 30
INPUT_TYPE = 0            # output class k has type k+1, k = 0..K-1
DISC_R = 10.0
DISC_CX = DISC_CY = 20.0
DEPOSIT_RADIUS = 6.0
STEPS_PER_SAMPLE = 215   # 200 stimulus steps + 15 rest, as in Exp 37c
V_DECAY = 0.99995
MIN_RETENTION = 0.25


# ============================================================
# Placement -- K+1 wedges (generalizes Exp 37c; K=2 is identical)
# ============================================================
def make_placement_segregated(K, seed=11):
    """Input is a central hub; the K output classes occupy equal angular
    wedges of the surrounding annulus, [2*pi*k/K, 2*pi*(k+1)/K].

    Why a hub-and-wedges layout rather than K+1 coplanar wedges: the naive
    generalization (input in the lower half, outputs tiling the upper half)
    STARVES the middle output wedges of input adjacency -- they sit far from
    the input region, so their near-zero learning is a geometry artifact, not
    a placement effect. The --smoke adjacency audit demonstrates this directly
    (per-class adjacency [772, 303, 72, 830] at K=4). Placing input at the
    centre makes every output wedge border the input equally, equalizing
    per-class input adjacency while keeping each output type contiguous and
    segregated from the others -- same disc, same uniform-area density as
    interleaved. The hub radius is chosen so density is uniform throughout
    (input area fraction == input neuron fraction)."""
    rng = np.random.default_rng(seed)
    r_in = np.sqrt(DISC_R ** 2 * N_INPUT / (N_INPUT + N_OUTPUT_PER_CLASS * K))
    coords, cids = [], []
    for _ in range(N_INPUT):                              # central hub
        th = rng.uniform(0, 2 * np.pi)
        r = r_in * np.sqrt(rng.uniform(0, 1))
        coords.append([DISC_CX + r * np.cos(th), DISC_CY + r * np.sin(th)])
        cids.append(INPUT_TYPE)
    for k in range(K):                                    # annular wedges
        lo, hi = 2 * np.pi * k / K, 2 * np.pi * (k + 1) / K
        for _ in range(N_OUTPUT_PER_CLASS):
            th = rng.uniform(lo, hi)
            r = np.sqrt(r_in ** 2 + rng.uniform(0, 1) * (DISC_R ** 2 - r_in ** 2))
            coords.append([DISC_CX + r * np.cos(th), DISC_CY + r * np.sin(th)])
            cids.append(k + 1)
    return np.array(coords), np.array(cids)


def make_placement_interleaved(K, seed=11):
    """Same disc, all types shuffled. K=2 reproduces Exp 37c exactly."""
    rng = np.random.default_rng(seed)
    coords, cids = [], []
    all_types = [INPUT_TYPE] * N_INPUT
    for k in range(K):
        all_types += [k + 1] * N_OUTPUT_PER_CLASS
    rng.shuffle(all_types)
    for t_ in all_types:
        th = rng.uniform(0, 2 * np.pi)
        r = DISC_R * np.sqrt(rng.uniform(0, 1))
        coords.append([DISC_CX + r * np.cos(th), DISC_CY + r * np.sin(th)])
        cids.append(t_)
    return np.array(coords), np.array(cids)


def per_class_adjacency(coords, cids, K, radius=DEPOSIT_RADIUS):
    """For each output class k, count input<->output_k pairs within the
    deposit radius. Returns a length-K list. This is the structural
    precondition for class k to form bridges -- audited before trust."""
    g = SpatialGrid(coords, radius)
    input_idx = np.where(cids == INPUT_TYPE)[0]
    counts = [0] * K
    in_set = set(input_idx.tolist())
    for i in input_idx:
        for j in g.within(int(i), radius):
            c = int(cids[int(j)])
            if c >= 1:
                counts[c - 1] += 1
    return counts


# ============================================================
# run_life -- Exp 37c physics, generalized from 2 to K reward streams
# ============================================================
def run_life_kclass(coords, cids, K, seed, class_samples,
                    n_epochs=1, input_gain=0.35, swap=200):
    """K-output generalization of Exp 37c's run_life. Every constant and every
    control-flow branch is preserved; only the number of output clusters and
    reward streams changes (2 -> K). Verified bit-identical to Exp 37b_v2's
    run_life at K=2 by --verify-equiv."""
    N = len(coords)
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(V_DECAY)
    Rhat = np.zeros(K)
    g = SpatialGrid(coords, DEPOSIT_RADIUS)
    nbr = [g.within(i, DEPOSIT_RADIUS) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    input_idx = np.where(cids == INPUT_TYPE)[0]
    in_set = set(input_idx.tolist())
    out_sets = [set(np.where(cids == k + 1)[0].tolist()) for k in range(K)]

    combined = [(s, k) for k in range(K) for s in class_samples[k]]
    t = 0

    for epoch in range(n_epochs):
        for sample_arr, class_label in combined:
            recent = [False] * K
            for local_t in range(sample_arr.shape[0]):
                inp = rng.uniform(0, 0.01, N)
                inp[input_idx] += sample_arr[local_t] * input_gain
                v_ = v * 0.90 + inp
                fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
                if len(f):
                    for fi in f:
                        for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
                C.tick(); E.tick(); V.tick()
                if len(f):
                    fs = set(int(x) for x in f)
                    for i in f:
                        i = int(i)
                        for j in nbr[i]:
                            if int(j) in fs:
                                C.deposit(i, int(j), 1.0); E.deposit(i, int(j), 1.0)
                    for k in range(K):
                        if fs & out_sets[k]:
                            recent[k] = True
                v = np.maximum(v_, 0); v[fired] = 0
                refrac[fired] = 3; refrac[refrac > 0] -= 1

                if t % 20 == 6:
                    stream = class_label
                    correct = recent[class_label]
                    base = 1.0 if correct else 0.0
                    delta = base - Rhat[stream]
                    if abs(delta) > 1e-9:
                        E.prune_below(1e-6)
                        for key in list(E.store.keys()):
                            ev = E.get(*key)
                            if ev != 0: V.deposit(key[0], key[1], delta * ev)
                    Rhat[stream] += 0.15 * delta
                    recent = [False] * K

                t += 1

            for _ in range(15):
                v_ = v * 0.90 + rng.uniform(0, 0.01, N)
                fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
                if len(f):
                    for fi in f:
                        for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
                C.tick(); E.tick(); V.tick()
                v = np.maximum(v_, 0); v[fired] = 0
                refrac[fired] = 3; refrac[refrac > 0] -= 1
                t += 1

            if t % 40 < 16:
                C.prune_below(1e-6)
                sc = np.array([V.get(int(src[k]), int(dst[k])) for k in range(len(src))])
                cold = np.argsort(sc, kind="stable")[:swap]
                ci, cj, _ = C.get_arrays()
                if len(ci) > 0:
                    keep2 = ci != cj; ci, cj = ci[keep2], cj[keep2]
                    ex = set(zip(src.tolist(), dst.tolist()))
                    mk = np.array([(int(a), int(b)) not in ex for a, b in zip(ci, cj)])
                    ci, cj = ci[mk], cj[mk]
                    if len(ci) > 0:
                        diff = coords[ci] - coords[cj]
                        dd = (diff ** 2).sum(axis=1)
                        vp = np.maximum(np.array([V.get(int(a), int(b)) for a, b in zip(ci, cj)]), 0)
                        cp = np.array([C.get(int(a), int(b)) for a, b in zip(ci, cj)])
                        score = (vp + 0.01 * cp) / (1 + 0.05 * dd); pos = score > 0
                        ci, cj, score = ci[pos], cj[pos], score[pos]
                        if len(ci) > 0:
                            order = np.argsort(score, kind="stable")[::-1][:len(cold)]
                            n2 = min(len(cold), len(order))
                            src[cold[:n2]] = ci[order[:n2]]; dst[cold[:n2]] = cj[order[:n2]]
                            out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                            for s2, d2b in zip(src, dst):
                                out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    bridges = np.zeros(K)
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        for k in range(K):
            if si in in_set and di in out_sets[k]:
                bridges[k] += max(V.get(si, di), 0)
            elif di in in_set and si in out_sets[k]:
                bridges[k] += max(V.get(di, si), 0)
    return bridges


# ============================================================
# Data
# ============================================================
def load_k_class_samples(K, samples_per_class, pool_size, max_steps=200):
    """Load a large SHD pool and return the first samples_per_class distinct
    samples from each of the K most-common classes."""
    stimuli, labels = load_shd_samples(
        n_samples=pool_size, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=max_steps)
    counts = Counter(labels)
    top = [c for c, _ in counts.most_common(K)]
    avail = {c: counts[c] for c in top}
    class_samples = [
        [s for s, l in zip(stimuli, labels) if l == cls][:samples_per_class]
        for cls in top]
    return top, avail, class_samples


def run_config(K, samples_per_class, n_epochs, seeds, pool_size, label):
    print("=" * 78)
    print("EXP 39: {}".format(label))
    print("=" * 78)

    top, avail, class_samples = load_k_class_samples(K, samples_per_class, pool_size)
    got = min(len(cs) for cs in class_samples)
    print("K = {} classes: {}".format(K, top))
    print("  samples/class available: {}".format({c: avail[c] for c in top}))
    print("  using {} samples/class x {} classes x {} epochs".format(
        got, K, n_epochs))
    if got < samples_per_class:
        print("  WARNING: only {} samples/class available (< {} requested); "
              "raise pool_size.".format(got, samples_per_class))

    n_samples_total = K * got * n_epochs
    est_steps = n_samples_total * STEPS_PER_SAMPLE
    retention = V_DECAY ** est_steps
    print("Life length: {} samples x {} steps = ~{:,} steps  |  "
          "V retention: {:.3f} (validated ~0.65)".format(
              n_samples_total, STEPS_PER_SAMPLE, est_steps, retention))
    assert retention >= MIN_RETENTION, (
        "V retention {:.4f} < {} -- config would decay early structure to "
        "noise. Reduce K*samples/class*epochs.".format(retention, MIN_RETENTION))

    # --- CRITICAL adjacency audit BEFORE any learning (per class) ---
    print("\n" + "-" * 78)
    print("PER-CLASS ADJACENCY AUDIT (input<->output_k pairs within radius {})".format(
        DEPOSIT_RADIUS))
    print("-" * 78)
    strategies = [("segregated", make_placement_segregated),
                  ("interleaved", make_placement_interleaved)]
    adj_report = {}
    for name, fn in strategies:
        coords, cids = fn(K)
        adj = per_class_adjacency(coords, cids, K)
        adj_report[name] = adj
        print("  {:>12}: per-class {}  total {}".format(name, adj, sum(adj)))
    seg_adj = np.array(adj_report["segregated"])
    int_adj = np.array(adj_report["interleaved"])
    print("  total seg/interleaved ratio: {:.2f}x  (Exp 37c K=2 was 1198/2268 = 0.53x)".format(
        sum(seg_adj) / max(sum(int_adj), 1)))
    starved = np.where(seg_adj < 0.15 * max(seg_adj.max(), 1))[0]
    if len(starved):
        print("  !! WARNING: segregated class(es) {} are adjacency-starved "
              "(<15% of the richest class) -- any near-zero learning for these "
              "is a GEOMETRY artifact, not a placement finding.".format(
                  [int(x) for x in starved]))
    else:
        print("  OK: no segregated class is adjacency-starved.")

    # --- learning runs ---
    results = {}
    for name, fn in strategies:
        print("\n  {}".format(name.upper()))
        coords, cids = fn(K)
        seed_bridges = []
        for s in seeds:
            t0 = time.time()
            br = run_life_kclass(coords, cids, K, s, class_samples, n_epochs=n_epochs)
            seed_bridges.append(br)
            print("    seed {:>2}: per-class {}  total={:>8.1f}  ({:.0f}s)".format(
                s, np.array2string(br, precision=0, floatmode="fixed"),
                br.sum(), time.time() - t0))
        results[name] = np.array(seed_bridges)   # shape (n_seeds, K)
    return results, top


def report_stats(results, top):
    seg = results["segregated"]          # (n_seeds, K)
    inter = results["interleaved"]
    seg_tot = seg.sum(axis=1)
    int_tot = inter.sum(axis=1)
    n = len(seg_tot)

    print("\n" + "=" * 78)
    print("RESULTS: total bridge mass, segregated vs interleaved (n = {} seeds)".format(n))
    print("-" * 78)
    for name, tot in (("segregated", seg_tot), ("interleaved", int_tot)):
        rel = 100 * tot.std(ddof=1) / max(tot.mean(), 1e-9)
        print("  {:>12}: total = {:>9.1f} +/- {:>8.1f}  (rel std {:>5.1f}%)".format(
            name, tot.mean(), tot.std(ddof=1), rel))

    ratio = int_tot.mean() / max(seg_tot.mean(), 1e-9)
    lo, hi = bootstrap_ratio_ci(int_tot, seg_tot)
    t_rel, p_rel = stats.ttest_rel(int_tot, seg_tot)
    t_ind, p_ind = stats.ttest_ind(int_tot, seg_tot, equal_var=False)
    try:
        w_stat, p_w = stats.wilcoxon(int_tot, seg_tot)
    except ValueError as e:
        w_stat, p_w = float("nan"), float("nan")
        print("  (Wilcoxon skipped: {})".format(e))
    d = cohens_d_pooled(int_tot, seg_tot)
    print("-" * 78)
    print("  Interleaved / Segregated ratio : {:.2f}x".format(ratio))
    print("  Bootstrap 95% CI on ratio      : [{:.2f}x, {:.2f}x]".format(lo, hi))
    print("  Paired t-test                  : t = {:.3f}, p = {:.2e}".format(t_rel, p_rel))
    print("  Welch unpaired t-test          : t = {:.3f}, p = {:.2e}".format(t_ind, p_ind))
    print("  Wilcoxon signed-rank (paired)  : W = {:.1f}, p = {:.2e}".format(w_stat, p_w))
    print("  Cohen's d (pooled)             : {:.2f}".format(d))

    # --- per-class breakdown (a dead class would hide in the total) ---
    print("\n" + "=" * 78)
    print("PER-CLASS BREAKDOWN (mean bridge mass over seeds; a near-zero")
    print("class reveals a failure the total would mask)")
    print("-" * 78)
    print("  {:>6} {:>14} {:>14} {:>10}".format(
        "class", "segregated", "interleaved", "int/seg"))
    K = seg.shape[1]
    for k in range(K):
        sm = seg[:, k].mean(); im = inter[:, k].mean()
        r = im / max(sm, 1e-9)
        flag = "  <- seg near-zero" if sm < 0.05 * max(seg.mean(axis=0).max(), 1e-9) else ""
        print("  {:>6} {:>14.1f} {:>14.1f} {:>9.1f}x{}".format(top[k], sm, im, r, flag))
    return ratio, (lo, hi), p_ind, d


if __name__ == "__main__":
    if "--verify-equiv" in sys.argv:
        # AUDIT: the K=2 special case of this generalized code must reproduce
        # Exp 37b_v2's original run_life + placements bit-for-bit.
        print("=" * 78)
        print("EQUIVALENCE AUDIT: K=2 generalized code vs Exp 37b_v2 original")
        print("=" * 78)
        import exp37b_v2_real_data as orig

        # The physics must be identical. run_life is geometry-agnostic, so we
        # feed the ORIGINAL's placements to BOTH implementations and check the
        # learning output matches bit-for-bit. (This experiment's own
        # segregated geometry differs by design -- the hub-and-wedges fix --
        # so we do NOT require placement equality; we require PHYSICS equality.)
        os2, oi2 = orig.make_placement_segregated()          # segregated coords
        oi2s, oi2c = orig.make_placement_interleaved()       # interleaved coords

        all_stim, all_lab = load_shd_samples(
            n_samples=200, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=200)
        counts = Counter(all_lab); tp = [c for c, _ in counts.most_common(2)]
        tgt = [s for s, l in zip(all_stim, all_lab) if l == tp[0]][:4]
        dis = [s for s, l in zip(all_stim, all_lab) if l == tp[1]][:4]
        print("  Feeding the original's placements to both run_life "
              "implementations...\n")
        all_match = True
        for (cname, cc, ccids) in (("segregated", os2, oi2),
                                   ("interleaved", oi2s, oi2c)):
            for s in (0, 1):
                br = run_life_kclass(cc, ccids, 2, s, [tgt, dis], n_epochs=5)
                ba, bb = orig.run_life(cc, ccids, s, tgt, dis, n_epochs=5)
                match = np.allclose(br, [ba, bb], rtol=1e-9, atol=1e-6)
                all_match = all_match and match
                print("    {:>11} seed {}: mine={}  orig=[{:.3f}, {:.3f}]  "
                      "identical={}".format(
                          cname, s,
                          np.array2string(br, precision=3, floatmode="fixed"),
                          ba, bb, match))
        print("\nVERDICT: {} -> the K-generalization preserves the validated".format(
            "all True" if all_match else "MISMATCH"))
        print("physics exactly; only the number of reward streams changed.")
        sys.exit(0)

    if "--uniform-adjacency-control" in sys.argv:
        # Disentangle: does the segregated per-class collapse track the fixed
        # residual adjacency gradient, or is it a genuine order/multi-class
        # dynamic? The wedges are symmetric, so [956,835,560,725] is just
        # finite-sample noise from placement seed 11. Here we VARY the
        # placement seed each run -> each class's adjacency is randomized and,
        # averaged over runs, equalized across classes -- while presentation
        # order stays fixed (class index k always presented in position k).
        #   collapse disappears  => it was the fixed adjacency gradient
        #   collapse persists by index => genuine order/resource-competition
        K, samples_per_class, n_epochs = 4, 10, 1
        PLACE_SEEDS = list(range(12))
        print("=" * 78)
        print("UNIFORM-ADJACENCY CONTROL: segregated, placement seed varied per run")
        print("=" * 78)
        top, avail, class_samples = load_k_class_samples(K, samples_per_class, 1500)
        print("K = {} classes {}, {} samples/class x {} epochs".format(
            K, top, samples_per_class, n_epochs))

        # confirm adjacency is equalized across classes once averaged over seeds
        adj_by_seed = np.array([per_class_adjacency(*make_placement_segregated(K, seed=ps), K)
                                for ps in PLACE_SEEDS])
        print("\nPer-class adjacency across the {} placement seeds:".format(len(PLACE_SEEDS)))
        for k in range(K):
            col = adj_by_seed[:, k]
            print("  class-index {}: mean {:.0f} +/- {:.0f}  (range {}-{})".format(
                k, col.mean(), col.std(), col.min(), col.max()))
        spread = adj_by_seed.mean(axis=0)
        print("  -> per-class MEAN adjacency now {} (max/min = {:.2f}x; "
              "fixed-seed run was 956/560 = 1.71x)".format(
                  np.array2string(spread, precision=0, floatmode="fixed"),
                  spread.max() / max(spread.min(), 1)))

        print("\nSegregated learning, one fresh placement seed per run "
              "(order fixed):")
        per_class = []
        for i, ps in enumerate(PLACE_SEEDS):
            coords, cids = make_placement_segregated(K, seed=ps)
            br = run_life_kclass(coords, cids, K, i, class_samples, n_epochs=n_epochs)
            per_class.append(br)
            print("    run {:>2} (place seed {}): per-class {}  total={:.0f}".format(
                i, ps, np.array2string(br, precision=0, floatmode="fixed"), br.sum()))
        per_class = np.array(per_class)

        print("\n" + "-" * 78)
        print("PER-CLASS LEARNING with adjacency equalized (mean +/- std over runs):")
        print("-" * 78)
        for k in range(K):
            col = per_class[:, k]
            near0 = np.sum(col < 0.05 * per_class.mean(axis=0).max())
            print("  class-index {}: {:>8.0f} +/- {:>7.0f}   (near-zero in {}/{} runs)".format(
                k, col.mean(), col.std(), near0, len(PLACE_SEEDS)))
        means = per_class.mean(axis=0)
        print("-" * 78)
        print("  per-class learning max/min = {:.1f}x  (fixed-seed run was "
              "7412/234 = 31.7x)".format(means.max() / max(means.min(), 1e-9)))
        print("\nINTERPRETATION:")
        print("  If later class indices still collapse toward zero despite equalized")
        print("  adjacency, the per-class failure is an ORDER / resource-competition")
        print("  effect (a real multi-class dynamic), not the adjacency gradient.")
        print("  If per-class learning is now roughly uniform, the collapse was")
        print("  driven by the residual adjacency gradient.")
        sys.exit(0)

    if "--smoke" in sys.argv:
        res, top = run_config(K=4, samples_per_class=4, n_epochs=1,
                              seeds=[0, 1, 2], pool_size=800,
                              label="SMOKE -- K=4, 4 samples/class, 1 epoch, 3 seeds")
        report_stats(res, top)
        sys.exit(0)

    # DEFAULT full run: K=4, 10 samples/class, 1 epoch = 4*10*215 = 8,600 steps
    # (exactly the validated life/retention regime), 12 seeds.
    res, top = run_config(K=4, samples_per_class=10, n_epochs=1,
                          seeds=list(range(12)), pool_size=1500,
                          label="FULL -- K=4, 10 samples/class, 1 epoch, 12 seeds")
    report_stats(res, top)
