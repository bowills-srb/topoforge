"""Exp 37b (v2): Placement effect on REAL SHD data.
Rebuilt directly from diagnose_exp37e.py, whose run-loop is PROVEN to
produce healthy, stable bridge mass (~720-765) across all 15 epochs.
The previous version had a broken indentation: the rewire block and
`t += 1` ended up nested inside the `if t % 20 == 6:` reward block,
making the rewire condition unreachable (t%20==6 and (t+1)%40==0 are
mutually exclusive) and the step counter increment 20x too slow.

V DECAY: 0.99995 (was 0.999). Over the ~8,600-step life,\n0.999 decays to 2e-4 -- structure formed early was unmeasurable at\nreadout, so results reflected only the last few hundred steps.\n0.99995 retains ~0.65, making cumulative learning measurable.\n\nSEGREGATED GEOMETRY: clusters moved from ~30 units apart to ~14,\nso some input-output pairs fall within the 6.0 deposit radius.\nPreviously segregation made cross-type learning geometrically\nIMPOSSIBLE (exact zero); now it is possible but disadvantaged,\nmatching what the synthetic benchmark actually measures.\n\nLIFE LENGTH: 4 samples/class x 2 classes x 5 epochs x ~215 steps
~= 8,600 steps, matching the 800-1,200-step scale of every other
validated experiment in this project. The earlier 8-sample/15-epoch
config ran ~51,600 steps, over which V (decay 0.999) falls to
0.999^51600 ~= 4e-23 -- deposits happened the whole time but could not
survive to the final readout. Diagnosed by A/B-ing the experiment's own
run_life() against the diagnostic on identical inputs: run_life returned
1235/769 on the short config and 0 on the long one, isolating the cause
to life length rather than the physics or the measurement code.

Run: python src/experiments/exp37b_v2_real_data.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
from sparse_state import SparsePairState
from spatial import SpatialGrid
from shd_loader import load_shd_samples

N_INPUT = 140
N_OUTPUT_A = 30
N_OUTPUT_B = 30
N = N_INPUT + N_OUTPUT_A + N_OUTPUT_B
INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE = 0, 1, 2


def make_placement_segregated():
    """Same disc, same density as interleaved -- but each type occupies a
    contiguous angular WEDGE instead of being mixed. This keeps total
    input-output adjacency broadly comparable between conditions, so the
    comparison isolates ARRANGEMENT rather than raw opportunity count.
    An earlier version placed types in distant clusters, which gave
    segregated 63 adjacent pairs vs interleaved's 3,601 -- a 57x
    opportunity gap that made the measured ratio meaningless."""
    rng = np.random.default_rng(11)
    coords, cids = [], []
    # input occupies the lower half-disc, outputs share the upper half
    for _ in range(N_INPUT):
        th = rng.uniform(np.pi, 2*np.pi)       # lower half
        r = 10 * np.sqrt(rng.uniform(0, 1))
        coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)]); cids.append(INPUT_TYPE)
    for _ in range(N_OUTPUT_A):
        th = rng.uniform(0, np.pi/2)            # upper-right quarter
        r = 10 * np.sqrt(rng.uniform(0, 1))
        coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)]); cids.append(OUTPUT_A_TYPE)
    for _ in range(N_OUTPUT_B):
        th = rng.uniform(np.pi/2, np.pi)        # upper-left quarter
        r = 10 * np.sqrt(rng.uniform(0, 1))
        coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)]); cids.append(OUTPUT_B_TYPE)
    return np.array(coords), np.array(cids)


def make_placement_interleaved():
    rng = np.random.default_rng(11)
    coords, cids = [], []
    all_types = ([INPUT_TYPE]*N_INPUT + [OUTPUT_A_TYPE]*N_OUTPUT_A +
                 [OUTPUT_B_TYPE]*N_OUTPUT_B)
    rng.shuffle(all_types)
    for t_ in all_types:
        th = rng.uniform(0, 2*np.pi)
        r = 10 * np.sqrt(rng.uniform(0, 1))   # uniform-area, matches segregated
        coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)]); cids.append(t_)
    return np.array(coords), np.array(cids)


def run_life(coords, cids, seed, target_samples, distractor_samples,
             n_epochs=5, input_gain=0.35, swap=200):
    """Structure copied verbatim from the validated diagnostic."""
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.99995)
    Rhat = np.zeros(2)
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    input_idx = np.where(cids == INPUT_TYPE)[0]
    a_idx = np.where(cids == OUTPUT_A_TYPE)[0]
    b_idx = np.where(cids == OUTPUT_B_TYPE)[0]
    in_set = set(input_idx.tolist())
    a_set = set(a_idx.tolist()); b_set = set(b_idx.tolist())

    combined = [(s, 0) for s in target_samples] + [(s, 1) for s in distractor_samples]
    t = 0

    for epoch in range(n_epochs):
        for sample_arr, class_label in combined:
            recent_a = False; recent_b = False
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
                    if fs & a_set: recent_a = True
                    if fs & b_set: recent_b = True
                v = np.maximum(v_, 0); v[fired] = 0
                refrac[fired] = 3; refrac[refrac > 0] -= 1

                if t % 20 == 6:
                    stream = class_label
                    correct = recent_a if class_label == 0 else recent_b
                    base = 1.0 if correct else 0.0
                    delta = base - Rhat[stream]
                    if abs(delta) > 1e-9:
                        E.prune_below(1e-6)
                        for key in list(E.store.keys()):
                            ev = E.get(*key)
                            if ev != 0: V.deposit(key[0], key[1], delta * ev)
                    Rhat[stream] += 0.15 * delta
                    recent_a = False; recent_b = False

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
                cold = np.argsort(sc)[:swap]
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
                            order = np.argsort(score)[::-1][:len(cold)]
                            n2 = min(len(cold), len(order))
                            src[cold[:n2]] = ci[order[:n2]]; dst[cold[:n2]] = cj[order[:n2]]
                            out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                            for s2, d2b in zip(src, dst):
                                out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    bridge_a = 0.0; bridge_b = 0.0
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if si in in_set and di in a_set: bridge_a += max(V.get(si, di), 0)
        elif di in in_set and si in a_set: bridge_a += max(V.get(di, si), 0)
        if si in in_set and di in b_set: bridge_b += max(V.get(si, di), 0)
        elif di in in_set and si in b_set: bridge_b += max(V.get(di, si), 0)
    return bridge_a, bridge_b


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 37b (v2): DUAL-OUTPUT REAL SHD DATA")
    print("Rebuilt from the validated diagnostic run-loop")
    print("=" * 74)

    all_stimuli, all_labels = load_shd_samples(
        n_samples=200, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=200)
    from collections import Counter
    counts = Counter(all_labels)
    top = [c for c, _ in counts.most_common(2)]
    target_samples = [s for s, l in zip(all_stimuli, all_labels) if l == top[0]][:4]
    distractor_samples = [s for s, l in zip(all_stimuli, all_labels) if l == top[1]][:4]
    print("Target class {} ({} samples), distractor class {} ({} samples)".format(
        top[0], len(target_samples), top[1], len(distractor_samples)))
    est_steps = (len(target_samples) + len(distractor_samples)) * 5 * 215
    print("Life length: ~{:,} steps (V retention at end: {:.3f})".format(
        est_steps, 0.99995 ** est_steps))

    SEEDS = [0, 1, 2, 3, 4]
    strategies = [("segregated", make_placement_segregated),
                  ("interleaved", make_placement_interleaved)]

    # verify both conditions have SOME input-output adjacency (else the
    # comparison is impossible-vs-possible, not harder-vs-easier)
    from spatial import SpatialGrid as _SG
    for _name, _fn in strategies:
        _c, _ci = _fn()
        _g = _SG(_c, 6.0)
        _in = set(np.where(_ci == INPUT_TYPE)[0].tolist())
        _out = set(np.where(_ci != INPUT_TYPE)[0].tolist())
        _adj = sum(1 for i in _in for j in _g.within(i, 6.0) if int(j) in _out)
        print("  {:>12}: {:,} input-output adjacent pairs within radius 6.0".format(
            _name, _adj))
    print()

    all_results = {}
    for strat_name, placement_fn in strategies:
        print("\n  {}".format(strat_name.upper()))
        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = placement_fn()
            ba, bb = run_life(coords, cids, s, target_samples, distractor_samples)
            seed_results.append((ba, bb))
            print("    seed {}: bridge_A={:.1f}  bridge_B={:.1f}  total={:.1f}  ({:.0f}s)".format(
                s, ba, bb, ba + bb, time.time() - t0))
        all_results[strat_name] = np.array(seed_results)

    print("\n" + "=" * 74)
    print("RESULTS on REAL SHD DATA")
    means = {}
    for strat_name, _ in strategies:
        vals = all_results[strat_name]
        total = vals.sum(axis=1)
        means[strat_name] = total.mean()
        print("  {:>12}: total={:.1f} +/- {:.1f}".format(
            strat_name, total.mean(), total.std()))

    seg = means.get("segregated", 0); inter = means.get("interleaved", 0)
    ratio = inter / max(seg, 1e-9)
    print("\n" + "=" * 74)
    print("VERDICT: Interleaved/Segregated on REAL DATA = {:.2f}x".format(ratio))
    print("(Synthetic-pattern PLB reference: ~4.0x at N=900)")
