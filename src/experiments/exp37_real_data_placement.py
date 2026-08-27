"""Exp 37: Placement effect on REAL SHD data.
Every prior experiment used synthetic, hand-designed periodic patterns.
This is the fix for the single biggest gap identified in the skeptical
review: does the segregated-vs-interleaved effect survive when the
input is REAL speech-derived spike data with real statistical structure,
not a clean synthetic cycle?

Design: same proven engine mechanics as Exp 35b (local deposit + V+C
rewire, spatial placement), but the input layer is driven by real SHD
samples via shd_loader.py instead of synthetic firing patterns. Two
SHD classes: one rewarded ("target" class), one not ("distractor").
Segregated vs interleaved placement of input vs output layers.

Run: python src/experiments/exp37_real_data_placement.py
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
N_OUTPUT = 60
N = N_INPUT + N_OUTPUT
INPUT_TYPE, OUTPUT_TYPE = 0, 1
COINCIDENCE_WINDOW = 8
SAMPLE_GAP = 20  # steps of quiet between sample presentations


def make_placement_segregated():
    """Input neurons on one side, output neurons on the other -- far apart."""
    rng = np.random.default_rng(11)
    coords, cids = [], []
    for _ in range(N_INPUT):
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 8)
        coords.append([5 + r*np.cos(th), 5 + r*np.sin(th)])
        cids.append(INPUT_TYPE)
    for _ in range(N_OUTPUT):
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 8)
        coords.append([35 + r*np.cos(th), 35 + r*np.sin(th)])
        cids.append(OUTPUT_TYPE)
    return np.array(coords), np.array(cids)


def make_placement_interleaved():
    """Input and output neurons physically mixed together."""
    rng = np.random.default_rng(11)
    coords, cids = [], []
    all_types = [INPUT_TYPE] * N_INPUT + [OUTPUT_TYPE] * N_OUTPUT
    rng.shuffle(all_types)
    for t in all_types:
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 10)
        coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)])
        cids.append(t)
    return np.array(coords), np.array(cids)


def run_life(coords, cids, seed, target_samples, distractor_samples,
             n_epochs=15, input_gain=0.35):
    """target_samples/distractor_samples: lists of (T, N_INPUT) real
    spike-count arrays from shd_loader. Cycles through them repeatedly."""
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(2)  # 0=target, 1=distractor
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 200

    input_idx = np.where(cids == INPUT_TYPE)[0]
    output_idx = np.where(cids == OUTPUT_TYPE)[0]

    # build the full presentation sequence: alternating target/distractor
    # samples, repeated for n_epochs
    all_samples = []
    for _ in range(n_epochs):
        combined = [(s, 0) for s in target_samples] + [(s, 1) for s in distractor_samples]
        rng3.shuffle(combined)
        all_samples.extend(combined)

    t = 0
    for sample_arr, class_label in all_samples:
        T_sample = sample_arr.shape[0]
        output_fired_count = 0

        recent_output_fired = False
        for local_t in range(T_sample):
            inp = rng3.uniform(0, 0.01, N)
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
            v = np.maximum(v_, 0); v[fired] = 0
            refrac[fired] = 3; refrac[refrac > 0] -= 1

            if len(f):
                output_fired_now = len(set(int(x) for x in f) & set(output_idx.tolist())) > 0
                if output_fired_now:
                    output_fired_count += 1
                    recent_output_fired = True

            # REWARD APPLIED EVERY 20 STEPS, not once at the end -- matches
            # the cadence used throughout the rest of the project. Waiting
            # until a 200-step sample fully finished let E (decay=0.90)
            # decay to ~9e-10 before reward ever arrived -- structurally
            # impossible for any deposit to register. Caught by the fast
            # diagnostic script + reasoning about decay^steps, not by luck.
            if t % 20 == 6:
                base = 1.0 if (class_label == 0 and recent_output_fired) else \
                       (0.0 if class_label == 0 else (-0.3 if recent_output_fired else 0.3))
                delta = base - Rhat[class_label]
                if abs(delta) > 1e-9:
                    E.prune_below(1e-6)
                    for key in list(E.store.keys()):
                        ev = E.get(*key)
                        if ev != 0: V.deposit(key[0], key[1], delta * ev)
                Rhat[class_label] += 0.15 * delta
                recent_output_fired = False

            t += 1

        # quiet gap between samples
        for _ in range(SAMPLE_GAP):
            inp = rng3.uniform(0, 0.01, N)
            v_ = v * 0.90 + inp
            fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
            if len(f):
                for fi in f:
                    for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
            C.tick(); E.tick(); V.tick()
            v = np.maximum(v_, 0); v[fired] = 0
            refrac[fired] = 3; refrac[refrac > 0] -= 1
            t += 1

        if t % 40 == 0:
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

    # final readout: input-output bridge mass
    in_set = set(input_idx.tolist())
    out_set = set(output_idx.tolist())
    bridge_mass = 0.0
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if si in in_set and di in out_set:
            bridge_mass += max(V.get(si, di), 0)
        elif di in in_set and si in out_set:
            bridge_mass += max(V.get(di, si), 0)
    return bridge_mass


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 37: PLACEMENT EFFECT ON REAL SHD DATA")
    print("Real speech-derived spike data drives the input layer,")
    print("not synthetic periodic patterns. Segregated vs interleaved.")
    print("=" * 74)

    print("\nLoading real SHD samples...")
    all_stimuli, all_labels = load_shd_samples(
        n_samples=200, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=200)

    from collections import Counter
    counts = Counter(all_labels)
    top_classes = [c for c, _ in counts.most_common(2)]
    target_class, distractor_class = top_classes[0], top_classes[1]
    print("Using target class {} ({} samples) vs distractor class {} ({} samples)".format(
        target_class, counts[target_class], distractor_class, counts[distractor_class]))

    target_samples = [s for s, l in zip(all_stimuli, all_labels) if l == target_class][:8]
    distractor_samples = [s for s, l in zip(all_stimuli, all_labels) if l == distractor_class][:8]
    print("Using {} target + {} distractor real samples".format(
        len(target_samples), len(distractor_samples)))

    SEEDS = [0, 1, 2, 3, 4]
    strategies = [
        ("segregated", make_placement_segregated),
        ("interleaved", make_placement_interleaved),
    ]

    all_results = {}
    for strat_name, placement_fn in strategies:
        print("\n  {}".format(strat_name.upper()))
        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = placement_fn()
            bridge = run_life(coords, cids, s, target_samples, distractor_samples,
                              n_epochs=15)
            seed_results.append(bridge)
            print("    seed {}: bridge_mass={:.1f}  ({:.0f}s)".format(
                s, bridge, time.time() - t0))
        all_results[strat_name] = np.array(seed_results)

    print("\n" + "=" * 74)
    print("RESULTS: Input-Output Bridge Mass on REAL SHD Data")
    means = {}
    for strat_name, _ in strategies:
        vals = all_results[strat_name]
        means[strat_name] = vals.mean()
        print("  {:>12}: {:.1f} +/- {:.1f}".format(strat_name, vals.mean(), vals.std()))

    print("\n" + "=" * 74)
    seg = means.get("segregated", 0)
    inter = means.get("interleaved", 0)
    ratio = inter / max(seg, 1)
    print("VERDICT:")
    print("  Interleaved/Segregated ratio on REAL DATA: {:.2f}x".format(ratio))
    print("  (Compare to synthetic-pattern PLB result: ~4.0x at N=900)")
    if ratio > 1.5:
        print("  The placement effect SURVIVES on real speech-derived data.")
        print("  This closes the biggest gap in the paper -- the finding is")
        print("  not an artifact of synthetic, hand-designed stimulus.")
    else:
        print("  The effect is weak or absent on real data -- this is an")
        print("  honest, important negative result that must be reported.")
