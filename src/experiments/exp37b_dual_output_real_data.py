"""Exp 37b: Placement effect on REAL SHD data -- FIXED task design.
Exp 37 proved the mechanism works powerfully on real data (bridge mass
swung -125 to +409 across just 6 samples) but ended at exactly 0.0
because ONE shared output cluster got conflicting reward from both
classes, causing oscillation that averaged to zero over a long run.

FIX: two SEPARATE, dedicated output clusters -- one per class -- each
with its OWN reward stream. This mirrors the (0,3)/(1,4) dual-pattern
structure that has worked in every other successful experiment in this
project. Target trials only ever reward/punish the target-dedicated
cluster's connections; distractor trials only touch the distractor-
dedicated cluster. No shared pool, no tug-of-war.

Run: python src/experiments/exp37b_dual_output_real_data.py
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
N_OUTPUT_A = 30  # dedicated to target class
N_OUTPUT_B = 30  # dedicated to distractor class
N = N_INPUT + N_OUTPUT_A + N_OUTPUT_B
INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE = 0, 1, 2


def make_placement_segregated():
    rng = np.random.default_rng(11)
    coords, cids = [], []
    for _ in range(N_INPUT):
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 8)
        coords.append([5 + r*np.cos(th), 5 + r*np.sin(th)])
        cids.append(INPUT_TYPE)
    for _ in range(N_OUTPUT_A):
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 6)
        coords.append([35 + r*np.cos(th), 30 + r*np.sin(th)])
        cids.append(OUTPUT_A_TYPE)
    for _ in range(N_OUTPUT_B):
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 6)
        coords.append([35 + r*np.cos(th), 40 + r*np.sin(th)])
        cids.append(OUTPUT_B_TYPE)
    return np.array(coords), np.array(cids)


def make_placement_interleaved():
    rng = np.random.default_rng(11)
    coords, cids = [], []
    all_types = ([INPUT_TYPE] * N_INPUT + [OUTPUT_A_TYPE] * N_OUTPUT_A +
                 [OUTPUT_B_TYPE] * N_OUTPUT_B)
    rng.shuffle(all_types)
    for t_ in all_types:
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 10)
        coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)])
        cids.append(t_)
    return np.array(coords), np.array(cids)


def run_life(coords, cids, seed, target_samples, distractor_samples,
             n_epochs=15, input_gain=0.35):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(2)  # 0=target/OutputA stream, 1=distractor/OutputB stream
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 200

    input_idx = np.where(cids == INPUT_TYPE)[0]
    output_a_idx = np.where(cids == OUTPUT_A_TYPE)[0]
    output_b_idx = np.where(cids == OUTPUT_B_TYPE)[0]
    out_a_set = set(output_a_idx.tolist())
    out_b_set = set(output_b_idx.tolist())

    all_samples = []
    for _ in range(n_epochs):
        combined = [(s, 0) for s in target_samples] + [(s, 1) for s in distractor_samples]
        rng3.shuffle(combined)
        all_samples.extend(combined)

    t = 0
    for sample_arr, class_label in all_samples:
        T_sample = sample_arr.shape[0]
        recent_a_fired = False
        recent_b_fired = False

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
                fs = set(int(x) for x in f)
                if fs & out_a_set: recent_a_fired = True
                if fs & out_b_set: recent_b_fired = True

            # dual, INDEPENDENT reward streams -- no shared pool
            if t % 20 == 6:
                stream = class_label  # 0 for target, 1 for distractor
                correct_fired = recent_a_fired if class_label == 0 else recent_b_fired
                base = 1.0 if correct_fired else 0.0
                delta = base - Rhat[stream]
                if abs(delta) > 1e-9:
                    E.prune_below(1e-6)
                    for key in list(E.store.keys()):
                        ev = E.get(*key)
                        if ev != 0: V.deposit(key[0], key[1], delta * ev)
                Rhat[stream] += 0.15 * delta
                recent_a_fired = False
                recent_b_fired = False

            # REWIRE CHECK INSIDE THE STEP LOOP -- this is where every other
            # experiment in the project puts it. When it sat outside the loop
            # (evaluated once per ~215-step sample), t%40==0 hit only by
            # coincidence: 1 rewire per 8 samples, measured empirically.
            # Result: src/dst kept the original random edges forever while
            # V decayed to ~0 on them over 50K+ steps -- learning was real
            # but never migrated into the edges being measured.
            if (t + 1) % 40 == 0:
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

            t += 1

        for _ in range(15):  # quiet gap
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

    in_set = set(input_idx.tolist())
    bridge_a = 0.0
    bridge_b = 0.0
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if si in in_set and di in out_a_set: bridge_a += max(V.get(si, di), 0)
        elif di in in_set and si in out_a_set: bridge_a += max(V.get(di, si), 0)
        if si in in_set and di in out_b_set: bridge_b += max(V.get(si, di), 0)
        elif di in in_set and si in out_b_set: bridge_b += max(V.get(di, si), 0)
    return bridge_a, bridge_b


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 37b: DUAL-OUTPUT REAL SHD DATA -- fixed task design")
    print("Separate output clusters per class, no shared-pool reward conflict")
    print("=" * 74)

    print("\nLoading real SHD samples...")
    all_stimuli, all_labels = load_shd_samples(
        n_samples=200, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=200)
    from collections import Counter
    counts = Counter(all_labels)
    top_classes = [c for c, _ in counts.most_common(2)]
    target_class, distractor_class = top_classes[0], top_classes[1]
    target_samples = [s for s, l in zip(all_stimuli, all_labels) if l == target_class][:8]
    distractor_samples = [s for s, l in zip(all_stimuli, all_labels) if l == distractor_class][:8]
    print("Target class {} ({} samples), distractor class {} ({} samples)".format(
        target_class, len(target_samples), distractor_class, len(distractor_samples)))

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
            ba, bb = run_life(coords, cids, s, target_samples, distractor_samples, n_epochs=15)
            seed_results.append((ba, bb))
            print("    seed {}: bridge_A(target)={:.1f}  bridge_B(distractor)={:.1f}  ({:.0f}s)".format(
                s, ba, bb, time.time() - t0))
        all_results[strat_name] = np.array(seed_results)

    print("\n" + "=" * 74)
    print("RESULTS: Dual Discriminative Bridge Mass on REAL SHD Data")
    means = {}
    for strat_name, _ in strategies:
        vals = all_results[strat_name]
        total = vals.sum(axis=1)
        means[strat_name] = total.mean()
        print("  {:>12}: bridge_A={:.1f}±{:.1f}  bridge_B={:.1f}±{:.1f}  total={:.1f}±{:.1f}".format(
            strat_name, vals[:, 0].mean(), vals[:, 0].std(),
            vals[:, 1].mean(), vals[:, 1].std(), total.mean(), total.std()))

    print("\n" + "=" * 74)
    seg = means.get("segregated", 0)
    inter = means.get("interleaved", 0)
    ratio = inter / max(seg, 1)
    print("VERDICT:")
    print("  Interleaved/Segregated ratio on REAL DATA: {:.2f}x".format(ratio))
    print("  (Compare to synthetic-pattern PLB result: ~4.0x at N=900)")
    if ratio > 1.5:
        print("  The placement effect SURVIVES on real speech-derived data.")
        print("  This closes the biggest gap in the paper.")
    else:
        print("  Effect weak/absent on real data with the corrected task design --")
        print("  now a trustworthy result, not a task-design artifact.")
