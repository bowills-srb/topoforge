"""Exp 29: Spatial Consensus as Noise Filter (The Ant Colony)
After 600 steps of baseline learning, inject a persistent anomaly:
  Arm 1: no anomaly (control)
  Arm 2: COHERENT anomaly — 50 adjacent neurons get a small persistent bias
  Arm 3: RANDOM anomaly — 50 scattered neurons get the same bias
No reward on the anomaly. Purely local correlation dynamics.
Does the coherent anomaly write new structure where the random one can't?
Run: python src/exp29_ant_colony.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

PATTERNS = [(0, 3), (1, 4), (2,)]
N, NC = 1000, 5
ANOMALY_START = 600
TOTAL = 1000
ANOMALY_STRENGTH = 0.15
N_ANOMALY = 50

def make_salt_clustered():
    rng = np.random.default_rng(11)
    centers = np.random.default_rng(42).uniform(5, 35, size=(5, 2))
    pts = []
    for cx, cy in centers:
        for _ in range(200):
            th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 8)
            pts.append([cx + r*np.cos(th), cy + r*np.sin(th)])
    coords = np.array(pts)
    cids = np.repeat(np.arange(NC), 200)
    rng.shuffle(cids)
    return coords, cids

def pick_coherent_group(coords, n):
    """Pick n neurons that are spatially adjacent — a real physical anomaly."""
    rng = np.random.default_rng(77)
    center_idx = rng.integers(0, len(coords))
    dists = ((coords - coords[center_idx]) ** 2).sum(1)
    return np.argsort(dists)[:n]

def pick_random_group(n):
    """Pick n neurons scattered randomly — noise coincidence."""
    rng = np.random.default_rng(77)
    return rng.choice(N, size=n, replace=False)

def run_arm(coords, cids, seed, anomaly_type):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 10000); dst = rng2.integers(0, N, 10000)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(3)
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 500

    if anomaly_type == "coherent":
        anomaly_neurons = pick_coherent_group(coords, N_ANOMALY)
    elif anomaly_type == "random":
        anomaly_neurons = pick_random_group(N_ANOMALY)
    else:
        anomaly_neurons = np.array([], dtype=int)

    # snapshot edges involving anomaly neurons before and after
    def count_anomaly_edges(s, d, group):
        if len(group) == 0:
            return 0, 0
        gset = set(int(x) for x in group)
        internal = sum(1 for si, di in zip(s, d) if int(si) in gset and int(di) in gset)
        touching = sum(1 for si, di in zip(s, d) if int(si) in gset or int(di) in gset)
        return internal, touching

    edges_before_int, edges_before_touch = 0, 0

    for t in range(TOTAL):
        p = (t // 20) % 3
        inp = rng3.uniform(0, 0.02, N)
        if (t % 20) < 5:
            for c in PATTERNS[p]: inp[cids == c] += 0.5

        # inject anomaly: persistent small bias on affected neurons
        if t >= ANOMALY_START and len(anomaly_neurons) > 0:
            inp[anomaly_neurons] += ANOMALY_STRENGTH

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
                    if int(j) in fs: C.deposit(i, int(j), 1.0); E.deposit(i, int(j), 1.0)
        v = np.maximum(v_, 0); v[fired] = 0; refrac[fired] = 3; refrac[refrac > 0] -= 1

        if (t % 20) == 6:
            base = {0: 1.0, 1: 0.0, 2: -1.0}[p]
            delta = base - Rhat[p]
            if abs(delta) > 1e-9:
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0: V.deposit(key[0], key[1], delta * ev)
            Rhat[p] += 0.15 * delta

        if (t + 1) % 40 == 0:
            # snapshot at anomaly start
            if t + 1 == ANOMALY_START:
                edges_before_int, edges_before_touch = count_anomaly_edges(
                    src, dst, anomaly_neurons)

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
                    dd = D2[ci, cj]
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

    edges_after_int, edges_after_touch = count_anomaly_edges(
        src, dst, anomaly_neurons)

    # also measure: how many of the anomaly neurons' LOCAL co-firing pairs
    # accumulated meaningful C-stock during the anomaly window?
    new_c_pairs = 0
    if len(anomaly_neurons) > 0:
        aset = set(int(x) for x in anomaly_neurons)
        for i in anomaly_neurons:
            i = int(i)
            for j in nbr[i]:
                if int(j) in aset:
                    cval = C.get(i, int(j))
                    if cval > 0.1:
                        new_c_pairs += 1

    return (edges_before_int, edges_before_touch,
            edges_after_int, edges_after_touch,
            new_c_pairs)

if __name__ == "__main__":
    print("=" * 70)
    print("EXP 29: THE ANT COLONY — spatial consensus as noise filter")
    print("50 neurons get a persistent anomaly bias after step 600.")
    print("Coherent = adjacent cluster. Random = scattered. Control = none.")
    print("No reward on the anomaly. Pure correlation dynamics.")
    print("=" * 70)

    SEEDS = [0, 1, 2, 3, 4]
    arms = [
        ("control", "none"),
        ("coherent", "coherent"),
        ("random", "random"),
    ]

    all_results = {}
    for arm_name, atype in arms:
        results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_salt_clustered()
            bi, bt, ai, at_, nc = run_arm(coords, cids, s, atype)
            results.append((bi, bt, ai, at_, nc))
            d_int = ai - bi
            d_touch = at_ - bt
            print("  {} seed {}: edges_internal {}->{} ({:+d})  edges_touching {}->{} ({:+d})  C_pairs={} ({:.0f}s)".format(
                arm_name, s, bi, ai, d_int, bt, at_, d_touch, nc, time.time() - t0))
        all_results[arm_name] = np.array(results, dtype=float)
        A = all_results[arm_name]
        print("  >> delta_internal={:+.0f}  delta_touching={:+.0f}  C_pairs={:.0f}\n".format(
            (A[:, 2] - A[:, 0]).mean(),
            (A[:, 3] - A[:, 1]).mean(),
            A[:, 4].mean()))

    print("=" * 70)
    print("THE ANT COLONY VERDICT")
    print("{:>10} {:>16} {:>16} {:>10}".format(
        "arm", "delta_internal", "delta_touching", "C_pairs"))
    print("-" * 55)
    for arm_name, _ in arms:
        A = all_results[arm_name]
        di = (A[:, 2] - A[:, 0]).mean()
        dt = (A[:, 3] - A[:, 1]).mean()
        cp = A[:, 4].mean()
        print("{:>10} {:>+14.0f} {:>+14.0f} {:>10.0f}".format(arm_name, di, dt, cp))

    coh = all_results["coherent"]
    ran = all_results["random"]
    coh_di = (coh[:, 2] - coh[:, 0]).mean()
    ran_di = (ran[:, 2] - ran[:, 0]).mean()
    coh_cp = coh[:, 4].mean()
    ran_cp = ran[:, 4].mean()

    print("")
    print("=" * 70)
    if coh_di > ran_di + 5 and coh_cp > ran_cp * 1.5:
        print("  VERDICT: SPATIAL CONSENSUS CONFIRMED")
        print("  Coherent anomaly wrote new structure; random did not.")
        print("  The architecture is an ant colony: local agreement = signal,")
        print("  scattered coincidence = noise. Geometry filters for free.")
    elif coh_di > ran_di:
        print("  VERDICT: MODERATE CONSENSUS EFFECT")
        print("  Coherent anomaly wrote more structure than random,")
        print("  but the gap is modest. Geometry helps but doesn't dominate.")
    else:
        print("  VERDICT: NO CONSENSUS EFFECT")
        print("  Coherent and random anomalies produced similar structural change.")
        print("  Spatial coherence doesn't help — the filter is elsewhere.")
    print("")
    print("For Adept: if confirmed, the sensor doesn't need a volatility")
    print("detector as a separate mechanism. The spatial embedding IS the")
    print("detector. Real faults are spatially coherent; noise isn't.")
