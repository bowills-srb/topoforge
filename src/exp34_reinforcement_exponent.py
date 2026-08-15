"""Exp 34: Reinforcement Exponent Sweep (Physarum-inspired)
Tero et al. 2010 Science: tube growth ~ flow^gamma. Low gamma = redundant
fault-tolerant network. High gamma = winner-take-all single path.
We test the equivalent: V-deposit ~ sign(delta)*|delta|^gamma instead of
linear delta. Does this control redundancy vs efficiency in OUR architecture?
5 gamma values x 5 seeds. Run: python src/exp34_reinforcement_exponent.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

PATTERNS = [(0, 3), (1, 4), (2,)]
N, NC = 1000, 5
REV = 600
TOTAL = 1200

GAMMAS = [0.5, 0.75, 1.0, 1.5, 2.0]

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

def signed_power(x, gamma):
    """sign(x) * |x|^gamma -- the Physarum-inspired nonlinearity."""
    return np.sign(x) * (np.abs(x) ** gamma)

def run_life(coords, cids, seed, gamma):
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
    crossover_step = None

    for t in range(TOTAL):
        p = (t // 20) % 3
        if t < REV:
            inp = rng3.uniform(0, 0.02, N)
            if (t % 20) < 5:
                for c in PATTERNS[p]: inp[cids == c] += 0.5
        else:
            inp = rng3.uniform(0, 0.02, N)
            new_pats = [(1, 4), (0, 3), (2,)]
            pat = new_pats[p]
            if (t % 20) < 5:
                for c in pat: inp[cids == c] += 0.5

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
            if t < REV:
                base = {0: 1.0, 1: 0.0, 2: -1.0}[p]
            else:
                base = {0: 0.0, 1: 1.0, 2: -1.0}[p]
            delta = base - Rhat[p]
            if abs(delta) > 1e-9:
                # PHYSARUM-INSPIRED NONLINEARITY: reinforcement ~ |delta|^gamma
                delta_shaped = signed_power(delta, gamma)
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0: V.deposit(key[0], key[1], delta_shaped * ev)
            Rhat[p] += 0.15 * delta

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

            if t >= REV:
                M = np.zeros((NC, NC), dtype=int)
                np.add.at(M, (cids[src], cids[dst]), 1)
                old = M[0, 3] + M[3, 0]
                new_val = M[1, 4] + M[4, 1]
                if crossover_step is None and new_val > old:
                    crossover_step = t + 1

    # readouts: taught mass AND redundancy (distinct edges serving the champion association)
    M = np.zeros((NC, NC), dtype=int)
    np.add.at(M, (cids[src], cids[dst]), 1)
    old = M[0, 3] + M[3, 0]
    new_val = M[1, 4] + M[4, 1]
    sup = M.sum() - np.trace(M) - old - new_val
    energy = D2[src, dst].sum()

    # redundancy metric: how many DISTINCT src-neurons in cluster 0 have
    # at least one edge to cluster 3 with meaningful V (path diversity)
    # vs total edges (0,3) -- ratio near 1 = many distinct paths (redundant)
    # ratio near 0 = few neurons carry all the traffic (winner-take-all)
    c0_neurons_in_03 = set()
    total_03_edges = 0
    for si, di in zip(src, dst):
        if cids[si] == 0 and cids[di] == 3:
            c0_neurons_in_03.add(int(si))
            total_03_edges += 1
        elif cids[si] == 3 and cids[di] == 0:
            c0_neurons_in_03.add(int(di))
            total_03_edges += 1
    redundancy = len(c0_neurons_in_03) / max(total_03_edges, 1)

    return old, new_val, sup, energy, redundancy, len(c0_neurons_in_03), total_03_edges, crossover_step

if __name__ == "__main__":
    print("=" * 74)
    print("EXP 34: REINFORCEMENT EXPONENT SWEEP (Physarum-inspired)")
    print("Tero et al. 2010 Science: tube growth ~ flow^gamma")
    print("Testing: V-deposit ~ sign(delta)*|delta|^gamma")
    print("Low gamma = redundant paths (fault tolerant, less efficient)")
    print("High gamma = winner-take-all (efficient, fragile)")
    print("5 gamma values x 5 seeds. Acquire 0-600, reverse 600-1200.")
    print("=" * 74)

    SEEDS = [0, 1, 2, 3, 4]
    all_data = {}

    for gamma in GAMMAS:
        results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_salt_clustered()
            old, new_val, sup, energy, redun, n_paths, n_edges, cross = run_life(
                coords, cids, s, gamma)
            results.append((old, new_val, sup, energy, redun, n_paths, n_edges, cross))
            cx = str(cross) if cross else "never"
            print("  gamma={} seed {}: old={} new={} redun={:.2f} (paths={},edges={}) cross={} ({:.0f}s)".format(
                gamma, s, old, new_val, redun, n_paths, n_edges, cx, time.time() - t0))
        all_data[gamma] = results
        A = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in results], dtype=float)
        crosses = [r[7] for r in results]
        valid = [c for c in crosses if c is not None]
        cx_pct = len(valid) * 100 // len(SEEDS)
        print("  >> old={:.0f} new={:.0f} redundancy={:.3f} paths={:.1f} cross={}%\n".format(
            A[:, 0].mean(), A[:, 1].mean(), A[:, 4].mean(), A[:, 5].mean(), cx_pct))

    print("=" * 74)
    print("THE REDUNDANCY SPECTRUM")
    print("{:>7} {:>8} {:>8} {:>10} {:>8} {:>10} {:>8}".format(
        "gamma", "old", "new", "redundancy", "paths", "energy", "cross%"))
    print("-" * 68)
    for gamma in GAMMAS:
        R = all_data[gamma]
        A = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in R], dtype=float)
        crosses = [r[7] for r in R]
        valid = [c for c in crosses if c is not None]
        cx_pct = len(valid) * 100 // len(SEEDS)
        print("{:>7} {:>8.0f} {:>8.0f} {:>10.3f} {:>8.1f} {:>10,.0f} {:>7}%".format(
            gamma, A[:, 0].mean(), A[:, 1].mean(), A[:, 4].mean(),
            A[:, 5].mean(), A[:, 3].mean(), cx_pct))

    print("")
    print("=" * 74)
    print("HOW TO READ REDUNDANCY:")
    print("  redundancy = (distinct neurons carrying the association) / (total edges)")
    print("  HIGH redundancy (~1.0): many different paths share the load")
    print("    -> fault tolerant, a single edge dying doesn't break the association")
    print("  LOW redundancy (~0.3-0.5): few neurons dominate, most edges pile onto them")
    print("    -> efficient but fragile, a single failure could be catastrophic")
    print("")
    print("  If low gamma -> high redundancy and high gamma -> low redundancy,")
    print("  this confirms the Physarum finding IN OUR ARCHITECTURE: the")
    print("  reinforcement nonlinearity is a real fault-tolerance/efficiency dial.")
    print("")
    print("  For Adept: a sensor with redundant internal pathways degrades")
    print("  gracefully if individual neurons/connections fail. This gamma")
    print("  parameter would be a literal reliability spec.")
