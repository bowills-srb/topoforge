"""Exp 27: C-Weight Sweep — how thick should the granite be?
Sweep the correlation basement (0.01*C term) from 0 to 0.10.
At C=0: pure value economy, no structural floor.
At C=0.10: thick basement, maximum persistence.
5 seeds per weight. Reversal at 600.
Run: python src/exp27_cweight_sweep.py
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

CWEIGHTS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

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

def run_life(coords, cids, seed, cweight):
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
    old_at_rev = 0

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
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0: V.deposit(key[0], key[1], delta * ev)
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
                    score = (vp + cweight * cp) / (1 + 0.05 * dd)
                    pos = score > 0
                    ci, cj, score = ci[pos], cj[pos], score[pos]
                    if len(ci) > 0:
                        order = np.argsort(score)[::-1][:len(cold)]
                        n2 = min(len(cold), len(order))
                        src[cold[:n2]] = ci[order[:n2]]; dst[cold[:n2]] = cj[order[:n2]]
                        out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

            M = np.zeros((NC, NC), dtype=int)
            np.add.at(M, (cids[src], cids[dst]), 1)
            old = M[0, 3] + M[3, 0]; new_val = M[1, 4] + M[4, 1]
            if t + 1 == REV: old_at_rev = old
            if crossover_step is None and t >= REV and new_val > old:
                crossover_step = t + 1

    M = np.zeros((NC, NC), dtype=int)
    np.add.at(M, (cids[src], cids[dst]), 1)
    final_old = M[0, 3] + M[3, 0]
    final_new = M[1, 4] + M[4, 1]
    sup = M.sum() - np.trace(M) - final_old - final_new
    energy = D2[src, dst].sum()
    return old_at_rev, final_old, final_new, sup, energy, crossover_step

if __name__ == "__main__":
    print("=" * 74)
    print("EXP 27: C-WEIGHT SWEEP — how thick is the granite?")
    print("7 C-weights x 5 seeds. Acquire 0-600, reverse 600-1200.")
    print("C=0: pure value economy. C=0.10: thick structural basement.")
    print("=" * 74)
    SEEDS = [0, 1, 2, 3, 4]
    all_data = {}
    for cw in CWEIGHTS:
        results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_salt_clustered()
            at_rev, f_old, f_new, sup, energy, cross = run_life(coords, cids, s, cw)
            results.append((at_rev, f_old, f_new, sup, energy, cross))
            cx = str(cross) if cross else "never"
            print("  C={} seed {}: @rev={} old={} new={} sup={} cross={} ({:.0f}s)".format(
                cw, s, at_rev, f_old, f_new, sup, cx, time.time() - t0))
        all_data[cw] = results
        A = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in results], dtype=float)
        crosses = [r[5] for r in results]
        valid = [c for c in crosses if c is not None]
        cx_pct = len(valid) * 100 // len(SEEDS)
        cx_str = "{:.0f}".format(np.mean(valid)) if valid else "never"
        print("  >> @rev={:.0f} old={:.0f} new={:.0f} cross={}% ({})\n".format(
            A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(), cx_pct, cx_str))

    print("=" * 74)
    print("THE GRANITE SPECTRUM")
    print("{:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "C-wt", "@rev", "fin_old", "fin_new", "supstit", "cross%", "cross@"))
    print("-" * 62)
    for cw in CWEIGHTS:
        R = all_data[cw]
        A = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in R], dtype=float)
        crosses = [r[5] for r in R]
        valid = [c for c in crosses if c is not None]
        cx_pct = len(valid) * 100 // len(SEEDS)
        cx_str = "{:.0f}".format(np.mean(valid)) if valid else "never"
        print("{:>7} {:>7.0f} {:>7.0f} {:>7.0f} {:>7.0f} {:>7}% {:>8}".format(
            cw, A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(),
            A[:, 3].mean(), cx_pct, cx_str))

    print("")
    print("=" * 74)
    print("THE DECISIVE ROW: C=0.0 (pure value, no basement)")
    R0 = all_data[0.0]
    A0 = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in R0], dtype=float)
    crosses0 = [r[5] for r in R0]
    valid0 = [c for c in crosses0 if c is not None]
    if valid0:
        print("  CROSSOVER ACHIEVED at C=0: mean step {:.0f}".format(np.mean(valid0)))
        print("  -> The granite IS the 0.01*C term. Remove it and the empire falls.")
        print("  -> This constant is the pencil-vs-chisel dial.")
    else:
        print("  Even at C=0, no crossover.")
        print("  -> The persistence comes from somewhere deeper than the C-weight.")
        print("  -> Structural inertia of existing edges alone holds the empire.")
    print("")
    print("For Adept: this table IS the durability spec sheet.")
