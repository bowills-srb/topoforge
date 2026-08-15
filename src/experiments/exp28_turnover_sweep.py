"""Exp 28: Turnover Sweep — how many slots must be contested to break the empire?
Sweep swap count from 500 (5%) to 5000 (50%) of the edge budget.
If persistence is topological, high turnover should finally produce crossover.
5 seeds per rate. Reversal at 600.
Run: python src/exp28_turnover_sweep.py
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
N_EDGES = 10000

SWAP_VALUES = [500, 1000, 2000, 3000, 4000, 5000]

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

def run_life(coords, cids, seed, swap_count):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, N_EDGES); dst = rng2.integers(0, N, N_EDGES)
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
    crossover_step = None
    old_at_rev = 0
    peak_old = 0

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
            cold = np.argsort(sc)[:swap_count]
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
                        actual_swap = min(len(cold), len(ci))
                        order = np.argsort(score)[::-1][:actual_swap]
                        n2 = min(len(cold), len(order))
                        src[cold[:n2]] = ci[order[:n2]]; dst[cold[:n2]] = cj[order[:n2]]
                        out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

            M = np.zeros((NC, NC), dtype=int)
            np.add.at(M, (cids[src], cids[dst]), 1)
            old = M[0, 3] + M[3, 0]; new_val = M[1, 4] + M[4, 1]
            if old > peak_old: peak_old = old
            if t + 1 == REV: old_at_rev = old
            if crossover_step is None and t >= REV and new_val > old:
                crossover_step = t + 1

    M = np.zeros((NC, NC), dtype=int)
    np.add.at(M, (cids[src], cids[dst]), 1)
    final_old = M[0, 3] + M[3, 0]
    final_new = M[1, 4] + M[4, 1]
    sup = M.sum() - np.trace(M) - final_old - final_new
    energy = D2[src, dst].sum()
    return peak_old, old_at_rev, final_old, final_new, sup, energy, crossover_step

if __name__ == "__main__":
    print("=" * 74)
    print("EXP 28: TURNOVER SWEEP — how many slots to break the empire?")
    print("6 swap rates x 5 seeds. Acquire 0-600, reverse 600-1200.")
    print("Swap counts: " + ", ".join(
        ["{}({:.0f}%)".format(s, s/N_EDGES*100) for s in SWAP_VALUES]))
    print("=" * 74)
    SEEDS = [0, 1, 2, 3, 4]
    all_data = {}
    for sw in SWAP_VALUES:
        results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_salt_clustered()
            peak, at_rev, f_old, f_new, sup, energy, cross = run_life(
                coords, cids, s, sw)
            results.append((peak, at_rev, f_old, f_new, sup, energy, cross))
            cx = str(cross) if cross else "never"
            print("  swap={} seed {}: peak={} @rev={} old={} new={} cross={} ({:.0f}s)".format(
                sw, s, peak, at_rev, f_old, f_new, cx, time.time() - t0))
        all_data[sw] = results
        A = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in results], dtype=float)
        crosses = [r[6] for r in results]
        valid = [c for c in crosses if c is not None]
        cx_pct = len(valid) * 100 // len(SEEDS)
        cx_str = "{:.0f}".format(np.mean(valid)) if valid else "never"
        print("  >> peak={:.0f} @rev={:.0f} old={:.0f} new={:.0f} cross={}% ({})\n".format(
            A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(), A[:, 3].mean(), cx_pct, cx_str))

    print("=" * 74)
    print("THE PLASTICITY SURFACE")
    print("{:>6} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8}".format(
        "swap", "%", "peak", "@rev", "f_old", "f_new", "cross%", "cross@"))
    print("-" * 62)
    for sw in SWAP_VALUES:
        R = all_data[sw]
        A = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in R], dtype=float)
        crosses = [r[6] for r in R]
        valid = [c for c in crosses if c is not None]
        cx_pct = len(valid) * 100 // len(SEEDS)
        cx_str = "{:.0f}".format(np.mean(valid)) if valid else "never"
        print("{:>6} {:>4.0f}% {:>6.0f} {:>6.0f} {:>6.0f} {:>6.0f} {:>6}% {:>8}".format(
            sw, sw/N_EDGES*100, A[:, 0].mean(), A[:, 1].mean(),
            A[:, 2].mean(), A[:, 3].mean(), cx_pct, cx_str))

    print("")
    print("=" * 74)
    print("WHAT THIS TELLS US:")
    print("  If crossover appears at high turnover: the empire's persistence")
    print("  IS about occupying slots, and turnover rate is the real plasticity")
    print("  dial — the one V-decay and C-weight couldn't be.")
    print("")
    print("  If crossover NEVER appears even at 50% turnover: the empire's")
    print("  persistence is even deeper — the VALUE RANKING itself protects")
    print("  old edges regardless of how many slots are contested, because")
    print("  old edges still outscore new candidates on accumulated V.")
    print("")
    print("  Either answer completes the persistence theory.")
