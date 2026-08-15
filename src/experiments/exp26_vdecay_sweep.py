"""Exp 26: V-Decay Sweep — mapping the retention/adaptability trade-off.
Sweep V decay from 0.990 to 0.9999 across 6 values.
For each: build empire (600 steps), reverse, measure adaptation speed.
5 seeds per decay rate. The output is the design surface customers need.
Run: python src/exp26_vdecay_sweep.py
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

DECAY_VALUES = [0.990, 0.993, 0.995, 0.997, 0.999, 0.9995]

def make_salt_clustered():
    rng = np.random.default_rng(11)
    centers = np.random.default_rng(42).uniform(5, 35, size=(5, 2))
    pts = []
    for cx, cy in centers:
        for _ in range(200):
            th, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 8)
            pts.append([cx + r * np.cos(th), cy + r * np.sin(th)])
    coords = np.array(pts)
    cids = np.repeat(np.arange(NC), 200)
    rng.shuffle(cids)
    return coords, cids

def run_life(coords, cids, seed, v_decay):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 10000); dst = rng2.integers(0, N, 10000)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95)
    E = SparsePairState(0.90)
    V = SparsePairState(v_decay)
    Rhat = np.zeros(3)
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 500
    crossover_step = None
    peak_old = 0
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
                    score = (vp + 0.01 * cp) / (1 + 0.05 * dd); pos = score > 0
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
            old = M[0, 3] + M[3, 0]
            new_val = M[1, 4] + M[4, 1]
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
    print("EXP 26: V-DECAY SWEEP — the retention/adaptability trade-off surface")
    print("6 decay rates x 5 seeds. Acquire 0-600, reverse 600-1200.")
    print("Half-lives: " + ", ".join(
        ["{}: {:.0f} steps".format(d, -1/np.log(d)) for d in DECAY_VALUES]))
    print("=" * 74)

    SEEDS = [0, 1, 2, 3, 4]
    all_data = {}

    for vd in DECAY_VALUES:
        hl = -1 / np.log(vd)
        results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_salt_clustered()
            peak, at_rev, f_old, f_new, sup, energy, cross = run_life(
                coords, cids, s, vd)
            results.append((peak, at_rev, f_old, f_new, sup, energy, cross))
            cx = str(cross) if cross else "never"
            print("  decay={} seed {}: peak={} @rev={} final_old={} final_new={} cross={} ({:.0f}s)".format(
                vd, s, peak, at_rev, f_old, f_new, cx, time.time() - t0))
        all_data[vd] = results
        crosses = [r[6] for r in results]
        valid = [c for c in crosses if c is not None]
        cx_pct = len(valid) * 100 // len(SEEDS)
        cx_str = "{:.0f}".format(np.mean(valid)) if valid else "never"
        A = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in results], dtype=float)
        print("  >> hl={:.0f} peak={:.0f} @rev={:.0f} final_old={:.0f} final_new={:.0f} cross={}% ({})\n".format(
            hl, A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean(), A[:, 3].mean(), cx_pct, cx_str))

    print("=" * 74)
    print("THE TRADE-OFF SURFACE")
    print("{:>7} {:>7} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8}".format(
        "decay", "hl", "peak", "@rev", "fin_old", "fin_new", "cross%", "cross@"))
    print("-" * 74)
    for vd in DECAY_VALUES:
        R = all_data[vd]
        A = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in R], dtype=float)
        crosses = [r[6] for r in R]
        valid = [c for c in crosses if c is not None]
        cx_pct = len(valid) * 100 // len(SEEDS)
        cx_str = "{:.0f}".format(np.mean(valid)) if valid else "never"
        hl = -1 / np.log(vd)
        print("{:>7} {:>5.0f}s {:>7.0f} {:>7.0f} {:>8.0f} {:>8.0f} {:>7}% {:>8}".format(
            vd, hl, A[:, 0].mean(), A[:, 1].mean(),
            A[:, 2].mean(), A[:, 3].mean(), cx_pct, cx_str))

    print("")
    print("=" * 74)
    print("HOW TO READ THIS:")
    print("  Fast decay (0.990, hl=100): small empires, easy reversal, fragile memory")
    print("  Slow decay (0.9995, hl=2000): huge empires, hard reversal, tenacious memory")
    print("  The SWEET SPOT: highest final_new with lowest residual fin_old")
    print("    = the decay rate where the system adapts fastest after a regime change")
    print("    while still building enough structure to be useful before the change.")
    print("")
    print("  For Adept (pump monitor, stable world, rare changes): slower decay")
    print("  For fleet sensor (changing drivers/loads frequently): faster decay")
    print("  This surface IS the product configuration guide.")
