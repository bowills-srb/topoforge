"""Experiment 22: Confrontation Dreams - can facing the old world clear sunk cost?
Four arms x 5 seeds. Run: python src/exp22.py
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
DREAM_START, DREAM_END = 600, 750
TOTAL = 1400


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


def make_salt_gas():
    rng = np.random.default_rng(11)
    coords = rng.uniform(0, 40, size=(N, 2))
    cids = np.repeat(np.arange(NC), 200)
    rng.shuffle(cids)
    return coords, cids


def run_arm(coords, cids, seed, arm_type):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 10000)
    dst = rng2.integers(0, N, 10000)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N)
    refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95)
    E = SparsePairState(0.90)
    V = SparsePairState(0.999)
    Rhat = np.zeros(3)
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    out_t = [[] for _ in range(N)]
    out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b)
        out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 500
    crossover_step = None

    for t in range(TOTAL):
        p = (t // 20) % 3
        in_dream = (arm_type in ("replace", "confront", "confront_gas")
                    and DREAM_START <= t < DREAM_END)

        # --- STIMULUS ---
        if in_dream:
            if arm_type == "replace":
                inp = rng3.uniform(0, 0.02, N)
                new_pats = [(1, 4), (0, 3), (2,)]
                pat = new_pats[p]
                if (t % 20) < 5:
                    for c in pat:
                        inp[cids == c] += 0.35
            else:
                cycle = (t - DREAM_START) // 20
                if cycle % 2 == 0:
                    inp = rng3.uniform(0, 0.02, N)
                    pat = PATTERNS[p]
                    if (t % 20) < 5:
                        for c in pat:
                            inp[cids == c] += 0.35
                else:
                    inp = rng3.uniform(0, 0.02, N)
                    new_pats = [(1, 4), (0, 3), (2,)]
                    pat = new_pats[p]
                    if (t % 20) < 5:
                        for c in pat:
                            inp[cids == c] += 0.35
        elif t < REV:
            inp = rng3.uniform(0, 0.02, N)
            if (t % 20) < 5:
                for c in PATTERNS[p]:
                    inp[cids == c] += 0.5
        else:
            inp = rng3.uniform(0, 0.02, N)
            new_pats = [(1, 4), (0, 3), (2,)]
            pat = new_pats[p]
            if (t % 20) < 5:
                for c in pat:
                    inp[cids == c] += 0.5

        # --- LIF ---
        v_ = v * 0.90 + inp
        fired = (v_ >= 1.0) & (refrac == 0)
        f = np.where(fired)[0]
        if len(f):
            for fi in f:
                for ti, wi in zip(out_t[fi], out_w[fi]):
                    v_[ti] += wi
        C.tick()
        E.tick()
        V.tick()
        if len(f):
            fs = set(int(x) for x in f)
            for i in f:
                i = int(i)
                for j in nbr[i]:
                    if int(j) in fs:
                        C.deposit(i, int(j), 1.0)
                        E.deposit(i, int(j), 1.0)
        v = np.maximum(v_, 0)
        v[fired] = 0
        refrac[fired] = 3
        refrac[refrac > 0] -= 1

        # --- REWARD ---
        if (t % 20) == 6:
            if in_dream and arm_type in ("confront", "confront_gas"):
                cycle = (t - DREAM_START) // 20
                if cycle % 2 == 0:
                    base = {0: 0.0, 1: 0.0, 2: -1.0}[p]
                else:
                    base = {0: 0.0, 1: 1.0, 2: -1.0}[p]
            elif t < REV:
                base = {0: 1.0, 1: 0.0, 2: -1.0}[p]
            else:
                base = {0: 0.0, 1: 1.0, 2: -1.0}[p]

            delta = base - Rhat[p]
            if abs(delta) > 1e-9:
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0:
                        V.deposit(key[0], key[1], delta * ev)
            Rhat[p] += 0.15 * delta

        # --- REWIRE ---
        if (t + 1) % 40 == 0:
            C.prune_below(1e-6)
            sc = np.array([V.get(int(src[k]), int(dst[k]))
                           for k in range(len(src))])
            cold = np.argsort(sc)[:swap]
            ci, cj, _ = C.get_arrays()
            if len(ci) > 0:
                keep2 = ci != cj
                ci, cj = ci[keep2], cj[keep2]
                ex = set(zip(src.tolist(), dst.tolist()))
                mk = np.array([(int(a), int(b)) not in ex
                               for a, b in zip(ci, cj)])
                ci, cj = ci[mk], cj[mk]
                if len(ci) > 0:
                    dd = D2[ci, cj]
                    vp = np.maximum(np.array([V.get(int(a), int(b))
                                              for a, b in zip(ci, cj)]), 0)
                    cp = np.array([C.get(int(a), int(b))
                                   for a, b in zip(ci, cj)])
                    score = (vp + 0.01 * cp) / (1 + 0.05 * dd)
                    pos = score > 0
                    ci, cj, score = ci[pos], cj[pos], score[pos]
                    if len(ci) > 0:
                        order = np.argsort(score)[::-1][:len(cold)]
                        n2 = min(len(cold), len(order))
                        src[cold[:n2]] = ci[order[:n2]]
                        dst[cold[:n2]] = cj[order[:n2]]
                        out_t = [[] for _ in range(N)]
                        out_w = [[] for _ in range(N)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b)
                            out_w[s2].append(-0.60 if inhib[s2] else 0.30)

            if t >= REV:
                M = np.zeros((NC, NC), dtype=int)
                np.add.at(M, (cids[src], cids[dst]), 1)
                old = M[0, 3] + M[3, 0]
                new_val = M[1, 4] + M[4, 1]
                if crossover_step is None and new_val > old:
                    crossover_step = t + 1

    M = np.zeros((NC, NC), dtype=int)
    np.add.at(M, (cids[src], cids[dst]), 1)
    old = M[0, 3] + M[3, 0]
    new_val = M[1, 4] + M[4, 1]
    sup = M.sum() - np.trace(M) - old - new_val
    energy = D2[src, dst].sum()
    return old, new_val, sup, energy, crossover_step


if __name__ == "__main__":
    print("=" * 70)
    print("EXP 22: CONFRONTATION DREAMS")
    print("4 arms x 5 seeds. Reversal at 600. Dreams 600-750.")
    print("Arm 3: old-at-zero + new-at-reward (confrontation)")
    print("=" * 70)

    SEEDS = [0, 1, 2, 3, 4]
    arms = [
        ("1:real_only", "salt_clust", "real"),
        ("2:replace", "salt_clust", "replace"),
        ("3:confront", "salt_clust", "confront"),
        ("4:confront+gas", "salt_gas", "confront_gas"),
    ]

    all_results = {}
    for arm_name, body_type, arm_code in arms:
        results = []
        for s in SEEDS:
            t0 = time.time()
            if body_type == "salt_clust":
                coords, cids = make_salt_clustered()
            else:
                coords, cids = make_salt_gas()
            old, nv, sup, energy, cross = run_arm(coords, cids, s, arm_code)
            results.append((old, nv, sup, energy, cross))
            cx = str(cross) if cross else "never"
            elapsed = time.time() - t0
            print("  {} seed {}: old={} new={} sup={} e={:,.0f} cross={} ({:.0f}s)".format(
                arm_name, s, old, nv, sup, energy, cx, elapsed))
        all_results[arm_name] = results
        crosses = [r[4] for r in results]
        valid = [c for c in crosses if c is not None]
        if valid:
            cx_str = "{:.0f}".format(np.mean(valid))
        else:
            cx_str = "never"
        ct = len(valid)
        print("  >> crossover: {}/{} seeds, mean step: {}".format(
            ct, len(SEEDS), cx_str))
        print("")

    print("=" * 70)
    print("SUMMARY")
    print("{:>20}  {:>12}  {:>12}  {:>12}  {:>12}  {:>8}".format(
        "arm", "old", "new", "superstit", "energy", "cross"))
    print("-" * 82)
    for arm_name, _, _ in arms:
        R = all_results[arm_name]
        A = np.array([(r[0], r[1], r[2], r[3]) for r in R], dtype=float)
        crosses = [r[4] for r in R]
        valid = [c for c in crosses if c is not None]
        if valid:
            cx_str = "{:.0f}".format(np.mean(valid))
        else:
            cx_str = "never"
        ct = len(valid)
        print("{:>20}  {:>5.0f}+/-{:<4.0f}  {:>5.0f}+/-{:<4.0f}  {:>5.0f}+/-{:<4.0f}  {:>10.0f}  {}/{}={}".format(
            arm_name,
            A[:, 0].mean(), A[:, 0].std(),
            A[:, 1].mean(), A[:, 1].std(),
            A[:, 2].mean(), A[:, 2].std(),
            A[:, 3].mean(),
            ct, len(SEEDS), cx_str))

    print("")
    print("=" * 70)
    print("The decisive comparison:")
    print("  Arms 1,2: should replicate Exp 21 (never crosses)")
    print("  Arm 3 vs 1,2: does CONFRONTATION demolish the empire?")
    print("  Arm 4 vs 3: does gas geometry accelerate it?")
    print("")
    print("Bring this back to Grok.")
