"""Exp 25: Within-Type Blind Spot
Patterns sharing a cluster: does devaluing one bleed into the other?
(0,3) and (0,4) both involve C0. Both rewarded during acquisition.
At step 600: (0,4)->0 while (0,3) stays rewarded.
Does C0's shared structure cause interference?
Run: python src/exp25.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

# 4 patterns now: (0,3) rewarded, (0,4) rewarded, (1,2) neutral, (lone 2) punished
# After reversal: (0,3) stays rewarded, (0,4) -> 0
PATTERNS = [(0, 3), (0, 4), (1,), (2,)]
N, NC = 1000, 5
REV = 600
TOTAL = 1200

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

def rewards_pre(p):
    # (0,3)->+1, (0,4)->+1, (1)->0, (2)->-1
    return {0: 1.0, 1: 1.0, 2: 0.0, 3: -1.0}[p]

def rewards_post(p):
    # (0,3)->+1, (0,4)->0 (DEVALUED), (1)->0, (2)->-1
    return {0: 1.0, 1: 0.0, 2: 0.0, 3: -1.0}[p]

def run_arm(coords, cids, seed, arm_type):
    """arm_type: 'selective' (devalue 0,4 only) | 'full' (devalue both) | 'control' (no change)"""
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 10000); dst = rng2.integers(0, N, 10000)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(len(PATTERNS))
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 500

    for t in range(TOTAL):
        p = (t // 20) % len(PATTERNS)
        pat = PATTERNS[p]

        inp = rng3.uniform(0, 0.02, N)
        if (t % 20) < 5:
            for c in pat:
                inp[cids == c] += 0.5

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
                base = rewards_pre(p)
            else:
                if arm_type == "control":
                    base = rewards_pre(p)  # no change
                elif arm_type == "selective":
                    base = rewards_post(p)  # only (0,4) devalued
                else:  # full
                    # both (0,3) and (0,4) devalued
                    base = {0: 0.0, 1: 0.0, 2: 0.0, 3: -1.0}[p]

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

    # readouts: bridge masses for each association
    M = np.zeros((NC, NC), dtype=int)
    np.add.at(M, (cids[src], cids[dst]), 1)
    bridge_03 = M[0, 3] + M[3, 0]  # (0,3) — should stay strong
    bridge_04 = M[0, 4] + M[4, 0]  # (0,4) — devalued in selective arm
    c0_internal = M[0, 0]           # C0 self-wiring (shared resource)
    energy = D2[src, dst].sum()
    return bridge_03, bridge_04, c0_internal, energy

if __name__ == "__main__":
    print("=" * 70)
    print("EXP 25: WITHIN-TYPE BLIND SPOT")
    print("(0,3) and (0,4) share cluster 0. Both rewarded until step 600.")
    print("Then: selective=(0,4)->0; full=both->0; control=no change")
    print("Question: does devaluing (0,4) bleed into (0,3)?")
    print("=" * 70)

    SEEDS = [0, 1, 2, 3, 4]
    arms = [
        ("control",   "control"),
        ("selective",  "selective"),
        ("full_deval", "full"),
    ]

    all_results = {}
    for arm_name, arm_code in arms:
        results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_salt_clustered()
            b03, b04, c0int, energy = run_arm(coords, cids, s, arm_code)
            results.append((b03, b04, c0int, energy))
            print("  {} seed {}: bridge(0,3)={} bridge(0,4)={} C0_self={} e={:,.0f} ({:.0f}s)".format(
                arm_name, s, b03, b04, c0int, energy, time.time() - t0))
        all_results[arm_name] = np.array(results, dtype=float)
        print("")

    print("=" * 70)
    print("SUMMARY")
    print("{:>12}  {:>14}  {:>14}  {:>10}  {:>10}".format(
        "arm", "bridge(0,3)", "bridge(0,4)", "C0_self", "energy"))
    print("-" * 65)
    for arm_name, _ in arms:
        A = all_results[arm_name]
        print("{:>12}  {:>6.0f}+/-{:<5.0f}  {:>6.0f}+/-{:<5.0f}  {:>5.0f}+/-{:<4.0f}  {:>10.0f}".format(
            arm_name,
            A[:, 0].mean(), A[:, 0].std(),
            A[:, 1].mean(), A[:, 1].std(),
            A[:, 2].mean(), A[:, 2].std(),
            A[:, 3].mean()))

    print("")
    print("=" * 70)
    print("THE BLIND SPOT TEST:")
    ctrl = all_results["control"]
    sel = all_results["selective"]
    full = all_results["full_deval"]

    bleed = (ctrl[:, 0].mean() - sel[:, 0].mean()) / ctrl[:, 0].mean() * 100
    target = (ctrl[:, 1].mean() - sel[:, 1].mean()) / ctrl[:, 1].mean() * 100

    print("  Selective devaluation of (0,4):")
    print("    Target (0,4) suppression: {:.0f}% (should be large)".format(target))
    print("    Collateral (0,3) damage:  {:.0f}% (the BLIND SPOT)".format(bleed))
    print("")
    if bleed > 15:
        print("  VERDICT: BLIND SPOT CONFIRMED — devaluing one association")
        print("  that shares a cluster damages its innocent neighbor by {:.0f}%".format(bleed))
    elif bleed > 5:
        print("  VERDICT: MODERATE INTERFERENCE — {:.0f}% bleed detected".format(bleed))
    else:
        print("  VERDICT: CLEAN SEPARATION — within-type devaluation does not")
        print("  significantly damage the preserved association")
    print("")
    print("Bring this back to Grok.")
