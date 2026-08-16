"""Exp 35c: Diagnosing WHY perfect synchrony underperforms moderate jitter.
Hypothesis from Exp 35b: rigid zero-jitter determinism keeps reinforcing
the same narrow set of edges, while moderate jitter spreads deposits
across more distinct candidate edges, more of which survive rewire.

This directly measures, at each jitter level:
  - n_distinct_edges: how many unique (i,j) pairs ever received a deposit
  - concentration: what fraction of total V-mass sits in the TOP edge
  - top3_concentration: same, for the top 3 edges combined

If the hypothesis is right: jitter=0 should show FEWER distinct edges
and HIGHER concentration than jitter=2 (the peak).

Run: python src/experiments/exp35c_diversity_diagnostic.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

N, NC = 1000, 5
TOTAL = 800
CHANNEL_A, CHANNEL_B = 0, 1
FUSION_TARGET = 3
ITDP_SIGMA = 3.0
ITDP_AMPLITUDE = 1.0
COINCIDENCE_WINDOW = 8


def itdp_gaussian(delta_t):
    return ITDP_AMPLITUDE * np.exp(-(delta_t ** 2) / (2 * ITDP_SIGMA ** 2))


def make_placement_near():
    rng = np.random.default_rng(11)
    coords = []
    cids = []
    centers = {
        CHANNEL_A: [10, 10], CHANNEL_B: [12, 10],
        FUSION_TARGET: [11, 12], 2: [30, 10], 4: [30, 30],
    }
    for cid, (cx, cy) in centers.items():
        for _ in range(200):
            th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 3)
            coords.append([cx + r*np.cos(th), cy + r*np.sin(th)])
            cids.append(cid)
    return np.array(coords), np.array(cids)


def run_life_with_diagnostics(coords, cids, seed, jitter_max):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 8000); dst = rng2.integers(0, N, 8000)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    V = SparsePairState(0.999)
    C = SparsePairState(0.95)
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 400

    fire_times_A = []
    fire_times_B = []
    fusion_deposits = {}  # (i,j) -> cumulative raw deposit count, for diagnostics

    for t in range(TOTAL):
        inp = rng3.uniform(0, 0.02, N)
        a_fired_now = False
        b_fired_now = False

        if (t % 20) < 3:
            inp[cids == CHANNEL_A] += 0.5
            a_fired_now = True

        if (t % 20) < 3:
            jitter = rng3.integers(-jitter_max, jitter_max + 1) if jitter_max > 0 else 0
            if 0 <= (t + jitter) % 20 < 5:
                inp[cids == CHANNEL_B] += 0.5
                b_fired_now = True

        if a_fired_now:
            fire_times_A.append(t)
        if b_fired_now:
            fire_times_B.append(t)
        fire_times_A = [x for x in fire_times_A if t - x <= COINCIDENCE_WINDOW]
        fire_times_B = [x for x in fire_times_B if t - x <= COINCIDENCE_WINDOW]

        v_ = v * 0.90 + inp
        fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
        if len(f):
            for fi in f:
                for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
        V.tick(); C.tick()

        if len(f):
            fs = set(int(x) for x in f)
            for i in f:
                i = int(i)
                for j in nbr[i]:
                    if int(j) in fs:
                        C.deposit(i, j, 1.0)

        if fire_times_A and fire_times_B:
            best_delta = min(abs(a - b) for a in fire_times_A[-3:]
                             for b in fire_times_B[-3:])
            fusion_strength = itdp_gaussian(best_delta)
            if fusion_strength > 0.05 and len(f):
                fs = set(int(x) for x in f)
                for i in f:
                    i = int(i)
                    ci = cids[i]
                    if ci not in (CHANNEL_A, CHANNEL_B):
                        continue
                    for j in nbr[i]:
                        if int(j) in fs and cids[j] == FUSION_TARGET:
                            V.deposit(i, j, fusion_strength * 0.1)
                            key = (i, int(j))
                            fusion_deposits[key] = fusion_deposits.get(key, 0) + 1

        v = np.maximum(v_, 0); v[fired] = 0
        refrac[fired] = 3; refrac[refrac > 0] -= 1

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

    # final readouts
    a_idx = set(np.where(cids == CHANNEL_A)[0].tolist())
    b_idx = set(np.where(cids == CHANNEL_B)[0].tolist())
    t_idx = set(np.where(cids == FUSION_TARGET)[0].tolist())
    edge_final_v = {}
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if (si in a_idx or si in b_idx) and di in t_idx:
            vval = max(V.get(si, di), 0)
            if vval > 0:
                edge_final_v[(si, di)] = vval
        elif (di in a_idx or di in b_idx) and si in t_idx:
            vval = max(V.get(di, si), 0)
            if vval > 0:
                edge_final_v[(di, si)] = vval

    fused_mass = sum(edge_final_v.values())
    n_distinct_deposit_edges = len(fusion_deposits)  # how many EVER got a deposit
    n_distinct_surviving_edges = len(edge_final_v)    # how many have nonzero V NOW

    if fused_mass > 0:
        sorted_v = sorted(edge_final_v.values(), reverse=True)
        top1_concentration = sorted_v[0] / fused_mass
        top3_concentration = sum(sorted_v[:3]) / fused_mass
    else:
        top1_concentration = 0.0
        top3_concentration = 0.0

    return {
        "fused_mass": fused_mass,
        "n_distinct_deposit_edges": n_distinct_deposit_edges,
        "n_distinct_surviving_edges": n_distinct_surviving_edges,
        "top1_concentration": top1_concentration,
        "top3_concentration": top3_concentration,
    }


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 35c: DIAGNOSING THE ZERO-JITTER UNDERPERFORMANCE")
    print("Testing hypothesis: does rigid synchrony concentrate deposits")
    print("onto fewer edges, while moderate jitter spreads them wider?")
    print("=" * 74)

    SEEDS = [0, 1, 2, 3, 4]
    JITTER_VALUES = [0, 1, 2, 4, 8]

    all_results = {}
    for jmax in JITTER_VALUES:
        print("\n  JITTER_MAX={}".format(jmax))
        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_placement_near()
            r = run_life_with_diagnostics(coords, cids, s, jmax)
            seed_results.append(r)
            print("    seed {}: fused={:.0f}  deposit_edges={}  surviving_edges={}  top1={:.1%}  top3={:.1%}  ({:.0f}s)".format(
                s, r["fused_mass"], r["n_distinct_deposit_edges"],
                r["n_distinct_surviving_edges"], r["top1_concentration"],
                r["top3_concentration"], time.time() - t0))
        all_results[jmax] = seed_results

    print("\n" + "=" * 74)
    print("SUMMARY: Diversity and Concentration by Jitter Level")
    print("{:>10} {:>12} {:>14} {:>14} {:>10} {:>10}".format(
        "jitter", "fused_mass", "deposit_edges", "surviving_edges", "top1_conc", "top3_conc"))
    print("-" * 78)
    for jmax in JITTER_VALUES:
        R = all_results[jmax]
        fm = np.mean([r["fused_mass"] for r in R])
        de = np.mean([r["n_distinct_deposit_edges"] for r in R])
        se = np.mean([r["n_distinct_surviving_edges"] for r in R])
        t1 = np.mean([r["top1_concentration"] for r in R])
        t3 = np.mean([r["top3_concentration"] for r in R])
        print("{:>10} {:>12.0f} {:>14.1f} {:>14.1f} {:>9.1%} {:>9.1%}".format(
            jmax, fm, de, se, t1, t3))

    print("\n" + "=" * 74)
    print("VERDICT:")
    j0 = all_results[0]
    j2 = all_results[2]
    de0 = np.mean([r["n_distinct_deposit_edges"] for r in j0])
    de2 = np.mean([r["n_distinct_deposit_edges"] for r in j2])
    t1_0 = np.mean([r["top1_concentration"] for r in j0])
    t1_2 = np.mean([r["top1_concentration"] for r in j2])

    if de2 > de0 * 1.2:
        print("  CONFIRMED: jitter=2 recruits MORE distinct edges ({:.0f}) than".format(de2))
        print("  jitter=0 ({:.0f}). Diversity hypothesis SUPPORTED.".format(de0))
    else:
        print("  Diversity hypothesis NOT clearly supported: deposit_edges")
        print("  jitter=0={:.0f} vs jitter=2={:.0f}".format(de0, de2))

    if t1_0 > t1_2 * 1.2:
        print("  CONFIRMED: jitter=0 shows HIGHER concentration on top edge ({:.1%})".format(t1_0))
        print("  than jitter=2 ({:.1%}). Rigid synchrony over-focuses on few edges.".format(t1_2))
    else:
        print("  Concentration hypothesis NOT clearly supported: top1_conc")
        print("  jitter=0={:.1%} vs jitter=2={:.1%}".format(t1_0, t1_2))
