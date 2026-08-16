"""Exp 35f: Does widening the deposit radius delay jitter=0's growth ceiling?
35e found: jitter=0 hits a hard growth ceiling at ~80% through the life
because it exhausts the fixed pool of nearby candidate edges. Prediction:
a WIDER local deposit radius gives jitter=0 more candidates to claim,
delaying or eliminating the ceiling.

Sweeps r_deposit (6, 10, 15, 20) for BOTH jitter=0 and jitter=2, tracking
final fused_mass and whether the ceiling still appears in the survival
curve. If the theory is right: jitter=0 should improve MORE than jitter=2
as radius widens (since jitter=2 was never pool-limited to begin with).

Run: python src/experiments/exp35f_radius_sweep.py
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


def run_life(coords, cids, seed, jitter_max, r_deposit):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 8000); dst = rng2.integers(0, N, 8000)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    V = SparsePairState(0.999)
    C = SparsePairState(0.95)
    g = SpatialGrid(coords, r_deposit)
    nbr = [g.within(i, r_deposit) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 400

    a_idx = set(np.where(cids == CHANNEL_A)[0].tolist())
    b_idx = set(np.where(cids == CHANNEL_B)[0].tolist())
    t_idx = set(np.where(cids == FUSION_TARGET)[0].tolist())

    fire_times_A = []
    fire_times_B = []
    checkpoint_edges = []  # track surviving edge count over time for ceiling detection

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

            n_survive = 0
            for si, di in zip(src, dst):
                si, di = int(si), int(di)
                if (si in a_idx or si in b_idx) and di in t_idx:
                    if V.get(si, di) > 0.001:
                        n_survive += 1
                elif (di in a_idx or di in b_idx) and si in t_idx:
                    if V.get(di, si) > 0.001:
                        n_survive += 1
            checkpoint_edges.append(n_survive)

    fused_mass = 0.0
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if (si in a_idx or si in b_idx) and di in t_idx:
            fused_mass += max(V.get(si, di), 0)
        elif (di in a_idx or di in b_idx) and si in t_idx:
            fused_mass += max(V.get(di, si), 0)

    # ceiling detection: did growth stop in the last 4 checkpoints?
    last4 = checkpoint_edges[-4:]
    ceiling_hit = (max(last4) - min(last4)) < 5  # essentially flat

    return fused_mass, checkpoint_edges[-1], ceiling_hit


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 35f: DOES A WIDER DEPOSIT RADIUS DELAY THE GROWTH CEILING?")
    print("Prediction: jitter=0 improves MORE than jitter=2 as radius widens")
    print("(since jitter=2 was never pool-limited to begin with)")
    print("=" * 74)

    SEEDS = [0, 1, 2, 3, 4]
    RADII = [6, 10, 15, 20]
    JITTER_VALUES = [0, 2]

    all_results = {}
    for jmax in JITTER_VALUES:
        for r_dep in RADII:
            key = (jmax, r_dep)
            print("\n  jitter={}, r_deposit={}".format(jmax, r_dep))
            seed_results = []
            for s in SEEDS:
                t0 = time.time()
                coords, cids = make_placement_near()
                fm, final_edges, ceiling = run_life(coords, cids, s, jmax, r_dep)
                seed_results.append((fm, final_edges, ceiling))
                print("    seed {}: fused={:.0f}  final_edges={}  ceiling_hit={}  ({:.0f}s)".format(
                    s, fm, final_edges, ceiling, time.time() - t0))
            all_results[key] = seed_results

    print("\n" + "=" * 74)
    print("SUMMARY: Fused Mass by Jitter and Radius")
    print("{:>10} {:>12}".format("radius", "") + "".join(
        "  {:>14}".format("j={}".format(j)) for j in JITTER_VALUES))
    means = {}
    for r_dep in RADII:
        row = "{:>10}".format(r_dep)
        for jmax in JITTER_VALUES:
            R = all_results[(jmax, r_dep)]
            fm = np.mean([r[0] for r in R])
            means[(jmax, r_dep)] = fm
            row += "  {:>14.0f}".format(fm)
        print(row)

    print("\n" + "=" * 74)
    print("RATIO (j=2 / j=0) BY RADIUS -- does the gap close as radius widens?")
    for r_dep in RADII:
        j0 = means[(0, r_dep)]
        j2 = means[(2, r_dep)]
        ratio = j2 / max(j0, 1)
        print("  radius={}: j=0={:.0f}  j=2={:.0f}  ratio={:.2f}x".format(
            r_dep, j0, j2, ratio))

    print("\nCEILING HIT RATE (fraction of seeds where growth froze):")
    for jmax in JITTER_VALUES:
        for r_dep in RADII:
            R = all_results[(jmax, r_dep)]
            ceiling_rate = np.mean([r[2] for r in R])
            print("  jitter={} radius={}: ceiling_hit_rate={:.0%}".format(
                jmax, r_dep, ceiling_rate))

    print("\n" + "=" * 74)
    print("VERDICT:")
    r6_ratio = means[(2, 6)] / max(means[(0, 6)], 1)
    r20_ratio = means[(2, 20)] / max(means[(0, 20)], 1)
    if r20_ratio < r6_ratio * 0.7:
        print("  CONFIRMED: the j=2/j=0 gap SHRINKS substantially as radius widens")
        print("  (ratio {:.2f}x at r=6 -> {:.2f}x at r=20)".format(r6_ratio, r20_ratio))
        print("  This confirms the candidate-pool-exhaustion mechanism directly:")
        print("  jitter=0's disadvantage IS about running out of nearby candidates.")
    else:
        print("  Gap does NOT shrink as predicted (r=6: {:.2f}x, r=20: {:.2f}x)".format(
            r6_ratio, r20_ratio))
        print("  The pool-exhaustion mechanism may be incomplete or radius isn't")
        print("  the limiting factor in the way hypothesized.")
