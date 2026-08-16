"""Exp 35g: Radius sweep, corrected for the geometry confound in 35f.
35f found radius=10 already reached every neuron in the tightly-packed
200-neuron clusters, so widening further couldn't test anything. This
fixes it two ways: (1) bigger clusters (500 neurons instead of 200),
(2) wider PHYSICAL SPREAD per cluster (radius 8 instead of 3), so a
growing search radius genuinely reveals more candidates step by step
instead of saturating almost instantly.

Run: python src/experiments/exp35g_radius_sweep_fixed.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

CHANNEL_A, CHANNEL_B = 0, 1
FUSION_TARGET = 3
ITDP_SIGMA = 3.0
ITDP_AMPLITUDE = 1.0
COINCIDENCE_WINDOW = 8
TOTAL = 800

# bigger, more physically spread clusters -- the actual fix
N_MAIN = 500   # A, B, target each get this many
N_BG = 100     # background clusters 2, 4 stay small
N = N_MAIN * 3 + N_BG * 2
SPREAD = 8.0   # physical radius of each cluster (was 3.0 -- too tight)


def itdp_gaussian(delta_t):
    return ITDP_AMPLITUDE * np.exp(-(delta_t ** 2) / (2 * ITDP_SIGMA ** 2))


def make_placement_near():
    rng = np.random.default_rng(11)
    coords = []
    cids = []
    # wider spacing between centers too, to give radius room to matter
    centers = {
        CHANNEL_A: (N_MAIN, [15, 15]),
        CHANNEL_B: (N_MAIN, [25, 15]),
        FUSION_TARGET: (N_MAIN, [20, 25]),
        2: (N_BG, [50, 15]),
        4: (N_BG, [15, 50]),
    }
    for cid, (n_cluster, (cx, cy)) in centers.items():
        for _ in range(n_cluster):
            th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, SPREAD)
            coords.append([cx + r*np.cos(th), cy + r*np.sin(th)])
            cids.append(cid)
    return np.array(coords), np.array(cids)


def run_life(coords, cids, seed, jitter_max, r_deposit):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, N * 8); dst = rng2.integers(0, N, N * 8)
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
    swap = int(N * 0.4)

    a_idx = set(np.where(cids == CHANNEL_A)[0].tolist())
    b_idx = set(np.where(cids == CHANNEL_B)[0].tolist())
    t_idx = set(np.where(cids == FUSION_TARGET)[0].tolist())

    # diagnostic: how many neighbors does a typical channel-A neuron
    # actually have at this radius? (measures whether radius is
    # genuinely revealing more candidates)
    sample_a = list(a_idx)[0]
    avg_nbr_count = len(nbr[sample_a])

    fire_times_A = []
    fire_times_B = []

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

    fused_mass = 0.0
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if (si in a_idx or si in b_idx) and di in t_idx:
            fused_mass += max(V.get(si, di), 0)
        elif (di in a_idx or di in b_idx) and si in t_idx:
            fused_mass += max(V.get(di, si), 0)

    return fused_mass, avg_nbr_count


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 35g: RADIUS SWEEP -- FIXED (bigger, more spread clusters)")
    print("N={} total ({} each in A/B/target, spread={} units)".format(
        N, N_MAIN, SPREAD))
    print("Checking neighbor count grows meaningfully with radius this time")
    print("=" * 74)

    SEEDS = [0, 1, 2]
    RADII = [4, 8, 15, 25]
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
                fm, nbr_count = run_life(coords, cids, s, jmax, r_dep)
                seed_results.append((fm, nbr_count))
                print("    seed {}: fused={:.0f}  avg_neighbors={}  ({:.0f}s)".format(
                    s, fm, nbr_count, time.time() - t0))
            all_results[key] = seed_results

    print("\n" + "=" * 74)
    print("NEIGHBOR COUNT CHECK (is radius actually revealing more candidates?)")
    for r_dep in RADII:
        R = all_results[(0, r_dep)]
        nc = np.mean([r[1] for r in R])
        print("  r_deposit={}: avg_neighbors={:.0f}".format(r_dep, nc))

    print("\n" + "=" * 74)
    print("SUMMARY: Fused Mass by Jitter and Radius")
    print("{:>10}".format("radius") + "".join(
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

    print("\nRATIO (j=2 / j=0) BY RADIUS:")
    for r_dep in RADII:
        j0 = means[(0, r_dep)]
        j2 = means[(2, r_dep)]
        ratio = j2 / max(j0, 1)
        print("  radius={}: j=0={:.0f}  j=2={:.0f}  ratio={:.2f}x".format(
            r_dep, j0, j2, ratio))

    print("\n" + "=" * 74)
    print("VERDICT:")
    r_first = means[(2, RADII[0])] / max(means[(0, RADII[0])], 1)
    r_last = means[(2, RADII[-1])] / max(means[(0, RADII[-1])], 1)
    if r_last < r_first * 0.7:
        print("  CONFIRMED (this time with room to test it): the gap SHRINKS")
        print("  substantially as radius widens ({:.2f}x -> {:.2f}x).".format(r_first, r_last))
        print("  Pool-exhaustion mechanism directly confirmed.")
    else:
        print("  Gap still does not shrink as predicted ({:.2f}x -> {:.2f}x)".format(
            r_first, r_last))
        print("  even with a properly spread-out body. The mechanism may be")
        print("  about something other than raw candidate pool size --")
        print("  worth reconsidering the ITDP timing-window interaction itself.")
