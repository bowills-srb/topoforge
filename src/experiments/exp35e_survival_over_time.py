"""Exp 35e: Watching the die-off happen, checkpoint by checkpoint.
35c/35d ruled out targeting diversity and credit-realization opportunity.
This tracks surviving fusion-edge count and total fused mass at EVERY
rewire checkpoint (every 40 steps) through the whole life, for jitter=0
vs jitter=2, to see WHEN and HOW the die-off actually happens.

Does jitter=0 die off early and fast? Steadily throughout? Right after
the first few rewire rounds? The shape of this curve is the answer.

Run: python src/experiments/exp35e_survival_over_time.py
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


def run_life_track_over_time(coords, cids, seed, jitter_max):
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

    a_idx = set(np.where(cids == CHANNEL_A)[0].tolist())
    b_idx = set(np.where(cids == CHANNEL_B)[0].tolist())
    t_idx = set(np.where(cids == FUSION_TARGET)[0].tolist())

    fire_times_A = []
    fire_times_B = []
    checkpoint_log = []  # (step, n_surviving_edges, total_fused_mass)

    def snapshot(step):
        fm = 0.0
        n_survive = 0
        for si, di in zip(src, dst):
            si, di = int(si), int(di)
            if (si in a_idx or si in b_idx) and di in t_idx:
                vv = max(V.get(si, di), 0)
            elif (di in a_idx or di in b_idx) and si in t_idx:
                vv = max(V.get(di, si), 0)
            else:
                continue
            if vv > 0.001:
                n_survive += 1
                fm += vv
        checkpoint_log.append((step, n_survive, fm))

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

            # SNAPSHOT at every rewire checkpoint
            snapshot(t + 1)

    return checkpoint_log


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 35e: SURVIVAL OVER TIME -- watching the die-off happen")
    print("Tracking surviving edges + fused mass at every 40-step checkpoint")
    print("=" * 74)

    SEEDS = [0, 1, 2]  # fewer seeds, more checkpoints per life
    JITTER_VALUES = [0, 2]  # just the two conditions that matter most

    all_logs = {}
    for jmax in JITTER_VALUES:
        print("\n  JITTER_MAX={}".format(jmax))
        seed_logs = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_placement_near()
            log = run_life_track_over_time(coords, cids, s, jmax)
            seed_logs.append(log)
            print("    seed {} done ({:.0f}s)".format(s, time.time() - t0))
        all_logs[jmax] = seed_logs

    print("\n" + "=" * 74)
    print("SURVIVAL CURVE: mean surviving_edges by checkpoint")
    print("(20 checkpoints across the 800-step life)")
    print("=" * 74)

    n_checkpoints = len(all_logs[0][0])
    print("{:>10}".format("checkpoint") + "".join(
        "  {:>8}".format("j={}".format(j)) for j in JITTER_VALUES))
    for cp_idx in range(n_checkpoints):
        step = all_logs[0][0][cp_idx][0]
        row = "{:>10}".format(step)
        for jmax in JITTER_VALUES:
            vals = [all_logs[jmax][s][cp_idx][1] for s in range(len(SEEDS))]
            row += "  {:>8.1f}".format(np.mean(vals))
        print(row)

    print("\n" + "=" * 74)
    print("FUSED MASS CURVE: mean total fused_mass by checkpoint")
    print("{:>10}".format("checkpoint") + "".join(
        "  {:>10}".format("j={}".format(j)) for j in JITTER_VALUES))
    for cp_idx in range(n_checkpoints):
        step = all_logs[0][0][cp_idx][0]
        row = "{:>10}".format(step)
        for jmax in JITTER_VALUES:
            vals = [all_logs[jmax][s][cp_idx][2] for s in range(len(SEEDS))]
            row += "  {:>10.1f}".format(np.mean(vals))
        print(row)

    print("\n" + "=" * 74)
    print("READ THIS FOR: does jitter=0 peak early then decay (die-off),")
    print("or does it just grow slower from the start (never catches up)?")
    print("Does jitter=2 show a clear early advantage, or does the gap")
    print("open up gradually over many rewire rounds?")
