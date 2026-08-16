"""Exp 35d: Testing the refractory-resonance hypothesis directly.
Exp 35c found: same edges get touched regardless of jitter, but far
fewer SURVIVE at jitter=0 (41%) vs jitter=2 (60%). Hypothesis: rigid
periodic stimulation resonates badly with the fixed 3-step refractory
period, systematically locking the fusion-target neuron out of firing
at credit-assignment moments.

This tracks, per jitter level:
  - raw target-cluster firing count (total fires across the life)
  - "coincidence windows": steps where both A and B fired recently
  - "credit realized": of those coincidence windows, how often did
    the target ALSO fire nearby (within 2 steps)?
  - "credit missed": coincidence existed but target never fired near it

If the resonance hypothesis is right: jitter=0 should show a LOWER
credit-realized rate (more missed opportunities) than jitter=2, even
though raw target firing might be similar.

Run: python src/experiments/exp35d_refractory_resonance.py
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
CREDIT_WINDOW = 2  # how close target firing must be to count as "realized"


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


def run_life_track_target(coords, cids, seed, jitter_max):
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

    target_idx_set = set(np.where(cids == FUSION_TARGET)[0].tolist())

    fire_times_A = []
    fire_times_B = []
    target_fire_times = []  # every step any target-cluster neuron fired
    total_target_fires = 0

    coincidence_steps = []  # steps where a co-fire window existed

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

        # track target firing
        if len(f):
            fset = set(int(x) for x in f)
            if fset & target_idx_set:
                target_fire_times.append(t)
                total_target_fires += len(fset & target_idx_set)

        if len(f):
            fs = set(int(x) for x in f)
            for i in f:
                i = int(i)
                for j in nbr[i]:
                    if int(j) in fs:
                        C.deposit(i, j, 1.0)

        if fire_times_A and fire_times_B:
            coincidence_steps.append(t)
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

    # final fused mass (same as before, for cross-reference)
    a_idx = set(np.where(cids == CHANNEL_A)[0].tolist())
    b_idx = set(np.where(cids == CHANNEL_B)[0].tolist())
    fused_mass = 0.0
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if (si in a_idx or si in b_idx) and di in target_idx_set:
            fused_mass += max(V.get(si, di), 0)
        elif (di in a_idx or di in b_idx) and si in target_idx_set:
            fused_mass += max(V.get(di, si), 0)

    # credit realization: of the coincidence steps, how many had target
    # firing within CREDIT_WINDOW steps?
    target_fire_arr = np.array(target_fire_times) if target_fire_times else np.array([])
    credit_realized = 0
    for cs in coincidence_steps:
        if len(target_fire_arr) > 0:
            if np.any(np.abs(target_fire_arr - cs) <= CREDIT_WINDOW):
                credit_realized += 1

    n_coincidence = len(coincidence_steps)
    credit_rate = credit_realized / max(n_coincidence, 1)

    return {
        "fused_mass": fused_mass,
        "total_target_fires": total_target_fires,
        "n_coincidence_steps": n_coincidence,
        "credit_realized": credit_realized,
        "credit_rate": credit_rate,
    }


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 35d: TESTING THE REFRACTORY-RESONANCE HYPOTHESIS")
    print("Does rigid jitter=0 periodicity lock the target out of firing")
    print("at credit-assignment moments, vs jitter=2 breaking that lock?")
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
            r = run_life_track_target(coords, cids, s, jmax)
            seed_results.append(r)
            print("    seed {}: fused={:.0f}  target_fires={}  coincidences={}  credit_realized={}  credit_rate={:.1%}  ({:.0f}s)".format(
                s, r["fused_mass"], r["total_target_fires"],
                r["n_coincidence_steps"], r["credit_realized"],
                r["credit_rate"], time.time() - t0))
        all_results[jmax] = seed_results

    print("\n" + "=" * 74)
    print("SUMMARY: Target Firing and Credit Realization by Jitter")
    print("{:>10} {:>12} {:>14} {:>14} {:>12}".format(
        "jitter", "fused_mass", "target_fires", "coincidences", "credit_rate"))
    print("-" * 66)
    for jmax in JITTER_VALUES:
        R = all_results[jmax]
        fm = np.mean([r["fused_mass"] for r in R])
        tf = np.mean([r["total_target_fires"] for r in R])
        nc = np.mean([r["n_coincidence_steps"] for r in R])
        cr = np.mean([r["credit_rate"] for r in R])
        print("{:>10} {:>12.0f} {:>14.1f} {:>14.1f} {:>11.1%}".format(
            jmax, fm, tf, nc, cr))

    print("\n" + "=" * 74)
    print("VERDICT:")
    cr0 = np.mean([r["credit_rate"] for r in all_results[0]])
    cr2 = np.mean([r["credit_rate"] for r in all_results[2]])
    tf0 = np.mean([r["total_target_fires"] for r in all_results[0]])
    tf2 = np.mean([r["total_target_fires"] for r in all_results[2]])

    if cr2 > cr0 * 1.2:
        print("  REFRACTORY-RESONANCE CONFIRMED: credit_rate at jitter=0 ({:.1%})".format(cr0))
        print("  is substantially lower than jitter=2 ({:.1%}).".format(cr2))
        print("  Rigid periodicity misses credit-assignment windows more often.")
    else:
        print("  Refractory-resonance NOT clearly supported by credit_rate:")
        print("  jitter=0={:.1%} vs jitter=2={:.1%}".format(cr0, cr2))

    if tf0 < tf2 * 0.8:
        print("  Raw target firing rate is also LOWER at jitter=0 ({:.1f}) than".format(tf0))
        print("  jitter=2 ({:.1f}) -- consistent with systematic suppression.".format(tf2))
    else:
        print("  Raw target firing rate similar: jitter=0={:.1f} vs jitter=2={:.1f}".format(tf0, tf2))
        print("  If credit_rate differs but raw firing doesn't, the effect is")
        print("  specifically about TIMING ALIGNMENT, not overall target activity.")
