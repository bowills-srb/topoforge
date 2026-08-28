"""Exp 35b: ITDP Fusion Under Placement -- FIXED co-firing detection.
Corrects Exp 35's flaw: fusion deposits now trigger ONLY when A and B
have BOTH fired within a short recent window of each other -- not on
any network spike. This isolates genuine synchrony from background
propagation contamination.

Also switches from random rewire to our normal value-gated rewire,
so edges carrying real fusion structure are protected the way they
would be in the main engine.

3 conditions x 5 seeds, PLUS a 4th: a timing sweep to see if fused
mass follows the actual Gaussian shape as timing offset increases.

Run: python src/experiments/exp35b_itdp_fusion_fixed.py
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
COINCIDENCE_WINDOW = 8  # max steps apart to even consider "co-firing"

def itdp_gaussian(delta_t):
    return ITDP_AMPLITUDE * np.exp(-(delta_t ** 2) / (2 * ITDP_SIGMA ** 2))

def make_placement(condition):
    rng = np.random.default_rng(11)
    coords = []
    cids = []
    if condition in ("near", "no_timing"):
        centers = {
            CHANNEL_A: [10, 10], CHANNEL_B: [12, 10],
            FUSION_TARGET: [11, 12], 2: [30, 10], 4: [30, 30],
        }
    else:  # far
        centers = {
            CHANNEL_A: [5, 5], CHANNEL_B: [35, 35],
            FUSION_TARGET: [20, 20], 2: [30, 5], 4: [5, 30],
        }
    for cid, (cx, cy) in centers.items():
        for _ in range(200):
            th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 3)
            coords.append([cx + r*np.cos(th), cy + r*np.sin(th)])
            cids.append(cid)
    return np.array(coords), np.array(cids)

def run_life(coords, cids, seed, condition, jitter_max=2):
    """condition: 'near', 'far', 'no_timing', or an integer jitter value
    (for the timing sweep -- larger jitter = less synchronized)."""
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 8000); dst = rng2.integers(0, N, 8000)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    V = SparsePairState(0.999)
    C = SparsePairState(0.95)  # now tracking real co-activity too, for value-gated rewire
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = 400

    fire_times_A = []  # recent firing history, trimmed each step
    fire_times_B = []
    fusion_events = 0
    total_ab_cofire_attempts = 0

    for t in range(TOTAL):
        inp = rng3.uniform(0, 0.02, N)
        a_fired_now = False
        b_fired_now = False

        if (t % 20) < 3:
            inp[cids == CHANNEL_A] += 0.5
            a_fired_now = True

        if condition == "no_timing":
            if rng3.random() < 0.15:
                inp[cids == CHANNEL_B] += 0.5
                b_fired_now = True
        elif isinstance(condition, int):
            # timing sweep: fixed jitter offset from A's schedule
            if (t % 20) < 3:
                if rng3.random() < 0.9:  # mostly fires, offset by `condition` steps
                    b_fired_now = True
                    # apply as an input a few steps later (handled via schedule below)
            if (t % 20) == (3 + condition) % 20:
                inp[cids == CHANNEL_B] += 0.5
                b_fired_now = True
        else:  # near / far
            if (t % 20) < 3:
                jitter = rng3.integers(-jitter_max, jitter_max + 1)
                if 0 <= (t + jitter) % 20 < 5:
                    inp[cids == CHANNEL_B] += 0.5
                    b_fired_now = True

        if a_fired_now:
            fire_times_A.append(t)
        if b_fired_now:
            fire_times_B.append(t)
        # trim history to the coincidence window
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

        # FIXED: only attempt fusion deposit when BOTH A and B have fired
        # within the coincidence window of EACH OTHER (not just recently
        # in general, and not triggered by unrelated network spikes)
        if fire_times_A and fire_times_B:
            best_delta = min(abs(a - b) for a in fire_times_A[-3:]
                             for b in fire_times_B[-3:])
            total_ab_cofire_attempts += 1
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
                            fusion_events += 1

        v = np.maximum(v_, 0); v[fired] = 0
        refrac[fired] = 3; refrac[refrac > 0] -= 1

        if (t + 1) % 40 == 0:
            # VALUE-GATED rewire (not random) -- protects real fusion structure
            C.prune_below(1e-6)
            sc = np.array([V.get(int(src[k]), int(dst[k])) for k in range(len(src))])
            cold = np.argsort(sc, kind="stable")[:swap]
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
                        order = np.argsort(score, kind="stable")[::-1][:len(cold)]
                        n2 = min(len(cold), len(order))
                        src[cold[:n2]] = ci[order[:n2]]; dst[cold[:n2]] = cj[order[:n2]]
                        out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    fused_mass = 0.0
    a_idx = set(np.where(cids == CHANNEL_A)[0].tolist())
    b_idx = set(np.where(cids == CHANNEL_B)[0].tolist())
    t_idx = set(np.where(cids == FUSION_TARGET)[0].tolist())
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if (si in a_idx or si in b_idx) and di in t_idx:
            fused_mass += max(V.get(si, di), 0)
        elif (di in a_idx or di in b_idx) and si in t_idx:
            fused_mass += max(V.get(di, si), 0)

    return fused_mass, fusion_events, total_ab_cofire_attempts


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 35b: ITDP FUSION -- FIXED co-firing detection + value-gated rewire")
    print("=" * 74)

    SEEDS = [0, 1, 2, 3, 4]

    print("\nPART 1: Original 3-condition comparison, corrected mechanism")
    conditions = [
        ("near", "Channels A+B adjacent, synchronized"),
        ("far", "Channels A+B far apart, synchronized"),
        ("no_timing", "Channels A+B adjacent, uncorrelated"),
    ]
    all_results = {}
    for cond_name, description in conditions:
        print("\n  {} -- {}".format(cond_name.upper(), description))
        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_placement(cond_name)
            fused, events, attempts = run_life(coords, cids, s, cond_name)
            seed_results.append((fused, events, attempts))
            print("    seed {}: fused_mass={:.2f}  fusion_events={}  cofire_windows={}  ({:.0f}s)".format(
                s, fused, events, attempts, time.time() - t0))
        all_results[cond_name] = seed_results

    print("\n" + "=" * 74)
    print("PART 1 RESULTS")
    print("{:>12} {:>18} {:>16}".format("condition", "fused_mass", "fusion_events"))
    print("-" * 50)
    means = {}
    for cond_name, _ in conditions:
        R = all_results[cond_name]
        fm = np.mean([r[0] for r in R])
        fe = np.mean([r[1] for r in R])
        means[cond_name] = fm
        print("{:>12} {:>14.2f}+/-{:<4.2f} {:>16.0f}".format(
            cond_name, fm, np.std([r[0] for r in R]), fe))

    print("\nPART 2: Timing sweep -- does fused mass trace the Gaussian shape?")
    print("(near placement, jitter_max swept from 0 to 8 steps)")
    sweep_results = []
    for jmax in [0, 1, 2, 3, 4, 6, 8]:
        fms = []
        for s in SEEDS[:3]:  # fewer seeds for the sweep to keep runtime sane
            coords, cids = make_placement("near")
            fused, events, attempts = run_life(coords, cids, s, "near", jitter_max=jmax)
            fms.append(fused)
        mean_fm = np.mean(fms)
        sweep_results.append((jmax, mean_fm))
        print("  jitter_max={}: fused_mass={:.2f}+/-{:.2f}".format(
            jmax, mean_fm, np.std(fms)))

    print("\n" + "=" * 74)
    near_v = means.get("near", 0)
    far_v = means.get("far", 0)
    nt_v = means.get("no_timing", 0)
    print("VERDICT:")
    print("  far=0 placement gating: {}".format(
        "CONFIRMED" if far_v < 0.01 else "old result did not replicate: {:.2f}".format(far_v)))
    if near_v > nt_v:
        print("  Synchrony helps (near > no_timing): CONFIRMED after fix")
        print("  near={:.2f} > no_timing={:.2f}".format(near_v, nt_v))
    else:
        print("  Synchrony still doesn't show expected advantage:")
        print("  near={:.2f} vs no_timing={:.2f} -- needs further investigation".format(near_v, nt_v))

    fms_sweep = [r[1] for r in sweep_results]
    if fms_sweep[0] > fms_sweep[-1]:
        print("  Timing sweep shows DECLINING fused mass as jitter increases:")
        print("  consistent with Gaussian ITDP shape (synchrony = strongest fusion)")
    else:
        print("  Timing sweep does not show the expected declining pattern")
    print("\n  Citation: Kim, S.J., Zhao, R. et al. Nature Sensors 1, 535-544 (2026).")
