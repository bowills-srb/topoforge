"""Exp 31: The Four-Dimensional Filter
Five temporal modes of anomaly, same spatial coherence, same strength.
The TRAJECTORY of C_pairs discriminates signal from noise.
Run: python src/exp31_temporal_modes.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

PATTERNS = [(0, 3), (1, 4), (2,)]
N, NC = 1000, 5
ANOMALY_START = 600
TOTAL = 1000
N_ANOMALY = 50
STRENGTH = 0.15

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

def pick_coherent_group(coords, n):
    rng = np.random.default_rng(77)
    center_idx = rng.integers(0, len(coords))
    dists = ((coords - coords[center_idx]) ** 2).sum(1)
    return np.argsort(dists)[:n]

def run_arm(coords, cids, seed, mode):
    """
    Modes:
      'control'        - no anomaly
      'sync_persist'   - all 50 biased every step (the real fault)
      'sync_transient' - all 50 biased for 20 steps then off (coffee spill)
      'desync_persist' - each neuron biased on its own random 30% schedule (local noise)
      'cross_modal'    - 50 adjacent neurons from MIXED types, each type gets
                         a DIFFERENT bias pattern, all simultaneous (multi-modal fault)
    """
    rng3 = np.random.default_rng(seed)
    rng_anom = np.random.default_rng(seed + 200)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 10000); dst = rng2.integers(0, N, 10000)
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
    swap = 500

    anomaly_neurons = pick_coherent_group(coords, N_ANOMALY)
    aset = set(int(x) for x in anomaly_neurons)

    # for desync mode: each neuron has its own random on/off schedule
    desync_schedule = rng_anom.random((N_ANOMALY, TOTAL)) < 0.30

    # for cross_modal: group anomaly neurons by their cluster identity
    # each cluster type gets a different bias multiplier
    anom_cids = cids[anomaly_neurons]
    unique_types = np.unique(anom_cids)
    type_multipliers = {}
    for i, ut in enumerate(unique_types):
        # each type feels the anomaly differently: some in +, some shifted
        type_multipliers[int(ut)] = STRENGTH * (0.5 + 1.0 * i / max(len(unique_types) - 1, 1))

    timeline = []

    for t in range(TOTAL):
        p = (t // 20) % 3
        inp = rng3.uniform(0, 0.02, N)
        if (t % 20) < 5:
            for c in PATTERNS[p]: inp[cids == c] += 0.5

        # inject anomaly based on mode
        if t >= ANOMALY_START:
            if mode == "sync_persist":
                inp[anomaly_neurons] += STRENGTH
            elif mode == "sync_transient":
                if t < ANOMALY_START + 20:
                    inp[anomaly_neurons] += STRENGTH
                # else: off — the coffee spill ended
            elif mode == "desync_persist":
                for idx, neuron in enumerate(anomaly_neurons):
                    if desync_schedule[idx, t]:
                        inp[neuron] += STRENGTH
            elif mode == "cross_modal":
                for neuron in anomaly_neurons:
                    cid_of_neuron = int(cids[neuron])
                    inp[neuron] += type_multipliers.get(cid_of_neuron, STRENGTH)

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
            base = {0: 1.0, 1: 0.0, 2: -1.0}[p]
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

            # measure C_pairs in anomaly neighborhood
            if t >= ANOMALY_START:
                cpairs = 0
                for i in anomaly_neurons:
                    i = int(i)
                    for j in nbr[i]:
                        if int(j) in aset:
                            if C.get(i, int(j)) > 0.1:
                                cpairs += 1
                timeline.append((t + 1, cpairs))

    return timeline

if __name__ == "__main__":
    print("=" * 70)
    print("EXP 31: THE FOUR-DIMENSIONAL FILTER")
    print("5 modes x 3 seeds. Strength=0.15. Anomaly starts at step 600.")
    print("Tracking C_pairs trajectory — the SHAPE of detection over time.")
    print("=" * 70)

    SEEDS = [0, 1, 2]
    modes = [
        ("control",        "No anomaly — the flat baseline"),
        ("sync_persist",   "All 50 biased every step — THE REAL FAULT"),
        ("sync_transient", "All 50 biased for 20 steps then off — COFFEE SPILL"),
        ("desync_persist", "Each neuron on 30% random schedule — LOCAL NOISE"),
        ("cross_modal",    "50 adjacent, mixed types, different biases — MULTI-MODAL"),
    ]

    all_timelines = {}
    for mode_name, description in modes:
        print("\n  MODE: {} ({})".format(mode_name, description))
        seed_timelines = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_salt_clustered()
            tl = run_arm(coords, cids, s, mode_name)
            seed_timelines.append(tl)
            # print trajectory at key checkpoints
            traj = ""
            for step_check in [640, 680, 720, 800, 920, 1000]:
                matches = [tp for tp in tl if tp[0] == step_check]
                if matches:
                    traj += " @{}:{}".format(step_check, matches[0][1])
            print("    seed {}: ({:.0f}s){}".format(s, time.time() - t0, traj))
        all_timelines[mode_name] = seed_timelines

    print("\n" + "=" * 70)
    print("TRAJECTORY COMPARISON (mean C_pairs across seeds at each checkpoint)")
    print("{:>16} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "mode", "@640", "@680", "@720", "@800", "@920", "@1000"))
    print("-" * 70)
    for mode_name, _ in modes:
        vals = {}
        for step_check in [640, 680, 720, 800, 920, 1000]:
            readings = []
            for tl in all_timelines[mode_name]:
                matches = [tp for tp in tl if tp[0] == step_check]
                if matches:
                    readings.append(matches[0][1])
            vals[step_check] = np.mean(readings) if readings else 0
        print("{:>16} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.0f}".format(
            mode_name, vals[640], vals[680], vals[720], vals[800], vals[920], vals[1000]))

    print("\n" + "=" * 70)
    print("HOW TO READ THE TRAJECTORIES:")
    print("  control:        flat near zero — the reference")
    print("  sync_persist:   high and HOLDS or CLIMBS — the real fault signature")
    print("  sync_transient: spike then DECAYS — coffee spill, self-clearing")
    print("  desync_persist: stays LOW — noise, ants disagree on timing")
    print("  cross_modal:    high and holds — if STRONGER than sync_persist,")
    print("                  multi-modal consensus is the premium signal")
    print("")
    print("The SHAPE of the trajectory is the discriminator:")
    print("  Holds steady  = real fault, alert the human")
    print("  Spikes & fades = transient, ignore it")
    print("  Never rises    = noise, ignore it")
    print("  Rises & GROWS  = worsening fault, escalate urgency")
