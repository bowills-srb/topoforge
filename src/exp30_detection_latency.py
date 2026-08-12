"""Exp 30: Detection Latency — when do the ants decide to act?
Same coherent anomaly as Exp 29, but log C_pairs every epoch.
Also sweep anomaly strength: subtle vs moderate vs strong.
The crossing of a detection threshold = time to alert.
Run: python src/exp30_detection_latency.py
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
TOTAL = 1200
N_ANOMALY = 50

STRENGTHS = [0.05, 0.10, 0.15, 0.25, 0.40]
NOISE_FLOOR = 394  # Exp 29 random arm's C_pairs — the background level
DETECT_THRESHOLD = NOISE_FLOOR * 2  # 2x noise = "the ants are sure"

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

def run_arm(coords, cids, seed, strength):
    rng3 = np.random.default_rng(seed)
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

    timeline = []
    detect_step = None

    for t in range(TOTAL):
        p = (t // 20) % 3
        inp = rng3.uniform(0, 0.02, N)
        if (t % 20) < 5:
            for c in PATTERNS[p]: inp[cids == c] += 0.5
        if t >= ANOMALY_START:
            inp[anomaly_neurons] += strength

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
                if detect_step is None and cpairs >= DETECT_THRESHOLD:
                    detect_step = t + 1

    return timeline, detect_step

if __name__ == "__main__":
    print("=" * 70)
    print("EXP 30: DETECTION LATENCY — when do the ants decide to act?")
    print("5 anomaly strengths x 5 seeds. Coherent group of 50.")
    print("Detection threshold: {} C_pairs (2x noise floor from Exp 29)".format(
        DETECT_THRESHOLD))
    print("=" * 70)

    SEEDS = [0, 1, 2, 3, 4]
    all_latencies = {}

    for strength in STRENGTHS:
        latencies = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_salt_clustered()
            timeline, detect = run_arm(coords, cids, s, strength)
            latencies.append(detect)
            # show timeline at key points
            traj = ""
            for step_check in [640, 720, 800, 920, 1080, 1200]:
                matches = [tp for tp in timeline if tp[0] == step_check]
                if matches:
                    traj += " @{}:{}".format(step_check, matches[0][1])
            dx = "step {}".format(detect) if detect else "never"
            print("  str={} seed {}: detect={} ({:.0f}s){}".format(
                strength, s, dx, time.time() - t0, traj))
        all_latencies[strength] = latencies
        valid = [l for l in latencies if l is not None]
        if valid:
            mean_lat = np.mean(valid)
            steps_after = mean_lat - ANOMALY_START
            pct = len(valid) * 100 // len(SEEDS)
            print("  >> detected {}/5 seeds, mean latency: step {:.0f} ({:.0f} steps after onset)\n".format(
                len(valid), mean_lat, steps_after))
        else:
            print("  >> detected 0/5 seeds (sub-threshold at this strength)\n")

    print("=" * 70)
    print("DETECTION LATENCY TABLE")
    print("{:>10} {:>12} {:>15} {:>10}".format(
        "strength", "detect_rate", "mean_latency", "steps"))
    print("-" * 50)
    for strength in STRENGTHS:
        L = all_latencies[strength]
        valid = [l for l in L if l is not None]
        rate = len(valid) * 100 // len(SEEDS)
        if valid:
            ml = np.mean(valid)
            steps = ml - ANOMALY_START
            print("{:>10} {:>10}% {:>13.0f} {:>10.0f}".format(
                strength, rate, ml, steps))
        else:
            print("{:>10} {:>10}% {:>13} {:>10}".format(
                strength, rate, "never", "-"))

    print("")
    print("=" * 70)
    print("HOW TO READ THIS:")
    print("  Strength 0.05 = very subtle fault (bearing starting to wear)")
    print("  Strength 0.40 = obvious fault (loose component, clear rattle)")
    print("")
    print("  The latency column is the PRODUCT SPEC:")
    print("  'How many cycles after a fault begins will the sensor notice?'")
    print("  Multiply by the real pump's cycle time to get wall-clock hours.")
    print("")
    print("  If a pump cycles every 10 seconds and detection takes 200 steps,")
    print("  that's ~33 minutes from fault onset to detection.")
    print("  At every 30 seconds, that's ~100 minutes.")
    print("")
    print("  The strength-vs-latency curve IS the sensor's detection spec.")
