"""Exp 32b: The Placement-Learning Benchmark (PLB)
Publication-ready: 10 seeds, full statistics, benchmark table output.
The litmus test: does placement strategy affect what a network can learn?
Run: python src/exp32b_benchmark.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

PATTERNS = [(0, 3), (1, 4), (2,)]
NC = 5
CORES_X, CORES_Y = 10, 6
N_CORES = CORES_X * CORES_Y
NEURONS_PER_CORE = 15
N = N_CORES * NEURONS_PER_CORE
STEPS_FROZEN = 400
STEPS_PLASTIC = 400
TOTAL = 1200

def core_positions():
    positions = []
    for cy in range(CORES_Y):
        for cx in range(CORES_X):
            positions.append([cx * 5.0, cy * 5.0])
    return np.array(positions)

def neurons_in_core(core_id, core_pos):
    rng = np.random.default_rng(core_id + 1000)
    cx, cy = core_pos
    pts = []
    for _ in range(NEURONS_PER_CORE):
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 1.5)
        pts.append([cx + r*np.cos(th), cy + r*np.sin(th)])
    return np.array(pts)

def make_placement(strategy):
    core_pos = core_positions()
    all_coords = []
    all_cids = []
    if strategy == "random":
        cids_pool = np.repeat(np.arange(NC), N // NC)
        rng = np.random.default_rng(42)
        rng.shuffle(cids_pool)
        idx = 0
        for core_id in range(N_CORES):
            pts = neurons_in_core(core_id, core_pos[core_id])
            for p in pts:
                all_coords.append(p)
                all_cids.append(int(cids_pool[idx]))
                idx += 1
    elif strategy == "vlsi":
        cores_per_id = N_CORES // NC
        core_cid = []
        for cid in range(NC):
            for _ in range(cores_per_id):
                core_cid.append(cid)
        while len(core_cid) < N_CORES:
            core_cid.append(NC - 1)
        for core_id in range(N_CORES):
            pts = neurons_in_core(core_id, core_pos[core_id])
            for p in pts:
                all_coords.append(p)
                all_cids.append(core_cid[core_id])
    elif strategy == "topoforge":
        rng = np.random.default_rng(42)
        for core_id in range(N_CORES):
            pts = neurons_in_core(core_id, core_pos[core_id])
            local_cids = list(range(NC)) * (NEURONS_PER_CORE // NC)
            while len(local_cids) < NEURONS_PER_CORE:
                local_cids.append(rng.integers(0, NC))
            rng.shuffle(local_cids)
            for i, p in enumerate(pts):
                all_coords.append(p)
                all_cids.append(int(local_cids[i]))
    return np.array(all_coords), np.array(all_cids)

def run_life(coords, cids, seed):
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    n_edges = N * 10
    src = rng2.integers(0, N, n_edges); dst = rng2.integers(0, N, n_edges)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(3)
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g = SpatialGrid(coords, 5.0)
    nbr = [g.within(i, 5.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N // 2
    results = {}
    for t in range(TOTAL):
        p = (t // 20) % 3
        in_plastic = t >= STEPS_FROZEN
        in_reversal = t >= STEPS_FROZEN + STEPS_PLASTIC
        if in_reversal:
            inp = rng3.uniform(0, 0.02, N)
            new_pats = [(1, 4), (0, 3), (2,)]
            pat = new_pats[p]
            if (t % 20) < 5:
                for c in pat: inp[cids == c] += 0.5
        else:
            inp = rng3.uniform(0, 0.02, N)
            if (t % 20) < 5:
                for c in PATTERNS[p]: inp[cids == c] += 0.5
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
            if in_reversal:
                base = {0: 0.0, 1: 1.0, 2: -1.0}[p]
            else:
                base = {0: 1.0, 1: 0.0, 2: -1.0}[p]
            delta = base - Rhat[p]
            if abs(delta) > 1e-9:
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0: V.deposit(key[0], key[1], delta * ev)
            Rhat[p] += 0.15 * delta
        if (t + 1) % 40 == 0 and in_plastic:
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
                    dd = D2[ci, cj]
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
        for cp in [STEPS_FROZEN, STEPS_FROZEN + STEPS_PLASTIC, TOTAL]:
            if t + 1 == cp:
                M = np.zeros((NC, NC), dtype=int)
                np.add.at(M, (cids[src], cids[dst]), 1)
                wire_e = D2[src, dst].sum()
                taught = M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]
                selectivity = (M[0, 3] + M[3, 0]) / max(M[1, 4] + M[4, 1], 1)
                old_03 = M[0, 3] + M[3, 0]
                new_14 = M[1, 4] + M[4, 1]
                phase = {STEPS_FROZEN: "frozen",
                         STEPS_FROZEN + STEPS_PLASTIC: "plastic",
                         TOTAL: "reversal"}[cp]
                results[phase] = {"taught": taught, "energy": wire_e,
                                  "selectivity": selectivity,
                                  "old_03": old_03, "new_14": new_14}
    return results

if __name__ == "__main__":
    print("=" * 74)
    print("PLACEMENT-LEARNING BENCHMARK (PLB) v1.0")
    print("TopoForge — github.com/bowills-srb/topoforge")
    print("{} cores ({}x{}), {} neurons/core, {} total.".format(
        N_CORES, CORES_X, CORES_Y, NEURONS_PER_CORE, N))
    print("3 placements x 10 seeds x 3 phases = 90 measurements.")
    print("=" * 74)

    SEEDS = list(range(10))
    strategies = [
        ("random", "Random (unoptimized)"),
        ("vlsi", "Wire-length-optimized (segregated)"),
        ("topoforge", "TopoForge (interleaved)"),
    ]

    all_results = {}
    for strat_name, description in strategies:
        print("\n  {} — {}".format(strat_name.upper(), description))
        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_placement(strat_name)
            r = run_life(coords, cids, s)
            seed_results.append(r)
            print("    seed {:>2}: plastic_taught={:>5}  reversal_old={:>5}  reversal_new={:>4}  ({:.0f}s)".format(
                s, r["plastic"]["taught"],
                r["reversal"]["old_03"], r["reversal"]["new_14"],
                time.time() - t0))
        all_results[strat_name] = seed_results

    # === BENCHMARK TABLE ===
    print("\n" + "=" * 74)
    print("PLACEMENT-LEARNING BENCHMARK RESULTS")
    print("=" * 74)

    print("\nTable 1: Learning Quality (taught mass after 400 steps of plastic learning)")
    print("{:>20} {:>12} {:>8} {:>12}".format(
        "placement", "mean+/-std", "vs rand", "vs vlsi"))
    print("-" * 55)
    baselines = {}
    for sn, _ in strategies:
        vals = [r["plastic"]["taught"] for r in all_results[sn]]
        m, s = np.mean(vals), np.std(vals)
        baselines[sn] = m
        vs_rand = ""
        vs_vlsi = ""
        if sn != "random":
            vs_rand = "{:+.0f}%".format((m - baselines["random"]) / baselines["random"] * 100)
        if sn != "vlsi" and "vlsi" in baselines:
            vs_vlsi = "{:+.0f}%".format((m - baselines["vlsi"]) / baselines["vlsi"] * 100)
        print("{:>20} {:>7.0f}+/-{:<4.0f} {:>8} {:>12}".format(sn, m, s, vs_rand, vs_vlsi))

    print("\nTable 2: Adaptation After Workload Change (old vs new mass post-reversal)")
    print("{:>20} {:>10} {:>10} {:>8}".format(
        "placement", "old_03", "new_14", "ratio"))
    print("-" * 50)
    for sn, _ in strategies:
        old = np.mean([r["reversal"]["old_03"] for r in all_results[sn]])
        new = np.mean([r["reversal"]["new_14"] for r in all_results[sn]])
        ratio = old / max(new, 1)
        print("{:>20} {:>8.0f} {:>8.0f} {:>8.1f}x".format(sn, old, new, ratio))

    print("\nTable 3: Wire Energy by Phase")
    print("{:>20} {:>12} {:>12} {:>12}".format(
        "placement", "frozen", "plastic", "reversal"))
    print("-" * 58)
    for sn, _ in strategies:
        ef = np.mean([r["frozen"]["energy"] for r in all_results[sn]])
        ep = np.mean([r["plastic"]["energy"] for r in all_results[sn]])
        er = np.mean([r["reversal"]["energy"] for r in all_results[sn]])
        print("{:>20} {:>10,.0f} {:>10,.0f} {:>10,.0f}".format(sn, ef, ep, er))

    print("\n" + "=" * 74)
    print("BENCHMARK FINDING:")
    tf = baselines.get("topoforge", 0)
    vl = baselines.get("vlsi", 0)
    if vl > 0:
        ratio = tf / vl
        print("  Segregated placement (grouping functionally-correlated neurons")
        print("  into separate cores) produces {:.1f}x LESS learned structure".format(ratio))
        print("  than interleaved placement, with no energy advantage during")
        print("  frozen operation.")
        print("")
        print("  Mapping tools optimize communication energy, an objective not")
        print("  aligned with learnability on plastic hardware. Whether a given")
        print("  tool incurs this penalty depends on whether its output segregates")
        print("  correlated neurons. Exp 38: SpiNeMap on the graph available at")
        print("  map time DOES segregate them, and does incur the penalty.")
    print("")
    print("  Replicate: github.com/bowills-srb/topoforge")
    print("  Paper: [arXiv link pending]")
