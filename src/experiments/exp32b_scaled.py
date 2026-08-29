"""PLB v1.0 Scaled: N=5,000 on a 25x10 grid (250 cores, 20 neurons/core)
3 placements x 10 seeds. The benchmark at realistic chip scale.
Run: python src/exp32b_scaled.py

METRIC CORRECTION (2026-08-29, PROJECT_HISTORY gotcha #11): raw "taught"
mass is a raw edge count diluted by a non-local random-initialization
baseline shared by every placement (see exp32b_benchmark.py's docstring --
the N=900 benchmark this scales up). This script already records taught at
the frozen checkpoint, so growth = plastic_taught - frozen_taught
(plasticity-attributable) is now reported alongside raw taught as the
corrected primary metric.
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

PATTERNS = [(0, 3), (1, 4), (2,)]
NC = 5
CORES_X, CORES_Y = 25, 10
N_CORES = CORES_X * CORES_Y
NEURONS_PER_CORE = 20
N = N_CORES * NEURONS_PER_CORE
STEPS_FROZEN = 400
STEPS_PLASTIC = 400
TOTAL = 1200

print("N={}, D2 matrix={:.0f}MB".format(N, N*N*8/1e6))

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
    # D2 at N=5000 is ~200MB — should fit in RAM
    print("      building D2...", end=" ", flush=True)
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    print("done. building neighbors...", end=" ", flush=True)
    g = SpatialGrid(coords, 5.0)
    nbr = [g.within(i, 5.0) for i in range(N)]
    print("done.", flush=True)
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N // 2
    results = {}
    last_report = time.time()
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
        # progress heartbeat
        if (t + 1) % 200 == 0:
            elapsed = time.time() - last_report
            print("        step {}/{} ({:.0f}s)".format(t + 1, TOTAL, elapsed), flush=True)
            last_report = time.time()
        for checkpoint in [STEPS_FROZEN, STEPS_FROZEN + STEPS_PLASTIC, TOTAL]:
            if t + 1 == checkpoint:
                M = np.zeros((NC, NC), dtype=int)
                np.add.at(M, (cids[src], cids[dst]), 1)
                wire_e = D2[src, dst].sum()
                taught = M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]
                old_03 = M[0, 3] + M[3, 0]
                new_14 = M[1, 4] + M[4, 1]
                phase = {STEPS_FROZEN: "frozen",
                         STEPS_FROZEN + STEPS_PLASTIC: "plastic",
                         TOTAL: "reversal"}[checkpoint]
                results[phase] = {"taught": taught, "energy": wire_e,
                                  "old_03": old_03, "new_14": new_14}
    return results

if __name__ == "__main__":
    print("=" * 74)
    print("PLACEMENT-LEARNING BENCHMARK (PLB) v1.0 — SCALED")
    print("TopoForge — github.com/bowills-srb/topoforge")
    print("{} cores ({}x{}), {} neurons/core, {} total.".format(
        N_CORES, CORES_X, CORES_Y, NEURONS_PER_CORE, N))
    print("3 placements x 10 seeds. This will take a while.")
    print("=" * 74)

    SEEDS = list(range(10))
    strategies = [
        ("topoforge", "TopoForge (interleaved)"),
        ("random", "Random (unoptimized)"),
        ("vlsi", "VLSI-optimized (segregated)"),
    ]

    all_results = {}
    for strat_name, description in strategies:
        print("\n  {} — {}".format(strat_name.upper(), description))
        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            print("    seed {}:".format(s))
            coords, cids = make_placement(strat_name)
            r = run_life(coords, cids, s)
            seed_results.append(r)
            total_time = time.time() - t0
            print("    => taught={:>6}  energy={:>12,.0f}  ({:.0f}s)".format(
                r["plastic"]["taught"], r["plastic"]["energy"], total_time))
        all_results[strat_name] = seed_results

    print("\n" + "=" * 74)
    print("PLACEMENT-LEARNING BENCHMARK RESULTS (N={})".format(N))
    print("=" * 74)

    print("\nTable 1: Learning Quality (raw taught -- legacy, diluted by shared baseline)")
    print("{:>15} {:>14} {:>10}".format("placement", "taught", "vs vlsi"))
    print("-" * 42)
    baselines = {}
    for sn, _ in strategies:
        vals = [r["plastic"]["taught"] for r in all_results[sn]]
        m, s = np.mean(vals), np.std(vals)
        baselines[sn] = m
        vs = ""
        if sn != "vlsi" and "vlsi" in baselines:
            vs = "{:.1f}x".format(m / max(baselines["vlsi"], 1))
        print("{:>15} {:>7.0f}+/-{:<4.0f} {:>10}".format(sn, m, s, vs))

    print("\nTable 1b: Plasticity-Attributable Learning (growth = plastic - frozen taught)")
    print("CORRECTED primary result -- see module docstring.")
    print("{:>15} {:>16} {:>10}".format("placement", "growth", "sign"))
    print("-" * 44)
    growth_baselines = {}
    growth_vals = {}
    for sn, _ in strategies:
        g = np.array([r["plastic"]["taught"] - r["frozen"]["taught"] for r in all_results[sn]], float)
        growth_vals[sn] = g
        growth_baselines[sn] = g.mean()
        sign = "NET GAIN" if g.mean() > 0 else "NET LOSS"
        print("{:>15} {:>8.0f}+/-{:<6.0f} {:>10}".format(sn, g.mean(), g.std(), sign))
    if "topoforge" in growth_vals and "vlsi" in growth_vals:
        try:
            from scipy import stats
            t, p = stats.ttest_ind(growth_vals["topoforge"], growth_vals["vlsi"], equal_var=False)
            print("\n  interleaved vs segregated growth: Welch t={:.2f}, p={:.2e}".format(t, p))
        except ImportError:
            print("\n  (scipy unavailable -- skipping Welch test; means above still valid)")

    print("\nTable 2: Wire Energy (frozen phase)")
    print("{:>15} {:>14}".format("placement", "energy"))
    print("-" * 32)
    for sn, _ in strategies:
        vals = [r["frozen"]["energy"] for r in all_results[sn]]
        print("{:>15} {:>12,.0f}".format(sn, np.mean(vals)))

    print("\nTable 3: Adaptation")
    print("{:>15} {:>8} {:>8}".format("placement", "old_03", "new_14"))
    print("-" * 34)
    for sn, _ in strategies:
        old = np.mean([r["reversal"]["old_03"] for r in all_results[sn]])
        new = np.mean([r["reversal"]["new_14"] for r in all_results[sn]])
        print("{:>15} {:>8.0f} {:>8.0f}".format(sn, old, new))

    print("\n" + "=" * 74)
    tf = baselines.get("topoforge", 0)
    vl = baselines.get("vlsi", 1)
    rn = baselines.get("random", 0)
    print("BENCHMARK FINDING (N={}):".format(N))
    print("  TopoForge taught: {:.0f}+/-{:.0f}".format(
        np.mean([r["plastic"]["taught"] for r in all_results["topoforge"]]),
        np.std([r["plastic"]["taught"] for r in all_results["topoforge"]])))
    print("  Random taught:    {:.0f}+/-{:.0f}".format(
        np.mean([r["plastic"]["taught"] for r in all_results["random"]]),
        np.std([r["plastic"]["taught"] for r in all_results["random"]])))
    print("  VLSI taught:      {:.0f}+/-{:.0f}".format(
        np.mean([r["plastic"]["taught"] for r in all_results["vlsi"]]),
        np.std([r["plastic"]["taught"] for r in all_results["vlsi"]])))
    if vl > 0:
        print("  TopoForge/VLSI ratio: {:.1f}x".format(tf / vl))
        print("  TopoForge/Random ratio: {:.2f}x".format(tf / max(rn, 1)))
    print("\n  Consistent with N=900 benchmark (8.6x)? Check the ratio above.")
    print("  If ratio GROWS with N: the effect scales with chip size.")
    print("  If ratio HOLDS: robust across scales.")
    print("  If ratio SHRINKS: toy-scale artifact (honest negative).")

    print("\n  CORRECTED (plasticity-attributable, N={}):".format(N))
    tf_g = growth_baselines.get("topoforge")
    vl_g = growth_baselines.get("vlsi")
    if tf_g is not None and vl_g is not None:
        print("  interleaved growth: {:+.0f}   segregated growth: {:+.0f}".format(tf_g, vl_g))
        if vl_g < 0 < tf_g:
            print("  -> SIGN FLIP at N={}, matching the N=900 result: segregated doesn't".format(N))
            print("     learn weakly, it actively erodes non-mechanistic pre-wiring; only")
            print("     interleaved builds new cross-type structure.")
        elif vl_g > 0 and tf_g > 0:
            print("  -> Both net-positive at this scale; ratio = {:.2f}x -- the sign flip".format(tf_g / vl_g))
            print("     seen at N=900 does NOT replicate here as a sign flip, only as a ratio.")
        else:
            print("  -> Unexpected pattern (both non-positive or interleaved below segregated) --")
            print("     investigate before trusting either N=900 or N={} as representative.".format(N))
    print("\n  Replicate: github.com/bowills-srb/topoforge")
