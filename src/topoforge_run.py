"""TopoForge CLI — describe your problem, get an answer.
Usage: python topoforge_run.py config.yaml
       python topoforge_run.py config.yaml --custom my_placement.csv
"""
import numpy as np
import time
import sys
import os
import json

sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

# ============================================================
# CONFIG PARSER (reads YAML-like simple format or JSON)
# ============================================================
def parse_config(path):
    """Parse a simple JSON config file."""
    with open(path) as f:
        cfg = json.load(f)
    return cfg

def default_config():
    return {
        "hardware": {"cores_x": 8, "cores_y": 8, "neurons_per_core": 15},
        "neuron_types": 5,
        "associations": [
            {"types": [0, 3], "reward": 1.0},
            {"types": [1, 4], "reward": 0.0},
            {"types": [2], "reward": -1.0}
        ],
        "test": {
            "strategies": ["segregated", "interleaved", "random"],
            "seeds": 5,
            "steps_frozen": 400,
            "steps_plastic": 400,
            "steps_reversal": 400
        }
    }

# ============================================================
# PLACEMENT GENERATORS
# ============================================================
def make_core_grid(cx, cy, spacing=5.0):
    positions = []
    for y in range(cy):
        for x in range(cx):
            positions.append([x * spacing, y * spacing])
    return np.array(positions)

def place_neurons(core_pos, npc, n_types, strategy, custom_csv=None):
    n_cores = len(core_pos)
    N = n_cores * npc
    all_coords = []
    all_cids = []
    
    if custom_csv is not None:
        data = np.loadtxt(custom_csv, delimiter=",", skiprows=1)
        return data[:, 1:3], data[:, 3].astype(int)
    
    rng_place = np.random.default_rng(42)
    
    if strategy == "segregated":
        cores_per_type = n_cores // n_types
        core_types = []
        for t in range(n_types):
            for _ in range(cores_per_type):
                core_types.append(t)
        while len(core_types) < n_cores:
            core_types.append(n_types - 1)
        for core_id in range(n_cores):
            rng_c = np.random.default_rng(core_id + 1000)
            cx, cy = core_pos[core_id]
            for _ in range(npc):
                th, r = rng_c.uniform(0, 2*np.pi), rng_c.uniform(0, 1.5)
                all_coords.append([cx + r*np.cos(th), cy + r*np.sin(th)])
                all_cids.append(core_types[core_id])
    
    elif strategy == "interleaved":
        for core_id in range(n_cores):
            rng_c = np.random.default_rng(core_id + 1000)
            cx, cy = core_pos[core_id]
            local = list(range(n_types)) * (npc // n_types)
            while len(local) < npc:
                local.append(rng_place.integers(0, n_types))
            rng_place.shuffle(local)
            for i in range(npc):
                th, r = rng_c.uniform(0, 2*np.pi), rng_c.uniform(0, 1.5)
                all_coords.append([cx + r*np.cos(th), cy + r*np.sin(th)])
                all_cids.append(int(local[i]))
    
    elif strategy == "random":
        cids_pool = np.repeat(np.arange(n_types), N // n_types)
        while len(cids_pool) < N:
            cids_pool = np.append(cids_pool, rng_place.integers(0, n_types))
        rng_place.shuffle(cids_pool)
        idx = 0
        for core_id in range(n_cores):
            rng_c = np.random.default_rng(core_id + 1000)
            cx, cy = core_pos[core_id]
            for _ in range(npc):
                th, r = rng_c.uniform(0, 2*np.pi), rng_c.uniform(0, 1.5)
                all_coords.append([cx + r*np.cos(th), cy + r*np.sin(th)])
                all_cids.append(int(cids_pool[idx]))
                idx += 1
    
    return np.array(all_coords), np.array(all_cids)

# ============================================================
# SIMULATION ENGINE
# ============================================================
def run_simulation(coords, cids, seed, cfg):
    n_types = cfg["neuron_types"]
    assocs = cfg["associations"]
    test = cfg["test"]
    N = len(coords)
    sf = test["steps_frozen"]
    sp = test["steps_plastic"]
    sr = test.get("steps_reversal", 400)
    total = sf + sp + sr
    
    patterns = [tuple(a["types"]) for a in assocs]
    rewards_pre = {i: a["reward"] for i, a in enumerate(assocs)}
    rewards_post = {}
    for i, a in enumerate(assocs):
        if a["reward"] > 0:
            rewards_post[i] = 0.0
        elif a["reward"] == 0 and i == 1:
            rewards_post[i] = 1.0
        else:
            rewards_post[i] = a["reward"]
    
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    n_edges = N * 10
    src = rng2.integers(0, N, n_edges); dst = rng2.integers(0, N, n_edges)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(len(patterns))
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g = SpatialGrid(coords, 5.0)
    nbr = [g.within(i, 5.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N // 2
    results = {}
    
    for t in range(total):
        p = (t // 20) % len(patterns)
        in_plastic = t >= sf
        in_reversal = t >= sf + sp
        pat = patterns[p]
        inp = rng3.uniform(0, 0.02, N)
        if (t % 20) < 5:
            for c in pat:
                inp[cids == c] += 0.5
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
            rwd = rewards_post if in_reversal else rewards_pre
            base = rwd.get(p, 0.0)
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
        for checkpoint in [sf, sf + sp, total]:
            if t + 1 == checkpoint:
                M = np.zeros((n_types, n_types), dtype=int)
                np.add.at(M, (cids[src], cids[dst]), 1)
                wire_e = D2[src, dst].sum()
                taught = sum(M[a, b] + M[b, a] for a_cfg in assocs
                            if a_cfg["reward"] > 0
                            for a, b in [tuple(a_cfg["types"][:2])]
                            if len(a_cfg["types"]) >= 2)
                phase = {sf: "frozen", sf + sp: "plastic", total: "reversal"}[checkpoint]
                results[phase] = {"taught": taught, "energy": wire_e}
    return results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        cfg = parse_config(sys.argv[1])
        config_name = os.path.basename(sys.argv[1]).replace(".json", "")
    else:
        cfg = default_config()
        config_name = "default_benchmark"
    
    custom_csv = None
    if "--custom" in sys.argv:
        idx = sys.argv.index("--custom")
        if idx + 1 < len(sys.argv):
            custom_csv = sys.argv[idx + 1]
    
    hw = cfg["hardware"]
    test = cfg["test"]
    n_types = cfg["neuron_types"]
    core_pos = make_core_grid(hw["cores_x"], hw["cores_y"])
    N = len(core_pos) * hw["neurons_per_core"]
    
    print("=" * 60)
    print("TopoForge v1.0 — Placement-Learning Analysis")
    print("=" * 60)
    print("Config: {}".format(config_name))
    print("Hardware: {}x{} cores, {} neurons/core, {} total".format(
        hw["cores_x"], hw["cores_y"], hw["neurons_per_core"], N))
    print("Types: {}  Seeds: {}".format(n_types, test["seeds"]))
    print("=" * 60)
    
    strategies = test.get("strategies", ["segregated", "interleaved", "random"])
    if custom_csv:
        strategies.append("custom")
    
    all_results = {}
    for strat in strategies:
        print("\n  Testing: {} ...".format(strat.upper()))
        seed_results = []
        for s in range(test["seeds"]):
            t0 = time.time()
            if strat == "custom":
                coords, cids = place_neurons(core_pos, hw["neurons_per_core"],
                                            n_types, strat, custom_csv)
            else:
                coords, cids = place_neurons(core_pos, hw["neurons_per_core"],
                                            n_types, strat)
            r = run_simulation(coords, cids, s, cfg)
            seed_results.append(r)
            print("    seed {:>2}: learned={:>5}  energy={:>10,.0f}  ({:.0f}s)".format(
                s, r["plastic"]["taught"], r["plastic"]["energy"],
                time.time() - t0))
        all_results[strat] = seed_results
    
    # === REPORT ===
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nLearning Quality (higher = better)")
    print("{:>15} {:>12} {:>10}".format("strategy", "mean+/-std", "vs best"))
    print("-" * 40)
    means = {}
    for strat in strategies:
        vals = [r["plastic"]["taught"] for r in all_results[strat]]
        m, s = np.mean(vals), np.std(vals)
        means[strat] = m
        print("{:>15} {:>7.0f}+/-{:<4.0f}".format(strat, m, s))
    
    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    if means[worst] > 0:
        ratio = means[best] / means[worst]
        print("\n  Best: {} ({:.0f})".format(best, means[best]))
        print("  Worst: {} ({:.0f})".format(worst, means[worst]))
        print("  Ratio: {:.1f}x".format(ratio))
    
    print("\nWire Energy (lower = cheaper)")
    print("{:>15} {:>12}".format("strategy", "mean"))
    print("-" * 30)
    for strat in strategies:
        vals = [r["plastic"]["energy"] for r in all_results[strat]]
        print("{:>15} {:>10,.0f}".format(strat, np.mean(vals)))
    
    if "reversal" in all_results[strategies[0]][0]:
        print("\nAdaptation After Workload Change")
        print("{:>15} {:>10}".format("strategy", "taught"))
        print("-" * 28)
        for strat in strategies:
            vals = [r["reversal"]["taught"] for r in all_results[strat]]
            print("{:>15} {:>8.0f}".format(strat, np.mean(vals)))
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("  Use {} placement.".format(best.upper()))
    if best != worst and means[worst] > 0:
        print("  Expected improvement over {}: {:.0f}%".format(
            worst, (means[best] - means[worst]) / means[worst] * 100))
    print("\n  Replicate: github.com/bowills-srb/topoforge")
