"""Exp 33: TopoForge x Catalyst-SHD Hybrid
Real architecture shape from catalyst-benchmarks (Shulayev Barnes 2026):
700 input channels -> 1536 recurrent hidden -> 20 output classes.
Total ~2256 neurons, mapped onto a core grid.
Tests: does our placement-learning penalty hold on a real, externally-
designed multi-class architecture, not just our toy 5-cluster world?

Note: this uses TopoForge's own plasticity/RPE simulation, not Henry's
actual PyTorch/FPGA pipeline (which we don't have access to). It tests
whether our placement principle generalizes to a real network SHAPE.
Run: python src/exp33_shd_topology.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid

# Real architecture proportions from catalyst-benchmarks SHD model
N_INPUT = 700
N_HIDDEN = 1536
N_OUTPUT = 20
N = N_INPUT + N_HIDDEN + N_OUTPUT  # 2256

# Scale down for laptop feasibility while preserving proportions
SCALE = 0.35
N_INPUT_S = int(N_INPUT * SCALE)    # 245
N_HIDDEN_S = int(N_HIDDEN * SCALE)  # 537
N_OUTPUT_S = int(N_OUTPUT * SCALE) if int(N_OUTPUT*SCALE) >= 5 else 20
N_S = N_INPUT_S + N_HIDDEN_S + N_OUTPUT_S

CORES_X, CORES_Y = 16, 8
N_CORES = CORES_X * CORES_Y
# distribute neurons across cores as evenly as possible
NEURONS_PER_CORE = max(1, N_S // N_CORES)
N_FINAL = NEURONS_PER_CORE * N_CORES

print("Architecture: {}->{}->{} scaled to {}->{}->{} (total N={})".format(
    N_INPUT, N_HIDDEN, N_OUTPUT, N_INPUT_S, N_HIDDEN_S, N_OUTPUT_S, N_FINAL))

STEPS_FROZEN = 300
STEPS_PLASTIC = 500
TOTAL = STEPS_FROZEN + STEPS_PLASTIC

# 3 "layer types": 0=input, 1=hidden, 2=output
LAYER_INPUT, LAYER_HIDDEN, LAYER_OUTPUT = 0, 1, 2

def core_positions():
    positions = []
    for cy in range(CORES_Y):
        for cx in range(CORES_X):
            positions.append([cx * 5.0, cy * 5.0])
    return np.array(positions)

def neurons_in_core(core_id, core_pos, npc):
    rng = np.random.default_rng(core_id + 1000)
    cx, cy = core_pos
    pts = []
    for _ in range(npc):
        th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 1.5)
        pts.append([cx + r*np.cos(th), cy + r*np.sin(th)])
    return np.array(pts)

def make_placement(strategy):
    """
    Build a network with N_INPUT_S input neurons, N_HIDDEN_S hidden,
    N_OUTPUT_S output. Layer identity = 'cid' (0=input,1=hidden,2=output).
    """
    core_pos = core_positions()
    all_coords = []
    all_cids = []

    # build the layer-id pool matching real proportions
    layer_pool = ([LAYER_INPUT] * N_INPUT_S +
                   [LAYER_HIDDEN] * N_HIDDEN_S +
                   [LAYER_OUTPUT] * N_OUTPUT_S)
    while len(layer_pool) < N_FINAL:
        layer_pool.append(LAYER_HIDDEN)  # pad with hidden (most common)
    layer_pool = np.array(layer_pool[:N_FINAL])

    if strategy == "random":
        rng = np.random.default_rng(42)
        pool = layer_pool.copy()
        rng.shuffle(pool)
        idx = 0
        for core_id in range(N_CORES):
            pts = neurons_in_core(core_id, core_pos[core_id], NEURONS_PER_CORE)
            for p in pts:
                all_coords.append(p)
                all_cids.append(int(pool[idx]))
                idx += 1

    elif strategy == "segregated":
        # input neurons on first block of cores, hidden in middle, output at end
        # this mimics a naive "layer = physical region" mapping
        n_input_cores = max(1, int(N_CORES * (N_INPUT_S / N_FINAL)))
        n_output_cores = max(1, int(N_CORES * (N_OUTPUT_S / N_FINAL)))
        n_hidden_cores = N_CORES - n_input_cores - n_output_cores
        core_layer = ([LAYER_INPUT] * n_input_cores +
                      [LAYER_HIDDEN] * n_hidden_cores +
                      [LAYER_OUTPUT] * n_output_cores)
        while len(core_layer) < N_CORES:
            core_layer.append(LAYER_HIDDEN)
        for core_id in range(N_CORES):
            pts = neurons_in_core(core_id, core_pos[core_id], NEURONS_PER_CORE)
            for p in pts:
                all_coords.append(p)
                all_cids.append(core_layer[core_id])

    elif strategy == "interleaved":
        # every core gets a proportional mix of input/hidden/output
        rng = np.random.default_rng(42)
        # proportions per core, rounded
        prop_input = N_INPUT_S / N_FINAL
        prop_output = N_OUTPUT_S / N_FINAL
        for core_id in range(N_CORES):
            pts = neurons_in_core(core_id, core_pos[core_id], NEURONS_PER_CORE)
            n_in = max(1, round(NEURONS_PER_CORE * prop_input)) if prop_input > 0 else 0
            n_out = max(1, round(NEURONS_PER_CORE * prop_output)) if prop_output > 0 else 0
            n_hid = max(0, NEURONS_PER_CORE - n_in - n_out)
            local = [LAYER_INPUT]*n_in + [LAYER_OUTPUT]*n_out + [LAYER_HIDDEN]*n_hid
            while len(local) < NEURONS_PER_CORE:
                local.append(LAYER_HIDDEN)
            local = local[:NEURONS_PER_CORE]
            rng.shuffle(local)
            for i, p in enumerate(pts):
                all_coords.append(p)
                all_cids.append(int(local[i]))

    return np.array(all_coords), np.array(all_cids)

def run_life(coords, cids, seed):
    """
    Simplified multi-class task: 'classes' cycle through, each class
    fires a subset of input neurons -> should learn to activate the
    corresponding output-layer bridge. Tests input->output learnability
    under each placement, mediated by hidden layer.
    """
    N_local = len(coords)
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    n_edges = N_local * 8
    src = rng2.integers(0, N_local, n_edges)
    dst = rng2.integers(0, N_local, n_edges)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    inhib = rng2.random(N_local) < 0.20
    v = np.zeros(N_local); refrac = np.zeros(N_local, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)

    input_idx = np.where(cids == LAYER_INPUT)[0]
    output_idx = np.where(cids == LAYER_OUTPUT)[0]

    n_classes = 3  # simplified: 3 classes instead of 20 for tractability
    Rhat = np.zeros(n_classes)

    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g = SpatialGrid(coords, 5.0)
    nbr = [g.within(i, 5.0) for i in range(N_local)]
    out_t = [[] for _ in range(N_local)]; out_w = [[] for _ in range(N_local)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N_local // 3

    # class -> which input subset fires, which output neuron should respond
    rng_task = np.random.default_rng(99)
    class_input_subset = []
    class_output_target = []
    for c in range(n_classes):
        subset_size = max(1, len(input_idx) // n_classes)
        subset = rng_task.choice(input_idx, size=subset_size, replace=False)
        class_input_subset.append(subset)
        if len(output_idx) > 0:
            target = output_idx[c % len(output_idx)]
        else:
            target = 0
        class_output_target.append(target)

    for t in range(TOTAL):
        c = (t // 20) % n_classes
        in_plastic = t >= STEPS_FROZEN
        inp = rng3.uniform(0, 0.02, N_local)
        if (t % 20) < 5:
            inp[class_input_subset[c]] += 0.5

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
            target = class_output_target[c]
            target_fired = 1.0 if (target in f) else 0.0
            base = target_fired * 1.0
            delta = base - Rhat[c]
            if abs(delta) > 1e-9:
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0: V.deposit(key[0], key[1], delta * ev)
            Rhat[c] += 0.15 * delta

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
                        out_t = [[] for _ in range(N_local)]; out_w = [[] for _ in range(N_local)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    # readout: how much input->output structure exists per class
    total_io_bridges = 0
    for c in range(n_classes):
        subset = set(int(x) for x in class_input_subset[c])
        target = int(class_output_target[c])
        for si, di in zip(src, dst):
            if int(si) in subset and int(di) == target:
                total_io_bridges += 1
            if int(di) in subset and int(si) == target:
                total_io_bridges += 1
    wire_e = D2[src, dst].sum()
    return total_io_bridges, wire_e

if __name__ == "__main__":
    print("=" * 74)
    print("EXP 33: TOPOFORGE x CATALYST-SHD HYBRID")
    print("Real architecture shape (Shulayev Barnes 2026, catalyst-benchmarks)")
    print("700->1536->20 scaled to {}->{}->{} (N={})".format(
        N_INPUT_S, N_HIDDEN_S, N_OUTPUT_S, N_FINAL))
    print("Testing: does the placement-learning penalty hold on a REAL,")
    print("externally-designed multi-layer architecture, not our toy world?")
    print("=" * 74)

    SEEDS = [0, 1, 2, 3, 4]
    strategies = [
        ("segregated", "Layer-blocked (naive physical layer mapping)"),
        ("interleaved", "TopoForge (input/hidden/output mixed per core)"),
        ("random", "Random"),
    ]

    all_results = {}
    for strat_name, description in strategies:
        print("\n  {} — {}".format(strat_name.upper(), description))
        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            coords, cids = make_placement(strat_name)
            io_bridges, energy = run_life(coords, cids, s)
            seed_results.append((io_bridges, energy))
            print("    seed {}: io_bridges={:>4}  energy={:>10,.0f}  ({:.0f}s)".format(
                s, io_bridges, energy, time.time() - t0))
        all_results[strat_name] = seed_results

    print("\n" + "=" * 74)
    print("RESULTS: Input->Output Learnability by Placement")
    print("{:>14} {:>16} {:>14}".format("strategy", "io_bridges", "energy"))
    print("-" * 46)
    means = {}
    for sn, _ in strategies:
        vals = [r[0] for r in all_results[sn]]
        m, s = np.mean(vals), np.std(vals)
        means[sn] = m
        e = np.mean([r[1] for r in all_results[sn]])
        print("{:>14} {:>9.1f}+/-{:<4.1f} {:>12,.0f}".format(sn, m, s, e))

    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    print("\n" + "=" * 74)
    if means[worst] > 0:
        ratio = means[best] / means[worst]
        print("VERDICT: {} beats {} by {:.1f}x on a REAL SHD-shaped architecture".format(
            best, worst, ratio))
    else:
        print("VERDICT: {} produced measurable learning; {} produced none".format(
            best, worst))
    print("\nThis architecture shape is from Shulayev Barnes 2026")
    print("(catalyst-neuromorphic/catalyst-benchmarks), used here to test")
    print("generalization of the TopoForge placement-learning finding")
    print("beyond our synthetic 5-cluster world.")
