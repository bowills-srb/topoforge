"""Exp 36: Locality WITHOUT Geometry -- the sharpest objection test.
Does the interleaving-beats-segregation effect require SPATIAL embedding
specifically, or does it appear under ANY local-vs-global connectivity
constraint, with no coordinates, no distance, no geometry at all?

Design: replace physical cores + Euclidean deposit radius with pure
GRAPH COMMUNITIES + adjacency-list neighbors. A neuron's "neighbors"
are simply the other neurons assigned to the same community -- an
arbitrary label, not a physical location. No x,y coordinates exist
anywhere in this file. No distance is ever computed.

If segregated still loses to interleaved by a similar ratio here, the
effect is about LOCAL vs GLOBAL connectivity generically -- "spatial
topology" is not the most precise description of the mechanism, and
the paper's framing needs correcting. If the penalty disappears without
geometry, something about physical space specifically matters, which
would also be an important, honest finding.

Run: python src/experiments/exp36_locality_without_geometry.py
"""
import numpy as np
import time
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState

PATTERNS = [(0, 3), (1, 4), (2,)]
NC = 5
N_COMMUNITIES = 60
NEURONS_PER_COMMUNITY = 15
N = N_COMMUNITIES * NEURONS_PER_COMMUNITY  # 900, matches original PLB scale
STEPS_FROZEN = 400
STEPS_PLASTIC = 400
TOTAL = STEPS_FROZEN + STEPS_PLASTIC


def make_community_structure():
    """Pure graph partition. NO coordinates. Node i belongs to
    community i // NEURONS_PER_COMMUNITY -- an arbitrary label."""
    community_id = np.repeat(np.arange(N_COMMUNITIES), NEURONS_PER_COMMUNITY)
    return community_id


def assign_types(community_id, strategy):
    """Assigns one of NC 'functional types' to each neuron.
    NO spatial reasoning anywhere -- purely which community a node is in.
    """
    rng = np.random.default_rng(42)
    cids = np.zeros(N, dtype=int)

    if strategy == "segregated":
        # each community gets exactly ONE type (12 communities per type)
        communities_per_type = N_COMMUNITIES // NC
        community_type = []
        for t in range(NC):
            community_type += [t] * communities_per_type
        while len(community_type) < N_COMMUNITIES:
            community_type.append(NC - 1)
        for i in range(N):
            cids[i] = community_type[community_id[i]]

    elif strategy == "interleaved":
        # each community gets a proportional MIX of all types
        for c in range(N_COMMUNITIES):
            members = np.where(community_id == c)[0]
            local_types = list(range(NC)) * (NEURONS_PER_COMMUNITY // NC)
            while len(local_types) < len(members):
                local_types.append(rng.integers(0, NC))
            rng.shuffle(local_types)
            for idx, node in enumerate(members):
                cids[node] = local_types[idx]

    elif strategy == "random":
        pool = np.repeat(np.arange(NC), N // NC)
        rng.shuffle(pool)
        cids = pool.copy()

    return cids


def build_neighbor_lists(community_id):
    """A node's neighbors = every OTHER node in the same community.
    This is the ENTIRE locality structure. No coordinates, no distance,
    no radius -- just: are you labeled with the same community index."""
    nbr = [[] for _ in range(N)]
    for c in range(N_COMMUNITIES):
        members = np.where(community_id == c)[0].tolist()
        for i in members:
            nbr[i] = [j for j in members if j != i]
    return nbr


def run_life(cids, nbr, seed, w_c=0.01):
    """Same learning dynamics as the spatial engine, but with NO
    distance term anywhere in the scoring function -- candidates are
    restricted to graph neighbors only, scored by V + w_c*C alone."""
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    n_edges = N * 10
    src = rng2.integers(0, N, n_edges); dst = rng2.integers(0, N, n_edges)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(3)
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N // 2

    for t in range(TOTAL):
        p = (t // 20) % 3
        in_plastic = t >= STEPS_FROZEN
        inp = rng3.uniform(0, 0.02, N)
        if (t % 20) < 5:
            for c in PATTERNS[p]:
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
                    # NO DISTANCE TERM. Score = V + w_c*C. That's it.
                    # No coordinates exist in this file, so none can be used.
                    vp = np.maximum(np.array([V.get(int(a), int(b)) for a, b in zip(ci, cj)]), 0)
                    cp = np.array([C.get(int(a), int(b)) for a, b in zip(ci, cj)])
                    score = vp + w_c * cp
                    pos = score > 0
                    ci, cj, score = ci[pos], cj[pos], score[pos]
                    if len(ci) > 0:
                        order = np.argsort(score, kind="stable")[::-1][:len(cold)]
                        n2 = min(len(cold), len(order))
                        src[cold[:n2]] = ci[order[:n2]]; dst[cold[:n2]] = cj[order[:n2]]
                        out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    M = np.zeros((NC, NC), dtype=int)
    np.add.at(M, (cids[src], cids[dst]), 1)
    taught = M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]
    return taught


if __name__ == "__main__":
    print("=" * 74)
    print("EXP 36: LOCALITY WITHOUT GEOMETRY")
    print("No coordinates. No distance. No radius. Pure graph community")
    print("membership defines who can ever become neighbors.")
    print("{} communities x {} neurons/community, N={}".format(
        N_COMMUNITIES, NEURONS_PER_COMMUNITY, N))
    print("=" * 74)

    SEEDS = list(range(10))
    strategies = [
        ("segregated", "Each community = one type (naive baseline analog)"),
        ("interleaved", "Each community = mix of all types"),
        ("random", "Type assignment independent of community"),
    ]

    all_results = {}
    for strat_name, description in strategies:
        print("\n  {} -- {}".format(strat_name.upper(), description))
        community_id = make_community_structure()
        cids = assign_types(community_id, strat_name)
        nbr = build_neighbor_lists(community_id)

        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            taught = run_life(cids, nbr, s)
            seed_results.append(taught)
            print("    seed {:>2}: taught={:>5}  ({:.0f}s)".format(s, taught, time.time() - t0))
        all_results[strat_name] = np.array(seed_results, dtype=float)

    print("\n" + "=" * 74)
    print("RESULTS: Learning Quality With ZERO Spatial Embedding")
    print("{:>15} {:>16}".format("strategy", "taught (mean+/-std)"))
    print("-" * 34)
    means = {}
    for strat_name, _ in strategies:
        vals = all_results[strat_name]
        means[strat_name] = vals.mean()
        print("{:>15} {:>10.0f}+/-{:<5.0f}".format(strat_name, vals.mean(), vals.std()))

    print("\n" + "=" * 74)
    seg = means["segregated"]
    inter = means["interleaved"]
    ratio = inter / max(seg, 1)
    print("VERDICT:")
    print("  Interleaved/Segregated ratio (NO geometry): {:.2f}x".format(ratio))
    print("  Compare to spatial PLB result: ~4.0x at N=900")
    print("")
    if ratio > 2.0:
        print("  The effect SURVIVES with zero spatial embedding.")
        print("  This means the mechanism is LOCAL vs GLOBAL connectivity,")
        print("  not specifically about physical/spatial geometry.")
        print("  The paper's 'spatial topology' framing should be corrected")
        print("  to something like 'local connectivity structure' -- geometry")
        print("  is one way to implement locality, not the necessary cause.")
    else:
        print("  The effect WEAKENS OR DISAPPEARS without geometry.")
        print("  This would suggest physical/spatial embedding specifically")
        print("  matters beyond generic local-vs-global connectivity --")
        print("  a more interesting and more defensible claim for the paper.")
