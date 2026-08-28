"""Exp 38: SpiNeMap as a traceable placement baseline.

Motivation (preprint Section 6, "Baseline fidelity"): our "segregated"
condition is our OWN wire-length heuristic, not the output of a published
mapping tool. SpiNeMap (Balaji et al., arXiv:1909.01843) has no public
code release, so we implement its published two-step algorithm and run it
through the same Placement-Learning Benchmark as our hand-rolled
conditions. This makes the baseline traceable to the actual citation.

SpiNeMap, per the paper:
  1. SpiNeCluster -- partition the network into clusters via greedy
     Kernighan-Lin-style graph partitioning, minimizing inter-cluster
     (global) synapse count, clusters sized to fit a crossbar. We map
     "crossbar" onto our core capacity (NEURONS_PER_CORE) and "network
     synapses" onto the task's connectivity graph (see below).
  2. SpiNePlacer -- place those clusters onto physical cores via Particle
     Swarm Optimization, minimizing a fitness combining energy and spike
     latency. We map both onto our existing squared-wire-length metric
     (energy ~ sum of weight * core-distance).

Both steps are implemented from scratch in numpy (no new dependency; the
KL step is cross-checked against networkx's kernighan_lin_bisection in the
sanity mode). run_life and the benchmark physics are imported VERBATIM
from exp32b_benchmark.py so the comparison is apples-to-apples.

THE KEY MODELING CHOICE -- what graph does SpiNeCluster receive?
A plastic network's cross-type associations do not exist as synapses until
they are LEARNED. So the graph available at map time matters, and we test
both honestly:
  - "population": intra-type synapses only (same-type neurons form a
    population/layer). This is what a mapper knows BEFORE plasticity runs.
  - "functional": intra-type PLUS inter-type association synapses (types
    that co-occur in a PATTERN are wired). This is what a mapper would
    receive only if the to-be-learned associations were declared up front.
Reporting both directly answers the reviewer: is our segregated baseline
representative of a real tool, or a strawman?

Run (staged, audit-before-trust):
  python src/experiments/exp38_spinemap_baseline.py --sanity-cluster
  python src/experiments/exp38_spinemap_baseline.py --sanity-place
  python src/experiments/exp38_spinemap_baseline.py            # full benchmark
"""
import numpy as np
import time
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

# Physics + benchmark harness imported verbatim -- do NOT re-implement.
from exp32b_benchmark import (
    PATTERNS, NC, CORES_X, CORES_Y, N_CORES, NEURONS_PER_CORE, N,
    core_positions, neurons_in_core, run_life, make_placement,
)
from spatial import SpatialGrid


# ============================================================
# The synapse graph SpiNeCluster partitions
# ============================================================
def type_affinity(mode):
    """NC x NC affinity between neuron TYPES.
      population: 1 on the diagonal only (same-type population synapses).
      functional: 1 whenever two types co-occur in a PATTERN (the
                  association synapses the task must realize), plus the
                  diagonal for the within-population synapses.
    """
    A = np.zeros((NC, NC))
    for c in range(NC):
        A[c, c] = 1.0
    if mode == "functional":
        for pat in PATTERNS:
            for a in pat:
                for b in pat:
                    if a != b:
                        A[a, b] = 1.0
    elif mode != "population":
        raise ValueError("mode must be 'population' or 'functional'")
    return A


def build_synapse_graph(cids, mode):
    """N x N symmetric synapse-weight matrix W. W[i,j] is the affinity
    between neurons i and j given their types. Zero diagonal."""
    A = type_affinity(mode)
    W = A[np.ix_(cids, cids)].astype(np.float64)
    np.fill_diagonal(W, 0.0)
    return W


def inter_cluster_cut(W, part):
    """Total inter-cluster synapse weight (the quantity SpiNeCluster
    minimizes). Counts each undirected edge once."""
    same = part[:, None] == part[None, :]
    return 0.5 * (W * (~same)).sum()


# ============================================================
# SpiNeCluster: greedy Kernighan-Lin k-way partitioning
# ============================================================
def kernighan_lin_kway(W, cluster_size, n_clusters, seed=0, max_passes=200,
                       permute=True):
    """Partition N nodes into n_clusters balanced clusters of cluster_size,
    minimizing inter-cluster edge weight, by balance-preserving KL swaps.

    Each pass: recompute the affinity matrix A[n,c] (weight from node n into
    cluster c), then visit every unordered cluster pair (a,b) in random order
    and perform the single best positive-gain swap between their members,
    with node locking so a node moves at most once per pass. Repeat until a
    pass makes no swap. The exact gain of swapping na in a with nb in b is

        dInternal = (A[na,b] - A[na,a]) + (A[nb,a] - A[nb,b]) - 2*W[na,nb]

    (the last term corrects for the na-nb edge, which is counted in both
    node-level gains). Cut = total_weight - internal, so a positive dInternal
    is a cut reduction.

    HISTORY -- the original implementation of this function scored each node's
    single best destination cluster, bucketed the movers by (from, to), and
    paired them off in gain order. That heuristic STALLED on this graph, which
    is massively tie-degenerate (every same-type node has an identical row):
    on type-SORTED input -- exactly what make_placement_spinemap feeds it -- it
    finished 6.4% above the optimal cut, at essentially the random-partition
    cut (KL/random = 0.9965, mean cluster type-purity 0.38), i.e. it was not
    partitioning at all. It happened to reach the optimum on type-SHUFFLED
    input, which is what --sanity-cluster checked, so the failure went unseen.
    See PROJECT_HISTORY, "SpiNeCluster node-order stall". Two changes fix it:
    an exhaustive best-swap search within each cluster pair (rather than
    pairing top-gain movers, which the -2*W[na,nb] correction makes unsound),
    and an internal random node relabeling so the caller's node order cannot
    determine the result. Both --sanity-cluster and --audit-cluster now test
    the sorted input directly.
    """
    N_ = W.shape[0]
    assert cluster_size * n_clusters == N_, "clusters must tile N exactly"
    rng = np.random.default_rng(seed)
    if permute:
        perm = rng.permutation(N_)
        W = W[np.ix_(perm, perm)]
    part = np.repeat(np.arange(n_clusters), cluster_size).astype(int)
    rng.shuffle(part)

    idx = np.arange(N_)
    members = [np.where(part == c)[0] for c in range(n_clusters)]
    pairs = [(a, b) for a in range(n_clusters) for b in range(a + 1, n_clusters)]

    for _pass in range(max_passes):
        oh = np.zeros((N_, n_clusters))
        oh[idx, part] = 1.0
        A = W @ oh                       # A[n,c] = weight from n into cluster c
        locked = np.zeros(N_, dtype=bool)
        n_swaps = 0
        for k in rng.permutation(len(pairs)):
            a, b = pairs[k]
            ma, mb = members[a], members[b]
            ma = ma[~locked[ma]]
            mb = mb[~locked[mb]]
            if len(ma) == 0 or len(mb) == 0:
                continue
            # exact swap gain for every (na, nb) pair in this cluster pair
            G = ((A[ma, b] - A[ma, a])[:, None]
                 + (A[mb, a] - A[mb, b])[None, :]
                 - 2.0 * W[np.ix_(ma, mb)])
            f = int(np.argmax(G))
            i_, j_ = divmod(f, G.shape[1])
            if G[i_, j_] <= 1e-12:
                continue
            na, nb = int(ma[i_]), int(mb[j_])
            part[na], part[nb] = b, a
            members[a] = np.where(part == a)[0]
            members[b] = np.where(part == b)[0]
            locked[na] = locked[nb] = True
            n_swaps += 1
        if n_swaps == 0:
            break
    if permute:
        out = np.empty(N_, dtype=int)
        out[perm] = part
        return out
    return part


def spinecluster(W, cluster_size, n_clusters, seed=0, restarts=10):
    """SpiNeCluster: KL partitioning with multi-restart. Greedy KL can settle
    in a local minimum (see --sanity-cluster: the k=2 case), so we run
    several random starts and keep the lowest-cut partition -- standard
    practice and cheap here."""
    best_part, best_cut = None, np.inf
    for r in range(restarts):
        part = kernighan_lin_kway(W, cluster_size, n_clusters, seed=seed + r)
        cut = inter_cluster_cut(W, part)
        if cut < best_cut:
            best_cut, best_part = cut, part
    return best_part


# ============================================================
# SpiNePlacer: Particle Swarm Optimization over core assignment
# ============================================================
def cluster_traffic(W, part, n_clusters):
    """Inter-cluster synapse weight matrix Wc (n_clusters x n_clusters),
    zero diagonal. Wc[a,b] is the traffic SpiNePlacer must minimize the
    wire cost of."""
    N_ = W.shape[0]
    oh = np.zeros((N_, n_clusters))
    oh[np.arange(N_), part] = 1.0
    Wc = oh.T @ W @ oh
    np.fill_diagonal(Wc, 0.0)
    return Wc


def _placement_energy(assign, Wc, Dcore):
    """Wire energy of an assignment (core slot per cluster): sum over cluster
    pairs of traffic * squared core distance. Matches the benchmark's
    squared-wire-length energy metric."""
    DA = Dcore[np.ix_(assign, assign)]
    return 0.5 * (Wc * DA).sum()


def particle_swarm_placement(Wc, core_pos, seed=0, n_particles=40,
                             iters=250, w=0.72, c1=1.5, c2=1.5):
    """SpiNePlacer: assign each cluster to a physical core (a permutation,
    since #clusters == #cores) minimizing wire energy, via random-key PSO.
    A particle is a real vector in R^K; its rank-order decodes to a
    permutation (cluster -> core slot). Standard gbest PSO on those keys."""
    K = Wc.shape[0]
    Dcore = ((core_pos[:, None, :] - core_pos[None, :, :]) ** 2).sum(-1)
    rng = np.random.default_rng(seed)

    def decode(pos):
        return np.argsort(np.argsort(pos))     # ranks -> bijection 0..K-1

    pos = rng.uniform(0, 1, size=(n_particles, K))
    vel = rng.uniform(-0.1, 0.1, size=(n_particles, K))
    pbest = pos.copy()
    pbest_fit = np.array([_placement_energy(decode(p), Wc, Dcore) for p in pos])
    g = int(np.argmin(pbest_fit))
    gbest = pbest[g].copy()
    gbest_fit = float(pbest_fit[g])
    start_fit = gbest_fit

    for _ in range(iters):
        r1 = rng.uniform(0, 1, size=(n_particles, K))
        r2 = rng.uniform(0, 1, size=(n_particles, K))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = pos + vel
        for i in range(n_particles):
            fit = _placement_energy(decode(pos[i]), Wc, Dcore)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = pos[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = pos[i].copy()
    return decode(gbest), gbest_fit, start_fit


def make_placement_spinemap(mode, seed=0, verbose=False):
    """Full SpiNeMap placement: SpiNeCluster then SpiNePlacer. Deterministic
    for a fixed seed (like the other strategies' placements). Returns
    (coords, cids) in the same format as exp32b.make_placement, with neurons
    placed inside their assigned core's disc via the SAME neurons_in_core
    jitter used by every other strategy -- so geometry is apples-to-apples."""
    core_pos = core_positions()
    cids = np.repeat(np.arange(NC), N // NC).astype(int)   # 180 per type

    W = build_synapse_graph(cids, mode)
    part = spinecluster(W, NEURONS_PER_CORE, N_CORES, seed=seed)
    Wc = cluster_traffic(W, part, N_CORES)
    assign, _, _ = particle_swarm_placement(Wc, core_pos, seed=seed)

    coords = np.zeros((N, 2))
    for c in range(N_CORES):
        members = np.where(part == c)[0]
        core_id = int(assign[c])
        pts = neurons_in_core(core_id, core_pos[core_id])
        coords[members] = pts[:len(members)]
    if verbose:
        print("    [spinemap-{}] cut={:.0f}".format(mode, inter_cluster_cut(W, part)))
    return coords, cids


def cooc_adjacency(coords, cids, radius=5.0):
    """For each association pair, the fraction of source-type neurons that
    have a partner-type neuron within the plasticity radius. This is the
    structural precondition for bridges to form -- checked BEFORE trusting
    any learning numbers (audit-before-trust)."""
    g = SpatialGrid(coords, radius)
    out = {}
    for pat in PATTERNS:
        if len(pat) != 2:
            continue
        a, b = pat
        a_idx = np.where(cids == a)[0]
        hit = 0
        for i in a_idx:
            nbrs = g.within(int(i), radius)
            if any(cids[int(j)] == b for j in nbrs):
                hit += 1
        out[(a, b)] = hit / max(len(a_idx), 1)
    return out


def cluster_component_label(part, cids, n_clusters, mode):
    """Label each cluster by the graph component it belongs to (the set of
    types it contains, collapsed to a component id). Used to check that
    SpiNePlacer groups communicating clusters spatially."""
    if mode == "functional":
        comp_of_type = {0: 0, 3: 0, 1: 1, 4: 1, 2: 2}
    else:
        comp_of_type = {c: c for c in range(NC)}
    labels = np.full(n_clusters, -1, dtype=int)
    for c in range(n_clusters):
        types = cids[part == c]
        labels[c] = comp_of_type[int(np.bincount(types).argmax())]
    return labels


if __name__ == "__main__":
    if "--sanity-place" in sys.argv:
        print("=" * 74)
        print("SANITY CHECK 2: SpiNePlacer (Particle Swarm Optimization)")
        print("=" * 74)
        core_pos = core_positions()
        Dcore = ((core_pos[:, None, :] - core_pos[None, :, :]) ** 2).sum(-1)
        cids_full = np.repeat(np.arange(NC), N // NC)
        np.random.default_rng(0).shuffle(cids_full)

        for mode in ("population", "functional"):
            print("\n--- graph mode: {} ---".format(mode))
            W = build_synapse_graph(cids_full, mode)
            part = spinecluster(W, NEURONS_PER_CORE, N_CORES, seed=0, restarts=4)
            Wc = cluster_traffic(W, part, N_CORES)

            assign, gbest_fit, start_fit = particle_swarm_placement(
                Wc, core_pos, seed=0)

            # random-assignment baseline energy
            rand_e = []
            for s in range(300):
                rp = np.random.default_rng(s).permutation(N_CORES)
                rand_e.append(_placement_energy(rp, Wc, Dcore))
            rand_e = np.array(rand_e)

            print("  PSO energy: start {:.0f} -> final {:.0f}".format(start_fit, gbest_fit))
            print("  random-assignment energy: {:.0f} +/- {:.0f}".format(
                rand_e.mean(), rand_e.std()))
            print("  PSO / random: {:.3f}  (want < 1)".format(
                gbest_fit / max(rand_e.mean(), 1e-9)))

            # spatial grouping: mean core-distance between clusters of the
            # SAME communication component vs DIFFERENT components.
            labels = cluster_component_label(part, cids_full, N_CORES, mode)
            placed = np.empty(N_CORES, dtype=int)
            placed[np.arange(N_CORES)] = assign          # cluster -> core slot
            cpos = core_pos[placed]                       # coord of each cluster
            same_d, diff_d = [], []
            for a in range(N_CORES):
                for b in range(a + 1, N_CORES):
                    d = np.sqrt(((cpos[a] - cpos[b]) ** 2).sum())
                    (same_d if labels[a] == labels[b] else diff_d).append(d)
            print("  mean spatial distance: same-component {:.1f} vs "
                  "different-component {:.1f}".format(
                      np.mean(same_d), np.mean(diff_d)))
            print("  -> communicating clusters {} closer".format(
                "ARE" if np.mean(same_d) < np.mean(diff_d) else "are NOT"))

        print("\nVERDICT: PSO energy well below random, and same-component")
        print("(communicating) clusters placed spatially closer than others.")
        sys.exit(0)

    if "--sanity-cluster" in sys.argv:
        print("=" * 74)
        print("SANITY CHECK 1: SpiNeCluster (Kernighan-Lin partitioning)")
        print("=" * 74)

        # Small, fast test case: 60 neurons, 6/core, 10 clusters, 5 types.
        rng = np.random.default_rng(0)
        n_small, csz, kc = 60, 6, 10
        cids_small = np.repeat(np.arange(NC), n_small // NC)
        rng.shuffle(cids_small)

        for mode in ("population", "functional"):
            print("\n--- graph mode: {} ---".format(mode))
            W = build_synapse_graph(cids_small, mode)
            total_w = 0.5 * W.sum()

            # baseline: many random balanced partitions
            rand_cuts = []
            for s in range(200):
                rp = np.repeat(np.arange(kc), csz)
                np.random.default_rng(s).shuffle(rp)
                rand_cuts.append(inter_cluster_cut(W, rp))
            rand_cuts = np.array(rand_cuts)

            kl_part = spinecluster(W, csz, kc, seed=0)
            kl_cut = inter_cluster_cut(W, kl_part)

            # sizes must stay balanced
            sizes = np.bincount(kl_part, minlength=kc)
            print("  total edge weight in graph : {:.0f}".format(total_w))
            print("  random partition cut       : {:.1f} +/- {:.1f} (mean over 200)".format(
                rand_cuts.mean(), rand_cuts.std()))
            print("  Kernighan-Lin cut          : {:.1f}".format(kl_cut))
            print("  KL / random ratio          : {:.3f}  (want << 1)".format(
                kl_cut / max(rand_cuts.mean(), 1e-9)))
            print("  cluster sizes balanced     : {}  (all == {})".format(
                bool(np.all(sizes == csz)), csz))

            # cross-check the KL gain logic against networkx on a bisection
            try:
                import networkx as nx
                g = nx.Graph()
                g.add_nodes_from(range(n_small))
                ii, jj = np.where(np.triu(W, 1) > 0)
                for a, b in zip(ii, jj):
                    g.add_edge(int(a), int(b), weight=float(W[a, b]))
                part_a, part_b = nx.algorithms.community.kernighan_lin_bisection(
                    g, weight="weight", seed=0)
                bis = np.zeros(n_small, dtype=int)
                for nidx in part_b:
                    bis[nidx] = 1
                nx_cut = inter_cluster_cut(W, bis)
                # our KL restricted to a 2-way partition, same sizes
                our_bis = spinecluster(W, n_small // 2, 2, seed=0)
                our_cut = inter_cluster_cut(W, our_bis)
                print("  [cross-check bisection] networkx cut = {:.1f}, ours = {:.1f}".format(
                    nx_cut, our_cut))
            except Exception as e:
                print("  [cross-check skipped: {}]".format(e))

        # Real-scale optimality: at N=900, k=60 clusters of 15, the block
        # structure has a known optimum (each cluster within one graph
        # component). Confirm KL reaches it before trusting placement.
        print("\n--- real-scale optimality (N={}, {} clusters of {}) ---".format(
            N, N_CORES, NEURONS_PER_CORE))
        print("    Tested on BOTH node orderings. This check originally used only")
        print("    the shuffled ordering, while make_placement_spinemap calls the")
        print("    partitioner with type-SORTED labels -- and the old greedy stalled")
        print("    at the random-partition cut on exactly that input. Never audit an")
        print("    algorithm on an input the deployed path does not use.")

        def per_type_optimum(sizes_per_group):
            # a fully-connected group of g nodes tiled into clusters of csz:
            # cut = C(g,2) - (g/csz)*C(csz,2)
            csz = NEURONS_PER_CORE
            tot = 0.0
            for g in sizes_per_group:
                internal = (g // csz) * (csz * (csz - 1) / 2)
                tot += g * (g - 1) / 2 - internal
            return tot

        ok_real = True
        for order in ("sorted", "shuffled"):
            cids_full = np.repeat(np.arange(NC), N // NC)
            if order == "shuffled":
                np.random.default_rng(0).shuffle(cids_full)
            for mode in ("population", "functional"):
                W = build_synapse_graph(cids_full, mode)
                part = spinecluster(W, NEURONS_PER_CORE, N_CORES, seed=0, restarts=2)
                cut = inter_cluster_cut(W, part)
                if mode == "population":
                    groups = [np.sum(cids_full == c) for c in range(NC)]
                else:
                    groups = [360, 360, 180]  # {0,3}, {1,4}, {2}
                opt = per_type_optimum(groups)
                sizes = np.bincount(part, minlength=N_CORES)
                # random-partition reference: if KL/random ~ 1 the step is inert
                rc = []
                for s in range(30):
                    rp = np.repeat(np.arange(N_CORES), NEURONS_PER_CORE)
                    np.random.default_rng(s).shuffle(rp)
                    rc.append(inter_cluster_cut(W, rp))
                purity = np.mean([np.bincount(cids_full[part == c], minlength=NC).max()
                                  / NEURONS_PER_CORE for c in range(N_CORES)])
                balanced = bool(np.all(sizes == NEURONS_PER_CORE))
                ok_real &= balanced and (cut - opt) / opt < 1e-9
                print("  {:>8}/{:<11}: cut = {:.0f}, optimum = {:.0f}, gap = {:+.2%}, "
                      "KL/random = {:.4f}, purity = {:.3f}, balanced = {}".format(
                          order, mode, cut, opt, (cut - opt) / opt,
                          cut / np.mean(rc), purity, balanced))

        print("\nVERDICT: KL cut should be far below random for both modes,")
        print("sizes balanced, our bisection cut matching networkx's, and the")
        print("real-scale cut AT the theoretical optimum for BOTH node orderings.")
        print("real-scale verdict: {}".format("PASS" if ok_real else "FAIL"))
        sys.exit(0 if ok_real else 1)

    # ============================================================
    # FULL BENCHMARK: SpiNeMap vs our three hand-rolled conditions
    # ============================================================
    import numpy as _np
    print("=" * 74)
    print("EXP 38: SpiNeMap (real algorithm) through the Placement-Learning Benchmark")
    print("{} cores ({}x{}), {} neurons/core, {} total.".format(
        N_CORES, CORES_X, CORES_Y, NEURONS_PER_CORE, N))
    print("=" * 74)

    SEEDS = list(range(int(sys.argv[sys.argv.index("--seeds") + 1]))) \
        if "--seeds" in sys.argv else list(range(10))
    print("Seeds: {}".format(len(SEEDS)))

    # Precompute the two SpiNeMap placements once (deterministic, like the
    # other strategies). This is the expensive-ish step (cluster + PSO).
    print("\nComputing SpiNeMap placements (SpiNeCluster + SpiNePlacer)...")
    t0 = time.time()
    spinemap_placements = {
        "spinemap-pop": make_placement_spinemap("population", seed=0),
        "spinemap-func": make_placement_spinemap("functional", seed=0),
    }
    print("  done in {:.0f}s".format(time.time() - t0))

    strategies = [
        ("random", "Random (unoptimized)"),
        ("vlsi", "Our segregated baseline (hand-rolled wire-length heuristic)"),
        ("topoforge", "Our interleaved condition"),
        ("spinemap-pop", "SpiNeMap on population graph (pre-plasticity info)"),
        ("spinemap-func", "SpiNeMap on functional graph (associations declared)"),
    ]

    def placement_for(name):
        if name in spinemap_placements:
            return spinemap_placements[name]
        return make_placement(name)

    # --- audit-before-trust: structural co-location BEFORE learning ---
    print("\n" + "-" * 74)
    print("Structural audit (before any learning): fraction of source-type")
    print("neurons with an associated-partner neuron within plasticity radius 5.0")
    print("-" * 74)
    print("{:>16} {:>14} {:>14}".format("placement", "0->3 adjacency", "1->4 adjacency"))
    for name, _ in strategies:
        coords, cids = placement_for(name)
        adj = cooc_adjacency(coords, cids)
        print("{:>16} {:>13.1%} {:>13.1%}".format(
            name, adj.get((0, 3), 0), adj.get((1, 4), 0)))

    # --- run the benchmark ---
    all_results = {}
    for name, desc in strategies:
        print("\n  {} - {}".format(name.upper(), desc))
        coords, cids = placement_for(name)
        seed_results = []
        for s in SEEDS:
            t0 = time.time()
            r = run_life(coords, cids, s)
            seed_results.append(r)
            print("    seed {:>2}: plastic_taught={:>5}  ({:.0f}s)".format(
                s, r["plastic"]["taught"], time.time() - t0))
        all_results[name] = seed_results

    # === RESULTS ===
    print("\n" + "=" * 74)
    print("RESULTS: Learning Quality (taught mass after plastic phase)")
    print("=" * 74)
    print("{:>16} {:>14} {:>10} {:>10}".format(
        "placement", "mean+/-std", "vs vlsi", "vs topo"))
    print("-" * 54)
    means = {}
    for name, _ in strategies:
        vals = [r["plastic"]["taught"] for r in all_results[name]]
        means[name] = float(_np.mean(vals))
        print("{:>16} {:>9.0f}+/-{:<4.0f}".format(
            name, _np.mean(vals), _np.std(vals)), end="")
        vs_vlsi = "{:+.0f}%".format((means[name] - means["vlsi"]) / max(means["vlsi"], 1e-9) * 100) \
            if "vlsi" in means else ""
        vs_topo = "{:+.0f}%".format((means[name] - means["topoforge"]) / max(means["topoforge"], 1e-9) * 100) \
            if "topoforge" in means else ""
        print(" {:>10} {:>10}".format(vs_vlsi, vs_topo))

    print("\n" + "=" * 74)
    print("WHERE DOES SpiNeMap LAND?")
    print("=" * 74)
    vlsi_m = means["vlsi"]; topo_m = means["topoforge"]
    span = max(topo_m - vlsi_m, 1e-9)
    for name in ("spinemap-pop", "spinemap-func"):
        frac = (means[name] - vlsi_m) / span
        if frac < 0.15:
            verdict = "resembles our SEGREGATED baseline"
        elif frac > 0.85:
            verdict = "resembles INTERLEAVED"
        else:
            verdict = "lands BETWEEN segregated and interleaved"
        print("  {:>14}: {:.0f}  ({:.0%} of the way vlsi->topoforge)  -> {}".format(
            name, means[name], frac, verdict))
    print("\nInterpretation for the paper's 'invented baseline' limitation:")
    print("  - SpiNeMap on the PRE-PLASTICITY (population) graph is the honest")
    print("    map-time case: the to-be-learned associations are not yet synapses.")
    print("  - SpiNeMap on the FUNCTIONAL graph assumes the associations are")
    print("    declared up front (an upper bound a real plastic system rarely has).")
    print("\n  Replicate: github.com/bowills-srb/topoforge")
