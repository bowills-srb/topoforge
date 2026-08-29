"""Exp 48: how much foreknowledge does an association-aware mapper actually
need? A usefulness experiment, not a truth experiment.

MOTIVATION. Every result so far in this project (through Section 4.10)
establishes THAT placement matters and WHY. None of them establish
whether a real mapper could act on that fact under realistic conditions.
Exp 38/44 only tested two extremes: SpiNeMap given the "population" graph
(zero foreknowledge of which populations will need to associate -- the
honest map-time case) and the "functional" graph (complete foreknowledge
-- an upper bound no real deployment has). No real mapper has either.
This experiment asks the practically decisive question: if a mapper knows
only a FRACTION of the true future associations in advance, how much of
the functional graph's benefit does it still capture? If a small amount
of foreknowledge captures most of the benefit, this becomes something a
real tool builder could act on today (partial profiling, or an
association hint an engineer can specify without complete certainty). If
it requires near-complete knowledge, the practical recommendation is
weaker than Section 4.10's framing suggests.

DESIGN. Generalizes exp38's `type_affinity`: population uses a
cross-type affinity of 0 (only same-type synapses in the graph SpiNeCluster
sees), functional uses 1 (full association weight declared). This swaps in
a continuous `knowledge_frac` for that hard 0/1 -- the cross-type affinity
SpiNeCluster is given for the two association pairs, everything else about
the SpiNeCluster + SpiNePlacer pipeline held identical. This is a
WEIGHT-CONFIDENCE model of partial knowledge (the mapper believes the
association exists with some strength/confidence), not an EDGE-SUBSAMPLING
model (the mapper has observed some fraction of actual future synapses and
is fully certain about those, blind to the rest) -- the two are not
guaranteed to produce identical results, and edge-subsampling is flagged
as a natural follow-up, not run here.

PREDICTIONS, registered before running:
  (P1) Growth increases monotonically with knowledge_frac (more accurate
       information cannot hurt a mapper that is free to ignore it).
  (P2) THE REAL QUESTION: is the curve concave (most of the benefit
       captured by partial knowledge, e.g. >=50% of the population-to-
       functional gap closed by knowledge_frac<=0.5) or closer to linear/
       convex (you need to be nearly certain before it helps much)? No
       specific shape is assumed in advance -- this is what the sweep is
       for. The practical read-out is the knowledge_frac at which 50% and
       90% of the gap are closed.

Run: python src/experiments/exp48_partial_knowledge.py
"""
import numpy as np
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats

from exp32b_benchmark import NC, N, N_CORES, NEURONS_PER_CORE, PATTERNS, run_life, make_placement
from exp38_spinemap_baseline import (
    core_positions, neurons_in_core, spinecluster, cluster_traffic,
    particle_swarm_placement, inter_cluster_cut,
)

SEEDS = list(range(8))
KNOWLEDGE_FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def type_affinity_partial(knowledge_frac):
    """Generalizes exp38.type_affinity: diagonal always 1 (same-type
    synapses, always known), off-diagonal association pairs weighted by
    knowledge_frac instead of a hard 0 (population) / 1 (functional)."""
    A = np.zeros((NC, NC))
    for c in range(NC):
        A[c, c] = 1.0
    for pat in PATTERNS:
        for a in pat:
            for b in pat:
                if a != b:
                    A[a, b] = knowledge_frac
    return A


def build_synapse_graph_partial(cids, knowledge_frac):
    A = type_affinity_partial(knowledge_frac)
    W = A[np.ix_(cids, cids)].astype(np.float64)
    np.fill_diagonal(W, 0.0)
    return W


def make_placement_spinemap_partial(knowledge_frac, seed=0, verbose=False):
    core_pos = core_positions()
    cids = np.repeat(np.arange(NC), N // NC).astype(int)
    W = build_synapse_graph_partial(cids, knowledge_frac)
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
        print("    [spinemap-k{:.2f}] cut={:.0f}".format(knowledge_frac, inter_cluster_cut(W, part)))
    return coords, cids


def growth_of(coords, cids, seeds=SEEDS):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


if __name__ == "__main__":
    print("=" * 78)
    print("EXP 48: partial foreknowledge -- how much does a mapper actually need?")
    print("=" * 78)

    print("\n[0] Sanity: knowledge_frac=0.0 and 1.0 match exp38's population/functional")
    from exp38_spinemap_baseline import make_placement_spinemap
    c_pop, i_pop = make_placement_spinemap("population", seed=0)
    c_pop2, i_pop2 = make_placement_spinemap_partial(0.0, seed=0)
    c_fn, i_fn = make_placement_spinemap("functional", seed=0)
    c_fn2, i_fn2 = make_placement_spinemap_partial(1.0, seed=0)
    print("  knowledge_frac=0.0 matches 'population' exactly: {}".format(
        np.array_equal(c_pop, c_pop2) and np.array_equal(i_pop, i_pop2)))
    print("  knowledge_frac=1.0 matches 'functional' exactly: {}".format(
        np.array_equal(c_fn, c_fn2) and np.array_equal(i_fn, i_fn2)))

    print("\n[1] Growth vs knowledge_frac, {} seeds each".format(len(SEEDS)))
    results = {}
    for kf in KNOWLEDGE_FRACS:
        coords, cids = make_placement_spinemap_partial(kf, seed=0, verbose=True)
        g = growth_of(coords, cids)
        results[kf] = g
        print("  knowledge_frac={:>5.0%}  growth={:+8.1f} +/- {:<6.1f}".format(kf, g.mean(), g.std()))

    print("\n" + "=" * 78)
    print("[2] Recovery: what fraction of the population->functional gap does each level close?")
    print("=" * 78)
    pop_g = results[0.0].mean()
    fn_g = results[1.0].mean()
    span = fn_g - pop_g
    print("  gap: population {:+.1f} -> functional {:+.1f}  (span {:.1f})".format(pop_g, fn_g, span))
    print("  {:>14} {:>10} {:>12} {:>14}".format("knowledge", "growth", "%recovered", "Welch p vs pop"))
    frac_at_50, frac_at_90 = None, None
    for kf in KNOWLEDGE_FRACS:
        g = results[kf]
        pct = (g.mean() - pop_g) / span
        t, p = stats.ttest_ind(g, results[0.0], equal_var=False)
        print("  {:>13.0%} {:>+10.1f} {:>11.1%} {:>14.2e}".format(kf, g.mean(), pct, p))
        if frac_at_50 is None and pct >= 0.5:
            frac_at_50 = kf
        if frac_at_90 is None and pct >= 0.9:
            frac_at_90 = kf

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if frac_at_50 is not None and frac_at_50 <= 0.5:
        print("  CONCAVE / practically actionable: knowledge_frac={:.0%} already recovers".format(frac_at_50))
        print("  >=50% of the population-to-functional gap. Partial, uncertain foreknowledge")
        print("  is enough to capture most of the benefit -- a real mapper does not need")
        print("  complete certainty about future associations to be worth building.")
    else:
        shown = "{:.0%}".format(frac_at_50) if frac_at_50 is not None else "never (within this sweep)"
        print("  LINEAR OR WORSE: 50% recovery is not reached until knowledge_frac={}.".format(shown))
        print("  The practical case is weaker than Section 4.10 implies -- a mapper needs")
        print("  substantial, fairly reliable foreknowledge before this pays off, closer to")
        print("  full information than a cheap partial-profiling story would need.")
    if frac_at_90 is not None:
        print("  90% of the gap is recovered by knowledge_frac={:.0%}.".format(frac_at_90))
