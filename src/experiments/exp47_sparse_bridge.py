"""Exp 47: the real-cortex counterexample -- does a SMALL bridge budget
recover most of interleaved's learning, without full interleaving?

MOTIVATION. Real cortex is not interleaved at the neuron level. It is
aggressively segregated -- cortical columns, functional areas, ocular
dominance stripes -- and cross-domain association happens through SPARSE
long-range white-matter tracts, not local mixing. If real brains that
demonstrably build cross-domain associations are locally segregated, that
is a standing counterexample to "segregation blocks cross-type learning"
as stated, and it points at a cheaper design than "interleave everything":
segregate locally, but embed a small minority of the partner population
directly in each other's territory (the neuromorphic analogue of a sparse
long-range projection terminating in local circuitry that can actually use
it -- not a single wire between two neurons, which this project's local
plasticity rule cannot exploit any better than segregation itself, but a
handful of real neurons physically co-located so LOCAL plasticity can
build genuine structure around them).

DESIGN. Start from the segregated ("vlsi") placement. For each of the two
associated type pairs (0,3) and (1,4), relocate a small fraction
`bridge_frac` of one type's neurons (e.g. type 3) from their home block
into a randomly chosen core of the partner type's territory (type 0),
drawing their new coordinates from that core's own neuron-scatter
distribution -- i.e. real neurons, physically embedded, not a synthetic
edge. The remaining (1 - bridge_frac) stays in the original segregated
block. bridge_frac=0 reproduces vlsi exactly (verified). This is swept
from 0 to a substantial fraction to find where, if anywhere, learning
recovers.

PREDICTIONS, registered before running:
  (P1) Growth increases monotonically with bridge_frac (more embedded
       partners can only help or do nothing, never hurt).
  (P2) THE REAL QUESTION: does a SMALL bridge_frac (<=10%) recover MOST
       (say >=50%) of the gap between segregated (-727) and interleaved
       (+1462)? If yes, the practical recommendation "interleave" is
       overstated -- a cheap sparse-bridge design gets most of the
       benefit. If recovering most of the gap requires a LARGE
       bridge_frac (approaching full interleaving), the sparse-cortex
       counterexample does not transfer to this substrate, and full
       mixing (or an association-aware mapper, per Exp 38/44) remains
       the honest recommendation.
  No specific bridge_frac is predicted in advance to be the "recovery
  point" -- that is exploratory within this run. What is registered is
  the decision rule: <=10% budget recovering >=50% of the gap counts as
  support for the sparse-bridge story; otherwise it does not.

Run: python src/experiments/exp47_sparse_bridge.py
"""
import numpy as np
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats

from exp32b_benchmark import (
    NC, N, N_CORES, run_life, make_placement, core_positions, neurons_in_core,
)

SEEDS = list(range(8))
BRIDGE_FRACS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
BRIDGE_PAIRS = [(3, 0), (4, 1)]  # (source type to relocate, destination territory)


def make_placement_bridge(bridge_frac, seed=42):
    core_pos = core_positions()
    cores_per_id = N_CORES // NC
    core_cid = []
    for cid in range(NC):
        for _ in range(cores_per_id):
            core_cid.append(cid)
    while len(core_cid) < N_CORES:
        core_cid.append(NC - 1)

    coords, cids = [], []
    for core_id in range(N_CORES):
        pts = neurons_in_core(core_id, core_pos[core_id])
        for p in pts:
            coords.append(p)
            cids.append(core_cid[core_id])
    coords = np.array(coords, dtype=float)
    cids = np.array(cids, dtype=int)

    if bridge_frac <= 0:
        return coords, cids

    rng = np.random.default_rng(seed)

    def cores_of_type(t):
        return [cid for cid in range(N_CORES) if core_cid[cid] == t]

    for src_type, dst_type in BRIDGE_PAIRS:
        idx_src = np.where(cids == src_type)[0]
        n_bridge = int(round(bridge_frac * len(idx_src)))
        if n_bridge == 0:
            continue
        bridge_idx = rng.choice(idx_src, size=n_bridge, replace=False)
        dst_cores = cores_of_type(dst_type)
        for i in bridge_idx:
            target_core = dst_cores[rng.integers(0, len(dst_cores))]
            th = rng.uniform(0, 2 * np.pi)
            r = 1.5 * np.sqrt(rng.uniform(0, 1))
            cx, cy = core_pos[target_core]
            coords[i] = [cx + r * np.cos(th), cy + r * np.sin(th)]
    return coords, cids


def growth_of(coords, cids, seeds=SEEDS):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


if __name__ == "__main__":
    print("=" * 78)
    print("EXP 47: sparse long-range bridge -- how much budget recovers how much learning?")
    print("=" * 78)

    print("\n[0] Sanity: bridge_frac=0 reproduces vlsi exactly")
    c0, i0 = make_placement_bridge(0.0)
    cv, iv = make_placement("vlsi")
    exact = np.array_equal(c0, cv) and np.array_equal(i0, iv)
    print("  exact match: {}".format(exact))
    if not exact:
        print("  MISMATCH -- refusing to trust the sweep.")
        sys.exit(1)

    print("\n[1] Growth vs bridge fraction, {} seeds each".format(len(SEEDS)))
    results = {}
    for bf in BRIDGE_FRACS:
        coords, cids = make_placement_bridge(bf)
        g = growth_of(coords, cids)
        results[bf] = g
        print("  bridge_frac={:>5.0%}  growth={:+8.1f} +/- {:<6.1f}".format(bf, g.mean(), g.std()))

    coords_t, cids_t = make_placement("topoforge")
    g_inter = growth_of(coords_t, cids_t)
    print("  {:>16}  growth={:+8.1f} +/- {:<6.1f}  (full interleave, reference)".format(
        "interleaved", g_inter.mean(), g_inter.std()))

    print("\n" + "=" * 78)
    print("[2] Recovery: what fraction of the segregated->interleaved GAP does each budget close?")
    print("=" * 78)
    seg_g = results[0.0].mean()
    inter_g = g_inter.mean()
    span = inter_g - seg_g
    print("  gap: segregated {:+.1f} -> interleaved {:+.1f}  (span {:.1f})".format(seg_g, inter_g, span))
    print("  {:>12} {:>10} {:>12} {:>10}".format("bridge_frac", "growth", "%recovered", "Welch p vs seg"))
    recovery_at_10pct = None
    for bf in BRIDGE_FRACS:
        g = results[bf]
        pct = (g.mean() - seg_g) / span
        t, p = stats.ttest_ind(g, results[0.0], equal_var=False)
        print("  {:>11.0%} {:>+10.1f} {:>11.1%} {:>14.2e}".format(bf, g.mean(), pct, p))
        if bf == 0.10:
            recovery_at_10pct = pct

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if recovery_at_10pct is not None and recovery_at_10pct >= 0.5:
        print("  SUPPORTS the sparse-bridge story: a 10% embedded-neuron budget recovers")
        print("  {:.0%} of the segregated-to-interleaved gap. The practical".format(recovery_at_10pct))
        print("  recommendation 'interleave everything' is overstated -- a cheap sparse")
        print("  bridge design captures most of the benefit, closer to how real cortex")
        print("  is actually organized (locally segregated + sparse long-range links).")
    else:
        pct_str = "{:.0%}".format(recovery_at_10pct) if recovery_at_10pct is not None else "n/a"
        print("  Does NOT support the sparse-bridge story at a 10% budget ({} of the".format(pct_str))
        print("  gap recovered, below the 50% threshold). The sparse-cortex counterexample")
        print("  does not transfer cheaply to this substrate -- full interleaving (or an")
        print("  association-aware mapper, Exp 38/44) remains the defensible recommendation.")
