"""Fast tinkering: dynamic self-organizing placement. Neurons start in a
single mixed blob and PHYSICALLY migrate over the course of learning,
pulled toward the centroid of the partners they've built real value (V)
with -- the same signal that already drives synaptic rewiring, now also
driving position. A soft tether keeps them from collapsing to a point.
This capability does not exist anywhere else in the codebase (confirmed:
an old checkpoint comment says "migration is a later port" and nothing
implements it) -- this is a first prototype, not a formal experiment.

Run: python src/experiments/tinker_dynamic_placement.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from sparse_state import SparsePairState
from spatial import SpatialGrid
from exp32b_benchmark import (
    PATTERNS, NC, N, STEPS_FROZEN, STEPS_PLASTIC, TOTAL, make_placement, run_life,
)

ETA = 0.35          # migration step size (fraction of the way toward the pull target each reposition)
TETHER = 0.03        # spring-back-to-origin strength (prevents collapse / unbounded drift)
REPOSITION_EVERY = 40  # same cadence as synaptic rewiring


def make_blob_start(seed=42):
    """Single mixed blob, radius scaled like Exp 43's blobs (density-matched
    to a population of 900), all 5 types uniformly shuffled."""
    rng = np.random.default_rng(seed)
    R = 1.5 * np.sqrt(N / 15.0)  # exp43's blob_radius formula at n=900
    th = rng.uniform(0, 2 * np.pi, N)
    r = R * np.sqrt(rng.uniform(0, 1, N))
    coords = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    cids = np.repeat(np.arange(NC), N // NC)
    rng.shuffle(cids)
    return coords, cids


def run_life_dynamic(coords0, cids, seed, migrate=True):
    """Copied from exp32b_benchmark.run_life; adds a migration step at the
    same cadence as rewiring. Everything else is identical physics."""
    coords = coords0.copy()
    orig = coords0.copy()
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    n_edges = N * 10
    src = rng2.integers(0, N, n_edges); dst = rng2.integers(0, N, n_edges)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(3)

    def rebuild():
        D2_ = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        g_ = SpatialGrid(coords, 5.0)
        nbr_ = [g_.within(i, 5.0) for i in range(N)]
        return D2_, nbr_

    D2, nbr = rebuild()
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N // 2
    results = {}
    migration_log = []
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
        if (t + 1) % REPOSITION_EVERY == 0 and in_plastic:
            # --- MIGRATION: physically move neurons toward learned V-partners ---
            if migrate:
                vi, vj, vv = V.get_arrays()
                if len(vi):
                    vv = np.maximum(vv, 0.0)
                    force = np.zeros((N, 2)); weight = np.zeros(N)
                    np.add.at(force, vi, vv[:, None] * (coords[vj] - coords[vi]))
                    np.add.at(weight, vi, vv)
                    np.add.at(force, vj, vv[:, None] * (coords[vi] - coords[vj]))
                    np.add.at(weight, vj, vv)
                    has_pull = weight > 1e-9
                    pull = np.zeros((N, 2))
                    pull[has_pull] = force[has_pull] / weight[has_pull, None]
                    coords[has_pull] += ETA * pull[has_pull]
                    coords += TETHER * (orig - coords)
                    migration_log.append(float(np.abs(coords - orig).mean()))
                    D2, nbr = rebuild()
            # --- SYNAPTIC REWIRING (unchanged from canonical run_life) ---
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
                taught = M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]
                phase = {STEPS_FROZEN: "frozen", STEPS_FROZEN + STEPS_PLASTIC: "plastic", TOTAL: "reversal"}[cp]
                results[phase] = {"taught": taught}
    return results, coords, migration_log


if __name__ == "__main__":
    print("=" * 100)
    print("TINKER: dynamic self-organizing placement (blob start, migrate toward learned V-partners)")
    print("=" * 100)

    SEEDS = [0, 1, 2]
    coords0, cids = make_blob_start()

    print("\n[1] Dynamic (migrating) blob:")
    growth_dyn = []
    final_coords = None
    for s in SEEDS:
        res, final_coords, mig_log = run_life_dynamic(coords0, cids, s, migrate=True)
        g = res["plastic"]["taught"] - res["frozen"]["taught"]
        growth_dyn.append(g)
        print("  seed {}: growth={:+.1f}  mean drift over life={}".format(
            s, g, ["{:.2f}".format(x) for x in mig_log[:3]] + ["..."] if mig_log else "n/a"))
    growth_dyn = np.array(growth_dyn, float)
    print("  DYNAMIC BLOB  growth={:+.1f} +/- {:.1f}".format(growth_dyn.mean(), growth_dyn.std()))

    print("\n[2] Static (non-migrating) blob, same starting point:")
    growth_static = []
    for s in SEEDS:
        res, _, _ = run_life_dynamic(coords0, cids, s, migrate=False)
        g = res["plastic"]["taught"] - res["frozen"]["taught"]
        growth_static.append(g)
    growth_static = np.array(growth_static, float)
    print("  STATIC BLOB   growth={:+.1f} +/- {:.1f}".format(growth_static.mean(), growth_static.std()))

    print("\n[3] Reference: static interleaved (topoforge) and static segregated (vlsi)")
    for name in ("topoforge", "vlsi"):
        c, ci = make_placement(name)
        gg = []
        for s in SEEDS:
            r = run_life(c, ci, s)
            gg.append(r["plastic"]["taught"] - r["frozen"]["taught"])
        gg = np.array(gg, float)
        print("  {:<12}  growth={:+.1f} +/- {:.1f}".format(name, gg.mean(), gg.std()))

    print("\n" + "=" * 100)
    print("Did letting the blob physically reorganize itself help over staying put?")
    print("=" * 100)
    print("  dynamic - static = {:+.1f}".format(growth_dyn.mean() - growth_static.mean()))
