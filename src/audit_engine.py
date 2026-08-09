"""Chunk 4: The regression gate — original audit's five claims re-run
on the sparse engine (legacy mode). Same pre-registered criteria.
Run: python src/audit_engine.py    (long run — hours at current speeds;
prints progress per seed so you can watch it breathe)
"""
import numpy as np
import time
from engine import Life, make_salt_clustered
from spatial import SpatialGrid


def report(name, vals, criterion, crit_fn):
    vals = np.array(vals, dtype=float)
    m, s = vals.mean(), vals.std()
    ok = crit_fn(vals)
    print(f"\nCLAIM: {name}")
    print(f"  seeds: {np.round(vals, 2)}")
    print(f"  mean {m:,.2f}  std {s:,.2f}")
    print(f"  criterion: {criterion}")
    print(f"  VERDICT: {'GRANITE' if ok else 'SOFT'}")
    return ok


def make_segregated(N=1000, NC=5):
    coords, _ = make_salt_clustered(N, NC)
    cids = np.repeat(np.arange(NC), N // NC)
    return coords, cids


if __name__ == "__main__":
    print("=" * 66)
    print("CHUNK 4: REGRESSION GATE — five claims on the sparse engine")
    print("=" * 66)
    t_start = time.time()
    verdicts = {}
    SEEDS = [0, 1, 2, 3, 4]

    # ---- CLAIM 1: placement energy (engine-independent; grid-certified) ----
    # Identical to original audit — kNN energy ratio via SpatialGrid.
    ratios = []
    for gs in [42, 43, 44, 45, 46]:
        rngA = np.random.default_rng(gs)
        grid_pts = np.array([[(i % 32) * 1.25, (i // 32) * 1.25]
                             for i in range(1000)])
        centers = rngA.uniform(5, 35, size=(5, 2))
        pts = []
        for cx, cy in centers:
            for _ in range(200):
                th, r = rngA.uniform(0, 2 * np.pi), rngA.uniform(0, 1.0)
                pts.append([cx + r * np.cos(th), cy + r * np.sin(th)])
        tight = np.array(pts)

        def knn_e(coords):
            g = SpatialGrid(coords, 2.0)
            nb = g.all_k_nearest(10)
            return sum(((coords[nb[i]] - coords[i]) ** 2).sum()
                       for i in range(len(coords)))
        ratios.append(knn_e(grid_pts) / knn_e(tight))
    verdicts['placement'] = report(
        "Placement energy ratio (grid-backed kNN)",
        ratios, "mean > 20x", lambda v: v.mean() > 20)

    # ---- CLAIM 2: interleaving advantage ----
    adv = []
    for s in SEEDS:
        t0 = time.time()
        ci, ii = make_salt_clustered()
        cs, is_ = make_segregated()
        li = Life(ci, ii, rule='corr', seed_life=s); li.run(400)
        ls = Life(cs, is_, rule='corr', seed_life=s); ls.run(400)
        adv.append(li.taught_mass() / max(ls.taught_mass(), 1))
        print(f"  [interleave seed {s}: {adv[-1]:.2f}x  "
              f"({time.time()-t0:.0f}s)]")
    verdicts['interleave'] = report(
        "Interleaved beats segregated (taught mass)",
        adv, "mean > 1.25x and every seed > 1.0x",
        lambda v: v.mean() > 1.25 and (v > 1.0).all())

    # ---- CLAIM 3: utility selectivity vs correlation null ----
    rw = lambda t, p: {0: +1.0, 1: 0.0, 2: -1.0}[p]
    selU, selC = [], []
    for s in SEEDS:
        t0 = time.time()
        co, ci_ = make_salt_clustered()
        lu = Life(co, ci_, rule='util', rewards=rw, seed_life=s); lu.run(600)
        lc = Life(co, ci_, rule='corr', rewards=rw, seed_life=s); lc.run(600)
        Mu, Mc = lu.bridge_matrix(), lc.bridge_matrix()
        selU.append((Mu[0, 3] + Mu[3, 0]) / max(Mu[1, 4] + Mu[4, 1], 1))
        selC.append((Mc[0, 3] + Mc[3, 0]) / max(Mc[1, 4] + Mc[4, 1], 1))
        print(f"  [utility seed {s}: {selU[-1]:.1f}x vs ctrl {selC[-1]:.2f}x  "
              f"({time.time()-t0:.0f}s)]")
    print(f"  [control selectivities: {np.round(selC, 2)}]")
    verdicts['utility'] = report(
        "Utility selectivity >> correlation null",
        selU, "mean > 5x AND control mean < 1.5x",
        lambda v: v.mean() > 5 and np.mean(selC) < 1.5)

    # ---- CLAIMS 4 & 5: sunk cost and RPE, shared reversal schedule ----
    rw_flip = lambda t, p: ({0: +1.0, 1: 0.0, 2: -1.0} if t < 600
                            else {0: 0.0, 1: +1.0, 2: -1.0})[p]

    def crossed(life_obj):
        M = life_obj.bridge_matrix()
        return (M[1, 4] + M[4, 1]) > (M[0, 3] + M[3, 0])

    persist, fixed = [], []
    for s in SEEDS:
        t0 = time.time()
        co, ci_ = make_salt_clustered()
        lb = Life(co, ci_, rule='util', rewards=rw_flip, seed_life=s)
        lb.run(1200)
        persist.append(0.0 if crossed(lb) else 1.0)
        lr = Life(co, ci_, rule='rpe', rewards=rw_flip, seed_life=s)
        lr.run(1200)
        fixed.append(1.0 if crossed(lr) else 0.0)
        print(f"  [reversal seed {s}: util-crossed={not persist[-1]}, "
              f"rpe-crossed={bool(fixed[-1])}  ({time.time()-t0:.0f}s)]")
    verdicts['sunkcost'] = report(
        "Sunk-cost: no crossover under raw utility",
        persist, ">= 4/5 seeds persist", lambda v: v.sum() >= 4)
    verdicts['rpe'] = report(
        "RPE produces regime change in window",
        fixed, ">= 4/5 seeds cross", lambda v: v.sum() >= 4)

    print("\n" + "=" * 66)
    print(f"GATE COMPLETE in {(time.time()-t_start)/60:.0f} min")
    for k, ok in verdicts.items():
        print(f"  {k:>10}: {'GRANITE' if ok else 'SOFT'}")
    n = sum(verdicts.values())
    print(f"\n{n}/5 claims survived on the sparse engine.")
    print("5/5 = the rewrite is physics-preserving; Chunk 5 (local tracking)")
    print("proceeds against this baseline.")