"""Fast tinkering, v3: adds simulated-annealing-style exploration noise to
the wide-sensing migration mechanism (v2), to test whether it can break
the "whole block drifts together as a rigid clump" failure mode diagnosed
in v2 -- pure gradient-following never takes a step that looks locally
bad, even when a genuinely better configuration is one temporary bad step
away. Noise is added to every neuron's migration step, largest on the
first reposition event and linearly cooling to ~0 by the last one within
the plastic phase (10 reposition events total).

Run: python src/experiments/tinker_dynamic_placement3.py
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

ETA = 0.5
TETHER = 0.02
REPOSITION_EVERY = 40
SENSE_RADIUS = 25.0
N_REPOSITIONS = STEPS_PLASTIC // REPOSITION_EVERY  # 10
T0 = 10.0  # initial exploration-noise std, linearly cooled to 0 by the last reposition


def run_life_annealed(coords0, cids, seed, migrate=True, t0=T0, sense_radius=SENSE_RADIUS):
    coords = coords0.copy()
    orig = coords0.copy()
    rng3 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    rng_anneal = np.random.default_rng(seed + 10000)
    n_edges = N * 10
    src = rng2.integers(0, N, n_edges); dst = rng2.integers(0, N, n_edges)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Sense = SparsePairState(0.98)
    Rhat = np.zeros(3)
    reposition_count = 0

    def rebuild():
        D2_ = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        g_ = SpatialGrid(coords, 5.0)
        nbr_ = [g_.within(i, 5.0) for i in range(N)]
        g_wide = SpatialGrid(coords, sense_radius)
        nbr_wide_ = [g_wide.within(i, sense_radius) for i in range(N)] if migrate else None
        return D2_, nbr_, nbr_wide_

    D2, nbr, nbr_wide = rebuild()
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
        if migrate: Sense.tick()
        if len(f):
            fs = set(int(x) for x in f)
            for i in f:
                i = int(i)
                for j in nbr[i]:
                    if int(j) in fs: C.deposit(i, int(j), 1.0); E.deposit(i, int(j), 1.0)
                if migrate:
                    for j in nbr_wide[i]:
                        if int(j) in fs: Sense.deposit(i, int(j), 1.0)
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
            if migrate:
                si, sj, sv = Sense.get_arrays()
                force = np.zeros((N, 2)); weight = np.zeros(N)
                if len(si):
                    sv = np.maximum(sv, 0.0)
                    np.add.at(force, si, sv[:, None] * (coords[sj] - coords[si]))
                    np.add.at(weight, si, sv)
                    np.add.at(force, sj, sv[:, None] * (coords[si] - coords[sj]))
                    np.add.at(weight, sj, sv)
                has_pull = weight > 1e-9
                pull = np.zeros((N, 2))
                pull[has_pull] = force[has_pull] / weight[has_pull, None]
                # annealing: temperature cools linearly across the 10 reposition events
                temperature = max(0.0, t0 * (1.0 - reposition_count / N_REPOSITIONS))
                noise = rng_anneal.normal(0, temperature, size=(N, 2))
                coords[has_pull] += ETA * pull[has_pull]
                coords += noise
                coords += TETHER * (orig - coords)
                migration_log.append((float(np.abs(coords - orig).mean()), temperature))
                reposition_count += 1
                D2, nbr, nbr_wide = rebuild()
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
    print("TINKER v3: ANNEALED wide-sensing migration (T0={:.1f}, cooling over {} reposition events)".format(
        T0, N_REPOSITIONS))
    print("starting from SEGREGATED placement")
    print("=" * 100)
    coords0, cids = make_placement("vlsi")
    growth = []
    for s in [0, 1, 2]:
        res, final_coords, mig_log = run_life_annealed(coords0, cids, s, migrate=True)
        g = res["plastic"]["taught"] - res["frozen"]["taught"]
        growth.append(g)
        print("  seed {}: growth={:+.1f}  drift/temp trace={}".format(
            s, g, ["{:.1f}/{:.1f}".format(d, tp) for d, tp in mig_log]))
    growth = np.array(growth, float)
    print("\n  ANNEALED (wide sense) FROM SEGREGATED  growth={:+.1f} +/- {:.1f}".format(growth.mean(), growth.std()))
    print("  Reference: static segregated=-726, static interleaved=+1471,")
    print("             narrow-sense migration=-719, wide-sense (no annealing)=-649")
