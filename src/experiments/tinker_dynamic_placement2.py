"""Fast tinkering, v2: fixes the chicken-and-egg problem found in
tinker_dynamic_placement.py -- migration driven by V (learned value) can't
rescue segregated placement because V only ever accrues for pairs that
have already co-fired LOCALLY (radius 5, the same radius that makes
segregated placement bad in the first place), so there is no signal
telling a neuron which direction its true partners are in.

Fix: a second, WIDE-radius correlation tracker ("Sense") used ONLY to
compute the migration pull direction, completely separate from the C/E/V
state that drives actual synaptic learning. Models a neuron sensing weak
correlation over a wider range than it can actually build synapses over
-- e.g. diffuse activity correlation vs. actual axonal reach. If this
lets segregated placement migrate its way toward interleaved-level
performance, self-organizing placement is a real, useful idea for this
substrate. If it still can't, even a widened sensing radius isn't enough
and the idea needs a different mechanism entirely.

Run: python src/experiments/tinker_dynamic_placement2.py
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


def run_life_dynamic_v2(coords0, cids, seed, migrate=True, sense_radius=SENSE_RADIUS):
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
    Sense = SparsePairState(0.98)
    Rhat = np.zeros(3)

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
                if len(si):
                    sv = np.maximum(sv, 0.0)
                    force = np.zeros((N, 2)); weight = np.zeros(N)
                    np.add.at(force, si, sv[:, None] * (coords[sj] - coords[si]))
                    np.add.at(weight, si, sv)
                    np.add.at(force, sj, sv[:, None] * (coords[si] - coords[sj]))
                    np.add.at(weight, sj, sv)
                    has_pull = weight > 1e-9
                    pull = np.zeros((N, 2))
                    pull[has_pull] = force[has_pull] / weight[has_pull, None]
                    coords[has_pull] += ETA * pull[has_pull]
                    coords += TETHER * (orig - coords)
                    migration_log.append(float(np.abs(coords - orig).mean()))
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
    print("TINKER v2: wide-sensing-radius migration ({:.0f} units) starting from SEGREGATED placement".format(SENSE_RADIUS))
    print("=" * 100)
    coords0, cids = make_placement("vlsi")
    growth = []
    for s in [0, 1, 2]:
        res, final_coords, mig_log = run_life_dynamic_v2(coords0, cids, s, migrate=True)
        g = res["plastic"]["taught"] - res["frozen"]["taught"]
        growth.append(g)
        print("  seed {}: growth={:+.1f}  final drift={:.2f}".format(s, g, mig_log[-1] if mig_log else 0))
    growth = np.array(growth, float)
    print("  DYNAMIC (wide sense) FROM SEGREGATED  growth={:+.1f} +/- {:.1f}".format(growth.mean(), growth.std()))
    print("\n  Reference: static segregated=-726, static interleaved=+1471, narrow-sense migration=-719")
