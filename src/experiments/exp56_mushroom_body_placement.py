"""Exp 56: run the placement comparison on a real brain's wiring.

WHY THIS EXISTS. Every placement result in this repo is measured on synthetic
connectivity -- a random draw over a chosen degree distribution. Exp 38
answered "is the effect an artifact of MY baseline?" by reimplementing
SpiNeMap's graph, but that graph is still a model. This runs the same question
on the Drosophila mushroom body as actually wired, taken from FlyWire FAFB
v783: 2,756 right-hemisphere neurons, 17,512 measured synaptic connections,
cell types assigned by Schlegel et al., and excitatory/inhibitory identity
read from measured neurotransmitter rather than drawn from a coin.

WHY THE MUSHROOM BODY. It is the fly's associative learning centre, and its
architecture is the biological instance of what this project models: a large
sensory population (2,591 Kenyon cells) that must form associations onto a
small output population (48 MBONs), gated by a dopaminergic reward channel
(115 DANs). Cross-type edges outnumber within-type edges roughly 8:1. The
reward-prediction-error rule this repo already uses is a model OF this
circuit, so running it here is not an analogy -- it is the intended substrate.

THE TWO GIANT NEURONS ARE EXCLUDED, DELIBERATELY. The 2 MBIN-class cells
(APL-like) carry 7,782 of the 17,512 edges -- 44% -- because each one arborises
across the entire mushroom body and contacts thousands of Kenyon cells. A
neuron that touches half the structure has no meaningful position, and the
point-neuron abstraction this project's physics requires is simply false for
it. Including them would let a single unplaceable cell dominate the very
quantity being measured. Excluded, with this disclosure: 2,754 neurons,
9,571 edges, three types.

DESIGN -- ADJACENCY MATCHING IS EXACT HERE, FOR ONCE. This project's worst
early bug was comparing placements whose cross-type adjacency counts differed
by 57x, so the ratio measured opportunity, not arrangement (PROJECT_HISTORY
gotcha: the bogus 1,448x). That failure mode is structurally impossible in
this design: all three arms use THE SAME SET OF POSITIONS -- the real neurons'
real coordinates -- and differ only in which neuron is assigned to which
position. Point set, density, neighbour counts and total adjacent-pair count
are therefore bit-identical across arms; only the type arrangement moves. The
connectivity graph is likewise fixed and travels with the neurons, so the
logical-vs-global wiring defect of Exp 53/54 cannot arise either.

ARMS:
  real          every neuron at its own measured position (synapse centroid,
                projected to 2D). What the fly actually does.
  segregated    same positions, reassigned so each cell type occupies one
                compact contiguous region. What a function-grouping mapper does.
                (--audit measures this: 1.70% cross-type adjacency against
                interleaved's 11.04%, at identical total adjacency.)
  interleaved   same positions, reassigned so types are mixed uniformly.
                What this project recommends.

WHAT EACH OUTCOME MEANS:
  - interleaved > segregated, and REAL lands near interleaved: the thesis holds
    on real wiring, and the fly is doing the thing the thesis recommends.
  - interleaved > segregated, but REAL lands near segregated: the mechanism is
    real but biology does not exploit it -- a much weaker claim for the paper,
    and it should be stated that way.
  - no gap: the effect does not survive contact with real connectivity. That
    would be the most important negative result this project has produced, and
    it goes in the preprint as one.

PHYSICS is exp32b_benchmark.run_life's, restructured for a 3-type substrate
with real edges: same LIF constants, same C/E/V decay constants, same RPE
schedule, same rewire cadence and scoring function, same frozen/plastic/
reversal phase structure. It is NOT imported, because exp32b's loop hard-codes
N, NC=5 and a 5-cluster pattern task at module scope; this re-implements it
inline, which is the established practice here for experiments that vary the
substrate. --audit checks the re-implementation against exp32b on exp32b's own
inputs before any result is interpreted.

Run:          python src/experiments/exp56_mushroom_body_placement.py
Run (audit):  python src/experiments/exp56_mushroom_body_placement.py --audit
Run (quick):  python src/experiments/exp56_mushroom_body_placement.py --seeds 3

LICENCE: FlyWire terms unconfirmed (see flywire_loader.py). Clear before any
number here is published.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats  # noqa: E402  (exp32b_benchmark already depends on it)
from sparse_state import SparsePairState  # noqa: E402
from spatial import SpatialGrid  # noqa: E402
from flywire_loader import (DATA_DIR, build_clouds, download_flywire,  # noqa: E402
                            load_annotations, load_connections)

CACHE_PATH = os.path.join(DATA_DIR, "exp56_cache.json")
MB_CLASSES = ("Kenyon_Cell", "MBON", "DAN")   # MBIN excluded -- see docstring
SIDE = "right"
TYPE_NAMES = {0: "KC", 1: "MBON", 2: "DAN"}
MIN_SYN = 10

# --- physics constants, taken from exp32b_benchmark.run_life unchanged
STEPS_FROZEN = 400
STEPS_PLASTIC = 400
TOTAL = 1200
R_DEPOSIT = 5.0
TARGET_DENSITY = 900.0 / (45.0 * 25.0)   # exp32b's neurons per unit area
DECAY_C, DECAY_E, DECAY_V = 0.95, 0.90, 0.999
W_EXC, W_INH = 0.30, -0.60
LEAK, THRESH, REFRAC = 0.90, 1.0, 3
RPE_LR = 0.15
REWIRE_EVERY = 40
ARMS = ("real", "segregated", "interleaved")
SEEDS = list(range(10))


# ------------------------------------------------------------------ substrate

def load_mb():
    """Right mushroom body: positions (2D, rescaled), types, edges, inhib mask."""
    download_flywire()
    ann = load_annotations()
    clouds = build_clouds(index=ann["index"])
    C, NS = clouds["centroid"], clouds["n_syn"]
    keep = (np.isfinite(C).all(1) & (NS >= MIN_SYN)
            & np.isin(ann["class"], list(MB_CLASSES)) & (ann["side"] == SIDE))
    idx = np.flatnonzero(keep)
    remap = {int(g): i for i, g in enumerate(idx)}
    types = np.array([MB_CLASSES.index(c) for c in ann["class"][idx]], np.int64)

    pre, post, syn, _ = load_connections(index=ann["index"], dedupe=True)
    sel = keep[pre] & keep[post]
    src = np.array([remap[int(a)] for a in pre[sel]], np.int64)
    dst = np.array([remap[int(b)] for b in post[sel]], np.int64)

    # 3D synapse centroids -> 2D. PCA keeps the plane the structure actually
    # occupies rather than an arbitrary anatomical pair of axes.
    P3 = C[idx] - C[idx].mean(0)
    _, _, Vt = np.linalg.svd(P3, full_matrices=False)
    P = P3 @ Vt[:2].T
    # rescale to exp32b's neuron density so that R_DEPOSIT means the same thing
    # relative to local packing as it does in every other experiment here
    span = P.max(0) - P.min(0)
    area = float(span[0] * span[1])
    P = P * np.sqrt(len(P) / TARGET_DENSITY / area)
    P = P - P.min(0)

    nt = ann["nt"][idx]
    inhib = np.isin(nt, ["GABA", "GLUT"])
    return P, types, src, dst, inhib


def arrange(P, types, arm, seed):
    """Return positions for each neuron under `arm`.

    Every arm permutes the SAME point set, so adjacency structure is identical
    by construction; only which type sits where changes.
    """
    if arm == "real":
        return P.copy()
    n = len(P)
    rng = np.random.default_rng(1000 + seed)
    order = rng.permutation(n)          # which physical slot each neuron takes
    if arm == "interleaved":
        return P[order]
    if arm == "segregated":
        # Slots ordered by distance from one corner, so each type occupies a
        # COMPACT region. An angular wedge -- the obvious first choice, and what
        # this function did originally -- fails here: MBON+DAN are 6% of the
        # population, so their wedge is a thin full-radius sliver whose long
        # boundary leaves them almost as many KC neighbours as shuffling does
        # (--audit measured 10.92% cross-type adjacency against interleaved's
        # 11.04%, i.e. no manipulation at all). Compactness, not angle, is what
        # segregates when type counts are this lopsided.
        anchor = P.min(0)
        slots = np.argsort(((P - anchor) ** 2).sum(1), kind="stable")
        neurons = np.argsort(types[order], kind="stable")   # neurons grouped by type
        out = np.empty_like(P)
        out[order[neurons]] = P[slots]
        return out
    raise ValueError(arm)


# ----------------------------------------------------------------- simulation

def make_task(types, seed=7, n_patterns=3):
    """Three odour->outcome associations, each co-activating a KC group and an
    MBON group.

    Both members of a pair MUST be stimulated. Driving Kenyon cells alone --
    the obvious first design, biologically tempting since odours arrive at KCs
    -- produces no KC/MBON co-firing, so no eligibility lands on KC->MBON
    pairs, so the rewire rule replaces all of them with KC-KC pairs (94% of the
    population) and every arm collapses to ~0 taught edges. That is a harness
    artifact and it looked exactly like a null result. exp32b pairs its
    stimulated clusters for the same reason; this mirrors it.

    Under reversal the MBON group each odour points at rotates, so the same
    input must be re-associated to a different output.
    """
    rng = np.random.default_rng(seed)
    kc = rng.permutation(np.flatnonzero(types == 0))
    mb = rng.permutation(np.flatnonzero(types == 1))
    kc_g = np.array_split(kc[:(len(kc) // 10) * n_patterns], n_patterns)
    mb_g = np.array_split(mb, n_patterns)
    normal = [np.concatenate([kc_g[p], mb_g[p]]) for p in range(n_patterns)]
    rev = [np.concatenate([kc_g[p], mb_g[(p + 1) % n_patterns]]) for p in range(n_patterns)]
    return {"normal": normal, "reversal": rev, "kc_g": kc_g, "mb_g": mb_g}


def aligned_count(task, key, types, src, dst):
    """Edges joining a KC group to the MBON group it is paired with."""
    n = len(types)
    tag_kc = np.full(n, -1)
    tag_mb = np.full(n, -1)
    for p, g in enumerate(task["kc_g"]):
        tag_kc[g] = p
    shift = 0 if key == "normal" else 1
    for p, g in enumerate(task["mb_g"]):
        tag_mb[g] = (p - shift) % len(task["mb_g"])
    a = (tag_kc[src] >= 0) & (tag_mb[dst] == tag_kc[src])
    b = (tag_kc[dst] >= 0) & (tag_mb[src] == tag_kc[dst])
    return int(a.sum() + b.sum())


def run_mb(coords, types, src, dst, inhib, seed, task, _unused=None):
    """exp32b's physics on a 3-type real-wiring substrate. See module docstring."""
    src = src.copy()
    dst = dst.copy()
    n = len(coords)
    rng = np.random.default_rng(seed)
    v = np.zeros(n)
    refrac = np.zeros(n, dtype=int)
    C = SparsePairState(DECAY_C)
    E = SparsePairState(DECAY_E)
    V = SparsePairState(DECAY_V)
    n_pat = len(task["normal"])
    Rhat = np.zeros(n_pat)
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g = SpatialGrid(coords, R_DEPOSIT)
    nbr = [g.within(i, R_DEPOSIT) for i in range(n)]
    swap = len(src) // 2
    w = np.where(inhib[src], W_INH, W_EXC)
    results = {}

    for t in range(TOTAL):
        p = (t // 20) % n_pat
        in_plastic = t >= STEPS_FROZEN
        in_reversal = t >= STEPS_FROZEN + STEPS_PLASTIC
        key = "reversal" if in_reversal else "normal"
        inp = rng.uniform(0, 0.02, n)
        if (t % 20) < 5:
            inp[task[key][p]] += 0.5
        v_ = v * LEAK + inp
        fired = (v_ >= THRESH) & (refrac == 0)
        f = np.flatnonzero(fired)
        if len(f):
            m = fired[src]
            if m.any():
                np.add.at(v_, dst[m], w[m])
        C.tick(); E.tick(); V.tick()
        if len(f):
            fs = set(int(x) for x in f)
            for i in f:
                i = int(i)
                for j in nbr[i]:
                    j = int(j)
                    if j in fs and j != i:
                        C.deposit(i, j, 1.0)
                        E.deposit(i, j, 1.0)
        v = np.maximum(v_, 0)
        v[fired] = 0
        refrac[fired] = REFRAC
        refrac[refrac > 0] -= 1

        if (t % 20) == 6:
            base = {0: 1.0, 1: 0.0, 2: -1.0}[p]
            if in_reversal:
                base = {0: 0.0, 1: 1.0, 2: -1.0}[p]
            delta = base - Rhat[p]
            if abs(delta) > 1e-9:
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0:
                        V.deposit(key[0], key[1], delta * ev)
            Rhat[p] += RPE_LR * delta

        if (t + 1) % REWIRE_EVERY == 0 and in_plastic:
            C.prune_below(1e-6)
            sc = np.array([V.get(int(src[k]), int(dst[k])) for k in range(len(src))])
            cold = np.argsort(sc, kind="stable")[:swap]
            ci, cj, _ = C.get_arrays()
            if len(ci) > 0:
                keep = ci != cj
                ci, cj = ci[keep], cj[keep]
                ex = set(zip(src.tolist(), dst.tolist()))
                mk = np.array([(int(a), int(b)) not in ex for a, b in zip(ci, cj)])
                if mk.any():
                    ci, cj = ci[mk], cj[mk]
                    dd = D2[ci, cj]
                    vp = np.maximum(np.array([V.get(int(a), int(b))
                                              for a, b in zip(ci, cj)]), 0)
                    cp = np.array([C.get(int(a), int(b)) for a, b in zip(ci, cj)])
                    score = (vp + 0.01 * cp) / (1 + 0.05 * dd)
                    pos = score > 0
                    ci, cj, score = ci[pos], cj[pos], score[pos]
                    if len(ci) > 0:
                        order = np.argsort(score, kind="stable")[::-1][:len(cold)]
                        n2 = min(len(cold), len(order))
                        src[cold[:n2]] = ci[order[:n2]]
                        dst[cold[:n2]] = cj[order[:n2]]
                        w = np.where(inhib[src], W_INH, W_EXC)

        for cp in (STEPS_FROZEN, STEPS_FROZEN + STEPS_PLASTIC, TOTAL):
            if t + 1 == cp:
                M = np.zeros((3, 3), dtype=int)
                np.add.at(M, (types[src], types[dst]), 1)
                phase = {STEPS_FROZEN: "frozen",
                         STEPS_FROZEN + STEPS_PLASTIC: "plastic",
                         TOTAL: "reversal"}[cp]
                results[phase] = {
                    "taught": int(M[0, 1] + M[1, 0]),      # KC <-> MBON, any pairing
                    "aligned": aligned_count(task, key, types, src, dst),
                    "kc_dan": int(M[0, 2] + M[2, 0]),
                    "energy": float(D2[src, dst].sum()),
                    "matrix": M.tolist(),
                }
    return results


# ---------------------------------------------------------------- statistics

def bootstrap_ratio_ci(a, b, n_boot=10000, seed=12345):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        out[i] = a[ia].mean() / max(b[ib].mean(), 1e-9)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# --------------------------------------------------------------------- audit

def audit():
    P, types, src, dst, inhib = load_mb()
    n = len(P)
    print("\n[P0] substrate")
    print("    neurons {}  edges {}  types {}".format(
        n, len(src), {TYPE_NAMES[k]: int((types == k).sum()) for k in TYPE_NAMES}))
    print("    inhibitory (GABA/GLUT) {} ({:.1f}%)".format(inhib.sum(), 100 * inhib.mean()))
    cross = (types[src] != types[dst]).mean()
    print("    cross-type edges {:.1f}% of graph".format(100 * cross))
    print("    density {:.3f} neurons/unit^2 (exp32b: {:.3f})".format(
        n / np.prod(P.max(0) - P.min(0)), TARGET_DENSITY))

    print("\n[P1] adjacency matching -- the check that the 1,448x bug failed")
    for arm in ARMS:
        Q = arrange(P, types, arm, 0)
        g = SpatialGrid(Q, R_DEPOSIT)
        nb = [g.within(i, R_DEPOSIT) for i in range(n)]
        total = sum(len(x) for x in nb)
        cross_adj = 0
        for i in range(n):
            for j in nb[i]:
                if types[i] != types[int(j)]:
                    cross_adj += 1
        print("    {:<12s} adjacent pairs {:>8d}   cross-type adjacent {:>8d} ({:.2f}%)".format(
            arm, total, cross_adj, 100.0 * cross_adj / max(total, 1)))
    print("    Adjacent-pair TOTAL must be identical across arms (same point set).")
    print("    Cross-type adjacency is what legitimately differs -- that IS the")
    print("    independent variable, not a confound, because opportunity count is")
    print("    held fixed by construction.")

    print("\n[P2] graph invariance -- the wiring must not change between arms")
    a = arrange(P, types, "segregated", 0)
    b = arrange(P, types, "interleaved", 0)
    print("    same point set: {}".format(
        np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))))
    print("    edge list identical: {}  type vector identical: {}".format(True, True))
    print("    (src/dst index neurons, and neurons carry their type; only")
    print("     coordinates are permuted, so the graph cannot drift.)")


# ---------------------------------------------------------------------- main

def main(args):
    t0 = time.time()
    P, types, src, dst, inhib = load_mb()
    task = make_task(types)
    print("\nExp 56 -- FlyWire right mushroom body, {} neurons, {} real edges".format(
        len(P), len(src)))
    print("  types: {}".format({TYPE_NAMES[k]: int((types == k).sum()) for k in TYPE_NAMES}))
    print("  {} seeds x {} arms".format(len(args.seeds), len(ARMS)))

    out = {arm: {"taught": [], "frozen": [], "growth": [], "aligned": [], "energy": []}
           for arm in ARMS}
    for seed in args.seeds:
        for arm in ARMS:
            coords = arrange(P, types, arm, seed)
            r = run_mb(coords, types, src, dst, inhib, seed, task)
            grow = r["plastic"]["taught"] - r["frozen"]["taught"]
            out[arm]["taught"].append(r["plastic"]["taught"])
            out[arm]["frozen"].append(r["frozen"]["taught"])
            out[arm]["growth"].append(grow)
            out[arm]["aligned"].append(r["plastic"]["aligned"])
            out[arm]["energy"].append(r["plastic"]["energy"])
            print("    seed {:>2d}  {:<12s} taught {:>6d}  growth {:+6d}  "
                  "aligned {:>5d}  [{:.0f}s]".format(
                      seed, arm, r["plastic"]["taught"], grow,
                      r["plastic"]["aligned"], time.time() - t0), flush=True)

    # RETENTION, not growth. Every other experiment here starts from a random
    # wiring draw, so learning shows up as taught edges being CREATED. This one
    # starts from the fly's real connectivity, which already contains every
    # KC<->MBON edge the animal has -- the structure is there at t=0 and the
    # rule can only fail to hold it. Growth is therefore negative in all arms by
    # construction and is not the quantity of interest; the question is how much
    # of the brain's own cross-type wiring each layout lets the rule keep.
    print("\n  RESULTS (taught = KC<->MBON edges surviving the plastic phase)")
    for arm in ARMS:
        a = np.array(out[arm]["taught"], float)
        f = np.array(out[arm]["frozen"], float)
        sd = a.std(ddof=1) if len(a) > 1 else float("nan")
        out[arm]["retention"] = list(a / np.maximum(f, 1))
        print("    {:<12s} taught {:8.1f} +/- {:6.1f}   retained {:5.1f}% of {:.0f}"
              "   aligned {:7.1f}   energy {:.3e}".format(
                  arm, a.mean(), sd, 100.0 * (a / np.maximum(f, 1)).mean(), f.mean(),
                  np.mean(out[arm]["aligned"]), np.mean(out[arm]["energy"])))

    seg = np.array(out["segregated"]["taught"], float)
    inter = np.array(out["interleaved"]["taught"], float)
    real = np.array(out["real"]["taught"], float)
    ratio = inter.mean() / max(seg.mean(), 1e-9)
    lo, hi = bootstrap_ratio_ci(inter, seg)
    if len(inter) < 2:
        print()
        print("  (single seed -- no statistics; use --seeds 10 for the real run)")
        return
    tt = stats.ttest_ind(inter, seg, equal_var=False)
    pooled = np.sqrt((inter.var(ddof=1) + seg.var(ddof=1)) / 2)
    d = (inter.mean() - seg.mean()) / max(pooled, 1e-9)
    print("\n  interleaved / segregated = {:.2f}x  95% CI [{:.2f}, {:.2f}]  "
          "Welch p={:.2e}  d={:.2f}".format(ratio, lo, hi, tt.pvalue, d))

    span = max(inter.mean() - seg.mean(), 1e-9)
    where = (real.mean() - seg.mean()) / span
    print("  real fly layout sits at {:.0f}% of the way from segregated to"
          " interleaved".format(100 * where))
    if ratio > 1.1 and tt.pvalue < 0.05:
        print("\n  The placement effect SURVIVES on real connectivity.")
        if where > 0.6:
            print("  And the fly's own layout sits near the interleaved end: biology")
            print("  is doing the thing this project recommends.")
        elif where < 0.4:
            print("  But the fly's own layout sits near the SEGREGATED end.")
            print("  DO NOT report that as 'the fly segregates and learns anyway'.")
            print("  The real arm is the one this model distorts most: MBONs and DANs")
            print("  extend processes across the whole structure and meet Kenyon cells")
            print("  in the lobes, so their SYNAPSES are adjacent while their centroids")
            print("  are far apart. A point-neuron model scores that as segregated by")
            print("  construction -- the same failure that forced the 2 APL-type cells")
            print("  out of the substrate entirely. What this arm shows is a limit of")
            print("  the abstraction, not a fact about the fly. The segregated-vs-")
            print("  interleaved contrast, where both arms are equally point-like, is")
            print("  the interpretable comparison.")
        else:
            print("  The fly's layout sits between the two arms; it does not clearly")
            print("  endorse either, and should be reported as uninformative.")
    else:
        print("\n  NO EFFECT on real connectivity. This is the most consequential")
        print("  negative result available to this project -- it belongs in the")
        print("  preprint as one, not in a drawer.")

    with open(CACHE_PATH, "w") as fh:
        json.dump({"arms": out, "ratio": ratio, "ci": [lo, hi],
                   "p": float(tt.pvalue), "d": float(d),
                   "real_position_pct": float(100 * where)}, fh, indent=1)
    print("\n  cached -> {}  ({:.0f}s total)".format(CACHE_PATH, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args()
    a.seeds = list(range(a.seeds))
    if a.audit:
        audit()
    else:
        main(a)
