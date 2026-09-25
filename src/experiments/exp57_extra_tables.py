"""Exp 57: does the rule get better with more pair-state tables?

WHY THIS EXISTS. The structural plasticity rule keeps exactly three decaying
tables over neuron pairs -- C (co-firing, 0.95), E (eligibility, 0.90) and
V (value, 0.999) -- and that is the minimum that can express the mechanism:
without C there is no candidate pool, without E reward cannot reach anything
that happened before it, without V nothing accumulates across pruning passes.
Three is the floor. Nobody has checked whether it is also the ceiling.

This tests three additions, each chosen because it addresses a specific,
already-observed shortcoming rather than because it sounded plausible:

  ORDER (O)   A signed table recording WHO FIRED FIRST. The deployed rule is
              blind to temporal order -- C and E treat a pair that fires
              together identically regardless of which neuron led. That is
              conspicuous given Exp 35's unexplained timing-jitter inverted-U,
              which survived three refuted mechanism hypotheses and lives
              precisely in the dimension the rule cannot represent. This does
              not test Exp 35's phenomenon; it closes the representational gap
              so that a later experiment can.

  CONSOL (W)  A slow second value table that V bleeds into, and which decays
              far slower than V. Standard cascade answer to stability-vs-
              plasticity. Motivated by Exp 56, where a pruning pass destroyed
              structure that had already been earned.

  AGE (A)     A per-SLOT survival counter: how many rewire epochs this
              connection has lasted. Long-lived connections resist the cut.
              Metaplasticity, and the cheapest of the three.

PRE-REGISTERED, WRITTEN BEFORE THE FIRST RUN:
  PRIMARY   an addition HELPS if interleaved taught mass rises >=10% over
            base with non-overlapping bootstrap CIs. It HURTS on the same
            criterion downward. Anything between is NO EFFECT, and gets
            reported as no effect rather than mined for a story.
  SECONDARY does the interleaved/segregated ratio move? An addition that
            raises both arms equally changes the rule's power but says
            nothing about placement, and must not be reported as if it did.
  TERTIARY  reversal-phase taught mass. CONSOL is predicted to HURT here --
            protecting old structure is exactly wrong when the task reverses.
            Registering that prediction now so a bad reversal result counts
            as confirmation rather than a surprise to be explained away.

DESIGN. Substrate is the standard PLB benchmark (exp32b placements) under
CORRECTED wiring -- `make_wiring(..., 'logical')` with the seed resampled per
run, per Exp 53/54. Extensions are pure additions: with all three off, this
file's loop must be BIT-IDENTICAL to exp54.run_life_corrected, which --audit
asserts on the real inputs before any result here is interpreted. None of the
three extensions consumes random numbers, so the RNG call sequence is
untouched by construction.

A DIAGNOSTIC WORTH KEEPING, from the first --audit run. AGE looked inert: it
returned a taught mass identical to base on seed 0, which is what an extension
that never fires would do. It is not inert. Instrumenting the cold-set
selection shows AGE changes WHICH connections are cut in 19 of the 20 rewire
epochs, and changes them wholesale -- typically ~850 of the 900 selected slots
differ from base. The endpoint is unmoved anyway. The reason is structural: in
this rule, what gets CUT matters far less than what is available to GROW. The
freed slots refill from the same candidate ranking whichever incumbents were
evicted, so protecting incumbents reshuffles the graveyard without changing
the result. Treat "an extension produced the same number" as a question, not
an answer -- inert and busy-but-irrelevant look identical from the outside.

PARAMETERS, fixed before running and not tuned afterwards:
  ORDER   decay 0.95, window 3 steps, candidate score x (1 + 0.5*tanh(net))
  CONSOL  decay 0.99999, 10% of positive V transferred per rewire epoch
  AGE     retention bonus 0.25 * min(age,10)/10, scaled by the V spread so it
          is dimensionless

Run:          python src/experiments/exp57_extra_tables.py
Run (audit):  python src/experiments/exp57_extra_tables.py --audit
Run (quick):  python src/experiments/exp57_extra_tables.py --seeds 3
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

from scipy import stats  # noqa: E402
from sparse_state import SparsePairState  # noqa: E402
from spatial import SpatialGrid  # noqa: E402
from exp32b_benchmark import (N, NC, PATTERNS, STEPS_FROZEN,  # noqa: E402
                              STEPS_PLASTIC, TOTAL, make_placement)
from exp54_synthetic_wiring_audit import (LEGACY_WIRING_SEED,  # noqa: E402
                                          make_wiring, run_life_corrected,
                                          wiring_seed_for)

CACHE_PATH = os.path.join("shd_data", "exp57_cache.json")

ORDER_DECAY = 0.95
ORDER_WINDOW = 3
ORDER_BETA = 0.5
CONSOL_DECAY = 0.99999
CONSOL_RATE = 0.10
AGE_ETA = 0.25
AGE_CAP = 10

CONDITIONS = [
    ("base", dict()),
    ("+order", dict(order=True)),
    ("+consol", dict(consol=True)),
    ("+age", dict(age=True)),
    ("+all", dict(order=True, consol=True, age=True)),
]
PLACEMENTS = [("segregated", "vlsi"), ("interleaved", "topoforge")]


def run_life_tables(coords, cids, seed, wiring_seed=LEGACY_WIRING_SEED,
                    wiring_mode="logical", order=False, consol=False, age=False):
    """exp54.run_life_corrected with three optional extra pair tables.

    Every line not guarded by `order` / `consol` / `age` is verbatim from
    run_life_corrected. --audit asserts bit-identity when all three are off.
    """
    rng3 = np.random.default_rng(seed)
    src, dst, inhib = make_wiring(cids, wiring_seed, wiring_mode)
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    O = SparsePairState(ORDER_DECAY) if order else None
    W = SparsePairState(CONSOL_DECAY) if consol else None
    slot_age = np.zeros(len(src), dtype=np.int64) if age else None
    last_fired = np.full(N, -10 ** 6, dtype=np.int64) if order else None
    Rhat = np.zeros(3)
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g = SpatialGrid(coords, 5.0)
    nbr = [g.within(i, 5.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N // 2
    results = {}
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
        if O is not None: O.tick()
        if W is not None: W.tick()
        if len(f):
            fs = set(int(x) for x in f)
            for i in f:
                i = int(i)
                for j in nbr[i]:
                    if int(j) in fs: C.deposit(i, int(j), 1.0); E.deposit(i, int(j), 1.0)
        # ORDER: i led j if i fired in the last ORDER_WINDOW steps but not now.
        # Deposited on the ordered pair (i, j); SparsePairState keys are ordered
        # tuples, so (i,j) and (j,i) are distinct and direction is representable.
        if O is not None and len(f):
            fs2 = set(int(x) for x in f)
            for j in f:
                j = int(j)
                for i in nbr[j]:
                    i = int(i)
                    if i in fs2:
                        continue
                    lag = t - last_fired[i]
                    if 1 <= lag <= ORDER_WINDOW:
                        O.deposit(i, j, 1.0)
            for i in f:
                last_fired[int(i)] = t
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
        if (t + 1) % 40 == 0 and in_plastic:
            C.prune_below(1e-6)
            # CONSOL: bleed a slice of positive value into the slow table first,
            # so this epoch's retention decision already sees it.
            if W is not None:
                for key in list(V.store.keys()):
                    vv = V.get(*key)
                    if vv > 0:
                        W.deposit(key[0], key[1], CONSOL_RATE * vv)
            sc = np.array([V.get(int(src[k]), int(dst[k])) for k in range(len(src))])
            if W is not None:
                sc = sc + np.array([W.get(int(src[k]), int(dst[k])) for k in range(len(src))])
            # AGE: a connection that has survived resists the cut. Bonus is
            # scaled by the spread of sc so it stays dimensionless.
            if slot_age is not None:
                spread = float(sc.max() - sc.min())
                if spread > 0:
                    sc = sc + AGE_ETA * (np.minimum(slot_age, AGE_CAP) / AGE_CAP) * spread
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
                    score = (vp + 0.01 * cp) / (1 + 0.05 * dd)
                    # ORDER: favour candidates whose firing order is consistent
                    # with the direction the edge would take.
                    if O is not None:
                        net = np.array([O.get(int(a), int(b)) - O.get(int(b), int(a))
                                        for a, b in zip(ci, cj)])
                        score = score * (1.0 + ORDER_BETA * np.tanh(net))
                    pos = score > 0
                    ci, cj, score = ci[pos], cj[pos], score[pos]
                    if len(ci) > 0:
                        order_idx = np.argsort(score, kind="stable")[::-1][:len(cold)]
                        n2 = min(len(cold), len(order_idx))
                        src[cold[:n2]] = ci[order_idx[:n2]]; dst[cold[:n2]] = cj[order_idx[:n2]]
                        if slot_age is not None:
                            slot_age += 1
                            slot_age[cold[:n2]] = 0
                        out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
                    elif slot_age is not None:
                        slot_age += 1
            elif slot_age is not None:
                slot_age += 1
        for cp2 in [STEPS_FROZEN, STEPS_FROZEN + STEPS_PLASTIC, TOTAL]:
            if t + 1 == cp2:
                M = np.zeros((NC, NC), dtype=int)
                np.add.at(M, (cids[src], cids[dst]), 1)
                wire_e = D2[src, dst].sum()
                taught = M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]
                selectivity = (M[0, 3] + M[3, 0]) / max(M[1, 4] + M[4, 1], 1)
                old_03 = M[0, 3] + M[3, 0]
                new_14 = M[1, 4] + M[4, 1]
                phase = {STEPS_FROZEN: "frozen",
                         STEPS_FROZEN + STEPS_PLASTIC: "plastic",
                         TOTAL: "reversal"}[cp2]
                results[phase] = {"taught": int(taught), "energy": float(wire_e),
                                  "selectivity": float(selectivity),
                                  "old_03": int(old_03), "new_14": int(new_14)}
    return results


def boot_ci(a, n_boot=10000, seed=12345):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    m = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def audit():
    print("\n[P0] all extensions OFF must be bit-identical to exp54.run_life_corrected")
    ok = True
    for label, strategy in PLACEMENTS:
        for seed in (0, 1):
            coords, cids = make_placement(strategy)
            ws = wiring_seed_for(True, seed)
            a = run_life_corrected(coords, cids, seed, wiring_seed=ws, wiring_mode="logical")
            b = run_life_tables(coords, cids, seed, wiring_seed=ws, wiring_mode="logical")
            same = a == b
            ok = ok and same
            print("    {:<12s} seed {}  identical: {}  (taught {} vs {})".format(
                label, seed, same, a["plastic"]["taught"], b["plastic"]["taught"]))
    print("    {}".format("P0 PASSED -- extensions are pure additions" if ok else
                          "P0 FAILED -- do NOT interpret any result from this file"))
    if not ok:
        sys.exit(1)

    print("\n[P1] each table actually does something (seed 0, interleaved)")
    coords, cids = make_placement("topoforge")
    ws = wiring_seed_for(True, 0)
    base = run_life_tables(coords, cids, 0, wiring_seed=ws)["plastic"]["taught"]
    for name, kw in CONDITIONS[1:]:
        got = run_life_tables(coords, cids, 0, wiring_seed=ws, **kw)["plastic"]["taught"]
        flag = "MOVES" if got != base else "NO CHANGE -- extension may be inert"
        print("    {:<9s} taught {:>6d}  vs base {:>6d}   {}".format(name, got, base, flag))


def main(args):
    t0 = time.time()
    out = {}
    for name, kw in CONDITIONS:
        out[name] = {}
        for label, strategy in PLACEMENTS:
            coords, cids = make_placement(strategy)
            vals, rev = [], []
            for seed in args.seeds:
                ws = wiring_seed_for(True, seed)
                r = run_life_tables(coords, cids, seed, wiring_seed=ws,
                                    wiring_mode="logical", **kw)
                vals.append(r["plastic"]["taught"])
                rev.append(r["reversal"]["new_14"])
                print("    {:<9s} {:<12s} seed {:>2d}  taught {:>6d}  rev_new {:>6d}"
                      "  [{:.0f}s]".format(name, label, seed, vals[-1], rev[-1],
                                           time.time() - t0), flush=True)
            out[name][label] = {"taught": vals, "reversal_new": rev}

    print("\n  RESULTS -- taught mass after the plastic phase")
    base_int = np.array(out["base"]["interleaved"]["taught"], float)
    base_lo, base_hi = boot_ci(base_int)
    print("    {:<9s} {:>10s} {:>10s} {:>8s} {:>10s}".format(
        "cond", "segregated", "interleaved", "ratio", "rev_new"))
    verdicts = {}
    for name, _ in CONDITIONS:
        seg = np.array(out[name]["segregated"]["taught"], float)
        inter = np.array(out[name]["interleaved"]["taught"], float)
        rev = np.array(out[name]["interleaved"]["reversal_new"], float)
        ratio = inter.mean() / max(seg.mean(), 1e-9)
        print("    {:<9s} {:>10.1f} {:>10.1f} {:>7.2f}x {:>10.1f}".format(
            name, seg.mean(), inter.mean(), ratio, rev.mean()))
        if name == "base":
            continue
        lo, hi = boot_ci(inter)
        delta = (inter.mean() - base_int.mean()) / max(base_int.mean(), 1e-9)
        overlap = not (hi < base_lo or base_hi < lo)
        if abs(delta) >= 0.10 and not overlap:
            verdicts[name] = ("HELPS" if delta > 0 else "HURTS", delta, (lo, hi))
        else:
            verdicts[name] = ("NO EFFECT", delta, (lo, hi))

    print("\n  PRE-REGISTERED VERDICTS (>=10% and non-overlapping CIs)")
    for name, (verdict, delta, ci) in verdicts.items():
        print("    {:<9s} {:<10s} {:+6.1f}% vs base   95% CI [{:.0f}, {:.0f}]"
              " (base [{:.0f}, {:.0f}])".format(
                  name, verdict, 100 * delta, ci[0], ci[1], base_lo, base_hi))

    # TERTIARY. The first version of this check compared the two means and
    # printed CONFIRMED whenever consol's was lower. That is a sign test with no
    # threshold and no interval: on a metric whose base values span 135 to 279
    # across seeds it would "confirm" the registered prediction off pure noise,
    # and on the first run it did exactly that (-3.1%, with the per-seed
    # differences splitting 5 up and 5 down). Fixed to require what the primary
    # requires -- a paired comparison and complete separation -- and reporting
    # every condition, not just the one that had a prediction attached.
    base_rev = np.array(out["base"]["interleaved"]["reversal_new"], float)
    print("\n  TERTIARY -- reversal re-learning (new_14 after the task flips)")
    print("    registered prediction: CONSOL hurts here, because protecting old")
    print("    structure is exactly wrong when the task reverses.")
    for name, _ in CONDITIONS:
        if name == "base":
            continue
        r = np.array(out[name]["interleaved"]["reversal_new"], float)
        delta = (r.mean() - base_rev.mean()) / max(base_rev.mean(), 1e-9)
        paired = r - base_rev
        sep = (r.min() > base_rev.max()) or (r.max() < base_rev.min())
        note = ("SEPARATED across all seeds" if sep
                else "overlapping -- {} of {} seeds moved the same way".format(
                    int((np.sign(paired) == np.sign(paired.sum())).sum()), len(paired)))
        print("    {:<9s} {:>7.1f} vs base {:>7.1f}  ({:+5.1f}%)  {}".format(
            name, r.mean(), base_rev.mean(), 100 * delta, note))
    cons = np.array(out["+consol"]["interleaved"]["reversal_new"], float)
    cons_sep = (cons.max() < base_rev.min())
    print("    -> registered prediction {}".format(
        "CONFIRMED" if cons_sep else
        "NOT CONFIRMED: consol's reversal effect is within seed noise. Report as a miss."))

    with open(CACHE_PATH, "w") as fh:
        json.dump({"raw": out, "verdicts": {k: v[0] for k, v in verdicts.items()}}, fh, indent=1)
    print("\n  cached -> {}  ({:.0f}s)".format(CACHE_PATH, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    a.seeds = list(range(a.seeds))
    if a.audit:
        audit()
    else:
        main(a)
