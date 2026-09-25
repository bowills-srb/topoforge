"""Exp 58: why does a survival counter improve reversal re-learning by 67%?

WHAT PROMPTED THIS. Exp 57 added three optional pair tables to the rule and
found all three inert on the primary endpoint -- taught mass moved by under
0.2% and the placement ratio never budged. But one number moved hard and
nobody predicted it: with the AGE table on, reversal re-learning (new_14 after
the task flips) rose from 221.1 to 368.5, +66.7%, with every one of ten seeds
above every base seed.

THAT IS A LEAD, NOT A RESULT, AND THIS FILE TREATS IT AS ONE. It was
unpredicted, it sits in a metric Exp 57 ranked third, and it was noticed
precisely because it was the one number that moved -- the textbook forking
path. Exp 49 has already caught this project out in exactly this spot, where a
single-placement-seed artifact read as a refutation until it was re-run. Seed
consistency is not protection: the pinned-wiring numbers were perfectly
consistent too, and wrong for two years of work.

Worse, the obvious mechanism predicts the OPPOSITE sign. AGE adds a retention
bonus proportional to how many rewire epochs a slot has survived. Edges formed
after the reversal start at age 0, so they should be the FIRST cut, and
re-learning should get worse. It got much better. Something else is going on,
and the job here is to find out what rather than to fit a story to it.

THE PRIME SUSPECT IS NOT AGE AT ALL. The retention score is heavily
tie-degenerate: most edges never earn value, so their score sits at exactly
zero and `np.argsort(kind="stable")` resolves the tie by INDEX -- the same
degeneracy that made this project's results numpy-version-dependent until
`kind='stable'` was pinned at 21 sites. Base therefore cuts low-index slots
over and over. ANY per-slot perturbation of that score, age-derived or not,
breaks the degeneracy and rotates the cut. If that is the mechanism, "age" is
a misnomer and the finding is about tie-breaking.

ARMS -- four bonuses of IDENTICAL magnitude, differing only in what they know:
  none      no bonus. Must reproduce exp54.run_life_corrected bit-identically.
  graded    ETA * min(age, 10)/10 * spread. Exp 57's table, verbatim.
  random    ETA * u * spread, u ~ U(0,1) drawn fresh per slot per epoch from a
            DEDICATED generator. Carries zero information about age, history
            or value. Tests whether the effect is tie-breaking.
  binary    ETA * (age > 0) * spread. Knows only "was this slot rewired last
            epoch", not how long it has survived. Tests whether graded
            accumulation matters or mere recency is enough.
  inverted  ETA * (1 - min(age, 10)/10) * spread. Protects YOUNG slots --
            the exact opposite policy. Tests whether protecting established
            structure is what helps.

PRE-REGISTERED DECISIONS, WRITTEN BEFORE THE FIRST RUN:
  REPLICATION (primary). `graded` must beat `none` on reversal new_14 at 20
  seeds with non-overlapping 95% bootstrap CIs. If it does not, Exp 57's
  finding was a small-sample artifact, this file reports that, and no
  mechanism question arises.

  Given replication, the mechanism is read off this table and NOT renegotiated
  afterwards:
    random separates too      -> TIE-BREAKING. The table's content is
                                 irrelevant; the retention score's tie mass is
                                 the real story. "Age" is a misnomer and Exp
                                 57's framing must be withdrawn.
    random doesn't, binary
    does                      -> RECENCY. Rotating the cut away from
                                 just-rewired slots is what matters; graded
                                 accumulation is surplus.
    only graded separates     -> ACCUMULATION. Length of survival genuinely
                                 carries information the rule can use.
    inverted ALSO separates   -> whatever the mechanism is, it is NOT
                                 protection of established structure, since
                                 the opposite policy helps as much. Any
                                 explanation claiming otherwise is dead.

  SECONDARY. Plastic-phase taught mass must stay flat (Exp 57: ~2767 for every
  arm). An arm that moves it has changed the rule's basic behaviour and its
  reversal number cannot be attributed to the reversal mechanism.

  DIAGNOSTIC, not a decision input. Cut-set churn: mean Jaccard overlap
  between consecutive epochs' cut sets, and how many distinct slots are ever
  cut. TIE-BREAKING and RECENCY both predict base has high overlap (the same
  slots cut repeatedly) and the bonus arms low overlap. Measured so the
  mechanism story has to survive contact with the rule's actual behaviour.

SCOPE. Interleaved placement only -- reversal re-learning is the endpoint and
the segregated arm barely learns the first association, so it has little to
re-learn. Twenty seeds, double Exp 57's.

Run:          python src/experiments/exp58_age_reversal_mechanism.py
Run (audit):  python src/experiments/exp58_age_reversal_mechanism.py --audit
Run (quick):  python src/experiments/exp58_age_reversal_mechanism.py --seeds 4
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

from sparse_state import SparsePairState  # noqa: E402
from spatial import SpatialGrid  # noqa: E402
from exp32b_benchmark import (N, NC, PATTERNS, STEPS_FROZEN,  # noqa: E402
                              STEPS_PLASTIC, TOTAL, make_placement)
from exp54_synthetic_wiring_audit import (LEGACY_WIRING_SEED,  # noqa: E402
                                          make_wiring, run_life_corrected,
                                          wiring_seed_for)
from exp57_extra_tables import AGE_CAP, AGE_ETA, run_life_tables  # noqa: E402

CACHE_PATH = os.path.join("shd_data", "exp58_cache.json")
ARMS = ("none", "graded", "random", "binary", "inverted")


def run_life_age(coords, cids, seed, wiring_seed=LEGACY_WIRING_SEED,
                 wiring_mode="logical", age_mode="none"):
    """exp57.run_life_tables(age=...) with the bonus POLICY parameterised.

    Everything outside the bonus expression is verbatim from
    exp54.run_life_corrected. The `random` arm draws from its own generator so
    the main RNG call sequence is untouched and all arms stay comparable.
    Exp 57 is not modified -- its results are already recorded.
    """
    rng3 = np.random.default_rng(seed)
    bonus_rng = np.random.default_rng(90000 + seed)
    src, dst, inhib = make_wiring(cids, wiring_seed, wiring_mode)
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    slot_age = np.zeros(len(src), dtype=np.int64)
    Rhat = np.zeros(3)
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g = SpatialGrid(coords, 5.0)
    nbr = [g.within(i, 5.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N // 2
    results = {}
    prev_cut = None
    overlaps = []
    ever_cut = set()

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
                base_r = {0: 0.0, 1: 1.0, 2: -1.0}[p]
            else:
                base_r = {0: 1.0, 1: 0.0, 2: -1.0}[p]
            delta = base_r - Rhat[p]
            if abs(delta) > 1e-9:
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0: V.deposit(key[0], key[1], delta * ev)
            Rhat[p] += 0.15 * delta
        if (t + 1) % 40 == 0 and in_plastic:
            C.prune_below(1e-6)
            sc = np.array([V.get(int(src[k]), int(dst[k])) for k in range(len(src))])
            if age_mode != "none":
                spread = float(sc.max() - sc.min())
                if spread > 0:
                    frac = np.minimum(slot_age, AGE_CAP) / float(AGE_CAP)
                    if age_mode == "graded":
                        b = frac
                    elif age_mode == "random":
                        b = bonus_rng.random(len(sc))
                    elif age_mode == "binary":
                        b = (slot_age > 0).astype(float)
                    elif age_mode == "inverted":
                        b = 1.0 - frac
                    else:
                        raise ValueError(age_mode)
                    sc = sc + AGE_ETA * b * spread
            cold = np.argsort(sc, kind="stable")[:swap]
            cut_set = set(int(x) for x in cold)
            if prev_cut is not None:
                inter = len(cut_set & prev_cut)
                union = len(cut_set | prev_cut)
                overlaps.append(inter / float(max(union, 1)))
            prev_cut = cut_set
            ever_cut |= cut_set
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
                        slot_age += 1
                        slot_age[cold[:n2]] = 0
                        out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
                    else:
                        slot_age += 1
            else:
                slot_age += 1
        for cp2 in [STEPS_FROZEN, STEPS_FROZEN + STEPS_PLASTIC, TOTAL]:
            if t + 1 == cp2:
                M = np.zeros((NC, NC), dtype=int)
                np.add.at(M, (cids[src], cids[dst]), 1)
                phase = {STEPS_FROZEN: "frozen",
                         STEPS_FROZEN + STEPS_PLASTIC: "plastic",
                         TOTAL: "reversal"}[cp2]
                results[phase] = {
                    "taught": int(M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]),
                    "old_03": int(M[0, 3] + M[3, 0]),
                    "new_14": int(M[1, 4] + M[4, 1]),
                    "energy": float(D2[src, dst].sum()),
                }
    results["churn"] = {"mean_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
                        "distinct_slots_cut": int(len(ever_cut))}
    return results


def boot_ci(a, n_boot=10000, seed=12345):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    m = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def audit():
    coords, cids = make_placement("topoforge")
    print("\n[P0] age_mode='none' must be bit-identical to exp54.run_life_corrected")
    ok = True
    for seed in (0, 1):
        ws = wiring_seed_for(True, seed)
        a = run_life_corrected(coords, cids, seed, wiring_seed=ws, wiring_mode="logical")
        b = run_life_age(coords, cids, seed, wiring_seed=ws, age_mode="none")
        # Compare the fields both harnesses report. run_life_corrected also
        # returns `selectivity`, which this file does not; comparing the dicts
        # whole fails on the key sets alone and says nothing about the physics.
        fields = ("taught", "energy", "old_03", "new_14")
        same = all(a[ph][k] == b[ph][k]
                   for ph in ("frozen", "plastic", "reversal") for k in fields)
        ok = ok and same
        print("    seed {}  identical: {}  (new_14 {} vs {}, taught {} vs {})".format(
            seed, same, a["reversal"]["new_14"], b["reversal"]["new_14"],
            a["plastic"]["taught"], b["plastic"]["taught"]))

    print("\n[P1] age_mode='graded' must reproduce exp57.run_life_tables(age=True)")
    for seed in (0, 1):
        ws = wiring_seed_for(True, seed)
        a = run_life_tables(coords, cids, seed, wiring_seed=ws,
                            wiring_mode="logical", age=True)
        b = run_life_age(coords, cids, seed, wiring_seed=ws, age_mode="graded")
        same = (a["reversal"]["new_14"] == b["reversal"]["new_14"]
                and a["plastic"]["taught"] == b["plastic"]["taught"])
        ok = ok and same
        print("    seed {}  identical: {}  (new_14 {} vs {})".format(
            seed, same, a["reversal"]["new_14"], b["reversal"]["new_14"]))

    print("\n    {}".format(
        "AUDIT PASSED -- this file reproduces both prior harnesses exactly" if ok
        else "AUDIT FAILED -- do NOT interpret any result from this file"))
    if not ok:
        sys.exit(1)


def main(args):
    t0 = time.time()
    coords, cids = make_placement("topoforge")
    out = {a: {"new_14": [], "taught": [], "old_03": [],
               "overlap": [], "distinct": []} for a in ARMS}
    for arm in ARMS:
        for seed in args.seeds:
            ws = wiring_seed_for(True, seed)
            r = run_life_age(coords, cids, seed, wiring_seed=ws, age_mode=arm)
            out[arm]["new_14"].append(r["reversal"]["new_14"])
            out[arm]["old_03"].append(r["reversal"]["old_03"])
            out[arm]["taught"].append(r["plastic"]["taught"])
            out[arm]["overlap"].append(r["churn"]["mean_overlap"])
            out[arm]["distinct"].append(r["churn"]["distinct_slots_cut"])
            print("    {:<9s} seed {:>2d}  new_14 {:>5d}  taught {:>6d}  "
                  "cut-overlap {:.3f}  [{:.0f}s]".format(
                      arm, seed, r["reversal"]["new_14"], r["plastic"]["taught"],
                      r["churn"]["mean_overlap"], time.time() - t0), flush=True)

    base = np.array(out["none"]["new_14"], float)
    b_lo, b_hi = boot_ci(base)
    print("\n  PRIMARY -- reversal re-learning (new_14), {} seeds".format(len(args.seeds)))
    print("    {:<9s} {:>8s} {:>8s} {:>22s} {:>9s}".format(
        "arm", "new_14", "vs none", "95% CI", "separated"))
    sep = {}
    for arm in ARMS:
        r = np.array(out[arm]["new_14"], float)
        lo, hi = boot_ci(r)
        s = (lo > b_hi) or (hi < b_lo)
        sep[arm] = bool(s) and arm != "none"
        print("    {:<9s} {:>8.1f} {:>7.1f}% {:>16s} {:>9s}".format(
            arm, r.mean(),
            100 * (r.mean() - base.mean()) / max(base.mean(), 1e-9),
            "[{:.0f}, {:.0f}]".format(lo, hi),
            "--" if arm == "none" else ("YES" if s else "no")))

    print("\n  SECONDARY -- plastic taught mass must stay flat")
    for arm in ARMS:
        tm = np.array(out[arm]["taught"], float)
        print("    {:<9s} {:>8.1f}  ({:+.2f}% vs none)".format(
            arm, tm.mean(),
            100 * (tm.mean() - np.mean(out["none"]["taught"])) / np.mean(out["none"]["taught"])))

    print("\n  DIAGNOSTIC -- cut-set churn")
    for arm in ARMS:
        print("    {:<9s} consecutive-epoch overlap {:.3f}   distinct slots ever cut {:.0f}"
              .format(arm, np.mean(out[arm]["overlap"]), np.mean(out[arm]["distinct"])))

    print("\n  PRE-REGISTERED MECHANISM READ-OFF")
    if not sep["graded"]:
        verdict = ("NO REPLICATION. Exp 57's +67% does not survive 20 seeds. "
                   "Withdraw it; no mechanism question arises.")
    elif sep["random"]:
        verdict = ("TIE-BREAKING. A bonus carrying no information at all "
                   "reproduces the effect, so the retention score's tie mass is "
                   "the story and 'age' is a misnomer. Withdraw Exp 57's framing.")
    elif sep["binary"]:
        verdict = ("RECENCY. Knowing only 'was this slot just rewired' suffices; "
                   "graded accumulation is surplus.")
    else:
        verdict = ("ACCUMULATION. Length of survival carries information the "
                   "rule genuinely uses.")
    print("    " + verdict)
    if sep.get("inverted"):
        print("    AND: the INVERTED policy helps too. Whatever this is, it is NOT")
        print("    protection of established structure -- the opposite policy works")
        print("    as well. Any account claiming otherwise is refuted.")

    with open(CACHE_PATH, "w") as fh:
        json.dump({"raw": out, "separated": sep, "verdict": verdict}, fh, indent=1)
    print("\n  cached -> {}  ({:.0f}s)".format(CACHE_PATH, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    a.seeds = list(range(a.seeds))
    if a.audit:
        audit()
    else:
        main(a)
