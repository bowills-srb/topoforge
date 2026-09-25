"""Exp 60: does the recency flag buy ACCURACY, or only edges?

WHY THIS EXISTS. Exp 58 found a one-bit "was this slot just rewired" flag
improves reversal re-learning by 61.5%, and Exp 59 found it preferentially
rescues segregated placement, halving the adaptation gap (5.66x -> 2.76x).
Both are STRUCTURAL counts -- edges landing on the newly-rewarded type pair.

This project has been burned by exactly that gap before. Exp 52b measured the
accuracy endpoint for the first time and found a 3.59x structural ratio was
1.079x in accuracy terms: the structure moved enormously, held-out
classification moved 6.94 pp (p=0.012, d=0.57). Structural effects in this
harness are not safe proxies for performance, and nothing from Exp 58/59 may
be stated in performance terms until this file reports.

DESIGN. Exp 52b's real-data harness already contains a reward mapping --
class 0 should drive output A, class 1 should drive output B. Reversing it is
therefore native, not bolted on: phase 1 trains under that mapping, phase 2
continues the SAME network under the swapped mapping, and we ask how well the
network re-maps. Two placements x flag off/on = four cells.

THE METRIC PROBLEM, AND THE FIX. Exp 52b's endpoint is a ridge decoder free to
weight every output neuron. That decoder is BLIND to re-mapping: if the
network swaps which cluster carries which class, a free decoder simply
re-weights and reports the same accuracy. Applying it unchanged here would
measure the wrong thing and would likely return a null for a reason that has
nothing to do with the flag. So:

  PRIMARY (mapping-aware). A FIXED readout with no fitted parameters: predict
  the class from which output cluster is more active, under the CURRENT
  mapping. Post-reversal that means class 0 -> cluster B. This is the accuracy
  analogue of new_14 and it is the endpoint the pre-registration below refers
  to.

  CONTROL (Exp 52b's imported ridge decoder, verbatim). Confirms class
  information still reaches the output layer at all. If PRIMARY is flat while
  CONTROL is healthy, the network kept the information and failed to re-map --
  a real finding. If both collapse, the substrate degraded and no claim about
  re-mapping can be made either way.

Both come from the same feature matrix, so the control costs no extra
simulation.

PRE-REGISTERED, BEFORE THE FIRST RUN:
  P1 MANIPULATION CHECK. Post-phase-1 mapping-aware accuracy must exceed 60%
  in both placements. Below that there is no learned mapping to reverse and
  the reversal comparison is meaningless -- stop and report that, do not
  interpret P2.
  P2 PRIMARY. Within each placement, does the flag raise post-reversal
  mapping-aware accuracy? Reported with bootstrap CIs on the difference.
  P3 THE EXP 59 QUESTION IN ACCURACY TERMS. Does the placement gap in
  post-reversal accuracy narrow with the flag on, as it did structurally?
  Exp 59's structural answer was RESCUE (5.66x -> 2.76x). If accuracy does not
  reproduce that, Exp 59's practical reading must be withdrawn.

REGISTERED PREDICTION, stated so a miss is a miss. Given Exp 52b's precedent I
expect the accuracy effect to be REAL BUT MUCH SMALLER than the structural
one -- single-digit percentage points, not 61%. I have been wrong twice in
this thread already (tie-breaking, and the direction of Exp 59), so this is a
weak prior, recorded rather than relied on.

POWER, STATED HONESTLY UP FRONT. Exp 52b needed 40 seeds to resolve d=0.57 at
p=0.012. This runs 24 seeds across four cells (~2.7 h). At d=0.57 that is
roughly 50% power, so A NULL HERE IS NOT EVIDENCE OF ABSENCE -- it is an
underpowered null and must be reported as one. Results cache per cell per
seed, so the seed count can be extended to 40 later without recomputation.

REUSES, NOT REIMPLEMENTS. `load_split`, `evaluate_features`,
`ridge_cv_accuracy`, both placements and the type constants are imported from
exp52/exp52b. The simulation loop is exp52b.run_life_corrected with two
additions -- a reward-mapping parameter and the recency flag -- and --audit
asserts that with the flag off and no phase 2 it reproduces exp52b's
bridge values bit-identically.

Run:          python src/experiments/exp60_reversal_accuracy.py
Run (audit):  python src/experiments/exp60_reversal_accuracy.py --audit
Run (quick):  python src/experiments/exp60_reversal_accuracy.py --seeds 2
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
import exp52b_accuracy_corrected as E52B  # noqa: E402
from exp52_accuracy_endpoint import (  # noqa: E402
    N, INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE, evaluate_features,
    load_split, make_placement_interleaved, make_placement_segregated,
    ridge_cv_accuracy)

CACHE_PATH = os.path.join("shd_data", "exp60_cache.json")
SEEDS = list(range(300, 324))        # 24 seeds, disjoint from exp52b's 100-139
AGE_ETA = 0.25                       # same magnitude as exp57/58
P1_FLOOR = 60.0                      # registered manipulation-check floor
PLACEMENTS = [("segregated", make_placement_segregated),
              ("interleaved", make_placement_interleaved)]
MODES = ("none", "binary")


def _cache_load():
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH) as fh:
            return json.load(fh)
    return {}


def _cache_save(c):
    with open(CACHE_PATH, "w") as fh:
        json.dump(c, fh)


def run_two_phase(coords, cids, seed, target_samples, distractor_samples,
                  wiring_seed, inhib_mode, age_mode="none", n_epochs=1,
                  input_gain=0.35, swap=200, do_phase2=True):
    """exp52b.run_life_corrected, plus a reward-mapping switch and the flag.

    Phase 1 trains under class0->A / class1->B. Phase 2 continues the SAME
    network under the swapped mapping. Returns the connectivity after each
    phase so both can be evaluated. Everything not guarded by `age_mode` or
    `phase` is verbatim from exp52b.
    """
    rng = np.random.default_rng(seed)
    src, dst, inhib = E52B.make_wiring(cids, wiring_seed, inhib_mode)
    src = np.asarray(src).copy(); dst = np.asarray(dst).copy()
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.99995)
    Rhat = np.zeros(2)
    slot_age = np.zeros(len(src), dtype=np.int64)
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    input_idx = np.where(cids == INPUT_TYPE)[0]
    a_idx = np.where(cids == OUTPUT_A_TYPE)[0]
    b_idx = np.where(cids == OUTPUT_B_TYPE)[0]
    a_set = set(a_idx.tolist()); b_set = set(b_idx.tolist())

    combined = [(s, 0) for s in target_samples] + [(s, 1) for s in distractor_samples]
    t = 0
    snapshots = {}
    phases = [1, 2] if do_phase2 else [1]

    for phase in phases:
        for _epoch in range(n_epochs):
            for sample_arr, class_label in combined:
                recent_a = False; recent_b = False
                for local_t in range(sample_arr.shape[0]):
                    inp = rng.uniform(0, 0.01, N)
                    inp[input_idx] += sample_arr[local_t] * input_gain
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
                                if int(j) in fs:
                                    C.deposit(i, int(j), 1.0); E.deposit(i, int(j), 1.0)
                        if fs & a_set: recent_a = True
                        if fs & b_set: recent_b = True
                    v = np.maximum(v_, 0); v[fired] = 0
                    refrac[fired] = 3; refrac[refrac > 0] -= 1

                    if t % 20 == 6:
                        stream = class_label
                        # THE ONLY TASK CHANGE: phase 2 swaps which output
                        # cluster counts as correct for each class.
                        if phase == 1:
                            correct = recent_a if class_label == 0 else recent_b
                        else:
                            correct = recent_b if class_label == 0 else recent_a
                        base = 1.0 if correct else 0.0
                        delta = base - Rhat[stream]
                        if abs(delta) > 1e-9:
                            E.prune_below(1e-6)
                            for key in list(E.store.keys()):
                                ev = E.get(*key)
                                if ev != 0: V.deposit(key[0], key[1], delta * ev)
                        Rhat[stream] += 0.15 * delta
                        recent_a = False; recent_b = False
                    t += 1

                for _ in range(15):
                    v_ = v * 0.90 + rng.uniform(0, 0.01, N)
                    fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
                    if len(f):
                        for fi in f:
                            for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
                    C.tick(); E.tick(); V.tick()
                    v = np.maximum(v_, 0); v[fired] = 0
                    refrac[fired] = 3; refrac[refrac > 0] -= 1
                    t += 1

                if t % 40 < 16:
                    C.prune_below(1e-6)
                    sc = np.array([V.get(int(src[k]), int(dst[k])) for k in range(len(src))])
                    if age_mode == "binary":
                        spread = float(sc.max() - sc.min())
                        if spread > 0:
                            sc = sc + AGE_ETA * (slot_age > 0).astype(float) * spread
                    cold = np.argsort(sc, kind="stable")[:swap]
                    ci, cj, _ = C.get_arrays()
                    if len(ci) > 0:
                        keep2 = ci != cj; ci, cj = ci[keep2], cj[keep2]
                        ex = set(zip(src.tolist(), dst.tolist()))
                        mk = np.array([(int(a), int(b)) not in ex for a, b in zip(ci, cj)])
                        ci, cj = ci[mk], cj[mk]
                        if len(ci) > 0:
                            diff = coords[ci] - coords[cj]
                            dd = (diff ** 2).sum(axis=1)
                            vp = np.maximum(np.array([V.get(int(a), int(b))
                                                      for a, b in zip(ci, cj)]), 0)
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
                                    out_t[s2].append(d2b)
                                    out_w[s2].append(-0.60 if inhib[s2] else 0.30)
                            else:
                                slot_age += 1
                    else:
                        slot_age += 1
        snapshots[phase] = (src.copy(), dst.copy())

    bridge_a = 0.0; bridge_b = 0.0
    in_set = set(input_idx.tolist())
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if si in in_set and di in a_set: bridge_a += max(V.get(si, di), 0)
        elif di in in_set and si in a_set: bridge_a += max(V.get(di, si), 0)
        if si in in_set and di in b_set: bridge_b += max(V.get(si, di), 0)
        elif di in in_set and si in b_set: bridge_b += max(V.get(di, si), 0)
    return snapshots, inhib, bridge_a, bridge_b


def mapping_accuracy(feats, labels, n_a, phase):
    """Fixed readout, no fitted parameters: which cluster is more active?

    Columns of `feats` are [A neurons..., B neurons...] (evaluate_features
    orders them that way). Phase 1 maps class 0 -> A; phase 2 -> B.
    """
    sa = feats[:, :n_a].sum(1)
    sb = feats[:, n_a:].sum(1)
    pred_a_is_class0 = (phase == 1)
    pred = np.where(sa > sb, 0 if pred_a_is_class0 else 1,
                    1 if pred_a_is_class0 else 0)
    return 100.0 * float((pred == np.asarray(labels)).mean())


def boot_diff_ci(a, b, n_boot=10000, seed=12345):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.array([a[rng.integers(0, len(a), len(a))].mean()
                  - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n_boot)])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _data():
    top, avail, plastic, evalset = load_split()
    tgt = plastic[top[0]]; dis = plastic[top[1]]
    ev_s = list(evalset[top[0]]) + list(evalset[top[1]])
    ev_l = [0] * len(evalset[top[0]]) + [1] * len(evalset[top[1]])
    return tgt, dis, ev_s, ev_l


def audit():
    print("\n[P0] flag off + no phase 2 must reproduce exp52b.run_life_corrected")
    tgt, dis, _, _ = _data()
    coords, cids = make_placement_interleaved()
    ok = True
    for seed in (100, 101):
        ws = E52B.LEGACY_WIRING_SEED + (seed - 100)
        ba0, bb0, _, _, _ = E52B.run_life_corrected(
            coords, cids, seed, tgt, dis, wiring_seed=ws, inhib_mode=E52B.WIRING_MODE)
        _, _, ba1, bb1 = run_two_phase(
            coords, cids, seed, tgt, dis, wiring_seed=ws,
            inhib_mode=E52B.WIRING_MODE, age_mode="none", do_phase2=False)
        same = (abs(ba0 - ba1) < 1e-9) and (abs(bb0 - bb1) < 1e-9)
        ok = ok and same
        print("    seed {}  identical: {}  (bridge_a {:.3f} vs {:.3f})".format(
            seed, same, ba0, ba1))
    print("    {}".format("P0 PASSED" if ok else
                          "P0 FAILED -- do NOT interpret any result from this file"))
    if not ok:
        sys.exit(1)


def main(args):
    t0 = time.time()
    tgt, dis, ev_s, ev_l = _data()
    n_a = int((make_placement_interleaved()[1] == OUTPUT_A_TYPE).sum())
    cache = _cache_load()
    out = {}
    for label, mk in PLACEMENTS:
        coords, cids = mk()
        out[label] = {}
        for mode in MODES:
            cell = {"p1_map": [], "p2_map": [], "p2_ridge": [], "p1_ridge": []}
            for seed in args.seeds:
                key = "{}|{}|{}".format(label, mode, seed)
                if key in cache:
                    rec = cache[key]
                else:
                    ws = E52B.LEGACY_WIRING_SEED + (seed - SEEDS[0])
                    snaps, inhib, _, _ = run_two_phase(
                        coords, cids, seed, tgt, dis, wiring_seed=ws,
                        inhib_mode=E52B.WIRING_MODE, age_mode=mode)
                    rec = {}
                    for ph in (1, 2):
                        s, d = snaps[ph]
                        F = evaluate_features(coords, cids, s, d, inhib, ev_s,
                                              eval_seed=900 + seed)
                        rec["p{}_map".format(ph)] = mapping_accuracy(F, ev_l, n_a, ph)
                        rec["p{}_ridge".format(ph)] = float(ridge_cv_accuracy(F, ev_l))
                    cache[key] = rec
                    _cache_save(cache)
                for k in cell:
                    cell[k].append(rec[k])
                print("    {:<12s} {:<7s} seed {}  phase1 {:5.1f}%  phase2 {:5.1f}%"
                      "  (ridge {:5.1f}%)  [{:.0f}s]".format(
                          label, mode, seed, rec["p1_map"], rec["p2_map"],
                          rec["p2_ridge"], time.time() - t0), flush=True)
            out[label][mode] = cell

    print("\n[P1] MANIPULATION CHECK -- post-phase-1 mapping accuracy must exceed"
          " {:.0f}%".format(P1_FLOOR))
    p1_ok = True
    for label, _ in PLACEMENTS:
        m = np.mean(out[label]["none"]["p1_map"])
        good = m > P1_FLOOR
        p1_ok = p1_ok and good
        print("    {:<12s} {:5.1f}%   {}".format(label, m, "OK" if good else "BELOW FLOOR"))
    if not p1_ok:
        print("    P1 FAILED: there is no learned mapping to reverse. P2/P3 are")
        print("    uninterpretable and are NOT reported. This is a harness result,")
        print("    not a finding about the flag.")
        _cache_save(cache)
        return

    print("\n[P2] PRIMARY -- post-reversal mapping accuracy, flag off vs on")
    for label, _ in PLACEMENTS:
        a = np.array(out[label]["none"]["p2_map"], float)
        b = np.array(out[label]["binary"]["p2_map"], float)
        lo, hi = boot_diff_ci(b, a)
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        print("    {:<12s} none {:5.1f}%   binary {:5.1f}%   diff {:+5.2f} pp"
              "   95% CI [{:+.2f}, {:+.2f}]   {}".format(
                  label, a.mean(), b.mean(), b.mean() - a.mean(), lo, hi, sig))

    print("\n[CONTROL] free ridge decoder -- is class information still present?")
    for label, _ in PLACEMENTS:
        for mode in MODES:
            print("    {:<12s} {:<7s} phase1 {:5.1f}%   phase2 {:5.1f}%".format(
                label, mode, np.mean(out[label][mode]["p1_ridge"]),
                np.mean(out[label][mode]["p2_ridge"])))

    print("\n[P3] THE EXP 59 QUESTION -- placement gap in post-reversal accuracy")
    for mode in MODES:
        i = np.mean(out["interleaved"][mode]["p2_map"])
        s = np.mean(out["segregated"][mode]["p2_map"])
        print("    flag {:<7s} interleaved {:5.1f}%  segregated {:5.1f}%  gap {:+5.2f} pp"
              .format(mode, i, s, i - s))
    gap_none = (np.mean(out["interleaved"]["none"]["p2_map"])
                - np.mean(out["segregated"]["none"]["p2_map"]))
    gap_bin = (np.mean(out["interleaved"]["binary"]["p2_map"])
               - np.mean(out["segregated"]["binary"]["p2_map"]))
    print("    structural (Exp 59) was RESCUE: 5.66x -> 2.76x.")
    print("    accuracy gap {:+.2f} pp -> {:+.2f} pp: {}".format(
        gap_none, gap_bin,
        "same direction, Exp 59's reading holds" if gap_bin < gap_none
        else "OPPOSITE direction -- Exp 59's practical reading does NOT carry to accuracy"))

    print("\n  NOTE ON POWER: 24 seeds, ~50% power at Exp 52b's d=0.57. A null here")
    print("  is an UNDERPOWERED null, not evidence of absence. Cache is per-seed;")
    print("  extend SEEDS to 40 to match Exp 52b without recomputing.")
    _cache_save(cache)
    print("\n  cached -> {}  ({:.0f}s)".format(CACHE_PATH, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args()
    a.seeds = SEEDS[:a.seeds]
    if a.audit:
        audit()
    else:
        main(a)
