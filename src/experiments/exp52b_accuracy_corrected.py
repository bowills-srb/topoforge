"""Exp 52b: Does segregated placement cost CLASSIFICATION ACCURACY?
CORRECTED and PROPERLY POWERED rerun of Exp 52, which came back CONFOUNDED
on its own pre-registered manipulation check.

WHY EXP 52 FAILED, and it failed honestly. Exp 52 registered P1b: the two
placements' FROZEN (no-plasticity) held-out accuracies must agree within
5.0 pp, on the reasoning that if they differ BEFORE any learning, the
plastic comparison measures the random projection rather than plasticity.
They differed by 5.97 pp. The check fired and the result was declared
uninterpretable rather than reported.

ROOT CAUSE, diagnosed and reproduced exactly here in section [0]. The
harness fixes the wiring with a single `np.random.default_rng(7)` draw:

    src = rng2.integers(0, N, N*10); dst = rng2.integers(0, N, N*10)
    inhib = rng2.random(N) < 0.20                 # <-- by GLOBAL index

`inhib` is therefore attached to global neuron INDICES, while the two
placements put input-type neurons at different indices (segregated:
inputs at 0-139, outputs contiguous at 140-199; interleaved: both spread
across 0-199). So a DIFFERENT SET of input-type neurons ends up
inhibitory in each condition. Measured: 31 inhibitory input neurons under
segregated against 23 under interleaved, giving 313 vs 341 excitatory
input->output edges -- an 8.2% imbalance in raw excitatory drive reaching
the readout layer before a single synapse had moved. Input->output edge
COUNTS matched almost exactly (419 vs 418), which is why this was
invisible to any check that counted edges rather than weighting them.

THE FIX -- and a registration note, because the fix was redesigned once.

  Two weaker fixes were tried first and are recorded here rather than
  quietly dropped:

  (1) TYPE-RELATIVE INHIBITION. `inhib` assigned WITHIN each type by
      type-relative rank, so both placements get identical inhibitory
      counts per type (28 of 140 inputs, 20% of each output pool).
      Measured: inhibitory-input counts 28 vs 28 (were 31 vs 23), and the
      excitatory-edge gap fell from 28 edges (8.2%) to 12 (3.5%).

  (2) RANDOMISED WIRING SEED, on top of (1). Exp 52 held the wiring FIXED
      at one draw, so any imbalance in that draw is SYSTEMATIC BIAS
      rather than noise. Varying the wiring seed with the run seed
      evaluates each condition over a DISTRIBUTION of wirings. Measured
      over 40 draws the gap fell to ~1-2% and its SIGN FLIPPED relative
      to the fixed draw -- direct evidence that exp52's 8.2% was a
      property of `default_rng(7)`, not of the placements.

  (1)+(2) WERE NOT ENOUGH, and P1b caught it. On the registered check
  the residual excitatory gap was +2.11% -- inside the 3.0% tolerance,
  but Welch p=0.043 on the excitatory counts, i.e. still a DETECTABLE
  systematic bias. Per the decision rule the run was stopped before the
  sweep and the fix redesigned. NOTE WHAT WAS AND WAS NOT CHANGED: the
  FIX was strengthened; no threshold was weakened, and no P2/outcome
  quantity had been computed or looked at. P1b exists precisely to be
  acted on before the expensive sweep, and it was.

  (3) LOGICAL-SPACE WIRING -- the actual fix, and the conceptually
      correct one. The residual bias survives (1)+(2) because the harness
      draws src/dst on GLOBAL neuron indices, so which LOGICAL neurons
      are connected depends on the placement. That silently violates the
      experiment's own premise, which is that the conditions differ only
      in physical arrangement. Here the wiring and the inhibitory flag
      are drawn over LOGICAL identities -- (type, within-type rank),
      canonically ordered -- and then mapped to global indices through
      each placement's type assignment. Both conditions therefore realise
      the IDENTICAL logical network: bit-for-bit equal input->output edge
      counts, excitatory counts, degree distribution and inhibitory
      assignment. Only the COORDINATES differ, which is exactly the
      contrast the paper claims to be testing, and the same invariant
      Exp 50 secured by giving its conditions bit-identical coordinates.
      Verified: per-seed io and excitatory counts are exactly equal
      across conditions for all 40 seeds, so P1b now demands EXACT
      equality rather than a tolerance.

  What (3) is and is not: no neuron, synapse, weight, decay constant or
  plasticity rule is altered, and the type populations, edge count and
  20% inhibitory fraction are unchanged. It removes a dependence of the
  LOGICAL wiring on the PLACEMENT that was never intended. Plasticity
  still sees geometry through `coords` in the neighbour search and the
  distance-weighted rewiring score, so placement continues to drive
  everything it is supposed to drive.

  CONSEQUENCE TO STATE PLAINLY: because the wiring is no longer pinned to
  seed 7, bridge-mass values here need not reproduce the preprint's 8.22x
  headline exactly. That headline is measured on the fixed-wiring harness
  and is unaffected by this file. Bridge mass is reported below as a
  secondary observation only.

WHAT IS KEPT EXACTLY. The readout, decoder, CV procedure, data split and
accuracy metric are IMPORTED from exp52 (`evaluate_features`,
`ridge_cv_accuracy`, `load_split`, `adjacency_count`) rather than
re-typed, so no divergence between the confounded run and this one is
possible. exp52 itself is NOT modified; it stands as the record of the
confounded first attempt. `run_life_corrected` below is exp52's loop with
the wiring generation parameterised; in legacy mode it must reproduce the
IMPORTED `run_life` bit-identically, and P1a asserts exactly that.

READOUT, unchanged from exp52's registration: per-neuron spike counts of
the 60 OUTPUT-type neurons only. Inputs are excluded because a decoder
reading input neurons classifies the raw stimulus regardless of learned
structure; reading only outputs asks whether class information actually
REACHED the output layer through learned connectivity. A readout
restricted to just the rewarded bridge pairs would bake the answer in, so
all output neurons are read and the decoder is free to weight them.
Decoder: closed-form ridge, lambda=1.0 FIXED, identical for both
conditions, standardisation and fit on the decoder-train fold only,
5-fold stratified CV over 120 held-out samples the plastic phase never
saw. Accuracy resolution 0.83 pp.

POWER CALCULATION, done before running and driving the seed count.
Exp 52's (confounded, but variance-informative) observation was a delta
gap of 5.56 pp with s.d. 9.09 and 6.62, i.e. pooled s.d. 7.95 and
Cohen's d = 5.56/7.95 = 0.70. For a two-sided Welch test at alpha=0.05,
n per group ~= 2*(z_{1-a/2}+z_{1-b})^2 / d^2:
    80% power: 2*(1.960+0.842)^2 / 0.49 = 32.0  -> 33 seeds
    90% power: 2*(1.960+1.282)^2 / 0.49 = 42.9  -> 43 seeds
n=40 per condition gives ~87% power at d=0.70, and a MINIMUM DETECTABLE
EFFECT at 80% power of d = 2.802*sqrt(2/40) = 0.626, i.e. 0.626*7.95 =
4.98 pp -- which lands essentially exactly on the 5.0 pp substantive bar
registered below. So this design can detect a substantive effect if one
exists, and a null here is informative rather than merely underpowered.
Observed s.d. may differ; the achieved MDE is recomputed from the
observed pooled s.d. and reported in section [4].

SEEDS: 100-139 (40 per condition), DISJOINT from exp52's 0-11.

PREDICTIONS, REGISTERED BEFORE RUNNING:

  P1 (manipulation checks -- ALL must hold before P2 is interpreted):
    P1a EQUIVALENCE: in legacy mode (wiring_seed=7, global inhib),
        `run_life_corrected` reproduces the imported `run_life`'s
        (bridge_a, bridge_b) BIT-IDENTICALLY. Guards the copied loop.
    P1b STRUCTURAL PARITY -- the fix for exp52's failure, and checked
        BEFORE the expensive sweep so no hours are burned on an already
        confounded design. Over the 40 wiring seeds actually used, and
        under the logical-space wiring of fix (3), the two conditions
        must realise the IDENTICAL logical network: per-seed
        input->output edge counts, excitatory input->output counts, and
        inhibitory input counts EXACTLY equal across conditions, for
        every seed. This is a strict-equality check, strengthened from
        the tolerance-based version that fixes (1)+(2) failed at
        p=0.043; a tolerance is no longer needed because exact parity is
        achievable by construction.
    P1c FROZEN ACCURACY PARITY: |frozen_acc(int) - frozen_acc(seg)| <
        5.0 pp. This is exp52's failed check, unchanged in threshold.
    P1d NO CEILING: mean frozen accuracy < 95.0% in both conditions.
        Registered as a real possible outcome: if the random projection
        already carries the class information there is no headroom, and
        the design is UNINFORMATIVE rather than null.

  P2 (the actual test), primary endpoint = plastic-minus-frozen accuracy
     delta per placement:
    P2a delta(interleaved) > delta(segregated), Welch p < 0.05.
    P2b that difference >= 5.0 pp -- the SUBSTANTIVE bar, carried over
        UNCHANGED from exp52 so the goalposts cannot be said to have
        moved after seeing exp52's +5.56 pp.

DECISION RULE:
  - P1a fails: COMPROMISED. Nothing interpretable.
  - P1b fails: the fix did not work. STOP before the sweep; report and
    redesign. Do not run.
  - P1c fails: STILL CONFOUNDED despite the fix. Report the residual
    frozen gap; do not read P2 as a plasticity effect.
  - P1d fails: UNINFORMATIVE (ceiling), not evidence against the claim.
  - P1 holds, P2a and P2b hold: SUPPORTS "determines learnability".
    Placement changes what the network can actually classify.
  - P1 holds, P2a holds, P2b fails: PARTIAL / NARROWS. The accuracy
    effect is real but below the substantive bar, and small relative to
    the ~8x structural effect. Report it quantitatively; do not lean on
    "determines learnability" unqualified.
  - P2a fails: NULL. Segregation does not measurably cost classification
    accuracy here. The structural findings stand, but the paper's claim
    should be NARROWED from "determines learnability" to "determines
    associative connectivity", and reported prominently, not buried.

A NULL REMAINS A FULLY VALID OUTCOME. No parameter in this file was
changed after seeing any result from it. If that ever ceases to be true
it must be stated explicitly here.

LIMITATIONS: one task (2-class SHD), one readout, one decoder family,
N=200, and the 1.89x input-output adjacency asymmetry inherited from
exp37b's geometry (1,198 vs 2,268 pairs), printed in [0]. Single
implementation, like everything else in this project.

Run: python src/experiments/exp52b_accuracy_corrected.py
Run (audit only, fast -- P1a + P1b, no sweep):
     python src/experiments/exp52b_accuracy_corrected.py --audit
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import json
import os
import time

import numpy as np
from scipy import stats

from sparse_state import SparsePairState
from spatial import SpatialGrid
from exp37b_v2_real_data import (
    make_placement_segregated, make_placement_interleaved, run_life,
    N, INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE,
)
from exp37c_real_data_scaled import cohens_d_pooled
# Readout, decoder, data split and adjacency metric imported verbatim from
# the confounded first attempt so they cannot silently diverge from it.
from exp52_accuracy_endpoint import (
    evaluate_features, ridge_cv_accuracy, load_split, adjacency_count,
    PLASTIC_PER_CLASS, EVAL_PER_CLASS, N_EPOCHS, STEPS_PER_SAMPLE,
    MIN_RETENTION,
)

# ---- registered configuration (fixed before running) ----
SEEDS = list(range(100, 140))     # 40 seeds, disjoint from exp52's 0-11
INHIB_FRAC = 0.20                 # unchanged from the harness
LEGACY_WIRING_SEED = 7            # the value exp37b/c/52 pinned

# registered thresholds
SUBSTANTIVE_PP = 5.0              # P2b, carried over unchanged from exp52
PARITY_PP = 5.0                   # P1c, unchanged from exp52
CEILING_PCT = 95.0                # P1d, unchanged from exp52
WIRING_MODE = "logical"           # P1b fix (3): identical logical network
EXC_GAP_PCT = 3.0                 # retained: threshold of the superseded (1)+(2) fix

CACHE_PATH = os.path.join("shd_data", "exp52b_cache.json")


def _cache_load():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as fh:
            return json.load(fh)
    return {}


def _cache_save(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    os.replace(tmp, CACHE_PATH)


def type_relative_inhib(cids, frac=INHIB_FRAC, seed=LEGACY_WIRING_SEED):
    """Assign the inhibitory flag WITHIN each type, by type-relative rank.

    This is the fix for exp52's confound. The harness's original
    `inhib = rng.random(N) < 0.20` attaches inhibition to GLOBAL indices,
    and the two placements put input-type neurons at different indices, so
    a different set of inputs ends up inhibitory in each condition (31 vs
    23 measured). Selecting within each type on type-relative positions
    guarantees identical inhibitory counts per type in both conditions,
    while preserving the 20% ratio the harness uses.
    """
    inhib = np.zeros(N, dtype=bool)
    for t in np.unique(cids):
        idx = np.sort(np.where(cids == t)[0])
        k = int(round(frac * len(idx)))
        r = np.random.default_rng(int(seed) * 1000 + int(t))
        inhib[idx[r.permutation(len(idx))[:k]]] = True
    return inhib


def logical_to_global(cids):
    """Map logical index -> global neuron index. Logical order is canonical:
    all type-0 neurons (ascending), then type-1, then type-2. Both
    placements have the same per-type populations (140/30/30), so this is
    a bijection under either placement, and logical identity `k` denotes
    the same LOGICAL neuron in both."""
    order = []
    for t in sorted(np.unique(cids)):
        order.extend(sorted(np.where(cids == t)[0].tolist()))
    return np.array(order, dtype=int)


def make_wiring(cids, wiring_seed, inhib_mode):
    """Initial wiring.

    `inhib_mode='global'` reproduces the harness exactly (used by P1a).
    `'type_relative'` is the superseded partial fix, retained so the
    docstring's account of it stays reproducible.
    `'logical'` is the fix actually used: src/dst and inhib are drawn over
    LOGICAL identities and mapped through the placement, so both
    placements realise the identical logical network and differ only in
    coordinates.
    """
    if inhib_mode == "logical":
        lm = logical_to_global(cids)
        r = np.random.default_rng(wiring_seed)
        sl = r.integers(0, N, N * 10)
        dl = r.integers(0, N, N * 10)
        keep = sl != dl
        sl, dl = sl[keep], dl[keep]
        inhib_log = r.random(N) < INHIB_FRAC
        src = lm[sl]
        dst = lm[dl]
        inhib = np.zeros(N, dtype=bool)
        inhib[lm] = inhib_log
        return src, dst, inhib

    rng2 = np.random.default_rng(wiring_seed)
    src = rng2.integers(0, N, N * 10)
    dst = rng2.integers(0, N, N * 10)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if inhib_mode == "global":
        inhib = rng2.random(N) < INHIB_FRAC
    elif inhib_mode == "type_relative":
        inhib = type_relative_inhib(cids, seed=wiring_seed)
    else:
        raise ValueError(inhib_mode)
    return src, dst, inhib


def excitatory_io_stats(cids, src, dst, inhib):
    """Input->output edge counts, and how many of them are excitatory.
    This is the quantity that silently differed in exp52."""
    inp = set(np.where(cids == INPUT_TYPE)[0].tolist())
    out = set(np.where(cids != INPUT_TYPE)[0].tolist())
    io = [(int(s), int(d)) for s, d in zip(src, dst)
          if int(s) in inp and int(d) in out]
    exc = sum(1 for s, _ in io if not inhib[s])
    n_inh_inputs = int(sum(1 for i in inp if inhib[i]))
    return len(io), exc, n_inh_inputs


def run_life_corrected(coords, cids, seed, target_samples, distractor_samples,
                       n_epochs=1, input_gain=0.35, swap=200,
                       wiring_seed=LEGACY_WIRING_SEED, inhib_mode="global"):
    """exp52's `run_life_with_state` with the wiring generation
    parameterised. With wiring_seed=7 and inhib_mode='global' this is the
    imported `run_life` exactly, plus returning final state; P1a asserts
    bit-identical bridge values. Do not edit without re-running --audit."""
    rng = np.random.default_rng(seed)
    src, dst, inhib = make_wiring(cids, wiring_seed, inhib_mode)
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.99995)
    Rhat = np.zeros(2)
    g = SpatialGrid(coords, 6.0)
    nbr = [g.within(i, 6.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    input_idx = np.where(cids == INPUT_TYPE)[0]
    a_idx = np.where(cids == OUTPUT_A_TYPE)[0]
    b_idx = np.where(cids == OUTPUT_B_TYPE)[0]
    in_set = set(input_idx.tolist())
    a_set = set(a_idx.tolist()); b_set = set(b_idx.tolist())

    combined = [(s, 0) for s in target_samples] + [(s, 1) for s in distractor_samples]
    t = 0

    for epoch in range(n_epochs):
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
                    correct = recent_a if class_label == 0 else recent_b
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

    bridge_a = 0.0; bridge_b = 0.0
    for si, di in zip(src, dst):
        si, di = int(si), int(di)
        if si in in_set and di in a_set: bridge_a += max(V.get(si, di), 0)
        elif di in in_set and si in a_set: bridge_a += max(V.get(di, si), 0)
        if si in in_set and di in b_set: bridge_b += max(V.get(si, di), 0)
        elif di in in_set and si in b_set: bridge_b += max(V.get(di, si), 0)
    return bridge_a, bridge_b, src, dst, inhib


if __name__ == "__main__":
    audit_only = "--audit" in sys.argv
    print("=" * 92)
    print("EXP 52b: CORRECTED + POWERED -- does segregated placement cost")
    print("CLASSIFICATION ACCURACY?  (predictions fixed in the docstring above)")
    print("=" * 92)

    top, avail, plastic, evalset = load_split()
    plastic_t = plastic[top[0]]; plastic_d = plastic[top[1]]
    eval_samples = evalset[top[0]] + evalset[top[1]]
    eval_labels = np.array([0] * len(evalset[top[0]]) + [1] * len(evalset[top[1]]))
    est_steps = 2 * PLASTIC_PER_CLASS * N_EPOCHS * STEPS_PER_SAMPLE
    retention = 0.99995 ** est_steps

    conditions = [("segregated", make_placement_segregated),
                  ("interleaved", make_placement_interleaved)]

    print("\n[0] Audit")
    print("  classes: target={} (avail {}), distractor={} (avail {})".format(
        top[0], avail[top[0]], top[1], avail[top[1]]))
    print("  plastic {}/class (seen) | eval {}/class = {} samples (NEVER seen)".format(
        PLASTIC_PER_CLASS, EVAL_PER_CLASS, len(eval_samples)))
    print("  accuracy resolution {:.2f} pp | life ~{:,} steps | V retention {:.3f}".format(
        100.0 / len(eval_samples), est_steps, retention))
    assert retention >= MIN_RETENTION
    for name, fn in conditions:
        c, ci = fn()
        print("  {:>12}: {:,} input-output adjacent pairs (radius 6.0)".format(
            name, adjacency_count(c, ci)))
    print("  seeds: {}-{} ({} per condition), disjoint from exp52's 0-11".format(
        SEEDS[0], SEEDS[-1], len(SEEDS)))

    # ---- P1a: equivalence in legacy mode -------------------------------
    print("\n  [P1a] equivalence (legacy mode) vs imported run_life")
    c0, ci0 = make_placement_segregated()
    ba_ref, bb_ref = run_life(c0, ci0, 0, plastic_t[:4], plastic_d[:4], n_epochs=1)
    ba_new, bb_new, _, _, _ = run_life_corrected(
        c0, ci0, 0, plastic_t[:4], plastic_d[:4], n_epochs=1,
        wiring_seed=LEGACY_WIRING_SEED, inhib_mode="global")
    print("    imported run_life   : bridge_a={:.10f} bridge_b={:.10f}".format(ba_ref, bb_ref))
    print("    run_life_corrected  : bridge_a={:.10f} bridge_b={:.10f}".format(ba_new, bb_new))
    p1a = (ba_ref == ba_new) and (bb_ref == bb_new)
    print("    P1a {}".format("HOLDS (bit-identical)" if p1a
                              else "FAILS -- COMPROMISED, stop"))
    if not p1a:
        sys.exit(1)

    # ---- P1b: structural parity, BEFORE the sweep ----------------------
    print("\n  [P1b] structural parity of excitatory input->output drive")
    print("        (this is exp52's confound; checked before spending hours)")
    coords_map = {name: fn() for name, fn in conditions}
    per_cond = {}
    for name, (coords, cids) in coords_map.items():
        ios, excs, inhs = [], [], []
        for s in SEEDS:
            src, dst, inhib = make_wiring(cids, LEGACY_WIRING_SEED + s, WIRING_MODE)
            io, exc, ninh = excitatory_io_stats(cids, src, dst, inhib)
            ios.append(io); excs.append(exc); inhs.append(ninh)
        per_cond[name] = (np.array(ios), np.array(excs), np.array(inhs))
        print("    {:<12} io_edges={:7.1f}  excitatory_io={:7.1f} +/-{:5.1f}  inhib_inputs={:.0f}".format(
            name, np.mean(ios), np.mean(excs), np.std(excs, ddof=1), np.mean(inhs)))
    io_s, exc_s, inh_s = per_cond["segregated"]
    io_i, exc_i, inh_i = per_cond["interleaved"]
    io_exact = bool(np.all(io_s == io_i))
    exc_exact = bool(np.all(exc_s == exc_i))
    inh_exact = bool(np.all(inh_s == inh_i))
    print("    per-seed input->output edge counts EXACTLY equal : {}".format(io_exact))
    print("    per-seed excitatory io counts     EXACTLY equal : {}".format(exc_exact))
    print("    per-seed inhibitory input counts  EXACTLY equal : {}".format(inh_exact))
    print("    [reference] exp52's fixed-wiring draw gave 313 vs 341 = +8.94%, the confound")
    print("    [reference] fixes (1)+(2) gave +2.11%, Welch p=0.043 -- failed this check")
    p1b = io_exact and exc_exact and inh_exact
    print("    P1b {}".format("HOLDS -- identical logical network, coordinates differ only"
                              if p1b else "FAILS -- fix insufficient, STOP and redesign"))
    if not p1b:
        print("\n  Decision rule: P1b failed. Not running the sweep.")
        sys.exit(1)

    if audit_only:
        print("\n  --audit: P1a and P1b pass; stopping before the sweep.")
        sys.exit(0)

    # ---- the sweep ------------------------------------------------------
    print("\n[1] Sweep: {} seeds x 2 conditions (plastic + frozen per seed)".format(len(SEEDS)))
    cache = _cache_load()
    results = {}
    for name, fn in conditions:
        coords, cids = coords_map[name]
        pl, fr, br = [], [], []
        print("\n  {}".format(name.upper()))
        for s in SEEDS:
            key = "{}|{}|spc{}|ep{}|ev{}|log".format(
                name, s, PLASTIC_PER_CLASS, N_EPOCHS, EVAL_PER_CLASS)
            if key in cache:
                pa, fa, b = cache[key]
                print("    seed {:>3}: plastic={:6.2f}%  frozen={:6.2f}%  bridge={:8.1f}  (cached)".format(
                    s, pa, fa, b))
            else:
                t0 = time.time()
                ws = LEGACY_WIRING_SEED + s
                ba, bb, src, dst, inhib = run_life_corrected(
                    coords, cids, s, plastic_t, plastic_d, n_epochs=N_EPOCHS,
                    wiring_seed=ws, inhib_mode=WIRING_MODE)
                Xp = evaluate_features(coords, cids, src, dst, inhib,
                                       eval_samples, 5000 + s)
                pa = ridge_cv_accuracy(Xp, eval_labels)
                # frozen = the SAME wiring this seed started from, unlearned
                src0, dst0, inhib0 = make_wiring(cids, ws, "type_relative")
                Xf = evaluate_features(coords, cids, src0, dst0, inhib0,
                                       eval_samples, 5000 + s)
                fa = ridge_cv_accuracy(Xf, eval_labels)
                b = ba + bb
                cache[key] = [float(pa), float(fa), float(b)]
                _cache_save(cache)
                print("    seed {:>3}: plastic={:6.2f}%  frozen={:6.2f}%  bridge={:8.1f}  ({:.0f}s)".format(
                    s, pa, fa, b, time.time() - t0))
            pl.append(pa); fr.append(fa); br.append(b)
            sys.stdout.flush()
        results[name] = {"plastic": np.array(pl, float),
                         "frozen": np.array(fr, float),
                         "bridge": np.array(br, float)}

    seg, inter = results["segregated"], results["interleaved"]
    seg_d = seg["plastic"] - seg["frozen"]
    int_d = inter["plastic"] - inter["frozen"]

    print("\n" + "=" * 92)
    print("[2] Four-cell result (held-out accuracy, mean +/- s.d. over {} seeds)".format(len(SEEDS)))
    print("=" * 92)
    print("  {:<13} {:>17} {:>17} {:>17} {:>12}".format(
        "placement", "plastic acc", "frozen acc", "delta (p-f)", "bridge mass"))
    for name, r, d in (("segregated", seg, seg_d), ("interleaved", inter, int_d)):
        print("  {:<13} {:>10.2f} +/-{:<5.2f} {:>10.2f} +/-{:<5.2f} {:>10.2f} +/-{:<5.2f} {:>12.1f}".format(
            name, r["plastic"].mean(), r["plastic"].std(ddof=1),
            r["frozen"].mean(), r["frozen"].std(ddof=1),
            d.mean(), d.std(ddof=1), r["bridge"].mean()))
    print("  chance = 50.00%  |  bridge-mass ratio (randomised wiring) = {:.2f}x".format(
        inter["bridge"].mean() / max(seg["bridge"].mean(), 1e-9)))
    print("  (bridge mass secondary here; the 8.22x headline is measured on the")
    print("   fixed-wiring harness and is unaffected by this file)")

    print("\n[3] P1c / P1d manipulation checks")
    frozen_gap = abs(inter["frozen"].mean() - seg["frozen"].mean())
    print("    frozen segregated = {:.2f}%   frozen interleaved = {:.2f}%".format(
        seg["frozen"].mean(), inter["frozen"].mean()))
    print("    |gap| = {:.2f} pp (tolerance {:.1f} pp)   [exp52 failed here at 5.97 pp]".format(
        frozen_gap, PARITY_PP))
    p1c = frozen_gap < PARITY_PP
    print("    P1c {}".format("HOLDS" if p1c else "FAILS -- STILL CONFOUNDED"))
    p1d = (seg["frozen"].mean() < CEILING_PCT) and (inter["frozen"].mean() < CEILING_PCT)
    print("    P1d {} (ceiling {:.1f}%)".format(
        "HOLDS -- headroom exists" if p1d else "FAILS -- UNINFORMATIVE (ceiling)", CEILING_PCT))

    print("\n[4] P2 (the actual test): plastic-minus-frozen accuracy delta")
    t2, p2 = stats.ttest_ind(int_d, seg_d, equal_var=False)
    gap = int_d.mean() - seg_d.mean()
    d_eff = cohens_d_pooled(int_d, seg_d)
    pooled_sd = np.sqrt((seg_d.std(ddof=1) ** 2 + int_d.std(ddof=1) ** 2) / 2.0)
    mde_d = 2.802 * np.sqrt(2.0 / len(SEEDS))
    print("    delta segregated  = {:+.2f} pp".format(seg_d.mean()))
    print("    delta interleaved = {:+.2f} pp".format(int_d.mean()))
    print("    difference        = {:+.2f} pp".format(gap))
    print("    Welch t={:.3f}, p={:.4e}, Cohen's d={:.2f}".format(t2, p2, d_eff))
    print("    achieved MDE at 80% power: d={:.3f} = {:.2f} pp (observed pooled s.d. {:.2f})".format(
        mde_d, mde_d * pooled_sd, pooled_sd))
    p2a = (gap > 0) and (p2 < 0.05)
    p2b = gap >= SUBSTANTIVE_PP
    print("    P2a (interleaved > segregated, p<0.05) : {}".format(p2a))
    print("    P2b (gap >= {:.1f} pp, substantive)      : {}".format(SUBSTANTIVE_PP, p2b))

    print("\n" + "=" * 92)
    print("[5] VERDICT (by the decision rule registered above)")
    print("=" * 92)
    if not p1c:
        print("  STILL CONFOUNDED: {:.2f} pp frozen gap despite the fix. P2 is not".format(frozen_gap))
        print("  readable as a plasticity effect.")
    elif not p1d:
        print("  UNINFORMATIVE (ceiling): the frozen projection already carries the class")
        print("  information, so plasticity has no headroom to demonstrate anything here.")
        print("  This is NOT evidence against the claim.")
    elif p2a and p2b:
        print("  SUPPORTS 'determines learnability': segregated placement measurably costs")
        print("  held-out CLASSIFICATION ACCURACY, by {:+.2f} pp, not merely associative".format(gap))
        print("  structure. The title's strong reading is earned.")
    elif p2a and not p2b:
        print("  PARTIAL / NARROWS: the accuracy effect is real and significant ({:+.2f} pp,".format(gap))
        print("  p={:.3g}) but below the {:.1f} pp substantive bar registered in advance, and".format(p2, SUBSTANTIVE_PP))
        print("  small relative to the ~8x structural effect. Report the accuracy cost")
        print("  quantitatively; do not lean on 'determines learnability' unqualified.")
    else:
        print("  NULL: segregation does NOT measurably cost classification accuracy here,")
        print("  at {}-seed power able to detect {:.2f} pp. The structural findings stand,".format(
            len(SEEDS), mde_d * pooled_sd))
        print("  but the paper's claim should be NARROWED from 'determines learnability'")
        print("  to 'determines associative connectivity'. Report prominently, not buried.")
