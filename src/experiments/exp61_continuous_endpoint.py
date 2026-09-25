"""Exp 61: was the accuracy endpoint the effect, or the instrument?

WHY THIS EXISTS. Exp 52b measured the performance endpoint for the first time
and found a 3.59x structural effect was 1.079x in accuracy terms: held-out
classification moved 6.94 pp (Welch p=0.012, d=0.575). That number is the
single weakest joint in the chain this project's claim runs along -- structure
-> performance -- and every later file (Exp 58/59/60) inherits it as a caveat.

But look at what produces it. `ridge_cv_accuracy` fits a ridge REGRESSION onto
+/-1 targets and then calls `np.sign()` on the prediction. The continuous
decision value -- how far from the boundary each sample landed -- is computed
and then discarded. What survives is 120 binary outcomes at 0.83 pp
resolution, so the entire 6.94 pp effect is about 8 samples crossing the
boundary. The other 112 samples carried information about the size of the
effect and it was thrown away.

So Exp 52b's result is ambiguous between two readings that have opposite
consequences for the project:

  (a) structural effects genuinely do not transfer to performance -- the
      honest, claim-limiting reading Exp 52b adopted; or
  (b) 2-class accuracy on 120 samples is too blunt to see the transfer, and
      the instrument, not the physics, produced the shrinkage.

Nothing in Exp 52b distinguishes them. This file does, by scoring the SAME
fitted readout continuously.

THE SECOND QUESTION, WHICH IS THE MORE INTERESTING ONE. `RIDGE_LAMBDA = 1.0`,
fixed, identical for both conditions. Fair, but it is one arbitrary point on
the axis that matters most to the thesis. Lambda controls READOUT CAPACITY,
and the claim is that placement determines what the NETWORK can learn. A
sufficiently powerful readout does the work the network did not, which is a
mechanistic reason a large structural gap should compress into a small
decoded gap. That makes a prediction nobody here has tested: the placement gap
should depend on readout strength, and be LARGEST where the readout is weakest
and the network must supply the separability itself.

If that holds, "our accuracy effect is small" stops being a weakness and
becomes a measured property of the decoder with a predicted direction. It is
the same dose-response logic as Exp 40, applied to the readout instead of the
placement.

THIS IS A NEW ENDPOINT PROPOSED AFTER SEEING A WEAK RESULT ON THE OLD ONE, AND
THAT IS SAID OUT LOUD. Exp 52b's accuracy result STANDS AS REPORTED whatever
this file finds. Nothing here replaces it, retracts it, or licenses quoting
the continuous number in its place. Switching endpoints until one cooperates
is the failure mode this project has caught itself in before (Exp 49, Exp 57
-> 58) and the only defence is to register the new endpoint separately, run it
on the same seeds and the same physics, and report both side by side forever.
The physics is untouched: identical `run_life_corrected`, identical wiring,
identical seeds, identical features. ONLY the readout metric and lambda move.

--------------------------------------------------------------------------
A DEFECT IN EXP 52b's FROZEN ARM, FOUND WHILE BUILDING THIS, FIXED HERE
--------------------------------------------------------------------------
Exp 52b's plastic arm calls `make_wiring(..., WIRING_MODE)` with
WIRING_MODE='logical' -- the P1b fix that makes both placements realise the
IDENTICAL logical network so they differ only in coordinates. Its frozen arm,
on the line commented "the SAME wiring this seed started from, unlearned",
calls `make_wiring(cids, ws, "type_relative")` instead. That is the SUPERSEDED
partial fix, and it does not have the parity P1b certifies:

    wiring mode      segregated             interleaved
    logical          io=414  exc=290        io=414  exc=290     <- exactly equal
    type_relative    io=414  exc=336        io=394  exc=321     <- NOT equal

So the frozen baseline is asymmetric in exactly the quantity P1b exists to
equalise, and the primary endpoint is plastic-MINUS-frozen, so it enters the
primary directly. P1b was run on 'logical' and passed; the frozen arm then ran
'type_relative'. This is precisely the documented caution from the 2026-08-27
sweep -- a check only certifies the configuration it actually runs -- and it
recurred despite being written down.

HOW BAD, MEASURED BEFORE WRITING ANY OF THIS: not bad. From Exp 52b's own
cache, frozen accuracy is 56.96% segregated vs 56.46% interleaved, a gap of
-0.50 pp, despite the structural asymmetry. And the plastic-only contrast,
which never touches the frozen arm at all, is +6.44 pp at Welch p=0.005 --
STRONGER than the +6.94 pp delta contrast it was subtracted into. Exp 52b's
conclusion survives its own defect on both counts. It is reported here because
an unreported near-miss is still an unreported defect, not because the number
needs correcting.

WHAT THIS FILE DOES ABOUT IT. Computes BOTH frozen arms from the same cached
features: `frozen_legacy` ('type_relative', reproduces Exp 52b bit-identically
and gates P0) and `frozen_logical` ('logical', the baseline Exp 52b intended).
The corrected arm is primary here. The cost is one extra `evaluate_features`
per seed per condition, and the comparison is reported as a standing audit.

--------------------------------------------------------------------------
DESIGN
--------------------------------------------------------------------------
Seeds, placements, wiring, data split, feature extraction and fold structure
are IMPORTED from Exp 52b / Exp 52 unchanged. 40 seeds (100-139) x 2
placements x 3 arms (plastic, frozen_logical, frozen_legacy).

Features -- per-output-neuron spike counts over the 120 held-out samples, 60
output neurons -- are extracted once and CACHED to disk. Everything after that
is arithmetic on the cache, so the lambda sweep and both endpoints cost
seconds, and re-analysis never re-simulates. That is the point of the file's
structure: the expensive thing happens once.

METRICS, all computed from the SAME fitted ridge weights on the SAME held-out
folds, pooled across folds:

  acc    percent correct after sign() -- Exp 52b's endpoint, reproduced exactly
  rsq    squared Pearson correlation between the continuous prediction and the
         +/-1 target.  PRIMARY.
  r2     1 - SS_res/SS_tot -- predictive R^2, penalises ridge shrinkage
  nrmse  sqrt(MSE / Var(target))

PRIMARY IS rsq, AND THE REASON IS REGISTERED BEFORE RUNNING. Ridge shrinks
predictions toward zero by an amount that grows with lambda, so r2 falls at
large lambda from CALIBRATION loss even when the readout's input carries
undiminished information. Across a lambda sweep that would confound the
mechanism question with an artifact of the estimator. rsq is scale-invariant
and immune to it. It is also the reservoir-computing standard -- Jaeger's
memory capacity is a sum of exactly this quantity -- which matters if the
capacity measurement ever gets built on top of this harness. r2 and nrmse are
reported alongside at every lambda so the shrinkage is visible, not hidden.

LAMBDA GRID: 0.01 to 1000 in half-decade steps, 11 points, including 1.0
exactly. Features are standardised on the train fold, so lambda is on a
standardised scale and comparable across conditions.

--------------------------------------------------------------------------
PRE-REGISTERED PREDICTIONS (fixed in this docstring before the first run)
--------------------------------------------------------------------------
P0  GATE, not a prediction. `ridge_cv_scores(X, y, lam=1.0)['acc']` must equal
    the imported `ridge_cv_accuracy(X, y)` bit-identically on the same X, and
    the plastic / frozen_legacy arms must reproduce Exp 52b's CACHED per-seed
    accuracies bit-identically. If either fails, the readout diverged from the
    published one and NOTHING in this file may be interpreted. Hard stop.

P1  MANIPULATION CHECK. The continuous endpoint must have no quantisation
    floor. Registered as: at least 99% of the seed-condition cells take a
    DISTINCT rsq value, and rsq takes strictly more distinct values than
    accuracy does on the identical cells. Accuracy can only occupy 121 values
    (0.83 pp spacing on 120 samples) and ties heavily; rsq should tie almost
    never. Trivially expected -- it is here so that a failure, which would
    mean something is badly wrong with the readout, cannot pass unnoticed.

    SPECIFICATION FIX, DISCLOSED: this check was first written as "more than
    100 distinct values", an absolute count. That is unreachable on a reduced
    seed count and does not actually test the claim -- 100 distinct values out
    of 240 cells would be a severe tie problem and would still pass. Corrected
    to the relative form above BEFORE the registered 40-seed run, on the
    2-seed and 6-seed pilots only. No hypothesis endpoint was touched.

P2  PRIMARY. Cohen's d for the continuous endpoint at lambda=1.0 exceeds Exp
    52b's accuracy d of 0.575, with a paired-bootstrap CI on the difference
    that excludes zero.
      d_cont > d_acc, CI excludes 0  -> READING (b). Accuracy was the limiting
          instrument. The structure->performance transfer is larger than Exp
          52b could resolve, and Exp 52b's caveat is about the metric.
      CI includes 0                  -> INCONCLUSIVE on power. The effect is
          about that size and dichotomisation was not the bottleneck.
      d_cont < d_acc, CI excludes 0  -> READING (a), HARDENED. Accuracy was
          flattering; the transfer gap is real and worse than reported. This
          cuts against the project and is reported first if it happens.

P3  SECONDARY, the mechanism. Within the lambda band where the readout is
    functioning -- defined as both conditions having mean plastic r2 > 0, a
    criterion blind to placement -- Spearman rho between log10(lambda) and
    Cohen's d is >= +0.50. Stronger regularisation = weaker readout = the
    network must supply separability = larger placement gap. Falsified if
    rho < +0.50, flat included.

    A POSITIVE RHO DOES NOT ESTABLISH THE MECHANISM IT WAS ARGUED FROM, and
    the file says so itself. The gap can grow because segregated degrades as
    the readout weakens (the account above) or because interleaved improves
    as regularisation cuts variance -- which would be a fact about the
    estimator, not about the network supplying separability. `decompose_p3`
    prints which one moved, unconditionally, so a passing P3 can never be
    narrated as the former without someone having looked.

    NON-MONOTONICITY IS EXPECTED OUTSIDE THAT BAND AND IS NOT A FINDING. At
    lambda -> 0 the readout has 61 parameters and 96 training samples per fold
    and overfits, so held-out performance collapses for BOTH conditions; at
    lambda -> inf everything shrinks to the intercept and both go to zero. The
    band restriction is registered here, in advance, precisely so an inverted-U
    cannot be discovered afterwards and narrated as a result. This project
    already owns one unexplained inverted-U (Exp 35); it does not need a second
    one arrived at by surprise.

P4  ROBUSTNESS, reported whatever it says. The primary re-run against
    `frozen_legacy` instead of `frozen_logical`, and against the plastic-only
    contrast which uses no frozen baseline at all. If the three disagree, the
    frozen baseline is load-bearing and that is the headline instead.

--------------------------------------------------------------------------
SCOPE -- WHAT THIS STILL CANNOT SAY
--------------------------------------------------------------------------
One task (2-class SHD), one dataset, one readout family (linear ridge on
spike counts), one network size. A linear readout measures LINEAR separability
of the output state; if segregated placement encodes the same information
nonlinearly, every metric here understates it equally and the comparison is
still fair but the absolute numbers are a lower bound. Nothing here measures
memory capacity or nonlinear processing capacity, which is the obvious next
instrument and is deliberately out of scope.

Run:              python src/experiments/exp61_continuous_endpoint.py
Run (audit, ~2 seeds):
                  python src/experiments/exp61_continuous_endpoint.py --audit
Run (analysis only, no simulation):
                  python src/experiments/exp61_continuous_endpoint.py --from-features
Run (quick):      python src/experiments/exp61_continuous_endpoint.py --seeds 6
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import argparse
import json
import os
import time

import numpy as np
from scipy import stats

from exp37b_v2_real_data import (
    make_placement_segregated, make_placement_interleaved,
)
from exp37c_real_data_scaled import cohens_d_pooled
from exp52_accuracy_endpoint import (
    evaluate_features, ridge_cv_accuracy, load_split,
    PLASTIC_PER_CLASS, EVAL_PER_CLASS, N_EPOCHS, RIDGE_LAMBDA, N_FOLDS,
)
from exp52b_accuracy_corrected import (
    run_life_corrected, make_wiring, excitatory_io_stats,
    SEEDS, LEGACY_WIRING_SEED, WIRING_MODE,
)

# ---- registered configuration (fixed before running) ----
LAMBDA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
REGISTERED_LAMBDA = 1.0           # the value exp52b registered; must be in grid
EXP52B_ACC_D = 0.575              # d(delta) recomputed from exp52b's own cache
RHO_THRESHOLD = 0.50              # P3
DISTINCT_FRAC = 0.99              # P1: fraction of cells with distinct rsq
ARMS = ("plastic", "frozen_logical", "frozen_legacy")
PRIMARY_METRIC = "rsq"
PRIMARY_FROZEN = "frozen_logical"

FEAT_PATH = os.path.join("shd_data", "exp61_features.npz")
CACHE_PATH = os.path.join("shd_data", "exp61_cache.json")
EXP52B_CACHE = os.path.join("shd_data", "exp52b_cache.json")

CONDITIONS = [("segregated", make_placement_segregated),
              ("interleaved", make_placement_interleaved)]


# ------------------------------------------------------------------ readout
def ridge_cv_scores(X, y, lam=RIDGE_LAMBDA, n_folds=N_FOLDS, seed=999):
    """`ridge_cv_accuracy` with the continuous prediction KEPT.

    The fold construction, standardisation, intercept handling and linear
    solve are copied verbatim from exp52's `ridge_cv_accuracy` so the same
    weights are fitted on the same folds; `acc` is asserted bit-identical to
    it in P0. Everything else is read off the decision values that function
    computes and discards.
    """
    X = np.asarray(X, float); y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        folds[idx] = np.arange(len(idx)) % n_folds
    ytr_sign = np.where(y == y.min(), -1.0, 1.0)

    pred_all = np.zeros(len(y), dtype=float)
    correct = 0
    for k in range(n_folds):
        te = folds == k; tr = ~te
        Xtr, Xte = X[tr], X[te]
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
        Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
        Zte = np.hstack([Zte, np.ones((len(Zte), 1))])
        t_ = ytr_sign[tr]
        A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
        A[-1, -1] -= lam                      # do not penalise intercept
        w = np.linalg.solve(A, Ztr.T @ t_)
        raw = Zte @ w
        pred_all[te] = raw
        pred = np.sign(raw)
        pred[pred == 0] = 1.0
        correct += int((pred == ytr_sign[te]).sum())

    t = ytr_sign
    ss_res = float(np.sum((pred_all - t) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    denom = max(ss_tot, 1e-12)
    sd_p = pred_all.std()
    rsq = 0.0 if sd_p < 1e-12 else float(np.corrcoef(pred_all, t)[0, 1] ** 2)
    return {"acc": 100.0 * correct / len(y),
            "rsq": rsq,
            "r2": 1.0 - ss_res / denom,
            "nrmse": float(np.sqrt(ss_res / denom))}


# ------------------------------------------------------------- feature cache
def _feat_load():
    if not os.path.exists(FEAT_PATH):
        return {}
    with np.load(FEAT_PATH) as z:
        return {k: z[k] for k in z.files}


def _feat_save(store):
    tmp = FEAT_PATH + ".tmp.npz"
    np.savez_compressed(tmp, **store)
    os.replace(tmp, FEAT_PATH)


def feat_key(cond, seed, arm):
    return "{}|{}|{}|spc{}|ep{}|ev{}".format(
        cond, seed, arm, PLASTIC_PER_CLASS, N_EPOCHS, EVAL_PER_CLASS)


def extract_seed(coords, cids, cond, seed, plastic_t, plastic_d, eval_samples,
                 store):
    """Simulate one (condition, seed) and cache all three arms' features.

    plastic         -- network after the plastic phase, WIRING_MODE wiring
    frozen_logical  -- the same wiring this seed started from, unlearned
                       (mode='logical' -- what exp52b's comment describes)
    frozen_legacy   -- mode='type_relative' -- what exp52b's code actually
                       ran; kept so P0 can reproduce its numbers exactly
    """
    want = [a for a in ARMS if feat_key(cond, seed, a) not in store]
    if not want:
        return False
    ws = LEGACY_WIRING_SEED + seed
    if "plastic" in want:
        _, _, src, dst, inhib = run_life_corrected(
            coords, cids, seed, plastic_t, plastic_d, n_epochs=N_EPOCHS,
            wiring_seed=ws, inhib_mode=WIRING_MODE)
        store[feat_key(cond, seed, "plastic")] = evaluate_features(
            coords, cids, src, dst, inhib, eval_samples, 5000 + seed)
    for arm, mode in (("frozen_logical", "logical"),
                      ("frozen_legacy", "type_relative")):
        if arm in want:
            s0, d0, i0 = make_wiring(cids, ws, mode)
            store[feat_key(cond, seed, arm)] = evaluate_features(
                coords, cids, s0, d0, i0, eval_samples, 5000 + seed)
    return True


# ----------------------------------------------------------------- analysis
def score_all(store, seeds, labels):
    """scores[cond][arm][lam] -> array over seeds of the metric dict."""
    out = {}
    for cond, _ in CONDITIONS:
        out[cond] = {}
        for arm in ARMS:
            out[cond][arm] = {}
            for lam in LAMBDA_GRID:
                rows = []
                for s in seeds:
                    X = store[feat_key(cond, s, arm)]
                    rows.append(ridge_cv_scores(X, labels, lam=lam))
                out[cond][arm][lam] = rows
    return out


def deltas(scores, metric, lam, frozen_arm):
    """Per-seed plastic-minus-frozen for one metric at one lambda."""
    d = {}
    for cond, _ in CONDITIONS:
        p = np.array([r[metric] for r in scores[cond]["plastic"][lam]], float)
        f = np.array([r[metric] for r in scores[cond][frozen_arm][lam]], float)
        d[cond] = p - f
    return d["interleaved"], d["segregated"]


def d_with_ci(a, b, n_boot=10000, seed=4242):
    """Cohen's d for (a - b) with a PAIRED bootstrap over seed index.

    Seeds are matched across conditions -- the same seed value drives both,
    with the same wiring seed -- so resampling the index jointly preserves
    that structure. The point estimate uses `cohens_d_pooled`, the same
    function exp52b used, so the numbers are directly comparable.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = cohens_d_pooled(a[idx], b[idx])
    return (float(cohens_d_pooled(a, b)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def d_difference_ci(a1, b1, a0, b0, n_boot=10000, seed=4242):
    """CI on d(continuous) - d(accuracy), resampling seeds jointly so both
    metrics are recomputed on the SAME resampled seeds each draw."""
    a1, b1, a0, b0 = (np.asarray(x, float) for x in (a1, b1, a0, b0))
    rng = np.random.default_rng(seed)
    n = len(a1)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = cohens_d_pooled(a1[idx], b1[idx]) - cohens_d_pooled(a0[idx], b0[idx])
    point = cohens_d_pooled(a1, b1) - cohens_d_pooled(a0, b0)
    return (float(point),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# -------------------------------------------------------------------- gates
def p0_readout_identity(labels):
    """The readout must not have diverged from the published one."""
    print("\n  [P0a] ridge_cv_scores(lam=1.0)['acc'] == imported ridge_cv_accuracy")
    rng = np.random.default_rng(0)
    ok = True
    for trial in range(3):
        X = rng.normal(size=(len(labels), 60)) + labels[:, None] * 0.3
        a_ref = ridge_cv_accuracy(X, labels)
        a_new = ridge_cv_scores(X, labels, lam=RIDGE_LAMBDA)["acc"]
        same = a_ref == a_new
        ok = ok and same
        print("    trial {}: imported {:.10f}   this file {:.10f}   match: {}".format(
            trial, a_ref, a_new, same))
    print("    P0a {}".format("HOLDS (bit-identical)" if ok
                              else "FAILS -- readout diverged, STOP"))
    return ok


def p0_exp52b_reproduction(store, seeds, labels):
    """plastic / frozen_legacy at lambda=1.0 must reproduce exp52b's cache."""
    print("\n  [P0b] exp52b cached per-seed accuracies reproduced bit-identically")
    if not os.path.exists(EXP52B_CACHE):
        print("    exp52b cache missing -- cannot gate. STOP.")
        return False
    prior = json.load(open(EXP52B_CACHE))
    ok, checked = True, 0
    for cond, _ in CONDITIONS:
        for s in seeds:
            key52 = "{}|{}|spc{}|ep{}|ev{}|log".format(
                cond, s, PLASTIC_PER_CLASS, N_EPOCHS, EVAL_PER_CLASS)
            if key52 not in prior:
                continue
            if feat_key(cond, s, "plastic") not in store:
                continue
            pa_ref, fa_ref = prior[key52][0], prior[key52][1]
            pa = ridge_cv_scores(store[feat_key(cond, s, "plastic")],
                                 labels, lam=RIDGE_LAMBDA)["acc"]
            fa = ridge_cv_scores(store[feat_key(cond, s, "frozen_legacy")],
                                 labels, lam=RIDGE_LAMBDA)["acc"]
            same = (pa == pa_ref) and (fa == fa_ref)
            ok = ok and same
            checked += 1
            if not same or checked <= 4:
                print("    {:<12} seed {:>3}: plastic {:.6f} vs {:.6f} | "
                      "frozen {:.6f} vs {:.6f} | match: {}".format(
                          cond, s, pa, pa_ref, fa, fa_ref, same))
    if checked == 0:
        print("    no overlapping seeds cached yet -- gate not yet evaluable.")
        return None
    print("    {} seed-conditions checked".format(checked))
    print("    P0b {}".format(
        "HOLDS -- physics and readout match the published run" if ok
        else "FAILS -- this file does NOT reproduce exp52b. STOP."))
    return ok


def frozen_arm_audit(coords_map, seeds):
    """Standing report of the exp52b frozen-arm defect (see docstring)."""
    print("\n  [AUDIT] wiring parity by mode -- exp52b's plastic arm uses "
          "'{}', its frozen arm used 'type_relative'".format(WIRING_MODE))
    rows = {}
    for mode in ("logical", "type_relative"):
        for cond, _ in CONDITIONS:
            coords, cids = coords_map[cond]
            ios, excs = [], []
            for s in seeds:
                src, dst, inhib = make_wiring(cids, LEGACY_WIRING_SEED + s, mode)
                io, exc, _ = excitatory_io_stats(cids, src, dst, inhib)
                ios.append(io); excs.append(exc)
            rows[(mode, cond)] = (np.array(ios), np.array(excs))
    print("    {:<15} {:>12} {:>12} {:>14}".format(
        "mode", "seg exc_io", "int exc_io", "per-seed equal"))
    for mode in ("logical", "type_relative"):
        se, ie = rows[(mode, "segregated")][1], rows[(mode, "interleaved")][1]
        so, io_ = rows[(mode, "segregated")][0], rows[(mode, "interleaved")][0]
        eq = bool(np.all(se == ie) and np.all(so == io_))
        print("    {:<15} {:>12.1f} {:>12.1f} {:>14}".format(
            mode, se.mean(), ie.mean(), "YES" if eq else "NO  <- defect"))
    print("    frozen_logical is primary here; frozen_legacy retained for P0/P4.")


def decompose_p3(sweep, band):
    """WHICH CONDITION MOVES? A rising gap is consistent with two different
    mechanisms and the registered rho cannot tell them apart:

      (i)  segregated DEGRADES as the readout weakens -- the mechanism the P3
           prediction was argued from (a weak readout cannot substitute for
           separability the network failed to supply); or
      (ii) interleaved IMPROVES -- regularisation reducing variance, which
           helps the signal-rich condition more and says nothing about the
           network having to do the work.

    Both produce a positive rho. Printed unconditionally so a passing P3 can
    never be narrated as (i) without someone having checked."""
    lo, hi = band[0], band[-1]
    ds = sweep[hi]["seg"] - sweep[lo]["seg"]
    di = sweep[hi]["int"] - sweep[lo]["int"]
    tot = sweep[hi]["gap"] - sweep[lo]["gap"]
    print("\n    DECOMPOSITION -- which condition actually moves across the band?")
    print("      segregated  {:+.4f} -> {:+.4f}   change {:+.4f} ({:+.0f}% of gap growth)"
          .format(sweep[lo]["seg"], sweep[hi]["seg"], ds,
                  100.0 * (-ds) / tot if abs(tot) > 1e-12 else 0.0))
    print("      interleaved {:+.4f} -> {:+.4f}   change {:+.4f} ({:+.0f}% of gap growth)"
          .format(sweep[lo]["int"], sweep[hi]["int"], di,
                  100.0 * di / tot if abs(tot) > 1e-12 else 0.0))
    if abs(di) > 2.0 * abs(ds):
        print("      -> DRIVEN BY INTERLEAVED IMPROVING, not by segregated degrading.")
        print("         The registered DIRECTION holds, but mechanism (i) -- the")
        print("         argument P3 was motivated by -- is NOT what the data show.")
        print("         Do not state P3 as 'a weak readout exposes the placement")
        print("         penalty'. State it as measured: the decoded gap depends on")
        print("         readout strength, because regularisation benefits the")
        print("         signal-rich placement more. Separating the two needs a")
        print("         readout-capacity knob that is not also a variance knob.")
    elif abs(ds) > 2.0 * abs(di):
        print("      -> DRIVEN BY SEGREGATED DEGRADING. Consistent with mechanism (i),")
        print("         the account P3 was argued from.")
    else:
        print("      -> BOTH MOVE COMPARABLY. Mechanism undetermined; report the")
        print("         direction only.")


# --------------------------------------------------------------------- main
def main(args):
    t0 = time.time()
    print("=" * 92)
    print("EXP 61: was the accuracy endpoint the effect, or the instrument?")
    print("        continuous readout + readout-capacity sweep (predictions "
          "fixed in the docstring)")
    print("=" * 92)

    top, avail, plastic, evalset = load_split()
    plastic_t = plastic[top[0]]; plastic_d = plastic[top[1]]
    eval_samples = evalset[top[0]] + evalset[top[1]]
    labels = np.array([0] * len(evalset[top[0]]) + [1] * len(evalset[top[1]]))
    coords_map = {name: fn() for name, fn in CONDITIONS}
    seeds = SEEDS[:args.seeds] if args.seeds else SEEDS
    if args.audit:
        seeds = seeds[:2]

    print("\n[0] Configuration")
    print("  seeds {}-{} ({})  |  eval {} samples  |  60 output neurons".format(
        seeds[0], seeds[-1], len(seeds), len(eval_samples)))
    print("  accuracy resolution {:.2f} pp ({} possible values); the continuous "
          "endpoint has none".format(100.0 / len(eval_samples), len(eval_samples) + 1))
    print("  lambda grid: {}".format(", ".join("{:g}".format(x) for x in LAMBDA_GRID)))
    print("  primary metric '{}' vs frozen arm '{}'  |  exp52b accuracy d = {:.3f}"
          .format(PRIMARY_METRIC, PRIMARY_FROZEN, EXP52B_ACC_D))
    assert REGISTERED_LAMBDA in LAMBDA_GRID

    if not p0_readout_identity(labels):
        sys.exit(1)
    frozen_arm_audit(coords_map, seeds)

    # ---- features -----------------------------------------------------
    store = _feat_load()
    if args.from_features:
        missing = [feat_key(c, s, a) for c, _ in CONDITIONS
                   for s in seeds for a in ARMS if feat_key(c, s, a) not in store]
        if missing:
            print("\n  --from-features: {} arms missing from {}. Run without "
                  "the flag first.".format(len(missing), FEAT_PATH))
            sys.exit(1)
        print("\n[1] --from-features: {} cached arms, no simulation".format(
            len(seeds) * len(CONDITIONS) * len(ARMS)))
    else:
        print("\n[1] Feature extraction ({} seeds x {} conditions x {} arms)".format(
            len(seeds), len(CONDITIONS), len(ARMS)))
        for cond, _ in CONDITIONS:
            coords, cids = coords_map[cond]
            print("\n  {}".format(cond.upper()))
            for s in seeds:
                ts = time.time()
                did = extract_seed(coords, cids, cond, s, plastic_t, plastic_d,
                                   eval_samples, store)
                if did:
                    _feat_save(store)
                print("    seed {:>3}: {}  [{:.0f}s total]".format(
                    s, "simulated + cached" if did else "cached", time.time() - t0),
                    flush=True)

    gate = p0_exp52b_reproduction(store, seeds, labels)
    if gate is False:
        sys.exit(1)

    if args.audit:
        print("\n  --audit: P0a{} pass; stopping before the analysis.".format(
            " and P0b" if gate else ""))
        sys.exit(0 if gate is not False else 1)

    # ---- scoring ------------------------------------------------------
    print("\n[2] Scoring {} lambda values x {} arms (from cached features)".format(
        len(LAMBDA_GRID), len(ARMS)))
    scores = score_all(store, seeds, labels)

    print("\n  P1 manipulation check -- resolution of the continuous endpoint")
    cells = [(r[PRIMARY_METRIC], r["acc"]) for c, _ in CONDITIONS for a in ARMS
             for r in scores[c][a][REGISTERED_LAMBDA]]
    n_cells = len(cells)
    n_cont = len(set(np.round([c[0] for c in cells], 12)))
    n_acc = len(set(np.round([c[1] for c in cells], 12)))
    frac = n_cont / max(n_cells, 1)
    p1 = (frac >= DISTINCT_FRAC) and (n_cont > n_acc)
    print("    {:<5} {:>4}/{} cells distinct ({:.1%})".format(
        PRIMARY_METRIC, n_cont, n_cells, frac))
    print("    {:<5} {:>4}/{} cells distinct ({:.1%})  -- quantised to {} possible "
          "values".format("acc", n_acc, n_cells, n_acc / max(n_cells, 1),
                          len(labels) + 1))
    print("    P1 {} (threshold: >={:.0%} distinct AND more than accuracy)".format(
        "HOLDS" if p1 else "FAILS -- investigate the readout", DISTINCT_FRAC))

    # ---- P2: primary --------------------------------------------------
    print("\n[3] P2 PRIMARY -- continuous endpoint at lambda={:g}".format(
        REGISTERED_LAMBDA))
    ic, sc = deltas(scores, PRIMARY_METRIC, REGISTERED_LAMBDA, PRIMARY_FROZEN)
    ia, sa = deltas(scores, "acc", REGISTERED_LAMBDA, "frozen_legacy")
    d_cont, dc_lo, dc_hi = d_with_ci(ic, sc)
    d_acc, da_lo, da_hi = d_with_ci(ia, sa)
    diff, df_lo, df_hi = d_difference_ci(ic, sc, ia, sa)
    t_c, p_c = stats.ttest_ind(ic, sc, equal_var=False)
    print("    delta {} : segregated {:+.4f} +/-{:.4f}   interleaved {:+.4f} +/-{:.4f}"
          .format(PRIMARY_METRIC, sc.mean(), sc.std(ddof=1), ic.mean(), ic.std(ddof=1)))
    print("    Welch p = {:.4g}   gap = {:+.4f}".format(p_c, ic.mean() - sc.mean()))
    print("    d(continuous) = {:.3f}  95% CI [{:.3f}, {:.3f}]".format(d_cont, dc_lo, dc_hi))
    print("    d(accuracy)   = {:.3f}  95% CI [{:.3f}, {:.3f}]   "
          "[exp52b recorded {:.3f}]".format(d_acc, da_lo, da_hi, EXP52B_ACC_D))
    print("    difference    = {:+.3f}  95% CI [{:+.3f}, {:+.3f}]  (paired bootstrap)"
          .format(diff, df_lo, df_hi))
    if df_lo > 0:
        p2 = ("READING (b). The continuous endpoint resolves a LARGER effect than "
              "accuracy on identical data, CI excludes zero. Exp 52b's shrinkage "
              "was substantially the instrument: 2-class accuracy at 0.83 pp "
              "resolution could not see the transfer. Exp 52b's accuracy number "
              "still stands as reported -- this does not replace it.")
    elif df_hi < 0:
        p2 = ("READING (a), HARDENED, AND THIS CUTS AGAINST THE PROJECT. The "
              "continuous endpoint resolves a SMALLER effect than accuracy did. "
              "Accuracy was flattering; the structure->performance gap is real and "
              "worse than Exp 52b reported. Report this first and loudest.")
    else:
        p2 = ("INCONCLUSIVE ON POWER. The two endpoints do not differ "
              "detectably in effect size, so dichotomisation was not the "
              "bottleneck and Exp 52b's reading (a) is unchallenged. The "
              "registered prediction is a MISS and is reported as one.")
    print("    " + p2)

    # ---- P3: lambda sweep ---------------------------------------------
    print("\n[4] P3 SECONDARY -- readout-capacity sweep")
    print("    {:>8} {:>9} {:>9} {:>10} {:>10} {:>8} {:>8}".format(
        "lambda", "seg d" + PRIMARY_METRIC, "int d" + PRIMARY_METRIC,
        "gap", "cohen d", "seg r2", "int r2"))
    sweep, band = {}, []
    for lam in LAMBDA_GRID:
        i_, s_ = deltas(scores, PRIMARY_METRIC, lam, PRIMARY_FROZEN)
        dl = cohens_d_pooled(i_, s_)
        r2s = np.mean([r["r2"] for r in scores["segregated"]["plastic"][lam]])
        r2i = np.mean([r["r2"] for r in scores["interleaved"]["plastic"][lam]])
        in_band = (r2s > 0) and (r2i > 0)
        if in_band:
            band.append(lam)
        sweep[lam] = {"gap": float(i_.mean() - s_.mean()), "d": float(dl),
                      "seg": float(s_.mean()), "int": float(i_.mean()),
                      "r2_seg": float(r2s), "r2_int": float(r2i),
                      "in_band": bool(in_band)}
        print("    {:>8g} {:>9.4f} {:>9.4f} {:>10.4f} {:>10.3f} {:>8.3f} {:>8.3f}{}"
              .format(lam, s_.mean(), i_.mean(), i_.mean() - s_.mean(), dl,
                      r2s, r2i, "" if in_band else "   (out of band)"))

    if len(band) >= 3:
        xs = np.log10(band)
        ys = np.array([sweep[l]["d"] for l in band], float)
        rho, p_rho = stats.spearmanr(xs, ys)
        p3 = rho >= RHO_THRESHOLD
        print("\n    band = lambda in [{:g}, {:g}] ({} points, both r2 > 0)".format(
            band[0], band[-1], len(band)))
        print("    Spearman rho(log10 lambda, d) = {:+.3f}  (p={:.3g}, threshold "
              "+{:.2f})".format(rho, p_rho, RHO_THRESHOLD))
        print("    P3 {}".format(
            "HOLDS -- the placement gap grows as the readout weakens. The decoded "
            "effect size is a property of readout capacity with the predicted "
            "direction, not a fixed fact about the physics." if p3 else
            "FAILS -- the placement gap does not track readout strength. The "
            "readout-capacity account of exp52b's shrinkage is REFUTED and must "
            "not be used to explain it."))
        decompose_p3(sweep, band)
    else:
        rho, p_rho, p3 = float("nan"), float("nan"), False
        print("\n    band has {} points (<3) -- P3 not evaluable. Report as "
              "UNTESTED, not as a miss.".format(len(band)))

    # ---- P4: robustness -----------------------------------------------
    print("\n[5] P4 ROBUSTNESS -- primary under three baselines")
    robust = {}
    for lbl, arm in (("frozen_logical (primary)", "frozen_logical"),
                     ("frozen_legacy  (exp52b)", "frozen_legacy")):
        i_, s_ = deltas(scores, PRIMARY_METRIC, REGISTERED_LAMBDA, arm)
        dd, lo, hi = d_with_ci(i_, s_)
        robust[arm] = {"d": dd, "ci": [lo, hi],
                       "p": float(stats.ttest_ind(i_, s_, equal_var=False)[1])}
        print("    {:<26} d = {:.3f}  95% CI [{:.3f}, {:.3f}]  Welch p = {:.4g}"
              .format(lbl, dd, lo, hi, robust[arm]["p"]))
    ip = np.array([r[PRIMARY_METRIC] for r in
                   scores["interleaved"]["plastic"][REGISTERED_LAMBDA]], float)
    sp = np.array([r[PRIMARY_METRIC] for r in
                   scores["segregated"]["plastic"][REGISTERED_LAMBDA]], float)
    dd, lo, hi = d_with_ci(ip, sp)
    robust["plastic_only"] = {"d": dd, "ci": [lo, hi],
                              "p": float(stats.ttest_ind(ip, sp, equal_var=False)[1])}
    print("    {:<26} d = {:.3f}  95% CI [{:.3f}, {:.3f}]  Welch p = {:.4g}"
          .format("plastic only (no frozen)", dd, lo, hi, robust["plastic_only"]["p"]))
    ds = [robust[k]["d"] for k in robust]
    agree = (min(ds) > 0) == (max(ds) > 0) and (max(ds) - min(ds)) < 0.5
    print("    {}".format(
        "All three agree in sign and within 0.5 d -- the frozen baseline is not "
        "load-bearing." if agree else
        "THE THREE BASELINES DISAGREE. The frozen baseline IS load-bearing and "
        "that is the headline, not the primary above."))

    with open(CACHE_PATH, "w") as fh:
        json.dump({"seeds": list(seeds),
                   "primary": {"metric": PRIMARY_METRIC, "lambda": REGISTERED_LAMBDA,
                               "frozen_arm": PRIMARY_FROZEN,
                               "d_cont": d_cont, "d_cont_ci": [dc_lo, dc_hi],
                               "d_acc": d_acc, "d_acc_ci": [da_lo, da_hi],
                               "d_diff": diff, "d_diff_ci": [df_lo, df_hi],
                               "welch_p": float(p_c), "verdict": p2},
                   "sweep": {str(k): v for k, v in sweep.items()},
                   "band": band, "rho": float(rho), "rho_p": float(p_rho),
                   "p1": bool(p1), "p1_distinct": [n_cont, n_acc, n_cells],
                   "p3": bool(p3),
                   "robustness": robust}, fh, indent=1)
    print("\n  cached -> {}  ({:.0f}s)".format(CACHE_PATH, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true",
                    help="P0 gates on 2 seeds, then stop")
    ap.add_argument("--from-features", action="store_true",
                    help="analysis only, from the cached feature matrices")
    ap.add_argument("--seeds", type=int, default=0,
                    help="use only the first N registered seeds (0 = all 40)")
    main(ap.parse_args())
