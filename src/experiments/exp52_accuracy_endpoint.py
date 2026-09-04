"""Exp 52: Does segregated placement cost CLASSIFICATION ACCURACY?

MOTIVATION -- the gap this closes. Every endpoint in this project is
STRUCTURAL: taught mass (Sec 4.1), bridge mass (Sec 4.5/4.13), cross-type
edge fraction (Sec 4.12). The preprint is titled "Placement Determines
LEARNABILITY", but nothing in it measures whether a network can actually
CLASSIFY better under one placement than another. The step from "less
associative structure forms" to "the network can't learn" is currently an
INFERENCE supported by a mediation argument (Sec 4.7), not a measurement.
A reviewer will go straight at that step. This experiment measures it.

A NULL RESULT IS A VALID AND VALUABLE OUTCOME, and is registered as such
below. If segregation does not cost accuracy, the honest conclusion is
that the paper's claim should be NARROWED from "learnability" to
"associative connectivity" -- still true, still publishable, and far
better discovered here than by a referee. No parameter in this file was
changed after seeing any result; if that ever becomes false it must be
stated explicitly in this docstring.

WHAT IS KEPT EXACTLY. Placements (make_placement_segregated /
make_placement_interleaved) and the physics are taken from
exp37b_v2_real_data. The plastic phase is run by `run_life_with_state`,
which is run_life's loop with ONE addition: it also returns the final
wiring (src, dst, inhib) so the trained network can be evaluated. Because
a hand-copied loop is exactly how this project has injected bugs before
(PROJECT_HISTORY: unreachable rewire block, reward-timing, decay
mismatch), section [0] ASSERTS that run_life_with_state reproduces the
imported run_life's bridge values BIT-IDENTICALLY on the same inputs.
If that assert fires, no number below is trustworthy.

DESIGN.

  Data split (SHD, two most-common classes, 89 and 87 samples available):
    PLASTIC set  20/class -- the ONLY samples structural plasticity sees.
    EVAL set     60/class -- 120 samples, never seen during plasticity.
  The plastic config is exp37c's MATCHED-LIFE headline verbatim (20
  samples/class x 1 epoch, ~8,600 steps, V retention 0.651), so the
  network being evaluated is the same network that produces the 7.25x
  bridge-mass result. Accuracy resolution is 1/120 = 0.83 percentage
  points.

  Four cells: {segregated, interleaved} x {plastic, frozen}.

  FROZEN is exact, not approximate. In this harness synaptic WEIGHTS never
  change -- they are fixed at +0.30/-0.60 by `inhib`. The only thing
  plasticity alters is the wiring (src/dst) via the rewire block. So a
  network with plasticity disabled is EXACTLY the initial random wiring,
  and can be evaluated directly without simulating the training phase.
  This is an equivalence, not a shortcut. Note also that src/dst/inhib
  derive from a FIXED rng (default_rng(7)) independent of seed, so the
  frozen network is identical across seeds; its only seed-to-seed
  variation is evaluation noise, and near-zero frozen variance is
  expected rather than suspicious.

  The primary quantity is the PLASTIC MINUS FROZEN accuracy delta per
  placement -- what plasticity bought -- mirroring how this project
  already isolates growth from a shared non-mechanistic baseline
  (Sec 4.1 metric correction).

READOUT, REGISTERED IN ADVANCE. This choice decides the experiment, so it
is fixed here with its rationale before any run.

  WHAT IS READ: per-neuron spike counts of the 60 OUTPUT-type neurons
  (30 OUTPUT_A then 30 OUTPUT_B, index-sorted) accumulated over one
  sample presentation. Input-type neurons are EXCLUDED.
    Why exclude inputs: input neurons are driven directly by the stimulus,
    so a decoder reading them classifies from the raw stimulus regardless
    of any learned structure, which would swamp the comparison and make
    the experiment uninformative about placement. Reading only outputs
    asks whether class information actually REACHED the output layer
    through learned connectivity.
    Why not read only the bridge pairs: a readout restricted to the
    specific input->A / input->B pairs the reward targets would bake the
    answer in. Reading ALL output neurons, with the decoder free to
    weight them, is the version defensible to a hostile reviewer.

  DECODER: ridge (L2) linear classifier on z-scored features, closed
  form, lambda = 1.0 FIXED. Chosen over logistic regression because it is
  deterministic, has no optimiser, and has exactly one hyperparameter
  that is fixed identically for both conditions. Standardisation
  statistics and ridge fit are computed on the decoder-TRAIN fold only
  and applied to the held-out fold, so no test information leaks. Same
  decoder class, same lambda, same procedure for both placements; only
  the data differ. 5-fold stratified CV over the 120 EVAL samples;
  reported accuracy is over all 120 held-out predictions.

  A note on circularity, stated rather than hidden: the reward signal
  during plasticity rewards OUTPUT_A activity on class 0 and OUTPUT_B on
  class 1, and the readout reads output populations. That is not circular
  -- it is the standard supervised setup, where the objective defines
  training and generalisation to UNSEEN samples is the test. The
  non-trivial question is whether the learned wiring generalises, which
  held-out accuracy answers and training accuracy would not.

PREDICTIONS, REGISTERED BEFORE RUNNING:

  P1 (manipulation checks -- ALL must hold before P2 is interpreted):
    P1a EQUIVALENCE: run_life_with_state reproduces imported run_life's
        (bridge_a, bridge_b) exactly. Guards a hand-copied loop.
    P1b FROZEN PARITY: |frozen_acc(interleaved) - frozen_acc(segregated)|
        < 5.0 percentage points. If the two placements differ in
        decodability BEFORE any learning, the plastic comparison is
        confounded by the random projection rather than by plasticity.
    P1c NO CEILING: mean frozen accuracy < 95.0% in both conditions. A
        random projection of 140 input channels onto 60 outputs may
        already carry most of the class information; if frozen is at
        ceiling there is no headroom for plasticity to demonstrate
        anything, and the experiment is UNINFORMATIVE rather than null.
        This is registered as a real possible outcome, not a failure.

  P2 (the actual test), primary endpoint = plastic-minus-frozen accuracy
     delta, per placement:
    P2a delta(interleaved) > delta(segregated), Welch p < 0.05.
    P2b the difference between those deltas is >= 5.0 percentage points
        -- a SUBSTANTIVE bar, registered so that a statistically
        significant but negligible gap is not reported as support.

DECISION RULE:
  - P1a fails: COMPROMISED. Nothing below is interpretable.
  - P1b fails: CONFOUNDED. The placements differ pre-learning; report the
    frozen gap and do not interpret P2 as a plasticity effect.
  - P1c fails: UNINFORMATIVE (ceiling). Report frozen accuracies and
    state that this design cannot answer the question; a harder task or
    a narrower readout would be needed.
  - P1 holds, P2a and P2b both hold: SUPPORTS the "determines
    learnability" framing. Placement changes what the network can
    actually classify, not merely what structure forms.
  - P1 holds, P2a holds but P2b fails: PARTIAL / NARROWS. The accuracy
    effect is real but small relative to the 7.25x structural effect;
    the title's strong reading is not supported at the measured
    magnitude and the claim should be stated quantitatively.
  - P2a fails: NULL. Segregation does NOT measurably cost classification
    accuracy in this setup. The paper's claim should be narrowed from
    "determines learnability" to "determines associative connectivity",
    and this result reported prominently rather than buried.

LIMITATIONS known in advance: one task (2-class SHD), one readout, one
decoder family, N=200, and a 1.89x input-output adjacency asymmetry
inherited from exp37b's geometry (1,198 vs 2,268 pairs) which is printed
in section [0]. Tinkering-grade this is not -- it is pre-registered --
but it is a single implementation, like everything else here.

Run: python src/experiments/exp52_accuracy_endpoint.py
Run (audit only, fast): python src/experiments/exp52_accuracy_endpoint.py --audit
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import json
import os
import time
from collections import Counter

import numpy as np
from scipy import stats

from sparse_state import SparsePairState
from spatial import SpatialGrid
from shd_loader import load_shd_samples
from exp37b_v2_real_data import (
    make_placement_segregated, make_placement_interleaved, run_life,
    N, N_INPUT, INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE,
)
from exp37c_real_data_scaled import cohens_d_pooled, INPUT_TYPE_CHANNELS

# ---- registered configuration (fixed before running) ----
PLASTIC_PER_CLASS = 20        # exp37c matched-life config
EVAL_PER_CLASS = 60           # held out from plasticity entirely
N_EPOCHS = 1                  # matched-life: ~8,600 steps, retention 0.651
SEEDS = list(range(12))
POOL_SIZE = 1500              # cached; top-2 classes have 89 / 87 samples
RIDGE_LAMBDA = 1.0            # FIXED, identical for both conditions
N_FOLDS = 5
MIN_RETENTION = 0.25
STEPS_PER_SAMPLE = 215

# registered thresholds
SUBSTANTIVE_PP = 5.0          # P2b: substantive accuracy gap
PARITY_PP = 5.0               # P1b: max tolerated frozen gap
CEILING_PCT = 95.0            # P1c: ceiling threshold

CACHE_PATH = os.path.join("shd_data", "exp52_cache.json")


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


def load_split(pool_size=POOL_SIZE):
    """Two most-common SHD classes. First PLASTIC_PER_CLASS samples of each
    go to the plastic phase; the NEXT EVAL_PER_CLASS go to evaluation and
    are never seen during plasticity."""
    stimuli, labels = load_shd_samples(
        n_samples=pool_size, n_channels_out=INPUT_TYPE_CHANNELS,
        bin_ms=4.0, max_steps=200)
    counts = Counter(labels)
    top = [c for c, _ in counts.most_common(2)]
    per_class = {c: [s for s, l in zip(stimuli, labels) if l == c] for c in top}
    need = PLASTIC_PER_CLASS + EVAL_PER_CLASS
    for c in top:
        if len(per_class[c]) < need:
            raise RuntimeError(
                "class {} has {} samples, need {}; raise POOL_SIZE".format(
                    c, len(per_class[c]), need))
    plastic = {c: per_class[c][:PLASTIC_PER_CLASS] for c in top}
    evalset = {c: per_class[c][PLASTIC_PER_CLASS:need] for c in top}
    return top, {c: len(per_class[c]) for c in top}, plastic, evalset


def run_life_with_state(coords, cids, seed, target_samples, distractor_samples,
                        n_epochs=5, input_gain=0.35, swap=200):
    """EXACT copy of exp37b_v2_real_data.run_life, with one addition: it
    also returns the final (src, dst, inhib) so the trained network can be
    evaluated on held-out data. Section [0] asserts bit-identical bridge
    values against the imported original -- do not edit one without the
    other."""
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
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


def initial_wiring():
    """The frozen network: exactly what run_life starts from. Weights never
    change in this harness, so 'plasticity disabled' IS this wiring."""
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    return src, dst, inhib


def evaluate_features(coords, cids, src, dst, inhib, samples, eval_seed,
                      input_gain=0.35):
    """Present each sample to the FROZEN network (no deposits, no rewiring)
    and return per-output-neuron spike counts. Membrane state is reset
    between samples so presentations are independent."""
    rng = np.random.default_rng(eval_seed)
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

    input_idx = np.where(cids == INPUT_TYPE)[0]
    a_idx = np.sort(np.where(cids == OUTPUT_A_TYPE)[0])
    b_idx = np.sort(np.where(cids == OUTPUT_B_TYPE)[0])
    read_idx = np.concatenate([a_idx, b_idx])

    feats = np.zeros((len(samples), len(read_idx)), dtype=float)
    for k, sample_arr in enumerate(samples):
        v = np.zeros(N); refrac = np.zeros(N, dtype=int)
        counts = np.zeros(N, dtype=float)
        for local_t in range(sample_arr.shape[0]):
            inp = rng.uniform(0, 0.01, N)
            inp[input_idx] += sample_arr[local_t] * input_gain
            v_ = v * 0.90 + inp
            fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
            if len(f):
                for fi in f:
                    for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
                counts[f] += 1.0
            v = np.maximum(v_, 0); v[fired] = 0
            refrac[fired] = 3; refrac[refrac > 0] -= 1
        feats[k] = counts[read_idx]
    return feats


def ridge_cv_accuracy(X, y, lam=RIDGE_LAMBDA, n_folds=N_FOLDS, seed=999):
    """Stratified k-fold CV accuracy of a closed-form ridge linear
    classifier. Standardisation and fit use TRAIN-fold data only."""
    X = np.asarray(X, float); y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        folds[idx] = np.arange(len(idx)) % n_folds
    ytr_sign = np.where(y == y.min(), -1.0, 1.0)

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
        pred = np.sign(Zte @ w)
        pred[pred == 0] = 1.0
        correct += int((pred == ytr_sign[te]).sum())
    return 100.0 * correct / len(y)


def adjacency_count(coords, cids, radius=6.0):
    g = SpatialGrid(coords, radius)
    inp = set(np.where(cids == INPUT_TYPE)[0].tolist())
    out = set(np.where(cids != INPUT_TYPE)[0].tolist())
    return sum(1 for i in inp for j in g.within(i, radius) if int(j) in out)


if __name__ == "__main__":
    audit_only = "--audit" in sys.argv
    print("=" * 86)
    print("EXP 52: does segregated placement cost CLASSIFICATION ACCURACY?")
    print("Predictions and decision rule are fixed in the docstring above.")
    print("=" * 86)

    top, avail, plastic, evalset = load_split()
    plastic_t = plastic[top[0]]; plastic_d = plastic[top[1]]
    eval_samples = evalset[top[0]] + evalset[top[1]]
    eval_labels = np.array([0] * len(evalset[top[0]]) + [1] * len(evalset[top[1]]))
    est_steps = 2 * PLASTIC_PER_CLASS * N_EPOCHS * STEPS_PER_SAMPLE
    retention = 0.99995 ** est_steps

    print("\n[0] Audit")
    print("  classes: target={} (avail {}), distractor={} (avail {})".format(
        top[0], avail[top[0]], top[1], avail[top[1]]))
    print("  plastic set : {}/class (seen by plasticity)".format(PLASTIC_PER_CLASS))
    print("  eval set    : {}/class = {} samples (NEVER seen by plasticity)".format(
        EVAL_PER_CLASS, len(eval_samples)))
    print("  accuracy resolution: {:.2f} percentage points".format(100.0 / len(eval_samples)))
    print("  life ~{:,} steps, V retention {:.3f}".format(est_steps, retention))
    assert retention >= MIN_RETENTION, "retention {:.3f} below {}".format(
        retention, MIN_RETENTION)

    conditions = [("segregated", make_placement_segregated),
                  ("interleaved", make_placement_interleaved)]
    for name, fn in conditions:
        c, ci = fn()
        print("  {:>12}: {:,} input-output adjacent pairs (radius 6.0)".format(
            name, adjacency_count(c, ci)))

    print("\n  [P1a] equivalence: run_life_with_state vs imported run_life")
    c0, ci0 = make_placement_segregated()
    ba_ref, bb_ref = run_life(c0, ci0, 0, plastic_t[:4], plastic_d[:4], n_epochs=1)
    ba_new, bb_new, _, _, _ = run_life_with_state(
        c0, ci0, 0, plastic_t[:4], plastic_d[:4], n_epochs=1)
    print("    imported run_life      : bridge_a={:.10f} bridge_b={:.10f}".format(ba_ref, bb_ref))
    print("    run_life_with_state    : bridge_a={:.10f} bridge_b={:.10f}".format(ba_new, bb_new))
    p1a = (ba_ref == ba_new) and (bb_ref == bb_new)
    print("    P1a {}".format("HOLDS (bit-identical)" if p1a
                              else "FAILS -- COMPROMISED, stop here"))
    if not p1a:
        sys.exit(1)
    if audit_only:
        print("\n  --audit: stopping before the sweep.")
        sys.exit(0)

    print("\n[1] Running {} seeds x 2 conditions (plastic + frozen)".format(len(SEEDS)))
    cache = _cache_load()
    results = {}
    src0, dst0, inhib0 = initial_wiring()

    for name, fn in conditions:
        coords, cids = fn()
        plastic_accs, frozen_accs, bridges = [], [], []
        print("\n  {}".format(name.upper()))
        for s in SEEDS:
            key = "{}|{}|spc{}|ep{}|ev{}".format(name, s, PLASTIC_PER_CLASS,
                                                 N_EPOCHS, EVAL_PER_CLASS)
            if key in cache:
                pa, fa, br = cache[key]
                print("    seed {:>2}: plastic={:.2f}%  frozen={:.2f}%  bridge={:.1f}  (cached)".format(
                    s, pa, fa, br))
            else:
                t0 = time.time()
                ba, bb, src, dst, inhib = run_life_with_state(
                    coords, cids, s, plastic_t, plastic_d, n_epochs=N_EPOCHS)
                Xp = evaluate_features(coords, cids, src, dst, inhib,
                                       eval_samples, 5000 + s)
                pa = ridge_cv_accuracy(Xp, eval_labels)
                Xf = evaluate_features(coords, cids, src0, dst0, inhib0,
                                       eval_samples, 5000 + s)
                fa = ridge_cv_accuracy(Xf, eval_labels)
                br = ba + bb
                cache[key] = [float(pa), float(fa), float(br)]
                _cache_save(cache)
                print("    seed {:>2}: plastic={:.2f}%  frozen={:.2f}%  bridge={:.1f}  ({:.0f}s)".format(
                    s, pa, fa, br, time.time() - t0))
            plastic_accs.append(pa); frozen_accs.append(fa); bridges.append(br)
            sys.stdout.flush()
        results[name] = {
            "plastic": np.array(plastic_accs, float),
            "frozen": np.array(frozen_accs, float),
            "bridge": np.array(bridges, float),
        }

    seg, inter = results["segregated"], results["interleaved"]
    seg_delta = seg["plastic"] - seg["frozen"]
    int_delta = inter["plastic"] - inter["frozen"]

    print("\n" + "=" * 86)
    print("[2] Four-cell result (held-out accuracy, mean +/- s.d. over {} seeds)".format(len(SEEDS)))
    print("=" * 86)
    print("  {:<14} {:>16} {:>16} {:>16} {:>12}".format(
        "placement", "plastic acc", "frozen acc", "delta (p-f)", "bridge mass"))
    for name, r, d in (("segregated", seg, seg_delta), ("interleaved", inter, int_delta)):
        print("  {:<14} {:>9.2f} +/-{:<5.2f} {:>9.2f} +/-{:<5.2f} {:>9.2f} +/-{:<5.2f} {:>12.1f}".format(
            name, r["plastic"].mean(), r["plastic"].std(ddof=1),
            r["frozen"].mean(), r["frozen"].std(ddof=1),
            d.mean(), d.std(ddof=1), r["bridge"].mean()))
    print("  chance = 50.00%  |  bridge-mass ratio (this run) = {:.2f}x".format(
        inter["bridge"].mean() / max(seg["bridge"].mean(), 1e-9)))

    print("\n[3] P1b / P1c manipulation checks")
    frozen_gap = abs(inter["frozen"].mean() - seg["frozen"].mean())
    print("    frozen segregated  = {:.2f}%".format(seg["frozen"].mean()))
    print("    frozen interleaved = {:.2f}%".format(inter["frozen"].mean()))
    print("    |gap| = {:.2f} pp   (registered tolerance {:.1f} pp)".format(frozen_gap, PARITY_PP))
    p1b = frozen_gap < PARITY_PP
    print("    P1b {}".format("HOLDS" if p1b else "FAILS -- CONFOUNDED"))
    p1c = (seg["frozen"].mean() < CEILING_PCT) and (inter["frozen"].mean() < CEILING_PCT)
    print("    P1c {} (ceiling threshold {:.1f}%)".format(
        "HOLDS -- headroom exists" if p1c else "FAILS -- UNINFORMATIVE (ceiling)",
        CEILING_PCT))

    print("\n[4] P2 (the actual test): plastic-minus-frozen accuracy delta")
    t2, p2 = stats.ttest_ind(int_delta, seg_delta, equal_var=False)
    delta_gap = int_delta.mean() - seg_delta.mean()
    d_eff = cohens_d_pooled(int_delta, seg_delta)
    print("    delta segregated  = {:+.2f} pp".format(seg_delta.mean()))
    print("    delta interleaved = {:+.2f} pp".format(int_delta.mean()))
    print("    difference        = {:+.2f} pp".format(delta_gap))
    print("    Welch t={:.3f}, p={:.4e}, Cohen's d={:.2f}".format(t2, p2, d_eff))
    p2a = (delta_gap > 0) and (p2 < 0.05)
    p2b = delta_gap >= SUBSTANTIVE_PP
    print("    P2a (interleaved > segregated, p<0.05) : {}".format(p2a))
    print("    P2b (gap >= {:.1f} pp, substantive)      : {}".format(SUBSTANTIVE_PP, p2b))

    print("\n" + "=" * 86)
    print("[5] VERDICT (by the decision rule registered above)")
    print("=" * 86)
    if not p1b:
        print("  CONFOUNDED: the placements differ in decodability BEFORE any learning")
        print("  ({:.2f} pp frozen gap). P2 cannot be read as a plasticity effect.".format(frozen_gap))
    elif not p1c:
        print("  UNINFORMATIVE (ceiling): the frozen random projection already carries")
        print("  the class information, leaving no headroom for plasticity to demonstrate")
        print("  anything. This design cannot answer the question; a harder task or a")
        print("  narrower readout would be needed. This is NOT evidence against the claim.")
    elif p2a and p2b:
        print("  SUPPORTS the 'determines learnability' framing: segregated placement")
        print("  measurably costs held-out CLASSIFICATION ACCURACY, not merely associative")
        print("  structure. The title's strong reading is earned.")
    elif p2a and not p2b:
        print("  PARTIAL / NARROWS: the accuracy effect is real and significant but below")
        print("  the registered substantive bar of {:.1f} pp. Report the accuracy cost".format(SUBSTANTIVE_PP))
        print("  quantitatively; do not lean on 'determines learnability' unqualified.")
    else:
        print("  NULL: segregation does NOT measurably cost classification accuracy here.")
        print("  The structural finding stands, but the paper's claim should be NARROWED")
        print("  from 'determines learnability' to 'determines associative connectivity'.")
        print("  This should be reported prominently, not buried.")
