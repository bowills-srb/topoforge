"""Exp 62: is taught structure a valid proxy for decoded capability, and how
much capability does a given amount of structure actually buy?

WHY THIS EXISTS, AND WHY IT IS THE MOST LOAD-BEARING FILE IN THE PROJECT RIGHT
NOW. Every headline this project publishes is a STRUCTURAL ratio -- 3.8x-4.2x
synthetic, 2.92x pure graph locality, 2.74x real-data corrected, 3.59x on
exp52b's harness. Not one of them is a capability ratio. Exp 52b measured
capability for the first time and got 1.079x in accuracy terms; Exp 61 showed
that shrinkage is NOT an artifact of a coarse metric -- a continuous endpoint
with 4x the resolution (239/240 distinct cells against accuracy's 54/240)
returned the same effect size, d 0.590 vs 0.575, difference CI [-0.291,
+0.325]. So the structure-to-capability gap is real and the obvious excuse for
it is gone.

That leaves the question nobody has asked directly: does taught structure
predict capability AT ALL, and if so with what exchange rate? If it does not,
every structural headline in the preprint is measuring something only loosely
coupled to what a user of the hardware would care about, and the paper's
framing has to change. If it does, the exchange rate is the number that should
be quoted instead of a raw structural ratio.

THE NON-CIRCULAR VERSION OF THE QUESTION. Correlating structure with
performance across BOTH placements pooled is nearly guaranteed to succeed:
interleaved has more of both, so the correlation mostly re-measures the
placement difference already known. The pooled estimate on the discovery seeds
is r=+0.420, and it is not the interesting number. The interesting number is
the WITHIN-CONDITION partial correlation -- among networks of the SAME
placement, differing only by seed, does more taught structure mean better
held-out decoding? That cannot be produced by the placement contrast. On the
discovery seeds it is r=+0.361 (p=0.001, n=80, bootstrap CI [0.167, 0.545]).

--------------------------------------------------------------------------
DISCOVERY / CONFIRMATION, BECAUSE THE PRIMARY HAS ALREADY BEEN SEEN
--------------------------------------------------------------------------
The r=+0.361 above was computed at a prompt, post hoc, after Exp 61 finished.
Pre-registering a threshold it has already cleared would be theatre, and this
project has twice been caught by results that looked solid until re-run (Exp
49's single-placement-seed artifact; Exp 57's AGE lead, which happened to
survive). So the file is split:

  DISCOVERY  seeds 100-139, features already cached by Exp 61, bridge mass
             from Exp 52b's cache. Everything here is DESCRIPTIVE and is
             labelled as such in the output. It generated the hypothesis and
             set the threshold below. It proves nothing on its own.

  CONFIRM    seeds 140-179, FRESH, disjoint, never simulated before this
             file. Same physics, same wiring rule, same readout, same
             lambda. This is where the pre-registered predictions are
             evaluated. ~20 minutes of simulation, cached afterwards.

The threshold is shrunk from the discovery estimate on purpose. A correlation
found by looking is biased upward -- the winner's-curse arithmetic that made
Exp 57's +66.7% worth re-running at 20 seeds (it replicated at +69.1%, which
is how you tell a real effect from a selection artifact). Discovery r=+0.361
with a CI whose lower bound is +0.167, so the registered bar for confirmation
is r >= +0.20 with a bootstrap CI excluding zero. Clearing +0.361 again is not
required and should not be expected.

--------------------------------------------------------------------------
PRE-REGISTERED PREDICTIONS -- evaluated on CONFIRM seeds only
--------------------------------------------------------------------------
P0  GATE. `ridge_cv_scores` must be the function Exp 61 gated bit-identically
    against the published readout, and the discovery statistics recomputed
    here must match the values quoted above. If the discovery numbers do not
    reproduce, this file is reading the caches wrong and nothing in it counts.

P1  PRIMARY. Within-condition partial correlation between per-seed bridge mass
    and per-seed delta-rsq, on the CONFIRM seeds: r >= +0.20, bootstrap CI
    excluding zero.
      HOLDS   -> structure is a genuine, replicated predictor of capability
                 among networks of identical placement. The structural
                 endpoint is a VALID proxy and the preprint may say so.
      FAILS   -> the discovery correlation did not replicate. The structural
                 headlines measure something whose coupling to capability is
                 unestablished, and the preprint must say THAT instead. This
                 is the outcome that costs the project most and it is reported
                 first if it happens.

P2  EXCHANGE RATE, descriptive with a CI, no threshold. The structural ratio
    (interleaved/segregated bridge mass) and the decoded ratio
    (interleaved/segregated delta-rsq), and the COMPRESSION FACTOR between
    them. Discovery gave 3.59x structural -> 1.24x decoded, a compression of
    ~2.9x. This is the number that should replace raw structural ratios in any
    capability claim, so it gets a confidence interval and a replication.

P3  CROSS-CONDITION EXTRAPOLATION -- the circularity test, and the reason this
    file exists rather than a one-line correlation. Fit delta-rsq as a
    function of bridge mass using INTERLEAVED seeds ONLY, then predict
    SEGREGATED's mean delta-rsq from segregated's mean bridge mass, and vice
    versa. A curve fitted through two clusters will always reproduce the ratio
    between those two clusters -- that is tautological and must not be
    reported as a law. Extrapolating a fit from one cluster to the other is
    not.
      Registered: |predicted - observed| < 1.0 within-condition SD of
      delta-rsq, in BOTH directions.
      HOLDS -> one common structure-capability relation spans both
               placements; placement acts THROUGH structure and not in
               addition to it.
      FAILS -> the two placements sit on different curves, so placement does
               something to capability that taught mass does not capture.
               That would be a genuinely new finding and a bigger one than P1.

P4  FUNCTIONAL FORM, PRE-COMMITTED TO A NEGATIVE. Linear, sqrt and log10 fits
    compared by 5-fold cross-validated R^2. The bridge-mass values cluster in
    two groups ~2,200 and ~7,900, so the form is weakly constrained by
    construction, and on discovery the three were within 0.03 R^2 of each
    other (0.176 / 0.190 / 0.210). Registered: a form may be declared only if
    its CV R^2 exceeds both others by >= 0.05. Otherwise the output says
    UNDETERMINED and the preprint says "compressive" and NOT "logarithmic".
    Written down in advance so the best-fitting curve cannot quietly become
    "the law" in a later draft.

P5  DOES THE PROXY DEPEND ON THE DECODER? Within-condition partial r
    recomputed across Exp 61's lambda grid. Exp 61 found the decoded
    PLACEMENT gap grows monotonically as the readout weakens (rho=+1.000),
    driven by interleaved improving rather than segregated degrading. If the
    structure-capability coupling also moves with lambda, then "structure
    predicts capability" is itself decoder-dependent and every number here
    carries a lambda. Descriptive; reported whatever it shows.

P6  LIMITATION, REGISTERED AS SUCH SO IT CANNOT BE DROPPED. Report 1 - R^2,
    the fraction of seed-level capability variance that taught mass does NOT
    explain. On discovery this is ~0.79. A proxy that explains a fifth of the
    variance describes central tendency across a 3.6x structural range; it is
    not a per-network predictor and must never be presented as one.

--------------------------------------------------------------------------
SCOPE
--------------------------------------------------------------------------
No simulation physics is introduced or changed here. Placement, wiring,
plastic phase, feature extraction, readout and lambda are all imported. The
only new thing is 40 fresh seeds and the analysis. One task (2-class SHD), one
readout family (linear ridge on spike counts), one network size -- the same
scope Exp 52b and Exp 61 carry, inherited unchanged.

"Bridge mass" is Exp 52b's `ba + bb`: summed V-value on input->output pairs of
the rewarded types. It is the structural quantity the real-data headlines are
computed from, which is why it and not some other structural measure is the
one under test.

Run:              python src/experiments/exp62_structure_performance_law.py
Run (discovery only, instant -- no simulation):
                  python src/experiments/exp62_structure_performance_law.py --discovery-only
Run (analysis only, once CONFIRM seeds are cached):
                  python src/experiments/exp62_structure_performance_law.py --from-cache
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
from exp52_accuracy_endpoint import (
    evaluate_features, load_split,
    PLASTIC_PER_CLASS, EVAL_PER_CLASS, N_EPOCHS,
)
from exp52b_accuracy_corrected import (
    run_life_corrected, make_wiring, SEEDS, LEGACY_WIRING_SEED, WIRING_MODE,
)
from exp61_continuous_endpoint import (
    ridge_cv_scores, feat_key, _feat_load, LAMBDA_GRID,
    REGISTERED_LAMBDA, PRIMARY_FROZEN, PRIMARY_METRIC,
)

# ---- registered configuration (fixed before the CONFIRM seeds were run) ----
DISCOVERY_SEEDS = list(SEEDS)                 # 100-139, already cached
CONFIRM_SEEDS = list(range(140, 180))         # fresh, disjoint
R_THRESHOLD = 0.20                            # P1, shrunk from discovery 0.361
EXTRAP_SD_MULT = 1.0                          # P3
FORM_MARGIN = 0.05                            # P4
CONDITIONS = [("segregated", make_placement_segregated),
              ("interleaved", make_placement_interleaved)]

# values recomputed by P0 as a self-consistency check on the caches
DISCOVERY_QUOTED = {"partial_r": 0.3608, "pooled_r": 0.4197,
                    "struct_ratio": 3.59, "decoded_ratio": 1.24}

FEAT_PATH = os.path.join("shd_data", "exp62_features.npz")
BRIDGE_PATH = os.path.join("shd_data", "exp62_bridge.json")
CACHE_PATH = os.path.join("shd_data", "exp62_cache.json")
EXP52B_CACHE = os.path.join("shd_data", "exp52b_cache.json")


# ------------------------------------------------------------------ helpers
def partial_r(x, y, group, n_boot=20000, seed=7):
    """Within-group correlation: centre each group separately, then correlate.

    This is the non-circular statistic. Pooling without centring lets the
    between-placement difference in BOTH variables manufacture a correlation
    that merely re-states the placement contrast.
    """
    x = np.asarray(x, float).copy(); y = np.asarray(y, float).copy()
    group = np.asarray(group)
    for g in np.unique(group):
        m = group == g
        x[m] -= x[m].mean(); y[m] -= y[m].mean()
    r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = stats.pearsonr(x[idx], y[idx])[0]
    return (float(r), float(p),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def ratio_ci(num, den, n_boot=20000, seed=11):
    num = np.asarray(num, float); den = np.asarray(den, float)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        a = num[rng.integers(0, len(num), len(num))].mean()
        b = den[rng.integers(0, len(den), len(den))].mean()
        vals[i] = a / max(b, 1e-12)
    return (float(num.mean() / max(den.mean(), 1e-12)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def cv_r2(x, y, transform, n_folds=5, seed=321):
    """5-fold CV R^2 of a 1-D linear fit on transform(x)."""
    x = transform(np.asarray(x, float)); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    folds = np.arange(len(y)) % n_folds
    folds = folds[rng.permutation(len(y))]
    pred = np.empty(len(y))
    for k in range(n_folds):
        te = folds == k; tr = ~te
        sl, ic = np.polyfit(x[tr], y[tr], 1)
        pred[te] = sl * x[te] + ic
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


TRANSFORMS = (("linear", lambda v: v),
              ("sqrt", lambda v: np.sqrt(np.maximum(v, 0.0))),
              ("log10", lambda v: np.log10(np.maximum(v, 1.0))))


# ------------------------------------------------------------- data assembly
def discovery_rows(labels, lam=REGISTERED_LAMBDA):
    """Bridge mass and delta-metric for seeds 100-139, entirely from cache."""
    store = _feat_load()
    prior = json.load(open(EXP52B_CACHE))
    B, D, G = [], [], []
    for cond, _ in CONDITIONS:
        for s in DISCOVERY_SEEDS:
            k52 = "{}|{}|spc{}|ep{}|ev{}|log".format(
                cond, s, PLASTIC_PER_CLASS, N_EPOCHS, EVAL_PER_CLASS)
            p = ridge_cv_scores(store[feat_key(cond, s, "plastic")],
                                labels, lam=lam)[PRIMARY_METRIC]
            f = ridge_cv_scores(store[feat_key(cond, s, PRIMARY_FROZEN)],
                                labels, lam=lam)[PRIMARY_METRIC]
            B.append(prior[k52][2]); D.append(p - f); G.append(cond)
    return np.array(B, float), np.array(D, float), np.array(G)


def _f_load():
    if not os.path.exists(FEAT_PATH):
        return {}
    with np.load(FEAT_PATH) as z:
        return {k: z[k] for k in z.files}


def _f_save(store):
    tmp = FEAT_PATH + ".tmp.npz"
    np.savez_compressed(tmp, **store)
    os.replace(tmp, FEAT_PATH)


def confirm_extract(coords_map, plastic_t, plastic_d, eval_samples, t0):
    """Simulate the fresh seeds, caching features AND bridge mass.

    Exp 61's extractor discarded the bridge values `run_life_corrected`
    returns, so they are captured here rather than recovered later.
    """
    store = _f_load()
    bridge = json.load(open(BRIDGE_PATH)) if os.path.exists(BRIDGE_PATH) else {}
    for cond, _ in CONDITIONS:
        coords, cids = coords_map[cond]
        print("\n  {}".format(cond.upper()))
        for s in CONFIRM_SEEDS:
            kp = feat_key(cond, s, "plastic")
            kf = feat_key(cond, s, PRIMARY_FROZEN)
            kb = "{}|{}".format(cond, s)
            if kp in store and kf in store and kb in bridge:
                print("    seed {:>3}: cached".format(s))
                continue
            ws = LEGACY_WIRING_SEED + s
            ba, bb, src, dst, inhib = run_life_corrected(
                coords, cids, s, plastic_t, plastic_d, n_epochs=N_EPOCHS,
                wiring_seed=ws, inhib_mode=WIRING_MODE)
            store[kp] = evaluate_features(coords, cids, src, dst, inhib,
                                          eval_samples, 5000 + s)
            s0, d0, i0 = make_wiring(cids, ws, "logical")
            store[kf] = evaluate_features(coords, cids, s0, d0, i0,
                                          eval_samples, 5000 + s)
            bridge[kb] = float(ba + bb)
            _f_save(store)
            with open(BRIDGE_PATH, "w") as fh:
                json.dump(bridge, fh)
            print("    seed {:>3}: simulated + cached  bridge={:8.1f}  [{:.0f}s]".format(
                s, bridge[kb], time.time() - t0), flush=True)
    return store, bridge


def confirm_rows(store, bridge, labels, lam=REGISTERED_LAMBDA):
    B, D, G = [], [], []
    for cond, _ in CONDITIONS:
        for s in CONFIRM_SEEDS:
            p = ridge_cv_scores(store[feat_key(cond, s, "plastic")],
                                labels, lam=lam)[PRIMARY_METRIC]
            f = ridge_cv_scores(store[feat_key(cond, s, PRIMARY_FROZEN)],
                                labels, lam=lam)[PRIMARY_METRIC]
            B.append(bridge["{}|{}".format(cond, s)]); D.append(p - f); G.append(cond)
    return np.array(B, float), np.array(D, float), np.array(G)


# ---------------------------------------------------------------- reporting
def describe(tag, B, D, G, descriptive):
    r, p, lo, hi = partial_r(B, D, G)
    pooled = stats.pearsonr(B, D)[0]
    print("\n  {} SET  (n={} across {} placements){}".format(
        tag, len(B), len(np.unique(G)),
        "   [DESCRIPTIVE -- generated the hypothesis, proves nothing]"
        if descriptive else "   [CONFIRMATORY]"))
    for cond in [c for c, _ in CONDITIONS]:
        m = G == cond
        rc, pc = stats.pearsonr(B[m], D[m])
        print("    {:<12} bridge {:8.0f}   d{:<4} {:.4f}   own r = {:+.3f} (p={:.3f})"
              .format(cond, B[m].mean(), PRIMARY_METRIC, D[m].mean(), rc, pc))
    print("    within-condition partial r = {:+.4f}  (p={:.4g})  95% CI [{:+.3f}, {:+.3f}]"
          .format(r, p, lo, hi))
    print("    pooled r = {:+.4f}  <- inflated by the placement contrast; not the test"
          .format(pooled))
    return {"partial_r": r, "p": p, "ci": [lo, hi], "pooled_r": float(pooled)}


def exchange_rate(B, D, G):
    si = G == "segregated"; ii = G == "interleaved"
    sr, srl, srh = ratio_ci(B[ii], B[si])
    dr, drl, drh = ratio_ci(D[ii], D[si])
    comp = sr / max(dr, 1e-12)
    print("    structural ratio (bridge mass)  = {:.2f}x   95% CI [{:.2f}, {:.2f}]"
          .format(sr, srl, srh))
    print("    decoded ratio    (delta {:<4})   = {:.2f}x   95% CI [{:.2f}, {:.2f}]"
          .format(PRIMARY_METRIC, dr, drl, drh))
    print("    COMPRESSION: {:.2f}x of structure buys {:.2f}x of capability "
          "({:.1f}x compression)".format(sr, dr, comp))
    return {"struct_ratio": sr, "struct_ci": [srl, srh],
            "decoded_ratio": dr, "decoded_ci": [drl, drh], "compression": comp}


def extrapolate(B, D, G):
    """P3: fit on one placement, predict the other placement's mean."""
    print("\n[P3] CROSS-CONDITION EXTRAPOLATION (the circularity test)")
    ok = True
    out = {}
    for fit_on, predict in (("interleaved", "segregated"),
                            ("segregated", "interleaved")):
        mf = G == fit_on; mp = G == predict
        x = np.log10(np.maximum(B[mf], 1.0))
        sl, ic = np.polyfit(x, D[mf], 1)
        pred = sl * np.log10(max(B[mp].mean(), 1.0)) + ic
        obs = D[mp].mean()
        sd = D[mp].std(ddof=1)
        err = abs(pred - obs)
        passes = err < EXTRAP_SD_MULT * sd
        ok = ok and passes
        out[fit_on] = {"pred": float(pred), "obs": float(obs),
                       "sd": float(sd), "pass": bool(passes)}
        print("    fit on {:<12} -> predict {:<12} : {:.4f} vs observed {:.4f}"
              "   |err| = {:.4f} ({:.2f} SD)  {}".format(
                  fit_on, predict, pred, obs, err, err / max(sd, 1e-12),
                  "PASS" if passes else "FAIL"))
    print("    P3 {}".format(
        "HOLDS -- one structure-capability relation spans both placements. "
        "Placement acts THROUGH taught structure, not in addition to it." if ok else
        "FAILS -- the placements sit on DIFFERENT curves. Placement affects "
        "capability by a route taught mass does not capture. This is a larger "
        "finding than P1 and belongs at the top of the writeup."))
    out["pass"] = bool(ok)
    return out


def functional_form(B, D):
    print("\n[P4] FUNCTIONAL FORM (pre-committed to UNDETERMINED unless a form "
          "wins by {:.2f} CV R^2)".format(FORM_MARGIN))
    scores = {}
    for name, fn in TRANSFORMS:
        scores[name] = cv_r2(B, D, fn)
        print("    {:<8} CV R^2 = {:.4f}".format(name, scores[name]))
    order = sorted(scores, key=scores.get, reverse=True)
    margin = scores[order[0]] - scores[order[1]]
    decided = margin >= FORM_MARGIN
    print("    best '{}' leads by {:.4f}  ->  {}".format(
        order[0], margin,
        "'{}' ESTABLISHED".format(order[0]) if decided else
        "UNDETERMINED. Say 'compressive', never 'logarithmic'. The bridge-mass "
        "values cluster in two groups, so the form is not identifiable from "
        "this design and a best-fitting curve here is not a law."))
    return {"scores": scores, "best": order[0], "margin": float(margin),
            "decided": bool(decided)}


def lambda_dependence(store_rows_fn, labels):
    print("\n[P5] IS THE PROXY DECODER-DEPENDENT? within-condition r vs lambda")
    out = {}
    print("    {:>8} {:>10} {:>22}".format("lambda", "partial r", "95% CI"))
    for lam in LAMBDA_GRID:
        B, D, G = store_rows_fn(lam)
        r, p, lo, hi = partial_r(B, D, G, n_boot=4000)
        out[str(lam)] = {"r": r, "ci": [lo, hi]}
        print("    {:>8g} {:>10.3f} {:>22}".format(
            lam, r, "[{:+.3f}, {:+.3f}]".format(lo, hi)))
    rs = [out[str(l)]["r"] for l in LAMBDA_GRID]
    rho, prho = stats.spearmanr(np.log10(LAMBDA_GRID), rs)
    print("    Spearman rho(log10 lambda, r) = {:+.3f} (p={:.3g}); spread "
          "{:.3f}".format(rho, prho, max(rs) - min(rs)))
    print("    {}".format(
        "Coupling is stable across the decoder grid -- the proxy claim does not "
        "carry a lambda." if max(rs) - min(rs) < 0.15 else
        "Coupling MOVES with the decoder. Every 'structure predicts capability' "
        "number here must be quoted with its lambda."))
    out["rho"] = float(rho); out["spread"] = float(max(rs) - min(rs))
    return out


# --------------------------------------------------------------------- main
def main(args):
    t0 = time.time()
    print("=" * 92)
    print("EXP 62: is taught structure a valid proxy for decoded capability,")
    print("        and what is the exchange rate?  (discovery / confirmation split)")
    print("=" * 92)

    top, avail, plastic, evalset = load_split()
    plastic_t = plastic[top[0]]; plastic_d = plastic[top[1]]
    eval_samples = evalset[top[0]] + evalset[top[1]]
    labels = np.array([0] * len(evalset[top[0]]) + [1] * len(evalset[top[1]]))
    coords_map = {name: fn() for name, fn in CONDITIONS}

    print("\n[0] Configuration")
    print("  discovery seeds {}-{} ({})   confirm seeds {}-{} ({}, disjoint)".format(
        DISCOVERY_SEEDS[0], DISCOVERY_SEEDS[-1], len(DISCOVERY_SEEDS),
        CONFIRM_SEEDS[0], CONFIRM_SEEDS[-1], len(CONFIRM_SEEDS)))
    print("  metric '{}' at lambda={:g}, frozen arm '{}' -- all imported from exp61"
          .format(PRIMARY_METRIC, REGISTERED_LAMBDA, PRIMARY_FROZEN))
    print("  registered bar: within-condition partial r >= {:+.2f}, CI excluding 0"
          .format(R_THRESHOLD))

    # ---- P0 gate: discovery must reproduce -----------------------------
    print("\n[P0] GATE -- discovery statistics must reproduce the quoted values")
    Bd, Dd, Gd = discovery_rows(labels)
    disc = describe("DISCOVERY", Bd, Dd, Gd, descriptive=True)
    checks = {"partial_r": disc["partial_r"], "pooled_r": disc["pooled_r"]}
    ok = all(abs(checks[k] - DISCOVERY_QUOTED[k]) < 5e-4 for k in checks)
    for k in checks:
        print("    {:<12} recomputed {:+.4f}   docstring {:+.4f}   match: {}".format(
            k, checks[k], DISCOVERY_QUOTED[k],
            abs(checks[k] - DISCOVERY_QUOTED[k]) < 5e-4))
    print("    P0 {}".format("HOLDS" if ok else
                             "FAILS -- caches misread, STOP"))
    if not ok:
        sys.exit(1)
    print("\n  [discovery exchange rate, descriptive]")
    disc_ex = exchange_rate(Bd, Dd, Gd)

    if args.discovery_only:
        print("\n  --discovery-only: nothing above is confirmatory. Stopping.")
        sys.exit(0)

    # ---- confirmation --------------------------------------------------
    if args.from_cache:
        store, bridge = _f_load(), (json.load(open(BRIDGE_PATH))
                                    if os.path.exists(BRIDGE_PATH) else {})
        missing = [s for c, _ in CONDITIONS for s in CONFIRM_SEEDS
                   if "{}|{}".format(c, s) not in bridge]
        if missing:
            print("\n  --from-cache: {} confirm seeds missing. Run without the "
                  "flag first.".format(len(missing)))
            sys.exit(1)
        print("\n[1] --from-cache: {} confirm seed-conditions, no simulation".format(
            len(CONFIRM_SEEDS) * len(CONDITIONS)))
    else:
        print("\n[1] CONFIRMATION -- simulating {} fresh seeds x {} placements".format(
            len(CONFIRM_SEEDS), len(CONDITIONS)))
        store, bridge = confirm_extract(coords_map, plastic_t, plastic_d,
                                        eval_samples, t0)

    Bc, Dc, Gc = confirm_rows(store, bridge, labels)

    print("\n" + "=" * 92)
    print("[2] P1 PRIMARY -- does the correlation replicate on fresh seeds?")
    print("=" * 92)
    conf = describe("CONFIRMATION", Bc, Dc, Gc, descriptive=False)
    p1 = (conf["partial_r"] >= R_THRESHOLD) and (conf["ci"][0] > 0)
    shrink = conf["partial_r"] - disc["partial_r"]
    print("\n    registered bar r >= {:+.2f} with CI excluding zero".format(R_THRESHOLD))
    print("    discovery {:+.3f} -> confirmation {:+.3f}   (shrinkage {:+.3f})".format(
        disc["partial_r"], conf["partial_r"], shrink))
    print("    P1 {}".format(
        "HOLDS -- structure is a replicated predictor of capability among "
        "networks of identical placement. The structural endpoint is a VALID "
        "proxy and the preprint may say so." if p1 else
        "FAILS -- the discovery correlation did not replicate. The structural "
        "headlines measure a quantity whose coupling to capability is NOT "
        "established. Report this first; it costs the project the most."))

    print("\n[P2] EXCHANGE RATE -- what a structural ratio is worth")
    conf_ex = exchange_rate(Bc, Dc, Gc)
    print("    [discovery gave {:.2f}x -> {:.2f}x]".format(
        disc_ex["struct_ratio"], disc_ex["decoded_ratio"]))

    ext = extrapolate(Bc, Dc, Gc)
    form = functional_form(Bc, Dc)

    lam_out = lambda_dependence(
        lambda lam: confirm_rows(store, bridge, labels, lam=lam), labels)

    print("\n[P6] LIMITATION -- variance NOT explained by taught mass")
    r_all = stats.pearsonr(Bc, Dc)[0]
    rp = conf["partial_r"]
    print("    pooled R^2 = {:.3f}  -> {:.0%} of seed-level capability variance "
          "unexplained".format(r_all ** 2, 1 - r_all ** 2))
    print("    within-condition R^2 = {:.3f}  -> {:.0%} unexplained".format(
        rp ** 2, 1 - rp ** 2))
    print("    This proxy describes central tendency across a {:.1f}x structural "
          "range. It is NOT a per-network predictor and must not be presented "
          "as one.".format(conf_ex["struct_ratio"]))

    with open(CACHE_PATH, "w") as fh:
        json.dump({"discovery": disc, "discovery_exchange": disc_ex,
                   "confirmation": conf, "confirmation_exchange": conf_ex,
                   "p1": bool(p1), "shrinkage": float(shrink),
                   "p3": ext, "p4": form, "p5": lam_out,
                   "confirm_seeds": CONFIRM_SEEDS,
                   "bridge": Bc.tolist(), "delta": Dc.tolist(),
                   "group": Gc.tolist()}, fh, indent=1)
    print("\n  cached -> {}  ({:.0f}s)".format(CACHE_PATH, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery-only", action="store_true",
                    help="descriptive discovery statistics only, no simulation")
    ap.add_argument("--from-cache", action="store_true",
                    help="analysis only, from cached confirm-seed features")
    main(ap.parse_args())
