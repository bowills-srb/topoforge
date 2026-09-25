"""DIAGNOSTIC (descriptive, post-hoc): is Exp 62's failure -- and the
saturation curve proposed to explain it -- a property of the PHYSICS, or of
the linear ridge-regression readout and the bounded R^2 metric used to score
it?

NOTHING HERE CAN OVERTURN EXP 62. P1 (within-condition structure->capability
correlation) and P3 (cross-condition extrapolation) FAILED under the
pre-registered readout and metric, and those verdicts stand. This file asks a
narrower, legitimate question: were the registered instruments capable of
seeing the effect at all? A negative result obtained with a blind instrument
is weaker evidence than the same result obtained with a sighted one, and that
distinction changes what should be pre-registered NEXT -- not what Exp 62
concluded.

TWO SPECIFIC WORRIES, BOTH RAISED BEFORE RUNNING THIS.

  W1 LINEARITY. Every capability number in this project comes from ridge
     regression: a LINEAR map. If placement creates information that is
     nonlinearly decodable, a linear readout cannot see it by construction,
     and "structure does not predict capability" would be a claim about
     linear decodability rather than about capability. Exp 61 established
     that DICHOTOMISATION was not the bottleneck (continuous vs binary gave
     d=0.590 vs 0.575). Linear-vs-nonlinear is a different axis and is
     untested.

  W2 A SATURATING METRIC. The primary metric `rsq` is squared Pearson
     correlation between out-of-fold decision values and +/-1 targets. It is
     bounded [0, 1] and compresses as it rises. The post-hoc saturation curve
     (negative quadratic, interleaved knee at bridge ~8659) was measured in
     that metric. A bounded metric can MANUFACTURE apparent diminishing
     returns from a relationship that is not saturating at all. If the
     curvature vanishes under an UNBOUNDED separation metric, the saturation
     hypothesis is an artifact and must not be pre-registered.

DESIGN. The simulations are untouched -- every number below is re-scored from
the SAME cached feature matrices Exp 61/62 used, with the SAME stratified fold
construction (seed 999, copied verbatim) so all readouts see identical splits.
Only the readout and the metric change.

  READOUTS   ridge      exp61's solve, verbatim. Its `rsq` MUST reproduce
                        Exp 62's published confirm statistics -- that is the
                        gate that makes the other two trustworthy.
             logistic   L2-regularised logistic regression (Newton/IRLS).
                        Still linear, but a proper classification objective
                        rather than regression onto +/-1.
             krr_rbf    kernel ridge with an RBF kernel, bandwidth by the
                        median heuristic on the TRAIN fold only. Genuinely
                        nonlinear. Addresses W1.

  METRICS    rsq        the registered metric. Bounded.  [baseline]
             auc        rank-based. Bounded.
             auc_logit  log(auc/(1-auc)). UNBOUNDED.      addresses W2
             dprime     class-mean separation of decision values in pooled
                        SD units. UNBOUNDED.              addresses W2

All metrics are computed from the SAME out-of-fold decision values per cell,
so readout and metric are cleanly separated.

READ-OFF, FIXED BEFORE RUNNING:
  - P1 stays <= 0 under krr_rbf  -> linearity was not the limitation; Exp 62's
    failure is about the physics, and W1 is closed.
  - P1 turns clearly positive under krr_rbf -> the registered readout was
    partly blind. Exp 62's verdict still stands as reported, but the next
    block must be registered on a nonlinear readout.
  - curvature survives under dprime/auc_logit -> saturation is real and is the
    right thing to pre-register.
  - curvature vanishes under dprime/auc_logit -> saturation was a bounded-
    metric artifact; WITHDRAW it and do not build a third block on it.

Run:  venv/Scripts/python.exe <this file>
"""
import os, sys, json
sys.path.insert(0, "src")
sys.path.insert(0, os.path.join("src", "experiments"))

import numpy as np
from scipy import stats

from exp52_accuracy_endpoint import (
    load_split, PLASTIC_PER_CLASS, EVAL_PER_CLASS, N_EPOCHS,
)
from exp61_continuous_endpoint import (
    ridge_cv_scores, feat_key, _feat_load, N_FOLDS, PRIMARY_FROZEN,
)
from exp62_structure_performance_law import (
    _f_load, BRIDGE_PATH, EXP52B_CACHE, CONDITIONS,
    DISCOVERY_SEEDS, CONFIRM_SEEDS, REGISTERED_LAMBDA,
)

LAM = REGISTERED_LAMBDA          # 1.0, the registered point
EXP62_CONFIRM_QUOTED = {"partial_r": -0.1403, "seg_r": 0.362, "int_r": -0.509}


# ------------------------------------------------------------------- folds
def make_folds(y, n_folds=N_FOLDS, seed=999):
    """exp61's stratified fold construction, copied verbatim."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


# ---------------------------------------------------------------- readouts
def _standardise(Xtr, Xte):
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ro_ridge(Xtr, ttr, Xte, lam=LAM):
    """exp61's solve, verbatim (intercept unpenalised)."""
    Ztr, Zte = _standardise(Xtr, Xte)
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Zte = np.hstack([Zte, np.ones((len(Zte), 1))])
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
    A[-1, -1] -= lam
    w = np.linalg.solve(A, Ztr.T @ ttr)
    return Zte @ w


def ro_logistic(Xtr, ttr, Xte, lam=LAM, iters=60):
    """L2-regularised logistic regression by damped Newton/IRLS.

    Returns the linear logit on the test fold. Labels +/-1 -> {0,1}.
    """
    Ztr, Zte = _standardise(Xtr, Xte)
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Zte = np.hstack([Zte, np.ones((len(Zte), 1))])
    y01 = (ttr > 0).astype(float)
    d = Ztr.shape[1]
    w = np.zeros(d)
    R = lam * np.eye(d); R[-1, -1] = 0.0          # do not penalise intercept
    for _ in range(iters):
        eta = np.clip(Ztr @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1.0 - p), 1e-6, None)
        g = Ztr.T @ (p - y01) + R @ w
        H = Ztr.T @ (Ztr * W[:, None]) + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if not np.all(np.isfinite(w_new)):
            break
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new
    return Zte @ w


def ro_krr_rbf(Xtr, ttr, Xte, lam=LAM):
    """Kernel ridge regression, RBF kernel, median-heuristic bandwidth.

    Bandwidth is computed on the TRAIN fold only -- using test points to set
    it would leak. Genuinely nonlinear in the features.
    """
    Ztr, Zte = _standardise(Xtr, Xte)
    d2 = np.sum((Ztr[:, None, :] - Ztr[None, :, :]) ** 2, axis=2)
    iu = np.triu_indices(len(Ztr), k=1)
    med = np.median(d2[iu])
    gamma = 1.0 / max(med, 1e-9)
    Ktr = np.exp(-gamma * d2)
    d2te = np.sum((Zte[:, None, :] - Ztr[None, :, :]) ** 2, axis=2)
    Kte = np.exp(-gamma * d2te)
    tc = ttr.mean()
    a = np.linalg.solve(Ktr + lam * np.eye(len(Ktr)), ttr - tc)
    return Kte @ a + tc


READOUTS = (("ridge", ro_ridge), ("logistic", ro_logistic),
            ("krr_rbf", ro_krr_rbf))


# ----------------------------------------------------------------- metrics
def _auc(p, t):
    pos, neg = p[t > 0], p[t < 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def metrics_from_scores(p, t):
    p = np.asarray(p, float); t = np.asarray(t, float)
    sd = p.std()
    rsq = 0.0 if sd < 1e-12 else float(np.corrcoef(p, t)[0, 1] ** 2)
    auc = _auc(p, t)
    auc_c = min(max(auc, 1e-3), 1 - 1e-3)
    pos, neg = p[t > 0], p[t < 0]
    pooled = np.sqrt(0.5 * (pos.var(ddof=1) + neg.var(ddof=1)))
    dprime = 0.0 if pooled < 1e-12 else float((pos.mean() - neg.mean()) / pooled)
    return {"rsq": rsq, "auc": auc,
            "auc_logit": float(np.log(auc_c / (1 - auc_c))),
            "dprime": dprime}


METRIC_NAMES = ("rsq", "auc", "auc_logit", "dprime")
UNBOUNDED = {"auc_logit", "dprime"}


def score_cell(X, y, readout):
    """Out-of-fold decision values -> all metrics, one cached feature matrix."""
    X = np.asarray(X, float); y = np.asarray(y, int)
    folds = make_folds(y)
    t = np.where(y == y.min(), -1.0, 1.0)
    p = np.zeros(len(y), float)
    for k in range(N_FOLDS):
        te = folds == k; tr = ~te
        p[te] = readout(X[tr], t[tr], X[te])
    return metrics_from_scores(p, t)


# ------------------------------------------------------------ data assembly
def assemble(labels):
    """B (bridge), D per (readout, metric), G (condition), BLK (block)."""
    disc_store = _feat_load()
    conf_store = _f_load()
    prior = json.load(open(EXP52B_CACHE))
    bridge = json.load(open(BRIDGE_PATH))

    rows = []
    for blk, seeds, store in (("discovery", DISCOVERY_SEEDS, disc_store),
                              ("confirm", CONFIRM_SEEDS, conf_store)):
        for cond, _ in CONDITIONS:
            for s in seeds:
                if blk == "discovery":
                    k52 = "{}|{}|spc{}|ep{}|ev{}|log".format(
                        cond, s, PLASTIC_PER_CLASS, N_EPOCHS, EVAL_PER_CLASS)
                    bm = prior[k52][2]
                else:
                    bm = bridge["{}|{}".format(cond, s)]
                Xp = store[feat_key(cond, s, "plastic")]
                Xf = store[feat_key(cond, s, PRIMARY_FROZEN)]
                cell = {"cond": cond, "blk": blk, "seed": s, "bridge": bm}
                for rname, rfn in READOUTS:
                    mp = score_cell(Xp, labels, rfn)
                    mf = score_cell(Xf, labels, rfn)
                    for m in METRIC_NAMES:
                        cell["{}|{}".format(rname, m)] = mp[m] - mf[m]
                rows.append(cell)
        print("    assembled {:<10} ({} cells)".format(
            blk, sum(1 for r in rows if r["blk"] == blk)), flush=True)
    return rows


# --------------------------------------------------------------- statistics
def boot_partial_r(x, y, group, n_boot=20000, seed=7):
    """Within-group-centred correlation with a vectorised bootstrap."""
    x = np.asarray(x, float).copy(); y = np.asarray(y, float).copy()
    group = np.asarray(group)
    for g in np.unique(group):
        m = group == g
        x[m] -= x[m].mean(); y[m] -= y[m].mean()
    r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, (n_boot, n))
    xb, yb = x[idx], y[idx]
    xb = xb - xb.mean(axis=1, keepdims=True)
    yb = yb - yb.mean(axis=1, keepdims=True)
    num = (xb * yb).sum(axis=1)
    den = np.sqrt((xb ** 2).sum(axis=1) * (yb ** 2).sum(axis=1))
    v = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return float(r), float(p), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def curvature(b, d):
    x = (b - b.mean()) / b.std()
    X = np.column_stack([np.ones_like(x), x, x ** 2])
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    resid = d - X @ beta
    dof = len(d) - 3
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    tq = beta[2] / np.sqrt(cov[2, 2])
    return float(beta[2]), float(tq), float(2 * (1 - stats.t.cdf(abs(tq), dof)))


def extrapolate(b_from, d_from, b_to, d_to):
    sl, ic = np.polyfit(np.log10(np.maximum(b_from, 1.0)), d_from, 1)
    pred = sl * np.log10(max(b_to.mean(), 1.0)) + ic
    sd = d_to.std(ddof=1)
    return pred, float(d_to.mean()), abs(pred - d_to.mean()) / max(sd, 1e-12)


def main():
    top, avail, plastic, evalset = load_split()
    labels = np.array([0] * len(evalset[top[0]]) + [1] * len(evalset[top[1]]))

    print("=" * 100)
    print("DIAGNOSTIC -- was Exp 62 measured with a blind instrument?")
    print("            [DESCRIPTIVE, POST-HOC. Exp 62's P1/P3 verdicts stand regardless.]")
    print("=" * 100)
    print("\n[0] Re-scoring cached features (no simulation): 3 readouts x 4 metrics x 160 cells")
    rows = assemble(labels)

    B = np.array([r["bridge"] for r in rows], float)
    G = np.array([r["cond"] for r in rows])
    BLK = np.array([r["blk"] for r in rows])

    # ---------------------------------------------------------- [P0] gate
    print("\n[P0] GATE -- 'ridge' + 'rsq' must reproduce Exp 62's confirm statistics")
    Dg = np.array([r["ridge|rsq"] for r in rows], float)
    kc = BLK == "confirm"
    r, p, lo, hi = boot_partial_r(B[kc], Dg[kc], G[kc])
    seg = kc & (G == "segregated"); inter = kc & (G == "interleaved")
    rs = stats.pearsonr(B[seg], Dg[seg])[0]
    ri = stats.pearsonr(B[inter], Dg[inter])[0]
    print("    partial r  recomputed {:+.4f}   exp62 {:+.4f}".format(
        r, EXP62_CONFIRM_QUOTED["partial_r"]))
    print("    seg own r  recomputed {:+.3f}    exp62 {:+.3f}".format(
        rs, EXP62_CONFIRM_QUOTED["seg_r"]))
    print("    int own r  recomputed {:+.3f}    exp62 {:+.3f}".format(
        ri, EXP62_CONFIRM_QUOTED["int_r"]))
    ok = (abs(r - EXP62_CONFIRM_QUOTED["partial_r"]) < 2e-3
          and abs(rs - EXP62_CONFIRM_QUOTED["seg_r"]) < 2e-3
          and abs(ri - EXP62_CONFIRM_QUOTED["int_r"]) < 2e-3)
    print("    P0 {}".format(
        "HOLDS -- harness reproduces the published scoring; other readouts trustworthy"
        if ok else "FAILS -- harness disagrees with exp62, STOP"))
    if not ok:
        sys.exit(1)

    # -------------------------------------- [1] W1: does P1 survive nonlinearity
    print("\n" + "=" * 100)
    print("[1] W1 -- LINEARITY. Exp 62's P1 test (CONFIRM block) under each readout x metric")
    print("    registered bar was partial r >= +0.20 with CI excluding 0")
    print("=" * 100)
    print("    {:<10} {:<10} {:>9} {:>20} {:>9} {:>9}".format(
        "readout", "metric", "partial r", "95% CI", "seg r", "int r"))
    for rname, _ in READOUTS:
        for m in METRIC_NAMES:
            D = np.array([r["{}|{}".format(rname, m)] for r in rows], float)
            rr, pp, lo, hi = boot_partial_r(B[kc], D[kc], G[kc])
            rs = stats.pearsonr(B[seg], D[seg])[0]
            ri = stats.pearsonr(B[inter], D[inter])[0]
            flag = "  <-- clears bar" if (rr >= 0.20 and lo > 0) else ""
            print("    {:<10} {:<10} {:>+9.3f} {:>20} {:>+9.3f} {:>+9.3f}{}".format(
                rname, m, rr, "[{:+.3f}, {:+.3f}]".format(lo, hi), rs, ri, flag))

    # ------------------------------- [2] W2: does the saturation curve survive
    print("\n" + "=" * 100)
    print("[2] W2 -- SATURATION. Curvature on pooled data (both blocks), per condition.")
    print("    A bounded metric can manufacture diminishing returns. Compare the")
    print("    BOUNDED metrics (rsq, auc) against the UNBOUNDED ones (auc_logit, dprime).")
    print("=" * 100)
    for rname, _ in READOUTS:
        print("\n  readout '{}'".format(rname))
        print("    {:<10} {:<4} {:<12} {:>10} {:>8} {:>8} {:>10} {:>10}".format(
            "metric", "bnd", "condition", "quad coef", "t", "p", "low-half r", "high-half r"))
        for m in METRIC_NAMES:
            D = np.array([r["{}|{}".format(rname, m)] for r in rows], float)
            for cond, _ in CONDITIONS:
                mk = G == cond
                b, d = B[mk], D[mk]
                co, tq, pq = curvature(b, d)
                med = np.median(b)
                rlo = stats.pearsonr(b[b < med], d[b < med])[0]
                rhi = stats.pearsonr(b[b >= med], d[b >= med])[0]
                print("    {:<10} {:<4} {:<12} {:>10.5f} {:>8.2f} {:>8.3f} {:>10.3f} {:>10.3f}"
                      .format(m, "no" if m in UNBOUNDED else "YES", cond,
                              co, tq, pq, rlo, rhi))

    # ------------------------------------------ [3] P3 under each instrument
    print("\n" + "=" * 100)
    print("[3] Exp 62's P3 (cross-condition extrapolation), CONFIRM block, per instrument")
    print("    exp62 registered failure at > 1.0 SD; it measured 2.19 SD one way")
    print("=" * 100)
    print("    {:<10} {:<10} {:>26} {:>26}".format(
        "readout", "metric", "int->seg (|err| SD)", "seg->int (|err| SD)"))
    for rname, _ in READOUTS:
        for m in METRIC_NAMES:
            D = np.array([r["{}|{}".format(rname, m)] for r in rows], float)
            bs, ds = B[seg], D[seg]
            bi, di = B[inter], D[inter]
            _, _, e1 = extrapolate(bi, di, bs, ds)
            _, _, e2 = extrapolate(bs, ds, bi, di)
            print("    {:<10} {:<10} {:>26} {:>26}".format(
                rname, m,
                "{:.2f}  {}".format(e1, "FAIL" if e1 > 1.0 else "pass"),
                "{:.2f}  {}".format(e2, "FAIL" if e2 > 1.0 else "pass")))

    # ------------------------------------------------- [4] absolute capability
    print("\n" + "=" * 100)
    print("[4] CONTEXT -- absolute delta-metric means, to see if a readout decodes better")
    print("=" * 100)
    print("    {:<10} {:<10} {:>14} {:>14} {:>9}".format(
        "readout", "metric", "segregated", "interleaved", "ratio"))
    for rname, _ in READOUTS:
        for m in METRIC_NAMES:
            D = np.array([r["{}|{}".format(rname, m)] for r in rows], float)
            a, b_ = D[seg].mean(), D[inter].mean()
            print("    {:<10} {:<10} {:>14.4f} {:>14.4f} {:>9}".format(
                rname, m, a, b_,
                "{:.2f}x".format(b_ / a) if abs(a) > 1e-9 else "n/a"))

    print("\n" + "=" * 100)
    print("[5] READ-OFF (criteria fixed before running -- see module docstring)")
    print("=" * 100)
    print("    W1: compare krr_rbf's P1 partial r in [1] against ridge's. If it is")
    print("        still <= 0, linearity was NOT the limitation and W1 closes.")
    print("    W2: compare curvature t under dprime/auc_logit against rsq in [2].")
    print("        If curvature vanishes when unbounded, WITHDRAW saturation.")
    print("    Exp 62's P1 and P3 verdicts are unchanged by anything above.")


if __name__ == "__main__":
    main()
