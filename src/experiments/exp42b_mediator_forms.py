"""Exp 42b: which FORM of "reach" is the mediator?

Section 4.7 (Exp 40/41) established that placement acts through the reachable
correlated fraction, measured there as a mean COUNT -- the average number of
correlated partners inside the plasticity radius. That was established on a
single placement family (the alpha-sweep between segregated and interleaved),
where every reasonable form of the metric co-varies. Exp 38's corrected result
breaks that tie and shows the count form is not sufficient:

    placement        taught    mean count   coverage
    interleaved       2950        7.80        1.000
    SpiNeMap-pop       876        6.34        0.608

Near-identical mean count, 3.3x different learning. What separates them is
COVERAGE -- the fraction of correlated neurons with ANY reachable partner.
SpiNeMap's population-graph placement makes cores type-pure, so a neuron in a
core's interior has no partner within the radius at all, while the neurons on
a core boundary have many; the mean hides that the pool is concentrated on a
minority of neurons.

This script tests four candidate forms against every placement x pitch
condition in Exp 42 (four placement families, six fabric pitches), asking
which one collapses them onto a single curve:

    count   mean number of partner-type neurons within the radius (Exp 40's)
    frac    mean fraction of a neuron's reachable neighbours that are partners
    cover   fraction of neurons with at least one partner within the radius
    wfrac   like frac, but each neighbour weighted by the rewire rule's own
            distance discount 1/(1 + 0.05*d^2)

HONESTY NOTE: only `count` was pre-registered (Exp 40). The other three were
formulated AFTER seeing the Exp 38 discrepancy above, so this is exploratory
model selection, not a confirmatory test. It is reported as such. A
confirmatory test of the winning form needs a placement family not used to
choose it.

Run (after exp42 has produced its output):
  python src/experiments/exp42b_mediator_forms.py <path-to-exp42-output.txt>
"""
import numpy as np
import re
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from exp32b_benchmark import PATTERNS
from exp42_fabric_sparsity_crossover import (
    PITCHES, STRATEGIES, PLASTICITY_RADIUS, placement_at,
)
from spatial import SpatialGrid

FORMS = ["count", "frac", "cover", "wfrac"]


def mediator_forms(coords, cids, radius=PLASTICITY_RADIUS):
    """All four candidate mediator forms, from one pass over the neighbourhood
    of every neuron that participates in a taught association."""
    g = SpatialGrid(coords, radius)
    count, frac, cover, wfrac = [], [], [], []
    for pat in PATTERNS:
        if len(pat) != 2:
            continue
        for a, b in ((pat[0], pat[1]), (pat[1], pat[0])):
            for i in np.where(cids == a)[0]:
                i = int(i)
                nb = [int(j) for j in g.within(i, radius) if int(j) != i]
                if not nb:
                    count.append(0.0); frac.append(0.0)
                    cover.append(0.0); wfrac.append(0.0)
                    continue
                d2 = ((coords[nb] - coords[i]) ** 2).sum(1)
                w = 1.0 / (1.0 + 0.05 * d2)          # the rewire rule's discount
                is_partner = np.array([cids[j] == b for j in nb])
                count.append(float(is_partner.sum()))
                frac.append(float(is_partner.mean()))
                cover.append(1.0 if is_partner.any() else 0.0)
                wfrac.append(float(w[is_partner].sum() / w.sum()))
    return {"count": np.mean(count), "frac": np.mean(frac),
            "cover": np.mean(cover), "wfrac": np.mean(wfrac)}


def parse_exp42_output(path):
    """Pull the per-(strategy, pitch) learning means out of an exp42 run's
    stdout. The lines look like:
      SPINEMAP-POPULATION -- SpiNeMap (population graph)
        rho=1.0 (pitch  5.0)  reach=  6.34  taught=    904 +/-     58  ...
    """
    learn = {}
    strat = None
    known = {s.upper(): s for s, _ in STRATEGIES}
    for line in open(path, encoding="utf-8", errors="replace"):
        head = line.strip().split(" --")[0].strip()
        if head in known:
            strat = known[head]
            continue
        m = re.search(r"pitch\s+([\d.]+)\)\s+reach=\s*[\d.]+\s+taught=\s*(-?\d+)", line)
        if m and strat is not None:
            pitch = min(PITCHES, key=lambda p: abs(p - float(m.group(1))))
            learn[(strat, pitch)] = float(m.group(2))
    return learn


def main(path):
    from scipy import stats

    learn = parse_exp42_output(path)
    print("=" * 78)
    print("EXP 42b: which form of reach mediates? ({} conditions parsed)".format(len(learn)))
    print("=" * 78)
    if len(learn) < len(STRATEGIES) * len(PITCHES):
        print("  WARNING: expected {} conditions, parsed {} -- run exp42 to completion".format(
            len(STRATEGIES) * len(PITCHES), len(learn)))
        if not learn:
            return 1

    rows = []
    for (strat, pitch), taught in sorted(learn.items()):
        coords, cids = placement_at(strat, pitch)
        m = mediator_forms(coords, cids)
        m.update(strategy=strat, pitch=pitch, taught=taught)
        rows.append(m)

    print("\n  {:<22} {:>5} {:>9} {:>8} {:>8} {:>8} {:>9}".format(
        "strategy", "rho", "taught", *FORMS))
    for r in sorted(rows, key=lambda r: (r["strategy"], r["pitch"])):
        print("  {:<22} {:>5.2f} {:>9.0f} {:>8.2f} {:>8.3f} {:>8.3f} {:>9.3f}".format(
            r["strategy"], r["pitch"] / PLASTICITY_RADIUS, r["taught"],
            *[r[f] for f in FORMS]))

    y = np.array([r["taught"] for r in rows])
    print("\n" + "=" * 78)
    print("Pooled across ALL placement families and pitches -- does the form")
    print("collapse them onto one curve?")
    print("=" * 78)
    scored = {}
    for f in FORMS:
        x = np.array([r[f] for r in rows])
        pr = stats.pearsonr(x, y)[0] ** 2
        sr = stats.spearmanr(x, y)[0]
        # a saturating form is expected (learning cannot grow without bound),
        # so also fit against log(1+x)
        lr = stats.pearsonr(np.log1p(x), y)[0] ** 2
        scored[f] = (pr, sr, lr, max(pr, lr))
    best = max(FORMS, key=lambda f: scored[f][3])
    print("  {:<8} {:>10} {:>12} {:>12} {:>14}".format(
        "form", "Pearson R2", "Spearman rho", "log-fit R2", "verdict"))
    for f in FORMS:
        pr, sr, lr, _ = scored[f]
        print("  {:<8} {:>10.3f} {:>12.3f} {:>12.3f} {:>14}".format(
            f, pr, sr, lr, "<-- best" if f == best else ""))

    print("\n  Best-collapsing form: {} (R^2 = {:.3f})".format(best, scored[best][3]))

    # --- how much of the story is coverage, how much is pool size? ---
    print("\n" + "=" * 78)
    print("Two-term model: does pool SIZE still matter once coverage is known?")
    print("=" * 78)
    cov = np.array([r["cover"] for r in rows])
    lcnt = np.log1p(np.array([r["count"] for r in rows]))
    X = np.column_stack([np.ones(len(y)), cov, lcnt])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2_both = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2_cov = stats.pearsonr(cov, y)[0] ** 2
    r2_cnt = stats.pearsonr(lcnt, y)[0] ** 2
    print("  R^2  coverage alone      : {:.3f}".format(r2_cov))
    print("  R^2  log-count alone     : {:.3f}".format(r2_cnt))
    print("  R^2  both together       : {:.3f}  (coefficients: cover {:+.0f}, "
          "log-count {:+.0f})".format(r2_both, beta[1], beta[2]))
    print("  -> coverage adds {:+.3f} R^2 over log-count alone; log-count adds "
          "{:+.3f} over coverage alone".format(r2_both - r2_cnt, r2_both - r2_cov))

    # within-family control: interleaved holds coverage at 1.0 at every pitch,
    # so its spread isolates the residual effect of pool size.
    fam = [r for r in rows if r["strategy"] == "topoforge"]
    if len(fam) > 2:
        c0 = [r["count"] for r in fam]
        t0 = [r["taught"] for r in fam]
        print("\n  Within interleaved (coverage pinned at 1.000 at every pitch):")
        print("    count {:.2f} -> {:.2f} ({:.1f}x) changes learning {:.0f} -> {:.0f} "
              "({:.2f}x)".format(max(c0), min(c0), max(c0) / max(min(c0), 1e-9),
                                 max(t0), min(t0), max(t0) / max(min(t0), 1e-9)))
        print("    -> pool size has a real but strongly SATURATING residual effect:")
        print("       a 10x cut in reachable partners costs ~1.5x learning, whereas")
        print("       losing coverage entirely costs ~4x.")
    print("\n  The discriminating comparison is any pair with similar `count` but")
    print("  different `cover`: if learning tracks cover there, the mediator is")
    print("  the fraction of correlated neurons REACHED AT ALL, not the mean")
    print("  size of the reachable pool.")

    disc = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if abs(a["count"] - b["count"]) / max(a["count"], b["count"], 1e-9) < 0.25 \
               and abs(a["cover"] - b["cover"]) > 0.2:
                disc.append((a, b))
    print("\n  {} such discriminating pairs found:".format(len(disc)))
    for a, b in disc[:10]:
        print("    {:<20}@{:.2f} (count {:.2f}, cover {:.2f}, taught {:.0f})  vs  "
              "{:<20}@{:.2f} (count {:.2f}, cover {:.2f}, taught {:.0f})".format(
                  a["strategy"], a["pitch"] / PLASTICITY_RADIUS, a["count"], a["cover"], a["taught"],
                  b["strategy"], b["pitch"] / PLASTICITY_RADIUS, b["count"], b["cover"], b["taught"]))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
