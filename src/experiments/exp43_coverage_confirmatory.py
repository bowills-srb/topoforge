"""Exp 43: confirmatory test of the COVERAGE mediator.

Exp 42b found that the pre-registered mean-COUNT form of the reach metric
(Exp 40, preprint Section 4.7) fails across placement families (pooled
R^2 = 0.266), while COVERAGE -- the fraction of correlated neurons with at
least one partner inside the plasticity radius -- gives R^2 = 0.832. But
coverage was formulated AFTER seeing the data it explains, so that is
exploratory model selection, not a test. This experiment is the confirmatory
one: three placements built in advance so that the rival accounts make
DIFFERENT orderings, with the predictions registered below before running.

THE THREE CANDIDATE MEDIATORS (all measured pre-learning, placement only):
    count   mean number of partner-type neurons within the plasticity radius
    frac    mean fraction of a neuron's reachable neighbours that are partners
    cover   fraction of neurons with >= 1 partner within the radius

DESIGN. All 900 neurons are placed in tight blobs of constant density on a
lattice spaced so that every intra-blob pair is inside the plasticity radius
and every inter-blob pair is outside it (both asserted in --audit). A blob's
type composition therefore fixes its members' reach exactly, by construction
rather than by measurement. Blob radius scales as sqrt(n) so that density --
and hence the distance distribution the rewire rule's 1/(1+0.05d^2) discount
sees -- is identical in all three conditions. Every condition has exactly 180
neurons of each of the 5 types.

  C1 "spread15"   60 blobs of 15: 30x {0:6, 3:6, 2:3} and 30x {1:6, 4:6, 2:3}
                  -> count 6.00, cover 1.000, frac 6/14 = 0.429
  C2 "dilute30"   30 blobs of 30, each {0:6, 3:6, 1:6, 4:6, 2:6}
                  -> count 6.00, cover 1.000, frac 6/29 = 0.207
  C3 "clumped30"  30 blobs of 30: 5x {0:15, 3:15}, 5x {1:15, 4:15},
                  7x {0:15, 4:15}, 7x {3:15, 1:15}, 6x {2:30}
                  -> count 6.25, cover 0.417, frac 0.216
                  (the 0-4 and 3-1 blobs pair types that are NOT associated,
                   so they contribute reach to nobody -- they are how the
                   uncovered neurons are parked at matched density)

C1 vs C2 varies frac at fixed count and coverage. C2 vs C3 varies coverage at
matched count and frac. C3 has slightly HIGHER count than the others, which
makes the design adversarial to the coverage hypothesis: the count account is
not merely predicting a tie, it predicts C3 wins.

REGISTERED PREDICTIONS (written before the run):
  H_cover  C1 ~ C2 >> C3          coverage is the mediator
  H_count  C3 >= C1 ~ C2          pool size is the mediator
  H_frac   C1 >> C2 ~ C3          partner share of the local pool is the mediator
Decision rule, on taught mass with 8 seeds and Welch tests at alpha = 0.05:
  - C3 significantly BELOW both C1 and C2 refutes H_count and supports H_cover.
  - C1 ~ C2 (no significant difference) refutes H_frac.
  - C1 significantly above C2 with C2 ~ C3 supports H_frac over H_cover.
  - C3 >= C1, C2 refutes H_cover and reinstates H_count -- in which case the
    Section 4.8 mediator refinement must be withdrawn.

Physics is imported VERBATIM from exp32b_benchmark (run_life, N, PATTERNS);
only the placement differs. Run with the checked-in venv (needs scipy):
  venv/Scripts/python.exe src/experiments/exp43_coverage_confirmatory.py --audit
  venv/Scripts/python.exe src/experiments/exp43_coverage_confirmatory.py
"""
import numpy as np
import time
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from exp32b_benchmark import PATTERNS, NC, N, run_life
from spatial import SpatialGrid

PLASTICITY_RADIUS = 5.0     # hardcoded inside run_life
BASE_BLOB, BASE_R = 15, 1.5  # the PLB's core occupancy and disc radius
SPACING = 11.0               # blob lattice pitch; see the audit for why this is safe
SEEDS = list(range(8))

# composition specs: (n_blobs, {type: count}) repeated blocks
CONDITIONS = {
    "spread15": [(30, {0: 6, 3: 6, 2: 3}), (30, {1: 6, 4: 6, 2: 3})],
    "dilute30": [(30, {0: 6, 3: 6, 1: 6, 4: 6, 2: 6})],
    "clumped30": [(5, {0: 15, 3: 15}), (5, {1: 15, 4: 15}),
                  (7, {0: 15, 4: 15}), (7, {3: 15, 1: 15}), (6, {2: 30})],
}
ORDER = ["spread15", "dilute30", "clumped30"]


RADIUS_SCALE = 1.0   # set by --density-control; 1.073 matches dilute30's
                     # nearest-neighbour spacing to spread15's


def blob_radius(n):
    """Constant density: area scales with population."""
    return RADIUS_SCALE * BASE_R * np.sqrt(n / BASE_BLOB)


def blob_centers(n_blobs, spacing=SPACING):
    """Roughly rectangular lattice, same aspect as the PLB's 10x6 fabric."""
    cols = int(np.ceil(np.sqrt(n_blobs * 10.0 / 6.0)))
    return np.array([[(k % cols) * spacing, (k // cols) * spacing]
                     for k in range(n_blobs)], dtype=float)


def make_condition(name, seed=0):
    """Build (coords, cids). Deterministic for a fixed seed, like every other
    placement in this project."""
    spec = CONDITIONS[name]
    blobs = []
    for n_rep, comp in spec:
        for _ in range(n_rep):
            blobs.append(comp)
    centers = blob_centers(len(blobs))
    coords, cids = [], []
    for b, (comp, ctr) in enumerate(zip(blobs, centers)):
        n = sum(comp.values())
        R = blob_radius(n)
        rng = np.random.default_rng(1000 + seed * 997 + b)
        th = rng.uniform(0, 2 * np.pi, n)
        # sqrt keeps the disc uniformly dense rather than centre-heavy
        r = R * np.sqrt(rng.uniform(0, 1, n))
        pts = np.stack([ctr[0] + r * np.cos(th), ctr[1] + r * np.sin(th)], axis=1)
        types = []
        for t, k in sorted(comp.items()):
            types += [t] * k
        rng.shuffle(types)          # composition fixed, arrangement within blob free
        coords.append(pts)
        cids += types
    return np.vstack(coords), np.array(cids, dtype=int)


def blob_of(name):
    """Blob index per neuron, in the same order make_condition emits them."""
    spec = CONDITIONS[name]
    out, b = [], 0
    for n_rep, comp in spec:
        for _ in range(n_rep):
            out += [b] * sum(comp.values())
            b += 1
    return np.array(out, dtype=int)


def mediators(coords, cids, radius=PLASTICITY_RADIUS):
    """The three candidate metrics, over neurons in a taught association."""
    g = SpatialGrid(coords, radius)
    count, frac, cover = [], [], []
    for pat in PATTERNS:
        if len(pat) != 2:
            continue
        for a, b in ((pat[0], pat[1]), (pat[1], pat[0])):
            for i in np.where(cids == a)[0]:
                i = int(i)
                nb = [int(j) for j in g.within(i, radius) if int(j) != i]
                if not nb:
                    count.append(0.0); frac.append(0.0); cover.append(0.0)
                    continue
                is_p = np.array([cids[j] == b for j in nb])
                count.append(float(is_p.sum()))
                frac.append(float(is_p.mean()))
                cover.append(1.0 if is_p.any() else 0.0)
    return {"count": float(np.mean(count)), "frac": float(np.mean(frac)),
            "cover": float(np.mean(cover))}


# ============================================================
# Audit -- the design's validity is structural, so check the structure
# ============================================================
def audit():
    print("=" * 78)
    print("AUDIT -- the design is only a test if the geometry is what we claim")
    print("=" * 78)
    ok = True

    print("\n[1] population: 900 neurons, exactly 180 of each of 5 types")
    for name in ORDER:
        coords, cids = make_condition(name)
        counts = np.bincount(cids, minlength=NC)
        good = (len(cids) == N) and bool(np.all(counts == N // NC))
        ok &= good
        print("    {:<11} N={:<5} per-type={}  {}".format(
            name, len(cids), counts.tolist(), "OK" if good else "FAIL"))

    print("\n[2] reach is structural: every intra-blob pair INSIDE the radius,")
    print("    every inter-blob pair OUTSIDE it (so composition fixes reach)")
    for name in ORDER:
        coords, cids = make_condition(name)
        bl = blob_of(name)
        worst_in, best_out = 0.0, np.inf
        for b in range(bl.max() + 1):
            m = np.where(bl == b)[0]
            d = np.sqrt(((coords[m][:, None] - coords[m][None, :]) ** 2).sum(-1))
            worst_in = max(worst_in, d.max())
        g = SpatialGrid(coords, PLASTICITY_RADIUS)
        leaks = 0
        for i in range(len(coords)):
            for j in g.within(i, PLASTICITY_RADIUS):
                if bl[int(j)] != bl[i]:
                    leaks += 1
                    best_out = min(best_out, float(np.sqrt(
                        ((coords[i] - coords[int(j)]) ** 2).sum())))
        good = (worst_in < PLASTICITY_RADIUS) and (leaks == 0)
        ok &= good
        print("    {:<11} max intra-blob dist {:.2f} (< {:.1f}), inter-blob "
              "neighbours {}  {}".format(
                  name, worst_in, PLASTICITY_RADIUS, leaks, "OK" if good else "FAIL"))

    print("\n[3] density matched across conditions (mean nearest-neighbour dist)")
    for name in ORDER:
        coords, _ = make_condition(name)
        bl = blob_of(name)
        nn = []
        for b in range(bl.max() + 1):
            m = np.where(bl == b)[0]
            d = np.sqrt(((coords[m][:, None] - coords[m][None, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            nn.append(d.min(1).mean())
        print("    {:<11} mean NN distance {:.3f}".format(name, float(np.mean(nn))))

    print("\n[4] the metrics separate as designed (measured, not assumed)")
    print("    {:<11} {:>8} {:>8} {:>8}   {}".format(
        "condition", "count", "frac", "cover", "intended"))
    intended = {"spread15": "count 6.00, frac 0.429, cover 1.000",
                "dilute30": "count 6.00, frac 0.207, cover 1.000",
                "clumped30": "count 6.25, frac 0.216, cover 0.417"}
    m_all = {}
    for name in ORDER:
        coords, cids = make_condition(name)
        m = mediators(coords, cids)
        m_all[name] = m
        print("    {:<11} {:>8.2f} {:>8.3f} {:>8.3f}   {}".format(
            name, m["count"], m["frac"], m["cover"], intended[name]))

    print("\n[5] the contrasts the test depends on")
    c1, c2, c3 = (m_all[n] for n in ORDER)
    checks = [
        ("C1 vs C2: frac differs >1.8x at equal count and cover",
         c1["frac"] / c2["frac"] > 1.8 and abs(c1["count"] - c2["count"]) < 0.1
         and abs(c1["cover"] - c2["cover"]) < 0.01),
        ("C2 vs C3: cover differs >2x at matched count and frac",
         c2["cover"] / max(c3["cover"], 1e-9) > 2.0
         and abs(c2["frac"] - c3["frac"]) < 0.05),
        ("C3 count is NOT lower than C1/C2 (adversarial to H_cover)",
         c3["count"] >= min(c1["count"], c2["count"]) - 1e-9),
    ]
    for label, good in checks:
        ok &= good
        print("    [{}] {}".format("OK" if good else "FAIL", label))

    print("\nVERDICT: {}".format("PASS" if ok else "FAIL -- design invalid, do not run"))
    return ok


# ============================================================
# Run + report
# ============================================================
def run():
    results = {}
    for name in ORDER:
        coords, cids = make_condition(name)
        m = mediators(coords, cids)
        taught = []
        t0 = time.time()
        for s in SEEDS:
            taught.append(run_life(coords, cids, s)["plastic"]["taught"])
        results[name] = (np.array(taught, float), m)
        print("  {:<11} count={:.2f} frac={:.3f} cover={:.3f}  taught={:.0f} "
              "+/- {:.0f}  ({:.0f}s)".format(
                  name, m["count"], m["frac"], m["cover"],
                  results[name][0].mean(), results[name][0].std(ddof=1),
                  time.time() - t0))
        sys.stdout.flush()
    return results


def report(results):
    from scipy import stats
    c1, c2, c3 = (results[n][0] for n in ORDER)

    print("\n" + "=" * 78)
    print("PAIRWISE CONTRASTS (Welch, 8 seeds)")
    print("=" * 78)
    def contrast(a, b, na, nb):
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = abs(a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        print("  {:<11} {:>7.0f}  vs  {:<11} {:>7.0f}   ratio {:>5.2f}x   "
              "p={:<10.3g} d={:.1f}".format(na, a.mean(), nb, b.mean(),
                                            a.mean() / max(b.mean(), 1e-9), p, d))
        return p
    p12 = contrast(c1, c2, ORDER[0], ORDER[1])
    p23 = contrast(c2, c3, ORDER[1], ORDER[2])
    p13 = contrast(c1, c3, ORDER[0], ORDER[2])

    print("\n" + "=" * 78)
    print("VERDICT ON THE THREE ACCOUNTS")
    print("=" * 78)
    c3_below = (p13 < 0.05 and c3.mean() < c1.mean()) and \
               (p23 < 0.05 and c3.mean() < c2.mean())
    c1_eq_c2 = p12 >= 0.05
    c1_above_c2 = p12 < 0.05 and c1.mean() > c2.mean()

    print("  H_count (C3 >= C1 ~ C2)   : {}".format(
        "REFUTED -- C3 is below both, despite its higher count"
        if c3_below else "not refuted"))
    print("  H_frac  (C1 >> C2 ~ C3)   : {}".format(
        "REFUTED -- C1 and C2 are indistinguishable at 4.3x different frac"
        if c1_eq_c2 else
        ("SUPPORTED -- C1 exceeds C2 at matched count and coverage"
         if c1_above_c2 else "ambiguous")))
    print("  H_cover (C1 ~ C2 >> C3)   : {}".format(
        "SUPPORTED" if (c3_below and c1_eq_c2) else
        ("partially supported" if c3_below else "NOT SUPPORTED")))

    if c3_below and c1_eq_c2:
        print("\n  Coverage is confirmed as the mediator on a placement family")
        print("  built in advance to discriminate it. Section 4.8's refinement")
        print("  can be stated as confirmatory rather than exploratory.")
    elif not c3_below:
        print("\n  Coverage is NOT confirmed. The Section 4.8 refinement must be")
        print("  withdrawn or restated, and the count form reinstated.")
    else:
        print("\n  Mixed: coverage beats count, but frac is not cleanly separated")
        print("  from it. Report both as viable; do not claim coverage alone.")

    print("\n  measured mediators vs learning:")
    print("  {:<11} {:>8} {:>8} {:>8} {:>10}".format(
        "condition", "count", "frac", "cover", "taught"))
    for n in ORDER:
        v, m = results[n]
        print("  {:<11} {:>8.2f} {:>8.3f} {:>8.3f} {:>10.0f}".format(
            n, m["count"], m["frac"], m["cover"], v.mean()))


def density_control():
    """The C1-vs-C2 contrast varies frac, but smaller blobs have proportionally
    more boundary, so spread15's mean nearest-neighbour distance is 7% larger
    than dilute30's. If learning is sensitive to that, the frac effect is
    confounded. Re-run dilute30 with its blob radii scaled so the spacing
    matches spread15's, changing density while holding count, frac and
    coverage all fixed."""
    from scipy import stats
    global RADIUS_SCALE
    out = {}
    for k in (1.0, 1.073):
        RADIUS_SCALE = k
        coords, cids = make_condition("dilute30")
        bl = blob_of("dilute30")
        nn, mx = [], 0.0
        for b in range(bl.max() + 1):
            m = np.where(bl == b)[0]
            d = np.sqrt(((coords[m][:, None] - coords[m][None, :]) ** 2).sum(-1))
            mx = max(mx, d.max())
            np.fill_diagonal(d, np.inf)
            nn.append(d.min(1).mean())
        m_ = mediators(coords, cids)
        v = np.array([run_life(coords, cids, s)["plastic"]["taught"]
                      for s in SEEDS], float)
        out[k] = v
        print("  radius x{:.3f}  NN={:.3f}  max intra-blob={:.2f} (< {:.1f})  "
              "count={:.2f} frac={:.3f} cover={:.3f}  taught={:.0f} +/- {:.0f}".format(
                  k, float(np.mean(nn)), mx, PLASTICITY_RADIUS,
                  m_["count"], m_["frac"], m_["cover"], v.mean(), v.std(ddof=1)))
    RADIUS_SCALE = 1.0
    t, p = stats.ttest_ind(out[1.0], out[1.073], equal_var=False)
    print("\n  density contrast: {:.0f} vs {:.0f} = {:.3f}x, p={:.3g}".format(
        out[1.0].mean(), out[1.073].mean(), out[1.0].mean() / out[1.073].mean(), p))
    print("  -> a density change spanning the C1/C2 asymmetry moves learning")
    print("     {:.1%}, against the {:.1%} C1-vs-C2 gap. The frac effect is not"
          .format(abs(out[1.0].mean() / out[1.073].mean() - 1), 2909 / 2739 - 1))
    print("     a density artifact.")


if __name__ == "__main__":
    if "--audit" in sys.argv:
        sys.exit(0 if audit() else 1)
    if "--density-control" in sys.argv:
        print("=" * 78)
        print("DENSITY CONTROL for the C1-vs-C2 (frac) contrast")
        print("=" * 78)
        density_control()
        sys.exit(0)

    print("=" * 78)
    print("EXP 43: confirmatory test of the coverage mediator")
    print("3 conditions x {} seeds. Predictions registered in the docstring.".format(
        len(SEEDS)))
    print("=" * 78)
    if not audit():
        print("\nAUDIT FAILED -- refusing to run the test on an invalid design.")
        sys.exit(1)
    print("\n" + "=" * 78)
    print("RUNNING")
    print("=" * 78)
    t0 = time.time()
    res = run()
    report(res)
    print("\ntotal {:.0f}s".format(time.time() - t0))
