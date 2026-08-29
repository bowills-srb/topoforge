"""Exp 44: is there actually an energy-learnability tradeoff?

The preprint's Discussion currently says the two objectives "are not merely
independent but can trade off directly." That sentence has never been measured.
Every energy number in the paper is either a frozen-phase aside (Section 4.1)
or a placement-only proxy (Section 4.8's association-pair distance). What has
not been reported is the quantity a hardware designer actually pays: the
communication cost of the network's REALIZED connections, after plasticity has
rewired them, for each placement -- against what that placement learned.

That is what this measures. run_life already records both, at all three phase
checkpoints:
    energy = sum of squared distance over the current edge set
    taught = cross-type bridge mass
so the frontier is a matter of reporting them together rather than new physics.

WHAT MAKES THE ANSWER NON-OBVIOUS. The rewiring rule scores candidate pairs by
(V + 0.01*C) / (1 + 0.05*d^2) -- it already prefers nearby pairs. So a placement
that puts correlated neurons within reach does not merely learn more; it may
also build its new connections out of SHORTER wires than a placement that has
to reach further for whatever it can find. If so the two objectives are aligned
rather than opposed, and the Discussion is wrong in the paper's own favour.

A structural fact worth stating before the numbers, because it makes one
comparison exact rather than statistical: in the PLB, the segregated,
interleaved and random conditions place neurons at the SAME coordinates and
differ only in which type label sits at each position. The initial connectivity
is also seeded identically. Their frozen-phase energy is therefore identical by
construction, not merely similar -- checked in --audit. The SpiNeMap conditions
DO differ geometrically (clustering assigns different neurons to different
cores), so their frozen energy differs and is reported separately.

PREDICTIONS (registered before running):
  (P1) Frozen energy is bit-identical across segregated / interleaved / random.
       If not, the "zero inference-energy cost" claim in Section 4.1 needs
       restating, and every energy comparison below is confounded.
  (P2) After plasticity, interleaved does NOT cost more energy than segregated.
       Falsified if interleaved's post-plastic energy exceeds segregated's by
       more than seed noise -- which would mean the paper's tradeoff sentence
       is right and the placement recommendation carries a real energy price.
  (P3) SpiNeMap-functional, which co-locates the populations that must
       associate, is at or below interleaved on energy while at or above it on
       learning -- i.e. it dominates rather than trades off.

Run with the checked-in venv (needs scipy):
  venv/Scripts/python.exe src/experiments/exp44_energy_frontier.py --audit
  venv/Scripts/python.exe src/experiments/exp44_energy_frontier.py

METRIC CORRECTION (2026-08-29, PROJECT_HISTORY gotcha #11): raw "taught"
mass is diluted by a shared non-local random-initialization baseline (see
exp32b_benchmark.py docstring). This script already records taught at the
frozen checkpoint for every condition, so growth = plastic_taught -
frozen_taught (plasticity-attributable) is now reported alongside it
throughout, as the corrected primary learning quantity for the "E per
learned unit" and P3 dominance comparisons. Energy (P1/P2) is unaffected --
it was never taught-metric-dependent.
"""
import numpy as np
import time
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from exp32b_benchmark import make_placement, run_life, STEPS_FROZEN
from exp38_spinemap_baseline import make_placement_spinemap
from exp42_fabric_sparsity_crossover import plb_reach

SEEDS = list(range(8))
PHASES = ["frozen", "plastic", "reversal"]
CONDITIONS = [
    ("vlsi", "segregated (ours)"),
    ("spinemap-population", "SpiNeMap, map-time graph"),
    ("random", "random"),
    ("topoforge", "interleaved (ours)"),
    ("spinemap-functional", "SpiNeMap, associations declared"),
]


def placement(name):
    if name.startswith("spinemap-"):
        return make_placement_spinemap(name.split("-", 1)[1], seed=0)
    return make_placement(name)


def audit():
    print("=" * 78)
    print("AUDIT -- what is comparable to what, before any number is trusted")
    print("=" * 78)
    ok = True

    print("\n[1] P1: segregated / interleaved / random occupy IDENTICAL positions")
    print("    (they differ only in which type label sits where), so any energy")
    print("    difference between them can only come from learned connections.")
    base = None
    for name in ("vlsi", "topoforge", "random"):
        c, ci = placement(name)
        if base is None:
            base, base_name = c, name
        else:
            same = np.array_equal(np.sort(c, axis=0), np.sort(base, axis=0))
            exact = np.array_equal(c, base)
            ok &= exact
            print("    {:<10} coords identical to {}: {}".format(name, base_name, exact))

    print("\n[2] SpiNeMap conditions DO differ geometrically -- reported separately")
    for name in ("spinemap-population", "spinemap-functional"):
        c, ci = placement(name)
        same = np.array_equal(c, base)
        print("    {:<20} coords identical to vlsi: {}  (expected False)".format(
            name, same))

    print("\n[3] pre-learning structure of each condition")
    print("    {:<22} {:>8} {:>16}".format("condition", "reach", "mean type/core"))
    # core membership must come from GEOMETRY: SpiNeMap assigns clusters to
    # cores, so its neurons are not contiguous index blocks and slicing by
    # index would report a meaningless 1.00 for every SpiNeMap condition.
    from exp42_fabric_sparsity_crossover import core_of
    for name, _ in CONDITIONS:
        c, ci = placement(name)
        owner = core_of(c)
        per_core = [len(set(ci[owner == k].tolist())) for k in range(owner.max() + 1)]
        print("    {:<22} {:>8.2f} {:>16.2f}".format(
            name, plb_reach(c, ci), float(np.mean(per_core))))

    print("\nVERDICT: {}".format("PASS" if ok else "FAIL -- P1 violated"))
    return ok


def run():
    results = {}
    for name, desc in CONDITIONS:
        coords, cids = placement(name)
        rows = {p: {"energy": [], "taught": []} for p in PHASES}
        t0 = time.time()
        for s in SEEDS:
            r = run_life(coords, cids, s)
            for p in PHASES:
                rows[p]["energy"].append(r[p]["energy"])
                rows[p]["taught"].append(r[p]["taught"])
        results[name] = {p: {k: np.array(v, float) for k, v in d.items()}
                         for p, d in rows.items()}
        print("  {:<22} frozen E={:>10.3g}  plastic E={:>10.3g}  taught={:>6.0f}"
              "  ({:.0f}s)".format(
                  name, results[name]["frozen"]["energy"].mean(),
                  results[name]["plastic"]["energy"].mean(),
                  results[name]["plastic"]["taught"].mean(), time.time() - t0))
        sys.stdout.flush()
    return results


def report(results):
    from scipy import stats

    print("\n" + "=" * 78)
    print("THE FRONTIER -- communication energy vs what was learned")
    print("=" * 78)
    print("  {:<22} {:>12} {:>12} {:>10} {:>12} {:>10} {:>12}".format(
        "condition", "frozen E", "plastic E", "taught", "E/taught", "growth", "E/growth"))
    for name, _ in CONDITIONS:
        r = results[name]
        fe = r["frozen"]["energy"].mean()
        pe = r["plastic"]["energy"].mean()
        tm = r["plastic"]["taught"].mean()
        gm = tm - r["frozen"]["taught"].mean()
        eg = "{:>12.4g}".format(pe / gm) if gm > 0 else "{:>12}".format("n/a(<=0)")
        print("  {:<22} {:>12.4g} {:>12.4g} {:>10.0f} {:>12.4g} {:>+10.0f} {}".format(
            name, fe, pe, tm, pe / max(tm, 1e-9), gm, eg))

    seg, inter = results["vlsi"], results["topoforge"]
    print("\n" + "=" * 78)
    print("P1 -- is inference (frozen) energy really equal?")
    print("=" * 78)
    fe_s, fe_i = seg["frozen"]["energy"], inter["frozen"]["energy"]
    identical = np.array_equal(fe_s, fe_i)
    print("  segregated {:.6g} vs interleaved {:.6g}".format(fe_s.mean(), fe_i.mean()))
    print("  bit-identical across all seeds: {}".format(identical))
    print("  -> the 'zero inference-energy cost' claim is {}".format(
        "EXACT, not approximate" if identical else "approximate; restate it"))

    print("\n" + "=" * 78)
    print("P2 -- after plasticity, does the learning-preserving placement")
    print("      cost more communication energy?")
    print("=" * 78)
    pe_s, pe_i = seg["plastic"]["energy"], inter["plastic"]["energy"]
    t, p = stats.ttest_ind(pe_i, pe_s, equal_var=False)
    print("  segregated  {:.6g} +/- {:.3g}".format(pe_s.mean(), pe_s.std(ddof=1)))
    print("  interleaved {:.6g} +/- {:.3g}".format(pe_i.mean(), pe_i.std(ddof=1)))
    print("  interleaved / segregated = {:.4f}x   Welch p = {:.3g}".format(
        pe_i.mean() / pe_s.mean(), p))
    seg_growth = seg["plastic"]["taught"].mean() - seg["frozen"]["taught"].mean()
    inter_growth = inter["plastic"]["taught"].mean() - inter["frozen"]["taught"].mean()
    print("  [corrected] segregated growth {:+.0f}  interleaved growth {:+.0f}".format(
        seg_growth, inter_growth))
    if pe_i.mean() > pe_s.mean() and p < 0.05:
        print("  -> P2 FALSIFIED: interleaving does cost energy. The Discussion's")
        print("     tradeoff sentence stands and the recommendation has a price.")
        if seg_growth > 0:
            print("     Learning ratio (raw taught): {:.2f}x more.".format(
                inter["plastic"]["taught"].mean() / seg["plastic"]["taught"].mean()))
        else:
            print("     Learning ratio is now a SIGN FLIP (segregated growth is")
            print("     negative), not a simple ratio -- see gotcha #11.")
    elif p < 0.05:
        print("  -> P2 HOLDS, strongly: interleaving costs LESS energy.")
        if seg_growth > 0:
            print("     Also learns {:.2f}x more (raw taught). The objectives are".format(
                inter["plastic"]["taught"].mean() / seg["plastic"]["taught"].mean()))
            print("     aligned here, not opposed.")
        else:
            print("     On the corrected growth metric this is even sharper: segregated")
            print("     growth is negative ({:+.0f}) while interleaved's is positive".format(seg_growth))
            print("     ({:+.0f}) -- a sign flip, not a ratio.".format(inter_growth))
    else:
        print("  -> P2 holds weakly: no significant energy difference either way,")
        print("     so the learning gain is free rather than paid for.")

    print("\n" + "=" * 78)
    print("P3 -- does the association-aware mapper dominate?")
    print("=" * 78)
    fn = results["spinemap-functional"]
    fn_growth = fn["plastic"]["taught"].mean() - fn["frozen"]["taught"].mean()
    for other in ("topoforge", "vlsi", "spinemap-population"):
        o = results[other]
        o_growth = o["plastic"]["taught"].mean() - o["frozen"]["taught"].mean()
        de = fn["plastic"]["energy"].mean() / o["plastic"]["energy"].mean()
        dt = fn["plastic"]["taught"].mean() / o["plastic"]["taught"].mean()
        verdict = ("DOMINATES" if de <= 1.0 and dt >= 1.0 else
                   "dominated" if de >= 1.0 and dt <= 1.0 else "trades off")
        if o_growth <= 0 < fn_growth:
            growth_note = "growth: {:+.0f} vs {:+.0f} (sign flip, not a ratio)".format(fn_growth, o_growth)
        else:
            growth_note = "growth ratio {:.3f}x".format(fn_growth / o_growth) if o_growth != 0 else "n/a"
        print("  vs {:<22} energy {:.3f}x, taught {:.3f}x  -> {}  [{}]".format(
            other, de, dt, verdict, growth_note))

    print("\n" + "=" * 78)
    print("Read the frontier column, not any single number: if the best-learning")
    print("placements are also at or below the others on energy, then a")
    print("plasticity-aware mapper is not asking anyone to pay for learnability.")
    print("=" * 78)


if __name__ == "__main__":
    if "--audit" in sys.argv:
        sys.exit(0 if audit() else 1)
    print("=" * 78)
    print("EXP 44: the energy-learnability frontier")
    print("{} conditions x {} seeds".format(len(CONDITIONS), len(SEEDS)))
    print("=" * 78)
    if not audit():
        print("\nAUDIT FAILED -- energy comparison would be confounded.")
        sys.exit(1)
    print("\n" + "=" * 78)
    print("RUNNING")
    print("=" * 78)
    t0 = time.time()
    res = run()
    report(res)
    print("\ntotal {:.0f}s".format(time.time() - t0))
