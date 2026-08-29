"""Exp 42: where does a real mapping tool start to hurt? The fabric-sparsity
crossover.

Motivation (preprint Section 6, "Baseline fidelity" -- explicitly flagged as
future work): Exp 38 found that SpiNeMap, a published communication-minimizing
mapper, does NOT reproduce our pathological segregated baseline -- it lands
near interleaved. The stated reason was geometric: SpiNeMap compacts
correlated populations, and in the PLB the core pitch (5.0) happens to equal
the plasticity radius (5.0), so neurons in adjacent cores stay within reach.
The prediction that follows from the Exp 40/41 mechanism (learning is set by
REACH -- how much correlated structure the local rule can physically touch) is
sharp:

    SpiNeMap's placement is a fixed logical mapping. Spread the same mapping
    over a sparser fabric and its reach must collapse, so its learning must
    collapse with it. There should be a crossover ratio rho = pitch / radius
    beyond which a communication-optimal mapper re-incurs the placement
    penalty it avoided at rho = 1.

We test it by sweeping the fabric pitch with the plasticity radius held fixed
at 5.0 (the radius is a property of the substrate's plasticity mechanism; the
pitch is a property of the floorplan), rho = 0.5 .. 3.0.

PREDICTIONS (registered before running):
  (P1) INTERLEAVED is ~flat in pitch. Its cross-type partners are inside the
       same core disc (radius 1.5 << 5.0), so respacing cores cannot remove
       them. This is also the CONTROL for a confound: respacing changes the
       rewire rule's distance discount 1/(1+0.05*d^2) as well as reach, and
       if that discount alone drove the result interleaved would fall too.
  (P2) SPINEMAP degrades monotonically with rho and converges toward the
       segregated condition at large rho -- a crossover, not a constant.
  (P3) The degradation is MEDIATED by reach: pooled across strategy and pitch,
       pre-learning reach predicts learning.
  Falsified if SpiNeMap stays flat (its advantage is not geometric, and
  Section 6's explanation for Exp 38 is wrong), or if interleaved falls in
  step with it (the effect is the distance discount, not reach).

Method note (audit-before-trust): placements are NOT re-implemented here.
Each strategy's placement is generated at the native pitch by the VERBATIM
imported exp32b / exp38 functions, then each core disc is rigidly translated
to the new pitch. Rigid translation preserves within-core geometry exactly
and is the identity at pitch 5.0 -- both checked bit-exactly in --audit.

Run:
  python src/experiments/exp42_fabric_sparsity_crossover.py --audit   # ~10s
  python src/experiments/exp42_fabric_sparsity_crossover.py --smoke   # ~4min
  python src/experiments/exp42_fabric_sparsity_crossover.py           # full

METRIC CORRECTION (2026-08-29, PROJECT_HISTORY gotcha #11): "taught" is a
raw edge count inflated by a non-local random-initialization baseline
shared by every placement/pitch (see exp32b_benchmark.py's docstring).
This script now also tracks growth = plastic_taught - frozen_taught (the
plasticity-attributable signal) and reports P1/P2/P3 on BOTH metrics --
taught (legacy, kept for continuity) and growth (corrected, primary).
"""
import numpy as np
import time
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from exp32b_benchmark import (
    PATTERNS, NC, N, N_CORES, NEURONS_PER_CORE,
    core_positions, make_placement, run_life,
)
from exp38_spinemap_baseline import make_placement_spinemap
from spatial import SpatialGrid

NATIVE_PITCH = 5.0          # the pitch exp32b/exp38 were built at
PLASTICITY_RADIUS = 5.0     # hardcoded inside run_life; the fixed substrate property
CORE_DISC_RADIUS = 1.5      # from exp32b.neurons_in_core

PITCHES = [2.5, 5.0, 6.25, 7.5, 10.0, 15.0]  # rho = 0.5, 1.0, 1.25, 1.5, 2.0, 3.0
STRATEGIES = [
    ("topoforge", "interleaved (control)"),
    ("spinemap-population", "SpiNeMap (population graph)"),
    ("spinemap-functional", "SpiNeMap (functional graph)"),
    ("vlsi", "segregated (our baseline)"),
]


# ============================================================
# Placement: verbatim generators + rigid per-core respacing
# ============================================================
def base_placement(strategy):
    """Placement at the NATIVE pitch, from the verbatim imported generators."""
    if strategy.startswith("spinemap-"):
        return make_placement_spinemap(strategy.split("-", 1)[1], seed=0)
    return make_placement(strategy)


def core_of(coords):
    """Which core each neuron sits in, inferred from the native-pitch coords
    (every neuron is within CORE_DISC_RADIUS of its own core centre, and the
    native pitch is 5.0 > 2*1.5, so the assignment is unambiguous)."""
    base = core_positions()
    d2 = ((coords[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    owner = d2.argmin(1)
    worst = np.sqrt(d2[np.arange(len(coords)), owner]).max()
    assert worst < CORE_DISC_RADIUS + 1e-9, "core assignment ambiguous: {}".format(worst)
    return owner


def respace(coords, pitch):
    """Rigidly translate each core's disc to the new pitch. Within-core
    geometry is untouched; identity at pitch == NATIVE_PITCH."""
    base = core_positions()
    owner = core_of(coords)
    return coords + base[owner] * (pitch / NATIVE_PITCH - 1.0)


def placement_at(strategy, pitch):
    coords, cids = base_placement(strategy)
    return respace(coords, pitch), cids


# ============================================================
# Reach: the placement-only mediator, PLB flavour
# ============================================================
def plb_reach(coords, cids, radius=PLASTICITY_RADIUS):
    """Mean number of PARTNER-type neurons within the plasticity radius, over
    the neurons that participate in a taught association ((0,3) and (1,4)).
    The direct analogue of exp40's reach_metrics for the synthetic benchmark:
    how much of the to-be-learned correlated structure the local rule can
    physically touch, measured before any learning."""
    g = SpatialGrid(coords, radius)
    counts = []
    for pat in PATTERNS:
        if len(pat) != 2:
            continue
        for a, b in ((pat[0], pat[1]), (pat[1], pat[0])):
            for i in np.where(cids == a)[0]:
                nbrs = g.within(int(i), radius)
                counts.append(sum(1 for j in nbrs if cids[int(j)] == b))
    return float(np.mean(counts))


def wire_energy_proxy(coords, cids):
    """Mean squared distance between neurons of associated types -- the
    communication cost a mapper is trying to minimize."""
    tot, n = 0.0, 0
    for pat in PATTERNS:
        if len(pat) != 2:
            continue
        a = np.where(cids == pat[0])[0]
        b = np.where(cids == pat[1])[0]
        d2 = ((coords[a][:, None, :] - coords[b][None, :, :]) ** 2).sum(-1)
        tot += d2.sum()
        n += d2.size
    return tot / max(n, 1)


# ============================================================
# Audit
# ============================================================
def audit():
    print("=" * 74)
    print("AUDIT -- respacing must be exact before any learning number is trusted")
    print("=" * 74)

    print("\n[1] respace() is the IDENTITY at the native pitch (bit-exact)")
    ok = True
    for strat, _ in STRATEGIES:
        c, _ci = base_placement(strat)
        diff = np.abs(respace(c, NATIVE_PITCH) - c).max()
        ok &= (diff == 0.0)
        print("    {:<22} max|respaced - original| = {:.1e}".format(strat, diff))

    print("\n[2] within-core geometry preserved exactly under respacing")
    for strat, _ in STRATEGIES:
        c, _ci = base_placement(strat)
        owner = core_of(c)
        worst = 0.0
        for pitch in PITCHES:
            cp = respace(c, pitch)
            for k in range(N_CORES):
                m = np.where(owner == k)[0]
                d0 = ((c[m][:, None] - c[m][None, :]) ** 2).sum(-1)
                d1 = ((cp[m][:, None] - cp[m][None, :]) ** 2).sum(-1)
                worst = max(worst, np.abs(d0 - d1).max())
        # float64 round-off only: within-core d^2 values are O(1-9), and the
        # translation adds an offset up to 70, so ~1e-14 absolute is the noise
        # floor. Anything larger would mean the discs are being deformed.
        ok &= (worst < 1e-10)
        print("    {:<22} max within-core pairwise d^2 drift = {:.1e}".format(strat, worst))

    print("\n[3] SpiNeMap's logical mapping is pitch-invariant (uniform scaling")
    print("    cannot change the argmin of a wire-energy objective), so any")
    print("    change we measure is physics, not the mapper re-deciding.")
    for mode in ("population", "functional"):
        c0, _ = make_placement_spinemap(mode, seed=0)
        c1, _ = make_placement_spinemap(mode, seed=0)
        print("    spinemap-{:<12} deterministic re-run identical: {}".format(
            mode, bool(np.array_equal(c0, c1))))

    print("\n[4] reach and wire cost vs fabric pitch (pre-learning, placement only)")
    print("    {:<22} {:>6} {:>10} {:>10} {:>12}".format(
        "strategy", "rho", "pitch", "reach", "wire d^2"))
    for strat, _ in STRATEGIES:
        for pitch in PITCHES:
            c, ci = placement_at(strat, pitch)
            print("    {:<22} {:>6.1f} {:>10.1f} {:>10.2f} {:>12.0f}".format(
                strat, pitch / PLASTICITY_RADIUS, pitch,
                plb_reach(c, ci), wire_energy_proxy(c, ci)))

    print("\n    (P1 precondition: interleaved reach should stay well above zero")
    print("     at every pitch; SpiNeMap's should collapse toward segregated.)")
    print("\nVERDICT: {}".format("PASS" if ok else "FAIL -- do not trust downstream numbers"))
    return ok


# ============================================================
# Main sweep
# ============================================================
def run(seeds, pitches):
    rows = {}
    reach_of = {}
    energy_of = {}
    for strat, desc in STRATEGIES:
        print("\n  {} -- {}".format(strat.upper(), desc))
        for pitch in pitches:
            coords, cids = placement_at(strat, pitch)
            reach_of[(strat, pitch)] = plb_reach(coords, cids)
            energy_of[(strat, pitch)] = wire_energy_proxy(coords, cids)
            taught, relearn, growth = [], [], []
            t0 = time.time()
            for s in seeds:
                r = run_life(coords, cids, s)
                taught.append(r["plastic"]["taught"])
                relearn.append(r["reversal"]["new_14"])
                growth.append(r["plastic"]["taught"] - r["frozen"]["taught"])
            rows[(strat, pitch)] = (np.array(taught, float), np.array(relearn, float), np.array(growth, float))
            print("    rho={:.1f} (pitch {:>4.1f})  reach={:>6.2f}  taught={:>7.0f} +/- {:>6.0f}"
                  "  growth={:>+7.0f} +/- {:>6.0f}  relearn={:>6.0f}  ({:.0f}s)".format(
                      pitch / PLASTICITY_RADIUS, pitch, reach_of[(strat, pitch)],
                      rows[(strat, pitch)][0].mean(), rows[(strat, pitch)][0].std(ddof=1),
                      rows[(strat, pitch)][2].mean(), rows[(strat, pitch)][2].std(ddof=1),
                      rows[(strat, pitch)][1].mean(), time.time() - t0))
            sys.stdout.flush()
    return rows, reach_of, energy_of


def dump_json(rows, reach_of, energy_of, pitches, path="exp42_results.json"):
    """Persist every measurement BEFORE any analysis runs. A 50-minute sweep
    should never be lost to an import error or a stats bug in the report."""
    import json
    out = []
    for (strat, pitch), (taught, relearn, growth) in rows.items():
        out.append({"strategy": strat, "pitch": pitch,
                    "rho": pitch / PLASTICITY_RADIUS,
                    "reach": reach_of[(strat, pitch)],
                    "wire_d2": energy_of[(strat, pitch)],
                    "taught": taught.tolist(), "relearn": relearn.tolist(),
                    "growth": growth.tolist()})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("")
    print("  raw results written to {}".format(path))


def analyze_metric(rows, reach_of, energy_of, pitches, stats, idx, metric_label):
    """P1/P2/P3 + table for one metric (idx into the rows tuple: 0=taught, 2=growth)."""
    print("\n" + "=" * 74)
    print("TABLE -- {} vs fabric sparsity".format(metric_label))
    print("=" * 74)
    print("  {:>5} {:>7}".format("rho", "pitch"), end="")
    for strat, _ in STRATEGIES:
        print(" {:>22}".format(strat), end="")
    print()
    for pitch in pitches:
        print("  {:>5.1f} {:>7.1f}".format(pitch / PLASTICITY_RADIUS, pitch), end="")
        for strat, _ in STRATEGIES:
            m, s = rows[(strat, pitch)][idx].mean(), rows[(strat, pitch)][idx].std(ddof=1)
            print(" {:>13.0f} +/-{:>5.0f}".format(m, s), end="")
        print()

    print("\n" + "-" * 74)
    print("P1 [{}] -- is INTERLEAVED flat in pitch? (the distance-discount control)".format(metric_label))
    print("-" * 74)
    inter = np.array([rows[("topoforge", p)][idx].mean() for p in pitches])
    rho_i, p_i = stats.spearmanr(list(pitches), inter)
    print("  interleaved across rho {:.1f}->{:.1f}: {:.0f} -> {:.0f}, "
          "Spearman rho={:.2f} p={:.3f}".format(
              pitches[0] / PLASTICITY_RADIUS, pitches[-1] / PLASTICITY_RADIUS,
              inter[0], inter[-1], rho_i, p_i))
    t, p = stats.ttest_ind(rows[("topoforge", pitches[0])][idx],
                           rows[("topoforge", pitches[-1])][idx], equal_var=False)
    print("  extremes Welch t-test: p={:.4f}  -> interleaved is {}".format(
        p, "NOT flat (confound live)" if p < 0.05 else "flat (confound controlled)"))

    print("\n" + "-" * 74)
    print("P2 [{}] -- does SPINEMAP cross over? (vs interleaved at each rho)".format(metric_label))
    print("-" * 74)
    for strat, _ in STRATEGIES:
        if strat == "topoforge":
            continue
        print("\n  {}".format(strat))
        print("    {:>5} {:>10} {:>14} {:>12} {:>12}".format(
            "rho", "reach", "vs interleaved", "Welch p", "verdict"))
        crossover = None
        for pitch in pitches:
            a = rows[("topoforge", pitch)][idx]
            b = rows[(strat, pitch)][idx]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            hit = (p < 0.05 and b.mean() < a.mean())
            if hit and crossover is None:
                crossover = pitch / PLASTICITY_RADIUS
            print("    {:>5.1f} {:>10.2f} {:>12.0f} vs {:>7.0f} {:>12.4f} {:>12}".format(
                pitch / PLASTICITY_RADIUS, reach_of[(strat, pitch)], b.mean(), a.mean(), p,
                "PENALTY" if hit else "no penalty"))
        print("    crossover: {}".format(
            "rho >= {:.1f}".format(crossover) if crossover
            else "none within the swept range"))

    print("\n" + "-" * 74)
    print("P3 [{}] -- is the degradation mediated by REACH? (pooled over strategy x pitch)".format(metric_label))
    print("-" * 74)
    xs, ys, labels = [], [], []
    for strat, _ in STRATEGIES:
        for pitch in pitches:
            xs.append(reach_of[(strat, pitch)])
            ys.append(rows[(strat, pitch)][idx].mean())
            labels.append((strat, pitch))
    xs, ys = np.array(xs), np.array(ys)
    pr, pp = stats.pearsonr(xs, ys)
    sr, sp = stats.spearmanr(xs, ys)
    print("  n={} points  Pearson R^2={:.3f} (p={:.2e})  Spearman rho={:.3f} (p={:.2e})".format(
        len(xs), pr ** 2, pp, sr, sp))


def report(rows, reach_of, energy_of, pitches):
    try:
        from scipy import stats
    except ImportError:
        print("")
        print("  scipy not available in this interpreter -- raw results are")
        print("  saved; run the analysis with the checked-in venv:")
        print("    venv/Scripts/python.exe src/experiments/exp42b_mediator_forms.py <output.txt>")
        return

    print("\n" + "#" * 74)
    print("# LEGACY METRIC: raw taught mass (diluted by shared non-local init baseline --")
    print("# see exp32b_benchmark.py docstring, PROJECT_HISTORY gotcha #11). Kept for")
    print("# continuity; growth (below) is the corrected primary metric.")
    print("#" * 74)
    analyze_metric(rows, reach_of, energy_of, pitches, stats, 0, "raw taught mass")

    print("\n" + "#" * 74)
    print("# CORRECTED METRIC: growth = plastic_taught - frozen_taught")
    print("# (plasticity-attributable signal, isolated from the shared baseline)")
    print("#" * 74)
    analyze_metric(rows, reach_of, energy_of, pitches, stats, 2, "growth (plasticity-attributable)")

    print("\n" + "=" * 74)
    print("If interleaved is flat while SpiNeMap falls with reach, then the")
    print("Exp 38 result is regime-specific: a communication-minimizing mapper")
    print("avoids the placement penalty only while the fabric pitch keeps")
    print("correlated populations inside the plasticity radius.")
    print("=" * 74)


if __name__ == "__main__":
    if "--audit" in sys.argv:
        sys.exit(0 if audit() else 1)

    smoke = "--smoke" in sys.argv
    seeds = list(range(3)) if smoke else list(range(8))
    pitches = [2.5, 5.0, 10.0] if smoke else PITCHES

    print("=" * 74)
    print("EXP 42: FABRIC-SPARSITY CROSSOVER {}".format("(SMOKE)" if smoke else ""))
    print("plasticity radius fixed at {:.1f}; core pitch swept {}".format(
        PLASTICITY_RADIUS, pitches))
    print("{} strategies x {} pitches x {} seeds = {} runs".format(
        len(STRATEGIES), len(pitches), len(seeds),
        len(STRATEGIES) * len(pitches) * len(seeds)))
    print("=" * 74)
    t0 = time.time()
    rows, reach_of, energy_of = run(seeds, pitches)
    dump_json(rows, reach_of, energy_of, pitches)
    report(rows, reach_of, energy_of, pitches)
    print("\ntotal {:.0f}s".format(time.time() - t0))
