"""Exp 42b: does Section 4.8's fabric-sparsity crossover (exp42) share the
pinned-wiring defect, and does it survive correction?

WHY THIS EXISTS. Exp 42 imports `run_life` directly from
`exp32b_benchmark.py` (the same buggy harness Exp 54 audited for vlsi and
topoforge alone) across four strategies -- topoforge, spinemap-population,
spinemap-functional, vlsi -- and SIX fabric pitches. Exp 38b already
confirmed Exp 54's "geometry forces the result, no draw can rescue it"
finding extends to the SpiNeMap conditions AT THE NATIVE PITCH (rho=1.0).
What has not been checked is whether it extends across the WHOLE pitch
sweep: at large pitch (rho=3.0), Table 7 shows spinemap-population
converging to a value NUMERICALLY IDENTICAL to segregated (701 vs 701,
zero variance in the published table) -- i.e. the published claim is that
a real mapping tool becomes AS BAD AS the hand-rolled segregated baseline
at low fabric density. If the pinned draw shifts either condition's floor,
this specific "SpiNeMap = segregated" convergence claim is exactly the
kind of near-equality result a single arbitrary draw could produce or
destroy.

REUSES, NOT REIMPLEMENTS: `run_life_corrected`, `make_wiring`,
`wiring_seed_for`, `LEGACY_WIRING_SEED` from `exp54_synthetic_wiring_audit.py`
(already P0-audited there); `base_placement`, `respace`, `placement_at`,
`PITCHES`, `STRATEGIES` from `exp42_fabric_sparsity_crossover.py` verbatim,
so the placements under test are byte-identical to the published ones.
ARM B is skipped per the established finding (Exp 53/54/51b/38b) that it
contributes ~nothing beyond ARM A -> ARM C's total effect.

ARMS:
  ARM A  REPLICATION   wiring_seed = 7 (fixed), global-index wiring. Must
                       reproduce Exp 42's published Table 7 growth values
                       before ARM C is interpreted.
  ARM C  LOGICAL       wiring_seed varies per seed, wiring drawn over
                       logical identities and mapped through each
                       (strategy, pitch) placement's own cids.

WHAT EACH OUTCOME MEANS:
  - Growth values at every (strategy, pitch) cell stay close between arms,
    and in particular the "spinemap-population == segregated at rho>=1.5"
    near-equality survives: Section 4.8 is robust, matching Exp 38b and
    Exp 54; only the published SD needs a wiring-variance note.
  - The near-equality breaks, or growth shifts materially at one or more
    cells: Section 4.8 needs the same kind of numeric correction Section
    4.11 received.

Run:          python src/experiments/exp42b_fabric_sparsity_wiring_corrected.py
Run (audit):  python src/experiments/exp42b_fabric_sparsity_wiring_corrected.py --audit
Run (smoke):  python src/experiments/exp42b_fabric_sparsity_wiring_corrected.py --smoke
"""
import json
import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from scipy import stats

from exp32b_benchmark import run_life
from exp42_fabric_sparsity_crossover import (
    base_placement, placement_at, PITCHES, STRATEGIES, PLASTICITY_RADIUS,
)
from exp54_synthetic_wiring_audit import (
    run_life_corrected, wiring_seed_for, LEGACY_WIRING_SEED,
)

CACHE_PATH = os.path.join("shd_data", "exp42b_cache.json")

ARMS = [
    ("A_replication", "global", False),
    ("C_logical", "logical", True),
]


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


if __name__ == "__main__":
    audit_only = "--audit" in sys.argv
    smoke = "--smoke" in sys.argv
    SEEDS = list(range(3)) if smoke else list(range(8))
    pitches = [2.5, 5.0, 10.0] if smoke else PITCHES

    print("=" * 96)
    print("EXP 42b: Fabric-sparsity crossover (Section 4.8) under CORRECTED wiring{}".format(
        " (SMOKE)" if smoke else ""))
    print("=" * 96)

    # ---- P0 -----------------------------------------------------------------
    print("\n[0] P0 equivalence: run_life_corrected(legacy) vs imported run_life")
    coords0, cids0 = placement_at("vlsi", 5.0)
    ref = run_life(coords0, cids0, 0)
    new = run_life_corrected(coords0, cids0, 0, wiring_seed=LEGACY_WIRING_SEED, wiring_mode="global")
    p0 = all(ref[k][f] == new[k][f] for k in ("frozen", "plastic", "reversal")
             for f in ("taught", "old_03", "new_14"))
    print("  vlsi@pitch5.0 seed0: imported taught(frozen/plastic)=({},{})  corrected=({},{})  {}".format(
        ref["frozen"]["taught"], ref["plastic"]["taught"],
        new["frozen"]["taught"], new["plastic"]["taught"],
        "OK" if p0 else "MISMATCH"))
    print("  P0 {}".format("HOLDS" if p0 else "FAILS -- stopping"))
    if not p0:
        sys.exit(1)

    if audit_only:
        print("\n--audit: stopping before the sweep.")
        sys.exit(0)

    # ---- sweep ----------------------------------------------------------------
    n_total = len(ARMS) * len(STRATEGIES) * len(pitches) * len(SEEDS)
    print("\n[1] Running {} arms x {} strategies x {} pitches x {} seeds = {} runs".format(
        len(ARMS), len(STRATEGIES), len(pitches), len(SEEDS), n_total))
    cache = _cache_load()
    growth = {}
    t_start = time.time()
    first = True
    for arm_label, wmode, resample in ARMS:
        print("\n  === ARM {} ===".format(arm_label))
        for strat, _ in STRATEGIES:
            for pitch in pitches:
                coords, cids = placement_at(strat, pitch)
                gs = []
                for s in SEEDS:
                    key = "{}|{}|{}|{}".format(arm_label, strat, pitch, s)
                    if key in cache:
                        fr, pl = cache[key]
                    else:
                        t0 = time.time()
                        ws = wiring_seed_for(resample, s)
                        r = run_life_corrected(coords, cids, s, wiring_seed=ws, wiring_mode=wmode)
                        fr, pl = int(r["frozen"]["taught"]), int(r["plastic"]["taught"])
                        cache[key] = [fr, pl]
                        _cache_save(cache)
                        if first:
                            dt = time.time() - t0
                            print("    [first run {:.0f}s -> full sweep ~{:.0f} min]".format(
                                dt, n_total * dt / 60.0))
                            first = False
                    gs.append(pl - fr)
                growth[(arm_label, strat, pitch)] = np.array(gs, dtype=float)
                print("    {:<22} rho={:.2f}  growth={:+8.1f} +/-{:6.1f}".format(
                    strat, pitch / PLASTICITY_RADIUS,
                    growth[(arm_label, strat, pitch)].mean(),
                    growth[(arm_label, strat, pitch)].std(ddof=1)))
                sys.stdout.flush()
    print("\n  sweep wall time: {:.1f} min".format((time.time() - t_start) / 60.0))

    # ---- comparison -------------------------------------------------------
    print("\n" + "=" * 96)
    print("[2] ARM A vs ARM C, growth, per (strategy, pitch)")
    print("=" * 96)
    max_abs_change = 0.0
    max_cell = None
    for strat, _ in STRATEGIES:
        for pitch in pitches:
            a = growth[("A_replication", strat, pitch)].mean()
            c = growth[("C_logical", strat, pitch)].mean()
            denom = max(abs(a), 1.0)
            pct_change = 100.0 * (c - a) / denom
            if abs(pct_change) > max_abs_change:
                max_abs_change = abs(pct_change)
                max_cell = (strat, pitch)
            print("  {:<22} rho={:.2f}  A={:+8.1f}  C={:+8.1f}  change={:+7.1f}%".format(
                strat, pitch / PLASTICITY_RADIUS, a, c, pct_change))

    # ---- the specific near-equality claim ----------------------------------
    print("\n" + "=" * 96)
    print("[3] The specific claim at risk: spinemap-population == segregated at high rho")
    print("=" * 96)
    for arm_label, _, _ in ARMS:
        print("\n  ARM {}".format(arm_label))
        for pitch in pitches:
            if pitch / PLASTICITY_RADIUS < 1.5:
                continue
            sp = growth[(arm_label, "spinemap-population", pitch)]
            vl = growth[(arm_label, "vlsi", pitch)]
            t, p = stats.ttest_ind(sp, vl, equal_var=False)
            print("    rho={:.2f}: spinemap-pop={:+8.1f}  vlsi={:+8.1f}  Welch p={:.3f}  {}".format(
                pitch / PLASTICITY_RADIUS, sp.mean(), vl.mean(), p,
                "INDISTINGUISHABLE" if p > 0.05 else "DISTINGUISHABLE"))

    # ---- verdict ------------------------------------------------------------
    print("\n" + "=" * 96)
    print("[4] VERDICT")
    print("=" * 96)
    print("  max |growth change| across all (strategy, pitch) cells: {:.1f}%  (at {})".format(
        max_abs_change, max_cell))
    if max_abs_change < 25.0:
        print("  ROBUST: growth values are stable across the whole pitch sweep under corrected")
        print("  wiring, matching Exp 38b/Exp 54's finding. Only the published SD needs a note")
        print("  that it omits wiring-draw variance; no numeric correction needed for Section 4.8.")
    else:
        print("  NOT ROBUST: growth values move materially at one or more (strategy, pitch) cells")
        print("  under corrected wiring. Section 4.8 needs the same kind of correction Section")
        print("  4.11 received; re-derive the affected cells from these ARM C numbers.")
