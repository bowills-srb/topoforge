"""Exp 38b: does Section 4.6's SpiNeMap result (exp38_spinemap_baseline.py)
share the pinned-wiring defect, and does it survive correction?

WHY THIS EXISTS. Exp 54 audited exp32b_benchmark.run_life's pinned-wiring
defect for the "vlsi" (segregated) and "topoforge" (interleaved) conditions
only, and found the synthetic headline robust (point estimates survive,
only variance was understated) -- because segregated has ZERO adjacency on
the measured type pairs, a fact of geometry no wiring draw can rescue.
exp38_spinemap_baseline.py imports the identical `run_life` directly
(verified: `from exp32b_benchmark import ... run_life`) but adds two NEW
conditions -- "spinemap-pop" and "spinemap-func" -- that Exp 54 never
checked. These sit at INTERMEDIATE adjacency (Table 3: 66.7-71.1% for
population, 100% for functional), not at segregated's zero floor, so
Exp 54's "geometry forces the result, no draw can rescue it" argument does
not automatically transfer. Whether it does is an empirical question, not
an assumption -- this script answers it rather than assuming Exp 54's
finding generalizes.

A FURTHER REASON NOT TO ASSUME: exp32b.make_placement assigns a
STRATEGY-DEPENDENT cids order (segregated, interleaved and random each get
different global-index-to-type mappings; see its `all_cids` construction),
while exp38's SpiNeMap conditions use a FIXED, sorted cids
(`np.repeat(np.arange(NC), N // NC)`) regardless of graph mode. So the
logical-identity component of the defect (component (2) in Exp 53/54's
framing) is a live question BETWEEN spinemap and vlsi/topoforge, even
though Exp 54 found it contributed ~0 between vlsi and topoforge alone,
and even though the prior real-data correction (exp49c) found it can be a
non-issue (fixed cids) while resampling STILL moves point estimates
substantially through the dynamics alone. Registering this rather than
assuming either direction.

REUSES, NOT REIMPLEMENTS: `run_life_corrected`, `make_wiring`,
`logical_to_global`, `wiring_seed_for` are imported verbatim from
`exp54_synthetic_wiring_audit.py`, already P0-audited there against the
imported `run_life`. `make_placement_spinemap` is imported verbatim from
`exp38_spinemap_baseline.py`. ARM B (resampled, still global) is skipped
per Exp 53/54/51b's established finding that it contributes ~nothing
beyond ARM A -> ARM C's total effect; this is a runtime decision, not a
scope reduction, exactly as exp51b argued.

ARMS:
  ARM A  REPLICATION   wiring_seed = 7 (fixed), global-index wiring.
                       Must reproduce exp38's published Table 4 numbers
                       (frozen/growth for all five conditions) before
                       ARM C is interpreted.
  ARM C  LOGICAL       wiring_seed varies per seed, wiring drawn over
                       logical identities and mapped through each
                       condition's own placement.

WHAT EACH OUTCOME MEANS:
  - ARM C's "where does SpiNeMap land" fractions (7.9% pop, 102.5% func in
    the published table) stay close under correction: Section 4.6 is
    robust, matching Exp 54's finding for vlsi/topoforge; only its SD
    needs a wiring-variance note.
  - The fractions move materially: Section 4.6 needs the same kind of
    numeric correction Section 4.11 received, and this docstring's
    "does not automatically transfer" caveat was right to raise.

Run:          python src/experiments/exp38b_spinemap_wiring_corrected.py
Run (audit):  python src/experiments/exp38b_spinemap_wiring_corrected.py --audit
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

from exp32b_benchmark import make_placement, run_life, NC, N
from exp38_spinemap_baseline import make_placement_spinemap
from exp54_synthetic_wiring_audit import (
    run_life_corrected, make_wiring, wiring_seed_for, LEGACY_WIRING_SEED,
)
from exp37c_real_data_scaled import cohens_d_pooled, bootstrap_ratio_ci

SEEDS = list(range(20))
ARMS = [
    ("A_replication", "global", False),
    ("C_logical", "logical", True),
]
CACHE_PATH = os.path.join("shd_data", "exp38b_cache.json")

CONDITIONS = [
    ("random", lambda: make_placement("random")),
    ("vlsi", lambda: make_placement("vlsi")),
    ("topoforge", lambda: make_placement("topoforge")),
    ("spinemap-pop", lambda: make_placement_spinemap("population", seed=0)),
    ("spinemap-func", lambda: make_placement_spinemap("functional", seed=0)),
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
    print("=" * 96)
    print("EXP 38b: SpiNeMap (Section 4.6) under CORRECTED wiring")
    print("=" * 96)

    print("\n[0] Placements and per-type counts")
    place = {}
    t0 = time.time()
    for name, fn in CONDITIONS:
        coords, cids = fn()
        place[name] = (coords, cids)
        counts = [int((cids == t).sum()) for t in range(NC)]
        print("  {:<14} per-type counts {}".format(name, counts))
    print("  (placement computation: {:.0f}s)".format(time.time() - t0))

    # ---- P0 ---------------------------------------------------------------
    print("\n[1] P0 equivalence: run_life_corrected(legacy) vs imported run_life")
    c0, ci0 = place["vlsi"]
    ref = run_life(c0, ci0, 0)
    new = run_life_corrected(c0, ci0, 0, wiring_seed=LEGACY_WIRING_SEED, wiring_mode="global")
    p0 = all(ref[k][f] == new[k][f] for k in ("frozen", "plastic", "reversal")
             for f in ("taught", "old_03", "new_14"))
    print("  vlsi seed0: imported taught(frozen/plastic)=({},{})  corrected=({},{})  {}".format(
        ref["frozen"]["taught"], ref["plastic"]["taught"],
        new["frozen"]["taught"], new["plastic"]["taught"],
        "OK" if p0 else "MISMATCH"))
    print("  P0 {}".format("HOLDS" if p0 else "FAILS -- stopping"))
    if not p0:
        sys.exit(1)

    if audit_only:
        print("\n--audit: stopping before the sweep.")
        sys.exit(0)

    # ---- sweep --------------------------------------------------------------
    n_total = len(ARMS) * len(CONDITIONS) * len(SEEDS)
    print("\n[2] Running {} arms x {} conditions x {} seeds = {} runs".format(
        len(ARMS), len(CONDITIONS), len(SEEDS), n_total))
    cache = _cache_load()
    growth, taught = {}, {}
    t_start = time.time()
    first = True
    for arm_label, wmode, resample in ARMS:
        print("\n  === ARM {} ===".format(arm_label))
        for name, _ in CONDITIONS:
            coords, cids = place[name]
            gs, ts = [], []
            for s in SEEDS:
                key = "{}|{}|{}".format(arm_label, name, s)
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
                gs.append(pl - fr); ts.append(pl)
            growth[(arm_label, name)] = np.array(gs, dtype=float)
            taught[(arm_label, name)] = np.array(ts, dtype=float)
            print("    {:<14} growth={:+8.1f} +/-{:6.1f}  taught={:8.1f} +/-{:6.1f}".format(
                name, np.mean(gs), np.std(gs, ddof=1), np.mean(ts), np.std(ts, ddof=1)))
            sys.stdout.flush()
    print("\n  sweep wall time: {:.1f} min".format((time.time() - t_start) / 60.0))

    # ---- "where does SpiNeMap land" per arm --------------------------------
    print("\n" + "=" * 96)
    print("[3] Where does SpiNeMap land? (growth-based position, vlsi=0%, topoforge=100%)")
    print("=" * 96)
    arm_frac = {}
    for arm_label, _, _ in ARMS:
        vlsi_g = growth[(arm_label, "vlsi")].mean()
        topo_g = growth[(arm_label, "topoforge")].mean()
        span_g = max(topo_g - vlsi_g, 1e-9)
        print("\n  ARM {}  (vlsi growth={:+.1f}, topoforge growth={:+.1f})".format(
            arm_label, vlsi_g, topo_g))
        fracs = {}
        for name in ("spinemap-pop", "spinemap-func"):
            g = growth[(arm_label, name)].mean()
            frac = (g - vlsi_g) / span_g
            fracs[name] = frac
            t, p = stats.ttest_ind(growth[(arm_label, name)], growth[(arm_label, "vlsi")], equal_var=False)
            print("    {:<14} growth={:+8.1f}  position={:6.1%}  (vs vlsi Welch p={:.2e})".format(
                name, g, frac, p))
        arm_frac[arm_label] = fracs

    # ---- comparison ---------------------------------------------------------
    print("\n" + "=" * 96)
    print("[4] VERDICT")
    print("=" * 96)
    print("  published Table 4 positions: spinemap-pop=7.9%, spinemap-func=102.5%")
    max_abs_change = 0.0
    for name in ("spinemap-pop", "spinemap-func"):
        a = arm_frac["A_replication"][name]
        c = arm_frac["C_logical"][name]
        change_pp = 100.0 * (c - a)
        max_abs_change = max(max_abs_change, abs(change_pp))
        print("  {:<14} ARM A position={:6.1%}  ARM C position={:6.1%}  change={:+.1f}pp".format(
            name, a, c, change_pp))
    print("\n  max |position change| across the two SpiNeMap conditions: {:.1f} percentage points".format(
        max_abs_change))
    if max_abs_change < 15.0:
        print("  ROBUST: SpiNeMap's position between segregated and interleaved is stable under")
        print("  corrected wiring, matching Exp 54's finding for vlsi/topoforge alone. Only the")
        print("  published SD needs a note that it omits wiring-draw variance.")
    else:
        print("  NOT ROBUST: SpiNeMap's position moves materially under corrected wiring.")
        print("  Section 4.6 needs the same kind of correction Section 4.11 received.")
