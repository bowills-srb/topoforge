"""Exp 49c-corrected: what are Exp 49c's real-data partial-knowledge numbers
under CORRECTED wiring?

WHY THIS EXISTS, AND A CORRECTION TO A FIRST ATTEMPT AT THIS SAME QUESTION.
`exp49b_shd_partial_knowledge_corrected.py` set out to answer this question
but targeted the wrong baseline: it reproduced `exp49_shd_partial_knowledge.py`
(1 placement seed), whose own conclusion was already withdrawn as
placement-search-variance noise by `exp49b_shd_robustness.py` (3 seeds) and
superseded by `exp49c_shd_large_sweep.py` (10 placement seeds x 4 dynamics
seeds, 240 runs) -- the properly-powered design registered with a 2-SE
adjacent-point resolvability rule, and the one whose numbers ("50% recovery
at 75% knowledge, 90% at 100%") actually appear in the preprint's Section
4.11. That script's ARM A reproduced the already-discredited 25%-beats-100%
artifact bit-for-bit (P0 held), which is a correct replication of the wrong
experiment, not a correction of the right one. This script re-targets Exp
49c specifically.

THE DEFECT, unchanged from every other file in this family. `run_life`,
imported from `exp37b_v2_real_data.py` via `exp49_shd_partial_knowledge.py`'s
`bridge_of`, pins initial connectivity and the inhibitory mask to a FIXED
`np.random.default_rng(7)` draw that ignores the per-run seed. Exp 49c's
DYN_SEEDS = [0,1,2,3] therefore all share one wiring draw, so its
placement-seed-mean spread (registered as the unit of replication, since
Exp 49b found placement-seed variance dominates dynamics-seed variance)
reflects true placement-search variance, but each placement seed's OWN mean
across 4 dynamics seeds carries zero wiring-draw variance.

WHY THE FIXED-CIDS ARGUMENT FROM THE FIRST ATTEMPT STILL APPLIES, AND WAS
VERIFIED THERE (P0, P1b both held): `CIDS` is fixed at
`[INPUT_TYPE]*N_INPUT + [OUTPUT_A_TYPE]*N_OUTPUT_A + [OUTPUT_B_TYPE]*N_OUTPUT_B`
and `make_placement_real` returns it unchanged for every knowledge_frac and
every placement seed -- only coordinates vary. So the logical-vs-global
wiring distinction that did essentially all the damage on SHD/SSC (Exp 53)
is expected to be a no-op here too, exactly as the first attempt's P1b
confirmed. What that first attempt ALSO showed, unexpectedly and contrary
to its own docstring's prediction, is that resampling the wiring draw
across seeds still moves point estimates by up to 471% even with identical
logical connectivity -- because the rewiring dynamics are highly sensitive
to which of the ~2,000 initial random edges happen to land near which
coordinates, and one pinned draw is not representative. That finding is
carried over as the operative prediction here: expect ARM C's point
estimates to differ substantially from ARM A's, not just its error bars.

DESIGN, otherwise identical to Exp 49c: 10 placement seeds x 4 dynamics
seeds x 6 knowledge_fracs x 2 arms (ARM A replication, ARM C corrected),
using the already-audited `run_life_corrected` / `make_wiring` from
`exp52b_accuracy_corrected.py`. Placement-seed means are the unit of
replication (Exp 49c's registered choice); the 2-SE adjacent-point
resolvability rule is re-applied per arm.

ARMS:
  ARM A  REPLICATION   wiring_seed = 7 (fixed) for every dynamics seed,
                       global-index wiring. Must reproduce Exp 49c's
                       published numbers (P0, checked bit-identically
                       against the imported `run_life` before the sweep).
  ARM C  CORRECTED     wiring_seed = 7 + dynamics_seed (resampled),
                       logical-space wiring (predicted, and re-checked via
                       P1b, to coincide with global here since CIDS never
                       changes).

Run:          python src/experiments/exp49c_shd_large_sweep_corrected.py
Run (audit):  python src/experiments/exp49c_shd_large_sweep_corrected.py --audit
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

from exp37b_v2_real_data import run_life
from exp49_shd_partial_knowledge import (
    load_two_class_samples, channel_activity, build_real_graph,
    make_placement_real, SAMPLES_PER_CLASS, POOL_SIZE, KNOWLEDGE_FRACS,
)
from exp52b_accuracy_corrected import run_life_corrected, make_wiring, LEGACY_WIRING_SEED

CACHE_PATH = os.path.join("shd_data", "exp49c_corrected_cache.json")

PLACEMENT_SEEDS = list(range(10))
DYN_SEEDS = list(range(4))

ARMS = [
    ("A_replication", "global", False),
    ("C_corrected", "logical", True),
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


def wiring_seed_for(resample, dyn_seed):
    return LEGACY_WIRING_SEED if not resample else LEGACY_WIRING_SEED + dyn_seed


def bridge_of_corrected(coords, cids, target_samples, distractor_samples,
                        inhib_mode, resample, dyn_seeds, cache, cache_prefix):
    out = []
    for s in dyn_seeds:
        key = "{}|{}".format(cache_prefix, s)
        if key in cache:
            ba, bb = cache[key]
        else:
            ws = wiring_seed_for(resample, s)
            ba, bb = run_life_corrected(
                coords, cids, s, target_samples, distractor_samples,
                n_epochs=1, wiring_seed=ws, inhib_mode=inhib_mode)[:2]
            ba, bb = float(ba), float(bb)
            cache[key] = [ba, bb]
            _cache_save(cache)
        out.append(ba + bb)
    return np.array(out, dtype=float)


if __name__ == "__main__":
    audit_only = "--audit" in sys.argv
    print("=" * 96)
    print("EXP 49c-corrected: SHD large placement-seed sweep under CORRECTED wiring")
    print("=" * 96)

    print("\n[0] Loading real SHD samples (same as Exp 49/49b/49c)")
    t0 = time.time()
    top, avail, target_samples, distractor_samples = load_two_class_samples(SAMPLES_PER_CLASS, POOL_SIZE)
    print("  classes: target={}, distractor={}  (loaded in {:.0f}s)".format(top[0], top[1], time.time() - t0))
    act_t = channel_activity(target_samples)
    act_d = channel_activity(distractor_samples)

    # ---- [1] P0 equivalence -----------------------------------------------
    print("\n[1] P0 equivalence: run_life_corrected(legacy) vs imported run_life")
    p0 = True
    for kf in (0.0, 1.0):
        W = build_real_graph(act_t, act_d, kf)
        coords, cids = make_placement_real(W, seed=0)
        ba_ref, bb_ref = run_life(coords, cids, DYN_SEEDS[0], target_samples, distractor_samples, n_epochs=1)
        ba_new, bb_new = run_life_corrected(
            coords, cids, DYN_SEEDS[0], target_samples, distractor_samples, n_epochs=1,
            wiring_seed=LEGACY_WIRING_SEED, inhib_mode="global")[:2]
        ok = (ba_ref == ba_new) and (bb_ref == bb_new)
        p0 = p0 and ok
        print("  kf={:.2f} pseed=0: imported bridge=({:.6f},{:.6f})  corrected=({:.6f},{:.6f})  {}".format(
            kf, ba_ref, bb_ref, ba_new, bb_new, "OK" if ok else "MISMATCH"))
    print("  P0 {}".format("HOLDS" if p0 else "FAILS -- stopping"))
    if not p0:
        sys.exit(1)

    # ---- [2] P1b: logical vs global wiring ---------------------------------
    print("\n[2] P1b: logical vs global wiring (predicted identical, CIDS fixed)")
    p1b = True
    for kf in (0.0, 1.0):
        W = build_real_graph(act_t, act_d, kf)
        coords, cids = make_placement_real(W, seed=0)
        for s in DYN_SEEDS:
            g1 = make_wiring(cids, LEGACY_WIRING_SEED + s, "global")
            g2 = make_wiring(cids, LEGACY_WIRING_SEED + s, "logical")
            ok = all(np.array_equal(a, b) for a, b in zip(g1, g2))
            p1b = p1b and ok
    print("  P1b {}".format("HOLDS -- logical and global coincide, as predicted" if p1b
                            else "FAILS -- reconsider before interpreting ARM C"))

    if audit_only:
        print("\n--audit: stopping before the sweep.")
        sys.exit(0)

    # ---- [3] the sweep ------------------------------------------------------
    n_total = len(KNOWLEDGE_FRACS) * len(PLACEMENT_SEEDS) * len(DYN_SEEDS) * len(ARMS)
    print("\n[3] Growth per (knowledge_frac, placement_seed), {} dynamics seeds, {} arms -- {} total runs".format(
        len(DYN_SEEDS), len(ARMS), n_total))
    cache = _cache_load()
    seed_means_by_arm_kf = {}
    t_start = time.time()
    first = True
    for kf in KNOWLEDGE_FRACS:
        W = build_real_graph(act_t, act_d, kf)
        for arm_label, inhib_mode, resample in ARMS:
            seed_means = []
            for pseed in PLACEMENT_SEEDS:
                coords, cids = make_placement_real(W, seed=pseed)
                prefix = "{}|kf{:.2f}|ps{}".format(arm_label, kf, pseed)
                t0 = time.time()
                g = bridge_of_corrected(coords, cids, target_samples, distractor_samples,
                                        inhib_mode, resample, DYN_SEEDS, cache, prefix)
                seed_means.append(g.mean())
                if first:
                    dt = time.time() - t0
                    print("    [first (kf,arm,pseed) took {:.0f}s -> full sweep ~{:.0f} min]".format(
                        dt, n_total / len(DYN_SEEDS) * dt / 60.0))
                    first = False
            seed_means_by_arm_kf[(arm_label, kf)] = np.array(seed_means)
            m = seed_means_by_arm_kf[(arm_label, kf)]
            print("  {:<14} kf={:>5.0%}  placement-seed means: mean={:+8.1f}  sd={:>7.1f}  se={:>6.1f}".format(
                arm_label, kf, m.mean(), m.std(ddof=1), m.std(ddof=1) / np.sqrt(len(PLACEMENT_SEEDS))))
            sys.stdout.flush()
    print("\n  sweep wall time: {:.1f} min".format((time.time() - t_start) / 60.0))

    # ---- [4] recovery curve per arm ----------------------------------------
    print("\n" + "=" * 96)
    print("[4] Recovery curve per arm, unit of replication = placement seed (n={})".format(len(PLACEMENT_SEEDS)))
    print("=" * 96)
    arm_summary = {}
    for arm_label, _, _ in ARMS:
        pop_means = seed_means_by_arm_kf[(arm_label, 0.0)]
        fn_means = seed_means_by_arm_kf[(arm_label, 1.0)]
        pop_g, fn_g = pop_means.mean(), fn_means.mean()
        span = fn_g - pop_g
        print("\n  ARM {}: gap population {:+.1f} -> functional {:+.1f}  (span {:.1f})".format(
            arm_label, pop_g, fn_g, span))
        results_summary = []
        for kf in KNOWLEDGE_FRACS:
            m = seed_means_by_arm_kf[(arm_label, kf)]
            se = m.std(ddof=1) / np.sqrt(len(PLACEMENT_SEEDS))
            pct = (m.mean() - pop_g) / span if span != 0 else float("nan")
            t, p = stats.ttest_ind(m, pop_means, equal_var=False)
            results_summary.append((kf, m.mean(), se, pct, p))
            print("  {:>9.0%} {:>+10.1f} {:>8.1f} {:>11.1%} {:>14.2e}".format(kf, m.mean(), se, pct, p))
        arm_summary[arm_label] = results_summary

    # ---- [5] resolvability per arm -----------------------------------------
    print("\n" + "=" * 96)
    print("[5] Adjacent-point resolvability (2-SE rule), per arm")
    print("=" * 96)
    arm_resolvable = {}
    for arm_label, _, _ in ARMS:
        results_summary = arm_summary[arm_label]
        resolvable_curve = []
        print("\n  ARM {}".format(arm_label))
        for i in range(1, len(results_summary)):
            kf0, m0, se0, _, _ = results_summary[i - 1]
            kf1, m1, se1, _, _ = results_summary[i]
            diff = m1 - m0
            combined_se = np.sqrt(se0 ** 2 + se1 ** 2)
            resolvable = abs(diff) > 2 * combined_se
            resolvable_curve.append(resolvable)
            print("    {:>5.0%} -> {:>5.0%}: diff={:+8.1f}  2*SE={:>7.1f}  {}".format(
                kf0, kf1, diff, 2 * combined_se, "RESOLVABLE" if resolvable else "flat"))
        arm_resolvable[arm_label] = resolvable_curve

    # ---- [6] ARM A fidelity + verdict --------------------------------------
    print("\n" + "=" * 96)
    print("[6] VERDICT")
    print("=" * 96)

    def recovery_points(results_summary):
        f50 = next((kf for kf, m, se, pct, p in results_summary if pct >= 0.5), None)
        f90 = next((kf for kf, m, se, pct, p in results_summary if pct >= 0.9), None)
        return f50, f90

    a50, a90 = recovery_points(arm_summary["A_replication"])
    c50, c90 = recovery_points(arm_summary["C_corrected"])
    a50s = "{:.0%}".format(a50) if a50 is not None else "never"
    c50s = "{:.0%}".format(c50) if c50 is not None else "never"
    a90s = "{:.0%}".format(a90) if a90 is not None else "never"
    c90s = "{:.0%}".format(c90) if c90 is not None else "never"
    n_res_a = sum(arm_resolvable["A_replication"])
    n_res_c = sum(arm_resolvable["C_corrected"])
    print("  ARM A (should match Exp 49c's published '50% at 75%, 90% at 100%'):")
    print("    50% recovery at {}, 90% recovery at {}, {}/{} steps resolvable".format(
        a50s, a90s, n_res_a, len(arm_resolvable["A_replication"])))
    print("  ARM C (corrected wiring):")
    print("    50% recovery at {}, 90% recovery at {}, {}/{} steps resolvable".format(
        c50s, c90s, n_res_c, len(arm_resolvable["C_corrected"])))

    max_abs_pct_change = 0.0
    print("\n  Per-knowledge_frac point estimate, A vs C:")
    for kf in KNOWLEDGE_FRACS:
        a = seed_means_by_arm_kf[("A_replication", kf)].mean()
        c = seed_means_by_arm_kf[("C_corrected", kf)].mean()
        pct_change = 100.0 * (c - a) / abs(a) if a != 0 else float("nan")
        max_abs_pct_change = max(max_abs_pct_change, abs(pct_change))
        print("    {:>5.0%}: A={:+8.1f}  C={:+8.1f}  ({:+.1f}%)".format(kf, a, c, pct_change))

    print("\n  max |point-estimate change| across all knowledge_fracs: {:.1f}%".format(max_abs_pct_change))
    if a50 == c50 and a90 == c90 and max_abs_pct_change < 15.0:
        print("  ROBUST: recovery thresholds and point estimates are stable under corrected")
        print("  wiring. Section 4.11's real-data leg (Exp 49c) stands as published.")
    else:
        print("  NOT ROBUST: recovery thresholds and/or point estimates moved materially under")
        print("  corrected wiring. Section 4.11's real-data leg needs re-derivation from these")
        print("  ARM C numbers, the same way Section 4.5 was corrected from Exp 53.")
