"""Exp 49b: what are Exp 49's real-data partial-knowledge numbers under
CORRECTED wiring?

WHY THIS EXISTS. Exp 49 (commit 82e1911, 2026-08-29) measured the
SpiNeCluster partial-knowledge curve on a real, data-derived SHD
association graph and is folded into the preprint as Section 4.11's
real-data leg. It imports `run_life` verbatim from `exp37b_v2_real_data.py`
-- the same harness Exp 53 later showed (2026-09-03/04, five days after
Exp 49 was committed) pins initial connectivity and the inhibitory mask to
a FIXED `np.random.default_rng(7)` draw that ignores the per-run seed:

    rng2 = np.random.default_rng(7)         # FIXED -- ignores `seed`
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    inhib = rng2.random(N) < 0.20

Exp 49 was never re-run under the corrected wiring. Unlike Section 4.6/4.8
(exp38/exp42, synthetic-side, understated variance only per Exp 54), the
real-data correction (Exp 53/51b) shifted POINT ESTIMATES, not just error
bars, so this omission is a real gap rather than a cosmetic one.

WHAT IS DIFFERENT HERE, AND WHY THE PRIOR REAL-DATA BLAST RADIUS MAY NOT
APPLY IN FULL. Exp 53's mechanism had two components: (1) every seed
reuses the same wiring draw (zero wiring-draw variance), and (2) src/dst
index GLOBAL neuron ids while placement decides which id holds which
type, so the identical pinned array realises DIFFERENT logical
connectivity in segregated vs interleaved. Exp 49 has no such placement
dependency: `CIDS` is fixed at import time as
`[INPUT_TYPE]*N_INPUT + [OUTPUT_A_TYPE]*N_OUTPUT_A + [OUTPUT_B_TYPE]*N_OUTPUT_B`
-- already canonically ordered by type -- and `make_placement_real`
returns `CIDS.copy()` unchanged for every knowledge_frac; only the
COORDINATES (which core a cluster lands on) vary. So component (2), the
mechanism that did almost all of the damage on SHD/SSC (Exp 53 attributed
-6.11x of -6.18x total to component (1) alone, i.e. component (2)
contributed ~0), predicts even less here: cids order never changes, so the
pinned draw is expected to realise IDENTICAL logical connectivity at every
knowledge_frac. This is a prediction, registered before running, not an
assumption acted on: P0 and P1b below verify it rather than skip it.

Consequently the predicted (not yet confirmed) outcome is that this
correction mainly fixes UNDERSTATED VARIANCE across the 8 life-seeds
within each knowledge_frac (Exp 49 seeds were all life-seeds, and none of
them varied the wiring), similar to Exp 54's finding for the synthetic
benchmark, rather than shifting the recovery-curve shape the way Exp 53
shifted the SHD headline ratio. If the recovery percentages or the P2
verdict (cliff vs. smooth) move materially anyway, that is the important
finding and this docstring's prediction is wrong.

THE TWO ARMS, exactly Exp 51b's pattern, reusing the already-audited
`run_life_corrected` / `make_wiring` from `exp52b_accuracy_corrected.py`
rather than reimplementing:

  ARM A  REPLICATION   wiring_seed = 7 (fixed) for every life-seed,
                       global-index wiring. Exactly Exp 49's published
                       configuration; existence checked against the
                       imported `run_life` bit-identically (P0).
  ARM C  CORRECTED     wiring_seed = 7 + life_seed (resampled per seed),
                       wiring drawn over LOGICAL identities then mapped
                       through the (identity, for this experiment)
                       placement. Given CIDS never changes, logical and
                       global wiring are predicted to coincide here (P1b
                       checks this directly rather than assuming it).

Placement (`make_placement_real`, SpiNeCluster + particle-swarm placement,
seed=0) is UNCHANGED and untouched by either arm -- it depends on
`np.random.default_rng(seed)` inside `spinecluster`/`particle_swarm_placement`,
not on `run_life`'s internal wiring, so it is not part of the defect and is
reused verbatim from Exp 49.

WHAT EACH OUTCOME MEANS (fixed before running):
  - ARM C tracks ARM A closely at every knowledge_frac: the real-data leg
    of Section 4.11 is robust to the wiring draw, as predicted above by
    the fixed-cids argument, and only needs a note that its across-seed
    SD previously omitted wiring variance.
  - ARM C shifts materially at one or more knowledge_frac points, or the
    recovery thresholds (50%/90%) or the P2 cliff-vs-smooth verdict
    change: the prediction above is wrong, Section 4.11 needs correcting
    the way Section 4.5 was, and that is reported plainly.
  - P1b (logical == global wiring here) fails: the fixed-cids argument
    above is wrong and the two arms are not comparing what this docstring
    claims; stop and reconsider before interpreting anything.

Run:          python src/experiments/exp49b_shd_partial_knowledge_corrected.py
Run (audit):  python src/experiments/exp49b_shd_partial_knowledge_corrected.py --audit
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from scipy import stats

from exp37b_v2_real_data import N, INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE, run_life
from exp49_shd_partial_knowledge import (
    load_two_class_samples, channel_activity, build_real_graph,
    make_placement_real, KNOWLEDGE_FRACS, SAMPLES_PER_CLASS, POOL_SIZE,
    N_EPOCHS, SEEDS, CIDS,
)
from exp52b_accuracy_corrected import run_life_corrected, make_wiring, LEGACY_WIRING_SEED

CACHE_PATH = os.path.join("shd_data", "exp49b_cache.json")

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


def wiring_seed_for(resample, run_seed):
    return LEGACY_WIRING_SEED if not resample else LEGACY_WIRING_SEED + run_seed


def bridge_of_corrected(coords, cids, target_samples, distractor_samples,
                        inhib_mode, resample, seeds=SEEDS, cache=None, cache_prefix=""):
    out = []
    for s in seeds:
        key = "{}|{}".format(cache_prefix, s)
        if cache is not None and key in cache:
            ba, bb = cache[key]
        else:
            ws = wiring_seed_for(resample, s)
            ba, bb = run_life_corrected(
                coords, cids, s, target_samples, distractor_samples,
                n_epochs=N_EPOCHS, wiring_seed=ws, inhib_mode=inhib_mode)[:2]
            ba, bb = float(ba), float(bb)
            if cache is not None:
                cache[key] = [ba, bb]
                _cache_save(cache)
        out.append(ba + bb)
    return np.array(out, dtype=float)


if __name__ == "__main__":
    audit_only = "--audit" in sys.argv
    print("=" * 96)
    print("EXP 49b: SHD partial-knowledge numbers under CORRECTED wiring")
    print("=" * 96)

    print("\n[0] Loading real SHD samples ({} pool, {} per class)".format(POOL_SIZE, SAMPLES_PER_CLASS))
    t0 = time.time()
    top, avail, target_samples, distractor_samples = load_two_class_samples(SAMPLES_PER_CLASS, POOL_SIZE)
    got = min(len(target_samples), len(distractor_samples))
    print("  classes: target={} (avail {}), distractor={} (avail {})".format(
        top[0], avail[top[0]], top[1], avail[top[1]]))
    print("  using {} samples/class (loaded in {:.0f}s)".format(got, time.time() - t0))

    act_t = channel_activity(target_samples)
    act_d = channel_activity(distractor_samples)

    # ---- [1] P0 equivalence: run_life_corrected(legacy) vs imported run_life ----
    print("\n[1] P0 equivalence: run_life_corrected(legacy) vs imported run_life")
    p0 = True
    for kf in (0.0, 1.0):
        W = build_real_graph(act_t, act_d, kf)
        coords, cids = make_placement_real(W, seed=0)
        ba_ref, bb_ref = run_life(coords, cids, SEEDS[0], target_samples, distractor_samples, n_epochs=N_EPOCHS)
        ba_new, bb_new = run_life_corrected(
            coords, cids, SEEDS[0], target_samples, distractor_samples, n_epochs=N_EPOCHS,
            wiring_seed=LEGACY_WIRING_SEED, inhib_mode="global")[:2]
        ok = (ba_ref == ba_new) and (bb_ref == bb_new)
        p0 = p0 and ok
        print("  kf={:.2f}: imported run_life bridge_a={:.10f} bridge_b={:.10f}".format(kf, ba_ref, bb_ref))
        print("           run_life_corrected bridge_a={:.10f} bridge_b={:.10f}  {}".format(
            ba_new, bb_new, "OK" if ok else "MISMATCH"))
    print("  P0 {}".format("HOLDS (bit-identical) -- ARM C uses the same physics" if p0
                           else "FAILS -- COMPROMISED, stopping"))
    if not p0:
        sys.exit(1)

    # ---- [2] P1b: does logical wiring coincide with global here, as predicted? ----
    print("\n[2] P1b: logical vs global wiring, at kf=0.0 and kf=1.0 (CIDS is fixed, so these")
    print("    are predicted to be identical; this checks that rather than assuming it)")
    p1b = True
    for kf in (0.0, 1.0):
        W = build_real_graph(act_t, act_d, kf)
        coords, cids = make_placement_real(W, seed=0)
        for s in SEEDS[:3]:
            src_g, dst_g, inh_g = make_wiring(cids, LEGACY_WIRING_SEED + s, "global")
            src_l, dst_l, inh_l = make_wiring(cids, LEGACY_WIRING_SEED + s, "logical")
            ok = (np.array_equal(src_g, src_l) and np.array_equal(dst_g, dst_l)
                  and np.array_equal(inh_g, inh_l))
            p1b = p1b and ok
            if not ok:
                print("  kf={:.2f} seed={}: MISMATCH (logical != global)".format(kf, s))
    print("  P1b {}".format("HOLDS -- logical and global wiring coincide, as predicted"
                            if p1b else "FAILS -- the fixed-cids argument is wrong, reconsider"))

    if audit_only:
        print("\n--audit: stopping before the sweep.")
        sys.exit(0)

    # ---- [3] the sweep --------------------------------------------------
    print("\n[3] Growth (bridge_a+bridge_b) vs knowledge_frac, {} seeds each, {} arms".format(
        len(SEEDS), len(ARMS)))
    cache = _cache_load()
    results = {}
    t_start = time.time()
    first = True
    for kf in KNOWLEDGE_FRACS:
        W = build_real_graph(act_t, act_d, kf)
        coords, cids = make_placement_real(W, seed=0)
        for arm_label, inhib_mode, resample in ARMS:
            prefix = "{}|kf{:.2f}".format(arm_label, kf)
            t0 = time.time()
            g = bridge_of_corrected(coords, cids, target_samples, distractor_samples,
                                    inhib_mode, resample, cache=cache, cache_prefix=prefix)
            results[(arm_label, kf)] = g
            print("  {:<14} knowledge_frac={:>5.0%}  bridge={:+8.1f} +/- {:<6.1f}".format(
                arm_label, kf, g.mean(), g.std()))
            if first:
                dt = time.time() - t0
                n_total = len(KNOWLEDGE_FRACS) * len(ARMS)
                print("    [first knowledge_frac x arm took {:.0f}s -> full sweep ~{:.1f} min]".format(
                    dt, n_total * dt / 60.0))
                first = False
            sys.stdout.flush()
    print("\n  sweep wall time: {:.1f} min".format((time.time() - t_start) / 60.0))

    # ---- [4] per-arm recovery tables -------------------------------------
    print("\n" + "=" * 96)
    print("[4] Recovery: what fraction of the population->functional gap does each level close?")
    print("=" * 96)
    arm_recovery = {}
    for arm_label, _, _ in ARMS:
        pop_g = results[(arm_label, 0.0)].mean()
        fn_g = results[(arm_label, 1.0)].mean()
        span = fn_g - pop_g
        print("\n  ARM {}: gap population {:+.1f} -> functional {:+.1f}  (span {:.1f})".format(
            arm_label, pop_g, fn_g, span))
        print("  {:>14} {:>10} {:>12} {:>14}".format("knowledge", "bridge", "%recovered", "Welch p vs pop"))
        frac_at_50, frac_at_90 = None, None
        for kf in KNOWLEDGE_FRACS:
            g = results[(arm_label, kf)]
            pct = (g.mean() - pop_g) / span if span != 0 else float("nan")
            t, p = stats.ttest_ind(g, results[(arm_label, 0.0)], equal_var=False)
            print("  {:>13.0%} {:>+10.1f} {:>11.1%} {:>14.2e}".format(kf, g.mean(), pct, p))
            if frac_at_50 is None and pct >= 0.5:
                frac_at_50 = kf
            if frac_at_90 is None and pct >= 0.9:
                frac_at_90 = kf
        arm_recovery[arm_label] = (frac_at_50, frac_at_90, pop_g, fn_g, span)

    # ---- [5] arm comparison at each knowledge_frac ------------------------
    print("\n" + "=" * 96)
    print("[5] ARM A vs ARM C, per knowledge_frac")
    print("=" * 96)
    print("  {:>10} {:>18} {:>18} {:>10}".format("kf", "A (replication)", "C (corrected)", "% change"))
    max_abs_pct_change = 0.0
    for kf in KNOWLEDGE_FRACS:
        a = results[("A_replication", kf)].mean()
        c = results[("C_corrected", kf)].mean()
        pct_change = 100.0 * (c - a) / abs(a) if a != 0 else float("nan")
        max_abs_pct_change = max(max_abs_pct_change, abs(pct_change))
        print("  {:>9.0%} {:>+17.1f} {:>+17.1f} {:>+9.1f}%".format(kf, a, c, pct_change))

    # ---- [6] verdict -------------------------------------------------------
    print("\n" + "=" * 96)
    print("[6] VERDICT")
    print("=" * 96)
    a50, a90, a_pop, a_fn, a_span = arm_recovery["A_replication"]
    c50, c90, c_pop, c_fn, c_span = arm_recovery["C_corrected"]
    a50_s = "{:.0%}".format(a50) if a50 is not None else "never"
    c50_s = "{:.0%}".format(c50) if c50 is not None else "never"
    a90_s = "{:.0%}".format(a90) if a90 is not None else "never"
    c90_s = "{:.0%}".format(c90) if c90 is not None else "never"
    print("  50% recovery: ARM A at {}  ->  ARM C at {}".format(a50_s, c50_s))
    print("  90% recovery: ARM A at {}  ->  ARM C at {}".format(a90_s, c90_s))
    print("  max |point-estimate change| across all knowledge_fracs: {:.1f}%".format(max_abs_pct_change))
    print()
    if p1b and max_abs_pct_change < 15.0 and a50 == c50 and a90 == c90:
        print("  ROBUST, AS PREDICTED: the fixed-cids argument holds -- ARM C's point estimates")
        print("  track ARM A closely (max change {:.1f}%) and the recovery thresholds are")
        print("  unchanged (50% at {}, 90% at {}). Section 4.11's real-data leg stands; only".format(
            c50_s, c90_s))
        print("  its across-seed SD needs a note that it previously omitted wiring variance.")
    else:
        print("  NOT ROBUST, OR THE PREDICTION FAILED: either P1b failed (logical != global here,")
        print("  contradicting the fixed-cids argument above) or the point estimates / recovery")
        print("  thresholds moved materially under corrected wiring. Section 4.11's real-data leg")
        print("  needs the same kind of correction Section 4.5 received. Corrected values:")
        for kf in KNOWLEDGE_FRACS:
            g = results[("C_corrected", kf)]
            print("      knowledge_frac={:>5.0%}: bridge={:+8.1f} +/- {:<6.1f}".format(kf, g.mean(), g.std()))
        print("      50% recovery at {}, 90% recovery at {} (ARM C)".format(c50_s, c90_s))
