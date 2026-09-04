"""Exp 53: Is the headline real-data ratio a property of PLACEMENT, or of one
fixed wiring draw?

WHY THIS AUDIT EXISTS. `run_life` in exp37b_v2_real_data.py (lines 92-96)
draws the initial connectivity and the inhibitory mask from a HARDCODED
seed:

    rng  = np.random.default_rng(seed)      # per-run seed: input noise only
    rng2 = np.random.default_rng(7)         # FIXED -- ignores `seed`
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    inhib = rng2.random(N) < 0.20

Two consequences, both verified before this script was written:

  (1) Initial connectivity and inhibition are IDENTICAL across every
      "seed" of every real-data experiment. The `seed` argument varies
      only the input noise stream. So the published across-seed SDs and
      bootstrap CIs -- exp37c's [5.05x, 11.48x], the preprint's
      [7.03x, 10.05x] -- contain NO wiring-draw variance. They are
      conditional on one arbitrary draw, and understate uncertainty with
      respect to that factor. This is a scope limitation on the reported
      interval, not in itself an error in the point estimate.

  (2) `src`/`dst` index GLOBAL neuron ids, but which global id holds which
      TYPE depends on the placement: segregated puts inputs at indices
      0-139 with outputs contiguous at 140-199, while interleaved spreads
      both across 0-199. The identical index arrays therefore realise
      DIFFERENT logical connectivity in the two conditions. exp52 measured
      the consequence: input->output edge counts matched (419 vs 418) but
      EXCITATORY input->output counts did not (313 vs 341, an 8.2% gap),
      and exp52b showed that changing the wiring seed both shrank that gap
      to 1-2% and flipped its sign -- i.e. the imbalance is a property of
      default_rng(7), not of the placements.

That second point is the serious one. It means the published comparison
may differ in initial excitatory drive as well as in geometry, and
exp52b reported in passing that under resampled wiring the bridge-mass
ratio falls to about 3.6x rather than 8.2x. This script settles it.

THE THREE ARMS. All at exp37c's MATCHED-LIFE headline configuration
(20 distinct samples/class, 1 epoch, ~8,600 steps, V retention 0.651),
reusing `load_two_class_samples` and both placement functions verbatim:

  ARM A  REPLICATION      wiring_seed = 7 (fixed), global-index wiring.
                          Exactly the published configuration. Confirms
                          this harness is faithful before anything is
                          concluded from B or C.
  ARM B  RESAMPLED        wiring_seed varies per run seed, still
                          global-index wiring. Isolates the effect of
                          un-pinning the draw, while retaining the
                          logical-connectivity asymmetry of point (2).
  ARM C  LOGICAL SPACE    wiring_seed varies per run seed AND wiring is
                          drawn over LOGICAL identities (type, within-type
                          rank) then mapped through each placement, so both
                          conditions realise the IDENTICAL logical network
                          and differ only in coordinates. This is the
                          properly-controlled number, and the same
                          invariant Exp 50 secured by construction.

Arms B and C reuse `run_life_corrected` / `make_wiring` from
exp52b_accuracy_corrected.py rather than reimplementing them, so no
transcription divergence is possible; exp52b already asserts that in
legacy mode (wiring_seed=7, inhib_mode='global') it reproduces the
imported `run_life` BIT-IDENTICALLY, and P0 below re-asserts it here.

WHAT EACH OUTCOME MEANS (fixed before running):
  - ARM C ratio close to ARM A: the headline is robust to the wiring
    draw. Report a clean bill of health, and note only that published
    CIs omit wiring variance.
  - ARM C ratio materially below ARM A (e.g. near 3.6x): the published
    8.22x is inflated by one arbitrary draw and the headline needs
    correcting to ARM C's value and CI. Report that plainly. This
    project has retracted a 9.31x and a 1,448x before; a third
    correction is normal practice, not a crisis.
  - ARM B between A and C: quantifies how much is the pinned draw
    (A->B) versus the logical-connectivity asymmetry (B->C).
  - If ARM A fails to reproduce roughly the published ratio, this
    harness is not faithful and NOTHING below should be interpreted.

Scope note: exp51's SSC numbers import the same `run_life` and inherit
whatever this finds. This script does not re-run SSC.

Run:            python src/experiments/exp53_wiring_draw_audit.py
Run (audit):    python src/experiments/exp53_wiring_draw_audit.py --audit
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

from exp37b_v2_real_data import (
    make_placement_segregated, make_placement_interleaved, run_life,
    INPUT_TYPE, N,
)
from exp37c_real_data_scaled import (
    load_two_class_samples, cohens_d_pooled, bootstrap_ratio_ci,
    MIN_RETENTION, STEPS_PER_SAMPLE,
)
from exp52b_accuracy_corrected import (
    run_life_corrected, make_wiring, excitatory_io_stats,
    LEGACY_WIRING_SEED,
)

# ---- registered configuration (fixed before running) ----
SAMPLES_PER_CLASS = 20
N_EPOCHS = 1
POOL_SIZE = 1500
SEEDS = list(range(20))          # 20 seeds/condition/arm; 0-11 is the published set

ARMS = [
    ("A_replication", "global", False),   # (label, inhib_mode, resample_wiring)
    ("B_resampled", "global", True),
    ("C_logical", "logical", True),
]

CACHE_PATH = os.path.join("shd_data", "exp53_cache.json")


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


def wiring_seed_for(arm_resample, run_seed):
    """Fixed draw reproduces the published config; resampled arms vary the
    draw with the run seed so wiring variance enters the across-seed SD."""
    return LEGACY_WIRING_SEED if not arm_resample else LEGACY_WIRING_SEED + run_seed


def arm_structural_stats(cids, inhib_mode, resample):
    """Input->output edge counts and their excitatory subset, averaged over
    the registered seeds. This is the quantity that silently differed."""
    ios, excs = [], []
    for s in SEEDS:
        ws = wiring_seed_for(resample, s)
        src, dst, inhib = make_wiring(cids, ws, inhib_mode)
        io, exc, _ = excitatory_io_stats(cids, src, dst, inhib)
        ios.append(io); excs.append(exc)
    return float(np.mean(ios)), float(np.mean(excs)), float(np.std(excs, ddof=1))


def report_arm(label, seg, inter):
    ratio = inter.mean() / max(seg.mean(), 1e-9)
    lo, hi = bootstrap_ratio_ci(inter, seg)
    t_ind, p_ind = stats.ttest_ind(inter, seg, equal_var=False)
    d = cohens_d_pooled(inter, seg)
    rel_seg = 100 * seg.std(ddof=1) / max(seg.mean(), 1e-9)
    print("  {:<16} seg={:8.1f} +/-{:7.1f} (rel {:5.1f}%)  int={:8.1f} +/-{:7.1f}"
          .format(label, seg.mean(), seg.std(ddof=1), rel_seg,
                  inter.mean(), inter.std(ddof=1)))
    print("  {:<16} ratio={:6.2f}x  95% CI [{:.2f}x, {:.2f}x]  Welch p={:.3e}  d={:.2f}"
          .format("", ratio, lo, hi, p_ind, d))
    return dict(ratio=ratio, lo=lo, hi=hi, p=p_ind, d=d,
                seg_mean=seg.mean(), seg_sd=seg.std(ddof=1),
                int_mean=inter.mean(), int_sd=inter.std(ddof=1),
                rel_seg=rel_seg)


if __name__ == "__main__":
    print("=" * 96)
    print("EXP 53: is the headline real-data ratio a property of PLACEMENT,")
    print("        or of the single fixed wiring draw default_rng(7)?")
    print("=" * 96)

    top, avail, tgt, dis = load_two_class_samples(SAMPLES_PER_CLASS, POOL_SIZE)
    est_steps = 2 * len(tgt) * N_EPOCHS * STEPS_PER_SAMPLE
    retention = 0.99995 ** est_steps
    print("\n[0] Configuration (exp37c MATCHED-LIFE headline)")
    print("  classes: target={} (avail {}), distractor={} (avail {})".format(
        top[0], avail[top[0]], top[1], avail[top[1]]))
    print("  {} samples/class x {} epoch(s) -> ~{:,} steps, V retention {:.3f}".format(
        len(tgt), N_EPOCHS, est_steps, retention))
    assert retention >= MIN_RETENTION, "retention {:.3f} below {}".format(
        retention, MIN_RETENTION)
    print("  seeds: {}-{} ({} per condition per arm); published set was 0-11".format(
        SEEDS[0], SEEDS[-1], len(SEEDS)))

    conditions = [("segregated", make_placement_segregated),
                  ("interleaved", make_placement_interleaved)]
    coords_map = {name: fn() for name, fn in conditions}

    # ---- P0: equivalence, non-negotiable -------------------------------
    print("\n[1] P0 equivalence: run_life_corrected in legacy mode vs imported run_life")
    c0, ci0 = coords_map["segregated"]
    ba_ref, bb_ref = run_life(c0, ci0, 0, tgt[:4], dis[:4], n_epochs=1)
    ba_new, bb_new = run_life_corrected(
        c0, ci0, 0, tgt[:4], dis[:4], n_epochs=1,
        wiring_seed=LEGACY_WIRING_SEED, inhib_mode="global")[:2]
    print("    imported run_life  : bridge_a={:.10f}  bridge_b={:.10f}".format(ba_ref, bb_ref))
    print("    run_life_corrected : bridge_a={:.10f}  bridge_b={:.10f}".format(ba_new, bb_new))
    p0 = (ba_ref == ba_new) and (bb_ref == bb_new)
    print("    P0 {}".format("HOLDS (bit-identical) -- arms B/C use the same physics"
                             if p0 else "FAILS -- COMPROMISED, stopping"))
    if not p0:
        sys.exit(1)

    # ---- structural audit, before spending hours -----------------------
    print("\n[2] Structural audit: input->output edges and their excitatory subset")
    print("    (the quantity that silently differed between conditions)")
    struct = {}
    for arm_label, inhib_mode, resample in ARMS:
        print("  {}".format(arm_label))
        for cname, (coords, cids) in coords_map.items():
            io, exc, exc_sd = arm_structural_stats(cids, inhib_mode, resample)
            struct[(arm_label, cname)] = (io, exc, exc_sd)
            print("    {:<12} io_edges={:7.1f}  excitatory_io={:7.1f} +/-{:5.1f}".format(
                cname, io, exc, exc_sd))
        e_s = struct[(arm_label, "segregated")][1]
        e_i = struct[(arm_label, "interleaved")][1]
        gap = 100.0 * (e_i - e_s) / max(e_s, 1e-9)
        print("    excitatory io gap (interleaved vs segregated): {:+.2f}%".format(gap))

    if "--audit" in sys.argv:
        print("\n--audit: stopping before the sweep.")
        sys.exit(0)

    # ---- the sweep -----------------------------------------------------
    print("\n[3] Running {} arms x 2 conditions x {} seeds = {} runs".format(
        len(ARMS), len(SEEDS), len(ARMS) * 2 * len(SEEDS)))
    cache = _cache_load()
    results = {}
    t_start = time.time()
    n_done = 0
    n_total = len(ARMS) * 2 * len(SEEDS)

    for arm_label, inhib_mode, resample in ARMS:
        print("\n  === ARM {} (inhib_mode={}, resample_wiring={}) ===".format(
            arm_label, inhib_mode, resample))
        for cname, (coords, cids) in coords_map.items():
            totals = []
            for s in SEEDS:
                key = "{}|{}|{}|spc{}|ep{}".format(
                    arm_label, cname, s, SAMPLES_PER_CLASS, N_EPOCHS)
                if key in cache:
                    ba, bb = cache[key]
                    cached = True
                else:
                    ws = wiring_seed_for(resample, s)
                    t0 = time.time()
                    ba, bb = run_life_corrected(
                        coords, cids, s, tgt, dis, n_epochs=N_EPOCHS,
                        wiring_seed=ws, inhib_mode=inhib_mode)[:2]
                    ba, bb = float(ba), float(bb)
                    cache[key] = [ba, bb]
                    _cache_save(cache)
                    cached = False
                    if n_done == 0:
                        dt = time.time() - t0
                        print("    [first run {:.0f}s -> full sweep ~{:.1f}h]".format(
                            dt, n_total * dt / 3600.0))
                totals.append(ba + bb)
                n_done += 1
            results[(arm_label, cname)] = np.array(totals, dtype=float)
            print("    {:<12} n={}  mean={:8.1f} +/- {:7.1f}{}".format(
                cname, len(totals), np.mean(totals), np.std(totals, ddof=1),
                "  (cached)" if cached else ""))
            sys.stdout.flush()

    print("\n  sweep wall time: {:.1f} min".format((time.time() - t_start) / 60.0))

    # ---- per-arm statistics --------------------------------------------
    print("\n" + "=" * 96)
    print("[4] Per-arm results")
    print("=" * 96)
    summ = {}
    for arm_label, _, _ in ARMS:
        summ[arm_label] = report_arm(
            arm_label,
            results[(arm_label, "segregated")],
            results[(arm_label, "interleaved")])
        print()

    # ---- cross-arm comparison ------------------------------------------
    print("=" * 96)
    print("[5] Cross-arm comparison")
    print("=" * 96)
    a, b, c = summ["A_replication"], summ["B_resampled"], summ["C_logical"]
    print("  ARM A (published config, pinned draw) : {:.2f}x  CI [{:.2f}, {:.2f}]".format(
        a["ratio"], a["lo"], a["hi"]))
    print("  ARM B (draw resampled, global wiring) : {:.2f}x  CI [{:.2f}, {:.2f}]".format(
        b["ratio"], b["lo"], b["hi"]))
    print("  ARM C (resampled, LOGICAL wiring)     : {:.2f}x  CI [{:.2f}, {:.2f}]".format(
        c["ratio"], c["lo"], c["hi"]))
    print()
    print("  preprint headline for this regime     : 8.22x CI [7.03, 10.05]")
    print("  exp37c docstring for this regime      : 7.25x CI [5.05, 11.48]")
    print()
    print("  attribution of any change:")
    print("    A -> B  (un-pinning the wiring draw)      : {:+.2f}x".format(
        b["ratio"] - a["ratio"]))
    print("    B -> C  (removing logical asymmetry)      : {:+.2f}x".format(
        c["ratio"] - b["ratio"]))
    print("    A -> C  (total)                           : {:+.2f}x".format(
        c["ratio"] - a["ratio"]))
    print()
    print("  uncertainty width (bootstrap CI, hi-lo):")
    for lbl, sdict in (("A", a), ("B", b), ("C", c)):
        print("    ARM {}: {:.2f}x wide   segregated rel-SD {:.1f}%".format(
            lbl, sdict["hi"] - sdict["lo"], sdict["rel_seg"]))
    widen = (c["hi"] - c["lo"]) / max(a["hi"] - a["lo"], 1e-9)
    print("    resampling wiring widens the interval by {:.2f}x".format(widen))

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 96)
    print("[6] VERDICT")
    print("=" * 96)
    a_ok = 6.0 <= a["ratio"] <= 10.5
    print("  ARM A reproduces the published regime (6.0-10.5x): {}".format(
        "YES ({:.2f}x)".format(a["ratio"]) if a_ok
        else "NO ({:.2f}x) -- harness not faithful, do NOT interpret B/C".format(a["ratio"])))
    if not a_ok:
        sys.exit(0)

    overlap = not (c["hi"] < a["lo"] or a["hi"] < c["lo"])
    print("  ARM A and ARM C bootstrap intervals overlap: {}".format(overlap))
    rel_drop = 100.0 * (a["ratio"] - c["ratio"]) / max(a["ratio"], 1e-9)
    print("  ARM C is {:+.1f}% relative to ARM A".format(-rel_drop))
    print()
    if overlap and abs(rel_drop) < 25.0:
        print("  ROBUST: the headline survives resampling the wiring draw and")
        print("  removing the logical-connectivity asymmetry. The published point")
        print("  estimate stands; the only correction needed is that published CIs")
        print("  are conditional on one wiring draw and understate uncertainty.")
    else:
        print("  NOT ROBUST: the published ratio is substantially conditional on the")
        print("  fixed draw default_rng(7) and/or on the logical-connectivity")
        print("  asymmetry between conditions. ARM C is the properly-controlled")
        print("  number and the headline should be corrected to:")
        print("      {:.2f}x, bootstrap 95% CI [{:.2f}x, {:.2f}x], Welch p={:.2e}, d={:.2f}".format(
            c["ratio"], c["lo"], c["hi"], c["p"], c["d"]))
        print("  exp51's SSC numbers import the same run_life and inherit this.")
