"""Exp 51b: What are Exp 51's SSC numbers under CORRECTED wiring?

WHY THIS EXISTS. Exp 51 established a second-corpus replication on Spiking
Speech Commands -- 9.89x / 10.47x / 16.07x across three disjoint class
pairs, aggregate 11.61x, CI [9.33, 14.99] -- and those numbers are in the
preprint as Section 4.13. Exp 51 imports `run_life` verbatim from
exp37b_v2_real_data.py, and Exp 53 subsequently showed that `run_life`
contains a defect that inflates the SHD headline roughly threefold:

    rng  = np.random.default_rng(seed)      # per-run seed: input noise only
    rng2 = np.random.default_rng(7)         # FIXED -- ignores `seed`
    src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
    inhib = rng2.random(N) < 0.20

Consequences, both established before this script was written:
  (1) Initial connectivity and inhibition are IDENTICAL across every seed,
      so all published across-seed SDs and bootstrap CIs -- including
      Exp 51's [9.33, 14.99] -- contain ZERO wiring-draw variance and are
      conditional on one arbitrary draw.
  (2) src/dst index GLOBAL neuron ids while placement decides which id
      holds which type, so the identical arrays realise DIFFERENT logical
      connectivity in the two conditions.

Exp 53 found that on SHD, un-pinning the draw collapsed the matched-life
ratio from 8.92x to 2.74x (CI [2.27, 3.41]), with essentially the ENTIRE
change attributable to (1) rather than (2): `default_rng(7)` is an
unusually unfavourable draw for the SEGREGATED condition specifically,
and resampling raised segregated ~2.9x while interleaved barely moved.
Exp 54 found the SYNTHETIC benchmark barely moves (4.22x -> 3.83x),
because segregated's zero adjacency on the measured pairs there is a fact
of geometry that no wiring draw can rescue.

SSC is the remaining unknown in the blast radius. This script settles it.

THE TWO ARMS. Both at Exp 51's registered configuration verbatim -- the
same three class pairs, the same 16 seeds (100-115), the same matched-life
regime (20 samples/class, 1 epoch, ~8,600 steps, V retention 0.651),
reusing Exp 51's SSC loader and both placement functions:

  ARM A  REPLICATION   wiring_seed = 7 (fixed), global-index wiring.
                       Exactly Exp 51's published configuration. Confirms
                       this harness is faithful before anything is
                       concluded from ARM C.
  ARM C  CORRECTED     wiring_seed varies per run seed AND wiring is drawn
                       over LOGICAL identities (type, within-type rank)
                       then mapped through each placement, so both
                       conditions realise the IDENTICAL logical network and
                       differ only in coordinates. This is the number that
                       should go in the paper.

ARM B (resampled but still global-index) is DELIBERATELY SKIPPED. Exp 53
measured A->B = -6.11x against B->C = -0.07x, and Exp 54 measured
A->B = -0.40x against B->C = +0.01x. In both cases the logical-asymmetry
step contributed essentially nothing, so a third arm here would cost ~1.7
hours of simulation to re-establish a null that two independent audits
have already established. Skipping it is a runtime decision, not a
scope reduction: ARM C still isolates the total correction, which is the
quantity the preprint needs.

`run_life_corrected` and `make_wiring` are imported from
exp52b_accuracy_corrected.py rather than reimplemented, exactly as Exp 53
does, so no transcription divergence between the audits is possible. P0
below re-asserts bit-identical equivalence in legacy mode ON SSC DATA
(Exp 53 asserted it on SHD).

WHAT EACH OUTCOME MEANS (fixed before running):
  - ARM C ratios close to ARM A: SSC is robust to the wiring draw, unlike
    SHD. That would itself be a finding needing explanation, and Section
    4.13 would stand with only a note that its CI omits wiring variance.
  - ARM C materially below ARM A: Section 4.13's numbers are inflated by
    the same defect and must be corrected to ARM C's values.
  - ANY pair falling below Exp 51's registered 3.0x substantive floor:
    reported PROMINENTLY, not buried in the aggregate. Exp 51's headline
    conclusion was specifically that all three pairs cleared the floor
    ("not a lucky word pair"); if that no longer holds under corrected
    wiring, the conclusion must be qualified.
  - If ARM A fails to reproduce Exp 51's published per-pair ratios, this
    harness is not faithful and NOTHING below should be interpreted.

Exp 51's other registered predictions (P1a geometry regression, P1c decay
discipline) are geometry- and config-level and are unaffected by the
wiring draw; they are re-checked here for completeness.

Run:          python src/experiments/exp51b_ssc_corrected.py
Run (audit):  python src/experiments/exp51b_ssc_corrected.py --audit
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
)
from exp37c_real_data_scaled import (
    cohens_d_pooled, bootstrap_ratio_ci, MIN_RETENTION, STEPS_PER_SAMPLE,
)
# SSC loader and the registered design constants, imported verbatim from the
# experiment being corrected so the configuration cannot silently drift.
from exp51_ssc_confirmatory import (
    load_class_pair_samples_ssc, adjacency_counts,
    CLASS_PAIRS, SEEDS, SAMPLES_PER_CLASS, N_EPOCHS,
    EXPECTED_ADJ, RATIO_FLOOR,
)
from exp52b_accuracy_corrected import (
    run_life_corrected, make_wiring, excitatory_io_stats, LEGACY_WIRING_SEED,
)

# ---- arms (label, inhib_mode, resample_wiring) ----
ARMS = [
    ("A_replication", "global", False),
    ("C_corrected", "logical", True),
]

# Exp 51's published per-pair ratios, for the ARM A fidelity check.
EXP51_PUBLISHED = {"PAIR-A": 9.89, "PAIR-B": 10.47, "PAIR-C": 16.07}
EXP51_AGG = (11.61, 9.33, 14.99)

# Exp 53's corrected SHD number, for the cross-corpus comparison.
SHD_CORRECTED = (2.74, 2.27, 3.41)

CACHE_PATH = os.path.join("shd_data", "exp51b_cache.json")


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
    """Identical rule to Exp 53: pinned arms reproduce the published config,
    resampled arms vary the draw with the run seed so wiring variance enters
    the across-seed SD."""
    return LEGACY_WIRING_SEED if not resample else LEGACY_WIRING_SEED + run_seed


def arm_structural_stats(cids, inhib_mode, resample):
    """Input->output edge counts and their excitatory subset, over the
    registered seeds. This is the quantity that silently differed."""
    ios, excs = [], []
    for s in SEEDS:
        ws = wiring_seed_for(resample, s)
        src, dst, inhib = make_wiring(cids, ws, inhib_mode)
        io, exc, _ = excitatory_io_stats(cids, src, dst, inhib)
        ios.append(io)
        excs.append(exc)
    return (float(np.mean(ios)), float(np.mean(excs)),
            float(np.std(excs, ddof=1)))


def stats_for(seg, inter):
    ratio = inter.mean() / max(seg.mean(), 1e-9)
    lo, hi = bootstrap_ratio_ci(inter, seg)
    _, p_welch = stats.ttest_ind(inter, seg, equal_var=False)
    return dict(ratio=ratio, lo=lo, hi=hi, p=p_welch,
                d=cohens_d_pooled(inter, seg),
                seg_mean=seg.mean(), seg_sd=seg.std(ddof=1),
                int_mean=inter.mean(), int_sd=inter.std(ddof=1))


def print_stats(label, st):
    print("    {:<14} seg={:8.1f} +/-{:7.1f}   int={:8.1f} +/-{:7.1f}".format(
        label, st["seg_mean"], st["seg_sd"], st["int_mean"], st["int_sd"]))
    print("    {:<14} ratio={:6.2f}x  95% CI [{:.2f}x, {:.2f}x]  "
          "Welch p={:.3e}  d={:.2f}".format(
              "", st["ratio"], st["lo"], st["hi"], st["p"], st["d"]))


if __name__ == "__main__":
    print("=" * 96)
    print("EXP 51b: SSC second-corpus numbers under CORRECTED wiring")
    print("=" * 96)

    # ---- [0] config + geometry audit ------------------------------------
    print("\n[0] Configuration and geometry audit")
    adj = adjacency_counts()
    p1a = all(adj[nm] == EXPECTED_ADJ[nm] for nm in EXPECTED_ADJ)
    print("  adjacency: segregated={:,} interleaved={:,}  (exp37c: {:,}/{:,})".format(
        adj["segregated"], adj["interleaved"],
        EXPECTED_ADJ["segregated"], EXPECTED_ADJ["interleaved"]))
    print("  P1a geometry regression: {}".format("HOLDS" if p1a else "FAILS"))
    est_steps = 2 * SAMPLES_PER_CLASS * N_EPOCHS * STEPS_PER_SAMPLE
    retention = 0.99995 ** est_steps
    print("  life ~{:,} steps, V retention {:.3f} (floor {}) -> P1c {}".format(
        est_steps, retention, MIN_RETENTION,
        "HOLDS" if retention >= MIN_RETENTION else "FAILS"))
    print("  {} pairs x 2 conditions x {} seeds x {} arms = {} runs".format(
        len(CLASS_PAIRS), len(SEEDS), len(ARMS),
        len(CLASS_PAIRS) * 2 * len(SEEDS) * len(ARMS)))
    if not p1a:
        print("  COMPROMISED: geometry changed. Stopping.")
        sys.exit(1)

    conditions = [("segregated", make_placement_segregated),
                  ("interleaved", make_placement_interleaved)]
    coords_map = {name: fn() for name, fn in conditions}

    # ---- [1] P0 equivalence on SSC data, non-negotiable -----------------
    print("\n[1] P0 equivalence: run_life_corrected(legacy) vs imported run_life, on SSC")
    _, tgt0, dis0 = load_class_pair_samples_ssc(
        CLASS_PAIRS[0][1][0], CLASS_PAIRS[0][1][1], SAMPLES_PER_CLASS)
    c0, ci0 = coords_map["segregated"]
    ba_ref, bb_ref = run_life(c0, ci0, 100, tgt0[:4], dis0[:4], n_epochs=1)
    ba_new, bb_new = run_life_corrected(
        c0, ci0, 100, tgt0[:4], dis0[:4], n_epochs=1,
        wiring_seed=LEGACY_WIRING_SEED, inhib_mode="global")[:2]
    print("    imported run_life  : bridge_a={:.10f}  bridge_b={:.10f}".format(
        ba_ref, bb_ref))
    print("    run_life_corrected : bridge_a={:.10f}  bridge_b={:.10f}".format(
        ba_new, bb_new))
    p0 = (ba_ref == ba_new) and (bb_ref == bb_new)
    print("    P0 {}".format(
        "HOLDS (bit-identical) -- ARM C uses the same physics"
        if p0 else "FAILS -- COMPROMISED, stopping"))
    if not p0:
        sys.exit(1)

    # ---- [2] structural audit, before spending hours --------------------
    print("\n[2] Structural audit: input->output edges and their excitatory subset")
    for arm_label, inhib_mode, resample in ARMS:
        print("  {}".format(arm_label))
        excs = {}
        for cname, (coords, cids) in coords_map.items():
            io, exc, exc_sd = arm_structural_stats(cids, inhib_mode, resample)
            excs[cname] = exc
            print("    {:<12} io_edges={:7.1f}  excitatory_io={:7.1f} +/-{:5.1f}".format(
                cname, io, exc, exc_sd))
        gap = 100.0 * (excs["interleaved"] - excs["segregated"]) / max(
            excs["segregated"], 1e-9)
        print("    excitatory io gap (interleaved vs segregated): {:+.2f}%".format(gap))

    if "--audit" in sys.argv:
        print("\n--audit: stopping before the sweep.")
        sys.exit(0)

    # ---- [3] the sweep --------------------------------------------------
    print("\n[3] Running the sweep")
    cache = _cache_load()
    results = {}
    t_start = time.time()
    first = True

    for pair_label, (ca, cb) in CLASS_PAIRS:
        avail, tgt, dis = load_class_pair_samples_ssc(ca, cb, SAMPLES_PER_CLASS)
        got = min(len(tgt), len(dis))
        print("\n  {} classes ({}, {}) -- {} samples/class".format(
            pair_label, ca, cb, got))
        if got < SAMPLES_PER_CLASS:
            raise RuntimeError("{}: only {} samples/class".format(pair_label, got))
        for arm_label, inhib_mode, resample in ARMS:
            for cname, (coords, cids) in coords_map.items():
                totals = []
                for s in SEEDS:
                    key = "{}|{}|{}|{}|spc{}|ep{}".format(
                        arm_label, pair_label, cname, s,
                        SAMPLES_PER_CLASS, N_EPOCHS)
                    if key in cache:
                        ba, bb = cache[key]
                    else:
                        ws = wiring_seed_for(resample, s)
                        t0 = time.time()
                        ba, bb = run_life_corrected(
                            coords, cids, s, tgt, dis, n_epochs=N_EPOCHS,
                            wiring_seed=ws, inhib_mode=inhib_mode)[:2]
                        ba, bb = float(ba), float(bb)
                        cache[key] = [ba, bb]
                        _cache_save(cache)
                        if first:
                            dt = time.time() - t0
                            n_total = len(CLASS_PAIRS) * 2 * len(SEEDS) * len(ARMS)
                            print("    [first run {:.0f}s -> full sweep ~{:.1f}h]".format(
                                dt, n_total * dt / 3600.0))
                            first = False
                    totals.append(ba + bb)
                results[(arm_label, pair_label, cname)] = np.array(totals, dtype=float)
                print("    {:<14} {:<12} n={:2d}  mean={:8.1f} +/-{:7.1f}".format(
                    arm_label, cname, len(totals),
                    np.mean(totals), np.std(totals, ddof=1)))
                sys.stdout.flush()

    print("\n  sweep wall time: {:.1f} min".format((time.time() - t_start) / 60.0))

    # ---- [4] per-pair results, per arm ----------------------------------
    print("\n" + "=" * 96)
    print("[4] Per-pair results by arm")
    print("=" * 96)
    summ = {}
    for arm_label, _, _ in ARMS:
        print("\n  === ARM {} ===".format(arm_label))
        for pair_label, classes in CLASS_PAIRS:
            st = stats_for(results[(arm_label, pair_label, "segregated")],
                           results[(arm_label, pair_label, "interleaved")])
            summ[(arm_label, pair_label)] = st
            print_stats("{} {}".format(pair_label, classes), st)

    # ---- [5] aggregates --------------------------------------------------
    print("\n" + "=" * 96)
    print("[5] Aggregates (pooled across pairs; pairs share seeds -- per-pair is primary)")
    print("=" * 96)
    agg = {}
    for arm_label, _, _ in ARMS:
        seg = np.concatenate([results[(arm_label, p, "segregated")]
                              for p, _ in CLASS_PAIRS])
        inter = np.concatenate([results[(arm_label, p, "interleaved")]
                                for p, _ in CLASS_PAIRS])
        agg[arm_label] = stats_for(seg, inter)
        mean_of_ratios = float(np.mean(
            [summ[(arm_label, p)]["ratio"] for p, _ in CLASS_PAIRS]))
        agg[arm_label]["mean_of_ratios"] = mean_of_ratios
        print("\n  ARM {}".format(arm_label))
        print_stats("aggregate", agg[arm_label])
        print("    mean of per-pair ratios = {:.2f}x".format(mean_of_ratios))

    # ---- [6] fidelity, floor, cross-corpus -------------------------------
    print("\n" + "=" * 96)
    print("[6] Fidelity check, registered floor, and cross-corpus comparison")
    print("=" * 96)

    print("\n  ARM A vs Exp 51 published (fidelity -- must reproduce):")
    fidelity_ok = True
    for pair_label, _ in CLASS_PAIRS:
        got = summ[("A_replication", pair_label)]["ratio"]
        want = EXP51_PUBLISHED[pair_label]
        rel = 100.0 * (got - want) / want
        ok = abs(rel) < 15.0
        fidelity_ok = fidelity_ok and ok
        print("    {}: got {:.2f}x vs published {:.2f}x  ({:+.1f}%)  {}".format(
            pair_label, got, want, rel, "OK" if ok else "MISMATCH"))
    print("    aggregate: got {:.2f}x vs published {:.2f}x".format(
        agg["A_replication"]["ratio"], EXP51_AGG[0]))
    print("    fidelity: {}".format(
        "OK -- ARM C is interpretable" if fidelity_ok
        else "MISMATCH -- harness not faithful, do NOT interpret ARM C"))

    print("\n  Exp 51's registered substantive floor ({:.1f}x) under CORRECTED wiring:".format(
        RATIO_FLOOR))
    n_pass = 0
    for pair_label, _ in CLASS_PAIRS:
        st = summ[("C_corrected", pair_label)]
        ok = (st["p"] < 0.05) and (st["ratio"] >= RATIO_FLOOR)
        n_pass += ok
        print("    {}: ratio {:.2f}x, Welch p={:.2e}  -> {}".format(
            pair_label, st["ratio"], st["p"], "PASS" if ok else "FAIL"))
    print("    {}/{} pairs still clear the floor".format(n_pass, len(CLASS_PAIRS)))

    c_agg = agg["C_corrected"]
    overlap = (c_agg["lo"] <= SHD_CORRECTED[2]) and (SHD_CORRECTED[1] <= c_agg["hi"])
    print("\n  Cross-corpus, BOTH corrected:")
    print("    SSC (this, ARM C): {:.2f}x  CI [{:.2f}, {:.2f}]".format(
        c_agg["ratio"], c_agg["lo"], c_agg["hi"]))
    print("    SHD (Exp 53 ARM C): {:.2f}x  CI [{:.2f}, {:.2f}]".format(*SHD_CORRECTED))
    print("    intervals overlap: {}".format(overlap))

    # ---- [7] verdict -----------------------------------------------------
    print("\n" + "=" * 96)
    print("[7] VERDICT")
    print("=" * 96)
    a_r = agg["A_replication"]["ratio"]
    c_r = c_agg["ratio"]
    rel_drop = 100.0 * (a_r - c_r) / max(a_r, 1e-9)
    arms_overlap = not (c_agg["hi"] < agg["A_replication"]["lo"] or
                        agg["A_replication"]["hi"] < c_agg["lo"])
    print("  ARM A aggregate {:.2f}x -> ARM C aggregate {:.2f}x  ({:+.1f}%)".format(
        a_r, c_r, -rel_drop))
    print("  ARM A / ARM C bootstrap intervals overlap: {}".format(arms_overlap))
    print()
    if not fidelity_ok:
        print("  COMPROMISED: ARM A did not reproduce Exp 51. Do not interpret ARM C.")
    elif arms_overlap and abs(rel_drop) < 25.0:
        print("  ROBUST: SSC survives the wiring correction. Section 4.13's point")
        print("  estimates stand; only the CI needs a note that it omits wiring")
        print("  variance. NOTE this would differ from SHD's behaviour (Exp 53) and")
        print("  would itself need an explanation.")
    else:
        print("  NOT ROBUST: Section 4.13's numbers are inflated by the same pinned-")
        print("  draw defect Exp 53 found on SHD. Corrected values for the paper:")
        for pair_label, _ in CLASS_PAIRS:
            st = summ[("C_corrected", pair_label)]
            print("      {}: {:.2f}x  CI [{:.2f}, {:.2f}]  p={:.2e}  d={:.2f}".format(
                pair_label, st["ratio"], st["lo"], st["hi"], st["p"], st["d"]))
        print("      aggregate: {:.2f}x  CI [{:.2f}, {:.2f}]  p={:.2e}  d={:.2f}".format(
            c_agg["ratio"], c_agg["lo"], c_agg["hi"], c_agg["p"], c_agg["d"]))
        if n_pass < len(CLASS_PAIRS):
            print()
            print("  QUALIFICATION REQUIRED: only {}/{} pairs clear Exp 51's registered".format(
                n_pass, len(CLASS_PAIRS)))
            print("  {:.1f}x substantive floor under corrected wiring. Exp 51's".format(RATIO_FLOOR))
            print("  'not a lucky word pair' conclusion must be qualified accordingly.")
