"""Exp 54: Does the SYNTHETIC headline (PLB, 3.8x-4.2x) share the pinned-wiring
defect that collapsed the real-data headline from 8.92x to 2.74x (Exp 53)?

WHY THIS AUDIT EXISTS. `run_life` in exp32b_benchmark.py (lines 101-107) has
the identical pattern Exp 53 found in the real-data harness:

    def run_life(coords, cids, seed):
        rng3 = np.random.default_rng(seed)      # per-run seed: input noise only
        rng2 = np.random.default_rng(7)         # FIXED -- ignores `seed`
        src = rng2.integers(0, N, n_edges); dst = rng2.integers(0, N, n_edges)
        inhib = rng2.random(N) < 0.20

Two consequences, the same two Exp 53 established:

  (1) Initial connectivity and inhibition are IDENTICAL across every "seed"
      of every experiment built on this harness. `seed` varies only the input
      noise stream. Published across-seed SDs (e.g. Table 1b's +/-2.3 on
      segregated growth) therefore contain NO wiring-draw variance and are
      conditional on one arbitrary draw.

  (2) `src`/`dst` index GLOBAL neuron ids, but which id holds which TYPE
      depends on the placement. The identical index arrays therefore realise
      DIFFERENT logical connectivity per condition. The preprint's own Table 4
      shows the footprint: implied frozen taught mass is 703.3+726.6 = 1430.0
      for segregated against 2950.1-1462.3 = 1487.8 for interleaved -- a 4%
      head start for interleaved before any plasticity runs, from the draw
      alone.

A MECHANISTIC PREDICTION, REGISTERED BEFORE RUNNING. This harness may be
much more robust than the real-data one, for a reason specific to it: under
"vlsi" the geometric adjacency between the measured type pairs (0,3)/(1,4)
is EXACTLY ZERO (exp32b's own docstring verifies this against SpatialGrid).
Candidates for rewiring are drawn only from C, which is deposited only
between spatial neighbours within radius 5.0, so no (0,3) or (1,4) candidate
can EVER be proposed under segregated placement, whatever the wiring draw.
Segregated's net loss is therefore forced by GEOMETRY, not by the draw. If
that reasoning is right, resampling should move the taught-mass ratio only
as far as the ~4% baseline asymmetry allows, and should leave the growth
sign flip untouched. Exp 53 found the opposite on the real-data harness --
there, un-pinning the draw raised segregated 2.9x. Stating the prediction in
advance so it can be wrong.

THE THREE ARMS, mirroring Exp 53 exactly, on the standard PLB configuration
(N=900, 60 cores, 1200 steps) with `make_placement` used verbatim:

  ARM A  REPLICATION   wiring_seed = 7 (fixed), global-index wiring.
                       Exactly the published configuration. Must reproduce
                       the 3.8x-4.2x regime or nothing below is interpretable.
  ARM B  RESAMPLED     wiring_seed varies per run seed, still global-index
                       wiring. Isolates un-pinning the draw, retaining the
                       logical-connectivity asymmetry of point (2).
  ARM C  LOGICAL       wiring_seed varies per run seed AND wiring is drawn
                       over LOGICAL identities (type, within-type rank) then
                       mapped through each placement, so both conditions
                       realise the IDENTICAL logical network and differ only
                       in coordinates. The properly-controlled number, and
                       the invariant Exp 50 secured by construction.

METRIC. Per CLAUDE.md and the preprint, raw "taught mass" is inflated by a
shared non-mechanistic random-initialization baseline; GROWTH
(plastic_taught - frozen_taught) is the corrected primary quantity. Growth is
reported as primary here. Taught mass is reported secondarily, because the
3.8x-4.2x headline itself is a raw taught-mass ratio and must be directly
comparable. Note that a growth RATIO is not meaningful when segregated growth
is negative -- that comparison is a sign flip, so growth is reported as means,
difference, Welch p and Cohen's d rather than as a ratio.

WHAT EACH OUTCOME MEANS (fixed before running):
  - ARM C close to ARM A: the synthetic headline is robust; the defect's
    material impact was confined to the real-data harness. Report a clean
    bill of health and note only that published SDs omit wiring variance.
  - ARM C materially below ARM A: the synthetic headline is also inflated by
    the fixed draw and must be corrected to ARM C's value. Report plainly.
  - ARM B between A and C quantifies pinned-draw (A->B) versus
    logical-asymmetry (B->C) contributions.
  - If ARM A fails to reproduce the published regime, the harness is not
    faithful and NOTHING below should be interpreted.

Run:          python src/experiments/exp54_synthetic_wiring_audit.py
Run (audit):  python src/experiments/exp54_synthetic_wiring_audit.py --audit
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

from sparse_state import SparsePairState
from spatial import SpatialGrid

from exp32b_benchmark import (
    make_placement, run_life,
    PATTERNS, NC, N, STEPS_FROZEN, STEPS_PLASTIC, TOTAL,
)
from exp37c_real_data_scaled import cohens_d_pooled, bootstrap_ratio_ci

LEGACY_WIRING_SEED = 7
SEEDS = list(range(20))          # 20 seeds/condition/arm; published set was 0-9

ARMS = [
    ("A_replication", "global", False),   # (label, wiring_mode, resample)
    ("B_resampled", "global", True),
    ("C_logical", "logical", True),
]

CACHE_PATH = os.path.join("shd_data", "exp54_cache.json")


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


def logical_to_global(cids):
    """Map logical identity (type, within-type rank) -> global index.

    Logical index l = t * per_type + r. Every PLB placement has exactly
    N/NC neurons of each type (verified by the assertion below), so this
    map is a permutation and the logical index space is identical across
    placements -- which is what lets ARM C give both conditions the same
    logical network.
    """
    per_type = N // NC
    mapping = np.empty(N, dtype=np.int64)
    for t in range(NC):
        idx = np.where(cids == t)[0]
        assert len(idx) == per_type, (
            "type {} has {} neurons, expected {} -- logical mapping assumes "
            "balanced types".format(t, len(idx), per_type))
        mapping[t * per_type:(t + 1) * per_type] = idx
    return mapping


def make_wiring(cids, wiring_seed, wiring_mode):
    """Initial connectivity and inhibitory mask.

    'global' reproduces exp32b.run_life's draw EXACTLY (same RNG call
    sequence), so wiring_seed=7 is bit-identical to the published harness.
    'logical' draws over logical identities and maps through the placement,
    so both conditions realise the same logical network.
    """
    rng2 = np.random.default_rng(wiring_seed)
    n_edges = N * 10
    if wiring_mode == "global":
        src = rng2.integers(0, N, n_edges)
        dst = rng2.integers(0, N, n_edges)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        inhib = rng2.random(N) < 0.20
        return src, dst, inhib
    elif wiring_mode == "logical":
        src_l = rng2.integers(0, N, n_edges)
        dst_l = rng2.integers(0, N, n_edges)
        keep = src_l != dst_l
        src_l, dst_l = src_l[keep], dst_l[keep]
        inhib_l = rng2.random(N) < 0.20
        m = logical_to_global(cids)
        src = m[src_l]
        dst = m[dst_l]
        inhib = np.empty(N, dtype=bool)
        inhib[m] = inhib_l
        return src, dst, inhib
    raise ValueError("wiring_mode must be 'global' or 'logical'")


def taught_edge_stats(cids, src, dst, inhib):
    """Initial count of edges on the MEASURED type pairs (0,3)/(1,4), and
    their excitatory subset. This is the synthetic analogue of Exp 53's
    input->output / excitatory-input->output audit, and equals frozen taught
    mass exactly (no rewiring occurs before the frozen checkpoint)."""
    M = np.zeros((NC, NC), dtype=int)
    np.add.at(M, (cids[src], cids[dst]), 1)
    taught = int(M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1])
    st, dt = cids[src], cids[dst]
    on_pair = (((st == 0) & (dt == 3)) | ((st == 3) & (dt == 0)) |
               ((st == 1) & (dt == 4)) | ((st == 4) & (dt == 1)))
    exc = int(np.sum(on_pair & ~inhib[src]))
    return taught, exc


def run_life_corrected(coords, cids, seed, wiring_seed=LEGACY_WIRING_SEED,
                       wiring_mode="global"):
    """exp32b.run_life with the wiring draw parameterised. The physics below
    is copied verbatim from exp32b_benchmark.run_life; ONLY the two lines
    that drew src/dst/inhib are replaced by make_wiring(). P0 asserts
    bit-identity in legacy mode."""
    rng3 = np.random.default_rng(seed)
    src, dst, inhib = make_wiring(cids, wiring_seed, wiring_mode)
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
    Rhat = np.zeros(3)
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g = SpatialGrid(coords, 5.0)
    nbr = [g.within(i, 5.0) for i in range(N)]
    out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
    for s2, d2b in zip(src, dst):
        out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
    swap = N // 2
    results = {}
    for t in range(TOTAL):
        p = (t // 20) % 3
        in_plastic = t >= STEPS_FROZEN
        in_reversal = t >= STEPS_FROZEN + STEPS_PLASTIC
        if in_reversal:
            inp = rng3.uniform(0, 0.02, N)
            new_pats = [(1, 4), (0, 3), (2,)]
            pat = new_pats[p]
            if (t % 20) < 5:
                for c in pat: inp[cids == c] += 0.5
        else:
            inp = rng3.uniform(0, 0.02, N)
            if (t % 20) < 5:
                for c in PATTERNS[p]: inp[cids == c] += 0.5
        v_ = v * 0.90 + inp
        fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
        if len(f):
            for fi in f:
                for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
        C.tick(); E.tick(); V.tick()
        if len(f):
            fs = set(int(x) for x in f)
            for i in f:
                i = int(i)
                for j in nbr[i]:
                    if int(j) in fs: C.deposit(i, int(j), 1.0); E.deposit(i, int(j), 1.0)
        v = np.maximum(v_, 0); v[fired] = 0; refrac[fired] = 3; refrac[refrac > 0] -= 1
        if (t % 20) == 6:
            if in_reversal:
                base = {0: 0.0, 1: 1.0, 2: -1.0}[p]
            else:
                base = {0: 1.0, 1: 0.0, 2: -1.0}[p]
            delta = base - Rhat[p]
            if abs(delta) > 1e-9:
                E.prune_below(1e-6)
                for key in list(E.store.keys()):
                    ev = E.get(*key)
                    if ev != 0: V.deposit(key[0], key[1], delta * ev)
            Rhat[p] += 0.15 * delta
        if (t + 1) % 40 == 0 and in_plastic:
            C.prune_below(1e-6)
            sc = np.array([V.get(int(src[k]), int(dst[k])) for k in range(len(src))])
            cold = np.argsort(sc, kind="stable")[:swap]
            ci, cj, _ = C.get_arrays()
            if len(ci) > 0:
                keep2 = ci != cj; ci, cj = ci[keep2], cj[keep2]
                ex = set(zip(src.tolist(), dst.tolist()))
                mk = np.array([(int(a), int(b)) not in ex for a, b in zip(ci, cj)])
                ci, cj = ci[mk], cj[mk]
                if len(ci) > 0:
                    dd = D2[ci, cj]
                    vp = np.maximum(np.array([V.get(int(a), int(b)) for a, b in zip(ci, cj)]), 0)
                    cp = np.array([C.get(int(a), int(b)) for a, b in zip(ci, cj)])
                    score = (vp + 0.01 * cp) / (1 + 0.05 * dd); pos = score > 0
                    ci, cj, score = ci[pos], cj[pos], score[pos]
                    if len(ci) > 0:
                        order = np.argsort(score, kind="stable")[::-1][:len(cold)]
                        n2 = min(len(cold), len(order))
                        src[cold[:n2]] = ci[order[:n2]]; dst[cold[:n2]] = cj[order[:n2]]
                        out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
                        for s2, d2b in zip(src, dst):
                            out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)
        for cp in [STEPS_FROZEN, STEPS_FROZEN + STEPS_PLASTIC, TOTAL]:
            if t + 1 == cp:
                M = np.zeros((NC, NC), dtype=int)
                np.add.at(M, (cids[src], cids[dst]), 1)
                wire_e = D2[src, dst].sum()
                taught = M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]
                selectivity = (M[0, 3] + M[3, 0]) / max(M[1, 4] + M[4, 1], 1)
                old_03 = M[0, 3] + M[3, 0]
                new_14 = M[1, 4] + M[4, 1]
                phase = {STEPS_FROZEN: "frozen",
                         STEPS_FROZEN + STEPS_PLASTIC: "plastic",
                         TOTAL: "reversal"}[cp]
                results[phase] = {"taught": int(taught), "energy": float(wire_e),
                                  "selectivity": float(selectivity),
                                  "old_03": int(old_03), "new_14": int(new_14)}
    return results


def wiring_seed_for(resample, run_seed):
    return LEGACY_WIRING_SEED if not resample else LEGACY_WIRING_SEED + run_seed


def bootstrap_diff_ci(a, b, n_boot=10000, seed=12345):
    """Percentile bootstrap 95% CI on mean(a) - mean(b). Used for growth,
    where a ratio is meaningless because segregated growth is negative."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(a[idx].mean() - b[idx].mean())
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def report_arm(label, seg_g, int_g, seg_t, int_t):
    """Growth primary (means/difference/sign), taught mass secondary (ratio)."""
    print("  {} -- GROWTH (primary, plastic - frozen)".format(label))
    print("    segregated  {:+9.1f} +/- {:6.1f}   {}".format(
        seg_g.mean(), seg_g.std(ddof=1),
        "NET GAIN" if seg_g.mean() > 0 else "NET LOSS"))
    print("    interleaved {:+9.1f} +/- {:6.1f}   {}".format(
        int_g.mean(), int_g.std(ddof=1),
        "NET GAIN" if int_g.mean() > 0 else "NET LOSS"))
    dlo, dhi = bootstrap_diff_ci(int_g, seg_g)
    t_g, p_g = stats.ttest_ind(int_g, seg_g, equal_var=False)
    d_g = cohens_d_pooled(int_g, seg_g)
    sign_flip = (int_g.mean() > 0) and (seg_g.mean() < 0)
    print("    difference  {:+9.1f}  95% CI [{:+.1f}, {:+.1f}]  Welch p={:.3e}  d={:.2f}".format(
        int_g.mean() - seg_g.mean(), dlo, dhi, p_g, d_g))
    print("    sign flip (interleaved gains, segregated loses): {}".format(sign_flip))

    ratio = int_t.mean() / max(seg_t.mean(), 1e-9)
    lo, hi = bootstrap_ratio_ci(int_t, seg_t)
    t_t, p_t = stats.ttest_ind(int_t, seg_t, equal_var=False)
    d_t = cohens_d_pooled(int_t, seg_t)
    print("  {} -- TAUGHT MASS (secondary; the 3.8x-4.2x headline metric)".format(label))
    print("    segregated  {:9.1f} +/- {:6.1f}".format(seg_t.mean(), seg_t.std(ddof=1)))
    print("    interleaved {:9.1f} +/- {:6.1f}".format(int_t.mean(), int_t.std(ddof=1)))
    print("    ratio={:5.2f}x  95% CI [{:.2f}x, {:.2f}x]  Welch p={:.3e}  d={:.2f}".format(
        ratio, lo, hi, p_t, d_t))
    return dict(g_seg=seg_g.mean(), g_seg_sd=seg_g.std(ddof=1),
                g_int=int_g.mean(), g_int_sd=int_g.std(ddof=1),
                g_diff=int_g.mean() - seg_g.mean(), g_p=p_g, g_d=d_g,
                sign_flip=sign_flip,
                ratio=ratio, lo=lo, hi=hi, t_p=p_t, t_d=d_t,
                t_seg=seg_t.mean(), t_int=int_t.mean())


if __name__ == "__main__":
    print("=" * 96)
    print("EXP 54: does the SYNTHETIC headline (PLB 3.8x-4.2x) share the")
    print("        pinned-wiring defect Exp 53 found in the real-data harness?")
    print("=" * 96)

    conditions = [("segregated", "vlsi"), ("interleaved", "topoforge")]
    place = {name: make_placement(strat) for name, strat in conditions}
    print("\n[0] Configuration: N={}, {} steps, seeds {}-{} ({}/condition/arm)".format(
        N, TOTAL, SEEDS[0], SEEDS[-1], len(SEEDS)))
    for name, (coords, cids) in place.items():
        counts = [int((cids == t).sum()) for t in range(NC)]
        print("    {:<12} per-type counts {}".format(name, counts))

    # ---- P0: equivalence gate, non-negotiable --------------------------
    print("\n[1] P0 equivalence: run_life_corrected(legacy) vs imported exp32b.run_life")
    c0, ci0 = place["segregated"]
    t0 = time.time()
    ref = run_life(c0, ci0, 0)
    dt_one = time.time() - t0
    new = run_life_corrected(c0, ci0, 0,
                             wiring_seed=LEGACY_WIRING_SEED, wiring_mode="global")
    keys = ["frozen", "plastic", "reversal"]
    p0 = True
    for k in keys:
        for fld in ["taught", "old_03", "new_14"]:
            if ref[k][fld] != new[k][fld]:
                p0 = False
        if abs(ref[k]["energy"] - new[k]["energy"]) > 0:
            p0 = False
    for k in keys:
        print("    {:<9} imported taught={:<6} corrected taught={:<6} energy_delta={:.10g}".format(
            k, ref[k]["taught"], new[k]["taught"],
            ref[k]["energy"] - new[k]["energy"]))
    print("    P0 {}".format(
        "HOLDS (bit-identical) -- arms B/C run the published physics"
        if p0 else "FAILS -- COMPROMISED, stopping"))
    if not p0:
        sys.exit(1)
    print("    single run: {:.0f}s -> full sweep ~{:.2f}h".format(
        dt_one, len(ARMS) * 2 * len(SEEDS) * dt_one / 3600.0))

    # ---- structural audit, before spending hours -----------------------
    print("\n[2] Structural audit: initial edges on measured type pairs (0,3)/(1,4)")
    print("    (equals frozen taught mass; no rewiring occurs before that checkpoint)")
    for arm_label, wmode, resample in ARMS:
        print("  {}".format(arm_label))
        stats_by_cond = {}
        for cname, (coords, cids) in place.items():
            ts, es = [], []
            for s in SEEDS:
                ws = wiring_seed_for(resample, s)
                src, dst, inhib = make_wiring(cids, ws, wmode)
                tt, ee = taught_edge_stats(cids, src, dst, inhib)
                ts.append(tt); es.append(ee)
            stats_by_cond[cname] = (np.mean(ts), np.std(ts, ddof=1),
                                    np.mean(es), np.std(es, ddof=1))
            print("    {:<12} taught_edges={:7.1f} +/-{:5.1f}   excitatory={:7.1f} +/-{:5.1f}".format(
                cname, *stats_by_cond[cname]))
        ts_s = stats_by_cond["segregated"][0]
        ts_i = stats_by_cond["interleaved"][0]
        es_s = stats_by_cond["segregated"][2]
        es_i = stats_by_cond["interleaved"][2]
        print("    baseline gap (interleaved vs segregated): taught {:+.2f}%   excitatory {:+.2f}%".format(
            100.0 * (ts_i - ts_s) / max(ts_s, 1e-9),
            100.0 * (es_i - es_s) / max(es_s, 1e-9)))

    if "--audit" in sys.argv:
        print("\n--audit: stopping before the sweep.")
        sys.exit(0)

    # ---- the sweep -----------------------------------------------------
    n_total = len(ARMS) * 2 * len(SEEDS)
    print("\n[3] Running {} arms x 2 conditions x {} seeds = {} runs".format(
        len(ARMS), len(SEEDS), n_total))
    cache = _cache_load()
    growth, taught = {}, {}
    t_start = time.time()

    for arm_label, wmode, resample in ARMS:
        print("\n  === ARM {} (wiring_mode={}, resample={}) ===".format(
            arm_label, wmode, resample))
        for cname, (coords, cids) in place.items():
            gs, ts = [], []
            for s in SEEDS:
                key = "{}|{}|{}".format(arm_label, cname, s)
                if key in cache:
                    fr, pl = cache[key]
                else:
                    ws = wiring_seed_for(resample, s)
                    r = run_life_corrected(coords, cids, s,
                                           wiring_seed=ws, wiring_mode=wmode)
                    fr, pl = int(r["frozen"]["taught"]), int(r["plastic"]["taught"])
                    cache[key] = [fr, pl]
                    _cache_save(cache)
                gs.append(pl - fr); ts.append(pl)
            growth[(arm_label, cname)] = np.array(gs, dtype=float)
            taught[(arm_label, cname)] = np.array(ts, dtype=float)
            print("    {:<12} n={}  growth={:+8.1f} +/-{:6.1f}   taught={:8.1f} +/-{:6.1f}".format(
                cname, len(gs), np.mean(gs), np.std(gs, ddof=1),
                np.mean(ts), np.std(ts, ddof=1)))
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
            growth[(arm_label, "segregated")], growth[(arm_label, "interleaved")],
            taught[(arm_label, "segregated")], taught[(arm_label, "interleaved")])
        print()

    # ---- cross-arm comparison ------------------------------------------
    print("=" * 96)
    print("[5] Cross-arm comparison")
    print("=" * 96)
    a, b, c = summ["A_replication"], summ["B_resampled"], summ["C_logical"]
    print("  taught-mass ratio (the published headline metric):")
    print("    ARM A (published config, pinned draw) : {:5.2f}x  CI [{:.2f}, {:.2f}]".format(
        a["ratio"], a["lo"], a["hi"]))
    print("    ARM B (draw resampled, global wiring) : {:5.2f}x  CI [{:.2f}, {:.2f}]".format(
        b["ratio"], b["lo"], b["hi"]))
    print("    ARM C (resampled, LOGICAL wiring)     : {:5.2f}x  CI [{:.2f}, {:.2f}]".format(
        c["ratio"], c["lo"], c["hi"]))
    print("    preprint headline for this regime     : 3.8x-4.2x (Table 1/Table 4)")
    print()
    print("  growth (corrected primary), segregated -> interleaved:")
    for lbl, sd in (("A", a), ("B", b), ("C", c)):
        print("    ARM {}: seg {:+8.1f}  int {:+8.1f}  diff {:+8.1f}  sign flip {}".format(
            lbl, sd["g_seg"], sd["g_int"], sd["g_diff"], sd["sign_flip"]))
    print()
    print("  attribution (taught-mass ratio):")
    print("    A -> B  (un-pinning the wiring draw) : {:+.2f}x".format(b["ratio"] - a["ratio"]))
    print("    B -> C  (removing logical asymmetry) : {:+.2f}x".format(c["ratio"] - b["ratio"]))
    print("    A -> C  (total)                      : {:+.2f}x".format(c["ratio"] - a["ratio"]))
    print()
    print("  uncertainty width (bootstrap CI on taught ratio, hi-lo):")
    for lbl, sd in (("A", a), ("B", b), ("C", c)):
        print("    ARM {}: {:.2f}x wide".format(lbl, sd["hi"] - sd["lo"]))

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 96)
    print("[6] VERDICT")
    print("=" * 96)
    a_ok = 3.4 <= a["ratio"] <= 4.6
    print("  ARM A reproduces the published regime (3.4-4.6x): {}".format(
        "YES ({:.2f}x)".format(a["ratio"]) if a_ok
        else "NO ({:.2f}x) -- harness not faithful, do NOT interpret B/C".format(a["ratio"])))
    if not a_ok:
        sys.exit(0)

    overlap = not (c["hi"] < a["lo"] or a["hi"] < c["lo"])
    rel = 100.0 * (c["ratio"] - a["ratio"]) / max(a["ratio"], 1e-9)
    print("  ARM A and ARM C bootstrap intervals overlap: {}".format(overlap))
    print("  ARM C is {:+.1f}% relative to ARM A".format(rel))
    print("  growth sign flip preserved in ARM C: {}".format(c["sign_flip"]))
    print()
    if overlap and abs(rel) < 25.0:
        print("  ROBUST: the synthetic headline survives resampling the wiring draw")
        print("  and removing the logical-connectivity asymmetry. The published point")
        print("  estimate stands; the only correction needed is that published SDs")
        print("  are conditional on one wiring draw and omit its variance.")
        print("  Exp 53's collapse was therefore specific to the real-data harness.")
    else:
        print("  NOT ROBUST: the synthetic headline is also substantially conditional")
        print("  on the fixed draw and/or the logical-connectivity asymmetry. ARM C is")
        print("  the properly-controlled number; correct the headline to:")
        print("      taught-mass ratio {:.2f}x, 95% CI [{:.2f}x, {:.2f}x], p={:.2e}, d={:.2f}".format(
            c["ratio"], c["lo"], c["hi"], c["t_p"], c["t_d"]))
        print("      growth: segregated {:+.1f}, interleaved {:+.1f}, p={:.2e}".format(
            c["g_seg"], c["g_int"], c["g_p"]))
