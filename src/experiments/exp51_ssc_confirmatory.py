"""Exp 51: Confirmatory test that the real-data placement penalty is not
specific to the SHD corpus -- pre-registered replication on Spiking Speech
Commands (SSC), swept across three disjoint class pairs.

MOTIVATION. Every real-data number backing this project's preprint comes
from ONE dataset, the Spiking Heidelberg Digits (Exp 37/37b/37c, 39, 49).
The obvious reviewer objection is "your 5.1x-8.2x is a property of SHD, not
of placement." The exploratory script `tinker_ssc_real_data.py` addressed
this on SSC -- the Zenke Lab sibling corpus, 35 Google-Speech-Commands word
classes vs SHD's 20 spoken digits, different speakers, same HDF5 schema --
and found 11.04x (12 seeds, CI [7.72, 17.38], Welch p = 1.7e-12, d = 8.07)
against SHD's matched-life 7.25x (CI [5.05, 11.48]).

That pass is not sufficient, for two reasons this experiment fixes:
  1. It was TINKERING GRADE: no pre-registration, no decision rule. Per
     this project's own standard (Exp 43 after 42b, Exp 45c after 45/45b),
     an exploratory finding needs a confirmatory run with predictions
     fixed in advance before it is reportable.
  2. It tested exactly ONE class pair -- SSC's two most common classes,
     inherited from exp37c's default. So "is this an SHD artifact?" was
     answered while "is this a lucky-word-pair artifact?" was left wide
     open. A second dataset tested on a single pair barely narrows the
     original objection: it replaces "one corpus" with "one word pair in
     a second corpus."

HONESTY ABOUT WHAT KIND OF PREDICTION THIS IS. The predictions below are
NOT blind. They are derived from the exploratory pass's observed numbers,
which the author has already seen. This is a pre-registered REPLICATION --
the predictions, seeds, class pairs, statistical tests and decision rule
are fixed before this script is run, and the seeds are disjoint from the
exploratory run so the confirmation is on fresh draws rather than a rerun
of the same random state. It is not, and is not claimed to be, a blind
out-of-sample prediction.

DESIGN.
  - THREE DISJOINT CLASS PAIRS, chosen by sample availability alone
    (class-frequency rank in the pool), before any growth outcome was
    inspected for pairs B and C:
        PAIR-A = classes (13, 27)  -- pool ranks 1-2. The EXACT pair the
                 exploratory pass used, so PAIR-A is a direct replication
                 on fresh seeds; PAIR-B and PAIR-C are genuinely new.
        PAIR-B = classes (10, 25)  -- pool ranks 3-4.
        PAIR-C = classes (12,  5)  -- pool ranks 5-6.
    All six classes have >= 118 samples in the 3,000-sample pool, far
    above the 20/class the config consumes; no pair was swapped after
    seeing a result.
  - 16 SEEDS PER CONDITION PER PAIR: seeds 100-115, DISJOINT from the
    exploratory pass's 0-11, and more than its 12 for a tighter CI.
    96 total simulation runs (3 pairs x 2 conditions x 16 seeds).
  - PHYSICS UNCHANGED: `make_placement_segregated`,
    `make_placement_interleaved` and `run_life` are imported verbatim from
    exp37b_v2_real_data.py, exactly as exp37c and the exploratory pass do,
    so no PROJECT_HISTORY gotcha (unreachable rewire block, reward timing,
    always-true trigger, decay/life mismatch) can be reintroduced by a
    copy-paste edit. `cohens_d_pooled` and `bootstrap_ratio_ci` are
    imported verbatim from exp37c.
  - CONFIG: exp37c's MATCHED-LIFE regime verbatim -- 20 distinct samples
    per class x 1 epoch x 215 steps/sample ~= 8,600 steps, V retention
    ~0.651, the band the 0.99995 decay was calibrated for.

REPORTING NOTE (not a bug fix): exp37c's `report_stats` hardcodes the
header "RESULTS on REAL SHD DATA", which would be wrong on SSC data. This
script does NOT edit exp37c (it backs published results). Instead it
prints its own correctly-labelled output, computing the statistics with
exp37c's `cohens_d_pooled` and `bootstrap_ratio_ci` imported VERBATIM plus
the same `scipy.stats` calls exp37c makes (ttest_rel, ttest_ind Welch,
wilcoxon). Only the presentation layer is new; no statistical primitive is
re-implemented.

PREDICTIONS, REGISTERED BEFORE RUNNING:
  P1 (manipulation check -- interpreted BEFORE P2, not part of the test):
     P1a GEOMETRY REGRESSION. The imported placements are dataset-
         independent, so their adjacency must reproduce the values
         documented in exp37c exactly: segregated 1,198 and interleaved
         2,268 input-output pairs within radius 6.0. A mismatch means the
         geometry or an import changed and NOTHING below is trustworthy.
     P1b OPPORTUNITY CANNOT EXPLAIN THE EFFECT (gotcha #4). The adjacency
         ratio (2268/1198 ~= 1.89x) must remain far below the observed
         learning ratio. If a learning ratio ever falls near 1.89x, the
         result is confounded with opportunity count -- the exact failure
         that produced this project's retracted "1,448x".
     P1c DECAY DISCIPLINE (gotcha #2). V retention at end of life must sit
         in the validated band, >= MIN_RETENTION = 0.25 (expected ~0.651).
  P2 (the actual test -- per pair): for EACH of the three class pairs,
     interleaved bridge mass exceeds segregated with Welch p < 0.05 AND a
     ratio >= 3.0x. The 3.0x floor is a SUBSTANTIVE threshold, not just
     significance: it is below the low end of this project's established
     real-data range (exp37c's longer-life 3.62x) so it tests that the
     effect is of the established magnitude, not merely nonzero.
  P3 (consistency -- the new question the exploratory pass could not
     answer): all three pairs satisfy P2. If the effect were a lucky-word-
     pair artifact, pairs B and C would be expected to fall below the
     threshold.
  P4 (cross-dataset magnitude): the SSC aggregate ratio's bootstrap 95% CI
     OVERLAPS SHD's matched-life CI of [5.05x, 11.48x]. The exploratory
     SSC point estimate (11.04x) sat above SHD's (7.25x) but with heavily
     overlapping intervals, so the registered prediction is "same effect,
     same order of magnitude", NOT "SSC is genuinely larger."

DECISION RULE:
  - P1 FAILS (any of a/b/c): the run is COMPROMISED. Do not interpret
    P2-P4 until the cause is understood.
  - P1 holds, P2 holds for all 3 pairs (P3 holds): CONFIRMED at
    confirmatory grade. The placement penalty is neither an SHD artifact
    nor a single-word-pair artifact, and is reportable as a second-corpus
    replication.
  - P1 holds, P2 holds for 2 of 3 pairs: PARTIAL. The effect is real but
    pair-dependent; report the per-pair spread honestly and do NOT quote a
    single aggregate ratio as though it were uniform.
  - P1 holds, P2 holds for <= 1 pair: REFUTED as a general SSC result. The
    exploratory 11.04x was a lucky pair, and the SSC claim must be
    withdrawn or narrowed to the specific pair that survives.
  - P4 separately, and only if the run is not compromised: CIs overlap =>
    report SSC as concordant with SHD. CIs disjoint with SSC higher => SSC
    shows a LARGER effect than SHD; this would be a new finding requiring
    its own explanation (vocabulary size? speaker variability?) and must
    NOT be quietly folded into the existing 5.1x-8.2x range.

LIMITATIONS (stated before running, not discovered afterwards):
  - SPLIT. This uses SSC's `valid` split (9,981 samples, ~155MB), not the
    canonical 1.19GB `train` split, a download-size choice inherited from
    the exploratory pass. This IS a deviation from SSC's benchmark
    protocol and weakens any claim about SSC-as-a-benchmark. It does NOT
    threaten internal validity of the placement comparison: both
    conditions see byte-identical stimuli, and the comparison is between
    placements, not against a published SSC accuracy number.
  - The 2-class bridge task is inherited unchanged from exp37b. It is not
    adapted to SSC's larger vocabulary or its word/speaker structure
    beyond selecting which two classes are presented.
  - Aggregate statistics pool 48 runs per condition across three pairs
    that SHARE seeds, so pooled runs are not fully independent. Per-pair
    results are the primary evidence; the aggregate is reported as a
    summary, and the mean-of-per-pair-ratios is reported alongside it.
  - Placement geometry is fixed (one placement per strategy), so seed
    variance reflects simulation noise, not placement-draw variance --
    the same scope as exp37b/37c.

EXECUTION NOTE (resilience only -- changes no number and no part of the
registered design above): each (pair, condition, seed) cell is cached to
`shd_data/exp51_cache.json` as it completes, and a rerun skips cells
already present. The first execution of this experiment was killed by the
environment partway through (61 of 96 runs), and re-running 1.7 hours of
completed simulations to recover from an external interruption is pure
waste. Runs are deterministic -- fixed seeds, fixed cached stimuli, and
`kind='stable'` sorts throughout the engine since commits 3f8c502->cb9e9cc
-- so a cached cell and a fresh cell are the same number. `--verify-cache`
re-runs one cached cell and asserts it reproduces bit-identically, which is
the audit that licenses trusting the cache at all.

Run: python src/experiments/exp51_ssc_confirmatory.py
Run (smoke, timing check only -- NOT the registered design):
     python src/experiments/exp51_ssc_confirmatory.py --smoke
Run (audit the resume cache against a fresh simulation):
     python src/experiments/exp51_ssc_confirmatory.py --verify-cache
"""
import numpy as np
import time
import sys
import os
import json
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats

from ssc_loader import load_ssc_samples
from spatial import SpatialGrid
# Validated physics, imported verbatim -- do NOT re-implement (see docstring).
from exp37b_v2_real_data import (
    make_placement_segregated, make_placement_interleaved, run_life,
    INPUT_TYPE,
)
# Statistical primitives, imported verbatim from the canonical real-data experiment.
from exp37c_real_data_scaled import (
    cohens_d_pooled, bootstrap_ratio_ci,
    INPUT_TYPE_CHANNELS, MIN_RETENTION, STEPS_PER_SAMPLE,
)

# ---- Registered configuration (fixed before running) --------------------
CLASS_PAIRS = [
    ("PAIR-A", (13, 27)),   # pool ranks 1-2: the exploratory pass's pair
    ("PAIR-B", (10, 25)),   # pool ranks 3-4: new
    ("PAIR-C", (12, 5)),    # pool ranks 5-6: new
]
SEEDS = list(range(100, 116))    # 16 seeds, DISJOINT from exploratory 0-11
SAMPLES_PER_CLASS = 20           # exp37c matched-life regime
N_EPOCHS = 1
POOL_SIZE = 3000                 # matches the cached SSC conversion
MAX_STEPS = 200

# P1a: adjacency values documented in exp37c's docstring for this geometry.
EXPECTED_ADJ = {"segregated": 1198, "interleaved": 2268}
# P2: substantive ratio floor (below exp37c's longer-life 3.62x).
RATIO_FLOOR = 3.0
# P4: SHD matched-life bootstrap CI to compare against (exp37c).
SHD_CI = (5.05, 11.48)
SHD_RATIO = 7.25


CACHE_PATH = os.path.join("shd_data", "exp51_cache.json")


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


def _cache_key(pair_label, condition, seed, samples_per_class, n_epochs):
    """Key includes the config knobs that change the physics, so a smoke
    run's cells can never be mistaken for registered-design cells."""
    return "{}|{}|{}|spc{}|ep{}".format(
        pair_label, condition, seed, samples_per_class, n_epochs)


def load_class_pair_samples_ssc(cls_a, cls_b, samples_per_class,
                                pool_size=POOL_SIZE, max_steps=MAX_STEPS):
    """Return the first `samples_per_class` DISTINCT SSC samples for two
    EXPLICIT class ids. Same pooling recipe as exp37c's
    load_two_class_samples / the exploratory script's SSC variant, except
    the classes are named rather than taken as the two most common -- which
    is what makes the class-pair sweep possible."""
    stimuli, labels = load_ssc_samples(
        n_samples=pool_size, n_channels_out=INPUT_TYPE_CHANNELS,
        bin_ms=4.0, max_steps=max_steps)
    counts = Counter(labels)
    tgt = [s for s, l in zip(stimuli, labels) if l == cls_a][:samples_per_class]
    dis = [s for s, l in zip(stimuli, labels) if l == cls_b][:samples_per_class]
    return {cls_a: counts[cls_a], cls_b: counts[cls_b]}, tgt, dis


def adjacency_counts():
    """Input-output pairs within radius 6.0 for each placement. Dataset-
    independent (the placements do not see data), so this is a pure
    regression check on the imported geometry -- P1a."""
    out = {}
    for name, fn in (("segregated", make_placement_segregated),
                     ("interleaved", make_placement_interleaved)):
        c, ci = fn()
        g = SpatialGrid(c, 6.0)
        inp = set(np.where(ci == INPUT_TYPE)[0].tolist())
        outs = set(np.where(ci != INPUT_TYPE)[0].tolist())
        out[name] = sum(1 for i in inp for j in g.within(i, 6.0) if int(j) in outs)
    return out


def run_pair(pair_label, cls_a, cls_b, seeds, samples_per_class, n_epochs):
    """Run both placements across all seeds for one class pair."""
    avail, target_samples, distractor_samples = load_class_pair_samples_ssc(
        cls_a, cls_b, samples_per_class)
    got = min(len(target_samples), len(distractor_samples))
    print("\n  {} -- target class {} (avail {}), distractor class {} (avail {})".format(
        pair_label, cls_a, avail[cls_a], cls_b, avail[cls_b]))
    if got < samples_per_class:
        raise RuntimeError(
            "{}: only {} samples/class available (< {} registered) -- the "
            "registered design cannot be run as specified.".format(
                pair_label, got, samples_per_class))

    cache = _cache_load()
    results = {}
    for name, fn in (("segregated", make_placement_segregated),
                     ("interleaved", make_placement_interleaved)):
        seed_totals = []
        for s in seeds:
            key = _cache_key(pair_label, name, s, samples_per_class, n_epochs)
            if key in cache:
                ba, bb = cache[key]
                seed_totals.append(ba + bb)
                print("    {:>11} seed {:>3}: bridge_A={:>7.1f} bridge_B={:>7.1f} "
                      "total={:>8.1f}  (cached)".format(name, s, ba, bb, ba + bb))
                sys.stdout.flush()
                continue
            t0 = time.time()
            coords, cids = fn()
            ba, bb = run_life(coords, cids, s, target_samples,
                              distractor_samples, n_epochs=n_epochs)
            seed_totals.append(ba + bb)
            cache[key] = [float(ba), float(bb)]
            _cache_save(cache)
            print("    {:>11} seed {:>3}: bridge_A={:>7.1f} bridge_B={:>7.1f} "
                  "total={:>8.1f}  ({:.0f}s)".format(
                      name, s, ba, bb, ba + bb, time.time() - t0))
            sys.stdout.flush()
        results[name] = np.array(seed_totals, dtype=float)
    return results


def pair_stats(results):
    """Statistics for one class pair. Uses exp37c's imported primitives and
    the same scipy calls exp37c makes; only the presentation is local."""
    inter, seg = results["interleaved"], results["segregated"]
    ratio = inter.mean() / max(seg.mean(), 1e-9)
    lo, hi = bootstrap_ratio_ci(inter, seg)
    t_rel, p_rel = stats.ttest_rel(inter, seg)
    t_ind, p_ind = stats.ttest_ind(inter, seg, equal_var=False)
    try:
        w_stat, p_w = stats.wilcoxon(inter, seg)
    except ValueError:
        w_stat, p_w = float("nan"), float("nan")
    return {
        "seg_mean": seg.mean(), "seg_sd": seg.std(ddof=1),
        "inter_mean": inter.mean(), "inter_sd": inter.std(ddof=1),
        "ratio": ratio, "ci": (lo, hi),
        "p_rel": p_rel, "p_welch": p_ind, "p_wilcoxon": p_w,
        "d": cohens_d_pooled(inter, seg),
        "raw": results,
    }


def print_pair_stats(label, classes, st):
    print("\n  {} (classes {} vs {}), n = {} seeds/condition".format(
        label, classes[0], classes[1], len(st["raw"]["interleaved"])))
    for nm, m, sd in (("segregated", st["seg_mean"], st["seg_sd"]),
                      ("interleaved", st["inter_mean"], st["inter_sd"])):
        rel = 100 * sd / max(m, 1e-9)
        print("    {:>12}: total = {:>9.1f} +/- {:>8.1f}  (rel std {:>5.1f}%)".format(
            nm, m, sd, rel))
    print("    ratio = {:.2f}x   bootstrap 95% CI [{:.2f}x, {:.2f}x]".format(
        st["ratio"], st["ci"][0], st["ci"][1]))
    print("    paired t p = {:.2e} | Welch p = {:.2e} | Wilcoxon p = {:.2e} | "
          "Cohen's d = {:.2f}".format(
              st["p_rel"], st["p_welch"], st["p_wilcoxon"], st["d"]))


def verify_cache(tol=0.05):
    """Audit the resume cache: re-run one cached cell from scratch and
    require a match. If this fails, every cached number is suspect and the
    experiment must be re-run in full.

    TOLERANCE, STATED HONESTLY: the 61 cells recovered from the killed
    first execution were parsed from its stdout, which printed bridge mass
    at ONE decimal place ("%7.1f"). Those cells are therefore rounded to
    0.1, not bit-identical -- a relative error of ~1e-5 on masses of
    300-11,000, negligible for means, SDs, ratios and p-values, but real
    and worth naming rather than hiding behind a 1e-9 assertion that would
    spuriously fail. Cells computed by THIS script are cached at full float
    precision. The audit therefore asserts agreement to +/-{} and prints the
    observed difference so a reader can confirm it is rounding rather than
    divergence."""
    cache = _cache_load()
    keys = [k for k in cache if "|spc{}|ep{}".format(SAMPLES_PER_CLASS, N_EPOCHS) in k]
    if not keys:
        print("  cache empty for the registered config -- nothing to verify.")
        return True
    key = sorted(keys)[0]
    pair_label, condition, seed_s = key.split("|")[:3]
    seed = int(seed_s)
    classes = dict(CLASS_PAIRS)[pair_label]
    print("  verifying cached cell: {}".format(key))
    _, tgt, dis = load_class_pair_samples_ssc(classes[0], classes[1],
                                              SAMPLES_PER_CLASS)
    fn = (make_placement_segregated if condition == "segregated"
          else make_placement_interleaved)
    coords, cids = fn()
    ba, bb = run_life(coords, cids, seed, tgt, dis, n_epochs=N_EPOCHS)
    cba, cbb = cache[key]
    da, db = abs(ba - cba), abs(bb - cbb)
    ok = (da <= tol) and (db <= tol)
    print("    cached : bridge_A={:.6f} bridge_B={:.6f}".format(cba, cbb))
    print("    fresh  : bridge_A={:.6f} bridge_B={:.6f}".format(ba, bb))
    print("    abs diff: {:.6f}, {:.6f}  (tolerance {} = stdout rounding)".format(
        da, db, tol))
    print("    {}".format("MATCH -- cache is trustworthy"
                          if ok else "MISMATCH -- cache must be discarded"))
    return ok


if __name__ == "__main__":
    if "--verify-cache" in sys.argv:
        print("=" * 78)
        print("EXP 51 CACHE AUDIT")
        print("=" * 78)
        sys.exit(0 if verify_cache() else 1)

    smoke = "--smoke" in sys.argv
    seeds = SEEDS[:2] if smoke else SEEDS
    pairs = CLASS_PAIRS[:1] if smoke else CLASS_PAIRS
    spc = 6 if smoke else SAMPLES_PER_CLASS

    print("=" * 78)
    if smoke:
        print("EXP 51 SMOKE TEST -- timing/plumbing only, NOT the registered design")
    else:
        print("EXP 51: CONFIRMATORY SSC replication (predictions registered in docstring)")
    print("=" * 78)

    # ---------------- [0] Structural / config audit ----------------------
    print("\n[0] Structural and config audit")
    adj = adjacency_counts()
    print("  Adjacency (input-output pairs within radius 6.0), dataset-independent:")
    for nm in ("segregated", "interleaved"):
        print("    {:>12}: {:,}  (exp37c documented: {:,})".format(
            nm, adj[nm], EXPECTED_ADJ[nm]))
    adj_ratio = adj["interleaved"] / max(adj["segregated"], 1e-9)
    print("    adjacency ratio (interleaved/segregated): {:.2f}x".format(adj_ratio))

    est_steps = 2 * spc * N_EPOCHS * STEPS_PER_SAMPLE
    retention = 0.99995 ** est_steps
    print("  Life length: ~{:,} steps | V retention at end: {:.3f} "
          "(validated ~0.651; floor {})".format(est_steps, retention, MIN_RETENTION))
    print("  Seeds: {}  (exploratory pass used 0-11; disjoint)".format(
        "{}-{}".format(seeds[0], seeds[-1])))
    print("  Class pairs: {}".format(
        ", ".join("{}={}".format(l, c) for l, c in pairs)))

    # ---------------- [1] The run ----------------------------------------
    print("\n[1] Running {} pair(s) x 2 conditions x {} seeds = {} simulations".format(
        len(pairs), len(seeds), len(pairs) * 2 * len(seeds)))
    t_start = time.time()
    all_stats = {}
    for label, (ca, cb) in pairs:
        res = run_pair(label, ca, cb, seeds, spc, N_EPOCHS)
        all_stats[label] = pair_stats(res)
    print("\n  total wall time: {:.0f}s".format(time.time() - t_start))

    # ---------------- [2] P1: manipulation check -------------------------
    print("\n" + "=" * 78)
    print("[2] P1 (manipulation check) -- interpreted BEFORE P2")
    print("=" * 78)
    p1a = all(adj[nm] == EXPECTED_ADJ[nm] for nm in EXPECTED_ADJ)
    print("  P1a geometry regression (adjacency == exp37c documented): {}".format(
        "HOLDS" if p1a else "FAILS"))
    min_ratio_seen = min(s["ratio"] for s in all_stats.values())
    p1b = min_ratio_seen > 2 * adj_ratio
    print("  P1b opportunity cannot explain effect (min learning ratio {:.2f}x "
          "vs adjacency {:.2f}x): {}".format(
              min_ratio_seen, adj_ratio, "HOLDS" if p1b else "FAILS"))
    p1c = retention >= MIN_RETENTION
    print("  P1c decay discipline (retention {:.3f} >= {}): {}".format(
        retention, MIN_RETENTION, "HOLDS" if p1c else "FAILS"))
    p1 = p1a and p1b and p1c
    print("  P1 OVERALL: {}".format("HOLDS" if p1 else "FAILS"))

    # ---------------- [3] P2/P3: the actual test -------------------------
    print("\n" + "=" * 78)
    print("[3] P2 (per-pair test) and P3 (consistency across pairs)")
    print("=" * 78)
    print("  RESULTS on REAL SSC DATA  (Spiking Speech Commands, valid split)")
    for label, (ca, cb) in pairs:
        print_pair_stats(label, (ca, cb), all_stats[label])

    print("\n  P2 per-pair verdicts (require Welch p < 0.05 AND ratio >= {:.1f}x):".format(
        RATIO_FLOOR))
    p2_flags = {}
    for label, _ in pairs:
        st = all_stats[label]
        ok = (st["p_welch"] < 0.05) and (st["ratio"] >= RATIO_FLOOR)
        p2_flags[label] = ok
        print("    {}: ratio {:.2f}x, Welch p = {:.2e}  -> {}".format(
            label, st["ratio"], st["p_welch"], "PASS" if ok else "FAIL"))
    n_pass = sum(p2_flags.values())
    p3 = n_pass == len(pairs)
    print("  P3 consistency: {}/{} pairs pass -> {}".format(
        n_pass, len(pairs), "HOLDS" if p3 else "FAILS"))

    # ---------------- [4] P4 + verdict -----------------------------------
    print("\n" + "=" * 78)
    print("[4] P4 (cross-dataset magnitude) and VERDICT")
    print("=" * 78)
    pooled_inter = np.concatenate([all_stats[l]["raw"]["interleaved"] for l, _ in pairs])
    pooled_seg = np.concatenate([all_stats[l]["raw"]["segregated"] for l, _ in pairs])
    agg_ratio = pooled_inter.mean() / max(pooled_seg.mean(), 1e-9)
    agg_lo, agg_hi = bootstrap_ratio_ci(pooled_inter, pooled_seg)
    mean_of_ratios = float(np.mean([all_stats[l]["ratio"] for l, _ in pairs]))
    print("  Aggregate (pooled across pairs; shares seeds, see LIMITATIONS):")
    print("    segregated  {:.1f} +/- {:.1f}   interleaved {:.1f} +/- {:.1f}".format(
        pooled_seg.mean(), pooled_seg.std(ddof=1),
        pooled_inter.mean(), pooled_inter.std(ddof=1)))
    print("    aggregate ratio = {:.2f}x, bootstrap 95% CI [{:.2f}x, {:.2f}x]".format(
        agg_ratio, agg_lo, agg_hi))
    print("    mean of per-pair ratios = {:.2f}x".format(mean_of_ratios))
    print("  SHD matched-life reference (exp37c): {:.2f}x, CI [{:.2f}x, {:.2f}x]".format(
        SHD_RATIO, SHD_CI[0], SHD_CI[1]))
    overlap = (agg_lo <= SHD_CI[1]) and (SHD_CI[0] <= agg_hi)
    print("  P4 CIs overlap SHD: {}".format("HOLDS (concordant)" if overlap
                                            else "FAILS (disjoint intervals)"))

    print("\n  ---------------- VERDICT ----------------")
    if not p1:
        print("  COMPROMISED: P1 failed. Do not interpret P2-P4 until the cause is")
        print("  understood -- the geometry, decay regime, or opportunity-parity")
        print("  assumption this experiment rests on is not what was registered.")
    elif p3:
        print("  CONFIRMED at confirmatory grade. All {} class pairs show the".format(len(pairs)))
        print("  penalty at >= {:.1f}x with Welch p < 0.05, on seeds disjoint from the".format(RATIO_FLOOR))
        print("  exploratory pass. The placement penalty is neither an SHD artifact")
        print("  nor a single-word-pair artifact; reportable as a second-corpus")
        print("  replication.")
    elif n_pass == len(pairs) - 1:
        print("  PARTIAL. {}/{} pairs pass. The effect is real but pair-dependent --".format(
            n_pass, len(pairs)))
        print("  report the per-pair spread above; do NOT quote a single aggregate")
        print("  ratio as though the effect were uniform across the vocabulary.")
    else:
        print("  REFUTED as a general SSC result: only {}/{} pairs pass. The".format(
            n_pass, len(pairs)))
        print("  exploratory 11.04x should be treated as a lucky pair, and the SSC")
        print("  claim withdrawn or narrowed to the pair(s) that survive.")

    if p1 and not overlap and agg_ratio > SHD_CI[1]:
        print("\n  NOTE (P4 disjoint, SSC higher): SSC shows a LARGER effect than SHD.")
        print("  Per the registered decision rule this is a NEW finding needing its own")
        print("  explanation (vocabulary size? speaker variability?) and must NOT be")
        print("  quietly folded into the existing 5.1x-8.2x range.")
    if smoke:
        print("\n  (SMOKE RUN -- reduced pairs/seeds/samples; NOT the registered design.)")
