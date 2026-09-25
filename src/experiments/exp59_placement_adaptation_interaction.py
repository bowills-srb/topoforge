"""Exp 59: does segregated placement cost MORE when the task changes?

WHY THIS EXISTS. Everything this project claims about placement is measured on
learning an association from scratch. Nothing measures RE-learning. That gap
matters commercially more than the original claim does: if a mapping is going
to be trained once and frozen, nobody needs plasticity in silicon at all --
you train offline and ship weights. The reason to want structural plasticity
on a chip is that the task will change after deployment. So "segregated
placement adapts worse" is a sharper argument than "segregated placement
learns less", and nobody has checked whether it is true.

Exp 58 gave us the instrument. It found that a one-bit recency flag -- "was
this connection slot rewired last epoch" -- improves reversal re-learning by
61.5% (20 seeds, non-overlapping CIs), by turning continuous churn into a
one-epoch admission test. But it ran on INTERLEAVED placement only, by design.
This adds the segregated arm and asks the interaction question.

PRIMARY, PRE-REGISTERED. The quantity of interest is the PLACEMENT RATIO on
reversal re-learning -- interleaved new_14 / segregated new_14 -- measured
with the recency flag off and on. Does the placement gap WIDEN when the rule
is capable of adapting well?

  ratio widens (CIs on the two ratios do not overlap)
      -> INTERACTION. Placement and adaptation are not independent: segregation
         costs more under task change than it does under first learning. This
         is a genuinely new and stronger form of the claim and belongs in the
         preprint as its own section.
  ratio unchanged
      -> INDEPENDENT. Recency helps both placements equally; adaptation and
         placement are separate axes. A clean negative that stops the claim
         being over-extended later, and worth reporting as one.
  ratio narrows
      -> recency partially RESCUES segregated placement. That would be the
         most useful practical result available here -- a one-bit fix that
         buys back some of the penalty -- and it argues against this project's
         own interest, so it gets reported first and loudest if it happens.

REGISTERED PREDICTION, so a hit counts and a miss is a miss: I expect the
ratio to WIDEN. Interleaved has cross-type candidate pairs available to lock
in after the reversal; segregated cannot form them at all, so giving it a
better retention policy should have less to protect. Stated before running.

SECONDARY. Within-placement recency benefit (binary / none) for each
placement separately, with bootstrap CIs. This is what the interaction is
computed from and is reported whatever the primary says.

SCOPE AND WHAT IS DELIBERATELY NOT HERE. Only `none` and `binary` are run.
Exp 58 established that graded accumulation adds ~7 points over binary's 61.5%
and that the effect is recency, not survival length; re-running graded here
would cost a third of the runtime to re-confirm a settled question. The
endpoint is structural (new_14), NOT accuracy -- the same limitation Exp 52b
exposed, where a 3.59x structural effect was 1.079x in accuracy terms. No
claim from this file should be stated in performance terms until that is
measured.

SCALE IS NOT A FREE CHOICE HERE, AND THE TWO SCALES DISAGREE IN SIGN. The
primary is a RATIO, so "the placement gap narrowed" is a claim on the
MULTIPLICATIVE scale. The same four cells read on the ADDITIVE scale say the
opposite, and neither reading is marginal:

    placement       flag off   flag on   additive gain   multiplicative gain
    segregated          37.5     124.2          +86.7                 3.31x
    interleaved        212.4     342.9         +130.5                 1.61x

Multiplicatively the flag helps segregated twice as much and the gap narrows
5.66x -> 2.76x. Additively the flag buys interleaved 43.9 MORE edges than it
buys segregated, so on that scale segregated is helped LESS and the registered
prediction is effectively confirmed. Both interactions exclude zero.

The reason is a floor. Segregated's flag-off arm sits at mean 37.5 with a
minimum of 1 and CV 0.73, and proportional gains inflate without limit against
a small, noisy denominator -- the same arithmetic that makes per-seed ratios
useless here (they average 18.35x against a ratio-of-means of 5.66x, which is
why ratio-of-means is what this file reports). This is not a defect in the
measurement; it is a fact about what "the gap" means when one arm is near zero.

THE FILE THEREFORE REPORTS BOTH AND PICKS NEITHER SILENTLY. The multiplicative
reading is this project's native scale -- every placement headline it has ever
published (3.8x-4.2x, 2.92x, 2.74x) is a ratio, and reporting this one
additively while reporting those multiplicatively would be scale-shopping. So
the primary stays multiplicative, as registered. But the additive contradiction
is printed next to it, unprompted, because a reader can recompute it in thirty
seconds and any claim that survives only on one scale must say so itself.

WHAT MAY AND MAY NOT BE SAID. Sayable: "on the ratio scale the project states
its penalties on, a one-bit recency flag halves the adaptation gap." Not
sayable: "the flag preferentially rescues segregated placement" -- that is a
mechanism claim, and the additive result is evidence against it. Distinguishing
the two needs an arm where segregated is not at the floor, which this design
does not contain.

CAVEAT ON THE CIs, RECORDED NOT FIXED. `ratio_ci` resamples numerator and
denominator with independent index draws, but the arms are paired by seed
(same seed, same wiring, same placement; only `age_mode` differs). Discarding
that pairing widens the intervals, so it is conservative for the non-overlap
read-off the primary rests on and is left alone rather than changed under a
result. The additive statistics use the same convention so the two scales are
computed by identical machinery and stay comparable.

REUSES, NOT REIMPLEMENTS. `run_life_age` is imported verbatim from
exp58_age_reversal_mechanism, which itself passed a bit-identity audit against
both exp54.run_life_corrected and exp57.run_life_tables. --audit re-checks two
cells against Exp 58's recorded cache before anything here is interpreted.

Run:          python src/experiments/exp59_placement_adaptation_interaction.py
Run (audit):  python src/experiments/exp59_placement_adaptation_interaction.py --audit
Run (quick):  python src/experiments/exp59_placement_adaptation_interaction.py --seeds 4
Run (report): python src/experiments/exp59_placement_adaptation_interaction.py --from-cache
              re-prints the full analysis from the cached per-seed numbers
              without re-simulating. Analysis changes go through this path.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from exp32b_benchmark import make_placement  # noqa: E402
from exp54_synthetic_wiring_audit import wiring_seed_for  # noqa: E402
from exp58_age_reversal_mechanism import run_life_age  # noqa: E402

CACHE_PATH = os.path.join("shd_data", "exp59_cache.json")
EXP58_CACHE = os.path.join("shd_data", "exp58_cache.json")
PLACEMENTS = [("segregated", "vlsi"), ("interleaved", "topoforge")]
MODES = ("none", "binary")


def boot(fn, n_boot=10000, seed=12345):
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng) for _ in range(n_boot)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ratio_ci(num, den, n_boot=10000, seed=12345):
    num = np.asarray(num, float); den = np.asarray(den, float)
    return boot(lambda r: num[r.integers(0, len(num), len(num))].mean()
                / max(den[r.integers(0, len(den), len(den))].mean(), 1e-9),
                n_boot, seed)


def diff_ci(num, den, n_boot=10000, seed=12345):
    """Additive counterpart to ratio_ci: CI on mean(num) - mean(den).

    Deliberately uses the same independent-resampling convention as ratio_ci
    so the two scales are produced by identical machinery (see the CAVEAT ON
    THE CIs note in the module docstring).
    """
    num = np.asarray(num, float); den = np.asarray(den, float)
    return boot(lambda r: num[r.integers(0, len(num), len(num))].mean()
                - den[r.integers(0, len(den), len(den))].mean(),
                n_boot, seed)


def interaction_cis(out, n_boot=10000, seed=12345):
    """Interaction between placement and the recency flag, on BOTH scales.

    additive:  (interleaved gain) - (segregated gain), in edges.
    log-ratio: log(interleaved x-gain) - log(segregated x-gain). Negative means
               segregated gains proportionally more, i.e. the placement ratio
               narrows. The two can and here do disagree in sign.
    """
    arms = {(lab, m): np.array(out[lab][m]["new_14"], float)
            for lab, _ in PLACEMENTS for m in MODES}

    def draw(r):
        d = {}
        for k, v in arms.items():
            d[k] = v[r.integers(0, len(v), len(v))].mean()
        return d

    def add(d):
        return ((d[("interleaved", "binary")] - d[("interleaved", "none")])
                - (d[("segregated", "binary")] - d[("segregated", "none")]))

    def logr(d):
        safe = lambda x: max(x, 1e-9)
        return (np.log(safe(d[("interleaved", "binary")]) / safe(d[("interleaved", "none")]))
                - np.log(safe(d[("segregated", "binary")]) / safe(d[("segregated", "none")])))

    rng = np.random.default_rng(seed)
    draws = [draw(rng) for _ in range(n_boot)]
    a = np.array([add(d) for d in draws])
    l = np.array([logr(d) for d in draws])
    point = {k: v.mean() for k, v in arms.items()}
    return {
        "additive": {"point": float(add(point)),
                     "ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                     "p_opposite": float((a < 0).mean())},
        "log_ratio": {"point": float(logr(point)),
                      "ci": [float(np.percentile(l, 2.5)), float(np.percentile(l, 97.5))],
                      "p_opposite": float((l > 0).mean())},
    }


def audit():
    print("\n[P0] imported run_life_age must reproduce Exp 58's recorded numbers")
    if not os.path.isfile(EXP58_CACHE):
        print("    exp58 cache missing -- run exp58 first. Cannot audit.")
        sys.exit(1)
    prior = json.load(open(EXP58_CACHE))["raw"]
    coords, cids = make_placement("topoforge")
    ok = True
    for mode in MODES:
        for seed in (0, 1):
            ws = wiring_seed_for(True, seed)
            r = run_life_age(coords, cids, seed, wiring_seed=ws, age_mode=mode)
            want = prior[mode]["new_14"][seed]
            got = r["reversal"]["new_14"]
            same = got == want
            ok = ok and same
            print("    {:<7s} seed {}  new_14 {:>5d}  exp58 recorded {:>5d}  match: {}".format(
                mode, seed, got, want, same))
    print("    {}".format("P0 PASSED" if ok else
                          "P0 FAILED -- do NOT interpret any result from this file"))
    if not ok:
        sys.exit(1)


def simulate(args, t0):
    out = {}
    for label, strategy in PLACEMENTS:
        coords, cids = make_placement(strategy)
        out[label] = {}
        for mode in MODES:
            vals, taught = [], []
            for seed in args.seeds:
                ws = wiring_seed_for(True, seed)
                r = run_life_age(coords, cids, seed, wiring_seed=ws, age_mode=mode)
                vals.append(r["reversal"]["new_14"])
                taught.append(r["plastic"]["taught"])
                print("    {:<12s} {:<7s} seed {:>2d}  new_14 {:>5d}  taught {:>6d}"
                      "  [{:.0f}s]".format(label, mode, seed, vals[-1], taught[-1],
                                           time.time() - t0), flush=True)
            out[label][mode] = {"new_14": vals, "taught": taught}
    return out


def report(out, t0=None):

    print("\n  REVERSAL RE-LEARNING (new_14) -- BOTH SCALES")
    print("    {:<12s} {:>8s} {:>8s} {:>9s} {:>16s} {:>8s} {:>14s}".format(
        "placement", "none", "binary", "abs gain", "95% CI", "x gain", "95% CI"))
    gains, gains_abs = {}, {}
    for label, _ in PLACEMENTS:
        a = np.array(out[label]["none"]["new_14"], float)
        b = np.array(out[label]["binary"]["new_14"], float)
        rlo, rhi = ratio_ci(b, a)
        dlo, dhi = diff_ci(b, a)
        gains[label] = (b.mean() / max(a.mean(), 1e-9), rlo, rhi)
        gains_abs[label] = (b.mean() - a.mean(), dlo, dhi)
        print("    {:<12s} {:>8.1f} {:>8.1f} {:>+9.1f} {:>16s} {:>7.2f}x {:>14s}".format(
            label, a.mean(), b.mean(), gains_abs[label][0],
            "[{:+.1f}, {:+.1f}]".format(dlo, dhi), gains[label][0],
            "[{:.2f}, {:.2f}]".format(rlo, rhi)))

    print("\n  PRIMARY -- placement ratio on reversal, with the flag off vs on")
    print("    (multiplicative, as registered; see SCALE DISCLOSURE below)")
    ratios = {}
    for mode in MODES:
        i = np.array(out["interleaved"][mode]["new_14"], float)
        s = np.array(out["segregated"][mode]["new_14"], float)
        lo, hi = ratio_ci(i, s)
        ratios[mode] = (i.mean() / max(s.mean(), 1e-9), lo, hi)
        print("    flag {:<7s} interleaved/segregated = {:.2f}x   95% CI [{:.2f}, {:.2f}]"
              .format(mode, ratios[mode][0], lo, hi))

    r_none, r_bin = ratios["none"], ratios["binary"]
    overlap = not (r_bin[1] > r_none[2] or r_none[1] > r_bin[2])
    print("\n  PRE-REGISTERED READ-OFF")
    if overlap:
        verdict = ("INDEPENDENT. The placement gap is the same size whether or not "
                   "the rule can adapt well. Placement and adaptation are separate "
                   "axes; the registered prediction (widening) is a MISS and must be "
                   "reported as one.")
    elif r_bin[0] > r_none[0]:
        verdict = ("INTERACTION. The placement gap WIDENS when the rule adapts well "
                   "({:.2f}x -> {:.2f}x). Segregation costs more under task change "
                   "than under first learning -- a stronger and newer form of the "
                   "claim. Registered prediction CONFIRMED.".format(r_none[0], r_bin[0]))
    else:
        verdict = ("RESCUE. The placement gap NARROWS ({:.2f}x -> {:.2f}x): a one-bit "
                   "retention flag buys back part of the segregation penalty. This "
                   "cuts AGAINST this project's claim and is the most practically "
                   "useful result available here. Report it first, not last. "
                   "Registered prediction MISSED in the opposite direction."
                   .format(r_none[0], r_bin[0]))
    print("    " + verdict)

    scale = scale_disclosure(out)

    print("\n  SANITY -- first-learning taught mass should be unaffected by the flag")
    for label, _ in PLACEMENTS:
        a = np.mean(out[label]["none"]["taught"])
        b = np.mean(out[label]["binary"]["taught"])
        print("    {:<12s} none {:>8.1f}   binary {:>8.1f}   ({:+.2f}%)".format(
            label, a, b, 100 * (b - a) / max(a, 1e-9)))

    with open(CACHE_PATH, "w") as fh:
        json.dump({"raw": out, "gains": gains, "gains_abs": gains_abs,
                   "ratios": ratios, "verdict": verdict, "scale": scale},
                  fh, indent=1)
    tail = "" if t0 is None else "  ({:.0f}s)".format(time.time() - t0)
    print("\n  cached -> {}{}".format(CACHE_PATH, tail))


def scale_disclosure(out):
    """Print the same four cells on the additive scale, and say so when the two
    scales disagree. Printed unconditionally, including when they agree -- the
    point is that a reader never has to wonder whether it was checked."""
    inter = interaction_cis(out)
    add, log = inter["additive"], inter["log_ratio"]

    print("\n  SCALE DISCLOSURE -- the same four cells, on both scales")
    print("    additive interaction   (int gain - seg gain)   = {:+.1f} edges"
          "   95% CI [{:+.1f}, {:+.1f}]".format(add["point"], *add["ci"]))
    print("    log-ratio interaction  (log int x - log seg x) = {:+.2f}"
          "         95% CI [{:+.2f}, {:+.2f}]".format(log["point"], *log["ci"]))
    print("      convention: additive > 0 means the flag buys INTERLEAVED more edges;")
    print("                  log-ratio < 0 means SEGREGATED gains proportionally more.")

    floor = np.array(out["segregated"]["none"]["new_14"], float)
    inter_none = np.array(out["interleaved"]["none"]["new_14"], float)
    print("    floor check -- segregated, flag off: mean {:.1f}  sd {:.1f}  CV {:.2f}"
          "  min {:.0f}".format(floor.mean(), floor.std(ddof=1),
                                floor.std(ddof=1) / max(floor.mean(), 1e-9), floor.min()))
    print("    ratio-of-means is used, not mean-of-ratios: per-seed ratios (flag off)"
          " average {:.2f}x".format((inter_none / np.maximum(floor, 1e-9)).mean()))
    print("      against a ratio-of-means of {:.2f}x -- the skew a near-zero"
          " denominator produces.".format(inter_none.mean() / max(floor.mean(), 1e-9)))

    conflict = (add["point"] > 0) and (log["point"] < 0)
    both_clear = ((add["ci"][0] > 0 or add["ci"][1] < 0)
                  and (log["ci"][0] > 0 or log["ci"][1] < 0))

    if conflict:
        note = ("THE TWO SCALES DISAGREE IN SIGN{}. Multiplicatively the flag helps "
                "segregated more and the placement gap narrows; additively it buys "
                "interleaved MORE edges, so on that scale segregated is helped LESS "
                "and the registered widening prediction is effectively confirmed. The "
                "primary above stands as the multiplicative statement it was "
                "registered as -- every placement headline this project publishes is a "
                "ratio, and reporting this one additively would be scale-shopping -- "
                "but it is a ratio-scale result ONLY. Do NOT restate it as 'recency "
                "preferentially rescues segregated placement': that is a mechanism "
                "claim and the additive result is evidence against it. Segregated's "
                "flag-off arm sits at the floor, which is exactly where proportional "
                "gains inflate; separating floor from mechanism needs an arm this "
                "design does not contain."
                .format(", and both exclude zero" if both_clear else ""))
    else:
        note = ("The two scales AGREE in sign, so the primary is not a scale artifact "
                "and may be stated on either. Reported multiplicatively for "
                "consistency with the rest of the project.")
    print("    " + note)
    inter["conflict"] = bool(conflict)
    inter["note"] = note
    return inter


def main(args):
    t0 = time.time()
    if args.from_cache:
        if not os.path.isfile(CACHE_PATH):
            print("  no cache at {} -- run without --from-cache first.".format(CACHE_PATH))
            sys.exit(1)
        out = json.load(open(CACHE_PATH))["raw"]
        print("  replaying analysis from {} ({} seeds, no re-simulation)".format(
            CACHE_PATH, len(out["segregated"]["none"]["new_14"])))
    else:
        out = simulate(args, t0)
    report(out, t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--from-cache", action="store_true",
                    help="re-print the analysis from cached per-seed numbers")
    a = ap.parse_args()
    a.seeds = list(range(a.seeds))
    if a.audit:
        audit()
    else:
        main(a)
