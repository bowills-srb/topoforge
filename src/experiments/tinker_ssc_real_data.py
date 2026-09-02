"""Fast, low-rigor tinkering: does the interleaved-vs-segregated learning
penalty replicate on a SECOND real spike-encoded dataset, or is the
5.1x-8.2x SHD result (Exp 37b/37c) an artifact specific to that one corpus?

MOTIVATION (from conversation): every real-data number in this project's
preprint comes from one dataset, the Spiking Heidelberg Digits (SHD).
Spiking Speech Commands (SSC) is the sibling dataset from the same Zenke
Lab release (same HDF5 schema, same 700-channel encoding, same known
seconds-not-ms quirk) but a different, larger, harder task: 35 word
classes from Google Speech Commands re-encoded to spikes, vs SHD's 20
digit classes. If the penalty shows up here too, on different words,
different speakers, a 1.75x larger class count, that is real evidence
against "this is an SHD artifact" -- the single best-value-per-effort
item on the "strengthen the claim" list, since it needs no new hardware,
no new collaborator, and reuses the validated placement/run_life physics
verbatim.

MECHANISM AND WHAT IS KEPT EXACTLY: nothing about the physics changes.
`make_placement_segregated`, `make_placement_interleaved`, and `run_life`
are imported verbatim from exp37b_v2_real_data.py -- the same import
discipline exp37c already established, so none of the PROJECT_HISTORY
gotchas (unreachable rewire block, reward-timing, decay/life mismatch)
can be reintroduced by a copy-paste edit. `report_stats`, `cohens_d_pooled`,
and `bootstrap_ratio_ci` are imported verbatim from exp37c. The ONLY new
code is `load_two_class_samples_ssc`, a straight swap of exp37c's
`load_two_class_samples` to pull from `ssc_loader.load_ssc_samples`
instead of `shd_loader.load_shd_samples` -- same two-most-common-class
selection, same distinct-sample-per-class pooling.

SIZING: mirrors exp37c's MATCHED-LIFE config (the one that disentangled
sample diversity from life length on SHD) -- 20 distinct samples/class x
1 epoch x 12 seeds, same ~8,600-step life, same 0.99995 V-decay retention
band (~0.65), same MIN_RETENTION assertion. SSC's valid split (9,981
samples, downloaded once, ~155MB) comfortably supports a pool of 3,000
(observed top classes have 120-134 samples each, well above the 20/class
needed).

WEAKER THAN THE ORIGINAL, HONESTLY:
  - SSC's ssc_valid split is used (smallest of train/test/valid, chosen
    for download size), not train -- not SSC's canonical benchmark split.
  - Still a single dataset run at n=12 seeds, tinkering-grade: no fresh
    pre-registration, no bootstrap-CI cross-check against a held-out
    second class pair.
  - "Cross-type association" here means the same thing it means in SHD:
    binding an INPUT-type population's spike train to two OUTPUT-type
    "bridge" populations via the same two-class 2AFC-style setup exp37b
    built -- it is not adapted to SSC's larger vocabulary or word/speaker
    structure in any way beyond swapping the two most-common classes in.
  - Only one class pair tested (the two most common), same as exp37c's
    default -- not swept across multiple class pairs.

NOT a formal experiment -- tinkering grade throughout; would need its own
pre-registration to promote to a numbered experiment.

Run: python src/experiments/tinker_ssc_real_data.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import time
from collections import Counter

from ssc_loader import load_ssc_samples
from exp37b_v2_real_data import make_placement_segregated, make_placement_interleaved, run_life
from exp37c_real_data_scaled import (
    cohens_d_pooled, bootstrap_ratio_ci, report_stats,
    INPUT_TYPE_CHANNELS, MIN_RETENTION, STEPS_PER_SAMPLE,
)
import numpy as np
from spatial import SpatialGrid
from exp37b_v2_real_data import INPUT_TYPE


def load_two_class_samples_ssc(samples_per_class, pool_size, max_steps=200):
    """Same recipe as exp37c.load_two_class_samples, pointed at SSC."""
    stimuli, labels = load_ssc_samples(
        n_samples=pool_size, n_channels_out=INPUT_TYPE_CHANNELS,
        bin_ms=4.0, max_steps=max_steps)
    counts = Counter(labels)
    top = [c for c, _ in counts.most_common(2)]
    avail = {c: counts[c] for c in top}
    tgt = [s for s, l in zip(stimuli, labels) if l == top[0]][:samples_per_class]
    dis = [s for s, l in zip(stimuli, labels) if l == top[1]][:samples_per_class]
    return top, avail, tgt, dis


def run_config_ssc(samples_per_class, n_epochs, seeds, pool_size, label):
    print("=" * 74)
    print("TINKER (SSC): {}".format(label))
    print("=" * 74)

    top, avail, target_samples, distractor_samples = load_two_class_samples_ssc(
        samples_per_class, pool_size)
    got = min(len(target_samples), len(distractor_samples))
    print("Classes: target={} (avail {}), distractor={} (avail {})".format(
        top[0], avail[top[0]], top[1], avail[top[1]]))
    print("Using {} samples/class x {} classes x {} epochs".format(
        got, 2, n_epochs))
    if got < samples_per_class:
        print("  WARNING: only {} samples/class available (< {} requested); "
              "raise pool_size.".format(got, samples_per_class))

    est_steps = 2 * got * n_epochs * STEPS_PER_SAMPLE
    retention = 0.99995 ** est_steps
    print("Life length: ~{:,} steps  |  V retention at end: {:.3f} "
          "(validated ~0.65 at 8,600 steps)".format(est_steps, retention))
    assert retention >= MIN_RETENTION, (
        "V retention {:.4f} < {} -- config would decay early structure to "
        "noise (gotcha #2). Reduce samples_per_class*n_epochs.".format(
            retention, MIN_RETENTION))

    strategies = [("segregated", make_placement_segregated),
                  ("interleaved", make_placement_interleaved)]
    print("\nAdjacency parity (input-output pairs within radius 6.0):")
    for name, fn in strategies:
        c, ci = fn()
        g = SpatialGrid(c, 6.0)
        inp = set(np.where(ci == INPUT_TYPE)[0].tolist())
        out = set(np.where(ci != INPUT_TYPE)[0].tolist())
        adj = sum(1 for i in inp for j in g.within(i, 6.0) if int(j) in out)
        print("  {:>12}: {:,} adjacent pairs".format(name, adj))

    results = {}
    for name, fn in strategies:
        print("\n  {}".format(name.upper()))
        seed_totals = []
        for s in seeds:
            t0 = time.time()
            coords, cids = fn()
            ba, bb = run_life(coords, cids, s, target_samples,
                              distractor_samples, n_epochs=n_epochs)
            seed_totals.append(ba + bb)
            print("    seed {:>2}: bridge_A={:>7.1f}  bridge_B={:>7.1f}  "
                  "total={:>7.1f}  ({:.0f}s)".format(s, ba, bb, ba + bb,
                                                     time.time() - t0))
        results[name] = np.array(seed_totals, dtype=float)
    return results


if __name__ == "__main__":
    res = run_config_ssc(samples_per_class=20, n_epochs=1,
                          seeds=list(range(12)), pool_size=3000,
                          label="MATCHED-LIFE (SSC) -- 20 samples/class, 1 epoch, 12 seeds")
    report_stats(res)
    print("\nReference: SHD MATCHED-LIFE (exp37c) was 7.25x, CI [5.05x, 11.48x].")
