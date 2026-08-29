"""Exp 49: does the SpiNeCluster partial-knowledge cliff (Exp 48) replicate
on real SHD speech data, with a REAL data-derived association graph instead
of the synthetic pattern-based one used in every prior SpiNeCluster test?

MOTIVATION. Exp 48 found a sharp threshold, not a smooth curve: a mapper
given partial-confidence knowledge of future associations gets almost
nothing until that confidence is close to certain. That result used a
SYNTHETIC association graph (a fixed type-affinity matrix derived from the
task's pattern design, not real data). This experiment asks whether the
same cliff shows up when the "knowledge" fed to SpiNeCluster is a REAL,
data-derived signal instead of a clean synthetic one: for each of the 140
real SHD input channels, how much it actually fires during TARGET-class
samples versus DISTRACTOR-class samples (real spike counts, not a design
choice), used as the input-to-output-population affinity SpiNeCluster is
given. Real channel activity is noisy and graded, unlike the synthetic
model's uniform per-type weight -- if the cliff is a synthetic-graph
artifact (an artifact of every association having identical, uniform
weight), it might not survive contact with a noisy, non-uniform real
signal. If it does survive, the practical conclusion (partial-confidence
knowledge doesn't help much) is on much firmer ground.

DESIGN. Same-type affinity (input-input, A-A, B-B) fixed at 1.0, matching
Exp 38/48's synthetic-graph convention. Cross-type affinity: for input
channel c, W[c, A-population] = knowledge_frac * (real target-class
activity of channel c, normalized to [0,1] against the most active
channel), and symmetrically W[c, B-population] using real distractor-class
activity. A-B affinity is 0. SpiNeCluster (Kernighan-Lin) partitions the
resulting 200-neuron graph into clusters, SpiNePlacer places clusters onto
a small ring-shaped fabric (10 cores, matching the scale of the validated
SHD substrate -- same disc center, comparable radius to the segregated/
interleaved geometry validated in Exp 37b/37c), and neurons are scattered
within their assigned core exactly as every other placement in this
project is. Growth is measured as the SHD harness's own native metric,
bridge_a + bridge_b (cumulative V-weighted bridge mass) -- this metric is
already immune to the raw-taught-mass contamination documented in gotcha
#11, because V only ever accrues through spatially-gated eligibility
starting from zero (verified in the original 2026-08-29 audit), so no
correction is needed here.

Data: the SHD loader's real samples (Spiking Heidelberg Digits), 20
distinct samples per class from the two most common classes, 1 epoch --
40 total presentations, matching the presentation COUNT of the originally
validated 4-samples x 5-epochs x 2-classes regime (also 40), so total life
length and V retention stay in the previously validated band without
needing new decay calibration.

PREDICTIONS, registered before running:
  (P1) Population (knowledge_frac=0) vs functional (knowledge_frac=1)
       replicates Exp 38's qualitative finding on real data: functional
       learns substantially more than population.
  (P2) THE REAL QUESTION: does the partial-knowledge curve show the same
       cliff shape found in Exp 48 (flat until near-certainty, then a
       jump), or does real, graded, non-uniform channel activity produce
       a smoother, more front-loaded curve? No shape assumed in advance.

Run: python src/experiments/exp49_shd_partial_knowledge.py
"""
import numpy as np
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats

from shd_loader import load_shd_samples
from exp37b_v2_real_data import (
    N, N_INPUT, N_OUTPUT_A, N_OUTPUT_B, INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE,
    run_life,
)
from exp38_spinemap_baseline import spinecluster, cluster_traffic, particle_swarm_placement

SEEDS = list(range(8))
SAMPLES_PER_CLASS = 20
N_EPOCHS = 1
POOL_SIZE = 1500
N_CLUSTERS = 10
CLUSTER_SIZE = N // N_CLUSTERS  # 20
DISC_CENTER = (20.0, 20.0)
CORE_RING_RADIUS = 7.0
CORE_SCATTER = 1.8
KNOWLEDGE_FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

CIDS = np.array([INPUT_TYPE] * N_INPUT + [OUTPUT_A_TYPE] * N_OUTPUT_A + [OUTPUT_B_TYPE] * N_OUTPUT_B)


def load_two_class_samples(samples_per_class, pool_size, max_steps=200):
    stimuli, labels = load_shd_samples(n_samples=pool_size, n_channels_out=N_INPUT,
                                        bin_ms=4.0, max_steps=max_steps)
    counts = Counter(labels)
    top = [c for c, _ in counts.most_common(2)]
    avail = {c: counts[c] for c in top}
    tgt = [s for s, l in zip(stimuli, labels) if l == top[0]][:samples_per_class]
    dis = [s for s, l in zip(stimuli, labels) if l == top[1]][:samples_per_class]
    return top, avail, tgt, dis


def channel_activity(samples):
    total = np.zeros(N_INPUT)
    for s in samples:
        total += s.sum(axis=0)
    return total


def build_real_graph(act_t, act_d, knowledge_frac):
    norm_t = act_t / max(act_t.max(), 1e-9)
    norm_d = act_d / max(act_d.max(), 1e-9)
    W = np.zeros((N, N))
    same = CIDS[:, None] == CIDS[None, :]
    W[same] = 1.0
    np.fill_diagonal(W, 0.0)
    a_idx = np.where(CIDS == OUTPUT_A_TYPE)[0]
    b_idx = np.where(CIDS == OUTPUT_B_TYPE)[0]
    for c in range(N_INPUT):
        wt = knowledge_frac * norm_t[c]
        wd = knowledge_frac * norm_d[c]
        W[c, a_idx] = wt; W[a_idx, c] = wt
        W[c, b_idx] = wd; W[b_idx, c] = wd
    return W


def core_positions_ring(n_cores, center=DISC_CENTER, radius=CORE_RING_RADIUS):
    cx, cy = center
    return np.array([[cx + radius * np.cos(2 * np.pi * k / n_cores),
                       cy + radius * np.sin(2 * np.pi * k / n_cores)]
                      for k in range(n_cores)])


def neurons_in_core_disc(center, scatter=CORE_SCATTER, n=CLUSTER_SIZE, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    r = scatter * np.sqrt(rng.uniform(0, 1, n))
    cx, cy = center
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)


def make_placement_real(W, seed=0):
    part = spinecluster(W, CLUSTER_SIZE, N_CLUSTERS, seed=seed)
    Wc = cluster_traffic(W, part, N_CLUSTERS)
    core_pos = core_positions_ring(N_CLUSTERS)
    assign, _, _ = particle_swarm_placement(Wc, core_pos, seed=seed)
    coords = np.zeros((N, 2))
    for c in range(N_CLUSTERS):
        members = np.where(part == c)[0]
        core_id = int(assign[c])
        pts = neurons_in_core_disc(core_pos[core_id], seed=core_id + 5000)
        coords[members] = pts[:len(members)]
    return coords, CIDS.copy()


def bridge_of(coords, cids, target_samples, distractor_samples, seeds=SEEDS):
    out = []
    for s in seeds:
        ba, bb = run_life(coords, cids, s, target_samples, distractor_samples, n_epochs=N_EPOCHS)
        out.append(ba + bb)
    return np.array(out, dtype=float)


if __name__ == "__main__":
    print("=" * 78)
    print("EXP 49: partial knowledge on REAL SHD data-derived association graph")
    print("=" * 78)

    print("\n[0] Loading real SHD samples ({} pool, {} per class)".format(POOL_SIZE, SAMPLES_PER_CLASS))
    t0 = time.time()
    top, avail, target_samples, distractor_samples = load_two_class_samples(SAMPLES_PER_CLASS, POOL_SIZE)
    got = min(len(target_samples), len(distractor_samples))
    print("  classes: target={} (avail {}), distractor={} (avail {})".format(
        top[0], avail[top[0]], top[1], avail[top[1]]))
    print("  using {} samples/class (loaded in {:.0f}s)".format(got, time.time() - t0))
    if got < SAMPLES_PER_CLASS:
        print("  WARNING: fewer distinct samples available than requested.")

    act_t = channel_activity(target_samples)
    act_d = channel_activity(distractor_samples)
    print("  real channel activity: target mean={:.2f} max={:.2f}  distractor mean={:.2f} max={:.2f}".format(
        act_t.mean(), act_t.max(), act_d.mean(), act_d.max()))

    print("\n[1] Growth (bridge_a+bridge_b) vs knowledge_frac, {} seeds each".format(len(SEEDS)))
    results = {}
    for kf in KNOWLEDGE_FRACS:
        W = build_real_graph(act_t, act_d, kf)
        coords, cids = make_placement_real(W, seed=0)
        g = bridge_of(coords, cids, target_samples, distractor_samples)
        results[kf] = g
        print("  knowledge_frac={:>5.0%}  bridge={:+8.1f} +/- {:<6.1f}".format(kf, g.mean(), g.std()))

    print("\n" + "=" * 78)
    print("[2] Recovery: what fraction of the population->functional gap does each level close?")
    print("=" * 78)
    pop_g = results[0.0].mean()
    fn_g = results[1.0].mean()
    span = fn_g - pop_g
    print("  gap: population {:+.1f} -> functional {:+.1f}  (span {:.1f})".format(pop_g, fn_g, span))
    print("  {:>14} {:>10} {:>12} {:>14}".format("knowledge", "bridge", "%recovered", "Welch p vs pop"))
    frac_at_50, frac_at_90 = None, None
    for kf in KNOWLEDGE_FRACS:
        g = results[kf]
        pct = (g.mean() - pop_g) / span if span != 0 else float("nan")
        t, p = stats.ttest_ind(g, results[0.0], equal_var=False)
        print("  {:>13.0%} {:>+10.1f} {:>11.1%} {:>14.2e}".format(kf, g.mean(), pct, p))
        if frac_at_50 is None and pct >= 0.5:
            frac_at_50 = kf
        if frac_at_90 is None and pct >= 0.9:
            frac_at_90 = kf

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    p1_t, p1_p = stats.ttest_ind(results[1.0], results[0.0], equal_var=False)
    p1_holds = p1_p < 0.05 and fn_g > pop_g
    print("  P1 (functional > population on real data): {} (p={:.2e})".format(
        "HOLDS" if p1_holds else "FAILS", p1_p))
    if frac_at_50 is not None and frac_at_50 <= 0.5:
        print("  P2: CONCAVE on real data too -- knowledge_frac={:.0%} recovers >=50% of the".format(frac_at_50))
        print("  gap. The Exp 48 cliff does NOT replicate; real, graded channel activity gives")
        print("  partial knowledge more traction than the uniform synthetic graph did.")
    else:
        shown = "{:.0%}".format(frac_at_50) if frac_at_50 is not None else "never (within this sweep)"
        print("  P2: the cliff REPLICATES on real data -- 50% recovery not reached until")
        print("  knowledge_frac={}. Partial-confidence knowledge is not useful on real".format(shown))
        print("  data either; the mapper needs near-certain associations before they help,")
        print("  strengthening (not weakening) Exp 48's practical conclusion.")
