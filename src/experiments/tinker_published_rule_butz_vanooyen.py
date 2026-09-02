"""Fast, low-rigor tinkering: does the segregated-vs-interleaved penalty
survive under a structural plasticity rule this project did NOT design --
one with no value signal, no reward, and no co-firing correlation term at
all?

MOTIVATION. Every growth rule tested elsewhere in this project (`Life`/
`LocalLife` in src/engine.py, and every exp3x/4x benchmark) is driven by
this project's own C/E/V machinery: correlation, eligibility, and
reward-attributed value on candidate (i, j) pairs. That leaves open an
obvious objection: is the segregation penalty a fact about *placement*,
or an artifact of *this specific rule's* candidate-scoring design? Exp 38
already closed the matching gap on the mapper side (a real published
partitioner, SpiNeMap, reproduces the segregated baseline). This script
closes it on the growth-rule side using a real published PLASTICITY
model instead of a hand-rolled one.

MECHANISM (Butz & van Ooyen 2013, "A simple rule for dendritic spine and
axonal bouton formation can account for cortical reorganization after
focal retinal lesions", PLoS Comput Biol; the same "Gaussian" growth
curve NEST's synaptic-elements SP module implements). Each neuron carries
an axonal-element budget a_i and a dendritic-element budget d_i. Once per
epoch, each neuron's realized firing RATE (spike count / epoch length,
its own activity only -- no pairwise term of any kind) is compared to a
homeostatic target via the Gaussian growth curve

    F(rate) = 2 * exp(-((rate - eps) / zeta)**2) - 1

which is +1 exactly at the target rate and falls toward -1 the further
rate strays from it in EITHER direction -- growth below target, retraction
above it, in both directions, not just runaway potentiation. a_i and d_i
are nudged by nu * F(rate_i) each epoch (floored at 0, capped so a runaway
neuron can't blow up the search). WHO connects to WHOM is then decided
purely by a spatial Gaussian: available axonal "slots" are randomly
matched to available dendritic "slots" with probability falling off as
exp(-dist(i,j)^2 / (2*sigma^2)). There is no term anywhere that knows or
cares about neuron TYPE or co-firing history -- if segregated placement
still produces less cross-type structure than interleaved under this
rule, the effect is a fact about geometry interacting with ANY local
growth process, not an artifact of this project's own C/E/V scoring.

SIMPLIFICATION vs. the published model (disclosed, not hidden): the paper
grows/retracts individual BOUND elements incrementally and only frees a
partner's element when its own synapse is deleted. This script instead
recomputes the full target quota (floor(a_i), floor(d_i)) each epoch and
re-derives the whole edge set from those quotas via the same spatial-
Gaussian pairing, discarding the previous epoch's specific edges. This
keeps the two things the test actually needs -- (1) local, rate-driven
element budgets, (2) purely spatial, type-blind pairing -- while dropping
the incremental bookkeeping, at the cost of edges not literally persisting
synapse-by-synapse across epochs the way the real model's do. Tinkering
grade throughout: 5 seeds, no pre-registration.

COMPARISON. Segregated ("vlsi") vs interleaved ("topoforge") placement
(exp32b_benchmark.make_placement), same LIF dynamics, same PATTERNS
input schedule, same core layout as every other PLB-family script. Network
starts from ZERO edges and grows entirely under the rule above -- there is
no separate frozen/plastic split the way run_life has, because there is no
pre-existing non-mechanistic edge baseline to strip out; the final
connectivity IS the thing being measured. Reported: (a) the SAME specific
cross-type mass exp32b measures (types (0,3) and (1,4), the two pattern-
coupled pairs), for continuity, and (b) a rule-agnostic broad proxy --
fraction of ALL final edges that connect two DIFFERENT types -- since this
rule has no reason to prefer exp32b's particular pairs over any other
cross-type pair.

Run: python src/experiments/tinker_published_rule_butz_vanooyen.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np
from scipy import stats

from exp32b_benchmark import make_placement, PATTERNS, NC, N

QUICK_SEEDS = [0, 1, 2, 3, 4]

LEAK = 0.90
THRESH = 1.0
REFRAC_STEPS = 3
NOISE = 0.02

EPOCH = 60      # = len(PATTERNS)*20, one full pattern supercycle -- every
                # neuron's type gets exactly one active window per epoch,
                # so the epoch-level rate is phase-stable across epochs
                # instead of an EPOCH=40 window landing on a different,
                # inconsistent subset of the 3 pattern blocks each time
                # (empirically: that made rate bimodal at {0.0, 0.05} with
                # nothing near a population-average target, so homeostatic
                # growth could never settle -- caught by inspection below)
N_EPOCHS = 13
STEPS = EPOCH * N_EPOCHS   # 780, close to exp32b's 800-step frozen+plastic window

NU_AX = 1.5     # max axonal elements gained/lost per epoch (|F|=1)
NU_DEN = 1.5
MAX_ELEMENTS = 25.0
FORM_SIGMA = 4.0   # spatial pairing width; core spacing in exp32b is 5.0

PILOT_STEPS = EPOCH * 6  # multiple of EPOCH, for the same phase-stability reason


def weight_of(inhib, s):
    return -0.60 if inhib[s] else 0.30


def lif_pilot_rates(coords, cids, seed, steps=PILOT_STEPS):
    """Firing rate per neuron with NO recurrent connectivity at all -- pure
    feedforward drive from the PATTERNS schedule + noise. Used only to pick
    a homeostatic target rate that's actually reachable; since both
    placements have identical per-type population counts and the same
    input schedule, the feedforward-only rate distribution is (up to seed
    noise) the same for both, so this is calibrated once, not per
    condition, keeping the comparison fair."""
    rng = np.random.default_rng(seed)
    N_ = len(coords)
    v = np.zeros(N_)
    refrac = np.zeros(N_, dtype=int)
    spikes = np.zeros(N_, dtype=int)
    for t in range(steps):
        p = (t // 20) % len(PATTERNS)
        inputs = rng.uniform(0, NOISE, N_)
        if (t % 20) < 5:
            for cid in PATTERNS[p]:
                inputs[cids == cid] += 0.5
        v_ = v * LEAK + inputs
        fired = (v_ >= THRESH) & (refrac == 0)
        v = np.maximum(v_, 0)
        v[fired] = 0
        refrac[fired] = REFRAC_STEPS
        refrac[refrac > 0] -= 1
        spikes[fired] += 1
    return spikes / steps


def calibrate_target(coords, cids, seed=1234):
    rates = lif_pilot_rates(coords, cids, seed)
    eps = max(float(rates.mean()), 1e-3)
    zeta = max(float(rates.std()), eps * 0.5, 1e-3)
    return eps, zeta


def growth_curve(rate, eps, zeta):
    return 2.0 * np.exp(-((rate - eps) / zeta) ** 2) - 1.0


def form_edges(A, D, D2, sigma, rng):
    """Random axonal-slot -> dendritic-slot pairing, probability falling
    off as a spatial Gaussian. No neuron-type term anywhere."""
    N_ = len(A)
    den_remaining = D.copy()
    ax_slots = np.repeat(np.arange(N_), A)
    rng.shuffle(ax_slots)
    inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    src, dst = [], []
    for i in ax_slots:
        avail = den_remaining > 0
        avail[i] = False
        if not avail.any():
            continue
        w = np.exp(-D2[i] * inv2sig2)
        w[~avail] = 0.0
        s = w.sum()
        if s <= 0:
            continue
        j = rng.choice(N_, p=w / s)
        src.append(i)
        dst.append(int(j))
        den_remaining[j] -= 1
    return np.array(src, dtype=int), np.array(dst, dtype=int)


def run_butz_vanooyen(coords, cids, seed, eps, zeta, steps=STEPS, epoch=EPOCH,
                       nu_ax=NU_AX, nu_den=NU_DEN, sigma=FORM_SIGMA):
    rng = np.random.default_rng(seed)
    N_ = len(coords)
    D2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    inhib = rng.random(N_) < 0.20

    v = np.zeros(N_)
    refrac = np.zeros(N_, dtype=int)
    a = np.zeros(N_)
    d = np.zeros(N_)
    spike_count = np.zeros(N_, dtype=int)
    out_t = [[] for _ in range(N_)]
    src = np.empty(0, dtype=int)
    dst = np.empty(0, dtype=int)

    for t in range(steps):
        p = (t // 20) % len(PATTERNS)
        inputs = rng.uniform(0, NOISE, N_)
        if (t % 20) < 5:
            for cid in PATTERNS[p]:
                inputs[cids == cid] += 0.5

        v_ = v * LEAK + inputs
        fired = (v_ >= THRESH) & (refrac == 0)
        f = np.where(fired)[0]
        if len(f):
            for i in f:
                for j, w in out_t[i]:
                    v_[j] += w
        v = np.maximum(v_, 0)
        v[fired] = 0
        refrac[fired] = REFRAC_STEPS
        refrac[refrac > 0] -= 1
        spike_count[f] += 1

        if (t + 1) % epoch == 0:
            rate = spike_count / epoch
            spike_count[:] = 0
            F = growth_curve(rate, eps, zeta)
            a = np.clip(a + nu_ax * F, 0.0, MAX_ELEMENTS)
            d = np.clip(d + nu_den * F, 0.0, MAX_ELEMENTS)
            A = np.floor(a).astype(int)
            D = np.floor(d).astype(int)
            src, dst = form_edges(A, D, D2, sigma, rng)
            out_t = [[] for _ in range(N_)]
            for s2, d2 in zip(src, dst):
                out_t[s2].append((d2, weight_of(inhib, s2)))

    return src, dst


def measure(src, dst, cids):
    n_edges = len(src)
    if n_edges == 0:
        return dict(n_edges=0, taught=0, cross_frac=0.0, cross_count=0)
    M = np.zeros((NC, NC), dtype=int)
    np.add.at(M, (cids[src], cids[dst]), 1)
    taught = M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]
    cross = int((cids[src] != cids[dst]).sum())
    return dict(n_edges=n_edges, taught=int(taught),
                cross_frac=cross / n_edges, cross_count=cross)


def sweep(strategy, eps, zeta, seeds=QUICK_SEEDS):
    coords, cids = make_placement(strategy)
    rows = []
    for s in seeds:
        src, dst = run_butz_vanooyen(coords, cids, s, eps, zeta)
        rows.append(measure(src, dst, cids))
    return rows


def summarize(label, rows):
    n_edges = np.array([r["n_edges"] for r in rows], dtype=float)
    taught = np.array([r["taught"] for r in rows], dtype=float)
    cross_frac = np.array([r["cross_frac"] for r in rows], dtype=float)
    cross_count = np.array([r["cross_count"] for r in rows], dtype=float)
    print("  {:<24} edges={:<8.1f} taught(0,3/1,4)={:<7.1f}+/-{:<6.1f} "
          "cross_frac={:.3f}+/-{:.3f} cross_count={:.1f}".format(
              label, n_edges.mean(), taught.mean(), taught.std(),
              cross_frac.mean(), cross_frac.std(), cross_count.mean()))
    return n_edges, taught, cross_frac, cross_count


if __name__ == "__main__":
    print("=" * 100)
    print("TINKER: published structural-plasticity rule (Butz & van Ooyen 2013 Gaussian")
    print("growth curve) -- rate homeostasis + purely spatial pairing, NO value/correlation term")
    print("=" * 100)

    coords_int, cids_int = make_placement("topoforge")
    eps, zeta = calibrate_target(coords_int, cids_int)
    print("\n  calibrated homeostatic target: eps={:.4f}  zeta={:.4f}  (feedforward-only pilot rate)".format(eps, zeta))

    print("\n[final connectivity after {} epochs ({} steps), grown from zero]".format(N_EPOCHS, STEPS))
    rows_seg = sweep("vlsi", eps, zeta)
    n_seg, taught_seg, cf_seg, cc_seg = summarize("SEGREGATED (vlsi)", rows_seg)
    rows_int = sweep("topoforge", eps, zeta)
    n_int, taught_int, cf_int, cc_int = summarize("INTERLEAVED (topoforge)", rows_int)

    print("\n" + "-" * 100)
    if taught_seg.mean() > 0:
        print("  ratio interleaved/segregated, specific taught pairs (0,3)+(1,4): {:.2f}x".format(
            taught_int.mean() / max(taught_seg.mean(), 1e-9)))
    else:
        print("  segregated specific-pair taught mass is ~0 ({:.1f}); interleaved={:.1f} -- floor effect,".format(
            taught_seg.mean(), taught_int.mean()))
        print("  same qualitative pattern exp32b found for its raw (uncorrected) metric.")

    t1, p1 = stats.ttest_ind(cf_int, cf_seg, equal_var=False)
    t2, p2 = stats.ttest_ind(taught_int, taught_seg, equal_var=False)
    print("  cross-type FRACTION of edges: interleaved={:.3f}+/-{:.3f}  segregated={:.3f}+/-{:.3f}".format(
        cf_int.mean(), cf_int.std(), cf_seg.mean(), cf_seg.std()))
    print("  Welch t={:.2f}, p={:.3e}  (cross-type fraction, interleaved vs segregated)".format(t1, p1))
    print("  Welch t={:.2f}, p={:.3e}  (specific taught pairs, interleaved vs segregated)".format(t2, p2))

    if cf_int.mean() > cf_seg.mean() and p1 < 0.05:
        print("\n  PENALTY SURVIVES under a published, value-free, correlation-free plasticity rule:")
        print("  interleaved placement produces a higher cross-type-connectivity fraction than")
        print("  segregated placement even when nothing in the growth rule knows what a 'type' is.")
    else:
        print("\n  Penalty does NOT clearly survive under this rule at this seed count/power.")
