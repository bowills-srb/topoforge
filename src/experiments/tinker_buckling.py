"""Fast, low-rigor tinkering: does PASSIVE MECHANICAL buckling -- a growth-
rate mismatch between an expanding population and a fixed footprint,
with no learning signal involved at all -- reorganize a bad (segregated)
placement into something better, where VALUE-DRIVEN migration
(tinker_dynamic_placement*) already failed to?

MOTIVATION (from conversation, grounded in the differential-growth
mechanical-instability literature -- Tallinen et al.'s gel/PDMS models,
the axonal-tension work's competing "cortex grows faster than white
matter" framing): real cortical folding is now understood as a passive
mechanical consequence of a growth-rate mismatch between two attached
layers under a fixed outer boundary, NOT an activity-dependent process.
tinker_dynamic_placement's three attempts all drove neuron migration with
the same value/correlation signal (V, Sense) that drives synaptic
rewiring -- and all three failed, notably from a SEGREGATED start,
because of a chicken-and-egg problem: V can only accrue from pairs that
have already co-fired LOCALLY, the same locality that makes segregated
placement bad, so the migration signal is blind to the problem it's
meant to fix. Mechanical buckling has no such blind spot -- it doesn't
need value or correlation at all, only crowding.

MECHANISM. Neurons are treated as a 2D layer wanting to expand (a growing
"rest length" between neighbors, applied every growth event) while (a)
tethered weakly to their ORIGINAL positions (a slower-growing understructure
they stay attached to) and (b) confined to the ORIGINAL bounding box (a
fixed "skull" that does not grow at all). Local overcrowding relative to
the growing rest length produces a repulsive force; the tether and the
hard boundary mean that force cannot be resolved by uniform expansion, so
positions must reorganize locally -- the numerical analogue of buckling.
This happens ONCE, before any learning -- like real gyrification, which
is largely complete before most experience-dependent plasticity -- so the
buckled layout is just handed to the standard run_life() unmodified,
exactly like every other shape family in this project.

COMPARISON. Buckle from a SEGREGATED start ("vlsi") and from a BLOB start
(clumped30, as a sanity/side check -- does buckling disturb an
already-good layout?). Compare buckled-segregated against: static
segregated, static interleaved, static blob, AND the three failed
value-driven migration variants (narrow-sense -719, wide-sense -649,
annealed -707, all from segregated, all worse than static segregated's
own -726 is NOT true -- they're all closer to zero but still deeply
negative; see tinker_dynamic_placement3's printed reference line).

NOT a formal experiment -- reduced seeds, no pre-registration, tinkering
grade throughout.

Run: python src/experiments/tinker_buckling.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np

from exp32b_benchmark import make_placement, run_life
from exp43_coverage_confirmatory import make_condition, mediators

QUICK_SEEDS = [0, 1, 2, 3, 4]

PLASTICITY_RADIUS = 5.0   # matches run_life's hardcoded locality radius
GROWTH_EVENTS = 14
GROWTH_RATE = 0.13         # rest_length: 5.0 -> ~28 over 14 events, past the
                           # ~12-18 unit gap between blocks of measured types
REPULSION_K = 0.4
TETHER_K = 0.02
CUTOFF_MULT = 1.5
RELAX_ITERS = 40
DT = 0.15


def buckle(coords0, growth_events=GROWTH_EVENTS, growth_rate=GROWTH_RATE,
           repulsion_k=REPULSION_K, tether_k=TETHER_K, cutoff_mult=CUTOFF_MULT,
           relax_iters=RELAX_ITERS, dt=DT, start_rest_length=PLASTICITY_RADIUS):
    """Rest length starts at the plasticity radius itself (the physically
    meaningful threshold in this codebase -- crowding only matters once a
    region needs more room than local plasticity rules can reach across)
    and grows from there, not from raw nearest-neighbor spacing, which
    inside a single PLB core (15 neurons in a radius-1.5 disc) reflects
    micro-packing irrelevant to whether TYPE BLOCKS ever bridge each other.
    Force is averaged (not summed) over interacting neighbors so it stays
    bounded as rest_length grows to encompass most of the population,
    rather than blowing up once cutoff spans hundreds of neurons."""
    coords = coords0.copy().astype(float)
    rest_length = start_rest_length
    x0, x1 = coords0[:, 0].min(), coords0[:, 0].max()
    y0, y1 = coords0[:, 1].min(), coords0[:, 1].max()
    log = []
    for ev in range(growth_events):
        rest_length *= (1.0 + growth_rate)
        cutoff = rest_length * cutoff_mult
        for it in range(relax_iters):
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1)) + 1e-9
            too_close = (dist < rest_length) & (dist < cutoff)
            np.fill_diagonal(too_close, False)
            mag = np.where(too_close, repulsion_k * (rest_length - dist), 0.0)
            unit = diff / dist[..., None]
            neighbor_count = np.maximum(too_close.sum(axis=1), 1)
            force = (mag[..., None] * unit).sum(axis=1) / neighbor_count[:, None]
            force += tether_k * (coords0 - coords)
            coords = coords + dt * force
            coords[:, 0] = np.clip(coords[:, 0], x0, x1)
            coords[:, 1] = np.clip(coords[:, 1], y0, y1)
        disp = float(np.abs(coords - coords0).mean())
        log.append((rest_length, disp))
    return coords, log


def growth_of(coords, cids, seeds=QUICK_SEEDS):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


def report(label, coords, cids):
    m = mediators(coords, cids)
    g = growth_of(coords, cids)
    print("  {:<24} count={:<7.2f} cover={:<7.3f}  growth={:+8.1f} +/- {:<6.1f}".format(
        label, m["count"], m["cover"], g.mean(), g.std()))
    return m, g


if __name__ == "__main__":
    print("=" * 100)
    print("TINKER: passive mechanical buckling (growth-rate mismatch), no value/correlation signal")
    print("=" * 100)

    print("\n[reference: static baselines]")
    coords_seg, cids_seg = make_placement("vlsi")
    report("static segregated", coords_seg, cids_seg)
    coords_int, cids_int = make_placement("topoforge")
    report("static interleaved", coords_int, cids_int)
    coords_blob, cids_blob = make_condition("clumped30")
    report("static blob clumped30", coords_blob, cids_blob)
    print("  (for context, tinker_dynamic_placement3 reference line: narrow-sense migration=-719,")
    print("   wide-sense=-649, annealed=-707, all from segregated -- value-driven migration never")
    print("   got close to static interleaved's advantage)")

    print("\n[buckling from SEGREGATED start]")
    buckled_seg, log_seg = buckle(coords_seg)
    for i, (rl, disp) in enumerate(log_seg):
        print("  growth event {}: rest_length={:.2f}  mean displacement from origin={:.2f}".format(i, rl, disp))
    report("BUCKLED (from segregated)", buckled_seg, cids_seg)

    print("\n[buckling from BLOB start -- sanity check: does it disturb a good layout?]")
    buckled_blob, log_blob = buckle(coords_blob)
    report("BUCKLED (from blob)", buckled_blob, cids_blob)
