"""Fast, low-rigor tinkering: does blob's win depend on being given free,
unconstrained space to stay isolated in? Every prior blob-vs-filament
comparison (Exp 43/45/45c, the shape-optimizer sweeps) held coverage FIXED
by recalibrating composition -- meaning blob was never actually allowed to
get crowded. This script instead squeezes the physical footprint directly
(the "growing brain in a fixed skull" framing from conversation: cortical
folding is now understood as a differential-growth mechanical instability,
not literally "ran out of room" -- but a related, directly testable
question in THIS codebase is simpler: does blob's advantage survive being
forced into the same shrinking footprint filament already occupies, with
coverage left to rise HOWEVER the squeeze makes it rise, rather than being
re-pinned back down to a fixed target the way every prior sweep did?).

MECHANISM. Blob's isolated-cluster property is a straightforward function
of inter-blob spacing vs plasticity radius (5.0): once blob centers are
closer than roughly 2*blob_radius + 2*plasticity_radius, blobs start
falling inside each other's reach and "isolated" stops being true, no
matter what the composition says. Filament has no such threshold -- it is
already continuous -- so shrinking its bounding box just raises density
uniformly rather than destroying a structural property it depends on.

DESIGN. Sweep a squeeze fraction from 1.0 (today's clumped30 / filament-low
footprints) down to ~0.3. For blob: shrink the lattice pitch (`spacing` to
blob_centers) by the squeeze fraction, composition and blob radius
untouched. For filament: shrink BBOX area by the same fraction (each
dimension by sqrt(fraction)), geometry knobs (sigma/threshold/bias)
untouched. At each level, measure count/cover AS THEY NATURALLY FALL OUT
(no recalibration) and growth via run_life. Watch for: (1) the squeeze
level where blob's cover departs from ~0.417 and starts climbing toward
filament's, and (2) whether blob's growth advantage shrinks, holds, or
inverts as that happens.

NOT a formal experiment -- reduced seeds, no pre-registration, exploratory
grade. If a real effect shows up here, it earns a proper confirmatory
version (fixed squeeze levels, seeds, and a prediction registered before
running), same as every other finding in this thread.

Run: python src/experiments/tinker_realestate_squeeze.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np

from exp43_coverage_confirmatory import (
    CONDITIONS, blob_radius, mediators, PLASTICITY_RADIUS,
)
from exp45_cosmic_web_topology import make_filament_density_biased, BBOX
from exp32b_benchmark import run_life

QUICK_SEEDS = [0, 1, 2, 3, 4]
SQUEEZE_LEVELS = [1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.28]

FILAMENT_LOW_KW = dict(alpha=0.0, sigma=2.4, threshold=1.0, bias_strength=0.99)


def blob_centers_squeezed(n_blobs, spacing):
    cols = int(np.ceil(np.sqrt(n_blobs * 10.0 / 6.0)))
    return np.array([[(k % cols) * spacing, (k // cols) * spacing]
                     for k in range(n_blobs)], dtype=float)


def make_blob_squeezed(name, seed, spacing):
    """Same as exp43.make_condition, but with an adjustable lattice pitch
    instead of the fixed SPACING=11.0 -- the one knob that controls whether
    blobs stay outside each other's plasticity radius or start merging."""
    spec = CONDITIONS[name]
    blobs = []
    for n_rep, comp in spec:
        for _ in range(n_rep):
            blobs.append(comp)
    centers = blob_centers_squeezed(len(blobs), spacing)
    coords, cids = [], []
    for b, (comp, ctr) in enumerate(zip(blobs, centers)):
        n = sum(comp.values())
        R = blob_radius(n)
        rng = np.random.default_rng(1000 + seed * 997 + b)
        th = rng.uniform(0, 2 * np.pi, n)
        r = R * np.sqrt(rng.uniform(0, 1, n))
        pts = np.stack([ctr[0] + r * np.cos(th), ctr[1] + r * np.sin(th)], axis=1)
        types = []
        for t, k in sorted(comp.items()):
            types += [t] * k
        rng.shuffle(types)
        coords.append(pts)
        cids += types
    return np.vstack(coords), np.array(cids, dtype=int)


def make_filament_squeezed(seed, fraction):
    x0, x1, y0, y1 = BBOX
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2 * np.sqrt(fraction), (y1 - y0) / 2 * np.sqrt(fraction)
    bbox = (cx - hw, cx + hw, cy - hh, cy + hh)
    return make_filament_density_biased(seed=seed, bbox=bbox, **FILAMENT_LOW_KW)


def growth_of(coords, cids, seeds=QUICK_SEEDS):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


if __name__ == "__main__":
    print("=" * 100)
    print("TINKER: real-estate squeeze -- does blob's win survive losing its isolation?")
    print("squeeze=1.0 matches today's clumped30 (blob) / filament-low (filament) footprints")
    print("=" * 100)
    print("{:<8} {:<10} {:>8} {:>8} {:>10} {:>8}".format(
        "squeeze", "family", "count", "cover", "growth", "std"))

    base_spacing = 11.0  # exp43's SPACING, the squeeze=1.0 reference
    rows = []
    for frac in SQUEEZE_LEVELS:
        coords_b, cids_b = make_blob_squeezed("clumped30", seed=0, spacing=base_spacing * frac)
        m_b = mediators(coords_b, cids_b)
        g_b = growth_of(coords_b, cids_b)
        rows.append((frac, "blob", m_b, g_b))
        print("{:<8.2f} {:<10} {:>8.2f} {:>8.3f} {:>+10.1f} {:>8.1f}".format(
            frac, "blob", m_b["count"], m_b["cover"], g_b.mean(), g_b.std()))

        coords_f, cids_f = make_filament_squeezed(seed=0, fraction=frac)
        m_f = mediators(coords_f, cids_f)
        g_f = growth_of(coords_f, cids_f)
        rows.append((frac, "filament", m_f, g_f))
        print("{:<8.2f} {:<10} {:>8.2f} {:>8.3f} {:>+10.1f} {:>8.1f}".format(
            frac, "filament", m_f["count"], m_f["cover"], g_f.mean(), g_f.std()))
        sys.stdout.flush()

    print("\n" + "=" * 100)
    print("Ratio (blob growth / filament growth) by squeeze level, and where blob's cover")
    print("departs from its unsqueezed ~0.417 baseline:")
    print("=" * 100)
    by_frac = {}
    for frac, fam, m, g in rows:
        by_frac.setdefault(frac, {})[fam] = (m, g)
    for frac in SQUEEZE_LEVELS:
        mb, gb = by_frac[frac]["blob"]
        mf, gf = by_frac[frac]["filament"]
        ratio = gb.mean() / gf.mean() if gf.mean() != 0 else float("nan")
        print("  squeeze={:.2f}  blob_cover={:.3f}  filament_cover={:.3f}  ratio(blob/filament)={:.2f}x".format(
            frac, mb["cover"], mf["cover"], ratio))
