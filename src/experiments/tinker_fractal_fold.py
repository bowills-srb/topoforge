"""Fast, low-rigor tinkering: does a genuinely SELF-SIMILAR (fractal) fold
beat blob, where a single-scale filament (Exp 45/45c) could not?

MOTIVATION (from conversation, grounded in the "universal blueprint for
mammalian brain shape" literature -- Mota/Herculano-Houzel-style scaling):
real cortical folding is specifically fractal -- small folds resemble
large folds, a self-similar structure across scales, not just "a network
with branches." Every filament shape tried in this project so far
(exp45's Voronoi-ridge density field, and every tinker_shape/fragmentation
variant) is SINGLE-scale: one sigma, one characteristic tube thickness,
no hierarchy. None of the six failed shape-optimization attempts touched
multi-scale self-similarity as an independent axis -- this script does.

CONSTRUCTION. A serpentine backbone of ROWS root segments sweeps the
bounding box (coarse structure). Each root segment is then recursively
folded MAX_DEPTH times: at each level, the segment's midpoint is displaced
perpendicular by a random-signed amount, and the two resulting half-
segments are folded again with amplitude shrunk by AMP_DECAY. This is a
Koch-curve-style construction -- the finished path has genuine bends at
every scale from ROOT_AMPLITUDE down to ROOT_AMPLITUDE * AMP_DECAY^MAX_DEPTH,
each smaller fold a scaled copy of the pattern that produced it. Points
are then sampled near this path with the SAME density-field / threshold
machinery as exp45's filament (single sigma -- the self-similarity lives
in the skeleton's geometry, not in a tube-thickness hierarchy), and the
SAME per-segment dominant-type bias trick controls coverage, so this can
be compared to blob and to the existing (non-fractal) filament on equal
footing.

COMPARISON. At matched coverage (~0.417, blob clumped30's operating
point, alpha calibrated by a small grid search): fractal-fold vs
blob:clumped30 vs the existing FILAMENT-LOW config from Exp 45c. If
fractal-fold beats plain filament, self-similarity is doing real work.
If it also beats or matches blob, that's a genuinely new result. If it
lands with plain filament (no better), self-similarity isn't the missing
ingredient and the six-failed-attempts conclusion stands.

NOT a formal experiment -- reduced seeds, no pre-registration.

Run: python src/experiments/tinker_fractal_fold.py
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

import numpy as np

from exp32b_benchmark import NC, N, run_life
from exp43_coverage_confirmatory import make_condition, mediators
from exp45_cosmic_web_topology import _field, BBOX, make_filament_density_biased

QUICK_SEEDS = [0, 1, 2, 3, 4]
TARGET_COVER = 0.417
COVER_TOL = 0.06

ROWS = 5
MAX_DEPTH = 5
ROOT_AMPLITUDE = 6.0
AMP_DECAY = 0.62
SIGMA = 1.4
THRESHOLD = 0.35
BIAS_STRENGTH = 0.99


def serpentine_backbone(bbox, rows):
    x0, x1, y0, y1 = bbox
    ys = np.linspace(y0, y1, rows + 1)
    pts = []
    for i, y in enumerate(ys):
        pts.append((x0, y) if i % 2 == 0 else (x1, y))
        pts.append((x1, y) if i % 2 == 0 else (x0, y))
    return list(zip(pts[:-1], pts[1:]))


def fractal_fold_segments(a, b, depth, max_depth, amplitude, amp_decay, rng):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if depth >= max_depth:
        return [(a, b)]
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-9:
        return [(a, b)]
    perp = np.array([-d[1], d[0]]) / length
    sign = 1.0 if rng.random() < 0.5 else -1.0
    mid = (a + b) / 2 + perp * amplitude * sign
    segs = fractal_fold_segments(a, mid, depth + 1, max_depth, amplitude * amp_decay, amp_decay, rng)
    segs += fractal_fold_segments(mid, b, depth + 1, max_depth, amplitude * amp_decay, amp_decay, rng)
    return segs


def build_fractal_skeleton(seed, rows=ROWS, max_depth=MAX_DEPTH,
                            root_amplitude=ROOT_AMPLITUDE, amp_decay=AMP_DECAY, bbox=BBOX):
    """Returns seg_a, seg_b, root_id -- root_id says which backbone leg each
    fine leaf segment descends from, so dominant-type bias can be assigned
    per COARSE leg (matching how blob/filament bias by Voronoi cell / blob,
    a whole spatial region at once) rather than per fine fold twig, which
    would force local type-mixing regardless of bias just from siblings
    being geometrically close after several recursive splits."""
    rng = np.random.default_rng(9000 + seed)
    backbone = serpentine_backbone(bbox, rows)
    seg_a, seg_b, root_id = [], [], []
    for ridx, (a, b) in enumerate(backbone):
        for sa, sb in fractal_fold_segments(a, b, 0, max_depth, root_amplitude, amp_decay, rng):
            seg_a.append(sa); seg_b.append(sb); root_id.append(ridx)
    return np.array(seg_a), np.array(seg_b), np.array(root_id), len(backbone)


def make_fractal_fold(seed, alpha, sigma=SIGMA, threshold=THRESHOLD, bias_strength=BIAS_STRENGTH,
                       rows=ROWS, max_depth=MAX_DEPTH, root_amplitude=ROOT_AMPLITUDE,
                       amp_decay=AMP_DECAY, bbox=BBOX, batch=6000, max_batches=200):
    """Same density-field / per-segment-type-bias recipe as
    make_filament_density_biased, but the skeleton is a self-similar
    fractal fold instead of a single-scale Voronoi ridge network."""
    seg_a, seg_b, root_id, n_roots = build_fractal_skeleton(seed, rows, max_depth, root_amplitude, amp_decay, bbox)
    rng = np.random.default_rng(9500 + seed * 197 + int(sigma * 1000) + int(threshold * 1000))
    x0, x1, y0, y1 = bbox
    accepted = []
    for _ in range(max_batches):
        cand = rng.uniform([x0, y0], [x1, y1], size=(batch, 2))
        f = _field(cand, seg_a, seg_b, sigma)
        keep = cand[f > threshold]
        accepted.append(keep)
        if sum(len(a) for a in accepted) >= N:
            break
    coords = np.concatenate(accepted, axis=0)[:N]
    if len(coords) < N:
        raise RuntimeError(f"only sampled {len(coords)}/{N} points -- lower threshold or raise max_batches")
    P = coords[:, None, :]
    A = seg_a[None, :, :]
    B = seg_b[None, :, :]
    D = B - A
    denom = np.maximum((D * D).sum(-1), 1e-9)
    t = np.clip(((P - A) * D).sum(-1) / denom, 0.0, 1.0)
    proj = A + t[..., None] * D
    dist2 = ((P - proj) ** 2).sum(-1)
    nearest_seg = dist2.argmin(axis=1)
    nearest_root = root_id[nearest_seg]
    dominant_by_root = rng.integers(0, NC, size=n_roots)
    cids = np.empty(N, dtype=int)
    for k in range(N):
        if rng.random() < alpha:
            cids[k] = rng.integers(0, NC)
        else:
            cids[k] = dominant_by_root[nearest_root[k]] if rng.random() < bias_strength else rng.integers(0, NC)
    return coords, cids


def calibrate_coverage(seed=0, target=TARGET_COVER, tol=COVER_TOL):
    best = None
    for alpha in np.linspace(0.0, 1.0, 11):
        try:
            coords, cids = make_fractal_fold(seed, alpha=alpha)
        except RuntimeError:
            continue
        m = mediators(coords, cids)
        d = abs(m["cover"] - target)
        if best is None or d < best[0]:
            best = (d, alpha, m, coords, cids)
    return best


def growth_of(coords, cids, seeds=QUICK_SEEDS):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


if __name__ == "__main__":
    print("=" * 100)
    print("TINKER: fractal self-similar fold vs blob vs plain (single-scale) filament")
    print("=" * 100)

    dist, alpha, m_frac, coords_frac, cids_frac = calibrate_coverage()
    print("  calibrated alpha={:.2f} -> cover={:.3f} (target {:.3f}, miss={:.3f})".format(
        alpha, m_frac["cover"], TARGET_COVER, dist))
    g_frac = growth_of(coords_frac, cids_frac)
    print("  FRACTAL-FOLD   count={:.2f} cover={:.3f}  growth={:+.1f} +/- {:.1f}".format(
        m_frac["count"], m_frac["cover"], g_frac.mean(), g_frac.std()))

    coords_b, cids_b = make_condition("clumped30", seed=0)
    m_b = mediators(coords_b, cids_b)
    g_b = growth_of(coords_b, cids_b)
    print("  BLOB clumped30 count={:.2f} cover={:.3f}  growth={:+.1f} +/- {:.1f}".format(
        m_b["count"], m_b["cover"], g_b.mean(), g_b.std()))

    coords_f, cids_f = make_filament_density_biased(
        seed=0, alpha=0.0, sigma=2.4, threshold=1.0, bias_strength=0.99)
    m_f = mediators(coords_f, cids_f)
    g_f = growth_of(coords_f, cids_f)
    print("  PLAIN filament count={:.2f} cover={:.3f}  growth={:+.1f} +/- {:.1f}".format(
        m_f["count"], m_f["cover"], g_f.mean(), g_f.std()))

    from scipy import stats
    t1, p1 = stats.ttest_ind(g_frac, g_f, equal_var=False)
    t2, p2 = stats.ttest_ind(g_frac, g_b, equal_var=False)
    print("\n  Welch t={:.2f}, p={:.3e}  fractal-fold vs plain filament".format(t1, p1))
    print("  Welch t={:.2f}, p={:.3e}  fractal-fold vs blob".format(t2, p2))
    if g_frac.mean() > g_f.mean() and p1 < 0.05:
        print("  Self-similarity helps: fractal-fold beats plain filament at matched coverage.")
    else:
        print("  Self-similarity does not measurably help over plain filament at matched coverage.")
    if g_frac.mean() > g_b.mean() and p2 < 0.05:
        print("  AND fractal-fold beats blob -- a genuinely new result.")
    elif p2 >= 0.05:
        print("  Statistically indistinguishable from blob.")
    else:
        print("  Blob still wins over fractal-fold.")
