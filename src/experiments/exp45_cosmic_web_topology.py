"""Exp 45: Does filament/void TOPOLOGY matter beyond first-order reach stats?

MOTIVATION. The project's founding intuition (per the author, 2026-08-28) is
an analogy between the cosmic web (voids and filaments in galaxy-cluster
distribution) and synaptic/neuronal structure: "shape must matter." Exp
42b/43 already established that first-order LOCAL REACH statistics --
count, and especially coverage (fraction of a neuron's reachable
neighbourhood containing >=1 associated partner) -- are strong mediators
of learning (coverage R^2=0.832 pooling placement families; refuted as a
pure count effect in Exp 43's confirmatory test). But none of those
placement families were topologically filamentary/web-like -- they were
contiguous blocks, blobs on a lattice, or full shuffles. This experiment
asks the sharper, more direct version of the founding question: at MATCHED
count and coverage, does a genuinely different large-scale TOPOLOGY --
a connected filament network with voids, versus isolated compact blobs --
change what gets learned? If not, coverage is the whole story and the
cosmic-web analogy, while visually real, is not operative beyond what is
already measured. If so, this is the first result in the project that
earns the cosmic-web framing rather than merely being inspired by it.

DESIGN. FILAMENT (v2, 2026-08-29): a smooth scalar density field -- Gaussian
"tubes" summed around each ridge segment of a Voronoi diagram of K random
seed points, then thresholded -- rejection-sampled for N points. This
replaces a v1 that scattered points directly on the Voronoi ridge skeleton
(straight line segments), which a reviewer correctly flagged as having the
wrong texture: "the sharp angles are wrong, it's more shaped like a gel
poured over a skull." The density-field version tapers and thickens
continuously (tubes overlap near junctions, exactly where real cosmic-web
filaments thicken), and void boundaries are curved, not polygonal -- see
make_filament_density / make_filament_density_biased and the "Cosmic Web
Check" artifact for a visual before/after. Per-point type is either
uniformly shuffled (make_filament_density, composition free like exp43's
"dilute30") or biased toward each point's nearest ridge segment's assigned
dominant type at rate (1-alpha) (make_filament_density_biased), which sweeps
coverage down from ~1.0 while the underlying smooth geometry -- and hence
raw local density -- is held fixed. BLOB: reuses exp43's "dilute30" (C2)
and "spread15"/"clumped30" verbatim -- isolated, compact clusters,
structurally the opposite extreme from a connected sparse network. Both
families use the same N, NC, PATTERNS, and plasticity radius.

Alpha is swept so the density-field family's coverage AND count span a
range overlapping the blob family's (unlike the v1 jitter sweep, which
saturated at coverage~1.0 and left count unmatched), then compared at the
closest matched points. Growth (plastic - frozen taught mass; see Exp
32b/38's 2026-08-29 correction, PROJECT_HISTORY gotcha #11) is the outcome,
not raw taught mass, for the reasons documented there.

DECISION RULE, registered before running:
  - If growth at matched coverage is statistically indistinguishable
    between FILAMENT and BLOB (Welch p > 0.05, or a significant
    difference smaller than the smallest effect Exp 43 called real,
    ~1.06x), coverage is confirmed as sufficient and topology beyond it
    is decorative for this outcome measure.
  - If growth differs significantly and non-trivially at matched
    coverage, topology (filament/connected vs blob/isolated) has an
    effect coverage does not capture -- a genuinely new finding.
  - A pooled regression of growth on coverage across BOTH families,
    with a family indicator, is reported either way: a significant
    family term after controlling for coverage is the general-purpose
    version of the same test.

Run: python src/experiments/exp45_cosmic_web_topology.py
"""
import numpy as np
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")

from scipy import stats
from scipy.spatial import Voronoi

from exp32b_benchmark import NC, N, run_life
from exp43_coverage_confirmatory import make_condition, mediators, PLASTICITY_RADIUS

SEEDS = list(range(8))
BBOX = (0.0, 88.0, 0.0, 44.0)   # comparable extent to dilute30's blob lattice


def make_filament(seed, n_voronoi_seeds=14, jitter=1.2, bbox=BBOX):
    """N points scattered along a Voronoi ridge network -- filaments with
    voids in the cell interiors -- types uniformly shuffled across all
    points (composition free, arrangement fixed by the filament geometry)."""
    rng = np.random.default_rng(3000 + seed * 131 + int(jitter * 1000) + n_voronoi_seeds)
    x0, x1, y0, y1 = bbox
    seeds_pts = rng.uniform([x0, y0], [x1, y1], size=(n_voronoi_seeds, 2))
    vor = Voronoi(seeds_pts)
    segs = []
    for p1, p2 in vor.ridge_vertices:
        if p1 == -1 or p2 == -1:
            continue
        v1, v2 = vor.vertices[p1], vor.vertices[p2]
        if (x0 - 10 <= v1[0] <= x1 + 10 and y0 - 10 <= v1[1] <= y1 + 10 and
                x0 - 10 <= v2[0] <= x1 + 10 and y0 - 10 <= v2[1] <= y1 + 10):
            segs.append((v1, v2))
    if len(segs) < 3:
        raise RuntimeError("too few filament segments -- increase n_voronoi_seeds or bbox")
    lens = np.array([np.linalg.norm(b - a) for a, b in segs])
    probs = lens / lens.sum()
    idx = rng.choice(len(segs), size=N, p=probs)
    coords = np.zeros((N, 2))
    for k, si in enumerate(idx):
        a, b = segs[si]
        t = rng.uniform(0, 1)
        base = a + t * (b - a)
        d = b - a
        nrm = np.linalg.norm(d)
        perp = np.array([-d[1], d[0]]) / nrm if nrm > 1e-9 else np.zeros(2)
        coords[k] = base + perp * rng.normal(0, jitter)
    cids_pool = np.repeat(np.arange(NC), N // NC)
    while len(cids_pool) < N:
        cids_pool = np.append(cids_pool, NC - 1)
    rng.shuffle(cids_pool)
    return coords, cids_pool.astype(int)


def _voronoi_segments(seed_pts):
    vor = Voronoi(seed_pts)
    segs = []
    for p1, p2 in vor.ridge_vertices:
        if p1 == -1 or p2 == -1:
            continue
        v1, v2 = vor.vertices[p1], vor.vertices[p2]
        segs.append((v1, v2))
    return np.array([s[0] for s in segs]), np.array([s[1] for s in segs])


def _field(points, seg_a, seg_b, sigma):
    """Smooth scalar density: sum of Gaussian 'tubes' around each segment,
    evaluated by perpendicular distance to the (clamped) segment. Overlapping
    tubes near junctions add up -- filaments thicken near nodes exactly the
    way real cosmic-web filaments do -- and the field is continuous and
    curved everywhere, including at junctions, so a level-set of it has no
    hard polygonal angles (the defect a 2026-08-29 review flagged in the
    straight-Voronoi-ridge version: 'the sharp angles are wrong, it's more
    shaped like a gel poured over a skull'). This is also methodologically
    closer to how real cosmic-web filaments are identified -- smoothing a
    density field and thresholding it -- than the polygonal skeleton was."""
    P = points[:, None, :]
    A = seg_a[None, :, :]
    B = seg_b[None, :, :]
    D = B - A
    denom = np.maximum((D * D).sum(-1), 1e-9)
    t = np.clip(((P - A) * D).sum(-1) / denom, 0.0, 1.0)
    proj = A + t[..., None] * D
    dist2 = ((P - proj) ** 2).sum(-1)
    return np.exp(-dist2 / (2 * sigma ** 2)).sum(axis=1)


def make_filament_density(seed, sigma=1.6, threshold=0.35, n_voronoi_seeds=14, bbox=BBOX,
                           batch=6000, max_batches=200):
    """N points rejection-sampled from a smooth metaball-style density field
    (Gaussian tubes around a Voronoi ridge skeleton, summed and thresholded)
    instead of scattered directly on the polygonal skeleton. Filaments taper
    and thicken continuously; voids are wherever the field falls below
    threshold, with a curved (not straight-edged) boundary. Types uniformly
    shuffled across all accepted points."""
    rng = np.random.default_rng(4000 + seed * 197 + int(sigma * 1000) + int(threshold * 1000))
    x0, x1, y0, y1 = bbox
    seed_pts = rng.uniform([x0, y0], [x1, y1], size=(n_voronoi_seeds, 2))
    seg_a, seg_b = _voronoi_segments(seed_pts)
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
    cids_pool = np.repeat(np.arange(NC), N // NC)
    while len(cids_pool) < N:
        cids_pool = np.append(cids_pool, NC - 1)
    rng.shuffle(cids_pool)
    return coords, cids_pool.astype(int)


def make_filament_density_biased(seed, alpha, sigma=1.6, threshold=0.35, n_voronoi_seeds=14,
                                  bbox=BBOX, bias_strength=0.85, batch=6000, max_batches=200):
    """Same smooth density-field geometry as make_filament_density (fixed
    sigma/threshold -> fixed field, fixed raw local point density), but type
    assignment is biased per-segment like make_filament_biased: each point
    is assigned to its nearest Voronoi segment, and with probability
    (1-alpha)*bias_strength takes that segment's dominant type instead of a
    uniform draw. alpha sweeps coverage down from ~1.0 while the underlying
    smooth geometry is identical across alpha."""
    rng = np.random.default_rng(4000 + seed * 197 + int(sigma * 1000) + int(threshold * 1000))
    x0, x1, y0, y1 = bbox
    seed_pts = rng.uniform([x0, y0], [x1, y1], size=(n_voronoi_seeds, 2))
    seg_a, seg_b = _voronoi_segments(seed_pts)
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
    # nearest segment per point, for per-segment type bias
    P = coords[:, None, :]
    A = seg_a[None, :, :]
    B = seg_b[None, :, :]
    D = B - A
    denom = np.maximum((D * D).sum(-1), 1e-9)
    t = np.clip(((P - A) * D).sum(-1) / denom, 0.0, 1.0)
    proj = A + t[..., None] * D
    dist2 = ((P - proj) ** 2).sum(-1)
    nearest_seg = dist2.argmin(axis=1)
    dominant = rng.integers(0, NC, size=len(seg_a))
    cids = np.empty(N, dtype=int)
    for k in range(N):
        if rng.random() < alpha:
            cids[k] = rng.integers(0, NC)
        else:
            cids[k] = dominant[nearest_seg[k]] if rng.random() < bias_strength else rng.integers(0, NC)
    return coords, cids


def make_filament_biased(seed, alpha, jitter=1.2, n_voronoi_seeds=14, bbox=BBOX, bias_strength=0.85):
    """Same filament GEOMETRY as make_filament (fixed jitter -> fixed raw
    local point density / total reach), but type assignment is no longer
    uniformly shuffled across all points. Each Voronoi ridge SEGMENT gets a
    randomly assigned 'dominant type'; a point on that segment takes the
    dominant type with probability (1-alpha)*bias_strength, else a uniform
    random type. alpha=1 reproduces make_filament's full mixing (coverage
    saturates near 1.0); alpha->0 concentrates types by segment (coverage
    drops, mirroring how Exp 43's clumped30 depressed coverage by
    concentrating partner types into a few blobs), while the underlying
    POINT GEOMETRY -- and hence total local density / topology -- is
    identical to make_filament at the same jitter for every alpha. This is
    what makes the coverage sweep clean: only WHICH type sits where changes,
    not the network shape itself."""
    rng = np.random.default_rng(3000 + seed * 131 + int(jitter * 1000) + n_voronoi_seeds)
    x0, x1, y0, y1 = bbox
    seeds_pts = rng.uniform([x0, y0], [x1, y1], size=(n_voronoi_seeds, 2))
    vor = Voronoi(seeds_pts)
    segs = []
    for p1, p2 in vor.ridge_vertices:
        if p1 == -1 or p2 == -1:
            continue
        v1, v2 = vor.vertices[p1], vor.vertices[p2]
        if (x0 - 10 <= v1[0] <= x1 + 10 and y0 - 10 <= v1[1] <= y1 + 10 and
                x0 - 10 <= v2[0] <= x1 + 10 and y0 - 10 <= v2[1] <= y1 + 10):
            segs.append((v1, v2))
    if len(segs) < 3:
        raise RuntimeError("too few filament segments -- increase n_voronoi_seeds or bbox")
    lens = np.array([np.linalg.norm(b - a) for a, b in segs])
    probs = lens / lens.sum()
    idx = rng.choice(len(segs), size=N, p=probs)
    coords = np.zeros((N, 2))
    for k, si in enumerate(idx):
        a, b = segs[si]
        t = rng.uniform(0, 1)
        base = a + t * (b - a)
        d = b - a
        nrm = np.linalg.norm(d)
        perp = np.array([-d[1], d[0]]) / nrm if nrm > 1e-9 else np.zeros(2)
        coords[k] = base + perp * rng.normal(0, jitter)
    dominant = rng.integers(0, NC, size=len(segs))
    cids = np.empty(N, dtype=int)
    for k in range(N):
        if rng.random() < alpha:
            cids[k] = rng.integers(0, NC)
        else:
            cids[k] = dominant[idx[k]] if rng.random() < bias_strength else rng.integers(0, NC)
    return coords, cids


def growth_of(coords, cids, seeds=SEEDS):
    g = []
    for s in seeds:
        r = run_life(coords, cids, s)
        g.append(r["plastic"]["taught"] - r["frozen"]["taught"])
    return np.array(g, dtype=float)


DENSITY_CONFIGS = [
    # (label, alpha, bias_strength) -- sigma/threshold fixed at 1.6/0.35
    ("a1.00", 1.00, 0.85),
    ("a0.50", 0.50, 0.85),
    ("a0.20", 0.20, 0.85),
    ("a0.05", 0.05, 0.85),
    ("a0-b0.95", 0.00, 0.95),
    ("a0-b0.99", 0.00, 0.99),
]


if __name__ == "__main__":
    print("=" * 78)
    print("EXP 45 v2: smooth density-field filaments vs isolated blobs (matched coverage/count)")
    print("Geometry corrected 2026-08-29 per reviewer comment: straight Voronoi-ridge")
    print("skeleton (v1) replaced with a smoothed metaball-style density field --")
    print("continuous, curved, no polygonal angles. See module docstring / the")
    print("'Cosmic Web Check' artifact thread for before/after.")
    print("=" * 78)

    print("\n[1] Blob-family reference (Exp 43 conditions, seed 0 mediators)")
    blob_med = {}
    for name in ("spread15", "dilute30", "clumped30"):
        coords, cids = make_condition(name, seed=0)
        m = mediators(coords, cids)
        blob_med[name] = m
        print("  {:<10} count={:.2f} frac={:.3f} cover={:.3f}".format(name, m["count"], m["frac"], m["cover"]))

    print("\n[2] Density-field configs -> mediators (no simulation yet)")
    print("{:>10} {:>8} {:>8} {:>8}".format("label", "count", "frac", "cover"))
    density_med = {}
    for label, alpha, bias in DENSITY_CONFIGS:
        coords, cids = make_filament_density_biased(seed=0, alpha=alpha, bias_strength=bias)
        m = mediators(coords, cids)
        density_med[label] = m
        print("{:>10} {:>8.2f} {:>8.3f} {:>8.3f}".format(label, m["count"], m["frac"], m["cover"]))

    print("\n[3] Running the benchmark (growth = plastic - frozen taught mass), {} seeds each".format(len(SEEDS)))
    results = {}
    for name in ("spread15", "dilute30", "clumped30"):
        coords, cids = make_condition(name, seed=0)
        g = growth_of(coords, cids)
        results[("blob", name)] = (g, blob_med[name])
        print("  blob:{:<10} growth={:+7.1f} +/- {:<6.1f}  cover={:.3f}".format(name, g.mean(), g.std(), blob_med[name]["cover"]))
    for label, alpha, bias in DENSITY_CONFIGS:
        coords, cids = make_filament_density_biased(seed=0, alpha=alpha, bias_strength=bias)
        g = growth_of(coords, cids)
        results[("density", label)] = (g, density_med[label])
        print("  density:{:<10} growth={:+7.1f} +/- {:<6.1f}  cover={:.3f}  count={:.2f}".format(
            label, g.mean(), g.std(), density_med[label]["cover"], density_med[label]["count"]))

    print("\n" + "=" * 78)
    print("[4] Matched comparison: density(a1.00) vs blob:dilute30 (both cover~1.0, count~6-7)")
    print("=" * 78)
    g_fil, m_fil = results[("density", "a1.00")]
    g_blob, m_blob = results[("blob", "dilute30")]
    t, p = stats.ttest_ind(g_fil, g_blob, equal_var=False)
    print("  density   growth={:+.1f}+/-{:.1f}  cover={:.3f}  count={:.2f}".format(g_fil.mean(), g_fil.std(), m_fil["cover"], m_fil["count"]))
    print("  dilute30  growth={:+.1f}+/-{:.1f}  cover={:.3f}  count={:.2f}".format(g_blob.mean(), g_blob.std(), m_blob["cover"], m_blob["count"]))
    ratio = g_fil.mean() / g_blob.mean() if g_blob.mean() != 0 else float("nan")
    print("  Welch t={:.2f}, p={:.2e}, ratio(density/blob)={:.3f}x".format(t, p, ratio))
    if p > 0.05:
        print("  VERDICT: indistinguishable at matched coverage/count -> coverage sufficient,")
        print("  topology beyond it is decorative for this outcome.")
    else:
        print("  VERDICT: SIGNIFICANT difference at matched coverage/count -> topology beyond")
        print("  coverage has a real effect on learning.")

    print("\n" + "=" * 78)
    print("[5] Pooled regression: growth ~ coverage, with a family (blob/density-field) term")
    print("=" * 78)
    all_cover, all_growth, all_family = [], [], []
    for (family, _key), (g, m) in results.items():
        for gi in g:
            all_cover.append(m["cover"]); all_growth.append(gi); all_family.append(1.0 if family == "density" else 0.0)
    all_cover = np.array(all_cover); all_growth = np.array(all_growth); all_family = np.array(all_family)
    X = np.column_stack([np.ones_like(all_cover), all_cover, all_family])
    beta, _, _, _ = np.linalg.lstsq(X, all_growth, rcond=None)
    pred = X @ beta
    ss_tot = np.sum((all_growth - all_growth.mean()) ** 2)
    r2_full = 1 - np.sum((all_growth - pred) ** 2) / ss_tot
    X0 = np.column_stack([np.ones_like(all_cover), all_cover])
    beta0, _, _, _ = np.linalg.lstsq(X0, all_growth, rcond=None)
    pred0 = X0 @ beta0
    r2_cover_only = 1 - np.sum((all_growth - pred0) ** 2) / ss_tot
    print("  n points = {}  (coverage range {:.3f}-{:.3f})".format(len(all_cover), all_cover.min(), all_cover.max()))
    print("  R^2 (coverage only)           = {:.4f}".format(r2_cover_only))
    print("  R^2 (coverage + family term)  = {:.4f}".format(r2_full))
    print("  family coefficient (density-field vs blob, at fixed coverage) = {:+.1f}".format(beta[2]))
    print("  (a large jump in R^2 and/or family coefficient far from 0 means topology")
    print("   matters beyond coverage; little change means coverage already captures it)")
