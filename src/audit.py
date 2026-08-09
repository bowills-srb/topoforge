"""Statistics harness: seed-sweeps the five load-bearing claims.
Pre-registered verdicts. Run: python src/audit.py"""
import numpy as np, time

N, NC = 1000, 5
PATTERNS = [(0,3), (1,4), (2,)]
SEEDS = [0, 1, 2, 3, 4]

def make_body(kind, seed_body=11):
    rng = np.random.default_rng(seed_body)
    centers = np.random.default_rng(42).uniform(5, 35, size=(5, 2))
    pts = []
    for cx, cy in centers:
        for _ in range(200):
            th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 8)
            pts.append([cx + r*np.cos(th), cy + r*np.sin(th)])
    coords = np.array(pts)
    cids = np.repeat(np.arange(NC), 200)
    if kind == 'interleaved':
        rng.shuffle(cids)
    return coords, cids

def k_nearest_energy(coords, k=10):
    n = len(coords)
    E = 0.0
    for i in range(n):
        d2 = ((coords - coords[i])**2).sum(1)
        d2[i] = np.inf
        E += np.sort(d2)[:k].sum()
    return E

def live(coords, cids, seed_life, rule='corr', steps=400, reversal=None,
         rewards=None):
    D2 = ((coords[:,None,:]-coords[None,:,:])**2).sum(-1)
    rng2 = np.random.default_rng(7)
    src = rng2.integers(0, N, 10000); dst = rng2.integers(0, N, 10000)
    keep = src != dst; src, dst = src[keep], dst[keep]
    inhib = rng2.random(N) < 0.20
    def bW(s, d):
        W = np.zeros((N, N))
        np.add.at(W, (s, d), np.where(inhib[s], -0.60, 0.30))
        return W
    W = bW(src, dst)
    v = np.zeros(N); refrac = np.zeros(N, dtype=int)
    C = np.zeros((N, N)); E = np.zeros((N, N)); V = np.zeros((N, N))
    Rhat = np.zeros(3)
    rng3 = np.random.default_rng(seed_life)
    snap = {}
    for t in range(steps):
        p = (t // 20) % 3
        inputs = rng3.uniform(0, 0.02, N)
        if (t % 20) < 5:
            for cid in PATTERNS[p]: inputs[cids == cid] += 0.5
        v_ = v*0.90 + inputs
        fired = (v_ >= 1.0) & (refrac == 0)
        if fired.any():
            v_ += W.T @ fired.astype(float)
            f = np.where(fired)[0]
            C[np.ix_(f, f)] += 1.0
            E[np.ix_(f, f)] += 1.0
        v = np.maximum(v_, 0); v[fired] = 0
        refrac[fired] = 3; refrac[refrac > 0] -= 1
        C *= 0.95; E *= 0.90
        if rewards and (t % 20) == 6:
            base = rewards(t, p)
            if rule == 'rpe':
                delta = base - Rhat[p]
                if abs(delta) > 1e-9: V += delta * E
                Rhat[p] += 0.15 * delta
            else:
                if base != 0.0: V += base * E
        V *= 0.999
        if (t+1) % 40 == 0:
            score = V if rule in ('util', 'rpe') else C
            cold = np.argsort(score[src, dst])[:500]
            if rule in ('util', 'rpe'):
                mask = (np.maximum(V, 0) + 0.01*C) / (1.0 + 0.05*D2)
            else:
                mask = C / (1.0 + 0.05*D2)
            mask[src, dst] = -1; np.fill_diagonal(mask, -1)
            flat = np.argpartition(mask.ravel(), -500)[-500:]
            ns, nd = np.unravel_index(flat, (N, N))
            src[cold], dst[cold] = ns, nd
            W = bW(src, dst)
            M = np.zeros((NC, NC), dtype=int)
            np.add.at(M, (cids[src], cids[dst]), 1)
            snap[t+1] = (M[0,3]+M[3,0], M[1,4]+M[4,1], M[0,1]+M[1,0])
    return snap

def report(name, vals, criterion, crit_fn):
    vals = np.array(vals, dtype=float)
    m, s = vals.mean(), vals.std()
    ok = crit_fn(vals)
    print(f"\nCLAIM: {name}")
    print(f"  seeds: {np.round(vals, 2)}")
    print(f"  mean {m:,.2f}  std {s:,.2f}  (spread {s/max(abs(m),1e-9)*100:.0f}%)")
    print(f"  criterion: {criterion}")
    print(f"  VERDICT: {'GRANITE' if ok else 'SOFT'}")
    return ok

print("=" * 66)
print("THE AUDIT: five load-bearing claims x five life-seeds")
print("=" * 66)
t0 = time.time()
verdicts = {}

# ---- CLAIM 1 (Exp 3): placement energy ratio, geometry-seed swept ----
ratios = []
for gs in [42, 43, 44, 45, 46]:
    rngA = np.random.default_rng(gs)
    grid = np.array([[ (i%32)*1.25, (i//32)*1.25 ] for i in range(N)])
    centers = rngA.uniform(5, 35, size=(5,2))
    pts = []
    for cx, cy in centers:
        for _ in range(200):
            th, r = rngA.uniform(0,2*np.pi), rngA.uniform(0, 1.0)
            pts.append([cx+r*np.cos(th), cy+r*np.sin(th)])
    tight = np.array(pts)
    ratios.append(k_nearest_energy(grid) / k_nearest_energy(tight))
verdicts['placement'] = report(
    "Placement moves k-NN wire energy by large factor (Exp 3: 79x)",
    ratios, "mean ratio > 20x across geometry seeds",
    lambda v: v.mean() > 20)

# ---- CLAIM 2 (18c): interleaving beats segregation on taught mass ----
adv = []
for s in SEEDS:
    ci, ii = make_body('interleaved')
    cs, is_ = make_body('segregated')
    ti = live(ci, ii, s)[400]
    ts = live(cs, is_, s)[400]
    adv.append((ti[0] + ti[1]) / max(ts[0] + ts[1], 1))
verdicts['interleave'] = report(
    "Interleaved identity beats segregated on taught mass (18c: +62%)",
    adv, "mean advantage > 1.25x and every seed > 1.0x",
    lambda v: v.mean() > 1.25 and (v > 1.0).all())

# ---- CLAIM 3 (19): utility selectivity vs correlation null ----
selU, selC = [], []
rw = lambda t, p: {0:+1.0, 1:0.0, 2:-1.0}[p]
for s in SEEDS:
    co, ci_ = make_body('interleaved')
    u = live(co, ci_, s, rule='util', steps=600, rewards=rw)[600]
    c = live(co, ci_, s, rule='corr', steps=600, rewards=rw)[600]
    selU.append(u[0] / max(u[1], 1)); selC.append(c[0] / max(c[1], 1))
print(f"\n  [control selectivity across seeds: {np.round(selC, 2)}]")
verdicts['utility'] = report(
    "Utility selectivity >> correlation null (19: 18.7x vs 1.1x)",
    selU, "mean utility selectivity > 5x AND mean control < 1.5x",
    lambda v: v.mean() > 5 and np.mean(selC) < 1.5)

# ---- CLAIM 4 (19b): sunk cost — no crossover within 600 post-flip ----
persist = []
rw_flip = lambda t, p: ({0:+1.0,1:0.0,2:-1.0} if t < 600 else {0:0.0,1:+1.0,2:-1.0})[p]
for s in SEEDS:
    co, ci_ = make_body('interleaved')
    sn = live(co, ci_, s, rule='util', steps=1200, rewards=rw_flip)
    crossed = any(sn[k][1] > sn[k][0] for k in sn if k > 600)
    persist.append(0.0 if crossed else 1.0)
verdicts['sunkcost'] = report(
    "Sunk-cost: empire never overtaken in 600 post-reversal steps (19b)",
    persist, "no crossover in >= 4/5 seeds",
    lambda v: v.sum() >= 4)

# ---- CLAIM 5 (19c): RPE produces crossover where raw utility doesn't ----
fixed = []
for s in SEEDS:
    co, ci_ = make_body('interleaved')
    sn = live(co, ci_, s, rule='rpe', steps=1200, rewards=rw_flip)
    crossed = any(sn[k][1] > sn[k][0] for k in sn if k > 600)
    fixed.append(1.0 if crossed else 0.0)
verdicts['rpe'] = report(
    "RPE dial produces regime change within window (19c: step 1160)",
    fixed, "crossover in >= 4/5 seeds",
    lambda v: v.sum() >= 4)

print("\n" + "=" * 66)
print(f"AUDIT COMPLETE in {(time.time()-t0)/60:.1f} min")
for k, ok in verdicts.items():
    print(f"  {k:>10}: {'GRANITE' if ok else 'SOFT'}")
print(f"\n{sum(verdicts.values())}/5 claims survived their pre-registered criteria.")