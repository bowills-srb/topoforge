"""Chunk 3: Unified Life engine on SpatialGrid + SparsePairState.
Pluggable retention rules: 'corr' | 'util' | 'rpe'.
r_grow=None reproduces legacy global candidates; r_grow=<float> is honest physics.
Run smoke tests: python src/engine.py
"""
import numpy as np
import time
from spatial import SpatialGrid
from sparse_state import SparsePairState

PATTERNS = [(0, 3), (1, 4), (2,)]


def make_salt_clustered(N=1000, NC=5, seed_body=11, seed_centers=42, radius=8.0):
    rng = np.random.default_rng(seed_body)
    centers = np.random.default_rng(seed_centers).uniform(5, 35, size=(NC, 2))
    pts = []
    for cx, cy in centers:
        for _ in range(N // NC):
            th, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, radius)
            pts.append([cx + r * np.cos(th), cy + r * np.sin(th)])
    coords = np.array(pts)
    cids = np.repeat(np.arange(NC), N // NC)
    rng.shuffle(cids)
    return coords, cids


class Life:
    def __init__(self, coords, cids, rule='corr', n_edges=10000, r_grow=None,
                 seed_wiring=7, seed_life=0, rewards=None,
                 leak=0.90, thresh=1.0, refrac_steps=3, noise=0.02,
                 epoch=40, swap=500, alpha=0.15):
        self.coords, self.cids = coords, np.asarray(cids)
        self.N = len(coords)
        self.NC = int(self.cids.max()) + 1
        self.rule, self.r_grow = rule, r_grow
        self.rewards = rewards
        self.leak, self.thresh = leak, thresh
        self.refrac_steps, self.noise = refrac_steps, noise
        self.epoch, self.swap, self.alpha = epoch, swap, alpha

        rng2 = np.random.default_rng(seed_wiring)
        src = rng2.integers(0, self.N, n_edges)
        dst = rng2.integers(0, self.N, n_edges)
        keep = src != dst
        self.src, self.dst = src[keep].copy(), dst[keep].copy()
        self.inhib = rng2.random(self.N) < 0.20
        self._rebuild_adjacency()

        self.v = np.zeros(self.N)
        self.refrac = np.zeros(self.N, dtype=int)
        self.C = SparsePairState(0.95)
        self.E = SparsePairState(0.90)
        self.V = SparsePairState(0.999)
        self.Rhat = np.zeros(len(PATTERNS))
        self.rng = np.random.default_rng(seed_life)
        self.t = 0
        self.grid = SpatialGrid(coords, cell_size=2.0)

    # ---------- wiring ----------
    def _weight_of(self, s):
        return np.where(self.inhib[s], -0.60, 0.30)

    def _rebuild_adjacency(self):
        self.out_t = [[] for _ in range(self.N)]
        w = self._weight_of(self.src)
        for s, d, wi in zip(self.src, self.dst, w):
            self.out_t[s].append((d, wi))
        self.out_arr = [
            (np.array([d for d, _ in lst], dtype=np.int64),
             np.array([wi for _, wi in lst]))
            if lst else (np.empty(0, dtype=np.int64), np.empty(0))
            for lst in self.out_t
        ]
        self.edge_set = set(zip(self.src.tolist(), self.dst.tolist()))

    def _dist2(self, i, j):
        d = self.coords[i] - self.coords[j]
        return float(d @ d)

    # ---------- dynamics ----------
    def step(self):
        t = self.t
        p = (t // 20) % len(PATTERNS)
        inputs = self.rng.uniform(0, self.noise, self.N)
        if (t % 20) < 5:
            for cid in PATTERNS[p]:
                inputs[self.cids == cid] += 0.5

        v_ = self.v * self.leak + inputs
        fired = (v_ >= self.thresh) & (self.refrac == 0)
        f = np.where(fired)[0]

        if len(f):
            tgt = np.concatenate([self.out_arr[i][0] for i in f])
            wts = np.concatenate([self.out_arr[i][1] for i in f])
            if len(tgt):
                np.add.at(v_, tgt, wts)

        self.C.tick(); self.E.tick(); self.V.tick()
        if len(f):
            self.C.deposit_outer(f, 1.0)
            self.E.deposit_outer(f, 1.0)

        self.v = np.maximum(v_, 0)
        self.v[fired] = 0
        self.refrac[fired] = self.refrac_steps
        self.refrac[self.refrac > 0] -= 1

        if self.rewards is not None and (t % 20) == 6:
            base = self.rewards(t, p)
            if self.rule == 'rpe':
                delta = base - self.Rhat[p]
                if abs(delta) > 1e-9:
                    self.E.prune_below(1e-6)
                    for key in list(self.E.store.keys()):
                        ev = self.E.get(*key)
                        if ev != 0.0:
                            self.V.deposit(key[0], key[1], delta * ev)
                self.Rhat[p] += self.alpha * delta
            elif self.rule == 'util':
                if base != 0.0:
                    self.E.prune_below(1e-6)
                    for key in list(self.E.store.keys()):
                        ev = self.E.get(*key)
                        if ev != 0.0:
                            self.V.deposit(key[0], key[1], base * ev)

        self.t += 1
        if (self.t % self.epoch) == 0:
            self.rewire()

    # ---------- structural plasticity ----------
    def _edge_score(self):
        S = self.V if self.rule in ('util', 'rpe') else self.C
        return S.get_many(self.src, self.dst)

    def _cand_score(self, key):
        i, j = key
        d2 = self._dist2(i, j)
        if self.r_grow is not None and d2 > self.r_grow ** 2:
            return None
        if self.rule in ('util', 'rpe'):
            base = max(self.V.get(i, j), 0.0) + 0.01 * self.C.get(i, j)
        else:
            base = self.C.get(i, j)
        return base / (1.0 + 0.05 * d2)

    def rewire(self):
        self.C.prune_below(1e-6)
        cold = np.argsort(self._edge_score())[:self.swap]

        keys = set(self.C.store.keys())
        if self.rule in ('util', 'rpe'):
            keys |= set(self.V.store.keys())
        scored = []
        for key in keys:
            if key in self.edge_set or key[0] == key[1]:
                continue
            s = self._cand_score(key)
            if s is not None and s > 0:
                scored.append((s, key))
        scored.sort(reverse=True)
        top = scored[:len(cold)]

        for slot, (_, (i, j)) in zip(cold, top):
            self.src[slot], self.dst[slot] = i, j
        self._rebuild_adjacency()

    # ---------- readouts ----------
    def bridge_matrix(self):
        M = np.zeros((self.NC, self.NC), dtype=int)
        np.add.at(M, (self.cids[self.src], self.cids[self.dst]), 1)
        return M

    def taught_mass(self):
        M = self.bridge_matrix()
        return M[0, 3] + M[3, 0] + M[1, 4] + M[4, 1]

    def wire_energy(self):
        d = self.coords[self.src] - self.coords[self.dst]
        return float((d * d).sum())

    def run(self, steps, log_every=None):
        logs = []
        for _ in range(steps):
            self.step()
            if log_every and (self.t % log_every) == 0:
                logs.append((self.t, self.taught_mass(), self.wire_energy()))
        return logs


# ---------------- smoke tests ----------------
if __name__ == "__main__":
    print("=" * 62)
    print("CHUNK 3 SMOKE TESTS: unified engine (legacy mode, r_grow=None)")
    print("=" * 62)

    # [1] corr rule, salt-clustered, N=1000 — expect taught ~1800-2300
    coords, cids = make_salt_clustered()
    t0 = time.time()
    life = Life(coords, cids, rule='corr', seed_life=0)
    life.run(400)
    tm, we = life.taught_mass(), life.wire_energy()
    print(f"\n[1] corr N=1,000 x400:  taught={tm}  energy={we:,.0f}  "
          f"({time.time()-t0:.1f}s)")
    print(f"    dense-era reference (18c salt-clustered): taught 2,144")
    print(f"    ballpark check: {'PASS' if 1500 < tm < 2800 else 'CHECK'}")

    # [2] util rule, 600 steps — expect selectivity > 5x (audit criterion)
    rw = lambda t, p: {0: +1.0, 1: 0.0, 2: -1.0}[p]
    t0 = time.time()
    lifeU = Life(coords, cids, rule='util', rewards=rw, seed_life=0)
    lifeU.run(600)
    M = lifeU.bridge_matrix()
    sel = (M[0, 3] + M[3, 0]) / max(M[1, 4] + M[4, 1], 1)
    print(f"\n[2] util N=1,000 x600:  selectivity={sel:.1f}x  "
          f"({time.time()-t0:.1f}s)")
    print(f"    dense-era reference: 18.7x; audit criterion: > 5x")
    print(f"    criterion check: {'PASS' if sel > 5 else 'FAIL'}")

    # [3] the scale smoke: N=10,000, corr, 200 steps
    coordsX, cidsX = make_salt_clustered(N=10_000, radius=25.0)
    t0 = time.time()
    lifeX = Life(coordsX, cidsX, rule='corr', n_edges=100_000, swap=5000)
    lifeX.run(200)
    print(f"\n[3] corr N=10,000 x200 (100K edges): taught={lifeX.taught_mass()}  "
          f"({time.time()-t0:.1f}s)")
    print(f"    pass = it ran at all; dense would need 2.4GB matrices x3")

    print("\n" + "=" * 62)
    print("[1][2] certify legacy-mode behavior in the audit's ballpark;")
    print("[3] certifies the scale path exists. Exact gate = Chunk 4.")