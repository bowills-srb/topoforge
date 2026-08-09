"""Chunk 5: Local candidate tracking — deposits restricted to spatially local pairs."""
import numpy as np
import time
from engine import Life, make_salt_clustered
from spatial import SpatialGrid
from engine import PATTERNS


def make_segregated(N=1000, NC=5):
    coords, _ = make_salt_clustered(N, NC)
    cids = np.repeat(np.arange(NC), N // NC)
    return coords, cids


class LocalLife(Life):
    def __init__(self, *args, r_deposit=6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_deposit = r_deposit
        self._build_neighbor_cache()

    def _build_neighbor_cache(self):
        if self.r_deposit is None:
            self.nbr_cache = None
            return
        g = SpatialGrid(self.coords, cell_size=max(2.0, self.r_deposit))
        self.nbr_cache = [g.within(i, self.r_deposit) for i in range(self.N)]

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
            if self.r_deposit is None:
                self.C.deposit_outer(f, 1.0)
                self.E.deposit_outer(f, 1.0)
            else:
                fset = set(int(x) for x in f)
                for i in f:
                    i = int(i)
                    for j in self.nbr_cache[i]:
                        if int(j) in fset:
                            self.C.deposit(i, int(j), 1.0)
                            self.E.deposit(i, int(j), 1.0)

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


if __name__ == "__main__":
    print("=" * 66)
    print("CHUNK 5: LOCAL DEPOSIT — speed + honest physics, characterized")
    print("=" * 66)

    coords, cids = make_salt_clustered()

    t0 = time.time()
    g_global = Life(coords, cids, rule="corr", seed_life=0)
    g_global.run(400)
    t_global = time.time() - t0
    tm_global = g_global.taught_mass()

    t0 = time.time()
    g_local = LocalLife(coords, cids, rule="corr", seed_life=0, r_deposit=6.0)
    g_local.run(400)
    t_local = time.time() - t0
    tm_local = g_local.taught_mass()

    print("\n[A] N=1,000, corr, 400 steps:")
    print(f"    GLOBAL (teleporter): taught={tm_global}  {t_global:.0f}s")
    print(f"    LOCAL  (honest):     taught={tm_local}  {t_local:.0f}s")
    print(f"    speedup: {t_global/max(t_local,0.1):.1f}x   "
          f"taught shift: {(tm_local-tm_global)/max(tm_global,1)*100:+.0f}%")

    print("\n[B] Phenomenon spot-check under local physics:")
    ci, ii = make_salt_clustered(); cs, is_ = make_segregated()
    li = LocalLife(ci, ii, rule="corr", seed_life=0, r_deposit=6.0); li.run(400)
    ls = LocalLife(cs, is_, rule="corr", seed_life=0, r_deposit=6.0); ls.run(400)
    adv = li.taught_mass() / max(ls.taught_mass(), 1)
    print(f"    interleave advantage: {adv:.2f}x  (baseline 1.46x)  "
          f"{'HOLDS' if adv > 1.0 else 'SHIFTED'}")
    rw = lambda t, p: {0:+1.0, 1:0.0, 2:-1.0}[p]
    lu = LocalLife(ci, ii, rule="util", rewards=rw, seed_life=0, r_deposit=6.0)
    lu.run(600)
    M = lu.bridge_matrix()
    sel = (M[0,3]+M[3,0]) / max(M[1,4]+M[4,1], 1)
    print(f"    utility selectivity:  {sel:.1f}x  (baseline 17.4x)  "
          f"{'HOLDS' if sel > 5 else 'SHIFTED'}")

    print("\n[C] Scale with local deposit (the point):")
    for N in [10_000, 100_000]:
        cN, idN = make_salt_clustered(N=N, radius=8.0*np.sqrt(N/1000))
        t0 = time.time()
        gN = LocalLife(cN, idN, rule="corr", n_edges=N*10,
                       swap=N//2, seed_life=0, r_deposit=6.0)
        gN.run(200)
        print(f"    N={N:>7,} x200 steps: {time.time()-t0:.0f}s  "
              f"taught={gN.taught_mass()}")
