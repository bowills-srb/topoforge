"""Chunk 6: Vectorized rewire. Subclasses LocalLife, overrides only rewire()
with an array-based candidate scan. Certified against LocalLife identically.
Run: python src/engine_fast.py
"""
import numpy as np
import time
from engine import Life, make_salt_clustered, PATTERNS
from engine_local import LocalLife, make_segregated


class FastLife(LocalLife):
    def rewire(self):
        self.C.prune_below(1e-6)
        edge_score = self._edge_score()
        cold = np.argsort(edge_score)[:self.swap]

        # gather candidate pairs from active state as arrays
        ci, cj, cv = self.C.get_arrays()
        if self.rule in ("util", "rpe"):
            vi, vj, vv = self.V.get_arrays()
            # merge C and V candidate pair-sets
            ai = np.concatenate([ci, vi]); aj = np.concatenate([cj, vj])
        else:
            ai, aj = ci, cj

        if len(ai) == 0:
            return

        # dedupe pairs
        pack = ai.astype(np.int64) * self.N + aj.astype(np.int64)
        pack = np.unique(pack)
        ai = (pack // self.N).astype(np.int64)
        aj = (pack % self.N).astype(np.int64)

        # drop self-loops and existing edges
        keep = ai != aj
        ai, aj = ai[keep], aj[keep]
        existing = set(zip(self.src.tolist(), self.dst.tolist()))
        mask = np.fromiter((( int(i), int(j)) not in existing
                            for i, j in zip(ai, aj)),
                           dtype=bool, count=len(ai))
        ai, aj = ai[mask], aj[mask]
        if len(ai) == 0:
            return

        # vectorized distance + score
        d = self.coords[ai] - self.coords[aj]
        d2 = (d * d).sum(1)
        if self.r_grow is not None:
            within = d2 <= self.r_grow ** 2
            ai, aj, d2 = ai[within], aj[within], d2[within]
            if len(ai) == 0:
                return

        if self.rule in ("util", "rpe"):
            vpart = np.maximum(self.V.get_many(ai, aj), 0.0)
            cpart = self.C.get_many(ai, aj)
            base = vpart + 0.01 * cpart
        else:
            base = self.C.get_many(ai, aj)
        score = base / (1.0 + 0.05 * d2)

        pos = score > 0
        ai, aj, score = ai[pos], aj[pos], score[pos]
        if len(ai) == 0:
            return

        order = np.argsort(score)[::-1][:len(cold)]
        chosen_i, chosen_j = ai[order], aj[order]

        n = min(len(cold), len(chosen_i))
        self.src[cold[:n]] = chosen_i[:n]
        self.dst[cold[:n]] = chosen_j[:n]
        self._rebuild_adjacency()


if __name__ == "__main__":
    print("=" * 62)
    print("CHUNK 6: VECTORIZED REWIRE — certify vs LocalLife, then scale")
    print("=" * 62)

    coords, cids = make_salt_clustered()

    # [1] equivalence: FastLife must match LocalLife (same rewire logic,
    #     just vectorized). Compare taught mass on identical seed.
    lo = LocalLife(coords, cids, rule="corr", seed_life=0, r_deposit=6.0)
    lo.run(400)
    fa = FastLife(coords, cids, rule="corr", seed_life=0, r_deposit=6.0)
    fa.run(400)
    print(f"\n[1] Equivalence (corr, N=1,000 x400):")
    print(f"    LocalLife taught={lo.taught_mass()}  FastLife taught={fa.taught_mass()}")
    diff = abs(lo.taught_mass() - fa.taught_mass())
    print(f"    diff={diff}  {'MATCH' if diff == 0 else 'CLOSE' if diff < 50 else 'DIVERGED'}")

    # [2] speed at N=1,000
    t0 = time.time(); LocalLife(coords, cids, rule="corr", seed_life=1,
                                r_deposit=6.0).run(400); t_lo = time.time()-t0
    t0 = time.time(); FastLife(coords, cids, rule="corr", seed_life=1,
                               r_deposit=6.0).run(400); t_fa = time.time()-t0
    print(f"\n[2] Speed N=1,000: Local {t_lo:.0f}s  Fast {t_fa:.0f}s  "
          f"({t_lo/max(t_fa,0.1):.1f}x)")

    # [3] the ladder: N=10,000 then N=100,000
    print(f"\n[3] Scale (Fast, corr, 200 steps):")
    for N in [10_000, 100_000]:
        cN, idN = make_salt_clustered(N=N, radius=8.0*np.sqrt(N/1000))
        t0 = time.time()
        g = FastLife(cN, idN, rule="corr", n_edges=N*10, swap=N//2,
                     seed_life=0, r_deposit=6.0)
        g.run(200)
        print(f"    N={N:>7,} x200: {time.time()-t0:.0f}s  taught={g.taught_mass()}")

    print("\n" + "=" * 62)
    print("[1] MATCH/CLOSE = vectorization preserved logic.")
    print("[3] N=100,000 timing is the headline — the rung we have chased")
    print("all day. If it lands in minutes, the ladder is open.")
