"""Chunk 1: Grid-bucket spatial index for k-NN and radius queries.
Exact (not approximate) — the grid only prunes candidates; distances decide.
Run tests: python src/spatial.py
"""
import numpy as np
import time
from collections import defaultdict


class SpatialGrid:
    """Uniform grid over 2D points. Rebuild cheaply each epoch after migration."""

    def __init__(self, coords: np.ndarray, cell_size: float):
        self.coords = coords
        self.cell = cell_size
        self.buckets = defaultdict(list)
        keys = np.floor(coords / cell_size).astype(np.int64)
        for i, (kx, ky) in enumerate(keys):
            self.buckets[(kx, ky)].append(i)
        self.keys = keys

    def _candidates(self, i: int, ring: int = 1):
        """Indices in the (2*ring+1)^2 block of cells around point i."""
        kx, ky = self.keys[i]
        out = []
        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                b = self.buckets.get((kx + dx, ky + dy))
                if b:
                    out.extend(b)
        return out

    def k_nearest(self, i: int, k: int) -> np.ndarray:
        """Exact k nearest neighbors of point i (excluding i).
        Expands the search ring until k found AND the ring guarantees
        no closer point can exist outside it."""
        ring = 1
        n = len(self.coords)
        while True:
            cand = self._candidates(i, ring)
            if len(cand) > k:  # need k+1 (self is in there)
                cand = np.array(cand)
                d2 = ((self.coords[cand] - self.coords[i]) ** 2).sum(1)
                order = np.argsort(d2)
                cand, d2 = cand[order], d2[order]
                mask = cand != i
                cand, d2 = cand[mask], d2[mask]
                if len(cand) >= k:
                    # guarantee: kth distance must fit inside the ring
                    if np.sqrt(d2[k - 1]) <= ring * self.cell:
                        return cand[:k]
            ring += 1
            if ring * 2 + 1 > 3 * (int(np.sqrt(n)) + 2):  # degenerate fallback
                d2 = ((self.coords - self.coords[i]) ** 2).sum(1)
                d2[i] = np.inf
                return np.argsort(d2)[:k]

    def all_k_nearest(self, k: int) -> np.ndarray:
        """(N, k) neighbor matrix."""
        return np.array([self.k_nearest(i, k) for i in range(len(self.coords))])

    def within(self, i: int, r: float) -> np.ndarray:
        """All points within radius r of point i (excluding i). Exact."""
        ring = max(1, int(np.ceil(r / self.cell)))
        cand = np.array(self._candidates(i, ring))
        d2 = ((self.coords[cand] - self.coords[i]) ** 2).sum(1)
        mask = (d2 <= r * r) & (cand != i)
        return cand[mask]


# ---------------- tests ----------------
def brute_knn(coords, k):
    out = []
    for i in range(len(coords)):
        d2 = ((coords - coords[i]) ** 2).sum(1)
        d2[i] = np.inf
        out.append(np.argsort(d2)[:k])
    return np.array(out)


def knn_energy(coords, nbrs):
    e = 0.0
    for i, row in enumerate(nbrs):
        e += ((coords[row] - coords[i]) ** 2).sum()
    return e


if __name__ == "__main__":
    print("=" * 60)
    print("CHUNK 1 TESTS: SpatialGrid vs brute force")
    print("=" * 60)

    # --- Test 1: exactness on the standard die, N=1000 ---
    rng = np.random.default_rng(11)
    centers = np.random.default_rng(42).uniform(5, 35, size=(5, 2))
    pts = []
    for cx, cy in centers:
        for _ in range(200):
            th, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 8)
            pts.append([cx + r * np.cos(th), cy + r * np.sin(th)])
    coords = np.array(pts)

    t0 = time.time()
    ref = brute_knn(coords.copy(), 10)
    t_brute = time.time() - t0

    grid = SpatialGrid(coords, cell_size=2.0)
    t0 = time.time()
    got = grid.all_k_nearest(10)
    t_grid = time.time() - t0

    # energies must match exactly; neighbor SETS must match
    # (order can differ on distance ties, so compare as sets + energy)
    e_ref, e_got = knn_energy(coords, ref), knn_energy(coords, got)
    sets_match = all(set(ref[i]) == set(got[i]) for i in range(len(coords)))
    print(f"\n[1] Exactness N=1,000, k=10 (clustered die)")
    print(f"    energy brute={e_ref:.6f}  grid={e_got:.6f}  "
          f"{'MATCH' if abs(e_ref - e_got) < 1e-6 else 'FAIL'}")
    print(f"    neighbor sets: {'MATCH (1000/1000)' if sets_match else 'FAIL'}")
    print(f"    time: brute {t_brute*1000:.0f}ms  grid {t_grid*1000:.0f}ms")

    # --- Test 2: exactness on uniform gas (worst case for clustering) ---
    gas = np.random.default_rng(3).uniform(0, 40, size=(1000, 2))
    ref2 = brute_knn(gas.copy(), 10)
    got2 = SpatialGrid(gas, 2.0).all_k_nearest(10)
    ok2 = all(set(ref2[i]) == set(got2[i]) for i in range(1000))
    print(f"\n[2] Exactness N=1,000 uniform gas: {'MATCH' if ok2 else 'FAIL'}")

    # --- Test 3: radius query exactness (spot check 50 points) ---
    g3 = SpatialGrid(coords, 2.0)
    ok3 = True
    for i in range(0, 1000, 20):
        d2 = ((coords - coords[i]) ** 2).sum(1)
        want = set(np.where((d2 <= 25.0) & (np.arange(1000) != i))[0])
        got3 = set(g3.within(i, 5.0))
        if want != got3:
            ok3 = False
            break
    print(f"[3] Radius query (r=5.0, 50 spot checks): {'MATCH' if ok3 else 'FAIL'}")

    # --- Test 4: the scaling demo ---
    print(f"\n[4] Scaling (k=10, clustered layout, grid only — no brute):")
    for N in [10_000, 100_000]:
        rngS = np.random.default_rng(7)
        cN = rngS.uniform(0, 40 * np.sqrt(N / 1000), size=(N, 2))  # constant density
        t0 = time.time()
        gN = SpatialGrid(cN, 2.0)
        t_build = time.time() - t0
        t0 = time.time()
        _ = gN.all_k_nearest(10)
        t_query = time.time() - t0
        print(f"    N={N:>7,}: build {t_build:.2f}s | all-kNN {t_query:.1f}s")

    print("\n" + "=" * 60)
    print("Pass criteria: tests 1-3 MATCH. Test 4 is the payoff readout —")
    print("brute force at N=100,000 would need ~80GB and hours; the grid")
    print("should land in seconds-to-a-minute on laptop CPU.")