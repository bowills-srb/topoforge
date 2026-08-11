"""Chunk 2: Sparse pair-state (C/E/V) with lazy exponential decay.
Exact within the registered pair-set. Shadow-tested against dense matrices.
Run tests: python src/sparse_state.py
"""
import numpy as np
import time


class SparsePairState:
    """Tracks decaying scalars (e.g. correlation, eligibility, value) on a
    dynamic set of (i, j) pairs. Decay is lazy: value(t) = stored * decay^(t - touched).
    """

    def __init__(self, decay: float):
        self.decay = decay
        self.log_decay = np.log(decay) if decay > 0 else -np.inf
        self.store = {}      # (i, j) -> stored value at last touch
        self.touched = {}    # (i, j) -> step of last touch
        self.now = 0

    def tick(self, n: int = 1):
        """Advance the global clock. O(1) — this is the whole trick."""
        self.now += n

    def _settle(self, key):
        """Bring one pair's stored value up to the current step."""
        dt = self.now - self.touched[key]
        if dt > 0:
            self.store[key] *= np.exp(self.log_decay * dt)
            self.touched[key] = self.now

    def deposit(self, i: int, j: int, amount: float):
        """Add to pair (i, j) at the current step. Registers the pair if new."""
        key = (i, j)
        if key in self.store:
            self._settle(key)
            self.store[key] += amount
        else:
            self.store[key] = amount
            self.touched[key] = self.now

    def deposit_outer(self, indices: np.ndarray, amount: float):
        """Deposit on all ordered pairs (i, j), i != j, from `indices`.
        Mirrors dense C[np.ix_(f, f)] += amount."""
        for i in indices:
            for j in indices:
                if i != j:
                    self.deposit(int(i), int(j), amount)

    def get(self, i: int, j: int) -> float:
        key = (i, j)
        if key not in self.store:
            return 0.0
        dt = self.now - self.touched[key]
        return self.store[key] * (np.exp(self.log_decay * dt) if dt > 0 else 1.0)

    def get_many(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        return np.array([self.get(int(i), int(j)) for i, j in zip(src, dst)])

    def get_arrays(self):
        """Bulk-read all active pairs as (src, dst, values) arrays, settled
        to the current step. Vectorized counterpart to per-pair get()."""
        if not self.store:
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    np.empty(0))
        keys = np.array(list(self.store.keys()), dtype=np.int64)
        vals = np.array(list(self.store.values()), dtype=float)
        touched = np.array([self.touched[tuple(k)] for k in keys], dtype=np.int64)
        dt = self.now - touched
        vals = vals * np.exp(self.log_decay * dt)
        return keys[:, 0], keys[:, 1], vals

    def prune_below(self, eps: float = 1e-12):
        """Drop pairs whose current value has decayed to noise. Call rarely."""
        dead = [k for k in self.store if abs(self.get(*k)) < eps]
        for k in dead:
            del self.store[k]
            del self.touched[k]
        return len(dead)

    def n_pairs(self) -> int:
        return len(self.store)


# ---------------- shadow tests ----------------
if __name__ == "__main__":
    print("=" * 60)
    print("CHUNK 2 TESTS: SparsePairState vs dense matrices")
    print("=" * 60)

    N = 300
    STEPS = 200
    rng = np.random.default_rng(0)

    C_d = np.zeros((N, N)); E_d = np.zeros((N, N)); V_d = np.zeros((N, N))
    C_s = SparsePairState(0.95)
    E_s = SparsePairState(0.90)
    V_s = SparsePairState(0.999)

    max_err = 0.0
    t0 = time.time()
    for t in range(STEPS):
        f = rng.choice(N, size=rng.integers(5, 26), replace=False)

        C_d *= 0.95; E_d *= 0.90; V_d *= 0.999
        C_d[np.ix_(f, f)] += 1.0
        E_d[np.ix_(f, f)] += 1.0
        np.fill_diagonal(C_d, 0); np.fill_diagonal(E_d, 0)

        C_s.tick(); E_s.tick(); V_s.tick()
        C_s.deposit_outer(f, 1.0)
        E_s.deposit_outer(f, 1.0)

        if t % 20 == 6:
            V_d += 0.7 * E_d
            for key in list(E_s.store.keys()):
                e_val = E_s.get(*key)
                if e_val != 0.0:
                    V_s.deposit(key[0], key[1], 0.7 * e_val)

        for _ in range(50):
            i, j = rng.integers(0, N, 2)
            if i == j:
                continue
            for dense, sparse in ((C_d, C_s), (E_d, E_s), (V_d, V_s)):
                err = abs(dense[i, j] - sparse.get(int(i), int(j)))
                max_err = max(max_err, err)
    t_run = time.time() - t0

    print(f"\n[1] Shadow life: {STEPS} steps, N={N}, ~10k sampled comparisons")
    print(f"    max |dense - sparse| = {max_err:.3e}  "
          f"{'MATCH' if max_err < 1e-8 else 'FAIL'}")
    print(f"    active pairs: C={C_s.n_pairs():,}  V={V_s.n_pairs():,}  "
          f"(dense stores {N*N:,} each)")
    print(f"    runtime {t_run:.1f}s")

    print(f"\n[2] Memory projection:")
    for Nb in [1_000, 100_000]:
        dense_gb = 3 * Nb * Nb * 8 / 1e9
        sparse_mb = 3 * Nb * 40 * (8 * 3) / 1e6
        print(f"    N={Nb:>7,}: dense {dense_gb:>10,.2f} GB  |  "
              f"sparse ~{sparse_mb:,.0f} MB")

    S = SparsePairState(0.95)
    S.deposit(1, 2, 10.0)
    S.tick(137)
    want = 10.0 * 0.95 ** 137
    got = S.get(1, 2)
    ok3 = abs(want - got) < 1e-12
    print(f"\n[3] Lazy decay across 137-step gap: want {want:.6e} "
          f"got {got:.6e}  {'MATCH' if ok3 else 'FAIL'}")

    # test get_arrays matches get()
    ga_i, ga_j, ga_v = C_s.get_arrays()
    ga_ok = all(abs(C_s.get(int(i), int(j)) - v) < 1e-12
                for i, j, v in zip(ga_i[:100], ga_j[:100], ga_v[:100]))
    print(f"[3b] get_arrays matches get() (100 spot checks): "
          f"{'MATCH' if ga_ok else 'FAIL'}")

    S.tick(2000)
    removed = S.prune_below()
    print(f"[4] Prune after 2000 more steps: removed {removed} dead pair(s), "
          f"{S.n_pairs()} remain  {'OK' if S.n_pairs() == 0 else 'CHECK'}")

    print("\n" + "=" * 60)
    print("Pass: [1] err<1e-8, [3] MATCH, [3b] MATCH, [4] empty store.")