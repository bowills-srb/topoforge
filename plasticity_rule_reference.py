"""plasticity_rule_reference.py -- the TopoForge structural plasticity rule,
as a standalone specification.

Written for Catalyst Neuromorphic. This is not the research harness; it is the
rule extracted from it, with the inputs, the state, the update steps and the
outputs made explicit so it can be ported to a substrate where connectivity
lives in compiled tables rather than fixed wiring. Pure NumPy, no imports from
the research repo, no dependencies beyond NumPy itself. Run it directly for a
toy demonstration:

    python plasticity_rule_reference.py

WHAT THE RULE IS. A structural plasticity rule: it does not adjust synaptic
weights, it decides WHICH CONNECTIONS EXIST. Periodically it deletes
connections that have not earned their keep and creates new ones between
neurons that have fired together, earned reward, and are physically close.
Weights here are a fixed +/- per neuron (excitatory or inhibitory); all the
learning is in the connection set.

THE CONTRACT

  Inputs, per timestep:
      fired      indices of neurons that spiked this step
      reward     a scalar, supplied only on the steps you choose to evaluate
      positions  fixed (N, D) coordinates -- the placement being evaluated
  Persistent state:
      three decaying tables over neuron PAIRS (see below)
      optionally one bit per connection slot (see RECENCY)
  Output:
      a DELTA. Every rewire pass returns `removed` and `added`, each a list of
      (pre, post) pairs. For per-core connectivity tables these are exactly
      the rows to clear and the rows to write -- nothing else in the table
      changes, and you never need to diff two full edge lists to find out what
      moved. `rule.edges()` still gives the full list if you want it.

      Between rewire passes the connectivity is static: deposits and reward
      only update the pair tables. So a fabric only has to touch its tables
      every `rewire_every` steps, not every step.

  The rule never needs the full N x N matrix. All three tables are sparse:
  they hold only pairs that have actually co-fired, and entries decay away.

THE THREE TABLES. Each is a scalar per neuron pair with its own decay rate,
and each answers a different question. Three is the minimum that works --
removing any one breaks the rule in a specific way, and adding more has been
tested and does not help (see NOTES).

    C   decay 0.95    "have these two been active together recently?"
                      Defines the candidate pool. Without it you would have to
                      consider all N^2 pairs.
    E   decay 0.90    "this pair just did something."  A fading tag that
                      survives ~50 steps. Solves the credit-assignment delay:
                      reward arrives after the event that caused it.
    V   decay 0.999   "what has this pair earned, over the whole run?"
                      Survives pruning passes; this is what structure is.

  Decay is lazy -- value(t) = stored * decay^(t - last_touched) -- so advancing
  the clock is O(1) and costs nothing per pair. Worth keeping in hardware.

THE UPDATE STEPS

    1. DEPOSIT (every step). For every pair (i, j) where BOTH fired AND
       distance(i, j) <= r_deposit:  C += 1, E += 1.

       This line is the whole thesis. A pair that never falls within
       r_deposit of each other is never tagged, so reward can never credit
       it, so it never becomes a candidate. Distance does not make an
       association expensive here -- it makes it unreachable.

    2. REWARD (on evaluation steps). delta = reward - expected[context];
       for every pair holding eligibility:  V += delta * E.
       Then expected[context] += alpha * delta.

       Signed: a negative surprise actively removes value.

    3. PRUNE AND REGROW (every `rewire_every` steps).
         a. rank existing connections by V; the lowest `n_swap` are cut
         b. candidates = pairs in C that are not already connections, not
            self-loops, and within r_grow
         c. score = (max(V, 0) + 0.01 * C) / (1 + 0.05 * distance^2)
         d. the best-scoring candidates take the freed slots

       Locality enters three times: r_deposit in step 1, r_grow in 3b, and
       the distance term in 3c. The first is the one that decides what is
       learnable at all; the other two decide what is preferred.

RECENCY -- OPTIONAL, ONE BIT PER SLOT, NEW AND WORTH YOUR ATTENTION.

  Set `recency_bonus=True` and each connection slot carries one bit: was it
  rewritten at the last rewire pass? Slots that were not get a small bonus in
  the step-3a ranking. In effect a new connection gets exactly one epoch to
  prove itself; survive that and you are on the permanent roster.

  Measured on our synthetic benchmark, 20 seeds, pre-registered: re-learning
  after a task change improves 61.5% (CI on the ratio [1.46, 1.80]), with no
  effect on first-time learning. Four control policies were run to establish
  it is recency specifically -- a bonus of identical magnitude carrying no
  information at all does nothing (+2.1%), and the opposite policy (protect
  the newest) does nothing (-1.8%).

  Caveat stated plainly: this is measured as a count of connections formed,
  not as task accuracy. We have not yet been able to measure the accuracy
  endpoint for it. Treat the size of the number with suspicion; the direction
  and the mechanism are solid.

NOTES ON WHAT WE TRIED AND DROPPED. Three further tables were tested --
firing order (who led), a slow consolidation store behind V, and a graded
survival count -- across 10 seeds, pre-registered. None changed learning by
more than 0.2%, and none changed the placement effect at all. Three tables is
both the floor and, for this rule, the ceiling. That is a design result in
your favour: per-pair state is the expensive part in silicon.

PARAMETERS. Defaults below are the values our published results use. The two
that carry the science are `r_deposit` and `r_grow`; the rest are ordinary LIF
and RL settings we have not tuned. `r_grow=None` reverts to a global candidate
search, which is the non-physical legacy mode -- useful as a control, not as a
model of hardware.
"""
import numpy as np

__all__ = ["DecayTable", "StructuralPlasticityRule", "apply_delta"]


class DecayTable:
    """A sparse scalar per neuron pair, with lazy exponential decay.

    Lazy means the clock advances in O(1) regardless of how many pairs are
    held: a pair's value is brought up to date only when it is read or
    written. In hardware this is a stored value plus a timestamp per pair.
    """

    def __init__(self, decay):
        self.decay = float(decay)
        self.value = {}       # (i, j) -> value at last touch
        self.stamp = {}       # (i, j) -> step of last touch
        self.now = 0

    def tick(self, n=1):
        self.now += n

    def get(self, i, j):
        key = (i, j)
        if key not in self.value:
            return 0.0
        dt = self.now - self.stamp[key]
        return self.value[key] * (self.decay ** dt) if dt else self.value[key]

    def deposit(self, i, j, amount):
        key = (i, j)
        if key in self.value:
            dt = self.now - self.stamp[key]
            if dt:
                self.value[key] *= self.decay ** dt
            self.value[key] += amount
        else:
            self.value[key] = amount
        self.stamp[key] = self.now

    def drop_below(self, floor):
        """Forget pairs that have decayed to noise. Bounds memory."""
        for key in [k for k in self.value if abs(self.get(*k)) < floor]:
            del self.value[key]
            del self.stamp[key]

    def pairs(self):
        return list(self.value.keys())

    def __len__(self):
        return len(self.value)


class StructuralPlasticityRule:
    """The rule. Owns the edge list and rewrites it; owns no neuron dynamics.

    Drive it from whatever simulator or fabric you have:

        rule = StructuralPlasticityRule(positions, src, dst, inhibitory)
        for t in range(steps):
            fired = your_substrate.step(rule.src, rule.dst, rule.weights())
            rule.observe(fired)                  # step 1
            if t % 20 == 6:
                rule.reward(r, context=c)        # step 2
            delta = rule.maybe_rewire(t)         # step 3
            if delta:                            # rows to clear / rows to write
                your_fabric.update_tables(delta["removed"], delta["added"])
    """

    def __init__(self, positions, src, dst, inhibitory,
                 r_deposit=5.0, r_grow=None,
                 decay_c=0.95, decay_e=0.90, decay_v=0.999,
                 w_exc=0.30, w_inh=-0.60,
                 alpha=0.15, n_contexts=1,
                 rewire_every=40, n_swap=None,
                 recency_bonus=False, recency_eta=0.25,
                 prune_floor=1e-6):
        self.pos = np.asarray(positions, float)
        self.n = len(self.pos)
        self.src = np.asarray(src, np.int64).copy()
        self.dst = np.asarray(dst, np.int64).copy()
        self.inhibitory = np.asarray(inhibitory, bool)
        self.r_deposit = r_deposit
        self.r_grow = r_grow
        self.w_exc, self.w_inh = w_exc, w_inh
        self.alpha = alpha
        self.rewire_every = rewire_every
        self.n_swap = n_swap if n_swap is not None else max(1, self.n // 2)
        self.prune_floor = prune_floor
        self.recency_bonus = recency_bonus
        self.recency_eta = recency_eta

        self.C = DecayTable(decay_c)
        self.E = DecayTable(decay_e)
        self.V = DecayTable(decay_v)
        self.expected = np.zeros(max(1, n_contexts))
        self.fresh = np.zeros(len(self.src), bool)   # the one bit per slot

        # Neighbours within r_deposit, precomputed. Placement is fixed, so this
        # is computed once. On a fabric this is a compile-time table.
        self._nbr = _neighbours_within(self.pos, r_deposit)

    # ---- output -----------------------------------------------------------

    def weights(self):
        """Fixed sign per presynaptic neuron. No weight learning happens."""
        return np.where(self.inhibitory[self.src], self.w_inh, self.w_exc)

    def edges(self):
        return self.src.copy(), self.dst.copy()

    # ---- step 1: deposit --------------------------------------------------

    def observe(self, fired):
        """Tag nearby pairs that fired together. Call once per timestep."""
        self.C.tick(); self.E.tick(); self.V.tick()
        if len(fired) == 0:
            return
        hot = set(int(x) for x in fired)
        for i in hot:
            for j in self._nbr[i]:
                if j in hot and j != i:
                    self.C.deposit(i, j, 1.0)
                    self.E.deposit(i, j, 1.0)

    # ---- step 2: reward ---------------------------------------------------

    def reward(self, signal, context=0):
        """Pay the prediction error out to everything currently eligible."""
        delta = float(signal) - self.expected[context]
        if abs(delta) > 1e-9:
            self.E.drop_below(self.prune_floor)
            for (i, j) in self.E.pairs():
                e = self.E.get(i, j)
                if e:
                    self.V.deposit(i, j, delta * e)
        self.expected[context] += self.alpha * delta
        return delta

    # ---- step 3: prune and regrow ----------------------------------------

    def maybe_rewire(self, t):
        """Returns None on steps that are not rewire steps, else the delta."""
        if (t + 1) % self.rewire_every:
            return None
        return self.rewire()

    def rewire(self):
        """Cut the connections that earned least; regrow toward value+proximity.

        Returns {"removed": [(pre, post), ...], "added": [(pre, post), ...]}
        -- the rows to clear and the rows to write in a connectivity table.
        Both lists are the same length, and `added[k]` takes the slot vacated
        by `removed[k]`, so the total synapse count never changes.
        """
        self.C.drop_below(self.prune_floor)

        keep_score = np.array([self.V.get(int(a), int(b))
                               for a, b in zip(self.src, self.dst)])
        if self.recency_bonus:
            # One bit: slots NOT rewritten last pass get a bonus, i.e. a new
            # connection gets exactly one epoch of probation. Scaled by the
            # spread of the scores so it is dimensionless.
            spread = float(keep_score.max() - keep_score.min())
            if spread > 0:
                keep_score = keep_score + self.recency_eta * (~self.fresh) * spread
        cold = np.argsort(keep_score, kind="stable")[:self.n_swap]

        existing = set(zip(self.src.tolist(), self.dst.tolist()))
        cand, score = [], []
        for (i, j) in self.C.pairs():
            if i == j or (i, j) in existing:
                continue
            d2 = float(((self.pos[i] - self.pos[j]) ** 2).sum())
            if self.r_grow is not None and d2 > self.r_grow ** 2:
                continue
            s = (max(self.V.get(i, j), 0.0) + 0.01 * self.C.get(i, j)) / (1.0 + 0.05 * d2)
            if s > 0:
                cand.append((i, j)); score.append(s)
        if not cand:
            self.fresh[:] = False
            return {"removed": [], "added": []}

        best = np.argsort(np.array(score), kind="stable")[::-1][:len(cold)]
        n_new = min(len(cold), len(best))
        self.fresh[:] = False
        removed, added = [], []
        for slot, k in zip(cold[:n_new], best[:n_new]):
            removed.append((int(self.src[slot]), int(self.dst[slot])))
            self.src[slot], self.dst[slot] = cand[k]
            added.append(cand[k])
            self.fresh[slot] = True
        return {"removed": removed, "added": added}


def apply_delta(connectivity, delta):
    """Reference for applying a delta to your own connectivity list.

    `connectivity` is any container of (pre, post) pairs. Removals are applied
    before additions, so a slot freed this pass can be refilled in the same
    pass without the two operations colliding.
    """
    out = list(connectivity)
    for pair in delta["removed"]:
        out.remove(tuple(pair))
    out.extend(tuple(p) for p in delta["added"])
    return out


def _neighbours_within(pos, radius):
    """Exact radius query via uniform grid bucketing. Grid prunes, distance decides."""
    if radius is None:
        return [np.arange(len(pos)) for _ in range(len(pos))]
    cell = {}
    keys = np.floor(pos / radius).astype(np.int64)
    for i, k in enumerate(keys):
        cell.setdefault(tuple(k), []).append(i)
    dim = pos.shape[1]
    offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * dim)).T.reshape(-1, dim)
    out = []
    r2 = radius * radius
    for i in range(len(pos)):
        near = []
        for off in offsets:
            near.extend(cell.get(tuple(keys[i] + off), ()))
        near = np.array(near, np.int64)
        d2 = ((pos[near] - pos[i]) ** 2).sum(1)
        out.append(set(int(x) for x in near[d2 <= r2]))
    return out


# --------------------------------------------------------------------------
# Toy demonstration: two neuron types that must associate, under two
# placements. Same neurons, same starting connections, same stimulus --
# only the arrangement differs. This is the whole claim in 40 lines.
# --------------------------------------------------------------------------

def _demo(placement, steps=1200, seed=0, recency=False):
    rng = np.random.default_rng(seed)
    n_per, n_type = 60, 4
    n = n_per * n_type

    pos = rng.uniform(0, 30, (n, 2))
    if placement == "segregated":
        order = np.argsort(pos[:, 0], kind="stable")
        types = np.empty(n, int)
        for k, idx in enumerate(np.array_split(order, n_type)):
            types[idx] = k
    else:
        types = np.tile(np.arange(n_type), n_per)
        rng.shuffle(types)

    wiring = np.random.default_rng(1000 + seed)
    src = wiring.integers(0, n, n * 8)
    dst = wiring.integers(0, n, n * 8)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    inhib = wiring.random(n) < 0.20

    rule = StructuralPlasticityRule(pos, src, dst, inhib, r_deposit=5.0,
                                    n_swap=len(src) // 20, n_contexts=2,
                                    recency_bonus=recency)

    v = np.zeros(n); refrac = np.zeros(n, int)
    n_changed = 0
    pairs = [(0, 2), (1, 3)]          # type 0 must associate with type 2, 1 with 3
    for t in range(steps):
        ctx = (t // 20) % 2
        drive = rng.uniform(0, 0.02, n)
        if (t % 20) < 5:
            for c in pairs[ctx]:
                drive[types == c] += 0.5
        w = rule.weights()
        v_ = v * 0.90 + drive
        fired = (v_ >= 1.0) & (refrac == 0)
        f = np.flatnonzero(fired)
        if len(f):
            m = fired[rule.src]
            if m.any():
                np.add.at(v_, rule.dst[m], w[m])
        rule.observe(f)
        v = np.maximum(v_, 0); v[fired] = 0
        refrac[fired] = 3; refrac[refrac > 0] -= 1
        if (t % 20) == 6:
            rule.reward(1.0 if ctx == 0 else 0.0, context=ctx)
        delta = rule.maybe_rewire(t)
        if delta:
            n_changed += len(delta["added"])

    s, d = rule.edges()
    cross = sum(1 for a, b in zip(s, d)
                if (types[a], types[b]) in pairs or (types[b], types[a]) in pairs)
    return cross, n_changed


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("\nToy demo: 240 neurons, 4 types, types 0-2 and 1-3 must associate.")
    print("Same neurons and same starting connections in both rows; only the")
    print("arrangement differs.\n")
    print("  {:<14s} {:>22s}".format("placement", "associated connections"))
    for placement in ("segregated", "interleaved"):
        got = [_demo(placement, seed=s) for s in range(3)]
        print("  {:<14s} {:>22s}".format(placement, str([g[0] for g in got])))
    print("\n  The zeros are the point, not a broken demo: in this layout types 0")
    print("  and 2 occupy non-adjacent quarters, so no pair of them ever falls")
    print("  within r_deposit. They are never tagged, so reward can never credit")
    print("  them, so they never become candidates. The association is not")
    print("  expensive -- it is unreachable, at any budget.")
    print("\nRecency flag (one bit per slot), interleaved, same seeds:")
    got = [_demo("interleaved", seed=s, recency=True) for s in range(3)]
    print("  {:<14s} {:>22s}".format("recency on", str([g[0] for g in got])))
    print()
    print("Synapses rewritten over the run (the delta your tables would see):")
    print("  {:<14s} {:>22s}".format("per run", str([g[1] for g in got])))
    print("\nNumbers are illustrative at this toy scale -- the published")
    print("measurements are at N=900-5,000 with 10-20 seeds and pre-registered")
    print("criteria. The point of this file is the rule, not the numbers.")
