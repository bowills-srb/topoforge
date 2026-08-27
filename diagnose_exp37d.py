"""Fast diagnostic v4: trace the FULL path from deposit to final readout.
Both output clusters fire (confirmed by v3), so reward should be firing
and deposits should be happening. Yet final bridge mass is exactly 0.
This traces: deposits happening? V accumulating? rewire firing? and
crucially -- do any input->output edges actually EXIST in src/dst at
the end to be measured?

Runs a SHORT version (2 epochs, not 15) with full instrumentation.
Run: python diagnose_exp37d.py
"""
import numpy as np
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid
from shd_loader import load_shd_samples

N_INPUT = 140
N_OUTPUT_A = 30
N_OUTPUT_B = 30
N = N_INPUT + N_OUTPUT_A + N_OUTPUT_B
INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE = 0, 1, 2

all_stimuli, all_labels = load_shd_samples(
    n_samples=200, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=200)
from collections import Counter
counts = Counter(all_labels)
top = [c for c, _ in counts.most_common(2)]
target_samples = [s for s, l in zip(all_stimuli, all_labels) if l == top[0]][:2]
distractor_samples = [s for s, l in zip(all_stimuli, all_labels) if l == top[1]][:2]

rng = np.random.default_rng(11)
coords, cids = [], []
all_types = [INPUT_TYPE]*N_INPUT + [OUTPUT_A_TYPE]*N_OUTPUT_A + [OUTPUT_B_TYPE]*N_OUTPUT_B
rng.shuffle(all_types)
for t_ in all_types:
    th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 10)
    coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)])
    cids.append(t_)
coords = np.array(coords); cids = np.array(cids)

rng2 = np.random.default_rng(7)
src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
keep = src != dst; src, dst = src[keep], dst[keep]
inhib = rng2.random(N) < 0.20
v = np.zeros(N); refrac = np.zeros(N, dtype=int)
C = SparsePairState(0.95); E = SparsePairState(0.90); V = SparsePairState(0.999)
Rhat = np.zeros(2)
g = SpatialGrid(coords, 6.0)
nbr = [g.within(i, 6.0) for i in range(N)]
out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
for s2, d2b in zip(src, dst):
    out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

input_idx = np.where(cids == INPUT_TYPE)[0]
a_idx = np.where(cids == OUTPUT_A_TYPE)[0]
b_idx = np.where(cids == OUTPUT_B_TYPE)[0]
in_set = set(input_idx.tolist()); a_set = set(a_idx.tolist()); b_set = set(b_idx.tolist())

def count_io_edges():
    n = 0
    for s, d in zip(src, dst):
        s, d = int(s), int(d)
        if (s in in_set and (d in a_set or d in b_set)) or \
           (d in in_set and (s in a_set or s in b_set)):
            n += 1
    return n

def measure_bridge():
    total = 0.0
    for s, d in zip(src, dst):
        s, d = int(s), int(d)
        if s in in_set and (d in a_set or d in b_set):
            total += max(V.get(s, d), 0)
        elif d in in_set and (s in a_set or s in b_set):
            total += max(V.get(d, s), 0)
    return total

print("Initial input<->output edges in src/dst: {}".format(count_io_edges()))

combined = [(s, 0) for s in target_samples] + [(s, 1) for s in distractor_samples]
t = 0
total_deposits = 0
rewire_count = 0

for epoch in range(2):
    for sample_arr, class_label in combined:
        recent_a = False; recent_b = False
        for local_t in range(sample_arr.shape[0]):
            inp = rng.uniform(0, 0.01, N)
            inp[input_idx] += sample_arr[local_t] * 0.35
            v_ = v * 0.90 + inp
            fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
            if len(f):
                for fi in f:
                    for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
            C.tick(); E.tick(); V.tick()
            if len(f):
                fs = set(int(x) for x in f)
                for i in f:
                    i = int(i)
                    for j in nbr[i]:
                        if int(j) in fs:
                            C.deposit(i, int(j), 1.0); E.deposit(i, int(j), 1.0)
                if fs & a_set: recent_a = True
                if fs & b_set: recent_b = True
            v = np.maximum(v_, 0); v[fired] = 0
            refrac[fired] = 3; refrac[refrac > 0] -= 1

            if t % 20 == 6:
                stream = class_label
                correct = recent_a if class_label == 0 else recent_b
                base = 1.0 if correct else 0.0
                delta = base - Rhat[stream]
                nd = 0
                if abs(delta) > 1e-9:
                    E.prune_below(1e-6)
                    for key in list(E.store.keys()):
                        ev = E.get(*key)
                        if ev != 0:
                            V.deposit(key[0], key[1], delta * ev); nd += 1
                total_deposits += nd
                Rhat[stream] += 0.15 * delta
                recent_a = False; recent_b = False
            t += 1

        for _ in range(15):
            v_ = v * 0.90 + rng.uniform(0, 0.01, N)
            fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
            if len(f):
                for fi in f:
                    for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
            C.tick(); E.tick(); V.tick()
            v = np.maximum(v_, 0); v[fired] = 0
            refrac[fired] = 3; refrac[refrac > 0] -= 1
            t += 1

        if t % 40 == 0:
            rewire_count += 1
        print("  after sample (t={:>5}): deposits_so_far={:>7} io_edges={:>4} "
              "bridge={:+.3f} Rhat={} rewires={}".format(
                  t, total_deposits, count_io_edges(), measure_bridge(),
                  np.round(Rhat, 3), rewire_count))

print("\nFINAL:")
print("  total deposits made: {}".format(total_deposits))
print("  input<->output edges existing: {}".format(count_io_edges()))
print("  final bridge mass: {:.4f}".format(measure_bridge()))
print("  rewire triggered: {} times".format(rewire_count))
print("\nDIAGNOSIS:")
if total_deposits == 0:
    print("  No deposits at all -- reward path broken.")
elif count_io_edges() == 0:
    print("  Deposits happened but NO input<->output edges exist to measure!")
    print("  The rewire step must be destroying them, or they never formed.")
elif measure_bridge() == 0:
    print("  Deposits happened, edges exist, but V on those specific edges is 0.")
    print("  Deposits are landing on DIFFERENT pairs than the ones in src/dst.")
    print("  (C/E deposit on spatial NEIGHBORS; src/dst are the actual edges --")
    print("   if those sets barely overlap, V never lands where it's measured.)")
else:
    print("  Bridge mass is nonzero here -- the short run works, so the issue")
    print("  is something that emerges over the longer 15-epoch run.")
